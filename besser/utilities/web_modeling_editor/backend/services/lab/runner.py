"""One BESSER Lab run: a user's message through ``LabSession.handle``, streamed as Spec-Driven SSE events.

The session runs in a worker thread of the Spec-Driven runtime. Its callbacks fire from that thread and are
handed to the event loop with ``call_soon_threadsafe``, as ``services/spec_driven/runner.py`` does; a cost
event goes out every ``BESSER_LLM_COST_EMITTER_INTERVAL_SECONDS``. The run id is registered with
``active_runs``, so ``POST /spec-driven/cancel/{run_id}`` stops the run at its next turn. The route holds the
lab's lock for the run; :meth:`LabRunner.stream` releases it, with the run id, when the stream ends. A
stopped run ends with ``done`` like any other, ``cancelled`` and with the checks that ran.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, AsyncIterator, Optional

from besser.spec_driven_agent.errors import InvalidApiKeyError, UpstreamLLMError
from besser.spec_driven_agent.lab.contracts import ChangeKind, LabCallbacks, LabLimits, SessionOutcome
from besser.spec_driven_agent.lab.session import LabSession
from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from besser.spec_driven_agent.providers.llm_client import create_llm_client, default_model
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabCheck,
    LabChecklistItem,
    LabDone,
    LabMessageRequest,
    LabModelProposal,
    LabPlan,
)
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser
from besser.utilities.web_modeling_editor.backend.services.lab.model_proposal import (
    SPEC_ERRORS,
    model_spec_from_project,
    proposal_from_plan,
)
from besser.utilities.web_modeling_editor.backend.services.lab.service import LabService, repo_change_view
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runtime import SpecDrivenRuntime
from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
    BaseSseEvent,
    CostEvent,
    DoneEvent,
    ErrorEvent,
    PhaseEvent,
    StartEvent,
    TextDeltaEvent,
    ToolCallEvent,
    format_sse,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

logger = logging.getLogger(__name__)

# The changes whose build follows a class diagram of the extension. For any other kind the project's class
# diagram is the user's own application model, not part of the change.
_KINDS_WITH_MODEL = (ChangeKind.METAMODEL, ChangeKind.DIAGRAM_TYPE)
# The card lists this many changed paths; ``changedFileCount`` covers the rest (as a Spec-Driven run does).
_MAX_CHANGED_FILES = 50

# A run that stopped early but kept its work: a warning before ``done``.
_EARLY_STOPS = {
    "cost_cap": ("COST_CAP", "The run reached its cost cap; what it did so far is in the working copy."),
    "timeout": ("TIMEOUT", "The run reached its time limit; what it did so far is in the working copy."),
    "max_turns": ("INCOMPLETE", "The run reached its turn limit; what it did so far is in the working copy."),
    "api_error": ("INCOMPLETE", "The model provider failed during the run; what it did so far is in the "
                                "working copy."),
}


def lab_done(outcome: SessionOutcome) -> LabDone:
    """The ``lab`` field of the ``done`` event for ``outcome``.

    A plan whose class diagram cannot be converted gets the reason as an open question instead of a
    ``modelProposal``.
    """
    change, plan, build = outcome.change, outcome.plan, outcome.build
    # A plan that changes nothing proposes no model: there is no change for the editor to tie it to.
    proposal = proposal_from_plan(plan) if plan is not None and not plan.changes_nothing else None
    kind = change.kind if change is not None else plan.kind if plan is not None else ChangeKind.OTHER
    title = change.title if change is not None else plan.title if plan is not None else ""
    return LabDone(
        step=outcome.step,
        changeId=change.id if change is not None else "",
        title=title,
        kind=kind,
        plan=LabPlan(kind=plan.kind, title=plan.title, summary=plan.summary, steps=plan.steps, repos=plan.repos,
                     questions=plan.questions) if plan is not None else None,
        modelProposal=LabModelProposal(**proposal) if proposal is not None else None,
        repoChanges=[repo_change_view(c) for c in build.changes] if build is not None else [],
        checks=[LabCheck(id=r.spec_id, repo=r.repo, title=r.title, passed=r.passed, skipped=r.skipped,
                         summary=r.summary, durationSeconds=r.duration_seconds)
                for r in build.checks] if build is not None else [],
        checklist=[LabChecklistItem(id=i.id, text=i.text, done=i.done)
                   for i in build.checklist] if build is not None else [],
        summary=build.summary if build is not None else plan.summary if plan is not None else outcome.final_text,
        stopReason=outcome.stop_reason,
        finalText=build.loop.final_text if build is not None else "",
    )


def done_event(run_id: str, lab: LabDone) -> DoneEvent:
    """A ``done`` event the Spec-Driven run card renders: nothing to download, the changed paths per repository."""
    changed = [f"{change.repo}/{file.path}" for change in lab.repoChanges for file in change.files]
    return DoneEvent(runId=run_id, downloadUrl="", fileName="", isZip=False, recipe={}, mode="modify",
                     changedFiles=changed[:_MAX_CHANGED_FILES], changedFileCount=len(changed),
                     cancelled=lab.stopReason == "cancelled", lab=lab)


class LabRunner:
    """Drive one lab run and yield its SSE frames."""

    def __init__(self, request: LabMessageRequest, *, user: LabUser, lab: LabService, runtime: SpecDrivenRuntime,
                 settings: BackendSettings, run_id: str, cancel_event: asyncio.Event) -> None:
        self.request = request
        self.run_id = run_id
        self._user = user
        self._lab = lab
        self._runtime = runtime
        self._settings = settings
        self._cancel_event = cancel_event
        self._api_key = settings.lab_llm_api_key.get_secret_value()
        self._client: Any = None
        self._turns = 0
        self._started_at = time.monotonic()

    async def stream(self) -> AsyncIterator[bytes]:
        """The run's SSE frames; when they end, the run id and the lab are released."""
        try:
            # Closed explicitly: when this stream is closed early, the session is stopped and waited for
            # (``_frames``' finally) before the lab is released below.
            async with contextlib.aclosing(self._frames()) as frames:
                async for frame in frames:
                    yield frame
        except Exception:
            logger.exception("BESSER Lab run %s crashed", self.run_id)
            yield self._sse(ErrorEvent(code="INTERNAL", message="Internal server error"))
        finally:
            # The session has ended here; released before anything that can wait, so a cancelled
            # stream cannot leave the lab taken.
            self._lab.end_run()
            await self._runtime.active_runs.release(self.run_id, self._cancel_event)
            await self._runtime.run_blocking(self._lab.refresh_preview)

    def _sse(self, event: BaseSseEvent) -> bytes:
        return format_sse(event, secrets=(self._api_key,) if self._api_key else ())

    def _totals(self) -> dict[str, Any]:
        usage = getattr(self._client, "usage", None)
        return {"usd": round(float(getattr(usage, "estimated_cost", 0.0) or 0.0), 4), "turns": self._turns,
                "elapsedSeconds": round(time.monotonic() - self._started_at, 2)}

    def _model_spec(self) -> Optional[dict]:
        """The change's class diagram in the project, as the user corrected it, for a change that follows one.

        A plan revision starts from it as the build does, so a correction made on the canvas is kept.
        """
        if self.request.project is None:
            return None
        change = self._lab.workspace.active_change()
        if change is None or change.kind not in _KINDS_WITH_MODEL:
            return None
        return model_spec_from_project(self.request.project, change.id)

    async def _frames(self) -> AsyncIterator[bytes]:
        settings = self._settings
        provider = settings.lab_llm_provider
        yield self._sse(StartEvent(
            runId=self.run_id, provider=provider, llmModel=settings.lab_llm_model or default_model(provider) or "",
            maxCost=settings.lab_max_cost_usd, maxRuntime=int(settings.lab_max_runtime_seconds)))

        try:
            self._client = await self._runtime.run_blocking(
                create_llm_client, provider, api_key=self._api_key or None, model=settings.lab_llm_model)
        except ValueError as exc:  # InvalidApiKeyError included
            yield self._sse(ErrorEvent(code="INVALID_KEY", message=f"The lab's LLM key was refused: {exc}"))
            return
        try:
            model_spec = await self._runtime.run_blocking(self._model_spec)
        except SPEC_ERRORS as exc:
            yield self._sse(ErrorEvent(code="BAD_REQUEST",
                                       message=f"The project's class diagram could not be read: {exc}"))
            return
        except LabWorkspaceError as exc:
            yield self._sse(ErrorEvent(code="BAD_REQUEST", message=str(exc)))
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[BaseSseEvent]] = asyncio.Queue()
        text_state = {"emitted": False, "break_pending": False}

        def put(event: BaseSseEvent) -> None:
            if isinstance(event, PhaseEvent):
                event = event.model_copy(update=self._totals())  # stamped when the phase starts
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass  # the loop is closed: nobody reads the stream any more

        def on_progress(turn: int, tool: str, status: str, detail: Optional[str]) -> None:
            self._turns = max(self._turns, turn)
            if text_state["emitted"]:
                text_state["emitted"], text_state["break_pending"] = False, True
            put(ToolCallEvent(turn=turn, tool=tool, status=status, detail=detail or None))

        def on_text(delta: str) -> None:
            # Each turn's text starts a new paragraph; the model's next message has no leading whitespace.
            if delta:
                if text_state["break_pending"]:
                    text_state["break_pending"] = False
                    if not delta.startswith(("\n", " ")):
                        delta = "\n\n" + delta
                text_state["emitted"] = True
            put(TextDeltaEvent(delta=delta))

        callbacks = LabCallbacks(
            on_progress=on_progress,
            on_text=on_text,
            on_phase=lambda phase, message: put(PhaseEvent(phase=phase, message=message)),
            should_continue=lambda: not self._cancel_event.is_set(),
        )
        limits = LabLimits(max_cost_usd=settings.lab_max_cost_usd,
                           max_runtime_seconds=float(settings.lab_max_runtime_seconds))
        session = LabSession(self._lab.workspace, client=self._client, limits=limits, callbacks=callbacks)

        async def work() -> SessionOutcome:
            try:
                return await self._runtime.run_blocking(
                    session.handle, self.request.message, user=self._user.login, action=self.request.action,
                    model_spec=model_spec)
            finally:
                queue.put_nowait(None)

        async def emit_costs() -> None:
            while True:
                await asyncio.sleep(settings.llm_cost_emitter_interval_seconds)
                queue.put_nowait(CostEvent(**self._totals()))

        worker = asyncio.create_task(work(), name=f"lab-run-{self.run_id[:8]}")
        emitter = asyncio.create_task(emit_costs(), name=f"lab-cost-{self.run_id[:8]}")
        try:
            while (event := await queue.get()) is not None:
                yield self._sse(event)
        finally:
            emitter.cancel()
            if not worker.done():
                # The stream was closed first: stop the session and wait, so the lab is released only then.
                self._cancel_event.set()
                await asyncio.wait({worker})
                if worker.exception() is not None:
                    logger.warning("BESSER Lab run %s failed after its stream ended: %r",
                                   self.run_id, worker.exception())
        while not queue.empty():
            event = queue.get_nowait()
            if event is not None:
                yield self._sse(event)

        try:
            outcome = worker.result()
        except InvalidApiKeyError as exc:
            yield self._sse(ErrorEvent(code="INVALID_KEY", message=f"The lab's LLM key was refused: {exc}"))
            return
        except UpstreamLLMError as exc:
            yield self._sse(ErrorEvent(code="UPSTREAM_LLM", message=str(exc)))
            return
        except LabWorkspaceError as exc:
            yield self._sse(ErrorEvent(code="BAD_REQUEST", message=str(exc)))
            return

        lab = await self._runtime.run_blocking(lab_done, outcome)
        await self._runtime.run_blocking(self._lab.remember_run, lab)
        yield self._sse(CostEvent(**self._totals()))
        # A stop is the user's own: no warning; ``done`` says it (``cancelled``, ``lab.stopReason``).
        if outcome.stop_reason in _EARLY_STOPS:
            code, message = _EARLY_STOPS[outcome.stop_reason]
            yield self._sse(ErrorEvent(code=code, message=message))
        yield self._sse(done_event(self.run_id, lab))
