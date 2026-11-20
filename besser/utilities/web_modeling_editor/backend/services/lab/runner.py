"""One BESSER Lab run: a message to a change through ``LabSession.handle``, streamed as Spec-Driven SSE events.

The route lets the run start (``LabService.begin_run``: the change's lock, a run slot, the cost cap) and hands
the :class:`~.service.LabRunTicket` to :class:`LabRunner`. The runner writes the message and ``running`` to the
change's thread, runs the session in a worker thread of the Spec-Driven runtime and streams its callbacks,
handed to the event loop with ``call_soon_threadsafe`` as ``services/spec_driven/runner.py`` does; a cost event
goes out every ``BESSER_LLM_COST_EMITTER_INTERVAL_SECONDS``. A build's work is committed as the login that asked
for it. However the stream ends, :meth:`LabRunner.stream` gives the ticket back (``LabService.end_run``: the run
record, the thread's last line, the audit line, the budget) and releases the run id that
``POST /spec-driven/cancel/{run_id}`` stops. A stopped run ends with ``done`` like any other, ``cancelled`` and
with the checks that ran.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, AsyncIterator, Optional

from besser.spec_driven_agent.errors import InvalidApiKeyError, UpstreamLLMError
from besser.spec_driven_agent.lab.contracts import (
    ChangeKind,
    IssueRef,
    LabCallbacks,
    LabLimits,
    NoteDraft,
    SessionOutcome,
)
from besser.spec_driven_agent.lab.session import LabSession
from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from besser.spec_driven_agent.providers.llm_client import create_llm_client, default_model
from besser.utilities.web_modeling_editor.backend.models.lab import (
    LabBuildReport,
    LabChangeCost,
    LabCheck,
    LabChecklistItem,
    LabDone,
    LabIssue,
    LabMessageRequest,
    LabModelProposal,
    LabNoteProposal,
    LabPlan,
    LabPullRequest,
    LabRunRecord,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser
from besser.utilities.web_modeling_editor.backend.services.lab.github_reader import LabGitHubReader
from besser.utilities.web_modeling_editor.backend.services.lab.model_proposal import (
    SPEC_ERRORS,
    _change_diagram,
    model_spec_from_project,
    proposal_from_plan,
)
from besser.utilities.web_modeling_editor.backend.services.lab.pull_requests import PullRequestError
from besser.utilities.web_modeling_editor.backend.services.lab.service import (
    LabRefusal,
    LabRunTicket,
    LabService,
    repo_change_view,
)
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

# The changes whose steps follow a class diagram of the extension. For any other kind the project's class
# diagram is the user's own application model, not part of the change.
_KINDS_WITH_MODEL = (ChangeKind.METAMODEL, ChangeKind.DIAGRAM_TYPE)
# The card lists this many changed paths; ``changedFileCount`` covers the rest (as a Spec-Driven run does).
_MAX_CHANGED_FILES = 50
_STEPS = ("plan", "build")
# A run record's stop reason for a run that failed before its session gave one: the list has no "failed".
_FAILED_STOP_REASON = "api_error"

# A run that stopped early but kept its work: a warning before ``done``.
_EARLY_STOPS = {
    "cost_cap": ("COST_CAP", "The run reached its cost cap; what it did so far is in the working copy."),
    "timeout": ("TIMEOUT", "The run reached its time limit; what it did so far is in the working copy."),
    "max_turns": ("INCOMPLETE", "The run reached its turn limit; what it did so far is in the working copy."),
    "api_error": ("INCOMPLETE", "The model provider failed during the run; what it did so far is in the "
                                "working copy."),
}


def note_proposals(run_id: str, drafts: list[NoteDraft]) -> list[LabNoteProposal]:
    """The notes a build proposed, each with an id unique within its change (``<run id[:8]>-<n>``)."""
    return [LabNoteProposal(id=f"{run_id[:8]}-{number}", kind=draft.kind, title=draft.title, body=draft.body,
                            repos=list(draft.repos), changeKinds=list(draft.change_kinds),
                            docsTarget=draft.docs_target, why=draft.why)
            for number, draft in enumerate(drafts, start=1)]


def lab_done(outcome: SessionOutcome, *, run_id: str, cost: LabChangeCost,
             pull_requests: Optional[list[LabPullRequest]] = None,
             issues: Optional[list[LabIssue]] = None) -> LabDone:
    """The ``lab`` field of the ``done`` event for ``outcome``.

    ``cost`` is what the change has cost with this run; ``pull_requests`` and ``issues`` are the change's now.
    A plan whose class diagram cannot be converted gets the reason as an open question instead of a
    ``modelProposal``.
    """
    change, plan, build = outcome.change, outcome.plan, outcome.build
    # A plan that changes nothing proposes no model: there is no change for the editor to tie it to.
    proposal = proposal_from_plan(plan) if plan is not None and not plan.changes_nothing else None
    kind = change.kind if change is not None else plan.kind if plan is not None else ChangeKind.OTHER
    # A draft has no title until its first plan opens it: a plan that changes nothing keeps the plan's own.
    title = (change.title if change is not None else "") or (plan.title if plan is not None else "")
    report = build.report if build is not None else None
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
        report=LabBuildReport(summary=report.summary, deviations=list(report.deviations),
                              notDone=list(report.not_done)) if report is not None else None,
        pullRequests=list(pull_requests or []),
        cost=cost,
        issues=list(issues or []),
        finalText=build.loop.final_text if build is not None else "",
        noteProposals=note_proposals(run_id, outcome.note_proposals),
    )


def done_event(run_id: str, lab: LabDone) -> DoneEvent:
    """A ``done`` event the Spec-Driven run card renders: nothing to download, the changed paths per repository."""
    changed = [f"{change.repo}/{file.path}" for change in lab.repoChanges for file in change.files]
    return DoneEvent(runId=run_id, downloadUrl="", fileName="", isZip=False, recipe={}, mode="modify",
                     changedFiles=changed[:_MAX_CHANGED_FILES], changedFileCount=len(changed),
                     cancelled=lab.stopReason == "cancelled", lab=lab)


class LabRunner:
    """Drive one lab run of ``request`` for ``user`` and yield its SSE frames.

    ``ticket`` is what ``LabService.begin_run`` gave the run; :meth:`stream` gives it back when it ends.
    """

    def __init__(self, request: LabMessageRequest, *, user: LabUser, lab: LabService, runtime: SpecDrivenRuntime,
                 settings: BackendSettings, ticket: LabRunTicket, cancel_event: asyncio.Event) -> None:
        self.request = request
        self.ticket = ticket
        self.run_id = ticket.run_id
        self._user = user
        self._lab = lab
        self._runtime = runtime
        self._settings = settings
        self._cancel_event = cancel_event
        self._api_key = settings.lab_llm_api_key.get_secret_value()
        self._client: Any = None
        self._turns = 0
        self._started_at = time.monotonic()
        if request.review is not None:
            ticket.step = "build"
        elif request.action in _STEPS:
            ticket.step = request.action
        # How the run ended, for its ledgers: the session's outcome, the done payload, or an error.
        self._outcome: Optional[SessionOutcome] = None
        self._done: Optional[LabDone] = None
        self._error: Optional[dict[str, str]] = None
        # Warnings that go out before ``done`` (a stop before the end, work that could not be committed).
        self._warnings: list[ErrorEvent] = []

    async def stream(self) -> AsyncIterator[bytes]:
        """The run's SSE frames; when they end, the run's ledgers are written and the run id released."""
        try:
            # Closed explicitly: when this stream is closed early, the session is stopped and waited for
            # (``_frames``' finally) before the run is given back below.
            async with contextlib.aclosing(self._frames()) as frames:
                async for frame in frames:
                    yield frame
        except Exception:
            logger.exception("BESSER Lab run %s crashed", self.run_id)
            self._error = {"code": "INTERNAL", "message": "Internal server error"}
            yield self._sse(ErrorEvent(code="INTERNAL", message="Internal server error"))
        finally:
            # Nothing here waits before the change and the run slot are given back, so a cancelled stream
            # cannot leave them taken.
            self._end_run()
            await self._runtime.active_runs.release(self.run_id, self._cancel_event)

    async def abandon(self, reason: str) -> None:
        """Give the run back when its stream never started (the route could not start it); ``reason`` is logged
        as the run's error."""
        self._error = {"code": "INTERNAL", "message": reason}
        self._end_run()
        await self._runtime.active_runs.release(self.run_id, self._cancel_event)

    # -- Ending ------------------------------------------------------------------------------------------------

    def _end_run(self) -> None:
        outcome = self._outcome
        if outcome is not None and self._error is None:
            status = "cancelled" if outcome.stop_reason == "cancelled" else "done"
        elif self._error is None and self._cancel_event.is_set():
            status = "cancelled"
        else:
            status = "failed"
        if status == "failed" and self._error is None:
            self._error = {"code": "INTERNAL", "message": "The run ended before it reported a result."}
        step = (outcome.step if outcome is not None else None) or self.ticket.step or "plan"
        stop = outcome.stop_reason if outcome is not None else "cancelled" if status == "cancelled" \
            else _FAILED_STOP_REASON
        totals = self._totals()
        record = LabRunRecord(runId=self.run_id, step=step, login=self._user.login, model=self._model_name(step),
                              usd=totals["usd"], turns=totals["turns"], elapsedSeconds=totals["elapsedSeconds"],
                              stopReason=stop, startedAt=self.ticket.started_at)
        try:
            self._lab.end_run(self.ticket, status=status, record=record, done=self._done, error=self._error)
        except Exception:  # the release in end_run's own finally has run; the stream must still end
            logger.exception("BESSER Lab run %s: its end could not be recorded", self.run_id)

    def _model_name(self, step: str) -> str:
        settings = self._settings
        model = settings.lab_llm_model or default_model(settings.lab_llm_provider) or ""
        return (settings.lab_plan_llm_model or model) if step == "plan" else model

    # -- Helpers -----------------------------------------------------------------------------------------------

    def _sse(self, event: BaseSseEvent) -> bytes:
        return format_sse(event, secrets=(self._api_key,) if self._api_key else ())

    def _fail(self, code: str, message: str) -> bytes:
        self._error = {"code": code, "message": message}
        return self._sse(ErrorEvent(code=code, message=message))

    def _totals(self) -> dict[str, Any]:
        usage = getattr(self._client, "usage", None)
        return {"usd": round(float(getattr(usage, "estimated_cost", 0.0) or 0.0), 4), "turns": self._turns,
                "elapsedSeconds": round(time.monotonic() - self._started_at, 2)}

    def _record_start(self) -> None:
        """The message and ``running`` in the change's thread (blocking: files)."""
        project, change_id = self.request.project, self.request.changeId
        diagram = _change_diagram(project, change_id) if project is not None else None
        action = "build" if self.request.review is not None else self.request.action
        thread = self._lab.thread(change_id)
        try:
            thread.add_message(login=self._user.login, text=self.request.message, action=action,
                               model=diagram.model if diagram is not None and diagram.model else None)
            thread.run_started(self.run_id, self._user.login)
        except OSError as exc:
            logger.warning("BESSER Lab run %s: the thread of change %s was not written: %s", self.run_id, change_id,
                           exc)

    def _model_spec(self) -> Optional[dict]:
        """The change's class diagram in the project, as the user corrected it, for a change that follows one.

        A plan revision starts from it as the build does, so a correction made on the canvas is kept.
        """
        if self.request.project is None:
            return None
        change = self._lab.workspace.get_change(self.request.changeId)
        if change.kind not in _KINDS_WITH_MODEL:
            return None
        return model_spec_from_project(self.request.project, change.id)

    def _conclude(self, outcome: SessionOutcome) -> LabDone:
        """Commit a build's work as the user, then the ``done`` payload (blocking: git)."""
        workspace = self._lab.workspace
        change_id = self.request.changeId
        if outcome.step == "build":
            try:
                change = workspace.get_change(change_id)
                if change.status == "open":
                    workspace.commit_all(change, change.title, login=self._user.login)
            except LabWorkspaceError as exc:
                logger.warning("BESSER Lab run %s: the build's work was not committed: %s", self.run_id, exc)
                self._warnings.append(ErrorEvent(code="INCOMPLETE", message=(
                    f"The build's work is in the change's working copy but was not committed: {exc}")))
        totals = self._totals()
        try:
            detail = self._lab.change_detail(change_id)
        except LabRefusal:  # dropped while it ran
            detail = None
        before = detail.cost if detail is not None else LabChangeCost(usd=0.0, turns=0, elapsedSeconds=0.0, runs=0)
        cost = LabChangeCost(usd=round(before.usd + totals["usd"], 6), turns=before.turns + totals["turns"],
                             elapsedSeconds=round(before.elapsedSeconds + totals["elapsedSeconds"], 3),
                             runs=before.runs + 1)
        return lab_done(outcome, run_id=self.run_id, cost=cost,
                        pull_requests=detail.pullRequests if detail is not None else None,
                        issues=detail.view.issues if detail is not None and detail.view is not None else None)

    async def _settle(self, worker: asyncio.Task) -> None:
        """What the session's task ended with: an outcome (committed and turned into ``done``) or an error."""
        exc = worker.exception()
        if exc is None:
            self._outcome = worker.result()
            self._done = await self._runtime.run_blocking(self._conclude, self._outcome)
        elif isinstance(exc, InvalidApiKeyError):
            self._error = {"code": "INVALID_KEY", "message": f"The lab's LLM key was refused: {exc}"}
        elif isinstance(exc, UpstreamLLMError):
            self._error = {"code": "UPSTREAM_LLM", "message": str(exc)}
        elif isinstance(exc, LabWorkspaceError):
            self._error = {"code": "BAD_REQUEST", "message": str(exc)}
        else:
            logger.error("BESSER Lab run %s crashed", self.run_id, exc_info=exc)
            self._error = {"code": "INTERNAL", "message": "Internal server error"}

    # -- The run -----------------------------------------------------------------------------------------------

    async def _frames(self) -> AsyncIterator[bytes]:
        settings = self._settings
        provider = settings.lab_llm_provider
        await self._runtime.run_blocking(self._record_start)
        yield self._sse(StartEvent(
            runId=self.run_id, provider=provider, llmModel=settings.lab_llm_model or default_model(provider) or "",
            maxCost=self.ticket.cost_cap_usd, maxRuntime=int(settings.lab_max_runtime_seconds)))

        try:
            self._client = await self._runtime.run_blocking(
                create_llm_client, provider, api_key=self._api_key or None, model=settings.lab_llm_model)
        except ValueError as exc:  # InvalidApiKeyError included
            yield self._fail("INVALID_KEY", f"The lab's LLM key was refused: {exc}")
            return
        message, action = self.request.message, self.request.action
        if self.request.review is not None:
            # A review task is a build whose message the server composes from a fresh read of the pull requests.
            try:
                message = await self._lab.review_task_message(self._user, self.request.changeId,
                                                              self.request.review, self.request.message)
            except (PullRequestError, LabRefusal) as exc:
                yield self._fail("BAD_REQUEST", f"The pull requests could not be read: {exc}")
                return
            action = "build"
        try:
            model_spec = await self._runtime.run_blocking(self._model_spec)
        except SPEC_ERRORS as exc:
            yield self._fail("BAD_REQUEST", f"The project's class diagram could not be read: {exc}")
            return
        except LabWorkspaceError as exc:
            yield self._fail("BAD_REQUEST", str(exc))
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

        def on_phase(phase: str, text: str) -> None:
            if phase in _STEPS:
                self.ticket.step = phase  # the home's runs in progress and a busy refusal name it
            put(PhaseEvent(phase=phase, message=text))

        callbacks = LabCallbacks(on_progress=on_progress, on_text=on_text, on_phase=on_phase,
                                 should_continue=lambda: not self._cancel_event.is_set())
        limits = LabLimits(max_cost_usd=self.ticket.cost_cap_usd,
                           max_runtime_seconds=float(settings.lab_max_runtime_seconds))
        reader = LabGitHubReader(GitHubService(self._user.access_token), self._lab.workspace.base_repos, loop,
                                 login=self._user.login)
        session = LabSession(self._lab.workspace, client=self._client, environments=self._lab.environments,
                             limits=limits, callbacks=callbacks, sandbox_policy=self._lab.sandbox_policy(),
                             github_reader=reader, memory=self._lab.memory, plan_model=settings.lab_plan_llm_model)
        issue = self.request.issue
        issues = [IssueRef(issue.repo, issue.number)] if issue is not None else []

        async def work() -> SessionOutcome:
            try:
                return await self._runtime.run_blocking(
                    session.handle, message, change_id=self.request.changeId, login=self._user.login,
                    action=action, model_spec=model_spec, issues=issues)
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
                # The stream was closed first: stop the session and wait, so the run is given back only then.
                self._cancel_event.set()
                await asyncio.wait({worker})
            await self._settle(worker)
        while not queue.empty():
            event = queue.get_nowait()
            if event is not None:
                yield self._sse(event)

        if self._error is not None:
            yield self._sse(ErrorEvent(**self._error))
            return
        yield self._sse(CostEvent(**self._totals()))
        # A stop is the user's own: no warning; ``done`` says it (``cancelled``, ``lab.stopReason``).
        if self._outcome.stop_reason in _EARLY_STOPS:
            code, text = _EARLY_STOPS[self._outcome.stop_reason]
            yield self._sse(ErrorEvent(code=code, message=text))
        for warning in self._warnings:
            yield self._sse(warning)
        yield self._sse(done_event(self.run_id, self._done))
