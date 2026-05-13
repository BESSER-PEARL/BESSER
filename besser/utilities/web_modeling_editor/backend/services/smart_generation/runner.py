"""Runner that bridges ``LLMOrchestrator`` to an SSE event stream.

The orchestrator runs synchronously in a worker thread (via
``asyncio.to_thread``) and fires ``on_text`` / ``on_progress`` callbacks
from that thread. The runner uses ``loop.call_soon_threadsafe`` to push
pydantic event objects onto an ``asyncio.Queue`` owned by the main
event loop, then drains the queue from the async generator body and
yields SSE-framed bytes to the FastAPI ``StreamingResponse``.

A sibling task emits periodic ``CostEvent``s by reading
``client.usage.estimated_cost`` every ``LLM_COST_EMITTER_INTERVAL_SECONDS``.

On successful completion the output directory is walked, zipped if it
contains more than one user file, and registered with
``SMART_RUN_REGISTRY`` for the sibling download endpoint to serve.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from pydantic import ValidationError as PydanticValidationError

from besser.generators.llm.llm_client import create_llm_client
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    LLM_COST_EMITTER_INTERVAL_SECONDS,
    LLM_DOWNLOAD_TTL_SECONDS,
    LLM_ENABLE_CHECKPOINTING,
    LLM_ENABLE_TRACING,
    LLM_TEMP_DIR_PREFIX,
)
from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    ValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.model_assembly import (
    assemble_models_from_project,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.sse_events import (
    BaseSseEvent,
    CostEvent,
    DoneEvent,
    ErrorEvent,
    PhaseEvent,
    PhaseUpdateEvent,
    StartEvent,
    TextDeltaEvent,
    ToolCallEvent,
    format_sse,
)

logger = logging.getLogger(__name__)


_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}

_REGISTRY_SWEEP_INTERVAL_SECONDS = 60

# Upper bound on queued SSE events so a runaway worker can't exhaust
# memory. 2048 is comfortably above the steady-state queue depth (one
# event every ~50ms for a very chatty run × a drain loop that handles
# each event in well under that), while still capping worst-case
# memory at a few MB.
_EVENT_QUEUE_MAXSIZE = 2048

# Max size in bytes of the .besser_recipe.json file that's embedded in
# the final DoneEvent. A malformed or maliciously huge recipe would
# otherwise bloat the last SSE frame beyond any reasonable size.
_MAX_RECIPE_BYTES = 256 * 1024  # 256 KB


class _EmptyGenerationError(Exception):
    """Raised when the orchestrator returned but produced no output files.

    Not a user-facing error — mapped to the ``INTERNAL`` SSE code so
    the client knows something is wrong on our side, distinct from an
    upstream LLM failure (``UPSTREAM_LLM``) or an invalid key
    (``INVALID_KEY``).
    """


# ---------------------------------------------------------------------
# Registry of completed runs awaiting download
# ---------------------------------------------------------------------


@dataclass
class SmartRunEntry:
    """One entry in the download registry."""

    file_path: str
    file_name: str
    is_zip: bool
    temp_dir: str
    created_at: float


class SmartRunRegistry:
    """In-memory, TTL-bounded map of ``run_id → SmartRunEntry``.

    The sibling ``GET /download-smart/{run_id}`` endpoint pops the entry
    on first download (single-use semantics). Anything not claimed within
    ``LLM_DOWNLOAD_TTL_SECONDS`` is swept by the periodic task started
    from the backend's lifespan.
    """

    def __init__(self) -> None:
        self._entries: dict[str, SmartRunEntry] = {}
        self._lock = asyncio.Lock()

    async def put(self, run_id: str, entry: SmartRunEntry) -> None:
        async with self._lock:
            self._entries[run_id] = entry

    async def pop(self, run_id: str) -> Optional[SmartRunEntry]:
        async with self._lock:
            return self._entries.pop(run_id, None)

    async def periodic_sweep(
        self,
        ttl_seconds: int = LLM_DOWNLOAD_TTL_SECONDS,
        interval_seconds: int = _REGISTRY_SWEEP_INTERVAL_SECONDS,
    ) -> None:
        """Loop forever, sweeping expired entries every ``interval_seconds``.

        Caller is expected to start this as an ``asyncio.Task`` in the
        FastAPI lifespan and cancel it on shutdown.
        """
        while True:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise

            now = time.time()
            expired: list[tuple[str, SmartRunEntry]] = []
            async with self._lock:
                for run_id, entry in list(self._entries.items()):
                    # Clamp against backwards clock movements: if the
                    # stored ``created_at`` is in the future (system
                    # clock was adjusted after the entry was put), age
                    # is treated as 0 rather than a negative value that
                    # would let the entry live forever.
                    age = max(0.0, now - entry.created_at)
                    if age > ttl_seconds:
                        expired.append((run_id, entry))
                        self._entries.pop(run_id, None)

            for run_id, entry in expired:
                try:
                    shutil.rmtree(entry.temp_dir, ignore_errors=True)
                except Exception:
                    logger.exception(
                        "Failed to remove temp dir for expired run %s", run_id
                    )
            if expired:
                logger.info(
                    "SmartRunRegistry: swept %d expired smart-gen run(s)",
                    len(expired),
                )


SMART_RUN_REGISTRY = SmartRunRegistry()


# ---------------------------------------------------------------------
# Global concurrency cap
# ---------------------------------------------------------------------
# Each in-flight smart-generation run holds a worker thread, a temp
# dir, and a long-lived SSE connection. The semaphore protects against
# a malicious or buggy client spawning runs faster than they finish,
# which would otherwise exhaust threads / disk. The router tries a
# non-blocking acquire first and returns HTTP 429 if the cap is
# exhausted — we deliberately do NOT queue because queued SSE
# connections just look hung from the browser's perspective.

_CONCURRENCY_SEMAPHORE: asyncio.Semaphore | None = None


def _get_concurrency_semaphore() -> asyncio.Semaphore:
    """Lazy-construct the semaphore on first use.

    We can't build it at import time because the constant might be
    monkeypatched by tests and because asyncio.Semaphore attaches
    itself to the current running loop on creation.
    """
    global _CONCURRENCY_SEMAPHORE
    if _CONCURRENCY_SEMAPHORE is None:
        from besser.utilities.web_modeling_editor.backend.constants.constants import (
            LLM_MAX_CONCURRENT_RUNS,
        )
        _CONCURRENCY_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENT_RUNS)
    return _CONCURRENCY_SEMAPHORE


def _reset_concurrency_semaphore_for_tests() -> None:
    """Allow tests to rebuild the semaphore after patching the cap.

    Not part of the public API — prefix is underscore. Exported so the
    test module can call it without hacking private globals.
    """
    global _CONCURRENCY_SEMAPHORE
    _CONCURRENCY_SEMAPHORE = None


def try_acquire_run_slot() -> bool:
    """Non-blocking attempt to reserve a concurrency slot.

    Returns True on success (caller must eventually call
    ``release_run_slot``), False when the cap is reached.
    """
    sem = _get_concurrency_semaphore()
    return sem.locked() is False and _try_acquire_nowait(sem)


def _try_acquire_nowait(sem: asyncio.Semaphore) -> bool:
    # asyncio.Semaphore lacks a non-blocking acquire, so inspect the
    # private counter. Safe because we never await between the check
    # and the decrement — both happen on the event loop thread.
    if sem._value <= 0:
        return False
    sem._value -= 1
    return True


def release_run_slot() -> None:
    """Return a concurrency slot. Idempotent: if the counter is already
    at its max, this is a no-op rather than a crash.
    """
    sem = _get_concurrency_semaphore()
    sem.release()


# ---------------------------------------------------------------------
# Crash-recovery helpers
# ---------------------------------------------------------------------


def _locate_run_temp_dir(run_id: str) -> str | None:
    """Locate a smart-gen temp directory by ``run_id``.

    ``mkdtemp`` names temp dirs as ``{prefix}{run_id}_{suffix}`` so we
    can recover them after a process restart by scanning the system
    temp dir. Returns ``None`` when no matching directory exists — the
    caller surfaces that as "no run to resume".

    Only directories carrying a ``.besser_checkpoint.json`` are
    considered valid resume candidates; a lingering temp dir from a
    crashed mkdtemp-but-never-checkpointed run is ignored.
    """
    if not run_id:
        return None
    from besser.generators.llm.checkpoint import CHECKPOINT_FILENAME

    tempdir = tempfile.gettempdir()
    prefix = f"{LLM_TEMP_DIR_PREFIX}{run_id}_"
    try:
        candidates = [
            entry for entry in os.listdir(tempdir)
            if entry.startswith(prefix)
        ]
    except OSError:
        return None

    for name in candidates:
        full = os.path.join(tempdir, name)
        if os.path.isdir(full) and os.path.isfile(
            os.path.join(full, CHECKPOINT_FILENAME),
        ):
            return full
    return None


# ---------------------------------------------------------------------
# Live-run cancellation registry
# ---------------------------------------------------------------------

# Maps ``run_id`` of an in-flight smart-gen run to an asyncio.Event that
# the cancel endpoint can set. The runner registers itself here on start
# and removes itself in its ``finally`` block. The orchestrator polls
# the event between turns to stop cleanly. Closing the SSE stream alone
# is not enough — the orchestrator runs in a worker thread and would
# keep burning the user's BYOK budget until natural completion.
_ACTIVE_RUNS: dict[str, asyncio.Event] = {}
_ACTIVE_RUNS_LOCK = asyncio.Lock()


async def request_cancellation(run_id: str) -> bool:
    """Signal a live run to stop at its next turn boundary.

    Returns True if the run was found and signalled, False if no such
    run is currently active (already completed, never existed, or the
    download already happened).
    """
    async with _ACTIVE_RUNS_LOCK:
        event = _ACTIVE_RUNS.get(run_id)
    if event is None:
        return False
    event.set()
    return True


async def _register_active_run(run_id: str) -> asyncio.Event:
    event = asyncio.Event()
    async with _ACTIVE_RUNS_LOCK:
        _ACTIVE_RUNS[run_id] = event
    return event


async def _deregister_active_run(run_id: str) -> None:
    async with _ACTIVE_RUNS_LOCK:
        _ACTIVE_RUNS.pop(run_id, None)


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


class SmartGenerationRunner:
    """Drive one smart-generation run and yield SSE events."""

    def __init__(
        self,
        request: SmartGenerateRequest,
        *,
        resume_run_id: Optional[str] = None,
    ) -> None:
        self.request = request
        # Resuming a prior run reuses its run_id so the client can keep
        # the same identifier across the crash. Fresh runs get a new
        # UUID. Either way the id is hex[32] so the path regex in the
        # cancel / download / resume routes accepts it.
        self.run_id = resume_run_id or uuid.uuid4().hex
        self._resume_run_id = resume_run_id
        self.temp_dir: Optional[str] = None
        self._started_at: Optional[float] = None

    async def generate_and_stream(
        self,
        http_request: Any | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Run the pipeline and yield SSE frames.

        The generator always emits at least a ``start`` event first so
        the client can confirm the connection is live even if the run
        fails immediately afterwards.

        If ``http_request`` (a Starlette/FastAPI ``Request``) is provided
        a watcher task polls ``is_disconnected()`` every second and
        triggers cancellation when the client drops the connection —
        otherwise the worker thread would keep calling the user's
        provider until natural completion, burning their BYOK budget
        after the browser tab is already closed.
        """
        loop = asyncio.get_running_loop()
        self._started_at = time.monotonic()
        llm_model = (
            self.request.llm_model or _DEFAULT_MODELS.get(self.request.provider, "")
        )

        # ---- 1. Emit start (no api_key anywhere in the payload) --------
        yield format_sse(StartEvent(
            runId=self.run_id,
            provider=self.request.provider,
            llmModel=llm_model,
            maxCost=self.request.max_cost_usd,
            maxRuntime=self.request.max_runtime_seconds,
        ))

        # ---- 2. Create (or recover) the temp dir -----------------------
        # When resuming, we reattach to the temp dir from the crashed run
        # so the orchestrator sees its existing output + checkpoint. The
        # caller is expected to have verified (via the resume endpoint)
        # that the dir is still there; we double-check and surface a
        # clear BAD_REQUEST if it isn't — the user should start fresh.
        if self._resume_run_id:
            existing = _locate_run_temp_dir(self._resume_run_id)
            if existing is None:
                yield format_sse(ErrorEvent(
                    code="BAD_REQUEST",
                    message=(
                        "No recoverable workspace for this run_id — the "
                        "crashed run's temp directory has been swept. "
                        "Start a fresh run."
                    ),
                ))
                return
            self.temp_dir = existing
        else:
            try:
                self.temp_dir = tempfile.mkdtemp(
                    prefix=f"{LLM_TEMP_DIR_PREFIX}{self.run_id}_"
                )
            except OSError as exc:
                yield format_sse(ErrorEvent(
                    code="INTERNAL",
                    message="Failed to allocate a workspace for this run",
                ))
                logger.exception("mkdtemp failed for smart-generate run %s: %s", self.run_id, exc)
                return

        # ---- 3. Assemble BUML models from the project ------------------
        try:
            assembled = await asyncio.to_thread(
                assemble_models_from_project,
                self.request.project,
                getattr(self.request, "primary_kind_override", None),
            )
        except (
            ConversionError,
            ValidationError,
            ValueError,
            PydanticValidationError,
        ) as exc:
            # All four map to "the user sent us a malformed project" and
            # should surface as a 400-equivalent BAD_REQUEST event. We
            # explicitly include pydantic.ValidationError so a malformed
            # ProjectInput doesn't fall through to the INTERNAL branch.
            yield format_sse(ErrorEvent(code="BAD_REQUEST", message=str(exc)))
            self._cleanup_temp_dir()
            return
        except Exception:
            logger.exception(
                "Unexpected error while assembling models for smart-generate run %s",
                self.run_id,
            )
            yield format_sse(ErrorEvent(
                code="INTERNAL",
                message="Internal server error",
            ))
            self._cleanup_temp_dir()
            return

        yield format_sse(PhaseEvent(phase="select", message="Selecting generator"))

        # ---- 4. Build the LLM client (may raise on invalid key) --------
        try:
            client = await asyncio.to_thread(
                create_llm_client,
                provider=self.request.provider,
                api_key=self.request.resolved_api_key(),
                model=self.request.llm_model,
            )
        except ValueError as exc:
            yield format_sse(ErrorEvent(code="INVALID_KEY", message=str(exc)))
            self._cleanup_temp_dir()
            return
        except Exception as exc:
            logger.exception("Failed to build LLM client: %s", exc)
            yield format_sse(ErrorEvent(
                code="INTERNAL",
                message="Internal server error",
            ))
            self._cleanup_temp_dir()
            return

        # ---- 5. Build the orchestrator with thread-safe callbacks ------
        # Bounded queue so a runaway orchestrator can't pile up events
        # faster than the SSE consumer drains them. On full, events are
        # dropped rather than blocking the worker thread — with a log
        # line so we can detect the condition.
        queue: asyncio.Queue[Optional[BaseSseEvent]] = asyncio.Queue(
            maxsize=_EVENT_QUEUE_MAXSIZE,
        )
        # Flag — mutated from the worker thread inside ``on_progress``.
        # Python assignments to dict entries are atomic with respect to
        # the GIL, so we don't need a lock for a one-shot toggle.
        phase_state: dict[str, bool] = {"customize_sent": False}

        def _put(event: BaseSseEvent) -> None:
            try:
                loop.call_soon_threadsafe(_put_in_loop, event)
            except RuntimeError:
                # Event loop is closed (e.g. client aborted). Drop
                # silently — the consumer is already gone.
                pass

        def _put_in_loop(event: BaseSseEvent) -> None:
            """Runs on the main event loop; safe to call put_nowait here."""
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event and retry once. Losing a cost
                # tick or a tool_call is fine; what matters is that
                # the worker keeps making progress and the consumer
                # doesn't see the queue stall.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "smart-gen event queue full even after drop; "
                        "event of type %s was dropped",
                        type(event).__name__,
                    )

        def on_text(delta: str) -> None:
            _put(TextDeltaEvent(delta=delta))

        def on_phase_details(phase: str, details: str) -> None:
            """Attach details (e.g. gap task list) to an existing phase row.

            Currently only emitted for the gap phase by gap_analyzer
            after the planning LLM call returns. The frontend renders
            this behind a chevron on the smart-gen card.
            """
            if not details:
                return
            try:
                _put(PhaseUpdateEvent(phase=phase, details=details))
            except Exception:
                logger.debug(
                    "PhaseUpdateEvent emission failed (phase=%s)", phase, exc_info=True
                )

        def on_progress(turn: int, tool: str, status: str) -> None:
            if tool == "validation":
                _put(PhaseEvent(phase="validate", message=status))
                return
            if tool == "gap_analysis":
                # Sentinel emitted by gap_analyzer.analyze_gaps_via_llm
                # right before the planning LLM call. Surfaces the gap
                # phase in the smart-gen card; without this the user
                # sees a silent jump from `generate` to `customize`.
                _put(PhaseEvent(phase="gap", message="Analysing gaps"))
                return
            if turn == 0:
                if tool == "__skipped__":
                    # Phase 1 skipped — surface a clear reason on the
                    # smart-gen card so users don't think the run jumped
                    # straight to gap analysis by mistake.
                    if status == "no_generator":
                        msg = (
                            "skipped — no deterministic generator for this "
                            "stack; the LLM will scaffold from scratch"
                        )
                    elif status == "no_model":
                        msg = (
                            "skipped — no class diagram or quantum circuit "
                            "to build from"
                        )
                    else:
                        msg = "skipped"
                    _put(PhaseEvent(phase="generate", message=msg))
                    return
                _put(PhaseEvent(phase="generate", message=f"running {tool}"))
                return
            # turn >= 1 means we're inside the LLM customization loop.
            # Emit the `customize` phase marker once, before the first
            # tool_call, so the frontend can show the transition.
            if not phase_state["customize_sent"]:
                phase_state["customize_sent"] = True
                _put(
                    PhaseEvent(
                        phase="customize",
                        message="LLM customising generator output",
                    )
                )
            _put(ToolCallEvent(turn=turn, tool=tool, status="executing"))

        # Register this run for cancellation NOW that we've cleared all
        # the early-return paths (mkdtemp / model assembly / LLM client
        # build). ``POST /cancel-smart-gen/{run_id}`` will set this
        # event and the orchestrator will stop at the next turn boundary.
        cancel_event = await _register_active_run(self.run_id)

        # Bridge the asyncio.Event into a thread-safe boolean check the
        # synchronous orchestrator can poll between turns.
        def should_continue() -> bool:
            return not cancel_event.is_set()

        orchestrator = LLMOrchestrator(
            llm_client=client,
            domain_model=assembled.domain_model,
            gui_model=assembled.gui_model,
            agent_model=assembled.agent_model,
            agent_config=assembled.agent_config,
            object_model=assembled.object_model,
            state_machines=assembled.state_machines,
            quantum_circuit=assembled.quantum_circuit,
            output_dir=self.temp_dir,
            max_cost_usd=self.request.max_cost_usd,
            max_runtime_seconds=self.request.max_runtime_seconds,
            on_progress=on_progress,
            on_text=on_text,
            on_phase_details=on_phase_details,
            use_streaming=True,
            should_continue=should_continue,
            # Carry through the assembled primary so the orchestrator
            # doesn't re-detect it. ``assembled.primary_kind`` already
            # honours the user's override (if any) courtesy of
            # assemble_models_from_project.
            primary_kind=assembled.primary_kind,
            run_id=self.run_id,
            # Honour the deploy-wide feature flags. The orchestrator
            # defaults to both on; ops can disable per deploy via the
            # BESSER_LLM_ENABLE_* env vars without a code change.
            enable_tracing=LLM_ENABLE_TRACING,
            enable_checkpointing=LLM_ENABLE_CHECKPOINTING,
        )

        # ---- 6. Spawn the worker + the cost emitter --------------------
        async def cost_emitter() -> None:
            while True:
                try:
                    await asyncio.sleep(LLM_COST_EMITTER_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
                elapsed = time.monotonic() - (self._started_at or time.monotonic())
                try:
                    cost_usd = float(client.usage.estimated_cost)
                except Exception:
                    cost_usd = 0.0
                try:
                    queue.put_nowait(CostEvent(
                        usd=round(cost_usd, 4),
                        turns=getattr(orchestrator, "total_turns", 0),
                        elapsedSeconds=round(elapsed, 2),
                        servedModel=getattr(client.usage, "served_model", None),
                    ))
                except Exception:
                    pass

        async def run_orchestrator() -> str:
            try:
                if self._resume_run_id:
                    return await asyncio.to_thread(
                        orchestrator.resume, self.request.instructions
                    )
                return await asyncio.to_thread(
                    orchestrator.run, self.request.instructions
                )
            finally:
                # Sentinel: tells the drain loop to stop waiting for
                # events. If this ever fails, log loudly — the drain
                # loop would otherwise hang until the worker_task
                # completion race path fires, which is less deterministic.
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    # Clear one slot and retry; a single slot is always
                    # enough for the sentinel even if everything else
                    # is queued up.
                    try:
                        queue.get_nowait()
                        queue.put_nowait(None)
                    except Exception:
                        logger.error(
                            "Failed to enqueue smart-gen sentinel; "
                            "drain loop will fall back to worker-done race path"
                        )
                except Exception:
                    logger.exception(
                        "Unexpected error enqueueing smart-gen sentinel"
                    )

        emitter_task = asyncio.create_task(cost_emitter(), name="smart-gen-cost-emitter")
        worker_task = asyncio.create_task(run_orchestrator(), name="smart-gen-worker")

        # Watch for client disconnect. When the browser closes the SSE
        # stream we flip the cancel_event so the orchestrator stops at
        # its next turn boundary instead of silently burning the user's
        # LLM budget. Polling at 1Hz is well under the cost of a single
        # LLM turn so we don't waste visible wallclock time.
        async def disconnect_watcher() -> None:
            if http_request is None:
                return
            try:
                while not cancel_event.is_set():
                    try:
                        disconnected = await http_request.is_disconnected()
                    except Exception:
                        # If the transport is in a weird state, stop
                        # watching rather than spamming exceptions.
                        return
                    if disconnected:
                        logger.info(
                            "smart-gen client disconnected, cancelling run %s",
                            self.run_id,
                        )
                        cancel_event.set()
                        return
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise

        disconnect_task = asyncio.create_task(
            disconnect_watcher(), name="smart-gen-disconnect-watcher",
        )

        # ---- 7. Drain the queue, yielding events ----------------------
        worker_exception: Optional[BaseException] = None
        result_path: Optional[str] = None

        try:
            while True:
                get_task = asyncio.create_task(queue.get())
                try:
                    done, _pending = await asyncio.wait(
                        {get_task, worker_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    get_task.cancel()
                    raise

                if get_task in done:
                    event = get_task.result()
                    if event is None:
                        # Sentinel: worker put it after finishing.
                        break
                    yield format_sse(event)
                else:
                    # Worker finished (or crashed) without a sentinel.
                    # Cancel the pending get_task; suppress only the
                    # expected CancelledError, not every exception.
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                    break

            # ---- 8. Collect worker result or exception -----------------
            try:
                result_path = await worker_task
            except _EmptyGenerationError as exc:
                worker_exception = exc
                logger.error(
                    "Smart-gen worker %s produced no output files", self.run_id
                )
                yield format_sse(ErrorEvent(code="INTERNAL", message=str(exc)))
            except ValueError as exc:
                worker_exception = exc
                yield format_sse(ErrorEvent(code="INVALID_KEY", message=str(exc)))
            except RuntimeError as exc:
                worker_exception = exc
                yield format_sse(ErrorEvent(code="UPSTREAM_LLM", message=str(exc)))
            except Exception as exc:
                worker_exception = exc
                logger.exception(
                    "Unexpected error in smart_generate worker %s", self.run_id
                )
                yield format_sse(ErrorEvent(
                    code="INTERNAL",
                    message="Internal server error",
                ))
        finally:
            emitter_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await emitter_task
            disconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect_task

            # Drain any events the emitter enqueued between the sentinel
            # being put and the task being cancelled. These are cost
            # ticks — not critical — but missing them can make the
            # final `cost` event stale by up to the emitter interval.
            while True:
                try:
                    leftover = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if leftover is None:
                    continue
                yield format_sse(leftover)

            # Always deregister so a future cancel call returns False
            # cleanly and the dict doesn't leak entries.
            await _deregister_active_run(self.run_id)

            # If the user cancelled, surface that explicitly so the
            # frontend stops waiting for `done` and shows a clear status.
            if cancel_event.is_set():
                yield format_sse(ErrorEvent(
                    code="CANCELLED",
                    message="Smart generation cancelled by user request",
                ))

        # ---- 9. Surface cost / runtime cap warnings -------------------
        if worker_exception is None and result_path:
            elapsed = time.monotonic() - (self._started_at or time.monotonic())
            try:
                final_cost = float(client.usage.estimated_cost)
            except Exception:
                final_cost = 0.0

            # Emit a final cost snapshot so the client always sees the
            # exact end-of-run usage, not the last periodic tick.
            yield format_sse(
                CostEvent(
                    usd=round(final_cost, 4),
                    turns=getattr(orchestrator, "total_turns", 0),
                    elapsedSeconds=round(elapsed, 2),
                    servedModel=getattr(client.usage, "served_model", None),
                )
            )

            if final_cost > self.request.max_cost_usd:
                yield format_sse(ErrorEvent(
                    code="COST_CAP",
                    message=(
                        f"Cost cap reached (${final_cost:.4f} > "
                        f"${self.request.max_cost_usd}). "
                        "Output may be incomplete."
                    ),
                ))
            if elapsed > self.request.max_runtime_seconds:
                yield format_sse(ErrorEvent(
                    code="TIMEOUT",
                    message=(
                        f"Runtime cap reached ({elapsed:.1f}s > "
                        f"{self.request.max_runtime_seconds}s). "
                        "Output may be incomplete."
                    ),
                ))

            # ---- 10. Package the result and emit `done` ---------------
            try:
                done_event, entry = await asyncio.to_thread(
                    self._package_result, result_path
                )
                await SMART_RUN_REGISTRY.put(self.run_id, entry)
                yield format_sse(done_event)
            except Exception:
                logger.exception(
                    "Failed to package smart-generate output for run %s", self.run_id
                )
                yield format_sse(ErrorEvent(
                    code="INTERNAL",
                    message="Failed to package generated output",
                ))
                self._cleanup_temp_dir()
        else:
            # Worker errored — temp dir is no longer useful.
            self._cleanup_temp_dir()

    # ------------------------------------------------------------------

    def _package_result(self, result_path: str) -> tuple[DoneEvent, SmartRunEntry]:
        """Walk the output dir, zip if needed, read the recipe.

        Returns a tuple of ``(DoneEvent, SmartRunEntry)`` so the caller
        can both yield the event downstream and register the entry in
        the download registry.
        """
        if self.temp_dir is None:
            raise RuntimeError("Runner has no temp_dir")

        recipe = self._read_recipe(result_path)

        # Collect every user file (skip internal artefacts like .besser_recipe.json)
        user_files: list[str] = []
        for root, _dirs, files in os.walk(result_path):
            for name in files:
                if name.startswith(".besser_"):
                    continue
                user_files.append(os.path.join(root, name))

        if not user_files:
            # Distinct from an LLM upstream failure — the orchestrator
            # returned successfully but produced no files. Raise a
            # ValueError so the caller maps it to INTERNAL (backend
            # issue) rather than UPSTREAM_LLM (provider issue).
            raise _EmptyGenerationError("Generation produced no output files")

        if len(user_files) == 1:
            single = user_files[0]
            file_name = os.path.basename(single)
            entry = SmartRunEntry(
                file_path=single,
                file_name=file_name,
                is_zip=False,
                temp_dir=self.temp_dir,
                created_at=time.time(),
            )
        else:
            zip_name = f"besser_smart_{self.run_id}.zip"
            zip_path = os.path.join(self.temp_dir, zip_name)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in user_files:
                    arcname = os.path.relpath(path, result_path)
                    zf.write(path, arcname)
            entry = SmartRunEntry(
                file_path=zip_path,
                file_name=zip_name,
                is_zip=True,
                temp_dir=self.temp_dir,
                created_at=time.time(),
            )

        event = DoneEvent(
            downloadUrl=f"/besser_api/download-smart/{self.run_id}",
            fileName=entry.file_name,
            isZip=entry.is_zip,
            recipe=recipe,
        )
        return event, entry

    def _read_recipe(self, result_path: str) -> dict:
        recipe_path = os.path.join(result_path, ".besser_recipe.json")
        if not os.path.isfile(recipe_path):
            return {}
        try:
            size = os.path.getsize(recipe_path)
        except OSError:
            return {}
        if size > _MAX_RECIPE_BYTES:
            logger.warning(
                "Skipping .besser_recipe.json for run %s: size %d exceeds %d bytes",
                self.run_id, size, _MAX_RECIPE_BYTES,
            )
            return {"warning": "recipe file exceeded size limit and was dropped"}
        try:
            with open(recipe_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except Exception:
            logger.exception(
                "Failed to read .besser_recipe.json for run %s", self.run_id
            )
        return {}

    def _cleanup_temp_dir(self) -> None:
        if self.temp_dir and os.path.isdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                logger.exception(
                    "Failed to clean up temp dir for run %s", self.run_id
                )
        self.temp_dir = None
