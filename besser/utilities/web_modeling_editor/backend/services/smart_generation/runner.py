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
from typing import Any, AsyncGenerator, Optional

from pydantic import ValidationError as PydanticValidationError

from besser.generators.llm.errors import (
    CheckpointMismatchError,
    EmptyInstructionsError,
    InvalidApiKeyError,
    UpstreamLLMError,
)
from besser.generators.llm.llm_client import DEFAULT_MODELS, create_llm_client
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    LLM_COST_EMITTER_INTERVAL_SECONDS,
    LLM_DOWNLOAD_TTL_SECONDS,
    LLM_ENABLE_AUTO_FIX,
    LLM_ENABLE_SHELL_TOOLS,
    LLM_ENABLE_CHECKPOINTING,
    LLM_ENABLE_TOOLCHAIN_VALIDATION,
    LLM_ENABLE_TRACING,
    LLM_TEMP_DIR_PREFIX,
    LLM_WATCHDOG_GRACE_SECONDS,
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


# Provider default models — single source in llm_client.DEFAULT_MODELS
# (kept under the old module-local name so existing imports keep working).
_DEFAULT_MODELS = DEFAULT_MODELS

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

# Build-output dirs + runtime/artifact FILE globs that must never reach the
# download zip OR a GitHub push. Fed to shutil.ignore_patterns (used by both
# the modify/continue-from-repo SEED copy and the push copy), which fnmatches
# base names, so file globs work alongside the directory names. The file
# globs matter because a run leaves artifacts IN the workspace that the seed
# would otherwise carry into the next run and the push would commit: the
# download zip itself (besser_smart_*.zip) and the SQLite DB that seed_data /
# a validation run creates (*.db) were both landing in the pushed repo.
_EXCLUDED_OUTPUT_DIRS = {
    # build-output / dependency directories
    "target", "node_modules", "__pycache__", ".git", "dist", "build",
    ".next", ".gradle", "venv", ".venv", ".besser_snapshot", ".pytest_cache",
    # runtime / build artifact files (never belong in source or a push)
    "*.zip", "*.db", "*.sqlite", "*.sqlite3", "*.db-journal", "*.pyc", "*.log",
}


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
    # Set only by a vibe-MODIFY run whose instruction implied new domain
    # entities: the run's project export with the active ClassDiagram's
    # model replaced by the re-serialised, model-synced diagram. The push
    # endpoint prefers this over the request's projectExport so the pushed
    # buml/ stays in sync with the code. ``None`` for every other run.
    updated_project_export: Optional[dict] = None


class SmartRunRegistry:
    """In-memory, TTL-bounded map of ``run_id → SmartRunEntry``.

    The sibling ``GET /download-smart/{run_id}`` endpoint reads entries
    non-destructively (``get``) so a failed blob fetch can be retried.
    Cleanup is owned by the periodic sweep: anything older than
    ``LLM_DOWNLOAD_TTL_SECONDS`` is removed (entry + temp dir) by the
    task started from the backend's lifespan.
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

    async def get(self, run_id: str) -> Optional[SmartRunEntry]:
        """Non-destructive lookup — the entry stays re-downloadable
        until the TTL sweep removes it."""
        async with self._lock:
            return self._entries.get(run_id)

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


async def reserve_active_run(run_id: str) -> Optional[asyncio.Event]:
    """Atomically reserve ``run_id`` for one live runner."""
    event = asyncio.Event()
    async with _ACTIVE_RUNS_LOCK:
        if run_id in _ACTIVE_RUNS:
            return None
        _ACTIVE_RUNS[run_id] = event
    return event


async def release_active_run(run_id: str, event: asyncio.Event) -> bool:
    """Release ``run_id`` only when ``event`` is still its owner."""
    async with _ACTIVE_RUNS_LOCK:
        if _ACTIVE_RUNS.get(run_id) is not event:
            return False
        _ACTIVE_RUNS.pop(run_id, None)
        return True


# ---------------------------------------------------------------------
# Incremental vibe-modify: seed a new workspace from a previous run
# ---------------------------------------------------------------------


def _seed_workspace_from_base(base_dir: str, dest_dir: str) -> None:
    """Copy a previous run's files into a fresh workspace for editing.

    Copies ``base_dir`` into ``dest_dir`` (which already exists — the new
    run's own ``mkdtemp``), skipping build / dependency directories so a
    prior ``npm install`` / ``cargo build`` doesn't drag thousands of
    files along. The base is left untouched so it stays downloadable.

    Then strips the seed's crash-recovery internals — the copied
    ``.besser_checkpoint.json`` and ``.besser_snapshot/`` belong to the
    OLD run and would make ``modify()`` look resumable / mis-snapshotted —
    while deliberately KEEPING ``.besser_recipe.json`` so the orchestrator
    can replay the previous run's generator-file tags.
    """
    from besser.generators.llm.checkpoint import CHECKPOINT_FILENAME

    shutil.copytree(
        base_dir,
        dest_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*_EXCLUDED_OUTPUT_DIRS),
    )

    # Drop the copied checkpoint (old run's mid-flight state).
    checkpoint = os.path.join(dest_dir, CHECKPOINT_FILENAME)
    try:
        if os.path.isfile(checkpoint):
            os.remove(checkpoint)
    except OSError:
        logger.debug("Failed to strip seed checkpoint at %s", checkpoint, exc_info=True)

    # Drop the copied snapshot dir (old run's pre-Phase-3 rollback point).
    snapshot = os.path.join(dest_dir, ".besser_snapshot")
    try:
        if os.path.isdir(snapshot):
            shutil.rmtree(snapshot, ignore_errors=True)
    except OSError:
        logger.debug("Failed to strip seed snapshot at %s", snapshot, exc_info=True)


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
        reserved_cancel_event: Optional[asyncio.Event] = None,
        base_run_id: Optional[str] = None,
        mode: str = "generate",
    ) -> None:
        self.request = request
        # Resuming a prior run reuses its run_id so the client can keep
        # the same identifier across the crash. Fresh runs (including a
        # vibe-modify run seeded from a previous run) get a new UUID.
        # Either way the id is hex[32] so the path regex in the cancel /
        # download / resume routes accepts it.
        self.run_id = resume_run_id or uuid.uuid4().hex
        self._resume_run_id = resume_run_id
        self._reserved_cancel_event = reserved_cancel_event
        # Incremental vibe-modify. When ``mode == "modify"`` and
        # ``base_run_id`` still resolves to a live registry entry, this
        # run is SEEDED from that entry's files and edits them in place.
        # ``_seeded`` is flipped True once the copy succeeds; if the base
        # has expired we warn and fall back to a from-scratch ``run()``,
        # so ``_seeded`` stays False and ``run_orchestrator`` picks the
        # right entry point.
        self._base_run_id = base_run_id
        self._mode = mode
        self._seeded = False
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
            # Every non-resume run allocates its OWN fresh workspace first.
            # For a vibe-modify run that own workspace is then seeded from
            # the base run's files — the base itself is never touched, so it
            # stays downloadable.
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

            # ---- Incremental vibe-modify: seed from the base run --------
            if self._mode == "modify" and self._base_run_id:
                base_entry = await SMART_RUN_REGISTRY.get(self._base_run_id)
                if (
                    base_entry is None
                    or not base_entry.temp_dir
                    or not os.path.isdir(base_entry.temp_dir)
                ):
                    # Base-expired fallback: the previous run was swept past
                    # its TTL (or never existed). Warn — non-terminally —
                    # and fall through to a normal from-scratch run() so the
                    # user still gets output instead of a hard failure.
                    yield format_sse(ErrorEvent(
                        code="INCOMPLETE",
                        message=(
                            "The previous generation has expired, so there is "
                            "nothing to edit — rebuilding from scratch instead. "
                            "Re-run to seed a fresh base for future edits."
                        ),
                    ))
                    logger.info(
                        "smart-modify base %s expired for run %s; "
                        "falling back to from-scratch generation",
                        self._base_run_id, self.run_id,
                    )
                    # ``_seeded`` stays False → run_orchestrator uses run().
                else:
                    try:
                        await asyncio.to_thread(
                            _seed_workspace_from_base,
                            base_entry.temp_dir,
                            self.temp_dir,
                        )
                        self._seeded = True
                        logger.info(
                            "smart-modify run %s seeded from base %s",
                            self.run_id, self._base_run_id,
                        )
                    except Exception:
                        # A failed copy must not corrupt the run — warn and
                        # continue from-scratch against the (now partially
                        # populated) fresh temp dir. Clear it first so we
                        # don't edit a half-copied tree.
                        logger.exception(
                            "Failed to seed workspace for smart-modify run %s "
                            "from base %s; falling back to from-scratch",
                            self.run_id, self._base_run_id,
                        )
                        self._reset_temp_dir_after_failed_seed()
                        yield format_sse(ErrorEvent(
                            code="INCOMPLETE",
                            message=(
                                "Could not copy the previous generation — "
                                "rebuilding from scratch instead."
                            ),
                        ))
                        if self.temp_dir is None:
                            yield format_sse(ErrorEvent(
                                code="INTERNAL",
                                message="Failed to allocate a workspace for this run",
                            ))
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
            if tool == "__customize_skipped__":
                # The gap analyser judged the deterministic scaffold
                # sufficient — Phase 2 was skipped entirely. Surface it
                # so the run doesn't look like it silently dropped work.
                _put(PhaseEvent(
                    phase="customize",
                    message=f"skipped — {status}" if status else "skipped",
                ))
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
                            "skipped — no deterministic scaffold is available "
                            "for the primary model"
                        )
                    else:
                        msg = "skipped"
                    _put(PhaseEvent(phase="generate", message=msg))
                    return
                if status.startswith("failed:"):
                    # Generator failure — without this the card shows
                    # "generating" forever and the user never learns the
                    # run fell back to from-scratch LLM generation.
                    _put(PhaseEvent(phase="generate", message=f"{tool} {status}"))
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
        cancel_event = self._reserved_cancel_event
        if cancel_event is None:
            cancel_event = await reserve_active_run(self.run_id)
        if cancel_event is None:
            yield format_sse(ErrorEvent(
                code="BAD_REQUEST",
                message=(
                    "This smart-generation run is already active or being resumed."
                ),
            ))
            return

        # Bridge the asyncio.Event into a thread-safe boolean check the
        # synchronous orchestrator can poll between turns.
        def should_continue() -> bool:
            return not cancel_event.is_set()

        target_generator = getattr(
            self.request, "target_generator_override", None
        )
        skip_deterministic_generator = getattr(
            self.request, "skip_deterministic_generator", False
        )
        if self._mode == "modify" and not self._seeded:
            # The approved incremental plan skipped Phase 1 because it expected
            # a reusable workspace. If that base expired, rebuild with normal
            # generator selection instead of forcing an unnecessarily costly
            # LLM-only run.
            skip_deterministic_generator = False

        orchestrator = LLMOrchestrator(
            llm_client=client,
            domain_model=assembled.domain_model,
            gui_model=assembled.gui_model,
            agent_model=assembled.agent_model,
            agent_config=assembled.agent_config,
            object_model=assembled.object_model,
            state_machines=assembled.state_machines,
            quantum_circuit=assembled.quantum_circuit,
            bpmn_model=assembled.bpmn_model,
            nn_model=assembled.nn_model,
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
            # Heavy per-project compilers (tsc / cargo / kotlinc) are
            # opt-in for the web deployment — they were the main driver
            # of the duration/cost regression on non-Python stacks.
            enable_toolchain_validation=LLM_ENABLE_TOOLCHAIN_VALIDATION,
            # Close the agent's feedback loop: when Phase 3 finds BLOCKER
            # issues (syntax errors, broken Dockerfile refs, dep conflicts)
            # run the bounded fix loop instead of shipping the broken app
            # as a green success. Bounded 3x5 turns with snapshot/rollback;
            # honours the run's cancel/cost/runtime budget per fix turn.
            auto_fix_issues=LLM_ENABLE_AUTO_FIX,
            # Disable the arbitrary-shell tools on the hosted deploy — they are
            # user-steerable RCE / secret-exfil on a shared BYOK box. OFF by
            # default (BESSER_LLM_ENABLE_SHELL_TOOLS). The agent keeps every
            # static tool; only arbitrary `run_command`/`install_dependencies`
            # are withheld.
            allow_shell_tools=LLM_ENABLE_SHELL_TOOLS,
            # Binding generator choice from an approved preview plan. A
            # bound None explicitly skips Phase 1; an unbound None auto-selects.
            target_generator=target_generator,
            target_generator_bound=(
                target_generator is not None or skip_deterministic_generator
            ),
            # The run's original project export (ProjectInput → JSON-safe
            # dict). Used ONLY by modify()'s model-sync step to slot an
            # updated ClassDiagram into the push export; ignored by
            # run()/resume(). model_dump(mode="json") keeps datetimes as
            # ISO strings so the push can re-validate it as a ProjectInput.
            source_project_export=self._source_project_export_dict(),
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
                if self._mode == "modify" and self._seeded:
                    # Seeded vibe-modify: edit the copied files in place.
                    # (A base-expired / failed-seed fallback left _seeded
                    # False and drops through to the from-scratch run().)
                    return await asyncio.to_thread(
                        orchestrator.modify, self.request.instructions
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

        # Hard runtime watchdog. The orchestrator checks its runtime cap
        # only at turn boundaries, so a hung provider call (bounded by
        # the SDK timeout but still minutes) or a runaway Phase 3 could
        # exceed the cap by a lot. After max_runtime + grace, flip the
        # cancel event so the worker stops at the next opportunity; the
        # terminal event is reported as TIMEOUT, not CANCELLED.
        watchdog_fired = {"value": False}

        async def runtime_watchdog() -> None:
            # Poll rather than a single sleep so we honour the orchestrator's
            # ADAPTED runtime cap: _apply_adaptive_budget() raises
            # ``orchestrator.max_runtime_seconds`` for from-scratch runs AFTER
            # Phase 1, which is after this task starts. Reading the request cap
            # up-front (the old behaviour) would kill a legitimately-longer
            # from-scratch run at the original limit. Fall back to the request
            # cap until/unless the orchestrator raises its own.
            try:
                while True:
                    await asyncio.sleep(min(10.0, LLM_WATCHDOG_GRACE_SECONDS))
                    if cancel_event.is_set():
                        return
                    effective_cap = getattr(
                        orchestrator, "max_runtime_seconds",
                        self.request.max_runtime_seconds,
                    )
                    elapsed = time.monotonic() - (self._started_at or time.monotonic())
                    if elapsed > effective_cap + LLM_WATCHDOG_GRACE_SECONDS:
                        break
            except asyncio.CancelledError:
                raise
            if not cancel_event.is_set():
                effective_cap = getattr(
                    orchestrator, "max_runtime_seconds",
                    self.request.max_runtime_seconds,
                )
                logger.warning(
                    "smart-gen watchdog fired for run %s (cap %ds + %ds grace)",
                    self.run_id,
                    effective_cap,
                    LLM_WATCHDOG_GRACE_SECONDS,
                )
                watchdog_fired["value"] = True
                cancel_event.set()

        watchdog_task = asyncio.create_task(
            runtime_watchdog(), name="smart-gen-runtime-watchdog",
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
            # Specific (typed) exceptions first, then the built-in
            # fallbacks for legacy raise sites. Order matters: the typed
            # classes subclass ValueError/RuntimeError.
            try:
                result_path = await worker_task
            except (CheckpointMismatchError, EmptyInstructionsError) as exc:
                worker_exception = exc
                yield format_sse(ErrorEvent(code="BAD_REQUEST", message=str(exc)))
            except FileNotFoundError as exc:
                # resume() with no checkpoint on disk.
                worker_exception = exc
                yield format_sse(ErrorEvent(code="BAD_REQUEST", message=str(exc)))
            except InvalidApiKeyError as exc:
                worker_exception = exc
                yield format_sse(ErrorEvent(code="INVALID_KEY", message=str(exc)))
            except UpstreamLLMError as exc:
                worker_exception = exc
                yield format_sse(ErrorEvent(code="UPSTREAM_LLM", message=str(exc)))
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
            watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task

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
            await release_active_run(self.run_id, cancel_event)

            # Terminal events are mutually exclusive: CANCELLED (or
            # TIMEOUT when the watchdog fired) is emitted ONLY when the
            # cancellation actually interrupted the run. A run cancelled
            # at the very end that still produced output falls through
            # to the normal `done` path below — emitting both signals
            # contradicted the documented contract (CANCELLED is
            # terminal) and confused clients.
            if cancel_event.is_set() and (worker_exception is not None or not result_path):
                if watchdog_fired["value"]:
                    yield format_sse(ErrorEvent(
                        code="TIMEOUT",
                        message=(
                            "Runtime cap exceeded — the run was stopped by "
                            "the server watchdog before completing."
                        ),
                    ))
                else:
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

            # Compare against the orchestrator's EFFECTIVE caps, which
            # _apply_adaptive_budget() may have raised for from-scratch runs.
            # Using the original request caps would emit a spurious
            # COST_CAP/TIMEOUT warning on a run that completed cleanly within
            # its (adapted) budget.
            effective_cost_cap = getattr(
                orchestrator, "max_cost_usd", self.request.max_cost_usd
            )
            effective_runtime_cap = getattr(
                orchestrator, "max_runtime_seconds", self.request.max_runtime_seconds
            )
            if final_cost > effective_cost_cap:
                yield format_sse(ErrorEvent(
                    code="COST_CAP",
                    message=(
                        f"Cost cap reached (${final_cost:.4f} > "
                        f"${effective_cost_cap}). "
                        "Output may be incomplete."
                    ),
                ))
            if elapsed > effective_runtime_cap:
                yield format_sse(ErrorEvent(
                    code="TIMEOUT",
                    message=(
                        f"Runtime cap reached ({elapsed:.1f}s > "
                        f"{effective_runtime_cap}s). "
                        "Output may be incomplete."
                    ),
                ))

            # ---- 9b. Honest "incomplete" signal -----------------------
            # The orchestrator returns its output dir even when Phase 2
            # was cut short (a provider rate-limit, the turn cap, a late
            # cancellation). Without this the run would be reported as an
            # unqualified success even though requested changes never ran.
            exited_cleanly = bool(getattr(orchestrator, "_phase2_exited_cleanly", True))
            stop_reason = getattr(orchestrator, "_phase2_stop_reason", "completed")

            # Phase 3 can DETECT blocker-class issues (syntax / import /
            # dependency errors — the "won't compile / won't boot" class) that
            # the bounded auto-fix loop couldn't resolve. Those must also mark
            # the run incomplete: without this, an app that parsed but has an
            # unfixed blocker ships as an unqualified green "success" even
            # though it can't run. (The verdict previously keyed ONLY on Phase 2
            # emitting end_turn.)
            _unfixed_blockers = [
                getattr(i, "message", str(i))
                for i in (getattr(orchestrator, "_validation_issues", None) or [])
                if getattr(i, "severity", None) == "blocker"
            ]

            incomplete = (not exited_cleanly) or bool(_unfixed_blockers)
            incomplete_reason_msg: Optional[str] = None
            if incomplete and exited_cleanly and _unfixed_blockers:
                # Phase 2 finished cleanly, but Phase 3 left unfixed blockers.
                incomplete_reason_msg = (
                    f"The app was built but {len(_unfixed_blockers)} blocker-level "
                    "issue(s) remain that likely stop it from running "
                    "(syntax / import / dependency errors). First: "
                    + _unfixed_blockers[0][:160]
                )
                yield format_sse(ErrorEvent(
                    code="INCOMPLETE",
                    message=(
                        incomplete_reason_msg
                        + " The downloaded output may not run as-is."
                    ),
                ))
            if not exited_cleanly:
                api_err = (getattr(orchestrator, "_phase2_api_error", "") or "")[:160]
                _REASON_TEXT = {
                    "api_error": (
                        "The customization loop was cut short by a provider error"
                        + (f" ({api_err})" if api_err else "")
                        + " before all requested changes were applied."
                    ),
                    "max_turns": (
                        "The customization loop reached its step limit before "
                        "finishing every requested change."
                    ),
                    "cancelled": (
                        "The run was cancelled before the customization loop finished."
                    ),
                    "cost_cap": (
                        "The run hit its cost cap before finishing every requested change."
                    ),
                    "timeout": (
                        "The run hit its runtime cap before finishing every requested change."
                    ),
                }
                incomplete_reason_msg = _REASON_TEXT.get(
                    stop_reason, "The customization loop did not finish cleanly."
                )
                # cost_cap / timeout already emit their own dedicated warning
                # above — avoid a duplicate. Warn here for the other reasons.
                if stop_reason not in ("cost_cap", "timeout"):
                    yield format_sse(ErrorEvent(
                        code="INCOMPLETE",
                        message=(
                            incomplete_reason_msg
                            + " The downloaded output may be incomplete; you can "
                            "resume the run to continue from where it stopped."
                        ),
                    ))

            # ---- 10. Package the result and emit `done` ---------------
            try:
                done_event, entry = await asyncio.to_thread(
                    self._package_result, result_path
                )
                done_event.incomplete = incomplete
                done_event.incompleteReason = incomplete_reason_msg
                # Carry any vibe-MODIFY model-sync delta so the GitHub push
                # writes buml/ from the UPDATED model rather than the stale
                # request export. None for generate/resume runs (and modify
                # runs whose instruction implied no new domain entities).
                entry.updated_project_export = getattr(
                    orchestrator, "_updated_project_export", None
                )
                await SMART_RUN_REGISTRY.put(self.run_id, entry)
                yield format_sse(done_event)
            except _EmptyGenerationError as exc:
                # Raised inside _package_result when the orchestrator
                # returned but produced zero user files. Surfaced with
                # its real message (previously swallowed by the generic
                # packaging branch below).
                logger.error(
                    "Smart-gen worker %s produced no output files", self.run_id
                )
                yield format_sse(ErrorEvent(code="INTERNAL", message=str(exc)))
                self._cleanup_temp_dir()
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

        # Collect every user file. Skips internal artefacts
        # (.besser_recipe.json etc.) AND build-output directories —
        # cargo/npm runs during the customise or validate phases would
        # otherwise put thousands of dependency files (target/,
        # node_modules/) into the download zip.
        user_files: list[str] = []
        for root, dirs, files in os.walk(result_path):
            dirs[:] = [d for d in dirs if d not in _EXCLUDED_OUTPUT_DIRS]
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
            runId=self.run_id,
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

    def _reset_temp_dir_after_failed_seed(self) -> None:
        """Wipe a partially-seeded temp dir and re-allocate a clean one.

        On a copytree failure the fresh workspace may hold a half-copied
        tree; editing that would be worse than starting over. Clear it and
        re-``mkdtemp`` so the from-scratch fallback starts clean. Sets
        ``self.temp_dir`` to None if re-allocation itself fails (the caller
        then surfaces INTERNAL and returns).
        """
        old = self.temp_dir
        if old and os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
        try:
            self.temp_dir = tempfile.mkdtemp(
                prefix=f"{LLM_TEMP_DIR_PREFIX}{self.run_id}_"
            )
        except OSError:
            logger.exception(
                "Failed to re-allocate workspace after seed failure for run %s",
                self.run_id,
            )
            self.temp_dir = None

    def _source_project_export_dict(self) -> Optional[dict]:
        """JSON-safe dict of the request's project, for modify() model-sync.

        Best-effort: a serialisation failure returns ``None`` (model-sync
        simply won't be able to build an updated push export) rather than
        breaking the run.
        """
        try:
            return self.request.project.model_dump(mode="json")
        except Exception:
            logger.debug(
                "Failed to serialise project export for model-sync; "
                "modify() will skip the push-export update",
                exc_info=True,
            )
            return None

    def _cleanup_temp_dir(self) -> None:
        if self.temp_dir and os.path.isdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                logger.exception(
                    "Failed to clean up temp dir for run %s", self.run_id
                )
        self.temp_dir = None
