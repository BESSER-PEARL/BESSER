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
# Runner
# ---------------------------------------------------------------------


class SmartGenerationRunner:
    """Drive one smart-generation run and yield SSE events."""

    def __init__(self, request: SmartGenerateRequest) -> None:
        self.request = request
        self.run_id = uuid.uuid4().hex
        self.temp_dir: Optional[str] = None
        self._started_at: Optional[float] = None

    async def generate_and_stream(self) -> AsyncGenerator[bytes, None]:
        """Run the pipeline and yield SSE frames.

        The generator always emits at least a ``start`` event first so
        the client can confirm the connection is live even if the run
        fails immediately afterwards.
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

        # ---- 2. Create the temp dir ------------------------------------
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
                assemble_models_from_project, self.request.project
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

        def on_progress(turn: int, tool: str, status: str) -> None:
            if tool == "validation":
                _put(PhaseEvent(phase="validate", message=status))
                return
            if turn == 0:
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

        orchestrator = LLMOrchestrator(
            llm_client=client,
            domain_model=assembled.domain_model,
            gui_model=assembled.gui_model,
            agent_model=assembled.agent_model,
            agent_config=assembled.agent_config,
            output_dir=self.temp_dir,
            max_cost_usd=self.request.max_cost_usd,
            max_runtime_seconds=self.request.max_runtime_seconds,
            on_progress=on_progress,
            on_text=on_text,
            use_streaming=True,
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
                    ))
                except Exception:
                    pass

        async def run_orchestrator() -> str:
            try:
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
