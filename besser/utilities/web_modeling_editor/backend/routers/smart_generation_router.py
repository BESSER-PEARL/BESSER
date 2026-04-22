"""Smart Generation Router.

Server-Sent Events endpoint that drives ``besser.generators.llm.LLMOrchestrator``
and streams phase markers, tool calls, text deltas, cost ticks, and a
final download URL back to the browser.

Security caveat
---------------
``LLMOrchestrator`` executes shell commands produced by the LLM inside a
per-run temp directory (120s per-command timeout, no command denylist).
It runs in the same process as the rest of the BESSER backend. This
endpoint is BYOK — the user provides their own Anthropic or OpenAI API
key and pays for their own run. Container-level isolation (e.g. Render)
is the only sandbox between runs. Never deploy this endpoint on
infrastructure shared with untrusted workloads.

The user's API key is accepted only in the JSON POST body, never via
URL, query string, or headers. It is never logged, never echoed in SSE
events, and never stored on disk.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import shutil
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
    SmartPreviewRequest,
)
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.model_assembly import (
    assemble_models_from_project,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.preview import (
    build_preview,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    _locate_run_temp_dir,
    release_run_slot,
    request_cancellation,
    try_acquire_run_slot,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["smart-generation"])

_DOWNLOAD_CHUNK_SIZE = 65536

# Any character that could break a Content-Disposition header if naively
# interpolated (CR, LF, double quote, backslash, control chars) is
# replaced with a single underscore. Since file names come from
# LLM-generated output directories, we cannot trust them.
_UNSAFE_FILENAME_RE = re.compile(r"[^\w\-. ()\[\]]+")


def _safe_attachment_filename(filename: str, fallback: str) -> str:
    """Produce a Content-Disposition-safe ASCII filename.

    Strips newlines, CR, double quotes, backslashes, and any other
    character that could be used to split the HTTP header line or
    inject another directive. If the sanitised filename ends up empty
    (e.g. an LLM produced a filename consisting entirely of CJK
    characters), falls back to ``fallback``.
    """
    candidate = (filename or "").strip()
    candidate = _UNSAFE_FILENAME_RE.sub("_", candidate)
    candidate = candidate.strip("._ ")
    if not candidate:
        return fallback
    # Cap the length so a ludicrously long LLM filename can't bloat the
    # header. 120 chars matches common filesystem limits.
    return candidate[:120]


def _background_rmtree(temp_dir: str) -> None:
    """Delete a temp dir, logging failures instead of swallowing them."""
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        # Already gone — expected if the periodic sweep got there first.
        pass
    except Exception:
        logger.exception(
            "Failed to remove smart-gen temp dir in background task: %s", temp_dir
        )


async def _stream_with_slot_release(
    inner: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Yield from an SSE generator and return its concurrency slot
    when the stream ends, regardless of outcome.

    Used for both /smart-generate and /resume-smart-gen. The slot is
    always acquired by the caller before entering this wrapper; we
    only care about release semantics (happy path, error, or client
    disconnect — ``finally`` covers all three).
    """
    try:
        async for frame in inner:
            yield frame
    finally:
        release_run_slot()


# ---------------------------------------------------------------------
# POST /besser_api/smart-generate  (SSE stream)
# ---------------------------------------------------------------------


@router.post("/smart-generate", response_class=StreamingResponse)
async def smart_generate(request: SmartGenerateRequest, http_request: Request):
    """Stream an LLM-orchestrated code generation run as SSE events.

    Emits events in order: ``start`` → zero or more of
    ``phase`` / ``tool_call`` / ``text`` / ``cost`` → optional
    ``error(COST_CAP|TIMEOUT)`` warning → terminal ``done`` (carrying a
    one-time ``downloadUrl``) or terminal ``error``.

    The ``api_key`` field in the request body is a ``SecretStr`` and is
    never logged, never echoed in events, and never stored. The download
    URL returned in the ``done`` event is single-use — the first GET
    against ``/download-smart/{runId}`` serves the file; subsequent GETs
    return 404.
    """
    # Reserve a concurrency slot BEFORE allocating any resources.
    # ``try_acquire_run_slot`` is non-blocking: the client sees an
    # immediate 429 when the server is saturated, never a hung SSE
    # stream. The wrapper around the stream returns the slot on exit.
    if not try_acquire_run_slot():
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many smart-generation runs are in flight right now. "
                "Retry in a moment."
            ),
        )

    runner = SmartGenerationRunner(request)
    return StreamingResponse(
        _stream_with_slot_release(
            runner.generate_and_stream(http_request=http_request),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Disable response buffering in proxies (nginx, Render, etc.)
            # so events reach the browser in near-real-time.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------
# GET /besser_api/smart-gen/config
# ---------------------------------------------------------------------


@router.get("/smart-gen/config")
async def smart_gen_config():
    """Expose the server's current smart-generation configuration.

    The frontend reads this once at app startup (or before rendering
    the preview screen) so it can show the real hard caps in tooltips,
    clamp its own input fields to the server's limits, and surface
    whether tracing/checkpointing are on at this deploy.

    Nothing here is a secret — everything exposed is either a public
    cap or a feature flag. API keys and run IDs are never returned.
    """
    # Re-import the constants each call so tests that monkeypatch the
    # module see fresh values. The indirection has zero measurable
    # overhead vs. the LLM work these caps gate.
    from besser.utilities.web_modeling_editor.backend.constants import constants as C

    return {
        "caps": {
            "max_cost_usd_hard_cap": C.LLM_MAX_COST_USD_HARD_CAP,
            "max_runtime_seconds_hard_cap": C.LLM_MAX_RUNTIME_SECONDS_HARD_CAP,
            "default_max_cost_usd": C.LLM_DEFAULT_MAX_COST_USD,
            "default_max_runtime_seconds": C.LLM_DEFAULT_MAX_RUNTIME_SECONDS,
        },
        "download_ttl_seconds": C.LLM_DOWNLOAD_TTL_SECONDS,
        "cost_emitter_interval_seconds": C.LLM_COST_EMITTER_INTERVAL_SECONDS,
        "concurrency": {
            "max_concurrent_runs": C.LLM_MAX_CONCURRENT_RUNS,
        },
        "features": {
            "tracing_enabled": C.LLM_ENABLE_TRACING,
            "checkpointing_enabled": C.LLM_ENABLE_CHECKPOINTING,
            "resume_enabled": C.LLM_ENABLE_CHECKPOINTING,
        },
        "supported_providers": ["anthropic", "openai"],
    }


# ---------------------------------------------------------------------
# POST /besser_api/smart-preview
# ---------------------------------------------------------------------


@router.post("/smart-preview")
async def smart_preview(request: SmartPreviewRequest):
    """Return the plan smart-generate would run, without executing it.

    The response lets the UI show a confirmation screen before the user
    commits their API key and budget. Preview never calls an LLM — the
    classifier is pure heuristics plus the project's model presence —
    so no api_key is required and the response is instant.

    Returns (simplified)::

        {
          "primary_kind": "class",
          "auxiliary_kinds": ["gui", "agent"],
          "target_generator": "generate_web_app",
          "target_generator_confidence": 0.8,
          "summary": "Generate full-stack web app from Class Diagram; "
                     "also using GUI Model, Agent Model",
          "estimated_turns": 18,
          "estimated_cost_usd": 0.59,
          "estimated_duration_seconds": 195,
          "notes": [...],
          "model_summary": {"primary": "class", "present": [...]}
        }

    Errors are handled inline (not via the ``handle_endpoint_errors``
    decorator) because its ``functools.wraps`` wrapper confuses Pydantic
    forward-ref resolution: the wrapped function's ``__globals__`` point
    at the decorator module, so ``SmartPreviewRequest`` isn't visible
    when the schema is built.
    """
    try:
        assembled = assemble_models_from_project(
            request.project, primary_kind_override=request.primary_kind_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error building smart-preview")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    plan = build_preview(
        assembled,
        instructions=request.instructions,
        max_cost_usd=request.max_cost_usd,
        max_runtime_seconds=request.max_runtime_seconds,
    )
    payload = plan.to_dict()
    payload["model_summary"] = assembled.summary()
    return payload


# ---------------------------------------------------------------------
# POST /besser_api/resume-smart-gen/{run_id}
# ---------------------------------------------------------------------


@router.post("/resume-smart-gen/{run_id}", response_class=StreamingResponse)
async def resume_smart_gen(
    run_id: str,
    request: SmartGenerateRequest,
    http_request: Request,
):
    """Resume a smart-generate run that crashed before completion.

    Takes the same ``SmartGenerateRequest`` shape as ``/smart-generate``
    — the user re-supplies their API key and the current project, which
    we hash and compare against the checkpoint's fingerprint before
    accepting the resume. The orchestrator picks up from the last saved
    turn; earlier tool calls are not re-executed.

    Returns 404 when the run_id has no recoverable workspace (either
    never existed, was swept, or completed cleanly — in which case its
    checkpoint was intentionally deleted on success).

    The run_id path pattern matches hex[32] to keep malformed IDs out
    of the filesystem scan performed by ``_locate_run_temp_dir``.
    """
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(status_code=422, detail="Invalid run_id format")

    temp_dir = _locate_run_temp_dir(run_id)
    if temp_dir is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No recoverable workspace for this run_id. The crashed "
                "run's temp directory has either been swept or the run "
                "completed cleanly. Start a fresh run via /smart-generate."
            ),
        )

    # Same concurrency gate as /smart-generate — a resumed run consumes
    # the same resources as a fresh one.
    if not try_acquire_run_slot():
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many smart-generation runs are in flight right now. "
                "Retry in a moment."
            ),
        )

    runner = SmartGenerationRunner(request, resume_run_id=run_id)
    return StreamingResponse(
        _stream_with_slot_release(
            runner.generate_and_stream(http_request=http_request),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------
# POST /besser_api/cancel-smart-gen/{run_id}
# ---------------------------------------------------------------------


@router.post("/cancel-smart-gen/{run_id}")
@handle_endpoint_errors("cancel_smart_gen")
async def cancel_smart_gen(
    run_id: str = Path(
        ...,
        pattern=r"^[a-f0-9]{32}$",
        description="Hex run ID returned in the `start` SSE event",
    ),
):
    """Signal a live smart-generation run to stop at its next turn.

    Returns ``{"status": "cancelled"}`` if the run was found and
    signalled, ``{"status": "not_found"}`` if no live run exists for
    that ID (already finished, never existed, or download already
    happened).

    The user's BYOK budget stops accruing once the orchestrator hits
    the next turn boundary (typically within a few seconds). The SSE
    stream emits a final ``error(code="CANCELLED")`` event before
    closing so the frontend can show a definite "stopped by user"
    state instead of waiting for ``done``.
    """
    cancelled = await request_cancellation(run_id)
    return {"status": "cancelled" if cancelled else "not_found", "runId": run_id}


# ---------------------------------------------------------------------
# GET /besser_api/download-smart/{run_id}
# ---------------------------------------------------------------------


@router.get("/download-smart/{run_id}", response_class=StreamingResponse)
@handle_endpoint_errors("download_smart")
async def download_smart(
    run_id: str = Path(
        ...,
        pattern=r"^[a-f0-9]{32}$",
        description="Hex run ID returned in the `done` SSE event",
    ),
):
    """Single-use download of the output produced by a completed run.

    The first successful GET returns the file and deletes the run from
    the registry; subsequent GETs return 404. The underlying temp dir is
    removed after the response body is flushed, via a Starlette
    background task.
    """
    entry = await SMART_RUN_REGISTRY.pop(run_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown or expired run ID",
        )

    if not os.path.isfile(entry.file_path):
        # Should not happen, but guard against disk-level races.
        logger.error(
            "SmartRunRegistry had an entry for %s but file %s is missing",
            run_id,
            entry.file_path,
        )
        raise HTTPException(
            status_code=404,
            detail="Generated output is no longer available",
        )

    media_type = (
        "application/zip"
        if entry.is_zip
        else mimetypes.guess_type(entry.file_name)[0] or "application/octet-stream"
    )

    async def _iter_file() -> AsyncIterator[bytes]:
        with open(entry.file_path, "rb") as fh:
            while True:
                chunk = fh.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    safe_filename = _safe_attachment_filename(
        entry.file_name,
        fallback="besser_smart_output.zip" if entry.is_zip else "besser_smart_output.bin",
    )
    cleanup = BackgroundTask(_background_rmtree, entry.temp_dir)

    return StreamingResponse(
        _iter_file(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Cache-Control": "no-store",
        },
        background=cleanup,
    )
