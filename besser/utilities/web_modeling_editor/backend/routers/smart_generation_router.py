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

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    request_cancellation,
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


# ---------------------------------------------------------------------
# POST /besser_api/smart-generate  (SSE stream)
# ---------------------------------------------------------------------


@router.post("/smart-generate", response_class=StreamingResponse)
async def smart_generate(request: SmartGenerateRequest):
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
    runner = SmartGenerationRunner(request)
    return StreamingResponse(
        runner.generate_and_stream(),
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
