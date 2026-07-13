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

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Path, Request
from fastapi.responses import StreamingResponse

from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    ImportGitHubRunRequest,
    ImportGitHubRunResponse,
    PushSmartToGitHubRequest,
    PushSmartToGitHubResponse,
    SmartGenerateRequest,
    SmartPreviewRequest,
)
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
    SmartRunEntry,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.model_assembly import (
    assemble_models_from_project,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.preview import (
    build_preview,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    _EXCLUDED_OUTPUT_DIRS,
    _locate_run_temp_dir,
    release_active_run,
    release_run_slot,
    request_cancellation,
    reserve_active_run,
    try_acquire_run_slot,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    create_github_service,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_oauth import (
    get_user_token,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_deploy_api import (
    _extract_github_error_message,
    _sanitize_repo_name,
)
from besser.utilities.buml_code_builder import (
    agent_model_to_code,
    domain_model_to_code,
    gui_model_to_code,
)
from besser.generators.web_app.web_app_generator import agent_slug
from besser.generators.llm.llm_client import DEFAULT_MODELS as _LLM_DEFAULT_MODELS

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


async def _stream_with_slot_release(
    inner: AsyncIterator[bytes],
    run_reservation: Optional[tuple[str, asyncio.Event]] = None,
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
        if run_reservation is not None:
            await release_active_run(*run_reservation)
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

    # Incremental vibe-modify: when the request carries mode="modify" and a
    # base_run_id, the runner seeds this run's workspace from that previous
    # run's files and edits them in place (falling back to from-scratch if
    # the base has expired). Both fields default to the from-scratch path.
    runner = SmartGenerationRunner(
        request,
        base_run_id=request.base_run_id,
        mode=request.mode,
    )
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
            "toolchain_validation_enabled": C.LLM_ENABLE_TOOLCHAIN_VALIDATION,
        },
        "supported_providers": ["anthropic", "openai", "mistral"],
        # Per-provider default model names, sourced from the LLM client
        # layer (single source of truth — the BYOK dialog should read
        # these instead of hardcoding its own copies).
        "default_models": dict(_LLM_DEFAULT_MODELS),
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
        mode=request.mode,
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

    # Atomically reserve the checkpoint/run ID before the global slot.
    # This prevents the original run and a resume (or two resumes) from
    # sharing one workspace and checkpoint.
    cancel_event = await reserve_active_run(run_id)
    if cancel_event is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This smart-generation run is already active or being resumed."
            ),
        )

    # Same concurrency gate as /smart-generate - a resumed run consumes
    # the same resources as a fresh one.
    if not try_acquire_run_slot():
        await release_active_run(run_id, cancel_event)
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many smart-generation runs are in flight right now. "
                "Retry in a moment."
            ),
        )

    runner = SmartGenerationRunner(
        request,
        resume_run_id=run_id,
        reserved_cancel_event=cancel_event,
    )
    return StreamingResponse(
        _stream_with_slot_release(
            runner.generate_and_stream(http_request=http_request),
            run_reservation=(run_id, cancel_event),
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
    """Download the output produced by a completed run.

    Re-downloadable until the TTL sweep expires the entry
    (``LLM_DOWNLOAD_TTL_SECONDS``, default 30 min) — a transient network
    failure during the blob fetch must not permanently lose an artifact
    the user paid minutes and dollars to produce. Cleanup is owned
    entirely by ``SmartRunRegistry.periodic_sweep``; this endpoint no
    longer deletes anything.
    """
    entry = await SMART_RUN_REGISTRY.get(run_id)
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

    return StreamingResponse(
        _iter_file(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------
# POST /besser_api/push-smart-to-github
# ---------------------------------------------------------------------

# .env-family files whose content is safe to publish (templates, not
# real credentials). Anything else named `.env*` is scanned for secrets.
_ENV_SAFE_SUFFIXES = (".example", ".sample", ".template", ".dist")

# Env assignments whose *key* looks like a credential.
_SECRET_ENV_KEY_RE = re.compile(
    r"(?im)^\s*[A-Za-z0-9_]*"
    r"(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)"
    r"[A-Za-z0-9_]*\s*=\s*(.+)$"
)

# Value shapes that are unmistakably real provider credentials, wherever
# they appear (OpenAI / Anthropic / GitHub / AWS / Slack).
_SECRET_VALUE_TOKEN_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{12,}"
    r"|sk-[A-Za-z0-9_\-]{16,}"
    r"|gh[posru]_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,})"
)

_ENV_PLACEHOLDER_VALUES = {
    "", "changeme", "change-me", "your-key-here", "your_api_key",
    "xxx", "xxxx", "todo", "...", "none", "null",
}


def _looks_like_secret_env(content: str) -> bool:
    """Heuristic: does this ``.env`` body carry a real credential?

    Preserves template/config values (``VITE_API_URL=...``, placeholders)
    while catching an ``.env`` that an LLM populated with a live key.
    """
    if _SECRET_VALUE_TOKEN_RE.search(content):
        return True
    for match in _SECRET_ENV_KEY_RE.finditer(content):
        value = match.group(1).strip().strip('"').strip("'").strip()
        low = value.lower()
        if low in _ENV_PLACEHOLDER_VALUES:
            continue
        if (
            value.startswith("${")
            or value.startswith("<")
            or "your" in low
            or "placeholder" in low
            or "example" in low
        ):
            continue
        return True
    return False


def _scrub_secret_env_files(workdir: str) -> list[str]:
    """Delete any ``.env`` file carrying real secrets; keep ``.env.example``.

    Returns the repo-relative paths removed (for logging).
    """
    removed: list[str] = []
    for root, _dirs, files in os.walk(workdir):
        for name in files:
            base = name.lower()
            if not base.startswith(".env"):
                continue
            if any(base.endswith(sfx) for sfx in _ENV_SAFE_SUFFIXES):
                continue
            full = os.path.join(root, name)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                continue
            if _looks_like_secret_env(content):
                try:
                    os.remove(full)
                    removed.append(os.path.relpath(full, workdir).replace("\\", "/"))
                except OSError:
                    logger.warning(
                        "push-smart-to-github: failed to remove secret env file %s",
                        full, exc_info=True,
                    )
    return removed


def _write_smart_model_to_buml(workdir: str, project_export: Optional[dict]) -> None:
    """Inject the model source into ``workdir/buml/`` (best-effort).

    Reuses the same ``buml_code_builder`` helpers ``/deploy-webapp`` uses
    and the smart-generation project→BUML assembly, so this works for
    class / GUI / agent projects. Also writes ``buml/diagrams.json`` from
    the re-importable V2 project-export envelope. Never raises — a failed
    export logs a warning and the code push still succeeds.
    """
    buml_dir = os.path.join(workdir, "buml")
    os.makedirs(buml_dir, exist_ok=True)

    has_export = isinstance(project_export, dict) and bool(project_export)

    # ---- B-UML model .py files (domain / gui / agent) ----
    if has_export:
        try:
            assembled = assemble_models_from_project(
                ProjectInput.model_validate(project_export)
            )
            if assembled.domain_model is not None:
                domain_model_to_code(
                    model=assembled.domain_model,
                    file_path=os.path.join(buml_dir, "domain_model.py"),
                )
            if assembled.gui_model is not None:
                gui_model_to_code(
                    model=assembled.gui_model,
                    file_path=os.path.join(buml_dir, "gui_model.py"),
                    domain_model=assembled.domain_model,
                )
            if assembled.agent_model is not None:
                slug = agent_slug(assembled.agent_model.name)
                agent_model_to_code(
                    assembled.agent_model,
                    os.path.join(buml_dir, f"agent_model_{slug}.py"),
                )
        except Exception:
            logger.warning(
                "push-smart-to-github: failed to export B-UML model files; "
                "continuing without them",
                exc_info=True,
            )

    # ---- Re-importable diagrams.json ----
    if has_export:
        try:
            with open(
                os.path.join(buml_dir, "diagrams.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(project_export, f, indent=2, default=str)
        except Exception:
            logger.warning(
                "push-smart-to-github: failed to write buml/diagrams.json; "
                "continuing without it",
                exc_info=True,
            )
    else:
        logger.warning(
            "push-smart-to-github: request carried no 'projectExport'; the "
            "pushed repo will not include a re-importable buml/diagrams.json"
        )


def _pick_default_branch(branches: list[str]) -> str:
    """Best-effort repo default when only a branch list is available.

    Prefers the conventional default names over an arbitrary first entry
    so we never blindly take ``branches[0]``.
    """
    if not branches:
        return "main"
    for preferred in ("main", "master"):
        if preferred in branches:
            return preferred
    return branches[0]


def _resolve_target_branch(
    requested: Optional[str],
    available: Optional[list[str]],
    default_branch: str,
) -> str:
    """Pick the push target branch.

    Honours ``requested`` when it is present in ``available`` (``available
    is None`` means the branch set is not materialised yet — e.g. a
    freshly created repo — so the explicit request is honoured there).
    Falls back to ``default_branch`` otherwise.
    """
    if requested:
        requested = requested.strip()
    if requested:
        if available is None or requested in available:
            return requested
        logger.warning(
            "push-smart-to-github: requested branch %r not present; using %r",
            requested, default_branch,
        )
    return default_branch


@router.post("/push-smart-to-github", response_model=PushSmartToGitHubResponse)
async def push_smart_to_github(
    req: PushSmartToGitHubRequest,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Push a finished vibe/smart-generation run to a GitHub repository.

    Unlike ``/deploy-webapp`` (which regenerates deterministically and
    would discard the LLM's customizations), this pushes the *stored*
    artifact for ``run_id`` — the exact code the user saw and paid for —
    plus the re-importable model source under ``buml/``.

    Flow:
      1. Gate on the ``X-GitHub-Session`` header (same helper the other
         GitHub endpoints use — the real token never reaches the client).
      2. Resolve the stored run from ``SMART_RUN_REGISTRY`` (404 when it
         has been swept past its TTL — the user must re-generate).
      3. Copy the stored file tree into a fresh temp dir, skipping build/
         dependency dirs, ``.besser_*`` internals, and any ``.env`` that
         carries a real secret.
      4. Inject ``buml/`` model files + ``buml/diagrams.json``.
      5. Create (default private) or reuse the repo, resolve the branch.
      6. Replace the repo tree with this push (each vibe run is a full app).
    """
    # ---- 1. GitHub auth gate ----
    if not github_session:
        raise HTTPException(
            status_code=401,
            detail="GitHub authentication required. Please sign in with GitHub first.",
        )
    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="GitHub session expired. Please sign in again.",
        )

    # ---- 2. Resolve the stored run (non-destructive; None if swept) ----
    entry = await SMART_RUN_REGISTRY.get(req.run_id)
    if entry is None or not entry.temp_dir or not os.path.isdir(entry.temp_dir):
        # Machine-usable code the frontend can branch on, plus a human
        # hint: "This generation has expired — re-generate to push."
        raise HTTPException(status_code=404, detail="run_expired")

    deploy_config = req.deploy_config
    repo_name = _sanitize_repo_name(deploy_config.repo_name)

    try:
        github = create_github_service(access_token)

        user_info = await github.get_authenticated_user()
        owner = user_info.get("login")
        if not owner:
            raise HTTPException(
                status_code=401,
                detail="Could not resolve GitHub user from session.",
            )

        with tempfile.TemporaryDirectory(
            prefix=f"besser_smart_push_{uuid.uuid4().hex}_"
        ) as workdir:
            # ---- 3. Copy the stored tree, dropping build/dep dirs and
            # `.besser_*` internal files (recipe / checkpoint / snapshot).
            shutil.copytree(
                entry.temp_dir,
                workdir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*_EXCLUDED_OUTPUT_DIRS, ".besser_*"),
            )
            scrubbed = _scrub_secret_env_files(workdir)
            if scrubbed:
                logger.info(
                    "push-smart-to-github: scrubbed %d secret env file(s): %s",
                    len(scrubbed), scrubbed,
                )

            # ---- 4. Inject the model source into buml/ ----
            # Prefer the run's model-synced export (a vibe-MODIFY run whose
            # instruction added domain entities stores it on the registry
            # entry) so the pushed buml/ matches the code. Fall back to the
            # request's projectExport for every other run.
            effective_export = (
                getattr(entry, "updated_project_export", None)
                or req.projectExport
            )
            _write_smart_model_to_buml(workdir, effective_export)

            # ---- 5. Resolve the repo + target branch ----
            is_first_push = not deploy_config.use_existing
            if deploy_config.use_existing:
                try:
                    branches = await github.get_branches(owner, repo_name)
                except httpx.HTTPStatusError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        # Requested an existing repo that isn't there — surface
                        # a machine-usable code rather than silently creating.
                        raise HTTPException(
                            status_code=404, detail="repo_missing"
                        ) from exc
                    raise
                repo_default = _pick_default_branch(branches)
                target_branch = _resolve_target_branch(
                    deploy_config.branch, branches, repo_default
                )
            else:
                description = (
                    deploy_config.description
                    or "Vibe-generated app + model — Generated by BESSER"
                )
                repo_info = await github.create_repository(
                    repo_name=repo_name,
                    description=description,
                    is_private=deploy_config.is_private,
                    auto_init=True,
                )
                repo_default = repo_info.get("default_branch", "main")
                target_branch = _resolve_target_branch(
                    deploy_config.branch, None, repo_default
                )

            # ---- 6. Push the whole tree (replace — full app per run) ----
            push_results = await github.push_directory_to_repo(
                owner=owner,
                repo_name=repo_name,
                directory_path=workdir,
                commit_message=(
                    deploy_config.commit_message
                    or "Vibe-generated app + model — BESSER"
                ),
                branch=target_branch,
                preserve_existing_files=False,
            )
            if not push_results.get("commit_sha"):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create GitHub commit for pushed files.",
                )

            return PushSmartToGitHubResponse(
                success=True,
                repo_url=f"https://github.com/{owner}/{repo_name}",
                owner=owner,
                repo_name=repo_name,
                is_first_push=is_first_push,
                files_uploaded=push_results.get("total_files", 0),
            )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        # Mirror /deploy-webapp's GitHub error mapping.
        logger.warning("GitHub API error in push_smart_to_github", exc_info=True)
        upstream_status = exc.response.status_code if exc.response is not None else 502
        detail = _extract_github_error_message(exc)
        if upstream_status == 401:
            raise HTTPException(status_code=401, detail=detail) from exc
        if upstream_status == 403:
            raise HTTPException(status_code=403, detail=detail) from exc
        if upstream_status == 422:
            # 422 covers both "repository name already exists on this
            # account" AND a non-fast-forward ref update (a concurrent
            # change to the branch). Both map cleanly to 409 Conflict.
            raise HTTPException(status_code=409, detail=detail) from exc
        if 400 <= upstream_status < 500:
            raise HTTPException(status_code=upstream_status, detail=detail) from exc
        raise HTTPException(
            status_code=502, detail=f"GitHub upstream error: {detail}"
        ) from exc
    except Exception:
        logger.exception("Unexpected error in push_smart_to_github")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during GitHub push.",
        )


# ---------------------------------------------------------------------
# POST /besser_api/import-github-run
# ---------------------------------------------------------------------


@router.post("/import-github-run", response_model=ImportGitHubRunResponse)
async def import_github_run(
    req: ImportGitHubRunRequest,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Import an existing BESSER-created GitHub repo as a modify seed.

    The counterpart to ``/push-smart-to-github``: instead of *writing* a
    finished run to GitHub, this *reads* a repo we previously created (so
    it carries the generated code plus a re-importable ``buml/diagrams.json``)
    back into the editor so the user can continue from it.

    Flow:
      1. Gate on the ``X-GitHub-Session`` header (same helper the other
         GitHub endpoints use — the real token never reaches the client).
      2. Resolve the target branch (repo default when none is requested).
      3. Download + extract the repo tarball into a fresh temp dir.
      4. Register that code tree as a run in ``SMART_RUN_REGISTRY`` under a
         fresh ``run_id``, with ``temp_dir`` pointing at the extracted root.
         A later ``/smart-generate`` with ``mode="modify"`` and
         ``base_run_id=run_id`` seeds its workspace from exactly this tree.
      5. Read ``buml/diagrams.json`` (the re-importable model) if present.

    Returns ``{ run_id, project, has_model, owner, repo, branch, message }``.
    ``project`` is the parsed ``diagrams.json`` (or ``null`` when the repo
    has no BESSER model — the frontend must then tell the user to open the
    repo in the editor first before it can be smart-modified).
    """
    # ---- 1. GitHub auth gate (identical to push-smart-to-github) ----
    if not github_session:
        raise HTTPException(
            status_code=401,
            detail="GitHub authentication required. Please sign in with GitHub first.",
        )
    access_token = get_user_token(github_session)
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="GitHub session expired. Please sign in again.",
        )

    owner = req.owner.strip()
    repo = req.repo.strip()

    try:
        github = create_github_service(access_token)

        # ---- 2. Resolve the target branch (repo default when unset) ----
        try:
            branches = await github.get_branches(owner, repo)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # The repo (or the caller's access to it) doesn't exist —
                # surface a machine-usable code the frontend can branch on.
                raise HTTPException(status_code=404, detail="repo_missing") from exc
            raise
        branch = _resolve_target_branch(
            req.branch, branches, _pick_default_branch(branches)
        )

        # ---- 3. Download + extract the repo tarball ----
        extract_dir = await github.download_repo_tarball(owner, repo, branch)

        # ---- 4. Register the extracted code tree as a run ----
        # ``file_path`` / ``is_zip`` are minimal placeholders — the modify
        # seed reads only ``temp_dir`` (via ``_seed_workspace_from_base``),
        # and ``/download-smart`` is never called on an imported run.
        run_id = uuid.uuid4().hex
        entry = SmartRunEntry(
            file_path=extract_dir,
            file_name="github_import",
            is_zip=False,
            temp_dir=extract_dir,
            created_at=time.time(),
        )
        await SMART_RUN_REGISTRY.put(run_id, entry)

        # ---- 5. Read the re-importable model (buml/diagrams.json) ----
        project: Optional[dict] = None
        has_model = False
        message: Optional[str] = None
        diagrams_path = os.path.join(extract_dir, "buml", "diagrams.json")
        if os.path.isfile(diagrams_path):
            try:
                with open(diagrams_path, "r", encoding="utf-8") as fh:
                    project = json.load(fh)
                has_model = True
            except (OSError, ValueError):
                logger.warning(
                    "import-github-run: buml/diagrams.json present but "
                    "unreadable for %s/%s@%s",
                    owner, repo, branch, exc_info=True,
                )
                message = (
                    "This repo's BESSER model (buml/diagrams.json) could not "
                    "be parsed."
                )
        else:
            message = (
                "This repo has no BESSER model — open it in the editor first."
            )

        return ImportGitHubRunResponse(
            run_id=run_id,
            project=project,
            has_model=has_model,
            owner=owner,
            repo=repo,
            branch=branch,
            message=message,
        )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        # Mirror push-smart-to-github's GitHub error mapping.
        logger.warning("GitHub API error in import_github_run", exc_info=True)
        upstream_status = exc.response.status_code if exc.response is not None else 502
        detail = _extract_github_error_message(exc)
        if upstream_status == 401:
            raise HTTPException(status_code=401, detail=detail) from exc
        if upstream_status == 403:
            raise HTTPException(status_code=403, detail=detail) from exc
        if upstream_status == 404:
            # Repo missing / no access — machine-usable code.
            raise HTTPException(status_code=404, detail="repo_missing") from exc
        if upstream_status == 422:
            raise HTTPException(status_code=409, detail=detail) from exc
        if 400 <= upstream_status < 500:
            raise HTTPException(status_code=upstream_status, detail=detail) from exc
        raise HTTPException(
            status_code=502, detail=f"GitHub upstream error: {detail}"
        ) from exc
    except Exception:
        logger.exception("Unexpected error in import_github_run")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during GitHub import.",
        )
