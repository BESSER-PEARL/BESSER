"""Spec-Driven Generation Router.

Server-Sent Events endpoint that drives ``besser.spec_driven_agent.LLMOrchestrator``
and streams phase markers, tool calls, text deltas, cost ticks, and a
final download URL back to the browser.

Security caveat
---------------
The LLM's shell tool (``run_command``) is off unless the process starts
with ``BESSER_LLM_ENABLE_SHELL_TOOLS``; when enabled, each command runs in a
per-run bubblewrap sandbox (``spec_driven_agent/execution/sandbox.py``) and
fails closed if the sandbox cannot start. Generated code is still
model-authored: anything copied out of a run workspace (download, seed,
GitHub push) drops symlinks rather than following them.

The user's API key is accepted only in the JSON POST body, never via
URL, query string, or headers. It is never logged, never echoed in SSE
events, and never stored on disk.
"""

# No ``from __future__ import annotations`` here: FastAPI resolves string
# annotations against a decorated endpoint's ``__globals__``, which
# ``handle_endpoint_errors``' wrapper points at the decorator module.
import asyncio
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from besser.spec_driven_agent.planning.model_change import compare_with_base
from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
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
from besser.utilities.web_modeling_editor.backend.settings import is_demo_token
from besser.utilities.web_modeling_editor.backend.dependencies import SettingsDep, SpecDrivenRuntimeDep
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    DurableRunManager,
    SmartGenerationRunner,
    SpecDrivenRuntime,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.model_assembly import (
    assemble_models_from_project,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.preview import (
    build_preview,
)
# Underscored names below are shared on purpose; tests patch them by name.
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    _locate_run_temp_dir,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.github_sync import (
    import_run_from_github,
    push_run_to_github,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import UpstreamError
from besser.utilities.web_modeling_editor.backend.routers.github_oauth import (
    get_user_token,
)
from besser.utilities.web_modeling_editor.backend.routers.github_deploy_api import (
    _github_status_error,
    _sanitize_repo_name,
)
from besser.spec_driven_agent.providers.llm_client import DEFAULT_MODELS as _LLM_DEFAULT_MODELS
from besser.spec_driven_agent.providers.llm_client import default_model as _llm_default_model
from besser.spec_driven_agent.providers.model_catalog import load_model_catalog
from besser.spec_driven_agent.providers.llm_client import (
    free_alt_models,
    free_fallback_model,
    free_pilot_model,
    free_tier_available,
    free_tier_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["spec-driven"])
_MAX_SQLITE_SEQUENCE = (1 << 63) - 1

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
    runtime: SpecDrivenRuntime,
    run_reservation: Optional[tuple[str, asyncio.Event]] = None,
) -> AsyncIterator[bytes]:
    """Yield from an SSE generator and return its concurrency slot
    when the stream ends, regardless of outcome.

    Used for both /spec-driven/generate and /spec-driven resume. The slot is
    always acquired by the caller before entering this wrapper; we
    only care about release semantics (happy path, error, or client
    disconnect — ``finally`` covers all three).
    """
    import traceback as _tb

    from besser.utilities.web_modeling_editor.backend.services.spec_driven.incidents import (
        record_incident,
    )
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
        ErrorEvent,
        format_sse,
    )

    # Run context comes from the start frame; error frames don't carry it.
    run_ctx: dict = {}

    def _frame_payload(frame: bytes) -> dict:
        data_line = next(
            (ln for ln in frame.split(b"\n") if ln.startswith(b"data:")), None,
        )
        return json.loads(data_line[5:].decode("utf-8")) if data_line else {}

    def _sniff_frame(frame: bytes) -> None:
        """Persist every user-facing SSE error event as an incident."""
        try:
            if b"event: start" in frame:
                payload = _frame_payload(frame)
                run_ctx.update(
                    run_id=payload.get("runId"),
                    provider=payload.get("provider"),
                    model=payload.get("llmModel"),
                )
            elif b"event: error" in frame:
                payload = _frame_payload(frame)
                record_incident(
                    kind="sse_error",
                    code=payload.get("code"),
                    message=payload.get("message"),
                    **run_ctx,
                )
        except Exception:
            pass

    try:
        async for frame in inner:
            _sniff_frame(frame)
            yield frame
    except Exception as exc:
        # An exception escaping the run generator used to close the SSE
        # stream with NO final event — the client showed "stream closed
        # before reporting a final result" and the traceback died with
        # the container logs. Record it AND tell the client honestly.
        tb = _tb.format_exc()
        logger.exception("Run stream crashed without a final event")
        record_incident(
            kind="stream_crash",
            message=str(exc),
            traceback_text=tb,
            **run_ctx,
        )
        try:
            yield format_sse(
                ErrorEvent(code="INTERNAL", message="Internal server error")
            )
        except Exception:
            logger.exception("Could not deliver the terminal error frame")
    finally:
        if run_reservation is not None:
            await runtime.active_runs.release(*run_reservation)
        runtime.slots.release()


# ---------------------------------------------------------------------
# POST /besser_api/spec-driven/generate  (SSE stream)
# ---------------------------------------------------------------------

# ``Idempotency-Key`` -> ``(run_id, created_at)`` lives on the runtime. Starting
# a run is a non-idempotent POST, and behind a TLS-inspecting proxy that is
# exactly the request that fails ("Failed to fetch", no run to reconnect to). A
# client-generated key makes the retry attach to the run the first attempt
# started. In-process and best-effort on purpose: a missed hit costs one
# duplicate run, not corruption, and the durable store stays the source of truth.
_IDEMPOTENCY_TTL_SECONDS = 900


def _prune_idempotency_keys(keys: dict[str, tuple[str, float]], now: float) -> None:
    for key, (_, created_at) in list(keys.items()):
        if now - created_at > _IDEMPOTENCY_TTL_SECONDS:
            keys.pop(key, None)


def _require_demo_token(request: SmartGenerateRequest) -> None:
    """Reject a ``sponsored``-tier run that does not carry the demo secret.

    The sponsored tier spends the ORG's credits (endpoint + token live in
    server env), so without this check any caller could POST
    ``provider: "sponsored"`` and use our key as an open API proxy. Demo links
    carry the secret as ``?demo=<token>``.

    Fails CLOSED: an unset ``BESSER_DEMO_TOKEN`` refuses every sponsored run,
    so a half-finished env edit cannot leave the tier open. The message never
    says which half was wrong.

    Compared as bytes: ``compare_digest`` raises TypeError on a non-ASCII
    ``str``, which would turn a junk token into a 500 instead of a 403.
    """
    if request.provider != "sponsored":
        return
    if not is_demo_token(request.resolved_demo_token()):
        logger.warning("Rejected a sponsored-tier run without a valid demo token")
        raise HTTPException(
            status_code=403,
            detail="The sponsored tier is not available on this server.",
        )


def _run_stream_response(
    durable_runs: DurableRunManager, run_id: str, after_sequence: int = 0,
) -> StreamingResponse:
    return StreamingResponse(
        durable_runs.subscribe(run_id, after_sequence=after_sequence),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Disable response buffering in proxies (nginx, Render, etc.)
            # so events reach the browser in near-real-time.
            "X-Accel-Buffering": "no",
            "X-BESSER-Run-Id": run_id,
        },
    )


@router.post("/spec-driven/generate", response_class=StreamingResponse)
async def smart_generate(
    request: SmartGenerateRequest,
    runtime: SpecDrivenRuntimeDep,
    settings: SettingsDep,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> StreamingResponse:
    """Stream an LLM-orchestrated code generation run as SSE events.

    Emits events in order: ``start`` → zero or more of
    ``phase`` / ``tool_call`` / ``text`` / ``cost`` / ``model_update``
    (mid-run outage-fallback model switch) → optional
    ``error(COST_CAP|TIMEOUT)`` warning → terminal ``done`` (carrying a
    one-time ``downloadUrl``) or terminal ``error``.

    The ``api_key`` field in the request body is a ``SecretStr`` and is
    never logged, never echoed in events, and never stored. The download
    URL returned in the ``done`` event is single-use — the first GET
    against ``/spec-driven/download/{runId}`` serves the file; subsequent GETs
    return 404.
    """
    # Authorisation before anything is allocated: a sponsored-tier run spends
    # the org's own credits and needs the demo secret.
    _require_demo_token(request)

    # A retried start must attach to the original run, not spawn a second
    # one. Checked before the slot is acquired so a retry cannot 429 against
    # the very run it is trying to rejoin.
    idem_key = (idempotency_key or "").strip()[:200]
    if idem_key:
        async with runtime.idempotency_lock:
            _prune_idempotency_keys(runtime.idempotent_runs, time.time())
            known = runtime.idempotent_runs.get(idem_key)
            if known is not None:
                existing_run_id = known[0]
                if runtime.durable_runs.get_run(existing_run_id) is not None:
                    logger.info(
                        "Idempotent retry for run %s — replaying from sequence 0",
                        existing_run_id,
                    )
                    return _run_stream_response(runtime.durable_runs, existing_run_id)
                # The run aged out of the store; let this request start a
                # fresh one under the same key.
                runtime.idempotent_runs.pop(idem_key, None)

    # Reserve a concurrency slot BEFORE allocating any resources.
    # ``slots.try_acquire`` is non-blocking: the client sees an
    # immediate 429 when the server is saturated, never a hung SSE
    # stream. The wrapper around the stream returns the slot on exit.
    if not runtime.slots.try_acquire():
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many spec-driven generation runs are in flight right now. "
                "Retry in a moment."
            ),
        )

    # Incremental modify: when the request carries mode="modify" and a
    # base_run_id, the runner seeds this run's workspace from that previous
    # run's files and edits them in place (falling back to from-scratch if
    # the base has expired). Both fields default to the from-scratch path.
    run_id = uuid.uuid4().hex
    if idem_key:
        # Claim the key with this run id BEFORE the run starts, so a retry
        # that arrives while startup is still in flight waits for this run
        # rather than opening a second one.
        async with runtime.idempotency_lock:
            runtime.idempotent_runs[idem_key] = (run_id, time.time())
    cancel_event = await runtime.active_runs.reserve(run_id)
    if cancel_event is None:  # UUID collision is fantastically unlikely.
        runtime.slots.release()
        if idem_key:
            async with runtime.idempotency_lock:
                runtime.idempotent_runs.pop(idem_key, None)
        raise HTTPException(status_code=409, detail="Generated run ID is already active")

    runner = SmartGenerationRunner(
        request,
        runtime=runtime,
        run_id=run_id,
        reserved_cancel_event=cancel_event,
        base_run_id=request.base_run_id,
        mode=request.mode,
    )
    source = _stream_with_slot_release(
        runner.generate_and_stream(),
        runtime,
        run_reservation=(run_id, cancel_event),
    )
    try:
        await runtime.durable_runs.start(
            run_id,
            source,
            on_abandoned=(
                cancel_event.set if settings.llm_cancel_abandoned_runs else None
            ),
            resume_available=lambda: _locate_run_temp_dir(run_id) is not None,
            disconnect_grace_seconds=settings.llm_disconnected_grace_seconds,
        )
    except Exception:
        await runtime.active_runs.release(run_id, cancel_event)
        runtime.slots.release()
        if idem_key:
            async with runtime.idempotency_lock:
                runtime.idempotent_runs.pop(idem_key, None)
        raise
    return _run_stream_response(runtime.durable_runs, run_id)


# ---------------------------------------------------------------------
# Durable run status + event replay
# ---------------------------------------------------------------------


@router.get("/spec-driven/runs/{run_id}")
async def get_smart_run_status(
    runtime: SpecDrivenRuntimeDep,
    run_id: str = Path(..., pattern=r"^[a-f0-9]{32}$"),
) -> dict:
    """Return durable lifecycle metadata without exposing request secrets."""
    record = runtime.durable_runs.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Spec-driven run not found")
    return record.to_api_dict()


@router.get("/spec-driven/runs/{run_id}/events.json")
async def poll_smart_run_events(
    runtime: SpecDrivenRuntimeDep,
    run_id: str = Path(..., pattern=r"^[a-f0-9]{32}$"),
    after: int = Query(default=0, ge=0, le=_MAX_SQLITE_SEQUENCE),
    limit: int = Query(default=250, ge=1, le=1000),
) -> dict:
    """Polling transport for the durable event log — the proxy-safe fallback.

    A TLS-intercepting corporate proxy buffers the
    response body before releasing it, which an SSE stream never finishes
    producing: either the headers are held so the fetch never settles, or the
    connection is torn down ("Failed to fetch"). REST and the agent WebSocket
    are unaffected because both terminate.

    So this returns the same events as a *short, terminating* JSON response the
    proxy can buffer normally. The durable store assigns sequence numbers before
    any subscriber sees a frame, so polling and streaming share one cursor:
    pass the last sequence you saw as ``after``.

    ``events[].data`` is the decoded SSE payload, so a client can feed these to
    the same reducer it feeds streamed events. ``hasMore`` says to come straight
    back, ``status``/``terminalEvent`` say when to stop.
    """
    record = runtime.durable_runs.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Spec-driven run not found")

    stored = runtime.durable_runs.store.events_after(run_id, after, limit=limit)
    events = [
        {
            "sequence": item.sequence,
            "event": item.event_type,
            "data": item.payload,
        }
        for item in stored
    ]
    cursor = events[-1]["sequence"] if events else after
    return {
        **record.to_api_dict(),
        "events": events,
        "cursor": cursor,
        # A full page almost always means more is waiting; the poller should
        # loop immediately instead of sleeping through its interval.
        "hasMore": len(stored) >= limit,
    }


@router.get("/spec-driven/runs/{run_id}/events", response_class=StreamingResponse)
async def stream_smart_run_events(
    runtime: SpecDrivenRuntimeDep,
    run_id: str = Path(..., pattern=r"^[a-f0-9]{32}$"),
    after: int = Query(default=0, ge=0, le=_MAX_SQLITE_SEQUENCE),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Replay events after a sequence, then follow the live producer.

    ``after`` supports the fetch-based frontend. ``Last-Event-ID`` keeps the
    endpoint compatible with native EventSource clients. The larger valid
    cursor wins, preventing a stale query value from replaying duplicates.
    """
    record = runtime.durable_runs.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Spec-driven run not found")

    cursor = after
    if last_event_id:
        try:
            parsed_event_id = int(last_event_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid Last-Event-ID")
        if not 0 <= parsed_event_id <= _MAX_SQLITE_SEQUENCE:
            raise HTTPException(status_code=422, detail="Invalid Last-Event-ID")
        cursor = max(cursor, parsed_event_id)

    return StreamingResponse(
        runtime.durable_runs.subscribe(run_id, after_sequence=cursor),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-BESSER-Run-Id": run_id,
        },
    )


# ---------------------------------------------------------------------
# GET /besser_api/spec-driven/config
# ---------------------------------------------------------------------


def _free_tier_model_choices() -> list[dict]:
    """The free-tier models a request may explicitly pick between.

    Exactly the server-configured allowlist, in preference order: the primary
    (default), any extra models sharing the primary endpoint
    (``BESSER_FREE_LLM_ALT_MODELS`` — e.g. ``poolside/laguna-s-2.1-free``,
    which unlike the primary advertises no daily request quota), and, when a
    distinct fallback endpoint is configured, the fallback's model. Empty when
    the free tier itself is not configured.

    The list is deliberately flat: the frontend renders it verbatim, so adding
    a free model is a server-env change with no frontend deploy. What the ids
    do NOT carry is which endpoint serves them — that pairing lives in
    ``create_llm_client``, where an alt id resolves to the primary base
    URL + token and the fallback id to its own.
    """
    if not free_tier_available():
        return []
    primary = free_tier_model()
    choices = [{"id": primary, "default": True}]
    choices.extend({"id": alt, "default": False} for alt in free_alt_models())
    fallback = free_fallback_model()
    if fallback and fallback != primary:
        choices.append({"id": fallback, "default": False})
    return choices


@router.get("/spec-driven/config")
async def smart_gen_config(settings: SettingsDep) -> dict:
    """Expose the server's current spec-driven generation configuration.

    The frontend reads this once at app startup (or before rendering
    the preview screen) so it can show the real hard caps in tooltips,
    clamp its own input fields to the server's limits, and surface
    whether tracing/checkpointing are on at this deploy.

    Nothing here is a secret — everything exposed is either a public
    cap or a feature flag. API keys and run IDs are never returned.
    """
    return {
        "caps": {
            "max_cost_usd_hard_cap": settings.llm_max_cost_usd_hard_cap,
            "max_runtime_seconds_hard_cap": settings.llm_max_runtime_seconds_hard_cap,
            "max_turns_hard_cap": settings.llm_max_turns_hard_cap,
            "default_max_cost_usd": settings.llm_default_max_cost_usd,
            "default_max_runtime_seconds": settings.llm_default_max_runtime_seconds,
            "default_max_turns": settings.llm_default_max_turns,
        },
        "download_ttl_seconds": settings.llm_download_ttl_seconds,
        "cost_emitter_interval_seconds": settings.llm_cost_emitter_interval_seconds,
        "concurrency": {
            "max_concurrent_runs": settings.llm_max_concurrent_runs,
        },
        "features": {
            "tracing_enabled": settings.llm_enable_tracing,
            "checkpointing_enabled": settings.llm_enable_checkpointing,
            "resume_enabled": settings.llm_enable_checkpointing,
            "toolchain_validation_enabled": settings.llm_enable_toolchain_validation,
            # False on any deploy that has not opted in. Reported so a local/on-prem operator can confirm from outside
            # the process that BESSER_LLM_ENABLE_SHELL_TOOLS actually took
            # effect - it is a process-start env var, never a request field.
            "shell_tools_enabled": settings.llm_enable_shell_tools,
            "per_write_diagnostics_enabled": settings.llm_per_write_diagnostics,
            "durable_runs_enabled": True,
            "event_replay_enabled": True,
            "cancel_abandoned_runs": settings.llm_cancel_abandoned_runs,
            "disconnected_grace_seconds": settings.llm_disconnected_grace_seconds,
        },
        "supported_providers": ["anthropic", "openai", "mistral", "nebius"],
        # The key dialog's providers and models, from the model catalog
        # (BESSER_MODEL_CATALOG). Gateway endpoints and rates stay server-side.
        "providers": load_model_catalog().public_providers(),
        # Per-provider default model names, sourced from the LLM client
        # layer (the model catalog's default, else the built-in one).
        "default_models": {p: _llm_default_model(p) for p in _LLM_DEFAULT_MODELS},
        # Keyless free tier (server-hosted open-weight model). The frontend
        # shows the "Free" option only when ``available`` is true, so a deploy
        # without the endpoint configured never offers a button that 500s.
        "free_tier": {
            "available": free_tier_available(),
            "model": free_tier_model() or None,
            # The model ids a free-tier request may explicitly choose between.
            # The server honors exactly these ids (anything else pins to the
            # default): the primary (default), any alt models on the primary
            # endpoint, and the fallback endpoint's model when configured.
            "models": _free_tier_model_choices(),
            # The model a study session (?study=<label> or ?pilot=<label>) pre-selects.
            # Null when unset. Server-side so swapping it is an env edit, not a
            # frontend release -- the client never hardcodes a model id.
            "pilot_model": free_pilot_model() or None,
        },
    }


# ---------------------------------------------------------------------
# POST /besser_api/spec-driven/preview
# ---------------------------------------------------------------------


@router.post("/spec-driven/preview")
@handle_endpoint_errors("smart_preview")
async def smart_preview(request: SmartPreviewRequest, runtime: SpecDrivenRuntimeDep) -> dict:
    """Return the plan spec-driven generate would run, without executing it.

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
          "model_summary": {"primary": "class", "present": [...]},
          "model_change": {"verdict": "large", "gone": 0.54, ...}  # modify only
        }

    Errors are mapped by ``handle_endpoint_errors`` (``ValueError`` -> 400).
    """
    assembled = assemble_models_from_project(
        request.project, primary_kind_override=request.primary_kind_override,
    )
    plan = build_preview(
        assembled,
        instructions=request.instructions,
        max_cost_usd=request.max_cost_usd,
        max_runtime_seconds=request.max_runtime_seconds,
        mode=request.mode,
    )
    payload = plan.to_dict()
    payload["model_summary"] = assembled.summary()
    # For a modify: how the model relates to the base app's, so the UI can ask
    # "update the existing app / start fresh". None = unknown, keep modifying.
    model_change = None
    if request.mode == "modify" and request.base_run_id:
        base = await runtime.registry.get(request.base_run_id)
        if base is not None and base.temp_dir:
            change = compare_with_base(base.temp_dir, assembled.domain_model)
            model_change = change.to_dict() if change else None
    payload["model_change"] = model_change
    return payload


# ---------------------------------------------------------------------
# POST /besser_api/spec-driven/resume/{run_id}
# ---------------------------------------------------------------------


@router.post("/spec-driven/resume/{run_id}", response_class=StreamingResponse)
async def resume_smart_gen(
    run_id: str,
    request: SmartGenerateRequest,
    runtime: SpecDrivenRuntimeDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Resume a spec-driven generate run that crashed before completion.

    Takes the same ``SmartGenerateRequest`` shape as ``/spec-driven/generate``
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
    # Resume takes the same request shape as /generate and spends the same
    # way — gate it identically or it becomes the way around the gate.
    _require_demo_token(request)

    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(status_code=422, detail="Invalid run_id format")

    temp_dir = _locate_run_temp_dir(run_id)
    if temp_dir is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No recoverable workspace for this run_id. The crashed "
                "run's temp directory has either been swept or the run "
                "completed cleanly. Start a fresh run via /spec-driven/generate."
            ),
        )

    # Atomically reserve the checkpoint/run ID before the global slot.
    # This prevents the original run and a resume (or two resumes) from
    # sharing one workspace and checkpoint.
    cancel_event = await runtime.active_runs.reserve(run_id)
    if cancel_event is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This spec-driven generation run is already active or being resumed."
            ),
        )

    # Same concurrency gate as /spec-driven/generate - a resumed run consumes
    # the same resources as a fresh one.
    if not runtime.slots.try_acquire():
        await runtime.active_runs.release(run_id, cancel_event)
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many spec-driven generation runs are in flight right now. "
                "Retry in a moment."
            ),
        )

    runner = SmartGenerationRunner(
        request,
        runtime=runtime,
        resume_run_id=run_id,
        reserved_cancel_event=cancel_event,
    )
    source = _stream_with_slot_release(
        runner.generate_and_stream(),
        runtime,
        run_reservation=(run_id, cancel_event),
    )
    try:
        await runtime.durable_runs.start(
            run_id,
            source,
            resume=True,
            on_abandoned=(
                cancel_event.set if settings.llm_cancel_abandoned_runs else None
            ),
            resume_available=lambda: _locate_run_temp_dir(run_id) is not None,
            disconnect_grace_seconds=settings.llm_disconnected_grace_seconds,
        )
    except Exception:
        await runtime.active_runs.release(run_id, cancel_event)
        runtime.slots.release()
        raise
    return StreamingResponse(
        runtime.durable_runs.subscribe(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-BESSER-Run-Id": run_id,
        },
    )


# ---------------------------------------------------------------------
# POST /besser_api/spec-driven/cancel/{run_id}
# ---------------------------------------------------------------------


@router.post("/spec-driven/cancel/{run_id}")
@handle_endpoint_errors("cancel_smart_gen")
async def cancel_smart_gen(
    runtime: SpecDrivenRuntimeDep,
    run_id: str = Path(
        ...,
        pattern=r"^[a-f0-9]{32}$",
        description="Hex run ID returned in the `start` SSE event",
    ),
) -> dict:
    """Signal a live spec-driven generation run to stop at its next turn.

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
    cancelled = await runtime.active_runs.request_cancellation(run_id)
    return {"status": "cancelled" if cancelled else "not_found", "runId": run_id}


# ---------------------------------------------------------------------
# GET /besser_api/spec-driven/download/{run_id}
# ---------------------------------------------------------------------


@router.get("/spec-driven/download/{run_id}", response_class=StreamingResponse)
@handle_endpoint_errors("download_smart")
async def download_smart(
    runtime: SpecDrivenRuntimeDep,
    run_id: str = Path(
        ...,
        pattern=r"^[a-f0-9]{32}$",
        description="Hex run ID returned in the `done` SSE event",
    ),
) -> StreamingResponse:
    """Download the output produced by a completed run.

    Re-downloadable until the TTL sweep expires the entry
    (``BESSER_LLM_DOWNLOAD_TTL_SECONDS``, default 30 min) — a transient network
    failure during the blob fetch must not permanently lose an artifact
    the user paid minutes and dollars to produce. Cleanup is owned
    entirely by ``SmartRunRegistry.periodic_sweep``; this endpoint no
    longer deletes anything.
    """
    entry = await runtime.registry.get(run_id)
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
# POST /besser_api/spec-driven/push-to-github
# ---------------------------------------------------------------------


@router.post("/spec-driven/push-to-github", response_model=PushSmartToGitHubResponse)
async def push_spec_driven_to_github(
    req: PushSmartToGitHubRequest,
    runtime: SpecDrivenRuntimeDep,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
) -> PushSmartToGitHubResponse:
    """Push a finished spec-driven generation run to a GitHub repository.

    Unlike ``/deploy-webapp`` (which regenerates deterministically and
    would discard the LLM's customizations), this pushes the *stored*
    artifact for ``run_id`` — the exact code the user saw and paid for —
    plus the re-importable model source under ``buml/``.

    Flow:
      1. Gate on the ``X-GitHub-Session`` header (same helper the other
         GitHub endpoints use — the real token never reaches the client).
      2. Resolve the stored run from the run registry (404 when it
         has been swept past its TTL — the user must re-generate).
      3. Copy the stored file tree into a fresh temp dir, skipping build/
         dependency dirs, ``.besser_*`` internals, and any ``.env`` that
         carries a real secret.
      4. Inject ``buml/`` model files + ``buml/diagrams.json``.
      5. Create (default private) or reuse the repo, resolve the branch.
      6. Commit what the run changed relative to the commit it started from:
         files the run never held (build output, ``*.db``, symlinks, secret
         files) stay as they are in the repo.
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
    entry = await runtime.registry.get(req.run_id)
    if entry is None or not entry.temp_dir or not os.path.isdir(entry.temp_dir):
        # Machine-usable code the frontend can branch on, plus a human
        # hint: "This generation has expired — re-generate to push."
        raise HTTPException(status_code=404, detail="run_expired")

    deploy_config = req.deploy_config
    repo_name = _sanitize_repo_name(deploy_config.repo_name)

    try:
        return await push_run_to_github(entry, req, access_token, repo_name, runtime.registry)
    except UpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise _github_status_error(exc, "push_spec_driven_to_github") from exc
    except Exception:
        logger.exception("Unexpected error in push_spec_driven_to_github")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during GitHub push.",
        )


# ---------------------------------------------------------------------
# POST /besser_api/spec-driven/import-github-run
# ---------------------------------------------------------------------


@router.post("/spec-driven/import-github-run", response_model=ImportGitHubRunResponse)
async def import_github_run(
    req: ImportGitHubRunRequest,
    runtime: SpecDrivenRuntimeDep,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
) -> ImportGitHubRunResponse:
    """Import an existing BESSER-created GitHub repo as a modify seed.

    The counterpart to ``/spec-driven/push-to-github``: instead of *writing* a
    finished run to GitHub, this *reads* a repo we previously created (so
    it carries the generated code plus a re-importable ``buml/diagrams.json``)
    back into the editor so the user can continue from it.

    Flow:
      1. Gate on the ``X-GitHub-Session`` header (same helper the other
         GitHub endpoints use — the real token never reaches the client).
      2. Resolve the target branch (repo default when none is requested).
      3. Pin the branch head, then download + extract the repo at that
         commit into the shared run workspace.
      4. Register that code tree as a run in the run registry under a
         fresh ``run_id``. A later ``/spec-driven/generate`` with
         ``mode="modify"`` and ``base_run_id=run_id`` seeds its workspace from
         exactly this tree, in whichever container runs it.
      5. Read ``buml/diagrams.json`` (the re-importable model) if present.

    Returns ``{ run_id, project, has_model, owner, repo, branch, commit_sha,
    message }``.
    ``project`` is the parsed ``diagrams.json`` (or ``null`` when the repo
    has no BESSER model — the frontend must then tell the user to open the
    repo in the editor first before it can be smart-modified).
    """
    # ---- 1. GitHub auth gate (identical to spec-driven push-to-github) ----
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

    # An import downloads and unpacks up to the archive caps and may call an
    # LLM, so it takes a run slot (BESSER_LLM_MAX_CONCURRENT_RUNS) like a run.
    if not runtime.slots.try_acquire():
        raise HTTPException(
            status_code=429,
            detail="Too many imports and generation runs are in flight right now. Retry in a moment.",
        )
    try:
        return await import_run_from_github(req, access_token, runtime.registry)
    except UpstreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise _github_status_error(exc, "import_github_run", repo_missing_on_404=True) from exc
    except ValueError as exc:
        # download_repo_tarball's size / member-count / unsafe-path refusals.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error in import_github_run")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during GitHub import.",
        )
    finally:
        runtime.slots.release()
