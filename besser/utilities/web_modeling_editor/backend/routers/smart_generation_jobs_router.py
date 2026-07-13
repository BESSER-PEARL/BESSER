"""Owner-bound HTTP API for durable SmartGen runs."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path as FilePath
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.services.principal import (
    Principal,
    get_current_principal,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_jobs import (
    ApprovalConflictError,
    DURABLE_JOB_RUNTIME,
    DurableDispatchError,
    DurableJobsDisabledError,
    DurableQuotaError,
    DurableRequestTooLargeError,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state import (
    ArtifactRef,
    DurableStateConfigurationError,
    DurableStateError,
    IdempotencyConflictError,
    OptimisticLockError,
    RecordNotFoundError,
    ReplayCursor,
    RunRecord,
    RunStatus,
    StorageIntegrityError,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.secret_envelope import (
    SecretEnvelopeError,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/besser_api/smart-gen/runs", tags=["smart-generation"])

_RUN_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
_APPROVAL_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
_SSE_EVENT_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class RunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus


class DurableRunSummary(BaseModel):
    run_id: str
    status: RunStatus
    mode: str
    provider: str
    model: str | None
    estimated_cost_usd: float
    created_at: float
    updated_at: float
    started_at: float | None
    completed_at: float | None
    error_code: str | None


class DurableRunDetail(DurableRunSummary):
    max_cost_usd: float
    max_runtime_seconds: int
    error_message: str | None
    artifact_available: bool
    events_url: str
    artifact_url: str | None
    approvals: dict[str, str]


class DurableRunList(BaseModel):
    runs: list[DurableRunSummary]


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]


class ApprovalDecisionResponse(BaseModel):
    run_id: str
    approval_id: str
    status: Literal["approved", "rejected"]


def _summary(record: RunRecord) -> DurableRunSummary:
    return DurableRunSummary(
        run_id=record.run_id,
        status=record.status,
        mode=record.mode,
        provider=record.provider,
        model=record.model,
        estimated_cost_usd=record.estimated_cost_usd,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error_code=record.error_code,
    )


def _approval_statuses(record: RunRecord) -> dict[str, str]:
    raw_approvals = record.metadata.get("approvals")
    if not isinstance(raw_approvals, Mapping):
        return {}
    statuses: dict[str, str] = {}
    for approval_id, raw_approval in raw_approvals.items():
        if not isinstance(approval_id, str) or not isinstance(raw_approval, Mapping):
            continue
        approval_status = raw_approval.get("status")
        if approval_status in {"pending", "approved", "rejected", "timed_out"}:
            statuses[approval_id] = approval_status
    return statuses


def _detail(record: RunRecord) -> DurableRunDetail:
    artifact_available = bool(
        record.artifact_key
        and record.status in {RunStatus.SUCCEEDED, RunStatus.INCOMPLETE}
    )
    base = _summary(record).model_dump()
    return DurableRunDetail(
        **base,
        max_cost_usd=record.max_cost_usd,
        max_runtime_seconds=record.max_runtime_seconds,
        error_message=record.error_message,
        artifact_available=artifact_available,
        events_url=f"/besser_api/smart-gen/runs/{record.run_id}/events",
        artifact_url=(
            f"/besser_api/smart-gen/runs/{record.run_id}/artifact"
            if artifact_available
            else None
        ),
        approvals=_approval_statuses(record),
    )


def _raise_service_error(exc: Exception) -> None:
    if isinstance(exc, DurableRequestTooLargeError):
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if isinstance(exc, IdempotencyConflictError):
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was already used for a different request.",
        ) from exc
    if isinstance(exc, ApprovalConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, OptimisticLockError):
        raise HTTPException(
            status_code=409,
            detail="The run changed concurrently. Please retry.",
        ) from exc
    if isinstance(exc, DurableQuotaError):
        headers = None
        if exc.retry_after_seconds is not None:
            headers = {"Retry-After": str(max(1, exc.retry_after_seconds))}
        raise HTTPException(status_code=429, detail=str(exc), headers=headers) from exc
    if isinstance(exc, DurableDispatchError):
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to queue generation work. Retry with the same "
                "Idempotency-Key."
            ),
        ) from exc
    if isinstance(exc, DurableJobsDisabledError):
        raise HTTPException(
            status_code=503,
            detail="Durable SmartGen jobs are unavailable.",
        ) from exc
    if isinstance(
        exc,
        (DurableStateConfigurationError, DurableStateError, SecretEnvelopeError),
    ):
        raise HTTPException(
            status_code=503,
            detail="Durable SmartGen is temporarily unavailable.",
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.exception("Unexpected durable SmartGen API failure", exc_info=exc)
    raise HTTPException(status_code=500, detail="Internal server error") from exc


async def _owned_run(owner_id: str, run_id: str) -> RunRecord:
    try:
        record = await DURABLE_JOB_RUNTIME.get_owned_run(owner_id, run_id)
    except Exception as exc:
        _raise_service_error(exc)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunStatusResponse,
)
async def enqueue_run(
    payload: SmartGenerateRequest,
    response: Response,
    principal: Principal = Depends(get_current_principal),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunStatusResponse:
    if idempotency_key is None or not idempotency_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key header is required.",
        )
    try:
        enqueued = await DURABLE_JOB_RUNTIME.enqueue(
            payload,
            owner_id=principal.subject,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        _raise_service_error(exc)
    response.headers["Location"] = (
        f"/besser_api/smart-gen/runs/{enqueued.record.run_id}"
    )
    response.headers["Cache-Control"] = "no-store"
    return RunStatusResponse(
        run_id=enqueued.record.run_id,
        status=enqueued.record.status,
    )


@router.get("", response_model=DurableRunList)
async def list_runs(
    principal: Principal = Depends(get_current_principal),
    limit: int = Query(default=50, ge=1, le=100),
) -> DurableRunList:
    try:
        records = await DURABLE_JOB_RUNTIME.list_owned_runs(
            principal.subject,
            limit=limit,
        )
    except Exception as exc:
        _raise_service_error(exc)
    return DurableRunList(runs=[_summary(record) for record in records])


@router.get("/{run_id}", response_model=DurableRunDetail)
async def get_run(
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
    principal: Principal = Depends(get_current_principal),
) -> DurableRunDetail:
    return _detail(await _owned_run(principal.subject, run_id))


def _last_event_sequence(value: str | None) -> int:
    if value is None or value == "":
        return 0
    if not re.fullmatch(r"[0-9]+", value):
        raise HTTPException(
            status_code=400,
            detail="Last-Event-ID must be a non-negative integer.",
        )
    sequence = int(value)
    if sequence > 2**63 - 1:
        raise HTTPException(status_code=400, detail="Last-Event-ID is too large.")
    return sequence


def _sse_frame(sequence: int, event_type: str, payload: Mapping[str, object]) -> bytes:
    payload_event = payload.get("event")
    candidate = payload_event if isinstance(payload_event, str) else event_type
    safe_event = candidate if _SSE_EVENT_NAME.fullmatch(candidate) else event_type
    encoded = json.dumps(
        dict(payload),
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"id: {sequence}\nevent: {safe_event}\ndata: {encoded}\n\n".encode("utf-8")


async def _event_stream(
    request: Request,
    owner_id: str,
    run_id: str,
    after_sequence: int,
    follow: bool,
    first_page,
) -> AsyncIterator[bytes]:
    sequence = after_sequence
    page = first_page
    empty_terminal_polls = 0
    last_heartbeat = time.monotonic()
    while True:
        for event in page.events:
            sequence = event.sequence
            yield _sse_frame(event.sequence, event.event_type, event.payload)
        if page.has_more:
            page = await DURABLE_JOB_RUNTIME.foundation.state.read_events(
                run_id,
                cursor=ReplayCursor(run_id, sequence),
                limit=DURABLE_JOB_RUNTIME.foundation.config.event_page_size,
            )
            continue
        if not follow:
            return
        if await request.is_disconnected():
            return

        current = await DURABLE_JOB_RUNTIME.foundation.state.get_owned_run(
            owner_id,
            run_id,
        )
        if current is None:
            return
        stream_complete = current.terminal or bool(
            current.status == RunStatus.INCOMPLETE
            and current.metadata.get("worker_finalized") is True
        )
        if stream_complete and not page.events:
            empty_terminal_polls += 1
            if empty_terminal_polls >= 2:
                return
        else:
            empty_terminal_polls = 0

        await asyncio.sleep(0.5)
        now = time.monotonic()
        if now - last_heartbeat >= 15:
            yield b": keep-alive\n\n"
            last_heartbeat = now
        page = await DURABLE_JOB_RUNTIME.foundation.state.read_events(
            run_id,
            cursor=ReplayCursor(run_id, sequence),
            limit=DURABLE_JOB_RUNTIME.foundation.config.event_page_size,
        )


@router.get("/{run_id}/events", response_class=StreamingResponse)
async def replay_events(
    request: Request,
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
    principal: Principal = Depends(get_current_principal),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    follow: bool = Query(default=True),
) -> StreamingResponse:
    await _owned_run(principal.subject, run_id)
    after_sequence = _last_event_sequence(last_event_id)
    try:
        first_page = await DURABLE_JOB_RUNTIME.foundation.state.read_events(
            run_id,
            cursor=ReplayCursor(run_id, after_sequence),
            limit=DURABLE_JOB_RUNTIME.foundation.config.event_page_size,
        )
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except Exception as exc:
        _raise_service_error(exc)
    return StreamingResponse(
        _event_stream(
            request,
            principal.subject,
            run_id,
            after_sequence,
            follow,
            first_page,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{run_id}/cancel", response_model=RunStatusResponse)
async def cancel_run(
    response: Response,
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
    principal: Principal = Depends(get_current_principal),
) -> RunStatusResponse:
    try:
        result = await DURABLE_JOB_RUNTIME.request_cancellation(
            principal.subject,
            run_id,
        )
    except Exception as exc:
        _raise_service_error(exc)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    response.status_code = (
        status.HTTP_202_ACCEPTED if result.accepted else status.HTTP_200_OK
    )
    return RunStatusResponse(run_id=result.record.run_id, status=result.record.status)


@router.post(
    "/{run_id}/approvals/{approval_id}",
    response_model=ApprovalDecisionResponse,
)
async def resolve_approval(
    payload: ApprovalDecisionRequest,
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
    approval_id: str = Path(pattern=_APPROVAL_ID_PATTERN),
    principal: Principal = Depends(get_current_principal),
) -> ApprovalDecisionResponse:
    try:
        result = await DURABLE_JOB_RUNTIME.resolve_approval(
            principal.subject,
            run_id,
            approval_id,
            payload.decision,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Approval not found") from exc
    except Exception as exc:
        _raise_service_error(exc)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return ApprovalDecisionResponse(
        run_id=result.record.run_id,
        approval_id=result.approval_id,
        status=result.decision,
    )


def _artifact_ref(record: RunRecord) -> ArtifactRef:
    raw = record.metadata.get("artifact")
    if not isinstance(raw, Mapping):
        raise FileNotFoundError("Artifact metadata is missing")
    try:
        storage_key = raw["storage_key"]
        file_name = raw["file_name"]
        size_bytes = raw["size_bytes"]
        sha256 = raw["sha256"]
        content_type = raw["content_type"]
        created_at = raw["created_at"]
    except KeyError as exc:
        raise StorageIntegrityError("Artifact metadata is incomplete") from exc
    if (
        not isinstance(storage_key, str)
        or storage_key != record.artifact_key
        or not isinstance(file_name, str)
        or not file_name
        or len(file_name) > 160
        or FilePath(file_name).name != file_name
        or "\r" in file_name
        or "\n" in file_name
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or not isinstance(sha256, str)
        or not re.fullmatch(r"[a-f0-9]{64}", sha256)
        or not isinstance(content_type, str)
        or not content_type
        or "\r" in content_type
        or "\n" in content_type
        or isinstance(created_at, bool)
        or not isinstance(created_at, (int, float))
        or not math.isfinite(float(created_at))
        or float(created_at) < 0
    ):
        raise StorageIntegrityError("Artifact metadata failed validation")
    return ArtifactRef(
        storage_key=storage_key,
        file_name=file_name,
        size_bytes=size_bytes,
        sha256=sha256,
        content_type=content_type,
        created_at=float(created_at),
    )


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@router.get("/{run_id}/artifact")
async def download_artifact(
    run_id: str = Path(pattern=_RUN_ID_PATTERN),
    principal: Principal = Depends(get_current_principal),
):
    record = await _owned_run(principal.subject, run_id)
    if record.status == RunStatus.EXPIRED:
        raise HTTPException(status_code=410, detail="Artifact has expired")
    if record.status not in {RunStatus.SUCCEEDED, RunStatus.INCOMPLETE}:
        raise HTTPException(status_code=409, detail="Artifact is not ready")
    try:
        artifact = _artifact_ref(record)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    except StorageIntegrityError as exc:
        logger.exception("Invalid artifact metadata for SmartGen run %s", run_id)
        raise HTTPException(status_code=500, detail="Artifact metadata is invalid") from exc

    try:
        signed_url = await DURABLE_JOB_RUNTIME.foundation.artifacts.create_download_url(
            artifact,
            expires_seconds=(
                DURABLE_JOB_RUNTIME.foundation.config.artifact_url_ttl_seconds
            ),
        )
    except Exception as exc:
        logger.exception("Unable to create artifact URL for SmartGen run %s", run_id)
        raise HTTPException(
            status_code=503,
            detail="Artifact download is temporarily unavailable",
        ) from exc
    if signed_url is not None:
        if not signed_url.startswith("https://"):
            raise HTTPException(status_code=500, detail="Artifact URL is invalid")
        return RedirectResponse(
            signed_url,
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Cache-Control": "private, no-store"},
        )

    suffix = FilePath(artifact.file_name).suffix[:16]
    file_descriptor, destination = tempfile.mkstemp(
        prefix="besser-smartgen-download-",
        suffix=suffix,
    )
    os.close(file_descriptor)
    try:
        await DURABLE_JOB_RUNTIME.foundation.artifacts.download_artifact(
            artifact,
            destination,
        )
    except FileNotFoundError as exc:
        _remove_file(destination)
        raise HTTPException(status_code=410, detail="Artifact is no longer available") from exc
    except StorageIntegrityError as exc:
        _remove_file(destination)
        logger.exception("Artifact integrity failure for SmartGen run %s", run_id)
        raise HTTPException(status_code=500, detail="Artifact integrity check failed") from exc
    except Exception as exc:
        _remove_file(destination)
        logger.exception("Artifact download failure for SmartGen run %s", run_id)
        raise HTTPException(
            status_code=503,
            detail="Artifact download is temporarily unavailable",
        ) from exc
    return FileResponse(
        destination,
        media_type=artifact.content_type,
        filename=artifact.file_name,
        headers={"Cache-Control": "private, no-store"},
        background=BackgroundTask(_remove_file, destination),
    )
