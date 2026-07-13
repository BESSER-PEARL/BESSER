"""Durable SmartGen controller service and worker dispatch.

The existing ``/smart-generate`` streaming route remains available for local
compatibility.  This module powers the production path where an HTTP request
creates an owner-bound run record, encrypts BYOK material, enqueues a small job
reference, and starts an isolated worker.  Browser disconnects therefore do
not own or cancel paid work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Mapping

from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)

from .durable_state import (
    DurableStateFoundation,
    DurableStateConfigurationError,
    IdempotencyConflictError,
    JobMessage,
    OptimisticLockError,
    RecordAlreadyExistsError,
    RunRecord,
    RunStatus,
    build_durable_state,
)
from .secret_envelope import SecretEnvelope, build_secret_envelope


_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
logger = logging.getLogger(__name__)


class DurableJobsDisabledError(RuntimeError):
    """Raised when a caller uses queued endpoints before enabling them."""


class DurableDispatchError(RuntimeError):
    """Raised when a durable run was queued but no worker could be started."""

    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id


class DurableRequestTooLargeError(ValueError):
    """Raised when a validated request exceeds the durable payload limit."""


class ApprovalConflictError(RuntimeError):
    """Raised when an approval cannot be resolved in its current state."""


class DurableQuotaError(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def durable_jobs_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    mode = source.get("BESSER_SMARTGEN_STATE_MODE", "local").strip().lower()
    if mode not in {"local", "production"}:
        raise DurableStateConfigurationError(
            "BESSER_SMARTGEN_STATE_MODE must be 'local' or 'production'"
        )
    configured = source.get("BESSER_SMARTGEN_DURABLE_JOBS")
    if configured is None:
        return mode == "production"
    normalized = configured.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise DurableStateConfigurationError(
        "BESSER_SMARTGEN_DURABLE_JOBS must be a boolean"
    )


def canonical_request_hash(request: SmartGenerateRequest, owner_id: str) -> str:
    payload = request.model_dump(mode="json", exclude={"api_key"})
    encoded = json.dumps(
        {"owner_id": owner_id, "request": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _WorkerLauncher:
    async def launch(self, run_id: str) -> None:
        raise NotImplementedError


class _LocalWorkerLauncher(_WorkerLauncher):
    def __init__(self, runtime: "DurableJobRuntime") -> None:
        self._runtime = runtime

    async def launch(self, run_id: str) -> None:
        existing = self._runtime.local_task(run_id)
        if existing is not None and not existing.done():
            return
        from besser.utilities.web_modeling_editor.backend.workers.smart_generation_worker import (
            process_one_job,
        )

        task = asyncio.create_task(
            process_one_job(
                self._runtime.foundation,
                envelope=self._runtime.envelope,
            ),
            name=f"smartgen-local-worker-{run_id[:12]}",
        )
        self._runtime.track_local_task(run_id, task)


class _ManagedWorkerLauncher(_WorkerLauncher):
    """Production workers are a managed service polling the durable queue."""

    async def launch(self, run_id: str) -> None:
        del run_id


@dataclass(slots=True)
class EnqueuedRun:
    record: RunRecord
    created: bool


@dataclass(slots=True)
class CancellationResult:
    record: RunRecord
    accepted: bool


@dataclass(slots=True)
class ApprovalResolutionResult:
    record: RunRecord
    approval_id: str
    decision: str
    changed: bool


class DurableJobRuntime:
    def __init__(self) -> None:
        self._foundation: DurableStateFoundation | None = None
        self._envelope: SecretEnvelope | None = None
        self._launcher: _WorkerLauncher | None = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._local_tasks: dict[str, asyncio.Task[Any]] = {}
        self._admission_tasks: set[asyncio.Task[EnqueuedRun]] = set()

    @property
    def foundation(self) -> DurableStateFoundation:
        if self._foundation is None:
            raise DurableJobsDisabledError("Durable SmartGen has not been initialized")
        return self._foundation

    @property
    def envelope(self) -> SecretEnvelope:
        if self._envelope is None:
            raise DurableJobsDisabledError("Durable SmartGen has not been initialized")
        return self._envelope

    def local_task(self, run_id: str) -> asyncio.Task[Any] | None:
        return self._local_tasks.get(run_id)

    def track_local_task(self, run_id: str, task: asyncio.Task[Any]) -> None:
        self._local_tasks[run_id] = task

        def discard(completed: asyncio.Task[Any]) -> None:
            if self._local_tasks.get(run_id) is completed:
                self._local_tasks.pop(run_id, None)

        task.add_done_callback(discard)

    async def initialize(self) -> None:
        if not durable_jobs_enabled():
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            foundation = build_durable_state()
            try:
                await foundation.initialize()
                envelope = build_secret_envelope(foundation.config.mode.value)
                if foundation.config.mode.value == "production":
                    launcher: _WorkerLauncher = _ManagedWorkerLauncher()
                else:
                    launcher = _LocalWorkerLauncher(self)
            except BaseException:
                try:
                    await foundation.close()
                except Exception:
                    logger.exception("Failed to close partially initialized SmartGen state")
                raise
            self._foundation = foundation
            self._envelope = envelope
            self._launcher = launcher
            self._initialized = True

    async def close(self) -> None:
        admissions = tuple(self._admission_tasks)
        if admissions:
            await asyncio.gather(*admissions, return_exceptions=True)
        self._admission_tasks.clear()
        tasks = tuple(self._local_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._local_tasks.clear()
        try:
            if self._foundation is not None:
                await self._foundation.close()
        finally:
            self._initialized = False
            self._foundation = None
            self._envelope = None
            self._launcher = None

    async def ensure_ready(self) -> None:
        if not durable_jobs_enabled():
            raise DurableJobsDisabledError("Durable SmartGen jobs are disabled")
        await self.initialize()

    async def enqueue(
        self,
        request: SmartGenerateRequest,
        *,
        owner_id: str,
        idempotency_key: str,
    ) -> EnqueuedRun:
        await self.ensure_ready()
        foundation = self.foundation
        if not _IDEMPOTENCY_RE.fullmatch(idempotency_key or ""):
            raise ValueError("Idempotency-Key has an invalid format")
        if request.mode == "modify" and request.base_run_id is None:
            raise ValueError("base_run_id is required for durable modify runs")
        if request.mode == "generate" and request.base_run_id is not None:
            raise ValueError("base_run_id is only valid for durable modify runs")

        serialized_request = request.model_dump(mode="json", exclude={"api_key"})
        serialized_size = len(
            json.dumps(
                serialized_request,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if serialized_size > foundation.config.max_request_bytes:
            raise DurableRequestTooLargeError(
                "Project payload is too large for durable generation"
            )

        request_hash = canonical_request_hash(request, owner_id)
        run_id = uuid.uuid4().hex
        claim = await foundation.state.claim_idempotency(
            owner_id,
            idempotency_key,
            request_hash,
            run_id,
            ttl_seconds=foundation.config.idempotency_ttl_seconds,
        )
        run_id = claim.record.run_id
        if not claim.created:
            existing = await foundation.state.get_owned_run(owner_id, run_id)
            if existing is not None:
                if (
                    existing.status == RunStatus.QUEUED
                    and existing.metadata.get("dispatch_state") != "published"
                ):
                    await self._recover_pending_dispatch(existing)
                return EnqueuedRun(existing, created=False)

        return await self._run_admission_task(
            self._admit_and_persist(
                request,
                owner_id=owner_id,
                run_id=run_id,
                request_hash=request_hash,
                serialized_request=serialized_request,
            )
        )

    async def _run_admission_task(
        self,
        operation: Coroutine[Any, Any, EnqueuedRun],
    ) -> EnqueuedRun:
        task = asyncio.create_task(operation, name="smartgen-durable-admission")
        self._admission_tasks.add(task)

        def consume_result(completed: asyncio.Task[EnqueuedRun]) -> None:
            self._admission_tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(consume_result)
        return await asyncio.shield(task)

    async def _admit_and_persist(
        self,
        request: SmartGenerateRequest,
        *,
        owner_id: str,
        run_id: str,
        request_hash: str,
        serialized_request: Mapping[str, Any],
    ) -> EnqueuedRun:
        foundation = self.foundation
        reservation_suffix = uuid.uuid4().hex[:12]
        concurrent_reservation = f"concurrent:{run_id}:{reservation_suffix}"
        concurrent = await foundation.state.reserve_quota(
            owner_id,
            "concurrent-runs",
            concurrent_reservation,
            amount=1,
            limit=foundation.config.max_concurrent_runs_per_owner,
            ttl_seconds=max(
                foundation.config.lease_ttl_seconds * 2,
                request.max_runtime_seconds + 300,
            ),
        )
        if not concurrent.allowed:
            raise DurableQuotaError(
                "Your SmartGen concurrency limit is currently in use.",
                retry_after_seconds=concurrent.retry_after_seconds,
            )

        hour_bucket = int(time.time() // 3600)
        start_reservation = f"start:{run_id}:{reservation_suffix}"
        starts = await foundation.state.reserve_quota(
            owner_id,
            f"starts-hour:{hour_bucket}",
            start_reservation,
            amount=1,
            limit=foundation.config.max_starts_per_hour,
            ttl_seconds=3600,
        )
        if not starts.allowed:
            await foundation.state.release_quota(concurrent_reservation)
            raise DurableQuotaError(
                "Your hourly SmartGen start limit has been reached.",
                retry_after_seconds=starts.retry_after_seconds,
            )

        record_created = False
        try:
            encrypted_key = await asyncio.to_thread(
                self.envelope.encrypt,
                request.api_key.get_secret_value(),
                run_id=run_id,
                owner_id=owner_id,
            )
            record = RunRecord(
                run_id=run_id,
                owner_id=owner_id,
                request_hash=request_hash,
                mode=request.mode,
                provider=request.provider,
                model=request.llm_model,
                max_cost_usd=request.max_cost_usd,
                max_runtime_seconds=request.max_runtime_seconds,
                metadata={
                    "request": serialized_request,
                    "api_key_envelope": encrypted_key.to_dict(),
                    "concurrency_reservation": concurrent_reservation,
                    "start_reservation": start_reservation,
                    "dispatch_state": "pending",
                },
            )
            try:
                await foundation.state.create_run(record)
                record_created = True
            except RecordAlreadyExistsError:
                await self._release_reservations(record, include_start=True)
                existing = await foundation.state.get_owned_run(owner_id, run_id)
                if existing is None:
                    raise IdempotencyConflictError(
                        "Idempotency record exists without an accessible run"
                    )
                if (
                    existing.status == RunStatus.QUEUED
                    and existing.metadata.get("dispatch_state") != "published"
                ):
                    await self._recover_pending_dispatch(existing)
                return EnqueuedRun(existing, created=False)
            await foundation.state.append_event(
                run_id,
                "phase",
                {
                    "event": "phase",
                    "phase": "select",
                    "message": "Run queued",
                },
            )
            await self._dispatch(record)
            return EnqueuedRun(record, created=True)
        except Exception as exc:
            if record_created:
                if not isinstance(exc, DurableDispatchError):
                    await self._record_dispatch_failure_safely(run_id)
                    raise DurableDispatchError(
                        "Unable to dispatch the SmartGen worker",
                        run_id=run_id,
                    ) from exc
            else:
                await self._safe_release(concurrent_reservation)
                await self._safe_release(start_reservation)
            raise

    async def get_owned_run(self, owner_id: str, run_id: str) -> RunRecord | None:
        await self.ensure_ready()
        return await self.foundation.state.get_owned_run(owner_id, run_id)

    async def list_owned_runs(
        self,
        owner_id: str,
        *,
        limit: int = 50,
    ) -> tuple[RunRecord, ...]:
        await self.ensure_ready()
        return await self.foundation.state.list_runs(owner_id, limit=limit)

    async def request_cancellation(
        self,
        owner_id: str,
        run_id: str,
        *,
        attempts: int = 4,
    ) -> CancellationResult | None:
        await self.ensure_ready()
        for attempt in range(attempts):
            current = await self.foundation.state.get_owned_run(owner_id, run_id)
            if current is None:
                return None
            if current.status == RunStatus.CANCEL_REQUESTED:
                return CancellationResult(current, accepted=True)
            if current.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                return CancellationResult(current, accepted=False)
            try:
                updated = await self.foundation.state.update_run(
                    run_id,
                    current.version,
                    {"status": RunStatus.CANCEL_REQUESTED},
                )
                return CancellationResult(updated, accepted=True)
            except OptimisticLockError:
                if attempt + 1 == attempts:
                    raise
                await asyncio.sleep(0)
        raise AssertionError("unreachable")

    async def resolve_approval(
        self,
        owner_id: str,
        run_id: str,
        approval_id: str,
        decision: str,
        *,
        attempts: int = 4,
    ) -> ApprovalResolutionResult | None:
        await self.ensure_ready()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", approval_id or ""):
            raise ValueError("approval_id has an invalid format")
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")

        for attempt in range(attempts):
            current = await self.foundation.state.get_owned_run(owner_id, run_id)
            if current is None:
                return None
            approvals = current.metadata.get("approvals")
            if not isinstance(approvals, Mapping):
                raise KeyError(approval_id)
            raw_approval = approvals.get(approval_id)
            if not isinstance(raw_approval, Mapping):
                raise KeyError(approval_id)
            approval = dict(raw_approval)
            status = approval.get("status")
            if status == decision:
                return ApprovalResolutionResult(
                    current,
                    approval_id,
                    decision,
                    changed=False,
                )
            if status != "pending":
                raise ApprovalConflictError("Approval has already been resolved")
            if current.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                raise ApprovalConflictError(
                    "Approvals can only be resolved for queued or running runs"
                )

            metadata = dict(current.metadata)
            updated_approvals = dict(approvals)
            approval.update({
                "status": decision,
                "resolved_at": time.time(),
                "resolved_by": owner_id,
            })
            updated_approvals[approval_id] = approval
            metadata["approvals"] = updated_approvals
            try:
                updated = await self.foundation.state.update_run(
                    run_id,
                    current.version,
                    {"metadata": metadata},
                )
            except OptimisticLockError:
                if attempt + 1 == attempts:
                    raise
                await asyncio.sleep(0)
                continue
            await self.foundation.state.append_event(
                run_id,
                "approval_resolved",
                {
                    "event": "approval_resolved",
                    "approvalId": approval_id,
                    "decision": decision,
                },
            )
            return ApprovalResolutionResult(
                updated,
                approval_id,
                decision,
                changed=True,
            )
        raise AssertionError("unreachable")

    async def _dispatch(self, record: RunRecord) -> None:
        try:
            message_id = await self.foundation.queue.enqueue(
                JobMessage(run_id=record.run_id, owner_id=record.owner_id, payload={}),
                deduplication_id=record.run_id,
            )
            if self._launcher is None:
                raise DurableDispatchError(
                    "No SmartGen worker launcher is configured",
                    run_id=record.run_id,
                )
            await self._launcher.launch(record.run_id)
        except DurableDispatchError:
            await self._record_dispatch_failure_safely(record.run_id)
            raise
        except Exception as exc:
            await self._record_dispatch_failure_safely(record.run_id)
            raise DurableDispatchError(
                "Unable to dispatch the SmartGen worker",
                run_id=record.run_id,
            ) from exc
        await self._mark_queue_published_safely(record.run_id, message_id)

    async def _recover_pending_dispatch(self, record: RunRecord) -> None:
        current = record
        for _ in range(5):
            await asyncio.sleep(0.05)
            refreshed = await self.foundation.state.get_owned_run(
                record.owner_id,
                record.run_id,
            )
            if refreshed is None:
                return
            current = refreshed
            if (
                current.status != RunStatus.QUEUED
                or current.metadata.get("dispatch_state") == "published"
            ):
                return
        page = await self.foundation.state.read_events(current.run_id, limit=1)
        if not page.events:
            await self.foundation.state.append_event(
                current.run_id,
                "phase",
                {
                    "event": "phase",
                    "phase": "select",
                    "message": "Run queued",
                },
            )
        await self._dispatch(current)

    async def _mark_queue_published_safely(
        self,
        run_id: str,
        message_id: str,
    ) -> None:
        try:
            for _ in range(4):
                current = await self.foundation.state.get_run(run_id)
                if current is None:
                    return
                if current.metadata.get("dispatch_state") == "published":
                    return
                metadata = dict(current.metadata)
                metadata.update({
                    "dispatch_state": "published",
                    "queue_message_id": message_id,
                    "queue_published_at": time.time(),
                })
                try:
                    await self.foundation.state.update_run(
                        run_id,
                        current.version,
                        {"metadata": metadata},
                    )
                    return
                except OptimisticLockError:
                    await asyncio.sleep(0)
            logger.warning("Could not persist queue publication for SmartGen run %s", run_id)
        except Exception:
            logger.exception("Failed to persist SmartGen queue publication for %s", run_id)

    async def _record_dispatch_failure_safely(self, run_id: str) -> None:
        try:
            await self._record_dispatch_failure(run_id)
        except Exception:
            logger.exception("Failed to reconcile SmartGen dispatch failure for %s", run_id)

    async def _record_dispatch_failure(self, run_id: str) -> None:
        record = await self.foundation.state.get_run(run_id)
        transitioned = False
        for _ in range(4):
            if record is None or record.status != RunStatus.QUEUED:
                break
            try:
                record = await self.foundation.state.update_run(
                    run_id,
                    record.version,
                    {
                        "status": RunStatus.FAILED,
                        "error_code": "DISPATCH_FAILED",
                        "error_message": "Generation could not be queued for processing.",
                    },
                )
                transitioned = True
                break
            except OptimisticLockError:
                record = await self.foundation.state.get_run(run_id)
        if transitioned:
            try:
                await self.foundation.state.append_event(
                    run_id,
                    "error",
                    {
                        "event": "error",
                        "code": "INTERNAL",
                        "message": "Generation could not be queued for processing.",
                    },
                )
            except Exception:
                logger.exception("Failed to record SmartGen dispatch error for %s", run_id)
        if record is not None:
            await self._release_reservations(record, include_start=True)

    async def _release_reservations(
        self,
        record: RunRecord,
        *,
        include_start: bool,
    ) -> None:
        concurrent = record.metadata.get("concurrency_reservation")
        if isinstance(concurrent, str):
            await self._safe_release(concurrent)
        if include_start:
            start = record.metadata.get("start_reservation")
            if isinstance(start, str):
                await self._safe_release(start)

    async def _safe_release(self, reservation_id: str) -> None:
        try:
            await self.foundation.state.release_quota(reservation_id)
        except Exception:
            logger.exception("Failed to release SmartGen quota reservation")


DURABLE_JOB_RUNTIME = DurableJobRuntime()


async def release_run_concurrency(
    foundation: DurableStateFoundation,
    record: RunRecord,
) -> bool:
    """Release a run's concurrency slot after durable terminal persistence."""

    reservation_id = record.metadata.get("concurrency_reservation")
    if not isinstance(reservation_id, str):
        return False
    return await foundation.state.release_quota(reservation_id)


async def update_run_with_retry(
    foundation: DurableStateFoundation,
    run_id: str,
    changes: Mapping[str, Any],
    *,
    attempts: int = 4,
) -> RunRecord:
    for attempt in range(attempts):
        current = await foundation.state.get_run(run_id)
        if current is None:
            raise KeyError(run_id)
        try:
            return await foundation.state.update_run(
                run_id,
                current.version,
                changes,
            )
        except OptimisticLockError:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(0)
    raise AssertionError("unreachable")
