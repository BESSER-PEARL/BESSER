"""One-job durable SmartGen worker."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import mimetypes
import os
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from pydantic import ValidationError as PydanticValidationError

from besser.generators.llm.checkpoint import CHECKPOINT_FILENAME
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    LLM_TEMP_DIR_PREFIX,
)
from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_jobs import (
    canonical_request_hash,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state import (
    ArtifactRef,
    CheckpointRef,
    DurableStateFoundation,
    LeaseLostError,
    OptimisticLockError,
    QueuedJob,
    ReplayCursor,
    RunRecord,
    RunStatus,
    build_durable_state,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
    SmartRunEntry,
    release_active_run,
    reserve_active_run,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.secret_envelope import (
    EncryptedSecret,
    SecretEnvelope,
    SecretEnvelopeError,
    build_secret_envelope,
)

logger = logging.getLogger(__name__)

_TERMINAL_ERROR_CODES = {
    "INVALID_KEY", "UPSTREAM_LLM", "INTERNAL", "BAD_REQUEST", "CANCELLED",
}
_MAX_SSE_FRAME_BYTES = 2 * 1024 * 1024
_DEFAULT_VISIBILITY_TIMEOUT_SECONDS = 3600
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_DELAY_SECONDS = 5
_DEFAULT_APPROVAL_TIMEOUT_SECONDS = 600
_DEFAULT_TASK_PROTECTION_MINUTES = 30
_DEFAULT_TASK_PROTECTION_TIMEOUT_SECONDS = 3
_MAX_TASK_PROTECTION_RESPONSE_BYTES = 64 * 1024
_CHECKPOINT_EXCLUDED_DIRECTORIES = frozenset({
    ".git",
    ".besser_snapshot",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
})


class WorkerDisposition(str, Enum):
    NO_JOB = "no_job"
    ACKNOWLEDGED = "acknowledged"
    RETRY_RELEASED = "retry_released"


@dataclass(frozen=True, slots=True)
class WorkerResult:
    disposition: WorkerDisposition
    run_status: Optional[RunStatus] = None
    reason: Optional[str] = None


class _Runner(Protocol):
    run_id: str
    owner_id: str
    temp_dir: Optional[str]

    async def generate_and_stream(self, http_request: Any | None = None): ...


class _LeaseLostError(RuntimeError):
    pass


class _PermanentJobError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message


class _CancellationRequested(RuntimeError):
    pass


class _TaskProtectionError(RuntimeError):
    pass


@dataclass(slots=True)
class _QueueDecision:
    acknowledge: bool
    status: Optional[RunStatus] = None
    reason: Optional[str] = None
    retry_delay_seconds: int = 0


class _TaskProtection(Protocol):
    async def enable(self) -> None: ...

    async def refresh(self) -> None: ...

    async def disable(self) -> None: ...


class _NoopTaskProtection:
    async def enable(self) -> None:
        return None

    async def refresh(self) -> None:
        return None

    async def disable(self) -> None:
        return None


class _UnavailableTaskProtection:
    def __init__(self, message: str) -> None:
        self.message = message

    async def enable(self) -> None:
        raise _TaskProtectionError(self.message)

    async def refresh(self) -> None:
        raise _TaskProtectionError(self.message)

    async def disable(self) -> None:
        return None


class _EcsTaskProtection:
    def __init__(
        self,
        agent_uri: str,
        *,
        expires_in_minutes: int,
        request_timeout_seconds: float,
    ) -> None:
        parsed = urllib.parse.urlsplit(agent_uri.rstrip("/"))
        if (
            parsed.scheme != "http"
            or parsed.hostname != "169.254.170.2"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("ECS_AGENT_URI is not a trusted ECS agent endpoint")
        if not 1 <= expires_in_minutes <= 2880:
            raise ValueError("task protection duration must be 1 to 2880 minutes")
        if not 0 < request_timeout_seconds <= 10:
            raise ValueError("task protection timeout must be at most 10 seconds")
        self.endpoint = (
            f"{agent_uri.rstrip('/')}/task-protection/v1/state"
        )
        self.expires_in_minutes = expires_in_minutes
        self.request_timeout_seconds = request_timeout_seconds
        self.refresh_interval_seconds = max(
            5.0,
            min(60.0, expires_in_minutes * 20.0),
        )
        self._active = False
        self._may_be_active = False
        self._last_refresh = 0.0
        self._lock = asyncio.Lock()

    async def enable(self) -> None:
        async with self._lock:
            self._may_be_active = True
            await self._set_state(True)
            self._active = True
            self._last_refresh = time.monotonic()

    async def refresh(self) -> None:
        async with self._lock:
            if not self._active:
                return
            if time.monotonic() - self._last_refresh < self.refresh_interval_seconds:
                return
            await self._set_state(True)
            self._last_refresh = time.monotonic()

    async def disable(self) -> None:
        async with self._lock:
            if not self._may_be_active:
                return
            await self._set_state(False)
            self._active = False
            self._may_be_active = False

    async def _set_state(self, enabled: bool) -> None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._set_state_sync, enabled),
                timeout=self.request_timeout_seconds + 0.5,
            )
        except (asyncio.TimeoutError, OSError, ValueError) as exc:
            raise _TaskProtectionError(
                "ECS task protection could not be updated",
            ) from exc

    def _set_state_sync(self, enabled: bool) -> None:
        payload: dict[str, Any] = {"ProtectionEnabled": enabled}
        if enabled:
            payload["ExpiresInMinutes"] = self.expires_in_minutes
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                body = response.read(_MAX_TASK_PROTECTION_RESPONSE_BYTES + 1)
                status = int(getattr(response, "status", 0))
        except (TimeoutError, urllib.error.URLError) as exc:
            raise OSError("ECS agent request failed") from exc
        if status < 200 or status >= 300:
            raise OSError("ECS agent rejected task protection update")
        if len(body) > _MAX_TASK_PROTECTION_RESPONSE_BYTES:
            raise ValueError("ECS agent response was too large")
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("ECS agent returned invalid JSON") from exc
        protection = decoded.get("protection") if isinstance(decoded, Mapping) else None
        if not isinstance(protection, Mapping):
            raise ValueError("ECS agent response omitted protection state")
        if protection.get("ProtectionEnabled") is not enabled:
            raise ValueError("ECS agent returned an unexpected protection state")


def _build_task_protection(
    mode: str,
    environ: Mapping[str, str] | None = None,
) -> _TaskProtection:
    if mode != "production":
        return _NoopTaskProtection()
    source = os.environ if environ is None else environ
    agent_uri = source.get("ECS_AGENT_URI", "").strip()
    try:
        expires_in_minutes = int(source.get(
            "BESSER_SMARTGEN_TASK_PROTECTION_MINUTES",
            str(_DEFAULT_TASK_PROTECTION_MINUTES),
        ))
        timeout_seconds = float(source.get(
            "BESSER_SMARTGEN_TASK_PROTECTION_TIMEOUT_SECONDS",
            str(_DEFAULT_TASK_PROTECTION_TIMEOUT_SECONDS),
        ))
        if not agent_uri:
            raise ValueError("ECS_AGENT_URI is required in production")
        return _EcsTaskProtection(
            agent_uri,
            expires_in_minutes=expires_in_minutes,
            request_timeout_seconds=timeout_seconds,
        )
    except (TypeError, ValueError) as exc:
        return _UnavailableTaskProtection(str(exc))


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _shell_tools_enabled(foundation: DurableStateFoundation) -> bool:
    if not _env_flag("BESSER_SMARTGEN_ALLOW_SHELL_TOOLS"):
        return False
    if foundation.config.mode.value == "production":
        return True
    return _env_flag("BESSER_SMARTGEN_ALLOW_LOCAL_SHELL_TOOLS")


def _run_ref(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]


def _decode_sse_frame(frame: bytes) -> dict[str, Any]:
    if not isinstance(frame, bytes) or len(frame) > _MAX_SSE_FRAME_BYTES:
        raise ValueError("SmartGen runner emitted an invalid SSE frame")
    try:
        data = "\n".join(
            line[6:]
            for line in frame.decode("utf-8").splitlines()
            if line.startswith("data: ")
        )
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SmartGen runner emitted a malformed SSE frame") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("event"), str):
        raise ValueError("SmartGen SSE payload has an invalid shape")
    return payload


def _serialise_ref(reference: Any) -> dict[str, Any]:
    return asdict(reference)


def _content_type(entry: SmartRunEntry) -> str:
    if entry.is_zip:
        return "application/zip"
    return mimetypes.guess_type(entry.file_name)[0] or "application/octet-stream"


class _LeaseGuard:
    """Renew the fenced lease and verify it before every durable write."""

    def __init__(
        self,
        foundation: DurableStateFoundation,
        *,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        cancel_event: asyncio.Event,
        heartbeat_interval_seconds: Optional[float],
        queued_job: QueuedJob,
        visibility_timeout_seconds: int,
        task_protection: _TaskProtection,
    ) -> None:
        self.foundation = foundation
        self.run_id = run_id
        self.worker_id = worker_id
        self.fencing_token = fencing_token
        default_interval = max(1.0, foundation.config.lease_ttl_seconds / 3)
        self.heartbeat_interval_seconds = (
            heartbeat_interval_seconds
            if heartbeat_interval_seconds is not None
            else min(10.0, default_interval)
        )
        self.cancel_event = cancel_event
        self.queued_job = queued_job
        self.visibility_timeout_seconds = visibility_timeout_seconds
        self.task_protection = task_protection
        self.lost = asyncio.Event()
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task[None]] = None

    def start(self) -> None:
        self._task = asyncio.create_task(
            self._heartbeat(), name="smartgen-durable-lease-heartbeat",
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def ensure_current(self) -> None:
        async with self._lock:
            await self._renew_locked()

    async def append_event(
        self, event_type: str, payload: Mapping[str, Any],
    ) -> Any:
        async with self._lock:
            await self._renew_locked()
            try:
                return await self.foundation.state.append_event_fenced(
                    self.run_id,
                    self.worker_id,
                    self.fencing_token,
                    event_type,
                    payload,
                )
            except LeaseLostError as exc:
                self._mark_lost()
                raise _LeaseLostError("SmartGen worker lease was lost") from exc

    async def update_run(
        self,
        changes_builder: Callable[[RunRecord], Mapping[str, Any]],
        *,
        attempts: int = 4,
    ) -> RunRecord:
        async with self._lock:
            for attempt in range(attempts):
                await self._renew_locked()
                current = await self.foundation.state.get_run(self.run_id)
                if current is None:
                    self._mark_lost()
                    raise _LeaseLostError("Durable run disappeared while leased")
                try:
                    changes = dict(changes_builder(current))
                    return await self.foundation.state.update_run_fenced(
                        self.run_id,
                        current.version,
                        self.worker_id,
                        self.fencing_token,
                        changes,
                    )
                except OptimisticLockError:
                    if attempt + 1 == attempts:
                        raise
                    await asyncio.sleep(0)
                except LeaseLostError as exc:
                    self._mark_lost()
                    raise _LeaseLostError(
                        "SmartGen worker lease was lost",
                    ) from exc
            raise AssertionError("unreachable")

    async def _renew_locked(self) -> None:
        if self.lost.is_set():
            raise _LeaseLostError("SmartGen worker lease was lost")
        renewed = await self.foundation.state.renew_lease(
            self.run_id,
            self.worker_id,
            self.fencing_token,
            ttl_seconds=self.foundation.config.lease_ttl_seconds,
        )
        if renewed is None:
            self._mark_lost()
            raise _LeaseLostError("SmartGen worker lease was lost")

    def _mark_lost(self) -> None:
        self.lost.set()
        self.cancel_event.set()

    async def _heartbeat(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.heartbeat_interval_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    await self.ensure_current()
                    extended = await self.foundation.queue.extend_visibility(
                        self.queued_job,
                        visibility_timeout=self.visibility_timeout_seconds,
                    )
                    if extended is False:
                        self._mark_lost()
                        return
                    await self.task_protection.refresh()
                    record = await self.foundation.state.get_run(self.run_id)
                    if record is None:
                        self._mark_lost()
                        return
                    if record.status == RunStatus.CANCEL_REQUESTED:
                        self.cancel_event.set()
                except _LeaseLostError:
                    return
                except Exception:
                    logger.exception(
                        "Durable SmartGen lease heartbeat failed ref=%s",
                        _run_ref(self.run_id),
                    )
                    self._mark_lost()
                    return
        except asyncio.CancelledError:
            raise


def _bundle_workspace(
    workspace: str,
    excluded_paths: tuple[str, ...] = (),
) -> bytes:
    """Create a portable checkpoint snapshot without dependency caches."""

    root = Path(workspace).resolve()
    excluded = {
        Path(path).resolve()
        for path in excluded_paths
    }
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for current_root, directory_names, file_names in os.walk(root):
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in _CHECKPOINT_EXCLUDED_DIRECTORIES
                and not (Path(current_root) / name).is_symlink()
            )
            for file_name in sorted(file_names):
                source = Path(current_root) / file_name
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or source.resolve() in excluded
                ):
                    continue
                try:
                    relative = source.resolve().relative_to(root).as_posix()
                    archive.write(source, relative)
                except (FileNotFoundError, OSError, ValueError):
                    continue
    return output.getvalue()


def _extract_workspace_bundle(
    bundle: bytes,
    destination: str,
    *,
    max_uncompressed_bytes: int,
) -> None:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        members = archive.infolist()
        if len(members) > 10_000:
            raise ValueError("Workspace bundle contains too many files")
        total_size = sum(member.file_size for member in members)
        if total_size > max_uncompressed_bytes:
            raise ValueError("Workspace bundle exceeds the extraction limit")
        for member in members:
            member_path = Path(member.filename)
            if (
                member_path.is_absolute()
                or member_path.drive
                or ".." in member_path.parts
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Workspace bundle contains an unsafe path")
            target = (root / member_path).resolve()
            if target != root and root not in target.parents:
                raise ValueError("Workspace bundle escapes its destination")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


class _ApprovalBroker:
    """Bridge synchronous orchestrator approvals into durable async state."""

    def __init__(
        self,
        guard: _LeaseGuard,
        *,
        event_loop: asyncio.AbstractEventLoop,
        timeout_seconds: int,
    ) -> None:
        self.guard = guard
        self.event_loop = event_loop
        self.timeout_seconds = timeout_seconds

    def request(self, turn: int, tool: str, arguments: dict) -> bool:
        approval_id = uuid.uuid4().hex
        future = asyncio.run_coroutine_threadsafe(
            self._wait_for_decision(
                approval_id=approval_id,
                turn=turn,
                tool=tool,
                arguments=arguments,
            ),
            self.event_loop,
        )
        try:
            return bool(future.result(timeout=self.timeout_seconds + 30))
        except Exception:
            future.cancel()
            logger.warning(
                "Durable SmartGen approval wait failed ref=%s tool=%s",
                _run_ref(self.guard.run_id),
                tool,
                exc_info=True,
            )
            return False

    async def _wait_for_decision(
        self,
        *,
        approval_id: str,
        turn: int,
        tool: str,
        arguments: dict,
    ) -> bool:
        created_at = time.time()
        safe_arguments = json.loads(json.dumps(arguments, default=str))

        def add_pending(current: RunRecord) -> Mapping[str, Any]:
            metadata = dict(current.metadata)
            raw_approvals = metadata.get("approvals")
            approvals = (
                dict(raw_approvals)
                if isinstance(raw_approvals, Mapping)
                else {}
            )
            approvals[approval_id] = {
                "status": "pending",
                "turn": int(turn),
                "tool": tool,
                "arguments": safe_arguments,
                "created_at": created_at,
            }
            metadata["approvals"] = approvals
            return {"metadata": metadata}

        await self.guard.update_run(add_pending)
        await self.guard.append_event(
            "approval_required",
            {
                "event": "approval_required",
                "approvalId": approval_id,
                "turn": int(turn),
                "tool": tool,
                "summary": f"Allow {tool} to run?",
                "arguments": safe_arguments,
            },
        )

        deadline = self.event_loop.time() + self.timeout_seconds
        while self.event_loop.time() < deadline:
            await self.guard.ensure_current()
            current = await self.guard.foundation.state.get_run(self.guard.run_id)
            if current is None:
                raise _LeaseLostError("Durable run disappeared during approval")
            if current.status == RunStatus.CANCEL_REQUESTED:
                self.guard.cancel_event.set()
                return False
            if current.terminal or current.metadata.get("worker_finalized") is True:
                return False
            raw_approvals = current.metadata.get("approvals")
            if isinstance(raw_approvals, Mapping):
                approval = raw_approvals.get(approval_id)
                if isinstance(approval, Mapping):
                    status = approval.get("status")
                    if status == "approved":
                        return True
                    if status in {"rejected", "timed_out"}:
                        return False
            await asyncio.sleep(0.25)

        await self._mark_timed_out(approval_id)
        return False

    async def _mark_timed_out(self, approval_id: str) -> None:
        timed_out_at = time.time()

        def time_out(current: RunRecord) -> Mapping[str, Any]:
            metadata = dict(current.metadata)
            raw_approvals = metadata.get("approvals")
            if not isinstance(raw_approvals, Mapping):
                return {"metadata": metadata}
            approvals = dict(raw_approvals)
            raw_approval = approvals.get(approval_id)
            if not isinstance(raw_approval, Mapping):
                return {"metadata": metadata}
            approval = dict(raw_approval)
            if approval.get("status") != "pending":
                return {"metadata": metadata}
            approval.update({"status": "timed_out", "resolved_at": timed_out_at})
            approvals[approval_id] = approval
            metadata["approvals"] = approvals
            return {"metadata": metadata}

        updated = await self.guard.update_run(time_out)
        raw_approvals = updated.metadata.get("approvals")
        approval = (
            raw_approvals.get(approval_id)
            if isinstance(raw_approvals, Mapping)
            else None
        )
        if (
            isinstance(approval, Mapping)
            and approval.get("status") == "timed_out"
            and approval.get("resolved_at") == timed_out_at
        ):
            await self.guard.append_event(
                "approval_resolved",
                {
                    "event": "approval_resolved",
                    "approvalId": approval_id,
                    "decision": "timed_out",
                },
            )


class _JobProcessor:
    def __init__(
        self,
        foundation: DurableStateFoundation,
        job: QueuedJob,
        *,
        envelope: SecretEnvelope,
        runner_factory: Callable[..., _Runner],
        worker_id: str,
        visibility_timeout_seconds: int,
        heartbeat_interval_seconds: Optional[float],
        approval_timeout_seconds: int,
        max_attempts: int,
        retry_delay_seconds: int,
        task_protection: _TaskProtection,
    ) -> None:
        self.foundation = foundation
        self.job = job
        self.envelope = envelope
        self.runner_factory = runner_factory
        self.worker_id = worker_id
        self.visibility_timeout_seconds = visibility_timeout_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.approval_timeout_seconds = approval_timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.task_protection = task_protection
        self.cancel_event = asyncio.Event()
        self.guard: Optional[_LeaseGuard] = None
        self.runner: Optional[_Runner] = None
        self.reserved_cancel_event: Optional[asyncio.Event] = None
        self.execution_started = False
        self._checkpoint_signature: Optional[tuple[int, int]] = None
        self._hydrated_base: Optional[tuple[str, str]] = None
        self._hydrated_resume_dir: Optional[str] = None

    @property
    def run_id(self) -> str:
        return self.job.message.run_id

    @property
    def owner_id(self) -> str:
        return self.job.message.owner_id

    async def process(self) -> _QueueDecision:
        try:
            record = await self.foundation.state.get_run(self.run_id)
            if record is None:
                return _QueueDecision(
                    acknowledge=True,
                    reason="queued run no longer exists",
                )
            if record.owner_id != self.owner_id:
                logger.warning(
                    "Discarding owner-mismatched SmartGen job ref=%s",
                    _run_ref(self.run_id),
                )
                return _QueueDecision(
                    acknowledge=True,
                    reason="job owner does not match run owner",
                )

            lease = await self.foundation.state.acquire_lease(
                self.run_id,
                self.worker_id,
                ttl_seconds=self.foundation.config.lease_ttl_seconds,
            )
            if lease is None:
                return self._retry("run lease is already held")

            self.guard = _LeaseGuard(
                self.foundation,
                run_id=self.run_id,
                worker_id=self.worker_id,
                fencing_token=lease.fencing_token,
                cancel_event=self.cancel_event,
                heartbeat_interval_seconds=self.heartbeat_interval_seconds,
                queued_job=self.job,
                visibility_timeout_seconds=self.visibility_timeout_seconds,
                task_protection=self.task_protection,
            )
            self.guard.start()
            record = await self._record_attempt(lease)

            if record.metadata.get("worker_finalized") is True:
                await self._recover_finalized(record)
                return _QueueDecision(True, record.status, "finalized run recovered")
            if record.terminal:
                await self._release_concurrency(record)
                return _QueueDecision(True, record.status, "terminal run recovered")
            if record.status == RunStatus.CANCEL_REQUESTED:
                finalized = await self._finalize(
                    RunStatus.CANCELLED,
                    {
                        "event": "error",
                        "code": "CANCELLED",
                        "message": "Smart generation cancelled by user request",
                    },
                    error_code="CANCELLED",
                    error_message="Smart generation cancelled by user request",
                )
                return _QueueDecision(True, finalized.status, "cancelled before start")
            raw_checkpoint = record.metadata.get("checkpoint")
            resumable_checkpoint = bool(
                isinstance(raw_checkpoint, Mapping)
                and raw_checkpoint.get("resumable") is True
            )
            resume_from_checkpoint = bool(
                record.metadata.get("execution_started") is True
                and record.checkpoint_key
                and resumable_checkpoint
                and record.status in {RunStatus.RUNNING, RunStatus.INCOMPLETE}
                and self.job.receive_count <= self.max_attempts
            )
            if (
                record.metadata.get("execution_started") is True
                and not resume_from_checkpoint
            ):
                finalized = await self._reconcile_interrupted(record)
                return _QueueDecision(True, finalized.status, "interrupted run reconciled")
            if record.status not in {
                RunStatus.QUEUED,
                RunStatus.RUNNING,
                RunStatus.INCOMPLETE,
            }:
                finalized = await self._finalize(
                    RunStatus.INCOMPLETE,
                    {
                        "event": "error",
                        "code": "INTERNAL",
                        "message": "Generation stopped before durable recovery completed.",
                    },
                    error_code="WORKER_INTERRUPTED",
                    error_message="Generation stopped before durable recovery completed.",
                )
                return _QueueDecision(True, finalized.status, "non-runnable state reconciled")

            request = self._reconstruct_request(record)
            if resume_from_checkpoint:
                await self._hydrate_resume_workspace(record)
            elif request.mode == "modify" and request.base_run_id:
                await self._hydrate_modify_base(request.base_run_id)
            await self._mark_running()
            await self._build_runner(request, resume=resume_from_checkpoint)
            await self.task_protection.enable()
            await self._mark_execution_started()
            self.execution_started = True
            return await self._consume_runner()
        except _CancellationRequested:
            if self.guard is None:
                return self._retry("cancellation raced with worker startup")
            finalized = await self._finalize(
                RunStatus.CANCELLED,
                {
                    "event": "error",
                    "code": "CANCELLED",
                    "message": "Smart generation cancelled by user request",
                },
                error_code="CANCELLED",
                error_message="Smart generation cancelled by user request",
            )
            return _QueueDecision(True, finalized.status, "cancelled during startup")
        except _PermanentJobError as exc:
            if self.guard is None:
                return _QueueDecision(True, RunStatus.FAILED, exc.code)
            finalized = await self._finalize(
                RunStatus.FAILED,
                {
                    "event": "error",
                    "code": "QUOTA" if exc.code == "STORAGE_QUOTA" else "INTERNAL",
                    "message": exc.safe_message,
                },
                error_code=exc.code,
                error_message=exc.safe_message,
            )
            return _QueueDecision(True, finalized.status, exc.code)
        except _LeaseLostError:
            return self._retry("worker lease was lost")
        except _TaskProtectionError:
            logger.error(
                "Durable SmartGen task protection unavailable ref=%s",
                _run_ref(self.run_id),
            )
            return self._retry("ECS task protection is unavailable")
        except Exception:
            logger.exception(
                "Durable SmartGen worker failed ref=%s",
                _run_ref(self.run_id),
            )
            return await self._handle_unexpected_failure()

    def _retry(self, reason: str) -> _QueueDecision:
        return _QueueDecision(
            acknowledge=False,
            reason=reason,
            retry_delay_seconds=self.retry_delay_seconds,
        )

    async def _record_attempt(self, lease: Any) -> RunRecord:
        assert self.guard is not None

        def changes(current: RunRecord) -> Mapping[str, Any]:
            metadata = dict(current.metadata)
            metadata.setdefault("worker_started_at", time.time())
            metadata.update({
                "worker_id": self.worker_id,
                "retry_count": max(0, self.job.receive_count - 1),
                "worker_last_attempt_at": time.time(),
                "lease": _serialise_ref(lease),
            })
            return {"metadata": metadata}

        return await self.guard.update_run(changes)

    def _reconstruct_request(self, record: RunRecord) -> SmartGenerateRequest:
        raw_request = record.metadata.get("request")
        raw_envelope = record.metadata.get("api_key_envelope")
        if not isinstance(raw_request, Mapping) or not isinstance(raw_envelope, Mapping):
            raise _PermanentJobError(
                "INVALID_JOB",
                "Generation request metadata is incomplete.",
            )
        try:
            encrypted = EncryptedSecret.from_dict(raw_envelope)
            plaintext = self.envelope.decrypt(
                encrypted,
                run_id=self.run_id,
                owner_id=self.owner_id,
            )
            payload = dict(raw_request)
            payload["api_key"] = plaintext
            request = SmartGenerateRequest.model_validate(payload)
        except (PydanticValidationError, SecretEnvelopeError, TypeError, ValueError) as exc:
            raise _PermanentJobError(
                "INVALID_JOB",
                "Generation request metadata could not be verified.",
            ) from exc
        if canonical_request_hash(request, self.owner_id) != record.request_hash:
            raise _PermanentJobError(
                "REQUEST_HASH_MISMATCH",
                "Generation request integrity verification failed.",
            )
        return request

    @staticmethod
    def _artifact_from_record(record: RunRecord) -> Optional[ArtifactRef]:
        raw_artifact = record.metadata.get("artifact")
        if not isinstance(raw_artifact, Mapping):
            return None
        try:
            artifact = ArtifactRef(
                storage_key=str(raw_artifact["storage_key"]),
                file_name=str(raw_artifact["file_name"]),
                size_bytes=int(raw_artifact["size_bytes"]),
                sha256=str(raw_artifact["sha256"]),
                content_type=str(raw_artifact["content_type"]),
                created_at=float(raw_artifact["created_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if artifact.storage_key != record.artifact_key:
            return None
        return artifact

    async def _hydrate_resume_workspace(self, record: RunRecord) -> None:
        assert self.guard is not None
        await self.guard.ensure_current()
        bundle = await self.foundation.checkpoints.get_checkpoint(
            self.owner_id,
            self.run_id,
        )
        if bundle is None:
            raise _PermanentJobError(
                "CHECKPOINT_UNAVAILABLE",
                "The durable recovery checkpoint is no longer available.",
            )
        destination = tempfile.mkdtemp(
            prefix=f"{LLM_TEMP_DIR_PREFIX}{self.run_id}_",
        )
        try:
            await asyncio.to_thread(
                _extract_workspace_bundle,
                bundle,
                destination,
                max_uncompressed_bytes=(
                    self.foundation.config.max_storage_bytes_per_owner
                ),
            )
            if not (Path(destination) / CHECKPOINT_FILENAME).is_file():
                raise ValueError("Workspace bundle has no resumable checkpoint")
            await self.guard.ensure_current()
        except Exception as exc:
            await asyncio.to_thread(shutil.rmtree, destination, True)
            if isinstance(exc, _LeaseLostError):
                raise
            raise _PermanentJobError(
                "CHECKPOINT_UNAVAILABLE",
                "The durable recovery checkpoint could not be restored.",
            ) from exc
        self._hydrated_resume_dir = destination

    async def _hydrate_modify_base(self, base_run_id: str) -> None:
        assert self.guard is not None
        existing = await SMART_RUN_REGISTRY.get(base_run_id)
        if existing is not None:
            if (
                existing.owner_id == self.owner_id
                and existing.temp_dir
                and Path(existing.temp_dir).is_dir()
            ):
                return
            raise _PermanentJobError(
                "BASE_ARTIFACT_UNAVAILABLE",
                "The selected base generation is not available for modification.",
            )

        base_record = await self.foundation.state.get_owned_run(
            self.owner_id,
            base_run_id,
        )
        if base_record is None:
            raise _PermanentJobError(
                "BASE_ARTIFACT_UNAVAILABLE",
                "The selected base generation is not available for modification.",
            )
        destination = tempfile.mkdtemp(prefix=f"besser_durable_base_{base_run_id}_")
        try:
            hydrated = False
            if base_record.checkpoint_key:
                bundle = await self.foundation.checkpoints.get_checkpoint(
                    self.owner_id,
                    base_run_id,
                )
                if bundle is not None:
                    await asyncio.to_thread(
                        _extract_workspace_bundle,
                        bundle,
                        destination,
                        max_uncompressed_bytes=(
                            self.foundation.config.max_storage_bytes_per_owner
                        ),
                    )
                    hydrated = True
            artifact = self._artifact_from_record(base_record)
            artifact_path: Optional[Path] = None
            if not hydrated and artifact is not None:
                download_path = Path(destination) / ".durable-artifact"
                await self.foundation.artifacts.download_artifact(
                    artifact,
                    str(download_path),
                )
                if zipfile.is_zipfile(download_path):
                    bundle = await asyncio.to_thread(download_path.read_bytes)
                    await asyncio.to_thread(
                        _extract_workspace_bundle,
                        bundle,
                        destination,
                        max_uncompressed_bytes=(
                            self.foundation.config.max_storage_bytes_per_owner
                        ),
                    )
                    download_path.unlink(missing_ok=True)
                else:
                    artifact_path = Path(destination) / Path(artifact.file_name).name
                    download_path.replace(artifact_path)
                hydrated = True
            if not hydrated:
                raise ValueError("Base run has no durable workspace or artifact")
            await self.guard.ensure_current()
            entry = SmartRunEntry(
                file_path=str(artifact_path or destination),
                file_name=(artifact.file_name if artifact is not None else "workspace.zip"),
                is_zip=artifact is None or artifact.content_type == "application/zip",
                temp_dir=destination,
                created_at=time.time(),
                owner_id=self.owner_id,
            )
            await SMART_RUN_REGISTRY.put(base_run_id, entry)
            self._hydrated_base = (base_run_id, destination)
        except Exception as exc:
            await asyncio.to_thread(shutil.rmtree, destination, True)
            if isinstance(exc, (_LeaseLostError, _PermanentJobError)):
                raise
            raise _PermanentJobError(
                "BASE_ARTIFACT_UNAVAILABLE",
                "The selected base generation could not be restored.",
            ) from exc

    async def _build_runner(
        self,
        request: SmartGenerateRequest,
        *,
        resume: bool,
    ) -> None:
        self.reserved_cancel_event = await reserve_active_run(
            self.run_id, self.owner_id,
        )
        if self.reserved_cancel_event is None:
            raise _LeaseLostError("run is already active in this process")
        self.cancel_event = self.reserved_cancel_event
        assert self.guard is not None
        self.guard.cancel_event = self.cancel_event
        approval_broker = _ApprovalBroker(
            self.guard,
            event_loop=asyncio.get_running_loop(),
            timeout_seconds=self.approval_timeout_seconds,
        )
        runner_arguments: dict[str, Any] = {
            "mode": request.mode,
            "owner_id": self.owner_id,
            "reserved_cancel_event": self.cancel_event,
            "allow_shell_tools": _shell_tools_enabled(self.foundation),
            "request_tool_approval": approval_broker.request,
        }
        if resume:
            runner_arguments["resume_run_id"] = self.run_id
        else:
            runner_arguments.update({
                "base_run_id": request.base_run_id,
                "run_id": self.run_id,
            })
        self.runner = self.runner_factory(request, **runner_arguments)

    async def _mark_running(self) -> RunRecord:
        assert self.guard is not None

        def changes(current: RunRecord) -> Mapping[str, Any]:
            if current.status == RunStatus.CANCEL_REQUESTED:
                raise _CancellationRequested
            if current.status not in {
                RunStatus.QUEUED,
                RunStatus.RUNNING,
                RunStatus.INCOMPLETE,
            }:
                raise _PermanentJobError(
                    "INVALID_RUN_STATE",
                    "Generation run is no longer runnable.",
                )
            metadata = dict(current.metadata)
            metadata["execution_started"] = False
            return {"status": RunStatus.RUNNING, "metadata": metadata}

        return await self.guard.update_run(changes)

    async def _mark_execution_started(self) -> RunRecord:
        assert self.guard is not None

        def changes(current: RunRecord) -> Mapping[str, Any]:
            if current.status == RunStatus.CANCEL_REQUESTED:
                self.cancel_event.set()
                raise _CancellationRequested
            metadata = dict(current.metadata)
            metadata.update({
                "execution_started": True,
                "execution_started_at": time.time(),
            })
            return {"metadata": metadata}

        return await self.guard.update_run(changes)

    async def _consume_runner(self) -> _QueueDecision:
        assert self.guard is not None
        assert self.runner is not None
        terminal_payload: Optional[dict[str, Any]] = None
        deferred_warnings: list[dict[str, Any]] = []
        stream = self.runner.generate_and_stream()
        try:
            async for frame in stream:
                if self.guard.lost.is_set():
                    raise _LeaseLostError("SmartGen worker lease was lost")
                payload = _decode_sse_frame(frame)
                await self._sync_checkpoint(strict=False)
                event_type = payload["event"]
                code = payload.get("code")
                if event_type == "done" or (
                    event_type == "error" and code in _TERMINAL_ERROR_CODES
                ):
                    for warning in deferred_warnings:
                        await self.guard.append_event(warning["event"], warning)
                    deferred_warnings.clear()
                    terminal_payload = payload
                    break
                if event_type == "error":
                    deferred_warnings.append(payload)
                    continue
                for warning in deferred_warnings:
                    await self.guard.append_event(warning["event"], warning)
                deferred_warnings.clear()
                await self.guard.append_event(event_type, payload)
                if event_type == "cost":
                    await self._record_cost(payload)
        finally:
            close_stream = getattr(stream, "aclose", None)
            if callable(close_stream):
                with contextlib.suppress(Exception):
                    await close_stream()

        if terminal_payload is None and deferred_warnings:
            for warning in deferred_warnings[:-1]:
                await self.guard.append_event(warning["event"], warning)
            terminal_payload = deferred_warnings[-1]

        await self._sync_checkpoint(strict=True, force=True)
        if terminal_payload is None:
            if self.cancel_event.is_set():
                terminal_payload = {
                    "event": "error",
                    "code": "CANCELLED",
                    "message": "Smart generation cancelled by user request",
                }
            else:
                terminal_payload = {
                    "event": "error",
                    "code": "INTERNAL",
                    "message": "Generation worker stopped without a terminal result.",
                }

        if terminal_payload["event"] == "done":
            await self._persist_final_workspace()
            await self._persist_artifact()
            terminal_payload = dict(terminal_payload)
            terminal_payload["downloadUrl"] = (
                f"/besser_api/smart-gen/runs/{self.run_id}/artifact"
            )
            incomplete = terminal_payload.get("incomplete") is True
            status = RunStatus.INCOMPLETE if incomplete else RunStatus.SUCCEEDED
            finalized = await self._finalize(
                status,
                terminal_payload,
                error_code="INCOMPLETE" if incomplete else None,
                error_message=(
                    terminal_payload.get("incompleteReason")
                    if incomplete
                    else None
                ),
            )
            return _QueueDecision(True, finalized.status, "runner completed")

        code = str(terminal_payload.get("code") or "INTERNAL")
        cancelled = code == "CANCELLED" or (
            self.cancel_event.is_set() and code not in {"TIMEOUT", "COST_CAP"}
        )
        status = RunStatus.CANCELLED if cancelled else RunStatus.FAILED
        message = str(
            terminal_payload.get("message")
            or "Generation failed before producing an artifact."
        )[:500]
        finalized = await self._finalize(
            status,
            terminal_payload,
            error_code=code,
            error_message=message,
        )
        return _QueueDecision(True, finalized.status, code)

    async def _record_cost(self, payload: Mapping[str, Any]) -> None:
        assert self.guard is not None
        try:
            cost = float(payload.get("usd", 0.0))
        except (TypeError, ValueError):
            return
        if not 0.0 <= cost < 1_000_000.0:
            return
        await self.guard.update_run(
            lambda current: {
                "estimated_cost_usd": max(current.estimated_cost_usd, cost),
            },
        )

    async def _sync_checkpoint(
        self,
        *,
        strict: bool,
        force: bool = False,
    ) -> None:
        assert self.guard is not None
        if self.runner is None or not self.runner.temp_dir:
            return
        checkpoint_path = Path(self.runner.temp_dir) / CHECKPOINT_FILENAME
        try:
            stat = checkpoint_path.stat()
        except FileNotFoundError:
            return
        signature = (stat.st_mtime_ns, stat.st_size)
        if not force and signature == self._checkpoint_signature:
            return
        try:
            await self.guard.ensure_current()
            bundle = await asyncio.to_thread(
                _bundle_workspace,
                self.runner.temp_dir,
            )
            reference = await self.foundation.checkpoints.put_checkpoint(
                self.owner_id,
                self.run_id,
                bundle,
            )
            await self.guard.ensure_current()

            def persist(current: RunRecord) -> Mapping[str, Any]:
                metadata = dict(current.metadata)
                metadata["checkpoint"] = {
                    **_serialise_ref(reference),
                    "format": "workspace-zip-v1",
                    "checkpoint_file": CHECKPOINT_FILENAME,
                    "purpose": "recovery",
                    "resumable": True,
                }
                return {
                    "checkpoint_key": reference.storage_key,
                    "metadata": metadata,
                }

            await self.guard.update_run(persist)
            self._checkpoint_signature = signature
        except _LeaseLostError:
            raise
        except Exception:
            if strict:
                raise
            logger.warning(
                "Durable checkpoint persistence failed ref=%s",
                _run_ref(self.run_id),
                exc_info=True,
            )

    async def _persist_final_workspace(self) -> CheckpointRef:
        assert self.guard is not None
        if self.runner is None or not self.runner.temp_dir:
            raise _PermanentJobError(
                "WORKSPACE_MISSING",
                "Generation completed without a durable workspace.",
            )
        entry = await SMART_RUN_REGISTRY.get(self.run_id)
        excluded_paths = (
            (entry.file_path,)
            if entry is not None and entry.owner_id == self.owner_id
            else ()
        )
        await self.guard.ensure_current()
        bundle = await asyncio.to_thread(
            _bundle_workspace,
            self.runner.temp_dir,
            excluded_paths,
        )
        reference = await self.foundation.checkpoints.put_checkpoint(
            self.owner_id,
            self.run_id,
            bundle,
        )
        await self.guard.ensure_current()
        resumable = (
            Path(self.runner.temp_dir) / CHECKPOINT_FILENAME
        ).is_file()

        def persist(current: RunRecord) -> Mapping[str, Any]:
            metadata = dict(current.metadata)
            metadata["checkpoint"] = {
                **_serialise_ref(reference),
                "format": "workspace-zip-v1",
                "checkpoint_file": CHECKPOINT_FILENAME,
                "purpose": "final-workspace",
                "resumable": resumable,
            }
            return {
                "checkpoint_key": reference.storage_key,
                "metadata": metadata,
            }

        await self.guard.update_run(persist)
        return reference

    async def _persist_artifact(self) -> ArtifactRef:
        assert self.guard is not None
        entry = await SMART_RUN_REGISTRY.get(self.run_id)
        if entry is None or entry.owner_id != self.owner_id:
            raise _PermanentJobError(
                "ARTIFACT_MISSING",
                "Generation completed without a durable artifact.",
            )
        source = Path(entry.file_path)
        try:
            size_bytes = source.stat().st_size
        except (FileNotFoundError, OSError) as exc:
            raise _PermanentJobError(
                "ARTIFACT_MISSING",
                "Generation completed without a durable artifact.",
            ) from exc
        if not source.is_file():
            raise _PermanentJobError(
                "ARTIFACT_MISSING",
                "Generation completed without a durable artifact.",
            )

        reservation_id = f"storage:{self.run_id}"
        decision = await self.foundation.state.reserve_quota(
            self.owner_id,
            "storage-bytes",
            reservation_id,
            amount=max(1, size_bytes),
            limit=self.foundation.config.max_storage_bytes_per_owner,
            ttl_seconds=max(
                86400,
                self.foundation.config.idempotency_ttl_seconds,
            ),
        )
        if not decision.allowed:
            raise _PermanentJobError(
                "STORAGE_QUOTA",
                "Generated output exceeds the available storage quota.",
            )
        reference: Optional[ArtifactRef] = None
        try:
            await self.guard.ensure_current()
            reference = await self.foundation.artifacts.put_artifact(
                self.owner_id,
                self.run_id,
                str(source),
                file_name=entry.file_name,
                content_type=_content_type(entry),
            )
            await self.guard.ensure_current()

            def persist(current: RunRecord) -> Mapping[str, Any]:
                metadata = dict(current.metadata)
                metadata["artifact"] = _serialise_ref(reference)
                metadata["storage_reservation"] = {
                    "reservation_id": reservation_id,
                    "amount": max(1, size_bytes),
                    "expires_at": decision.expires_at,
                }
                return {
                    "artifact_key": reference.storage_key,
                    "metadata": metadata,
                }

            await self.guard.update_run(persist)
        except _LeaseLostError:
            raise
        except Exception:
            if reference is not None:
                with contextlib.suppress(Exception):
                    await self.foundation.artifacts.delete_artifact(reference)
            with contextlib.suppress(Exception):
                await self.foundation.state.release_quota(reservation_id)
            raise
        return reference

    async def _finalize(
        self,
        status: RunStatus,
        terminal_payload: Mapping[str, Any],
        *,
        error_code: Optional[str],
        error_message: Optional[str],
    ) -> RunRecord:
        assert self.guard is not None
        payload = dict(terminal_payload)

        def changes(current: RunRecord) -> Mapping[str, Any]:
            if current.metadata.get("worker_finalized") is True:
                return {"metadata": dict(current.metadata)}
            metadata = dict(current.metadata)
            metadata.pop("api_key_envelope", None)
            metadata.update({
                "execution_finished_at": time.time(),
                "worker_finalized": True,
                "terminal_event": payload,
            })
            return {
                "status": status,
                "error_code": error_code,
                "error_message": error_message,
                "metadata": metadata,
            }

        record = await self.guard.update_run(changes)
        terminal_sequence = await self._ensure_terminal_event(payload)
        record = await self._record_terminal_sequence(terminal_sequence)
        await self._release_concurrency(record)
        return record

    async def _ensure_terminal_event(self, payload: Mapping[str, Any]) -> int:
        assert self.guard is not None
        expected_payload = dict(payload)
        expected_event_type = str(payload.get("event") or "error")
        cursor = ReplayCursor(self.run_id)
        while True:
            page = await self.foundation.state.read_events(
                self.run_id,
                cursor=cursor,
                limit=self.foundation.config.event_page_size,
            )
            for event in page.events:
                if (
                    event.event_type == expected_event_type
                    and dict(event.payload) == expected_payload
                ):
                    return event.sequence
            if not page.has_more:
                break
            cursor = page.cursor
        event = await self.guard.append_event(expected_event_type, expected_payload)
        return int(event.sequence)

    async def _record_terminal_sequence(self, sequence: int) -> RunRecord:
        assert self.guard is not None

        def persist(current: RunRecord) -> Mapping[str, Any]:
            metadata = dict(current.metadata)
            metadata["terminal_event_sequence"] = sequence
            return {"metadata": metadata}

        return await self.guard.update_run(persist)

    async def _recover_finalized(self, record: RunRecord) -> None:
        raw_payload = record.metadata.get("terminal_event")
        if isinstance(raw_payload, Mapping):
            payload = dict(raw_payload)
        else:
            payload = {
                "event": "error",
                "code": record.error_code or "INTERNAL",
                "message": record.error_message or "Generation finished.",
            }
        sequence = await self._ensure_terminal_event(payload)
        await self._record_terminal_sequence(sequence)
        await self._release_concurrency(record)

    async def _release_concurrency(self, record: RunRecord) -> None:
        reservation_id = record.metadata.get("concurrency_reservation")
        if isinstance(reservation_id, str):
            await self.foundation.state.release_quota(reservation_id)

    async def _reconcile_interrupted(self, record: RunRecord) -> RunRecord:
        if record.status == RunStatus.CANCEL_REQUESTED:
            return await self._finalize(
                RunStatus.CANCELLED,
                {
                    "event": "error",
                    "code": "CANCELLED",
                    "message": "Smart generation cancelled by user request",
                },
                error_code="CANCELLED",
                error_message="Smart generation cancelled by user request",
            )
        resumable = bool(record.checkpoint_key)
        artifact_available = bool(record.artifact_key)
        status = (
            RunStatus.INCOMPLETE
            if artifact_available
            and record.status in {RunStatus.RUNNING, RunStatus.INCOMPLETE}
            else RunStatus.FAILED
        )
        code = "WORKER_INTERRUPTED" if resumable else "WORKER_CRASHED"
        message = (
            "Generation stopped, but a durable recovery checkpoint is available."
            if resumable
            else "Generation worker stopped before producing recoverable output."
        )
        return await self._finalize(
            status,
            {
                "event": "error",
                "code": "INCOMPLETE" if resumable else "INTERNAL",
                "message": message,
            },
            error_code=code,
            error_message=message,
        )

    async def _handle_unexpected_failure(self) -> _QueueDecision:
        if self.guard is None or self.guard.lost.is_set():
            return self._retry("transient worker failure")
        try:
            current = await self.foundation.state.get_run(self.run_id)
            if current is None:
                return _QueueDecision(True, reason="run disappeared")
            if current.metadata.get("worker_finalized") is True:
                await self._recover_finalized(current)
                return _QueueDecision(True, current.status, "finalization recovered")
            if self.execution_started or current.metadata.get("execution_started") is True:
                await self._sync_checkpoint(strict=False, force=True)
                current = await self.foundation.state.get_run(self.run_id) or current
                raw_checkpoint = current.metadata.get("checkpoint")
                if (
                    current.checkpoint_key
                    and isinstance(raw_checkpoint, Mapping)
                    and raw_checkpoint.get("resumable") is True
                    and self.job.receive_count < self.max_attempts
                ):
                    return self._retry("resumable execution will continue in a fresh worker")
                finalized = await self._reconcile_interrupted(current)
                return _QueueDecision(
                    True,
                    finalized.status,
                    "paid execution reconciled without rerun",
                )
            if self.job.receive_count >= self.max_attempts:
                finalized = await self._finalize(
                    RunStatus.FAILED,
                    {
                        "event": "error",
                        "code": "INTERNAL",
                        "message": "Generation worker could not start after bounded retries.",
                    },
                    error_code="WORKER_RETRY_EXHAUSTED",
                    error_message=(
                        "Generation worker could not start after bounded retries."
                    ),
                )
                return _QueueDecision(True, finalized.status, "retry limit reached")
        except _LeaseLostError:
            return self._retry("worker lease was lost during reconciliation")
        except Exception:
            logger.exception(
                "Durable SmartGen reconciliation failed ref=%s",
                _run_ref(self.run_id),
            )
        return self._retry("transient worker failure")

    async def close(self) -> None:
        if self.guard is not None:
            await self.guard.stop()
        try:
            await self.task_protection.disable()
        except _TaskProtectionError:
            logger.exception(
                "Durable SmartGen task protection cleanup failed ref=%s",
                _run_ref(self.run_id),
            )
        if self.reserved_cancel_event is not None:
            with contextlib.suppress(Exception):
                await release_active_run(self.run_id, self.reserved_cancel_event)
        if self.guard is not None:
            with contextlib.suppress(Exception):
                await self.foundation.state.release_lease(
                    self.run_id,
                    self.worker_id,
                    self.guard.fencing_token,
                )

        cleanup_directories: set[str] = set()
        if self._hydrated_base is not None:
            base_run_id, base_directory = self._hydrated_base
            base_entry = await SMART_RUN_REGISTRY.get(base_run_id)
            if base_entry is not None and base_entry.temp_dir == base_directory:
                await SMART_RUN_REGISTRY.pop(base_run_id)
            cleanup_directories.add(base_directory)
        if self._hydrated_resume_dir is not None:
            cleanup_directories.add(self._hydrated_resume_dir)
        entry = await SMART_RUN_REGISTRY.get(self.run_id)
        if entry is not None and entry.owner_id == self.owner_id:
            popped = await SMART_RUN_REGISTRY.pop(self.run_id)
            if popped is not None:
                cleanup_directories.add(popped.temp_dir)
        if self.runner is not None and self.runner.temp_dir:
            cleanup_directories.add(self.runner.temp_dir)
        for directory in cleanup_directories:
            if directory:
                await asyncio.to_thread(shutil.rmtree, directory, True)


async def process_one_job(
    foundation: DurableStateFoundation,
    *,
    envelope: Optional[SecretEnvelope] = None,
    runner_factory: Callable[..., _Runner] = SmartGenerationRunner,
    worker_id: Optional[str] = None,
    wait_seconds: int = 10,
    visibility_timeout_seconds: int = _DEFAULT_VISIBILITY_TIMEOUT_SECONDS,
    heartbeat_interval_seconds: Optional[float] = None,
    approval_timeout_seconds: int = _DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: int = _DEFAULT_RETRY_DELAY_SECONDS,
    task_protection: Optional[_TaskProtection] = None,
) -> WorkerResult:
    """Receive and safely process at most one durable SmartGen job."""

    if not 0 <= wait_seconds <= 20:
        raise ValueError("wait_seconds must be between 0 and 20")
    if not 1 <= visibility_timeout_seconds <= 43200:
        raise ValueError("visibility_timeout_seconds must be between 1 and 43200")
    if approval_timeout_seconds <= 0 or max_attempts <= 0:
        raise ValueError("approval timeout and max attempts must be positive")
    if not 0 <= retry_delay_seconds <= 900:
        raise ValueError("retry_delay_seconds must be between 0 and 900")

    jobs = await foundation.queue.receive(
        max_messages=1,
        wait_seconds=wait_seconds,
        visibility_timeout=visibility_timeout_seconds,
    )
    if not jobs:
        return WorkerResult(WorkerDisposition.NO_JOB)
    job = jobs[0]
    processor = _JobProcessor(
        foundation,
        job,
        envelope=envelope or build_secret_envelope(foundation.config.mode.value),
        runner_factory=runner_factory,
        worker_id=worker_id or f"smartgen-worker:{uuid.uuid4().hex}",
        visibility_timeout_seconds=visibility_timeout_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        approval_timeout_seconds=approval_timeout_seconds,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        task_protection=(
            task_protection
            if task_protection is not None
            else _build_task_protection(foundation.config.mode.value)
        ),
    )
    try:
        decision = await processor.process()
    finally:
        await processor.close()

    if decision.acknowledge:
        try:
            await foundation.queue.acknowledge(job)
        except Exception:
            logger.exception(
                "Durable SmartGen acknowledgement failed ref=%s",
                _run_ref(job.message.run_id),
            )
            await foundation.queue.release(
                job,
                delay_seconds=retry_delay_seconds,
            )
            return WorkerResult(
                WorkerDisposition.RETRY_RELEASED,
                decision.status,
                "acknowledgement failed",
            )
        return WorkerResult(
            WorkerDisposition.ACKNOWLEDGED,
            decision.status,
            decision.reason,
        )

    await foundation.queue.release(
        job,
        delay_seconds=decision.retry_delay_seconds,
    )
    return WorkerResult(
        WorkerDisposition.RETRY_RELEASED,
        decision.status,
        decision.reason,
    )


async def run_worker_until_job(
    foundation: DurableStateFoundation,
    **worker_options: Any,
) -> WorkerResult:
    """Long-poll until one message is handled, then exit for isolation."""

    worker_options.setdefault("wait_seconds", 20)
    while True:
        result = await process_one_job(foundation, **worker_options)
        if result.disposition != WorkerDisposition.NO_JOB:
            return result


async def main_async() -> WorkerResult:
    foundation = build_durable_state()
    await foundation.initialize()
    try:
        return await run_worker_until_job(
            foundation,
            envelope=build_secret_envelope(foundation.config.mode.value),
            wait_seconds=20,
            visibility_timeout_seconds=_positive_env_int(
                "BESSER_SMARTGEN_WORKER_VISIBILITY_TIMEOUT_SECONDS",
                _DEFAULT_VISIBILITY_TIMEOUT_SECONDS,
            ),
            approval_timeout_seconds=_positive_env_int(
                "BESSER_SMARTGEN_APPROVAL_TIMEOUT_SECONDS",
                _DEFAULT_APPROVAL_TIMEOUT_SECONDS,
            ),
            max_attempts=_positive_env_int(
                "BESSER_SMARTGEN_WORKER_MAX_ATTEMPTS",
                _DEFAULT_MAX_ATTEMPTS,
            ),
            retry_delay_seconds=_positive_env_int(
                "BESSER_SMARTGEN_WORKER_RETRY_DELAY_SECONDS",
                _DEFAULT_RETRY_DELAY_SECONDS,
            ),
        )
    finally:
        await foundation.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
