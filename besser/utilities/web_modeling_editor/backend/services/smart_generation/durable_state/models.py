"""Typed records shared by durable SmartGen state and storage adapters."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional

from .errors import InvalidRunTransitionError

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:@/\-]{1,256}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class RunStatus(str, Enum):
    """Durable lifecycle states for a SmartGen job."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    INCOMPLETE = "incomplete"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_RUN_STATUSES = frozenset({
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
    RunStatus.EXPIRED,
})

_ALLOWED_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({
        RunStatus.RUNNING,
        RunStatus.CANCEL_REQUESTED,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
        RunStatus.EXPIRED,
    }),
    RunStatus.RUNNING: frozenset({
        RunStatus.CANCEL_REQUESTED,
        RunStatus.INCOMPLETE,
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }),
    RunStatus.CANCEL_REQUESTED: frozenset({
        RunStatus.INCOMPLETE,
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }),
    RunStatus.INCOMPLETE: frozenset({
        RunStatus.QUEUED,
        RunStatus.RUNNING,
        RunStatus.CANCEL_REQUESTED,
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.EXPIRED,
    }),
    RunStatus.SUCCEEDED: frozenset({RunStatus.EXPIRED}),
    RunStatus.FAILED: frozenset({RunStatus.EXPIRED}),
    RunStatus.CANCELLED: frozenset({RunStatus.EXPIRED}),
    RunStatus.EXPIRED: frozenset(),
}


def validate_identifier(value: str, name: str = "identifier") -> str:
    """Return a stripped identifier or raise a safe validation error."""

    candidate = (value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(candidate):
        raise ValueError(f"{name} has an invalid format")
    return candidate


def validate_run_id(value: str) -> str:
    candidate = (value or "").strip()
    if not _RUN_ID_RE.fullmatch(candidate):
        raise ValueError("run_id has an invalid format")
    return candidate


def validate_request_hash(value: str) -> str:
    candidate = (value or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", candidate):
        raise ValueError("request_hash must be a SHA-256 hex digest")
    return candidate


def _finite_non_negative(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Durable metadata for one SmartGen run."""

    run_id: str
    owner_id: str
    request_hash: str
    status: RunStatus = RunStatus.QUEUED
    mode: str = "generate"
    provider: str = ""
    model: Optional[str] = None
    max_cost_usd: float = 0.0
    max_runtime_seconds: int = 0
    estimated_cost_usd: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    artifact_key: Optional[str] = None
    checkpoint_key: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "owner_id", validate_identifier(self.owner_id, "owner_id"))
        object.__setattr__(self, "request_hash", validate_request_hash(self.request_hash))
        object.__setattr__(self, "status", RunStatus(self.status))
        if self.mode not in {"generate", "modify", "resume", "import"}:
            raise ValueError("mode must be generate, modify, resume, or import")
        if self.max_runtime_seconds < 0:
            raise ValueError("max_runtime_seconds must be non-negative")
        _finite_non_negative(self.max_cost_usd, "max_cost_usd")
        _finite_non_negative(self.estimated_cost_usd, "estimated_cost_usd")
        if self.version < 1:
            raise ValueError("version must be positive")
        try:
            json.dumps(dict(self.metadata), default=str)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES


_MUTABLE_RUN_FIELDS = frozenset({
    "status",
    "estimated_cost_usd",
    "started_at",
    "completed_at",
    "artifact_key",
    "checkpoint_key",
    "error_code",
    "error_message",
    "metadata",
})


def apply_run_changes(
    current: RunRecord,
    changes: Mapping[str, Any],
    *,
    now: float,
) -> RunRecord:
    """Validate and apply a typed optimistic update to a run record."""

    unknown = set(changes) - _MUTABLE_RUN_FIELDS
    if unknown:
        raise ValueError(f"Unsupported run update fields: {', '.join(sorted(unknown))}")
    values = dict(changes)
    if "status" in values:
        target = RunStatus(values["status"])
        if target != current.status and target not in _ALLOWED_TRANSITIONS[current.status]:
            raise InvalidRunTransitionError(
                f"Cannot transition run {current.run_id} from {current.status.value} "
                f"to {target.value}"
            )
        values["status"] = target
        if target == RunStatus.RUNNING and current.started_at is None:
            values.setdefault("started_at", now)
        if target in TERMINAL_RUN_STATUSES:
            values.setdefault("completed_at", now)
    values["updated_at"] = now
    values["version"] = current.version + 1
    return replace(current, **values)


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One ordered, replayable event emitted by a run."""

    run_id: str
    sequence: int
    event_type: str
    payload: Mapping[str, Any]
    created_at: float

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.sequence < 1:
            raise ValueError("event sequence must be positive")
        validate_identifier(self.event_type, "event_type")
        json.dumps(dict(self.payload), default=str)


@dataclass(frozen=True, slots=True)
class ReplayCursor:
    """Opaque client cursor bound to one run and event sequence."""

    run_id: str
    sequence: int = 0

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.sequence < 0:
            raise ValueError("cursor sequence must be non-negative")

    def encode(self) -> str:
        raw = json.dumps(
            {"run_id": self.run_id, "sequence": self.sequence},
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "ReplayCursor":
        try:
            padded = value + "=" * (-len(value) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
            return cls(run_id=payload["run_id"], sequence=int(payload["sequence"]))
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            binascii.Error,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("Invalid event replay cursor") from exc


@dataclass(frozen=True, slots=True)
class EventPage:
    events: tuple[EventRecord, ...]
    cursor: ReplayCursor
    has_more: bool


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    owner_id: str
    key: str
    request_hash: str
    run_id: str
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class IdempotencyClaim:
    record: IdempotencyRecord
    created: bool


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    run_id: str
    worker_id: str
    fencing_token: int
    acquired_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    allowed: bool
    resource: str
    limit: int
    used: int
    remaining: int
    reservation_id: Optional[str] = None
    expires_at: Optional[float] = None
    retry_after_seconds: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    storage_key: str
    file_name: str
    size_bytes: int
    sha256: str
    content_type: str
    created_at: float


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    storage_key: str
    size_bytes: int
    sha256: str
    created_at: float


@dataclass(frozen=True, slots=True)
class JobMessage:
    run_id: str
    owner_id: str
    payload: Mapping[str, Any]
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        validate_identifier(self.owner_id, "owner_id")
        json.dumps(dict(self.payload), default=str)


@dataclass(frozen=True, slots=True)
class QueuedJob:
    message: JobMessage
    receipt_handle: str
    message_id: str
    receive_count: int = 1
