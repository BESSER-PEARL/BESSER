"""In-memory durable-state contract implementation for tests and local tools."""

from __future__ import annotations

import asyncio
import copy
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

from .errors import (
    IdempotencyConflictError,
    LeaseLostError,
    OptimisticLockError,
    RecordAlreadyExistsError,
    RecordNotFoundError,
)
from .models import (
    EventPage,
    EventRecord,
    IdempotencyClaim,
    IdempotencyRecord,
    LeaseRecord,
    QuotaDecision,
    ReplayCursor,
    RunRecord,
    apply_run_changes,
    validate_identifier,
    validate_request_hash,
    validate_run_id,
)


@dataclass(slots=True)
class _QuotaReservation:
    owner_id: str
    resource: str
    reservation_id: str
    amount: int
    created_at: float
    expires_at: float
    released_at: Optional[float] = None


def _clone_run(record: RunRecord) -> RunRecord:
    return replace(record, metadata=copy.deepcopy(dict(record.metadata)))


def _clone_event(record: EventRecord) -> EventRecord:
    return replace(record, payload=copy.deepcopy(dict(record.payload)))


class InMemoryStateStore:
    """Atomic reference adapter with the same semantics as SQL backends."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[EventRecord]] = {}
        self._idempotency: dict[tuple[str, str], IdempotencyRecord] = {}
        self._leases: dict[str, LeaseRecord] = {}
        self._quotas: dict[str, _QuotaReservation] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def create_run(self, record: RunRecord) -> RunRecord:
        async with self._lock:
            if record.run_id in self._runs:
                raise RecordAlreadyExistsError(f"Run {record.run_id} already exists")
            stored = _clone_run(record)
            self._runs[record.run_id] = stored
            self._events[record.run_id] = []
            return _clone_run(stored)

    async def get_run(self, run_id: str) -> Optional[RunRecord]:
        validate_run_id(run_id)
        async with self._lock:
            record = self._runs.get(run_id)
            return _clone_run(record) if record is not None else None

    async def get_owned_run(self, owner_id: str, run_id: str) -> Optional[RunRecord]:
        owner_id = validate_identifier(owner_id, "owner_id")
        record = await self.get_run(run_id)
        if record is None or record.owner_id != owner_id:
            return None
        return record

    async def update_run(
        self,
        run_id: str,
        expected_version: int,
        changes: Mapping[str, Any],
    ) -> RunRecord:
        validate_run_id(run_id)
        async with self._lock:
            current = self._runs.get(run_id)
            if current is None:
                raise RecordNotFoundError(f"Run {run_id} does not exist")
            if current.version != expected_version:
                raise OptimisticLockError(
                    f"Run {run_id} is version {current.version}, expected {expected_version}"
                )
            updated = apply_run_changes(current, changes, now=self._clock())
            self._runs[run_id] = _clone_run(updated)
            return _clone_run(updated)

    async def update_run_fenced(
        self,
        run_id: str,
        expected_version: int,
        worker_id: str,
        fencing_token: int,
        changes: Mapping[str, Any],
    ) -> RunRecord:
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        if fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        async with self._lock:
            self._require_fence_locked(run_id, worker_id, fencing_token)
            current = self._runs.get(run_id)
            if current is None:
                raise RecordNotFoundError(f"Run {run_id} does not exist")
            if current.version != expected_version:
                raise OptimisticLockError(
                    f"Run {run_id} is version {current.version}, expected {expected_version}"
                )
            updated = apply_run_changes(current, changes, now=self._clock())
            self._runs[run_id] = _clone_run(updated)
            return _clone_run(updated)

    async def list_runs(self, owner_id: str, *, limit: int = 50) -> tuple[RunRecord, ...]:
        owner_id = validate_identifier(owner_id, "owner_id")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        async with self._lock:
            records = [record for record in self._runs.values() if record.owner_id == owner_id]
            records.sort(key=lambda item: (item.created_at, item.run_id), reverse=True)
            return tuple(_clone_run(record) for record in records[:limit])

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EventRecord:
        validate_run_id(run_id)
        event_type = validate_identifier(event_type, "event_type")
        async with self._lock:
            if run_id not in self._runs:
                raise RecordNotFoundError(f"Run {run_id} does not exist")
            records = self._events.setdefault(run_id, [])
            event = EventRecord(
                run_id=run_id,
                sequence=len(records) + 1,
                event_type=event_type,
                payload=copy.deepcopy(dict(payload)),
                created_at=self._clock(),
            )
            records.append(event)
            return _clone_event(event)

    async def append_event_fenced(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EventRecord:
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        event_type = validate_identifier(event_type, "event_type")
        if fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        async with self._lock:
            self._require_fence_locked(run_id, worker_id, fencing_token)
            records = self._events.setdefault(run_id, [])
            event = EventRecord(
                run_id=run_id,
                sequence=len(records) + 1,
                event_type=event_type,
                payload=copy.deepcopy(dict(payload)),
                created_at=self._clock(),
            )
            records.append(event)
            return _clone_event(event)

    def _require_fence_locked(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> None:
        lease = self._leases.get(run_id)
        if (
            lease is None
            or lease.worker_id != worker_id
            or lease.fencing_token != fencing_token
            or lease.expires_at <= self._clock()
        ):
            raise LeaseLostError(f"Worker lease for run {run_id} is no longer valid")

    async def read_events(
        self,
        run_id: str,
        *,
        cursor: Optional[ReplayCursor] = None,
        limit: int = 200,
    ) -> EventPage:
        validate_run_id(run_id)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        after_sequence = 0
        if cursor is not None:
            if cursor.run_id != run_id:
                raise ValueError("Replay cursor belongs to a different run")
            after_sequence = cursor.sequence
        async with self._lock:
            if run_id not in self._runs:
                raise RecordNotFoundError(f"Run {run_id} does not exist")
            candidates = [event for event in self._events.get(run_id, ()) if event.sequence > after_sequence]
            selected = candidates[:limit]
            next_sequence = selected[-1].sequence if selected else after_sequence
            return EventPage(
                events=tuple(_clone_event(event) for event in selected),
                cursor=ReplayCursor(run_id=run_id, sequence=next_sequence),
                has_more=len(candidates) > limit,
            )

    async def claim_idempotency(
        self,
        owner_id: str,
        key: str,
        request_hash: str,
        run_id: str,
        *,
        ttl_seconds: int,
    ) -> IdempotencyClaim:
        owner_id = validate_identifier(owner_id, "owner_id")
        key = validate_identifier(key, "idempotency key")
        request_hash = validate_request_hash(request_hash)
        run_id = validate_run_id(run_id)
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = self._clock()
        identity = (owner_id, key)
        async with self._lock:
            existing = self._idempotency.get(identity)
            if existing is not None and existing.expires_at <= now:
                self._idempotency.pop(identity, None)
                existing = None
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for a different request"
                    )
                return IdempotencyClaim(record=existing, created=False)
            record = IdempotencyRecord(
                owner_id=owner_id,
                key=key,
                request_hash=request_hash,
                run_id=run_id,
                created_at=now,
                expires_at=now + ttl_seconds,
            )
            self._idempotency[identity] = record
            return IdempotencyClaim(record=record, created=True)

    async def acquire_lease(
        self,
        run_id: str,
        worker_id: str,
        *,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]:
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = self._clock()
        async with self._lock:
            if run_id not in self._runs:
                raise RecordNotFoundError(f"Run {run_id} does not exist")
            existing = self._leases.get(run_id)
            if existing is not None and existing.expires_at > now:
                if existing.worker_id != worker_id:
                    return None
                renewed = replace(existing, expires_at=now + ttl_seconds)
                self._leases[run_id] = renewed
                return renewed
            token = (existing.fencing_token if existing is not None else 0) + 1
            lease = LeaseRecord(
                run_id=run_id,
                worker_id=worker_id,
                fencing_token=token,
                acquired_at=now,
                expires_at=now + ttl_seconds,
            )
            self._leases[run_id] = lease
            return lease

    async def renew_lease(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        *,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]:
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        if ttl_seconds <= 0 or fencing_token < 1:
            raise ValueError("ttl_seconds and fencing_token must be positive")
        now = self._clock()
        async with self._lock:
            existing = self._leases.get(run_id)
            if (
                existing is None
                or existing.worker_id != worker_id
                or existing.fencing_token != fencing_token
                or existing.expires_at <= now
            ):
                return None
            renewed = replace(existing, expires_at=now + ttl_seconds)
            self._leases[run_id] = renewed
            return renewed

    async def release_lease(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> bool:
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        now = self._clock()
        async with self._lock:
            existing = self._leases.get(run_id)
            if (
                existing is None
                or existing.worker_id != worker_id
                or existing.fencing_token != fencing_token
            ):
                return False
            self._leases[run_id] = replace(existing, expires_at=now)
            return True

    async def reserve_quota(
        self,
        owner_id: str,
        resource: str,
        reservation_id: str,
        *,
        amount: int,
        limit: int,
        ttl_seconds: int,
    ) -> QuotaDecision:
        owner_id = validate_identifier(owner_id, "owner_id")
        resource = validate_identifier(resource, "quota resource")
        reservation_id = validate_identifier(reservation_id, "reservation_id")
        if amount <= 0 or limit <= 0 or ttl_seconds <= 0:
            raise ValueError("amount, limit, and ttl_seconds must be positive")
        if amount > limit:
            return QuotaDecision(False, resource, limit, 0, limit)
        now = self._clock()
        async with self._lock:
            existing = self._quotas.get(reservation_id)
            if existing is not None:
                if (
                    existing.owner_id != owner_id
                    or existing.resource != resource
                    or existing.amount != amount
                ):
                    raise ValueError("reservation_id was reused with different quota inputs")
                active = existing.released_at is None and existing.expires_at > now
                used = self._quota_used(owner_id, resource, now)
                return QuotaDecision(
                    allowed=active,
                    resource=resource,
                    limit=limit,
                    used=used,
                    remaining=max(0, limit - used),
                    reservation_id=reservation_id if active else None,
                    expires_at=existing.expires_at if active else None,
                )
            used = self._quota_used(owner_id, resource, now)
            if used + amount > limit:
                retry_candidates = [
                    row.expires_at
                    for row in self._quotas.values()
                    if row.owner_id == owner_id
                    and row.resource == resource
                    and row.released_at is None
                    and row.expires_at > now
                ]
                retry_after = None
                if retry_candidates:
                    retry_after = max(1, math.ceil(min(retry_candidates) - now))
                return QuotaDecision(
                    False,
                    resource,
                    limit,
                    used,
                    max(0, limit - used),
                    retry_after_seconds=retry_after,
                )
            expires_at = now + ttl_seconds
            self._quotas[reservation_id] = _QuotaReservation(
                owner_id=owner_id,
                resource=resource,
                reservation_id=reservation_id,
                amount=amount,
                created_at=now,
                expires_at=expires_at,
            )
            return QuotaDecision(
                True,
                resource,
                limit,
                used + amount,
                limit - used - amount,
                reservation_id=reservation_id,
                expires_at=expires_at,
            )

    async def release_quota(self, reservation_id: str) -> bool:
        reservation_id = validate_identifier(reservation_id, "reservation_id")
        async with self._lock:
            existing = self._quotas.get(reservation_id)
            if existing is None or existing.released_at is not None:
                return False
            existing.released_at = self._clock()
            return True

    def _quota_used(self, owner_id: str, resource: str, now: float) -> int:
        return sum(
            row.amount
            for row in self._quotas.values()
            if row.owner_id == owner_id
            and row.resource == resource
            and row.released_at is None
            and row.expires_at > now
        )
