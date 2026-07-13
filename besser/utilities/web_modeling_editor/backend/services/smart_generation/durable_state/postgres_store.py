"""PostgreSQL durable-state adapter for production SmartGen deployments."""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Callable, Mapping, Optional

from .dependencies import require_optional_dependency
from .errors import (
    DurableStateConfigurationError,
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
    RunStatus,
    apply_run_changes,
    validate_identifier,
    validate_request_hash,
    validate_run_id,
)


class PostgresStateStore:
    """PostgreSQL implementation with optimistic runs and fenced worker leases."""

    def __init__(self, database_url: str, *, clock: Callable[[], float] = time.time) -> None:
        if not database_url or not database_url.startswith(("postgres://", "postgresql://")):
            raise DurableStateConfigurationError("A PostgreSQL SmartGen database URL is required")
        self.database_url = database_url
        self._clock = clock
        self._psycopg = require_optional_dependency(
            "psycopg",
            "the 'psycopg[binary]' package",
        )
        rows = require_optional_dependency("psycopg.rows", "the 'psycopg[binary]' package")
        self._dict_row = rows.dict_row
        self._initialized = False

    def _connect(self):
        return self._psycopg.connect(self.database_url, row_factory=self._dict_row)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)
        self._initialized = True

    def _initialize_sync(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS smartgen_runs (
                run_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT,
                max_cost_usd DOUBLE PRECISION NOT NULL,
                max_runtime_seconds INTEGER NOT NULL,
                estimated_cost_usd DOUBLE PRECISION NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                started_at DOUBLE PRECISION,
                completed_at DOUBLE PRECISION,
                artifact_key TEXT,
                checkpoint_key TEXT,
                error_code TEXT,
                error_message TEXT,
                metadata_json JSONB NOT NULL,
                version INTEGER NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_smartgen_runs_owner_created
            ON smartgen_runs(owner_id, created_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS smartgen_event_cursors (
                run_id TEXT PRIMARY KEY REFERENCES smartgen_runs(run_id) ON DELETE CASCADE,
                next_sequence BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS smartgen_events (
                run_id TEXT NOT NULL REFERENCES smartgen_runs(run_id) ON DELETE CASCADE,
                sequence BIGINT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json JSONB NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (run_id, sequence)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS smartgen_idempotency (
                owner_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                run_id TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (owner_id, idempotency_key)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_smartgen_idempotency_expiry
            ON smartgen_idempotency(expires_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS smartgen_leases (
                run_id TEXT PRIMARY KEY REFERENCES smartgen_runs(run_id) ON DELETE CASCADE,
                worker_id TEXT NOT NULL,
                fencing_token BIGINT NOT NULL,
                acquired_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS smartgen_quota_reservations (
                reservation_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                resource TEXT NOT NULL,
                amount BIGINT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL,
                released_at DOUBLE PRECISION
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_smartgen_quota_active
            ON smartgen_quota_reservations(owner_id, resource, expires_at)
            """,
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    async def close(self) -> None:
        return None

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise DurableStateConfigurationError(
                "PostgresStateStore.initialize() must complete before use"
            )

    async def create_run(self, record: RunRecord) -> RunRecord:
        self._require_initialized()
        return await asyncio.to_thread(self._create_run_sync, record)

    def _create_run_sync(self, record: RunRecord) -> RunRecord:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO smartgen_runs (
                        run_id, owner_id, request_hash, status, mode, provider, model,
                        max_cost_usd, max_runtime_seconds, estimated_cost_usd,
                        created_at, updated_at, started_at, completed_at, artifact_key,
                        checkpoint_key, error_code, error_message, metadata_json, version
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
                    )
                    """,
                    self._run_values(record),
                )
                connection.execute(
                    "INSERT INTO smartgen_event_cursors(run_id, next_sequence) VALUES (%s, 1)",
                    (record.run_id,),
                )
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == "23505":
                raise RecordAlreadyExistsError(f"Run {record.run_id} already exists") from exc
            raise
        return record

    async def get_run(self, run_id: str) -> Optional[RunRecord]:
        self._require_initialized()
        return await asyncio.to_thread(self._get_run_sync, validate_run_id(run_id))

    def _get_run_sync(self, run_id: str) -> Optional[RunRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM smartgen_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    async def get_owned_run(self, owner_id: str, run_id: str) -> Optional[RunRecord]:
        self._require_initialized()
        owner_id = validate_identifier(owner_id, "owner_id")
        run_id = validate_run_id(run_id)
        return await asyncio.to_thread(self._get_owned_run_sync, owner_id, run_id)

    def _get_owned_run_sync(self, owner_id: str, run_id: str) -> Optional[RunRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM smartgen_runs WHERE owner_id = %s AND run_id = %s",
                (owner_id, run_id),
            ).fetchone()
        return self._row_to_run(row) if row else None

    async def update_run(
        self,
        run_id: str,
        expected_version: int,
        changes: Mapping[str, Any],
    ) -> RunRecord:
        self._require_initialized()
        return await asyncio.to_thread(
            self._update_run_sync,
            validate_run_id(run_id),
            expected_version,
            dict(changes),
        )

    def _update_run_sync(
        self,
        run_id: str,
        expected_version: int,
        changes: Mapping[str, Any],
    ) -> RunRecord:
        with self._connect() as connection:
            return self._update_run_in_transaction(
                connection,
                run_id,
                expected_version,
                changes,
            )

    async def update_run_fenced(
        self,
        run_id: str,
        expected_version: int,
        worker_id: str,
        fencing_token: int,
        changes: Mapping[str, Any],
    ) -> RunRecord:
        self._require_initialized()
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        if fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        return await asyncio.to_thread(
            self._update_run_fenced_sync,
            run_id,
            expected_version,
            worker_id,
            fencing_token,
            dict(changes),
        )

    def _update_run_fenced_sync(
        self,
        run_id: str,
        expected_version: int,
        worker_id: str,
        fencing_token: int,
        changes: Mapping[str, Any],
    ) -> RunRecord:
        with self._connect() as connection:
            self._require_fence_sync(connection, run_id, worker_id, fencing_token)
            return self._update_run_in_transaction(
                connection,
                run_id,
                expected_version,
                changes,
            )

    def _update_run_in_transaction(
        self,
        connection: Any,
        run_id: str,
        expected_version: int,
        changes: Mapping[str, Any],
    ) -> RunRecord:
        row = connection.execute(
            "SELECT * FROM smartgen_runs WHERE run_id = %s FOR UPDATE",
            (run_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Run {run_id} does not exist")
        current = self._row_to_run(row)
        if current.version != expected_version:
            raise OptimisticLockError(
                f"Run {run_id} is version {current.version}, expected {expected_version}"
            )
        updated = apply_run_changes(current, changes, now=self._clock())
        cursor = connection.execute(
            """
            UPDATE smartgen_runs SET
                status = %s, estimated_cost_usd = %s, updated_at = %s, started_at = %s,
                completed_at = %s, artifact_key = %s, checkpoint_key = %s, error_code = %s,
                error_message = %s, metadata_json = %s::jsonb, version = %s
            WHERE run_id = %s AND version = %s
            """,
            (
                updated.status.value,
                updated.estimated_cost_usd,
                updated.updated_at,
                updated.started_at,
                updated.completed_at,
                updated.artifact_key,
                updated.checkpoint_key,
                updated.error_code,
                updated.error_message,
                json.dumps(dict(updated.metadata), default=str, separators=(",", ":")),
                updated.version,
                run_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise OptimisticLockError(f"Run {run_id} changed during update")
        return updated

    async def list_runs(self, owner_id: str, *, limit: int = 50) -> tuple[RunRecord, ...]:
        self._require_initialized()
        owner_id = validate_identifier(owner_id, "owner_id")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        return await asyncio.to_thread(self._list_runs_sync, owner_id, limit)

    def _list_runs_sync(self, owner_id: str, limit: int) -> tuple[RunRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM smartgen_runs
                WHERE owner_id = %s
                ORDER BY created_at DESC, run_id DESC
                LIMIT %s
                """,
                (owner_id, limit),
            ).fetchall()
        return tuple(self._row_to_run(row) for row in rows)

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EventRecord:
        self._require_initialized()
        run_id = validate_run_id(run_id)
        event_type = validate_identifier(event_type, "event_type")
        payload_json = json.dumps(dict(payload), default=str, separators=(",", ":"))
        return await asyncio.to_thread(
            self._append_event_sync,
            run_id,
            event_type,
            payload_json,
        )

    def _append_event_sync(self, run_id: str, event_type: str, payload_json: str) -> EventRecord:
        with self._connect() as connection:
            return self._append_event_in_transaction(
                connection,
                run_id,
                event_type,
                payload_json,
            )

    async def append_event_fenced(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EventRecord:
        self._require_initialized()
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        event_type = validate_identifier(event_type, "event_type")
        if fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        payload_json = json.dumps(dict(payload), default=str, separators=(",", ":"))
        return await asyncio.to_thread(
            self._append_event_fenced_sync,
            run_id,
            worker_id,
            fencing_token,
            event_type,
            payload_json,
        )

    def _append_event_fenced_sync(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        event_type: str,
        payload_json: str,
    ) -> EventRecord:
        with self._connect() as connection:
            self._require_fence_sync(connection, run_id, worker_id, fencing_token)
            return self._append_event_in_transaction(
                connection,
                run_id,
                event_type,
                payload_json,
            )

    def _append_event_in_transaction(
        self,
        connection: Any,
        run_id: str,
        event_type: str,
        payload_json: str,
    ) -> EventRecord:
        created_at = self._clock()
        row = connection.execute(
            """
            UPDATE smartgen_event_cursors
            SET next_sequence = next_sequence + 1
            WHERE run_id = %s
            RETURNING next_sequence - 1 AS sequence
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Run {run_id} does not exist")
        sequence = int(row["sequence"])
        connection.execute(
            """
            INSERT INTO smartgen_events(run_id, sequence, event_type, payload_json, created_at)
            VALUES (%s, %s, %s, %s::jsonb, %s)
            """,
            (run_id, sequence, event_type, payload_json, created_at),
        )
        return EventRecord(run_id, sequence, event_type, json.loads(payload_json), created_at)

    def _require_fence_sync(
        self,
        connection: Any,
        run_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM smartgen_leases
            WHERE run_id = %s AND worker_id = %s AND fencing_token = %s AND expires_at > %s
            FOR UPDATE
            """,
            (run_id, worker_id, fencing_token, self._clock()),
        ).fetchone()
        if row is None:
            raise LeaseLostError(f"Worker lease for run {run_id} is no longer valid")

    async def read_events(
        self,
        run_id: str,
        *,
        cursor: Optional[ReplayCursor] = None,
        limit: int = 200,
    ) -> EventPage:
        self._require_initialized()
        run_id = validate_run_id(run_id)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        after_sequence = 0
        if cursor is not None:
            if cursor.run_id != run_id:
                raise ValueError("Replay cursor belongs to a different run")
            after_sequence = cursor.sequence
        return await asyncio.to_thread(
            self._read_events_sync,
            run_id,
            after_sequence,
            limit,
        )

    def _read_events_sync(self, run_id: str, after_sequence: int, limit: int) -> EventPage:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM smartgen_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone() is None:
                raise RecordNotFoundError(f"Run {run_id} does not exist")
            rows = connection.execute(
                """
                SELECT * FROM smartgen_events
                WHERE run_id = %s AND sequence > %s
                ORDER BY sequence ASC
                LIMIT %s
                """,
                (run_id, after_sequence, limit + 1),
            ).fetchall()
        selected = rows[:limit]
        events = tuple(self._row_to_event(row) for row in selected)
        next_sequence = events[-1].sequence if events else after_sequence
        return EventPage(
            events,
            ReplayCursor(run_id, next_sequence),
            len(rows) > limit,
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
        self._require_initialized()
        owner_id = validate_identifier(owner_id, "owner_id")
        key = validate_identifier(key, "idempotency key")
        request_hash = validate_request_hash(request_hash)
        run_id = validate_run_id(run_id)
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        return await asyncio.to_thread(
            self._claim_idempotency_sync,
            owner_id,
            key,
            request_hash,
            run_id,
            ttl_seconds,
        )

    def _claim_idempotency_sync(
        self,
        owner_id: str,
        key: str,
        request_hash: str,
        run_id: str,
        ttl_seconds: int,
    ) -> IdempotencyClaim:
        now = self._clock()
        with self._connect() as connection:
            self._advisory_lock(connection, f"idempotency:{owner_id}:{key}")
            row = connection.execute(
                """
                SELECT * FROM smartgen_idempotency
                WHERE owner_id = %s AND idempotency_key = %s
                FOR UPDATE
                """,
                (owner_id, key),
            ).fetchone()
            if row is not None and float(row["expires_at"]) <= now:
                connection.execute(
                    "DELETE FROM smartgen_idempotency WHERE owner_id = %s AND idempotency_key = %s",
                    (owner_id, key),
                )
                row = None
            if row is not None:
                record = self._row_to_idempotency(row)
                if record.request_hash != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for a different request"
                    )
                return IdempotencyClaim(record, False)
            record = IdempotencyRecord(owner_id, key, request_hash, run_id, now, now + ttl_seconds)
            connection.execute(
                """
                INSERT INTO smartgen_idempotency(
                    owner_id, idempotency_key, request_hash, run_id, created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (owner_id, key, request_hash, run_id, now, record.expires_at),
            )
            return IdempotencyClaim(record, True)

    async def acquire_lease(
        self,
        run_id: str,
        worker_id: str,
        *,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]:
        self._require_initialized()
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        return await asyncio.to_thread(self._acquire_lease_sync, run_id, worker_id, ttl_seconds)

    def _acquire_lease_sync(
        self,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]:
        now = self._clock()
        with self._connect() as connection:
            self._advisory_lock(connection, f"lease:{run_id}")
            if connection.execute(
                "SELECT 1 FROM smartgen_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone() is None:
                raise RecordNotFoundError(f"Run {run_id} does not exist")
            row = connection.execute(
                "SELECT * FROM smartgen_leases WHERE run_id = %s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if row is not None and float(row["expires_at"]) > now:
                if row["worker_id"] != worker_id:
                    return None
                lease = LeaseRecord(
                    run_id,
                    worker_id,
                    int(row["fencing_token"]),
                    float(row["acquired_at"]),
                    now + ttl_seconds,
                )
            else:
                token = (int(row["fencing_token"]) if row is not None else 0) + 1
                lease = LeaseRecord(run_id, worker_id, token, now, now + ttl_seconds)
            connection.execute(
                """
                INSERT INTO smartgen_leases(run_id, worker_id, fencing_token, acquired_at, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(run_id) DO UPDATE SET
                    worker_id = excluded.worker_id,
                    fencing_token = excluded.fencing_token,
                    acquired_at = excluded.acquired_at,
                    expires_at = excluded.expires_at
                """,
                (
                    lease.run_id,
                    lease.worker_id,
                    lease.fencing_token,
                    lease.acquired_at,
                    lease.expires_at,
                ),
            )
            return lease

    async def renew_lease(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        *,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]:
        self._require_initialized()
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        if fencing_token < 1 or ttl_seconds <= 0:
            raise ValueError("fencing_token and ttl_seconds must be positive")
        return await asyncio.to_thread(
            self._renew_lease_sync,
            run_id,
            worker_id,
            fencing_token,
            ttl_seconds,
        )

    def _renew_lease_sync(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
        ttl_seconds: int,
    ) -> Optional[LeaseRecord]:
        now = self._clock()
        expires_at = now + ttl_seconds
        with self._connect() as connection:
            row = connection.execute(
                """
                UPDATE smartgen_leases SET expires_at = %s
                WHERE run_id = %s AND worker_id = %s AND fencing_token = %s AND expires_at > %s
                RETURNING acquired_at
                """,
                (expires_at, run_id, worker_id, fencing_token, now),
            ).fetchone()
            if row is None:
                return None
        return LeaseRecord(run_id, worker_id, fencing_token, float(row["acquired_at"]), expires_at)

    async def release_lease(
        self,
        run_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> bool:
        self._require_initialized()
        run_id = validate_run_id(run_id)
        worker_id = validate_identifier(worker_id, "worker_id")
        return await asyncio.to_thread(
            self._release_lease_sync,
            run_id,
            worker_id,
            fencing_token,
        )

    def _release_lease_sync(self, run_id: str, worker_id: str, fencing_token: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE smartgen_leases SET expires_at = %s
                WHERE run_id = %s AND worker_id = %s AND fencing_token = %s
                """,
                (self._clock(), run_id, worker_id, fencing_token),
            )
            return cursor.rowcount == 1

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
        self._require_initialized()
        owner_id = validate_identifier(owner_id, "owner_id")
        resource = validate_identifier(resource, "quota resource")
        reservation_id = validate_identifier(reservation_id, "reservation_id")
        if amount <= 0 or limit <= 0 or ttl_seconds <= 0:
            raise ValueError("amount, limit, and ttl_seconds must be positive")
        if amount > limit:
            return QuotaDecision(False, resource, limit, 0, limit)
        return await asyncio.to_thread(
            self._reserve_quota_sync,
            owner_id,
            resource,
            reservation_id,
            amount,
            limit,
            ttl_seconds,
        )

    def _reserve_quota_sync(
        self,
        owner_id: str,
        resource: str,
        reservation_id: str,
        amount: int,
        limit: int,
        ttl_seconds: int,
    ) -> QuotaDecision:
        now = self._clock()
        with self._connect() as connection:
            self._advisory_lock(connection, f"quota:{owner_id}:{resource}")
            existing = connection.execute(
                "SELECT * FROM smartgen_quota_reservations WHERE reservation_id = %s FOR UPDATE",
                (reservation_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["owner_id"] != owner_id
                    or existing["resource"] != resource
                    or int(existing["amount"]) != amount
                ):
                    raise ValueError("reservation_id was reused with different quota inputs")
                used = self._quota_used(connection, owner_id, resource, now)
                active = existing["released_at"] is None and float(existing["expires_at"]) > now
                return QuotaDecision(
                    active,
                    resource,
                    limit,
                    used,
                    max(0, limit - used),
                    reservation_id=reservation_id if active else None,
                    expires_at=float(existing["expires_at"]) if active else None,
                )
            used = self._quota_used(connection, owner_id, resource, now)
            if used + amount > limit:
                retry = connection.execute(
                    """
                    SELECT MIN(expires_at) AS retry_at FROM smartgen_quota_reservations
                    WHERE owner_id = %s AND resource = %s
                      AND released_at IS NULL AND expires_at > %s
                    """,
                    (owner_id, resource, now),
                ).fetchone()
                retry_at = retry["retry_at"] if retry else None
                retry_after = max(1, math.ceil(float(retry_at) - now)) if retry_at is not None else None
                return QuotaDecision(
                    False,
                    resource,
                    limit,
                    used,
                    max(0, limit - used),
                    retry_after_seconds=retry_after,
                )
            expires_at = now + ttl_seconds
            connection.execute(
                """
                INSERT INTO smartgen_quota_reservations(
                    reservation_id, owner_id, resource, amount, created_at, expires_at, released_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NULL)
                """,
                (reservation_id, owner_id, resource, amount, now, expires_at),
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
        self._require_initialized()
        reservation_id = validate_identifier(reservation_id, "reservation_id")
        return await asyncio.to_thread(self._release_quota_sync, reservation_id)

    def _release_quota_sync(self, reservation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE smartgen_quota_reservations SET released_at = %s
                WHERE reservation_id = %s AND released_at IS NULL
                """,
                (self._clock(), reservation_id),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _advisory_lock(connection, identity: str) -> None:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (identity,),
        )

    @staticmethod
    def _quota_used(connection, owner_id: str, resource: str, now: float) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS used
            FROM smartgen_quota_reservations
            WHERE owner_id = %s AND resource = %s AND released_at IS NULL AND expires_at > %s
            """,
            (owner_id, resource, now),
        ).fetchone()
        return int(row["used"])

    @staticmethod
    def _run_values(record: RunRecord) -> tuple[Any, ...]:
        return (
            record.run_id,
            record.owner_id,
            record.request_hash,
            record.status.value,
            record.mode,
            record.provider,
            record.model,
            record.max_cost_usd,
            record.max_runtime_seconds,
            record.estimated_cost_usd,
            record.created_at,
            record.updated_at,
            record.started_at,
            record.completed_at,
            record.artifact_key,
            record.checkpoint_key,
            record.error_code,
            record.error_message,
            json.dumps(dict(record.metadata), default=str, separators=(",", ":")),
            record.version,
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    @classmethod
    def _row_to_run(cls, row: Mapping[str, Any]) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            owner_id=row["owner_id"],
            request_hash=row["request_hash"],
            status=RunStatus(row["status"]),
            mode=row["mode"],
            provider=row["provider"],
            model=row["model"],
            max_cost_usd=float(row["max_cost_usd"]),
            max_runtime_seconds=int(row["max_runtime_seconds"]),
            estimated_cost_usd=float(row["estimated_cost_usd"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            artifact_key=row["artifact_key"],
            checkpoint_key=row["checkpoint_key"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            metadata=cls._json_value(row["metadata_json"]),
            version=int(row["version"]),
        )

    @classmethod
    def _row_to_event(cls, row: Mapping[str, Any]) -> EventRecord:
        return EventRecord(
            run_id=row["run_id"],
            sequence=int(row["sequence"]),
            event_type=row["event_type"],
            payload=cls._json_value(row["payload_json"]),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _row_to_idempotency(row: Mapping[str, Any]) -> IdempotencyRecord:
        return IdempotencyRecord(
            owner_id=row["owner_id"],
            key=row["idempotency_key"],
            request_hash=row["request_hash"],
            run_id=row["run_id"],
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )
