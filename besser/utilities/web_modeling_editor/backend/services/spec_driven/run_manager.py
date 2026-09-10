"""Durable ownership and replay for spec-driven generation runs.

The LLM runner produces an asynchronous stream of SSE frames.  Historically
the HTTP response consumed that generator directly, which made the browser
connection the effective owner of the worker: closing the response closed the
generator and could release the run while its blocking worker was still
finishing.

This module inserts a durable boundary between the producer and subscribers:

* one background task consumes the runner until it really terminates;
* every redacted SSE frame is assigned a monotonically increasing sequence;
* frames and run status are stored in SQLite before subscribers see them;
* any number of short-lived HTTP subscribers can replay from a sequence;
* losing every subscriber starts a configurable grace period before the
  producer is cooperatively stopped, limiting unattended BYOK spend.

The SQLite store intentionally contains events and lifecycle metadata only.
It never stores the generation request or the user's API key.  A server restart
marks formerly-running records as ``interrupted``; the existing checkpoint
resume endpoint can then continue them when the client re-supplies its request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Literal, Optional

from .sse_events import ErrorEvent, format_sse

logger = logging.getLogger(__name__)

RunStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "partial",
    "failed",
    "cancelled",
    "abandoned",
    "interrupted",
]

TERMINAL_RUN_STATUSES = frozenset(
    {"succeeded", "partial", "failed", "cancelled", "abandoned", "interrupted"}
)
_NON_TERMINAL_ERROR_CODES = frozenset({"COST_CAP", "TIMEOUT", "INCOMPLETE"})
_DEFAULT_DB_NAME = "besser_spec_driven_runs.sqlite3"
_HEARTBEAT_SECONDS = 15.0


@dataclass(frozen=True)
class StoredRunEvent:
    run_id: str
    sequence: int
    event_type: str
    frame: bytes
    created_at: float
    payload: dict


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: RunStatus
    created_at: float
    updated_at: float
    last_sequence: int
    terminal_event: Optional[str] = None
    error: Optional[str] = None
    subscriber_count: int = 0
    last_subscriber_at: Optional[float] = None
    disconnected_at: Optional[float] = None
    abandonment_requested_at: Optional[float] = None
    resume_available: bool = False

    def to_api_dict(self) -> dict:
        return {
            "runId": self.run_id,
            "status": self.status,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "lastSequence": self.last_sequence,
            "terminalEvent": self.terminal_event,
            "error": self.error,
            "subscriberCount": self.subscriber_count,
            "lastSubscriberAt": self.last_subscriber_at,
            "disconnectedAt": self.disconnected_at,
            "abandonmentRequestedAt": self.abandonment_requested_at,
            "resumeAvailable": self.resume_available,
        }


@dataclass(frozen=True)
class _AbandonmentPolicy:
    grace_seconds: float
    cancel: Callable[[], None]
    resume_available: Optional[Callable[[], bool]] = None


def _default_store_path() -> str:
    configured = os.environ.get("BESSER_LLM_RUN_STORE_PATH", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(tempfile.gettempdir(), _DEFAULT_DB_NAME)


def _parse_frame(frame: bytes) -> tuple[str, dict]:
    """Return the event name and JSON body from one canonical SSE frame."""
    text = frame.decode("utf-8", errors="replace")
    event_type = "message"
    data_lines: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("event:"):
            event_type = line[6:].strip() or "message"
        elif line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    payload: dict = {}
    if data_lines:
        try:
            decoded = json.loads("\n".join(data_lines))
            if isinstance(decoded, dict):
                payload = decoded
        except (TypeError, ValueError):
            logger.warning("Could not parse generated SSE frame for durable replay")
    if isinstance(payload.get("event"), str):
        event_type = payload["event"]
    return event_type, payload


def _add_sequence(frame: bytes, sequence: int) -> tuple[bytes, str, dict]:
    """Add both SSE ``id`` and JSON ``sequence`` to a generated frame."""
    event_type, payload = _parse_frame(frame)
    payload = {**payload, "sequence": sequence}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sequenced = (
        f"id: {sequence}\n"
        f"event: {event_type}\n"
        f"data: {body}\n\n"
    ).encode("utf-8")
    return sequenced, event_type, payload


class SqliteRunEventStore:
    """Small synchronous SQLite event store protected by a process lock.

    SQLite calls are intentionally short (one indexed lookup/insert/update per
    emitted event).  WAL mode lets status/replay readers proceed while the
    producer appends.  The interface is kept independent from the run manager
    so a PostgreSQL implementation can replace it for a multi-instance deploy.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or _default_store_path()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=10.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS spec_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_sequence INTEGER NOT NULL DEFAULT 0,
                    terminal_event TEXT,
                    error TEXT,
                    subscriber_count INTEGER NOT NULL DEFAULT 0,
                    last_subscriber_at REAL,
                    disconnected_at REAL,
                    abandonment_requested_at REAL,
                    resume_available INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS spec_run_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    frame BLOB NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id) REFERENCES spec_runs(run_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_spec_runs_updated_at
                    ON spec_runs(updated_at);
                """
            )
            # Existing pilot databases predate subscriber lifecycle columns.
            # SQLite has no ADD COLUMN IF NOT EXISTS, so inspect and migrate
            # each additive field explicitly.
            existing_columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(spec_runs)"
                ).fetchall()
            }
            additive_columns = {
                "subscriber_count": "INTEGER NOT NULL DEFAULT 0",
                "last_subscriber_at": "REAL",
                "disconnected_at": "REAL",
                "abandonment_requested_at": "REAL",
                "resume_available": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in additive_columns.items():
                if column not in existing_columns:
                    self._conn.execute(
                        f"ALTER TABLE spec_runs ADD COLUMN {column} {declaration}"
                    )
            # No producer can survive a Python process restart.  Preserve its
            # events/checkpoint but tell reconnecting clients the truth.
            now = time.time()
            self._conn.execute(
                """
                UPDATE spec_runs
                   SET status = 'interrupted', updated_at = ?,
                       terminal_event = 'interrupted', subscriber_count = 0,
                       disconnected_at = COALESCE(disconnected_at, ?)
                 WHERE status IN ('queued', 'running')
                """,
                (now, now),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def begin_run(self, run_id: str, *, resume: bool = False) -> RunRecord:
        now = time.time()
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT run_id FROM spec_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    """
                    INSERT INTO spec_runs(
                        run_id, status, created_at, updated_at, last_sequence
                    ) VALUES (?, 'queued', ?, ?, 0)
                    """,
                    (run_id, now, now),
                )
            elif not resume:
                raise ValueError(f"Run {run_id} already exists")
            else:
                # A resume is a new attempt under the same capability ID.
                # Old terminal frames must not be replayed before the resumed
                # attempt's events, so reset the event stream atomically.
                self._conn.execute(
                    "DELETE FROM spec_run_events WHERE run_id = ?", (run_id,)
                )
                self._conn.execute(
                    """
                    UPDATE spec_runs
                       SET status = 'queued', updated_at = ?,
                           terminal_event = NULL, error = NULL,
                           subscriber_count = 0, disconnected_at = NULL,
                           abandonment_requested_at = NULL,
                           resume_available = 0, last_sequence = 0
                     WHERE run_id = ?
                    """,
                    (now, run_id),
                )
        record = self.get_run(run_id)
        if record is None:  # pragma: no cover - guarded by the transaction
            raise RuntimeError(f"Could not create run {run_id}")
        return record

    def append(self, run_id: str, frame: bytes) -> StoredRunEvent:
        created_at = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT last_sequence FROM spec_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run {run_id}")
            sequence = int(row["last_sequence"]) + 1
            sequenced, event_type, payload = _add_sequence(frame, sequence)
            self._conn.execute(
                """
                INSERT INTO spec_run_events(
                    run_id, sequence, event_type, frame, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sequence,
                    event_type,
                    sequenced,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    created_at,
                ),
            )
            self._conn.execute(
                """
                UPDATE spec_runs
                   SET last_sequence = ?, updated_at = ?
                 WHERE run_id = ?
                """,
                (sequence, created_at, run_id),
            )
        return StoredRunEvent(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            frame=sequenced,
            created_at=created_at,
            payload=payload,
        )

    def set_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        terminal_event: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE spec_runs
                   SET status = ?, updated_at = ?, terminal_event = ?, error = ?
                 WHERE run_id = ?
                """,
                (status, time.time(), terminal_event, error, run_id),
            )

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM spec_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            status=row["status"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_sequence=int(row["last_sequence"]),
            terminal_event=row["terminal_event"],
            error=row["error"],
            subscriber_count=int(row["subscriber_count"] or 0),
            last_subscriber_at=(
                float(row["last_subscriber_at"])
                if row["last_subscriber_at"] is not None
                else None
            ),
            disconnected_at=(
                float(row["disconnected_at"])
                if row["disconnected_at"] is not None
                else None
            ),
            abandonment_requested_at=(
                float(row["abandonment_requested_at"])
                if row["abandonment_requested_at"] is not None
                else None
            ),
            resume_available=bool(row["resume_available"]),
        )

    def set_subscriber_count(self, run_id: str, count: int) -> None:
        """Persist subscriber presence without exposing subscriber identity."""
        now = time.time()
        safe_count = max(0, count)
        with self._lock, self._conn:
            if safe_count > 0:
                self._conn.execute(
                    """
                    UPDATE spec_runs
                       SET subscriber_count = ?, last_subscriber_at = ?,
                           disconnected_at = NULL, updated_at = ?
                     WHERE run_id = ?
                    """,
                    (safe_count, now, now, run_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE spec_runs
                       SET subscriber_count = 0, last_subscriber_at = ?,
                           disconnected_at = ?, updated_at = ?
                     WHERE run_id = ?
                    """,
                    (now, now, now, run_id),
                )

    def mark_abandonment_requested(self, run_id: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE spec_runs
                   SET abandonment_requested_at = ?, updated_at = ?
                 WHERE run_id = ?
                """,
                (now, now, run_id),
            )

    def set_resume_available(self, run_id: str, available: bool) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE spec_runs
                   SET resume_available = ?, updated_at = ?
                 WHERE run_id = ?
                """,
                (1 if available else 0, time.time(), run_id),
            )

    def events_after(
        self,
        run_id: str,
        sequence: int,
        *,
        limit: int = 250,
    ) -> list[StoredRunEvent]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT run_id, sequence, event_type, frame, payload_json, created_at
                  FROM spec_run_events
                 WHERE run_id = ? AND sequence > ?
                 ORDER BY sequence ASC
                 LIMIT ?
                """,
                (run_id, max(0, sequence), max(1, limit)),
            ).fetchall()
        events: list[StoredRunEvent] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                payload = {}
            events.append(
                StoredRunEvent(
                    run_id=row["run_id"],
                    sequence=int(row["sequence"]),
                    event_type=row["event_type"],
                    frame=bytes(row["frame"]),
                    created_at=float(row["created_at"]),
                    payload=payload,
                )
            )
        return events

    def sweep_expired(self, ttl_seconds: int) -> int:
        cutoff = time.time() - max(1, ttl_seconds)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                DELETE FROM spec_runs
                 WHERE updated_at < ?
                   AND status NOT IN ('queued', 'running')
                """,
                (cutoff,),
            )
            return max(0, cursor.rowcount)

    def clear(self) -> None:
        """Remove all records. Intended for isolated tests only."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM spec_run_events")
            self._conn.execute("DELETE FROM spec_runs")


class DurableRunManager:
    """Own producer tasks and expose replayable event subscriptions."""

    def __init__(self, store: Optional[SqliteRunEventStore] = None) -> None:
        self.store = store or SqliteRunEventStore()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._conditions: dict[str, asyncio.Condition] = {}
        self._subscriber_counts: dict[str, int] = {}
        self._abandonment_policies: dict[str, _AbandonmentPolicy] = {}
        self._abandonment_tasks: dict[str, asyncio.Task[None]] = {}
        self._abandonment_triggered: set[str] = set()
        self._generations: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        run_id: str,
        source: AsyncIterator[bytes],
        *,
        resume: bool = False,
        on_abandoned: Optional[Callable[[], None]] = None,
        resume_available: Optional[Callable[[], bool]] = None,
        disconnect_grace_seconds: float = 300.0,
    ) -> RunRecord:
        async with self._lock:
            existing_task = self._tasks.get(run_id)
            if existing_task is not None and not existing_task.done():
                raise ValueError(f"Run {run_id} is already active")
            record = self.store.begin_run(run_id, resume=resume)
            generation = self._generations.get(run_id, 0) + 1
            self._generations[run_id] = generation
            condition = asyncio.Condition()
            self._conditions[run_id] = condition
            task = asyncio.create_task(
                self._drive(run_id, source, condition, generation),
                name=f"spec-driven-run-{run_id[:8]}",
            )
            self._tasks[run_id] = task
            self._subscriber_counts[run_id] = 0
            self.store.set_subscriber_count(run_id, 0)
            if on_abandoned is not None:
                self._abandonment_policies[run_id] = _AbandonmentPolicy(
                    grace_seconds=max(0.0, float(disconnect_grace_seconds)),
                    cancel=on_abandoned,
                    resume_available=resume_available,
                )
                self._arm_abandonment_locked(run_id, generation)
            return record

    def _arm_abandonment_locked(self, run_id: str, generation: int) -> None:
        """Start/restart the grace timer. Caller must hold ``self._lock``."""
        previous = self._abandonment_tasks.pop(run_id, None)
        if previous is not None and not previous.done():
            previous.cancel()
        policy = self._abandonment_policies.get(run_id)
        producer = self._tasks.get(run_id)
        if (
            policy is None
            or self._generations.get(run_id) != generation
            or self._subscriber_counts.get(run_id, 0) > 0
            or producer is None
            or producer.done()
        ):
            return
        self._abandonment_tasks[run_id] = asyncio.create_task(
            self._cancel_if_still_abandoned(
                run_id, generation, policy.grace_seconds
            ),
            name=f"spec-driven-abandon-{run_id[:8]}",
        )

    async def _cancel_if_still_abandoned(
        self,
        run_id: str,
        generation: int,
        grace_seconds: float,
    ) -> None:
        try:
            await asyncio.sleep(grace_seconds)
        except asyncio.CancelledError:
            return

        callback: Optional[Callable[[], None]] = None
        async with self._lock:
            if self._abandonment_tasks.get(run_id) is not asyncio.current_task():
                return
            self._abandonment_tasks.pop(run_id, None)
            producer = self._tasks.get(run_id)
            policy = self._abandonment_policies.get(run_id)
            if (
                policy is None
                or self._generations.get(run_id) != generation
                or producer is None
                or producer.done()
                or self._subscriber_counts.get(run_id, 0) > 0
            ):
                return
            self._abandonment_triggered.add(run_id)
            self.store.mark_abandonment_requested(run_id)
            callback = policy.cancel

        if callback is None:  # pragma: no cover - guarded under the lock
            return
        try:
            callback()
        except Exception:
            logger.exception("Could not cancel abandoned run %s", run_id)

    async def _subscriber_joined(
        self, run_id: str, generation: Optional[int]
    ) -> bool:
        async with self._lock:
            if (
                generation is not None
                and self._generations.get(run_id) != generation
            ):
                return False
            count = self._subscriber_counts.get(run_id, 0) + 1
            self._subscriber_counts[run_id] = count
            timer = self._abandonment_tasks.pop(run_id, None)
            if timer is not None and not timer.done():
                timer.cancel()
            self.store.set_subscriber_count(run_id, count)
            return True

    async def _subscriber_left(
        self, run_id: str, generation: Optional[int]
    ) -> None:
        async with self._lock:
            if (
                generation is not None
                and self._generations.get(run_id) != generation
            ):
                return
            count = max(0, self._subscriber_counts.get(run_id, 0) - 1)
            self._subscriber_counts[run_id] = count
            self.store.set_subscriber_count(run_id, count)
            if count == 0:
                if run_id in self._tasks:
                    current_generation = self._generations.get(run_id)
                    if current_generation is not None:
                        self._arm_abandonment_locked(run_id, current_generation)
                else:
                    self._subscriber_counts.pop(run_id, None)

    async def _notify(self, condition: asyncio.Condition) -> None:
        async with condition:
            condition.notify_all()

    async def _drive(
        self,
        run_id: str,
        source: AsyncIterator[bytes],
        condition: asyncio.Condition,
        generation: int,
    ) -> None:
        terminal_status: Optional[RunStatus] = None
        terminal_event: Optional[str] = None
        terminal_error: Optional[str] = None
        self.store.set_status(run_id, "running")
        await self._notify(condition)
        try:
            async for frame in source:
                source_event_type, source_payload = _parse_frame(frame)
                abandoned = run_id in self._abandonment_triggered
                if (
                    abandoned
                    and source_event_type == "error"
                    and source_payload.get("code") == "CANCELLED"
                ):
                    policy = self._abandonment_policies.get(run_id)
                    can_resume = False
                    if policy is not None and policy.resume_available is not None:
                        try:
                            can_resume = bool(policy.resume_available())
                        except Exception:
                            logger.exception(
                                "Could not inspect checkpoint for abandoned run %s",
                                run_id,
                            )
                    self.store.set_resume_available(run_id, can_resume)
                    frame = format_sse(
                        ErrorEvent(
                            code="CANCELLED",
                            reason="abandoned",
                            resumeAvailable=can_resume,
                            message=(
                                "Generation stopped after no browser reconnected "
                                "within the configured grace period. "
                                + (
                                    "A verified checkpoint was retained."
                                    if can_resume
                                    else "No resumable checkpoint was produced; "
                                    "start a new generation."
                                )
                            ),
                        )
                    )
                stored = self.store.append(run_id, frame)
                event_type = stored.event_type
                payload = stored.payload
                if event_type == "done":
                    incomplete = (
                        payload.get("incomplete")
                        or int(payload.get("blockerCount") or 0) > 0
                    )
                    terminal_status = (
                        "abandoned"
                        if abandoned and incomplete
                        else "partial" if incomplete else "succeeded"
                    )
                    terminal_event = "done"
                    if abandoned:
                        policy = self._abandonment_policies.get(run_id)
                        can_resume = False
                        if policy is not None and policy.resume_available is not None:
                            try:
                                can_resume = bool(policy.resume_available())
                            except Exception:
                                logger.exception(
                                    "Could not inspect checkpoint for abandoned run %s",
                                    run_id,
                                )
                        self.store.set_resume_available(run_id, can_resume)
                elif event_type == "error":
                    code = str(payload.get("code") or "INTERNAL")
                    if code not in _NON_TERMINAL_ERROR_CODES:
                        if code == "CANCELLED":
                            terminal_status = "abandoned" if abandoned else "cancelled"
                            terminal_event = "ABANDONED" if abandoned else code
                        else:
                            terminal_status = "failed"
                            terminal_event = code
                        terminal_error = str(payload.get("message") or "") or None
                await self._notify(condition)

            if terminal_status is None:
                message = "Generation worker ended without a terminal result."
                stored = self.store.append(
                    run_id,
                    format_sse(ErrorEvent(code="INTERNAL", message=message)),
                )
                terminal_status = "failed"
                terminal_event = str(stored.payload.get("code") or "INTERNAL")
                terminal_error = message
        except asyncio.CancelledError:
            terminal_status = "interrupted"
            terminal_event = "interrupted"
            terminal_error = "Backend process stopped before the run completed."
            raise
        except Exception:  # belt-and-braces around producer adapters
            logger.exception("Durable producer failed for run %s", run_id)
            message = "Internal server error"
            try:
                self.store.append(
                    run_id,
                    format_sse(ErrorEvent(code="INTERNAL", message=message)),
                )
            except Exception:
                logger.exception("Could not persist terminal error for run %s", run_id)
            terminal_status = "failed"
            terminal_event = "INTERNAL"
            terminal_error = message
        finally:
            self.store.set_status(
                run_id,
                terminal_status or "failed",
                terminal_event=terminal_event,
                error=terminal_error,
            )
            await self._notify(condition)
            async with self._lock:
                if (
                    self._tasks.get(run_id) is asyncio.current_task()
                    and self._generations.get(run_id) == generation
                ):
                    self._tasks.pop(run_id, None)
                    timer = self._abandonment_tasks.pop(run_id, None)
                    if timer is not None and not timer.done():
                        timer.cancel()
                    self._abandonment_policies.pop(run_id, None)
                    self._conditions.pop(run_id, None)
                    self._abandonment_triggered.discard(run_id)
                    if self._subscriber_counts.get(run_id, 0) == 0:
                        self._subscriber_counts.pop(run_id, None)

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        return self.store.get_run(run_id)

    async def subscribe(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[bytes]:
        """Replay stored frames, then follow the live producer until terminal."""
        if self.store.get_run(run_id) is None:
            return
        generation = self._generations.get(run_id)
        joined = await self._subscriber_joined(run_id, generation)
        if not joined:
            return
        cursor = max(0, after_sequence)
        condition = self._conditions.get(run_id)
        if condition is None:
            # Replaying a record from a previous process needs no in-memory
            # producer condition.  A private condition still gives the loop a
            # uniform shape; interrupted/terminal records exit immediately.
            condition = asyncio.Condition()

        try:
            while True:
                events = self.store.events_after(run_id, cursor)
                for event in events:
                    cursor = event.sequence
                    yield event.frame

                record = self.store.get_run(run_id)
                if record is None:
                    return
                if (
                    record.status in TERMINAL_RUN_STATUSES
                    and cursor >= record.last_sequence
                ):
                    return

                # Close the race between the status/sequence check above and
                # waiting: re-check under the same condition the producer uses to
                # notify.  The timeout emits a standards-compliant SSE heartbeat
                # without adding noise to the durable event log.
                heartbeat_due = False
                async with condition:
                    current = self.store.get_run(run_id)
                    if current is None:
                        return
                    if current.last_sequence > cursor:
                        continue
                    if current.status in TERMINAL_RUN_STATUSES:
                        return
                    try:
                        await asyncio.wait_for(
                            condition.wait(), timeout=_HEARTBEAT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        heartbeat_due = True
                if heartbeat_due:
                    yield b": durable-run-heartbeat\n\n"
        finally:
            await self._subscriber_left(run_id, generation)

    async def wait(self, run_id: str) -> None:
        """Wait for an in-process producer. Useful for shutdown/tests."""
        async with self._lock:
            task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.shield(task)

    async def periodic_sweep(
        self,
        ttl_seconds: int,
        interval_seconds: int = 60,
    ) -> None:
        while True:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise
            try:
                removed = self.store.sweep_expired(ttl_seconds)
            except Exception:
                # A transient SQLite failure must not permanently disable TTL
                # cleanup for the lifetime of the backend process.
                logger.exception("Could not sweep durable spec-driven runs")
                continue
            if removed:
                logger.info("Swept %d expired durable spec-driven run(s)", removed)


DURABLE_RUN_MANAGER = DurableRunManager()


__all__ = [
    "DURABLE_RUN_MANAGER",
    "DurableRunManager",
    "RunRecord",
    "RunStatus",
    "SqliteRunEventStore",
    "StoredRunEvent",
    "TERMINAL_RUN_STATUSES",
]
