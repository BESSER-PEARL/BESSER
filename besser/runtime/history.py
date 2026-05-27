"""HistoryStore — SQLite-backed time-series snapshot store for Digital Twin platforms.

Every tick the scheduler calls :meth:`HistoryStore.snapshot`, which writes one
row per primitive attribute per instance into a local SQLite database.  Frontend
charts query the ``/history`` endpoint, which in turn calls :meth:`HistoryStore.query`.

Schema
------
``ticks``    (tick_id INTEGER PK, ts REAL)        — one row per scheduler tick
``snapshots``(tick_id INTEGER, class_name TEXT,
              instance_id TEXT, attribute TEXT,
              value TEXT)                          — one row per attribute per tick

Values are stored as JSON-encoded text so that integers, floats, strings, and
booleans round-trip faithfully.

The database file defaults to ``dt_history.db`` in the working directory and can
be overridden via the ``BESSER_HISTORY_DB`` environment variable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


class HistoryStore:
    """Append-only SQLite store for per-tick instance attribute snapshots."""

    _CREATE_TICKS = """
    CREATE TABLE IF NOT EXISTS ticks (
        tick_id   INTEGER PRIMARY KEY,
        ts        REAL NOT NULL
    )
    """
    _CREATE_SNAPSHOTS = """
    CREATE TABLE IF NOT EXISTS snapshots (
        tick_id     INTEGER NOT NULL,
        class_name  TEXT NOT NULL,
        instance_id TEXT NOT NULL,
        attribute   TEXT NOT NULL,
        value       TEXT
    )
    """
    _INDEX_SNAPSHOTS = """
    CREATE INDEX IF NOT EXISTS idx_snap_lookup
        ON snapshots (class_name, instance_id, attribute, tick_id)
    """

    def __init__(self, instance_manager, db_path: str | Path | None = None):
        self._im = instance_manager
        if db_path is None:
            db_path = os.environ.get("BESSER_HISTORY_DB", "dt_history.db")
        self._db_path = str(db_path)
        # For `:memory:` databases, each call to `sqlite3.connect(":memory:")` opens
        # a fresh empty database, so we must reuse a single persistent connection.
        # For file-based databases, a per-operation connection is fine (WAL mode).
        self._persistent_conn: sqlite3.Connection | None = None
        if self._db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    # --------------------------------------------------------- init

    def _connect(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        if self._persistent_conn is not None:
            # In-memory: no context manager — commit manually.
            conn.execute(self._CREATE_TICKS)
            conn.execute(self._CREATE_SNAPSHOTS)
            conn.execute(self._INDEX_SNAPSHOTS)
            conn.commit()
        else:
            with conn:
                conn.execute(self._CREATE_TICKS)
                conn.execute(self._CREATE_SNAPSHOTS)
                conn.execute(self._INDEX_SNAPSHOTS)

    # --------------------------------------------------------- write

    def _execute(self, sql: str, params=(), many: bool = False) -> None:
        """Execute a write statement, committing in the right mode."""
        conn = self._connect()
        if self._persistent_conn is not None:
            if many:
                conn.executemany(sql, params)
            else:
                conn.execute(sql, params)
            conn.commit()
        else:
            with conn:
                if many:
                    conn.executemany(sql, params)
                else:
                    conn.execute(sql, params)

    def snapshot(self, tick_count: int) -> None:
        """Write a snapshot of every instance's primitive attributes for ``tick_count``."""
        ts = time.time()
        rows: list[tuple] = []

        for inst in self._im.get_all_instances():
            cn = inst["class_name"]
            iid = inst["id"]
            for attr, val in inst.get("attributes", {}).items():
                if isinstance(val, (int, float, str, bool)) or val is None:
                    rows.append((tick_count, cn, iid, attr, json.dumps(val)))

        self._execute("INSERT OR REPLACE INTO ticks (tick_id, ts) VALUES (?, ?)", (tick_count, ts))
        self._execute(
            "INSERT INTO snapshots (tick_id, class_name, instance_id, attribute, value) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
            many=True,
        )

    # --------------------------------------------------------- read

    def query(
        self,
        class_name: str,
        instance_id: str,
        attribute: str,
        since_tick: int = 0,
        limit: int = 1000,
    ) -> list[dict]:
        """Return up to ``limit`` snapshots for the given attribute slot.

        Each entry: ``{"tick_id": int, "ts": float, "value": Any}``
        """
        sql = """
            SELECT s.tick_id, t.ts, s.value
            FROM snapshots s
            JOIN ticks t ON s.tick_id = t.tick_id
            WHERE s.class_name = ?
              AND s.instance_id = ?
              AND s.attribute   = ?
              AND s.tick_id     >= ?
            ORDER BY s.tick_id ASC
            LIMIT ?
        """
        conn = self._connect()
        rows = conn.execute(sql, (class_name, instance_id, attribute, since_tick, limit)).fetchall()
        return [
            {"tick_id": r["tick_id"], "ts": r["ts"], "value": json.loads(r["value"])}
            for r in rows
        ]

    # --------------------------------------------------------- management

    def clear(self) -> None:
        """Delete all recorded ticks and snapshots (used by /reset)."""
        self._execute("DELETE FROM snapshots")
        self._execute("DELETE FROM ticks")

    def flush(self) -> None:
        """Ensure all pending writes are committed (called on shutdown)."""
        conn = self._connect()
        if self._persistent_conn is not None:
            conn.commit()
        else:
            with conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    @property
    def db_path(self) -> str:
        return self._db_path
