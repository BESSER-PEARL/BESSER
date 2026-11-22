"""A finished run whose events were swept answers 410 on both events endpoints.

The sweep deletes a finished run's events after the download TTL but keeps its
small row for 24 h (so a resume keeps numbering). In that window both events
endpoints answered 200 with nothing in it, and the editor's stream client, which
stops only on 404/410/422, retried for minutes.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3

from besser.utilities.web_modeling_editor.backend.services.spec_driven.run_manager import (
    SqliteRunEventStore,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
    DoneEvent,
    StartEvent,
    format_sse,
)

_TTL_SECONDS = 1800
_RUNS = "/besser_api/spec-driven/runs"


def _finished_run(store, run_id: str) -> int:
    """Store a two-event run that succeeded; return its last sequence."""
    store.begin_run(run_id)
    store.append(
        run_id,
        format_sse(StartEvent(runId=run_id, provider="openai", llmModel="m", maxCost=1.0, maxRuntime=60)),
    )
    store.append(
        run_id,
        format_sse(DoneEvent(runId=run_id, downloadUrl="/d", fileName="r.zip", isZip=True)),
    )
    store.set_status(run_id, "succeeded", terminal_event="done")
    return store.get_run(run_id).last_sequence


def _sweep_events(store, run_id: str) -> None:
    """Age the run past the events TTL (not the record retention) and sweep."""
    with store._conn:
        store._conn.execute(
            "UPDATE spec_runs SET updated_at = updated_at - ? WHERE run_id = ?",
            (_TTL_SECONDS + 60, run_id),
        )
    assert store.sweep_expired(_TTL_SECONDS) == 1
    assert store.get_run(run_id) is not None, "the run row must outlive its events"


def test_swept_run_answers_410_on_both_events_endpoints(client, spec_driven_runtime):
    store = spec_driven_runtime.durable_runs.store
    run_id = "a1" * 16
    _finished_run(store, run_id)
    _sweep_events(store, run_id)

    assert client.get(f"{_RUNS}/{run_id}/events.json").status_code == 410
    assert client.get(f"{_RUNS}/{run_id}/events").status_code == 410
    assert client.get(f"{_RUNS}/{run_id}/events", headers={"Last-Event-ID": "2"}).status_code == 410
    status = client.get(f"{_RUNS}/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"


def test_unswept_run_with_a_cursor_at_its_end_still_answers_200(client, spec_driven_runtime):
    store = spec_driven_runtime.durable_runs.store
    run_id = "b2" * 16
    last = _finished_run(store, run_id)

    polled = client.get(f"{_RUNS}/{run_id}/events.json", params={"after": last})
    assert polled.status_code == 200
    assert polled.json()["events"] == []
    streamed = client.get(f"{_RUNS}/{run_id}/events", params={"after": last})
    assert streamed.status_code == 200
    assert streamed.content == b""


def test_terminal_run_that_never_had_events_still_answers_200(client, spec_driven_runtime):
    store = spec_driven_runtime.durable_runs.store
    run_id = "c3" * 16
    store.begin_run(run_id)
    store.set_status(run_id, "failed", terminal_event="INTERNAL")

    assert client.get(f"{_RUNS}/{run_id}/events.json").status_code == 200
    assert client.get(f"{_RUNS}/{run_id}/events").status_code == 200


def test_resumed_run_stopped_before_its_first_frame_is_not_gone(client, spec_driven_runtime):
    """A resume drops the old frames but keeps ``last_sequence``; a shutdown
    before the new attempt's first frame left a finished run with no events
    that the sweep never touched, and it answered 410 instead of 200."""
    manager = spec_driven_runtime.durable_runs
    store = manager.store
    run_id = "e5" * 16
    _finished_run(store, run_id)
    store.set_status(run_id, "interrupted", terminal_event="interrupted")

    async def exercise() -> None:
        async def source():
            await asyncio.sleep(3600)
            yield b""

        await manager.start(run_id, source(), resume=True)
        task = manager._tasks[run_id]
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    record = store.get_run(run_id)
    assert record.status == "interrupted" and record.last_sequence == 2
    assert store.events_after(run_id, 0) == []

    polled = client.get(f"{_RUNS}/{run_id}/events.json")
    assert polled.status_code == 200
    assert polled.json()["status"] == "interrupted"
    assert client.get(f"{_RUNS}/{run_id}/events").status_code == 200


def test_resume_of_a_swept_run_is_no_longer_gone(spec_driven_runtime):
    store = spec_driven_runtime.durable_runs.store
    run_id = "f6" * 16
    _finished_run(store, run_id)
    _sweep_events(store, run_id)
    assert store.events_swept(run_id)

    store.begin_run(run_id, resume=True)
    assert not store.events_swept(run_id)


# The ``spec_runs`` table as releases before the sweep flag created it.
_LEGACY_SCHEMA = """
CREATE TABLE spec_runs (
    run_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL,
    updated_at REAL NOT NULL, last_sequence INTEGER NOT NULL DEFAULT 0,
    terminal_event TEXT, error TEXT, subscriber_count INTEGER NOT NULL DEFAULT 0,
    last_subscriber_at REAL, disconnected_at REAL, abandonment_requested_at REAL,
    resume_available INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE spec_run_events (
    run_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_type TEXT NOT NULL,
    frame BLOB NOT NULL, payload_json TEXT NOT NULL, created_at REAL NOT NULL,
    PRIMARY KEY (run_id, sequence),
    FOREIGN KEY (run_id) REFERENCES spec_runs(run_id) ON DELETE CASCADE
);
"""


def test_a_store_from_before_the_sweep_flag_opens_and_keeps_its_swept_runs_gone(tmp_path):
    path = str(tmp_path / "runs.sqlite3")
    swept, kept = "a7" * 16, "b8" * 16
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(_LEGACY_SCHEMA)
        conn.executemany(
            "INSERT INTO spec_runs(run_id, status, created_at, updated_at, last_sequence) "
            "VALUES (?, 'succeeded', 0, 0, 2)",
            [(swept,), (kept,)],
        )
        conn.execute(
            "INSERT INTO spec_run_events VALUES (?, 2, 'done', x'00', '{}', 0)", (kept,)
        )

    store = SqliteRunEventStore(path)
    try:
        assert store.events_swept(swept)
        assert not store.events_swept(kept)
        store.begin_run("c9" * 16)
        assert not store.events_swept("c9" * 16)
    finally:
        store.close()
