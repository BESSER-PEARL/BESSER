"""Tests for ``besser.generators.llm.tracing``.

The TraceWriter is append-only and best-effort — these tests lock
down the record shape and verify it doesn't raise on degraded
filesystems (e.g. read-only output dir).
"""

from __future__ import annotations

import json
import os

import pytest

from besser.generators.llm.tracing import (
    EVENT_PHASE_ENTER,
    EVENT_TOOL_CALL,
    EVENT_TURN_START,
    NullTraceWriter,
    TRACE_FILENAME,
    TraceWriter,
)


def _read_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_write_records_are_self_describing(tmp_path):
    writer = TraceWriter(str(tmp_path), run_id="abc123", primary_kind="class")
    writer.write(EVENT_PHASE_ENTER, phase="phase1")
    writer.write(EVENT_TURN_START, turn=1)
    writer.write(EVENT_TOOL_CALL, turn=1, tool="read_file", success=True)

    records = _read_records(writer.path)
    assert len(records) == 3
    for record in records:
        # Framing fields are stamped on every record so downstream
        # tooling never has to guess.
        assert record["run_id"] == "abc123"
        assert record["primary_kind"] == "class"
        assert "ts" in record
        assert "event" in record
        assert isinstance(record["payload"], dict)

    assert records[0]["event"] == EVENT_PHASE_ENTER
    assert records[1]["event"] == EVENT_TURN_START
    assert records[2]["payload"]["tool"] == "read_file"


def test_write_never_raises_on_bad_path():
    """Passing an invalid directory must degrade to a no-op, not crash."""
    writer = TraceWriter("/nonexistent\0/definitely/not/a/dir")
    # Should not raise; caller can keep running.
    writer.write(EVENT_TURN_START, turn=42)


def test_tail_returns_last_n_records(tmp_path):
    writer = TraceWriter(str(tmp_path), run_id="r", primary_kind="class")
    for i in range(10):
        writer.write(EVENT_TURN_START, turn=i)

    tail = writer.tail(limit=3)
    assert len(tail) == 3
    # Last three turns should be 7, 8, 9 in order
    assert [r["payload"]["turn"] for r in tail] == [7, 8, 9]


def test_tail_skips_corrupt_lines(tmp_path):
    """A partial write that leaves a malformed last line must not
    poison the whole tail — we skip it and return what parses.
    """
    path = tmp_path / TRACE_FILENAME
    good1 = '{"event": "turn_start", "payload": {"turn": 1}, "ts": 1, "run_id": "r", "primary_kind": "class"}'
    good2 = '{"event": "turn_start", "payload": {"turn": 2}, "ts": 2, "run_id": "r", "primary_kind": "class"}'
    malformed = '{"event": "turn_start", "payload": {"turn": 3'   # truncated, no closing braces
    path.write_text(good1 + "\n" + good2 + "\n" + malformed + "\n", encoding="utf-8")

    writer = TraceWriter(str(tmp_path), run_id="r", primary_kind="class")
    tail = writer.tail(limit=5)
    # The two complete lines should come through; the half-written one is skipped.
    assert len(tail) == 2


def test_tail_returns_empty_when_no_file(tmp_path):
    writer = TraceWriter(str(tmp_path))
    assert writer.tail() == []


def test_null_writer_is_interchangeable(tmp_path):
    """NullTraceWriter mirrors the public API so production code can
    swap in a no-op without conditional branches.
    """
    null = NullTraceWriter()
    null.write(EVENT_TURN_START, turn=1)   # no file produced
    assert null.tail() == []
    assert null.path == ""
    # Directory must remain empty
    assert os.listdir(tmp_path) == []
