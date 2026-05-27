"""Tests for besser.runtime.history.HistoryStore."""
from __future__ import annotations

import tempfile
import time

import pytest

from besser.runtime.history import HistoryStore


# ---------------------------------------------------------------------------
# Minimal instance manager stub
# ---------------------------------------------------------------------------
class _Store:
    def __init__(self, instances):
        self._instances = instances

    def get_all_instances(self):
        return list(self._instances)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHistoryStore:
    @pytest.fixture()
    def db_path(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return f.name

    @pytest.fixture()
    def store(self):
        return _Store([
            {"class_name": "Tank", "id": "t1", "attributes": {"level": 5.0, "name": "T1"}},
            {"class_name": "Pump", "id": "p1", "attributes": {"flowrate": 2.0}},
        ])

    @pytest.fixture()
    def history(self, store, db_path):
        return HistoryStore(instance_manager=store, db_path=db_path)

    def test_snapshot_writes_rows(self, history):
        history.snapshot(0)
        rows = history.query("Tank", "t1", "level")
        assert len(rows) == 1
        assert rows[0]["tick_id"] == 0
        assert rows[0]["value"] == pytest.approx(5.0)

    def test_query_returns_in_tick_order(self, history):
        for i in range(5):
            history.snapshot(i)
        rows = history.query("Pump", "p1", "flowrate", limit=100)
        assert [r["tick_id"] for r in rows] == list(range(5))

    def test_since_tick_filters_earlier_rows(self, history):
        for i in range(10):
            history.snapshot(i)
        rows = history.query("Tank", "t1", "level", since_tick=7)
        assert all(r["tick_id"] >= 7 for r in rows)
        assert len(rows) == 3  # 7, 8, 9

    def test_limit_caps_rows(self, history):
        for i in range(20):
            history.snapshot(i)
        rows = history.query("Tank", "t1", "level", limit=5)
        assert len(rows) == 5

    def test_clear_removes_all(self, history):
        history.snapshot(0)
        history.snapshot(1)
        history.clear()
        rows = history.query("Tank", "t1", "level")
        assert rows == []

    def test_snapshot_only_stores_primitives(self, history, store):
        """Non-primitive attribute values (lists, dicts) must be skipped."""
        complex_store = _Store([
            {"class_name": "X", "id": "x1", "attributes": {
                "val": 42.0,
                "nested": {"a": 1},   # dict — should be skipped
                "tags": [1, 2],       # list — should be skipped
            }}
        ])
        h = HistoryStore(instance_manager=complex_store, db_path=":memory:")
        h.snapshot(0)
        rows = h.query("X", "x1", "val")
        assert len(rows) == 1
        assert rows[0]["value"] == 42.0
        # dict/list attrs should not appear
        assert h.query("X", "x1", "nested") == []
        assert h.query("X", "x1", "tags") == []

    def test_flush_does_not_raise(self, history):
        history.snapshot(0)
        history.flush()  # must not raise

    def test_in_memory_db(self, store):
        """HistoryStore works with ':memory:' for test isolation."""
        h = HistoryStore(instance_manager=store, db_path=":memory:")
        h.snapshot(0)
        rows = h.query("Tank", "t1", "level")
        assert len(rows) == 1
