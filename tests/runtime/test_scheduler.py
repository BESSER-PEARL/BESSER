"""Tests for besser.runtime.scheduler.Scheduler.

The Scheduler integrates Bindings + Engine + HistoryStore.  These tests use
minimal stubs for all three so they can run without generating a platform.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from besser.runtime.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _Store:
    def __init__(self):
        self._rows: Dict[tuple, Dict[str, Any]] = {}

    def add(self, class_name, instance_id, attributes):
        self._rows[(class_name, instance_id)] = {
            "class_name": class_name, "id": instance_id, "attributes": dict(attributes),
        }

    def get_instance(self, class_name, instance_id):
        return self._rows[(class_name, instance_id)]

    def get_all_instances(self):
        return list(self._rows.values())

    def update_instance(self, class_name, instance_id, attributes):
        self._rows[(class_name, instance_id)]["attributes"] = dict(attributes)


class _Engine:
    """Records which instances were ticked."""
    def __init__(self, store: _Store):
        self._store = store
        self.calls: List[tuple] = []

    def tick(self, class_name: str, instance_id: str, dt: float, method_name: str = "step"):
        self.calls.append((class_name, instance_id, dt, method_name))
        inst = self._store.get_instance(class_name, instance_id)
        attrs = dict(inst["attributes"])
        # Increment 'counter' if present.
        if "counter" in attrs:
            attrs["counter"] = (attrs["counter"] or 0) + dt
        self._store.update_instance(class_name, instance_id, attrs)


class _NoOpBindings:
    def apply_pre_tick(self, tick_count, dt=1.0):
        pass


class _NoOpHistory:
    def __init__(self):
        self.snapshots: List[int] = []

    def snapshot(self, tick_count):
        self.snapshots.append(tick_count)

    def clear(self):
        self.snapshots.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store():
    s = _Store()
    s.add("Tank", "t1", {"counter": 0.0, "level": 0.0})
    return s


@pytest.fixture()
def engine(store):
    return _Engine(store)


@pytest.fixture()
def scheduler(engine, store):
    history = _NoOpHistory()
    bindings = _NoOpBindings()
    return Scheduler(
        engine=engine,
        history=history,
        bindings=bindings,
        instance_manager=store,
        steppable_classes={"Tank"},
        dt=1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_initial_state(self, scheduler):
        s = scheduler.state()
        assert s["running"] is False
        assert s["tick_count"] == 0
        assert s["dt"] == 1.0

    @pytest.mark.asyncio
    async def test_step_once_advances_tick(self, scheduler):
        result = await scheduler.step_once()
        assert result["tick_count"] == 1
        assert scheduler.tick_count == 1

    @pytest.mark.asyncio
    async def test_step_once_multiple_times(self, scheduler):
        for i in range(5):
            await scheduler.step_once()
        assert scheduler.tick_count == 5

    @pytest.mark.asyncio
    async def test_step_calls_engine_tick(self, scheduler, engine):
        await scheduler.step_once()
        assert len(engine.calls) == 1
        cls, iid, dt, method_name = engine.calls[0]
        assert cls == "Tank"
        assert iid == "t1"
        assert dt == pytest.approx(1.0)
        assert method_name == "step"

    @pytest.mark.asyncio
    async def test_start_and_pause(self, scheduler):
        await scheduler.start()
        assert scheduler.running is True
        # Let one tick fire.
        await asyncio.sleep(0.05)
        await scheduler.pause()
        assert scheduler.running is False
        # tick_count must have advanced at least once.
        assert scheduler.tick_count >= 1

    @pytest.mark.asyncio
    async def test_start_noop_if_already_running(self, scheduler):
        await scheduler.start()
        task1 = scheduler._task
        await scheduler.start()  # no-op
        assert scheduler._task is task1
        await scheduler.pause()

    @pytest.mark.asyncio
    async def test_pause_noop_if_stopped(self, scheduler):
        await scheduler.pause()  # already stopped — must not raise

    @pytest.mark.asyncio
    async def test_reset_clears_tick_count(self, scheduler):
        await scheduler.step_once()
        await scheduler.step_once()
        assert scheduler.tick_count == 2
        await scheduler.reset()
        assert scheduler.tick_count == 0
        assert scheduler.running is False

    @pytest.mark.asyncio
    async def test_step_does_not_affect_running_state(self, scheduler):
        # step_once while stopped keeps stopped=False
        assert not scheduler.running
        await scheduler.step_once()
        assert not scheduler.running

    @pytest.mark.asyncio
    async def test_history_snapshot_called_per_tick(self, scheduler):
        history = scheduler._history
        for _ in range(3):
            await scheduler.step_once()
        assert len(history.snapshots) == 3
        assert history.snapshots == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_listener_called_with_deltas(self, scheduler, store, engine):
        received = []

        async def _listener(tick_count, deltas):
            received.append((tick_count, deltas))

        scheduler.add_listener(_listener)
        await scheduler.step_once()
        # Engine increments 'counter', so a delta for 'counter' must have fired.
        assert len(received) == 1
        tick, deltas = received[0]
        assert tick == 0  # first tick fires at tick_count=0, then advances to 1
        counter_delta = next((d for d in deltas if d["attribute"] == "counter"), None)
        assert counter_delta is not None
        assert counter_delta["value"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_remove_listener(self, scheduler):
        received = []

        async def _l(tick, deltas):
            received.append(tick)

        scheduler.add_listener(_l)
        scheduler.remove_listener(_l)
        await scheduler.step_once()
        assert received == []

    @pytest.mark.asyncio
    async def test_steppable_only_dispatches_right_classes(self, engine, store):
        """Non-steppable classes must not receive a tick() call."""
        store.add("Pump", "p1", {"flowrate": 0.0})
        scheduler = Scheduler(
            engine=engine,
            history=_NoOpHistory(),
            bindings=_NoOpBindings(),
            instance_manager=store,
            steppable_classes={"Tank"},  # Pump is not steppable
            dt=1.0,
        )
        await scheduler.step_once()
        called_classes = {c for c, *_ in engine.calls}
        assert "Tank" in called_classes
        assert "Pump" not in called_classes
