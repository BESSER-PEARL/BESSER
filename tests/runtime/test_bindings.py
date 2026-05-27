"""Tests for besser.runtime.bindings.InputBindings."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from besser.runtime.bindings import InputBindings


# ---------------------------------------------------------------------------
# Minimal in-memory instance manager stub
# ---------------------------------------------------------------------------
class _Store:
    def __init__(self):
        self._rows = {}

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


@pytest.fixture()
def store():
    s = _Store()
    s.add("Pump", "p1", {"flowrate": 0.0})
    return s


@pytest.fixture()
def bindings(store):
    return InputBindings(instance_manager=store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBindingCRUD:
    def test_set_and_get(self, bindings):
        bindings.set_binding("Pump", "p1", "flowrate", "constant", {"value": 5.0})
        b = bindings.get_binding("Pump", "p1", "flowrate")
        assert b is not None
        assert b["kind"] == "constant"
        assert b["params"]["value"] == 5.0

    def test_clear(self, bindings):
        bindings.set_binding("Pump", "p1", "flowrate", "constant", {"value": 1.0})
        bindings.clear_binding("Pump", "p1", "flowrate")
        assert bindings.get_binding("Pump", "p1", "flowrate") is None

    def test_clear_nonexistent_is_noop(self, bindings):
        bindings.clear_binding("Pump", "p1", "nonexistent")  # must not raise

    def test_list_bindings_for_instance(self, bindings):
        bindings.set_binding("Pump", "p1", "flowrate", "sine", {"amplitude": 2})
        result = bindings.list_bindings_for_instance("Pump", "p1")
        assert len(result) == 1
        assert result[0]["attribute"] == "flowrate"
        assert result[0]["kind"] == "sine"

    def test_invalid_kind_raises(self, bindings):
        with pytest.raises(ValueError, match="Unknown binding kind"):
            bindings.set_binding("Pump", "p1", "flowrate", "kafka", {})


class TestSimulators:
    def test_constant(self, bindings, store):
        bindings.set_binding("Pump", "p1", "flowrate", "constant", {"value": 3.14})
        bindings.apply_pre_tick(0, dt=1.0)
        assert store.get_instance("Pump", "p1")["attributes"]["flowrate"] == pytest.approx(3.14)

    def test_sine_at_zero(self, bindings, store):
        # sin(0) = 0 → offset + 0 = offset
        bindings.set_binding("Pump", "p1", "flowrate", "sine",
                             {"amplitude": 1.0, "frequency": 1.0, "offset": 5.0})
        bindings.apply_pre_tick(0, dt=1.0)
        v = store.get_instance("Pump", "p1")["attributes"]["flowrate"]
        assert v == pytest.approx(5.0)

    def test_sine_at_quarter_period(self, bindings, store):
        # t = tick * dt = 4 * 0.25 = 1.0; freq=1 → t=1 → sin(2π*1*1)=0
        # t = 0.25, freq=1 → sin(2π*0.25)=sin(π/2)=1 → offset+amplitude*1
        bindings.set_binding("Pump", "p1", "flowrate", "sine",
                             {"amplitude": 2.0, "frequency": 1.0, "offset": 10.0})
        # tick=1, dt=0.25 → t=0.25 → value=10+2*sin(2π*0.25)=10+2=12
        bindings.apply_pre_tick(1, dt=0.25)
        v = store.get_instance("Pump", "p1")["attributes"]["flowrate"]
        assert v == pytest.approx(12.0, abs=1e-6)

    def test_ramp_increases(self, bindings, store):
        bindings.set_binding("Pump", "p1", "flowrate", "ramp",
                             {"start": 0.0, "rate": 2.0, "max_value": None})
        bindings.apply_pre_tick(5, dt=1.0)  # t=5*1=5 → 0+2*5=10
        v = store.get_instance("Pump", "p1")["attributes"]["flowrate"]
        assert v == pytest.approx(10.0)

    def test_ramp_clamped(self, bindings, store):
        bindings.set_binding("Pump", "p1", "flowrate", "ramp",
                             {"start": 0.0, "rate": 10.0, "max_value": 3.0})
        bindings.apply_pre_tick(100, dt=1.0)
        v = store.get_instance("Pump", "p1")["attributes"]["flowrate"]
        assert v == pytest.approx(3.0)

    def test_replay_csv_cycles(self, bindings, store):
        bindings.load_csv("src1", [10.0, 20.0, 30.0])
        bindings.set_binding("Pump", "p1", "flowrate", "replay_csv", {"source_id": "src1"})
        for i, expected in enumerate([10.0, 20.0, 30.0, 10.0, 20.0]):
            bindings.apply_pre_tick(i, dt=1.0)
            v = store.get_instance("Pump", "p1")["attributes"]["flowrate"]
            assert v == pytest.approx(expected), f"tick {i}: expected {expected} got {v}"


class TestPersistence:
    def test_save_and_load_roundtrip(self, bindings, store):
        bindings.set_binding("Pump", "p1", "flowrate", "sine",
                             {"amplitude": 1.5, "frequency": 0.5, "offset": 2.0})
        bindings.load_csv("csv1", [1.0, 2.0])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        bindings.save(path)
        # Load into a new InputBindings instance.
        fresh = InputBindings(instance_manager=store)
        fresh.load(path)
        b = fresh.get_binding("Pump", "p1", "flowrate")
        assert b is not None
        assert b["kind"] == "sine"
        assert b["params"]["amplitude"] == pytest.approx(1.5)

    def test_load_nonexistent_is_noop(self, store):
        b = InputBindings(instance_manager=store)
        b.load("/nonexistent/path/dt_bindings.json")  # must not raise

    def test_binding_visible_after_next_tick(self, bindings, store):
        """Mutations from apply_pre_tick are visible in the store immediately."""
        bindings.set_binding("Pump", "p1", "flowrate", "constant", {"value": 7.0})
        bindings.apply_pre_tick(0, dt=1.0)
        assert store.get_instance("Pump", "p1")["attributes"]["flowrate"] == pytest.approx(7.0)
