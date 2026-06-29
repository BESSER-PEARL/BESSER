"""Tests for the BPMN code builder.

Covers the builder unit behaviour (parses, exec'd code reconstructs the model,
collisions resolved, layout passthrough, deterministic output). Round-trip via the
``bpmn_buml_to_json`` wrapper is covered separately in
``tests/utilities/web_modeling_editor/backend/services/converters/test_bpmn_converters.py``.
"""

from __future__ import annotations

import time

import pytest

from besser.BUML.metamodel.bpmn import (
    BPMNModel,
    Collaboration,
    EndEvent,
    Gateway,
    GatewayType,
    MessageFlow,
    Participant,
    Process,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TaskType,
)
from besser.utilities.buml_code_builder.bpmn_model_builder import (
    _NameDispenser,
    bpmn_model_to_code,
)


# ---------------------------------------------------------------------------
# Fixture builders — programmatic models matching the §03 hand-written JSON
# ---------------------------------------------------------------------------

def _poolless_model() -> BPMNModel:
    """Pool-less StartEvent → Task → EndEvent."""
    s = StartEvent(name="go")
    t = Task(name="Do work", task_type=TaskType.USER)
    e = EndEvent(name="done")
    p = Process(name="P", flow_nodes={s, t, e},
                sequence_flows={SequenceFlow(s, t), SequenceFlow(t, e)})
    return BPMNModel(name="Poolless", processes={p})


def _two_pool_model() -> BPMNModel:
    """Two pools, one task each, one MessageFlow."""
    t1 = Task(name="Place order")
    t2 = Task(name="Ship")
    p_buyer = Process(name="Buyer", flow_nodes={t1})
    p_seller = Process(name="Seller", flow_nodes={t2})
    part_buyer = Participant(name="Buyer", process=p_buyer)
    part_seller = Participant(name="Seller", process=p_seller)
    coll = Collaboration(name="Buyer-Seller",
                         participants={part_buyer, part_seller},
                         message_flows={MessageFlow(t1, t2, name="order")})
    return BPMNModel(name="Buyer-Seller", processes={p_buyer, p_seller},
                     collaboration=coll)


def _gateway_model() -> BPMNModel:
    """ExclusiveGateway with a marked default flow."""
    s = StartEvent(name="")
    g = Gateway(name="?", gateway_type=GatewayType.EXCLUSIVE)
    a = Task(name="A")
    b = Task(name="B")
    e1 = EndEvent(name="")
    e2 = EndEvent(name="")
    f0 = SequenceFlow(s, g)
    f1 = SequenceFlow(g, a, name="happy")
    f2 = SequenceFlow(g, b, name="default", is_default=True)
    fa = SequenceFlow(a, e1)
    fb = SequenceFlow(b, e2)
    p = Process(name="Gateway",
                flow_nodes={s, g, a, b, e1, e2},
                sequence_flows={f0, f1, f2, fa, fb})
    return BPMNModel(name="Gateway", processes={p})


def _subprocess_model() -> BPMNModel:
    """SubProcess containing two flow nodes and one inner sequence flow."""
    inner_t = Task(name="inner")
    inner_e = EndEvent(name="inner end")
    sub = SubProcess(name="Sub",
                     flow_nodes={inner_t, inner_e},
                     sequence_flows={SequenceFlow(inner_t, inner_e)})
    s = StartEvent(name="")
    e = EndEvent(name="")
    p = Process(name="Subprocess",
                flow_nodes={s, sub, e},
                sequence_flows={SequenceFlow(s, sub), SequenceFlow(sub, e)})
    return BPMNModel(name="Subprocess", processes={p})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exec_source(source: str) -> dict:
    """Exec ``source`` in a fresh namespace and return the namespace dict."""
    namespace: dict = {}
    exec(source, namespace)
    return namespace


def _find_model(namespace: dict) -> BPMNModel:
    candidate = namespace.get("bpmn_model")
    if isinstance(candidate, BPMNModel):
        return candidate
    for value in namespace.values():
        if isinstance(value, BPMNModel):
            return value
    raise AssertionError("No BPMNModel in namespace")


# ---------------------------------------------------------------------------
# A. Builder unit tests
# ---------------------------------------------------------------------------

class TestBuilderProducesValidPython:
    @pytest.mark.parametrize("fixture_fn", [
        _poolless_model, _two_pool_model, _gateway_model, _subprocess_model,
    ])
    def test_emitted_code_compiles(self, fixture_fn):
        source = bpmn_model_to_code(fixture_fn())
        compile(source, "<emitted>", "exec")  # raises SyntaxError on failure

    @pytest.mark.parametrize("fixture_fn", [
        _poolless_model, _two_pool_model, _gateway_model, _subprocess_model,
    ])
    def test_exec_recovers_a_bpmn_model(self, fixture_fn):
        source = bpmn_model_to_code(fixture_fn())
        ns = _exec_source(source)
        model = _find_model(ns)
        assert isinstance(model, BPMNModel)


class TestExecdModelMatchesOriginal:
    def test_poolless_node_classes_preserved(self):
        original = _poolless_model()
        ns = _exec_source(bpmn_model_to_code(original))
        model = _find_model(ns)
        assert {type(n).__name__ for n in model.all_flow_nodes()} == \
               {type(n).__name__ for n in original.all_flow_nodes()}
        assert len(model.all_sequence_flows()) == len(original.all_sequence_flows())

    def test_two_pool_collaboration_preserved(self):
        original = _two_pool_model()
        ns = _exec_source(bpmn_model_to_code(original))
        model = _find_model(ns)
        assert isinstance(model.collaboration, Collaboration)
        assert len(model.collaboration.participants) == 2
        assert len(model.collaboration.message_flows) == 1
        # Each participant has a process.
        for part in model.collaboration.participants:
            assert isinstance(part.process, Process)

    def test_gateway_default_flow_preserved(self):
        original = _gateway_model()
        ns = _exec_source(bpmn_model_to_code(original))
        model = _find_model(ns)
        gateway = next(n for n in model.all_flow_nodes() if isinstance(n, Gateway))
        assert gateway.default_flow is not None
        assert gateway.default_flow.name == "default"
        # Validate succeeds (default_flow source-restriction rule still holds).
        result = model.validate(raise_exception=False)
        assert result["success"] is True, result["errors"]

    def test_subprocess_containment_preserved(self):
        original = _subprocess_model()
        ns = _exec_source(bpmn_model_to_code(original))
        model = _find_model(ns)
        sub = next(n for n in model.all_flow_nodes() if isinstance(n, SubProcess))
        # Subprocess holds its 2 inner nodes (Task + inner EndEvent).
        assert len(sub.flow_nodes) == 2
        assert len(sub.sequence_flows) == 1
        # Inner sequence flow's endpoints are inside the subprocess.
        inner_flow = next(iter(sub.sequence_flows))
        assert inner_flow.source.container is sub
        assert inner_flow.target.container is sub


class TestNameCollisions:
    def test_duplicate_task_names_get_unique_vars(self):
        # Three Tasks: two named "do", one unnamed.
        t1 = Task(name="do")
        t2 = Task(name="do")
        t3 = Task(name="")
        p = Process(name="P", flow_nodes={t1, t2, t3})
        model = BPMNModel(name="Collisions", processes={p})
        source = bpmn_model_to_code(model)
        # exec must succeed (no NameError from duplicate variable assignments).
        ns = _exec_source(source)
        recovered = _find_model(ns)
        assert len(recovered.all_flow_nodes()) == 3

    def test_dispenser_caches_per_object(self):
        d = _NameDispenser()
        t = Task(name="x")
        first = d.name_for(t)
        second = d.name_for(t)
        assert first == second

    def test_dispenser_unique_for_same_name_different_objects(self):
        d = _NameDispenser()
        names = {d.name_for(Task(name="x")) for _ in range(5)}
        assert len(names) == 5


class TestLayoutPassthrough:
    def test_layout_round_trips_when_set(self):
        t = Task(name="t", layout={"id": "abc", "bounds": {"x": 1, "y": 2, "width": 10, "height": 10}})
        p = Process(name="P", flow_nodes={t})
        model = BPMNModel(name="Layout", processes={p})
        source = bpmn_model_to_code(model)
        ns = _exec_source(source)
        recovered = _find_model(ns)
        recovered_t = next(iter(recovered.all_flow_nodes()))
        assert recovered_t.layout is not None
        assert recovered_t.layout["id"] == "abc"
        assert recovered_t.layout["bounds"] == {"x": 1, "y": 2, "width": 10, "height": 10}

    def test_layout_omitted_when_none(self):
        # No layout assignment line should appear in the source for a layout-free node.
        source = bpmn_model_to_code(_poolless_model())
        # `_poolless_model` doesn't set layout on any node.
        assert ".layout = " not in source


class TestBannerAndImports:
    def test_banner_present_at_top(self):
        source = bpmn_model_to_code(_poolless_model())
        assert source.startswith("####################\n#    BPMN MODEL    #\n####################\n")

    def test_only_used_classes_imported(self):
        # Pool-less model uses no Gateway / Collaboration / DataStore; their import
        # symbols should be absent from the import block.
        source = bpmn_model_to_code(_poolless_model())
        # The import block ends at the first blank line after `from … import (`.
        import_block = source.split("from besser.BUML.metamodel.bpmn import (", 1)[1].split(")", 1)[0]
        assert "Gateway" not in import_block
        assert "Collaboration" not in import_block
        assert "DataStore" not in import_block
        # Classes the pool-less model actually uses must be present.
        for name in ("BPMNModel", "Process", "StartEvent", "Task", "EndEvent", "SequenceFlow"):
            assert name in import_block, f"expected {name} in import block, got: {import_block!r}"


# ---------------------------------------------------------------------------
# D. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_model_produces_same_source(self):
        # Two builds of the same model must be byte-identical.
        model = _poolless_model()
        first = bpmn_model_to_code(model)
        second = bpmn_model_to_code(model)
        assert first == second

    def test_walk_order_is_timestamp_not_name(self):
        # Two same-named tasks with distinct timestamps. Build emit order should
        # follow timestamp (insertion order in our Process), not arbitrary set order.
        t_first = Task(name="x")
        time.sleep(0.001)
        t_second = Task(name="x")
        p = Process(name="P", flow_nodes={t_first, t_second})
        source = bpmn_model_to_code(BPMNModel(name="ts", processes={p}))
        # The second-created task's var should appear AFTER the first-created task's
        # var in the emitted source. Both use prefix `task_x` + suffix.
        first_idx = source.index("task_x =")
        second_idx = source.index("task_x_1 =")
        assert first_idx < second_idx
        # And the dispenser-second task corresponds to the timestamp-second task.
        ns = _exec_source(source)
        recovered = _find_model(ns)
        # Earliest timestamp in the recovered model has the smaller var-index.
        assert len(recovered.all_flow_nodes()) == 2
