"""Tests for the SEAA'25 agentic extensions in the BPMN code builder.

Mirrors ``test_bpmn_model_builder.py`` but exercises the
``AgenticTask`` / ``AgenticGateway`` / ``AgenticLane`` subclass branches
of ``_emit_flow_node`` and ``_emit_lane``. Covers:

* Single-class round-trips (B-1, B-3, B-4, B-6, B-7).
* Import-block correctness per subclass (B-2, B-5, B-8).
* Mixed-class emission (B-9) and absence of agentic imports for vanilla
  models (B-10).
* Layout passthrough (B-11) and determinism (B-12).
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    AgentRole,
    BPMNModel,
    CollaborationMode,
    EndEvent,
    Gateway,
    GatewayRole,
    GatewayType,
    MergingStrategy,
    Process,
    ReflectionMode,
    SequenceFlow,
    StartEvent,
    Task,
    TaskType,
)
from besser.utilities.buml_code_builder.bpmn_model_builder import (
    bpmn_model_to_code,
)


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


def _import_block(source: str) -> str:
    """Extract the inside of the ``from besser.BUML.metamodel.bpmn import (...)`` block."""
    return source.split(
        "from besser.BUML.metamodel.bpmn import (", 1
    )[1].split(")", 1)[0]


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _agentic_task_model() -> BPMNModel:
    """Single AgenticTask with non-default reflection_mode + trust_score."""
    t = AgenticTask(
        name="Review",
        task_type=TaskType.USER,
        reflection_mode=ReflectionMode.CROSS,
        trust_score=85,
    )
    p = Process(name="P", flow_nodes={t})
    return BPMNModel(name="AgenticTaskModel", processes={p})


def _agentic_gateway_merging_model(
    collaboration_mode: CollaborationMode,
    merging_strategy: MergingStrategy,
) -> BPMNModel:
    """Merging AgenticGateway with given collaboration_mode + merging_strategy."""
    g = AgenticGateway(
        name="Merge",
        gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.MERGING,
        collaboration_mode=collaboration_mode,
        merging_strategy=merging_strategy,
        trust_score=42,
    )
    p = Process(name="P", flow_nodes={g})
    return BPMNModel(name="AgenticGatewayMergingModel", processes={p})


def _agentic_gateway_diverging_model() -> BPMNModel:
    """Diverging AgenticGateway -- merging_strategy must be None."""
    g = AgenticGateway(
        name="Fork",
        gateway_type=GatewayType.INCLUSIVE,
        gateway_role=GatewayRole.DIVERGING,
        collaboration_mode=CollaborationMode.VOTING,
        trust_score=10,
    )
    p = Process(name="P", flow_nodes={g})
    return BPMNModel(name="AgenticGatewayDivergingModel", processes={p})


def _agentic_lane_model() -> BPMNModel:
    """Lane-only model with a single AgenticLane wrapping one Task."""
    t = Task(name="Code")
    lane = AgenticLane(
        name="Reviewer",
        role=AgentRole.MANAGER,
        trust_score=75,
        flow_nodes={t},
    )
    p = Process(name="P", flow_nodes={t}, lanes={lane})
    return BPMNModel(name="AgenticLaneModel", processes={p})


def _mixed_model() -> BPMNModel:
    """Mix of an AgenticTask and a base Task + an AgenticGateway and a base Gateway."""
    a_task = AgenticTask(
        name="Agentic",
        reflection_mode=ReflectionMode.SELF,
        trust_score=50,
    )
    b_task = Task(name="Vanilla", task_type=TaskType.USER)
    a_gw = AgenticGateway(
        name="ParaFork",
        gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.DIVERGING,
        collaboration_mode=CollaborationMode.VOTING,
    )
    b_gw = Gateway(name="ExFork", gateway_type=GatewayType.EXCLUSIVE)
    p = Process(name="P", flow_nodes={a_task, b_task, a_gw, b_gw})
    return BPMNModel(name="Mixed", processes={p})


def _vanilla_model() -> BPMNModel:
    """No agentic classes anywhere -- StartEvent -> Task -> EndEvent."""
    s = StartEvent(name="go")
    t = Task(name="Do")
    e = EndEvent(name="done")
    p = Process(name="P", flow_nodes={s, t, e},
                sequence_flows={SequenceFlow(s, t), SequenceFlow(t, e)})
    return BPMNModel(name="Vanilla", processes={p})


# ---------------------------------------------------------------------------
# B-1 / B-2 -- AgenticTask
# ---------------------------------------------------------------------------

class TestAgenticTask:
    def test_emit_agentic_task_round_trips(self):  # B-1
        original = _agentic_task_model()
        source = bpmn_model_to_code(original)
        ns = _exec_source(source)
        model = _find_model(ns)
        nodes = list(model.all_flow_nodes())
        assert len(nodes) == 1
        node = nodes[0]
        assert isinstance(node, AgenticTask)
        assert node.name == "Review"
        assert node.task_type == TaskType.USER
        assert node.reflection_mode == ReflectionMode.CROSS
        assert node.trust_score == 85

    def test_emit_agentic_task_imports(self):  # B-2
        source = bpmn_model_to_code(_agentic_task_model())
        block = _import_block(source)
        for name in ("AgenticTask", "ReflectionMode", "TaskType",
                     "LoopCharacteristics"):
            assert name in block, f"expected {name} in import block, got: {block!r}"
        # The Task class itself is shadowed by AgenticTask -- no plain `Task,` line
        # should appear in the import list.
        assert "Task," not in block.replace("AgenticTask,", "")


# ---------------------------------------------------------------------------
# B-3 / B-4 / B-5 / B-6 -- AgenticGateway
# ---------------------------------------------------------------------------

class TestAgenticGateway:
    @pytest.mark.parametrize("mode,strategy", [
        (CollaborationMode.VOTING, MergingStrategy.MAJORITY),
        (CollaborationMode.VOTING, MergingStrategy.ABSOLUTE_MAJORITY),
        (CollaborationMode.ROLE, MergingStrategy.LEADER_DRIVEN),
        (CollaborationMode.COMPETITION, MergingStrategy.FASTEST),
    ])
    def test_emit_agentic_gateway_merging_round_trips(self, mode, strategy):  # B-3
        original = _agentic_gateway_merging_model(mode, strategy)
        source = bpmn_model_to_code(original)
        ns = _exec_source(source)
        model = _find_model(ns)
        node = next(iter(model.all_flow_nodes()))
        assert isinstance(node, AgenticGateway)
        assert node.gateway_type == GatewayType.PARALLEL
        assert node.gateway_role == GatewayRole.MERGING
        assert node.collaboration_mode == mode
        assert node.merging_strategy == strategy
        assert node.trust_score == 42

    def test_emit_agentic_gateway_diverging_round_trips(self):  # B-4
        original = _agentic_gateway_diverging_model()
        source = bpmn_model_to_code(original)
        # Emitted source explicitly carries `merging_strategy=None`.
        assert "merging_strategy=None" in source
        ns = _exec_source(source)
        model = _find_model(ns)
        node = next(iter(model.all_flow_nodes()))
        assert isinstance(node, AgenticGateway)
        assert node.gateway_role == GatewayRole.DIVERGING
        assert node.merging_strategy is None
        assert node.gateway_type == GatewayType.INCLUSIVE

    def test_emit_agentic_gateway_imports(self):  # B-5
        source = bpmn_model_to_code(
            _agentic_gateway_merging_model(
                CollaborationMode.VOTING, MergingStrategy.MAJORITY
            )
        )
        block = _import_block(source)
        for name in ("AgenticGateway", "GatewayRole", "CollaborationMode",
                     "MergingStrategy", "GatewayType"):
            assert name in block, f"expected {name} in import block, got: {block!r}"

    def test_emit_agentic_gateway_debate_borrows_strategy(self):  # B-6
        original = _agentic_gateway_merging_model(
            CollaborationMode.DEBATE, MergingStrategy.LEADER_DRIVEN
        )
        source = bpmn_model_to_code(original)
        ns = _exec_source(source)
        model = _find_model(ns)
        node = next(iter(model.all_flow_nodes()))
        assert isinstance(node, AgenticGateway)
        assert node.collaboration_mode == CollaborationMode.DEBATE
        assert node.merging_strategy == MergingStrategy.LEADER_DRIVEN


# ---------------------------------------------------------------------------
# B-7 / B-8 -- AgenticLane
# ---------------------------------------------------------------------------

class TestAgenticLane:
    def test_emit_agentic_lane_round_trips(self):  # B-7
        original = _agentic_lane_model()
        source = bpmn_model_to_code(original)
        ns = _exec_source(source)
        model = _find_model(ns)
        process = next(iter(model.processes))
        assert len(process.lanes) == 1
        lane = next(iter(process.lanes))
        assert isinstance(lane, AgenticLane)
        assert lane.name == "Reviewer"
        assert lane.role == AgentRole.MANAGER
        assert lane.trust_score == 75
        # The lane still references the original flow_node.
        assert len(lane.flow_nodes) == 1

    def test_emit_agentic_lane_imports(self):  # B-8
        source = bpmn_model_to_code(_agentic_lane_model())
        block = _import_block(source)
        for name in ("AgenticLane", "AgentRole"):
            assert name in block, f"expected {name} in import block, got: {block!r}"

    def test_emit_agentic_lane_with_ref(self):  # S1-b-1
        """An AgenticLane with agent_diagram_ref round-trips through exec'd code."""
        t = Task(name="Code")
        lane = AgenticLane(name="Reviewer", role=AgentRole.MANAGER,
                           trust_score=75, agent_diagram_ref="ref-123",
                           flow_nodes={t})
        p = Process(name="P", flow_nodes={t}, lanes={lane})
        model = BPMNModel(name="RefModel", processes={p})
        source = bpmn_model_to_code(model)
        assert "agent_diagram_ref='ref-123'" in source
        recovered = _find_model(_exec_source(source))
        rec_lane = next(iter(next(iter(recovered.processes)).lanes))
        assert isinstance(rec_lane, AgenticLane)
        assert rec_lane.agent_diagram_ref == "ref-123"

    def test_emit_agentic_lane_without_ref_omits_kwarg(self):  # S1-b-2
        """An AgenticLane with no ref emits no agent_diagram_ref kwarg."""
        source = bpmn_model_to_code(_agentic_lane_model())
        assert "agent_diagram_ref" not in source


# ---------------------------------------------------------------------------
# B-9 / B-10 -- mixed model + vanilla model
# ---------------------------------------------------------------------------

class TestMixedAndVanilla:
    def test_mixed_diagram_emits_both_subclasses_and_bases(self):  # B-9
        source = bpmn_model_to_code(_mixed_model())
        # Both constructors must show up literally in the emitted code.
        assert "AgenticTask(" in source
        assert "AgenticGateway(" in source
        # And the base ones too -- check word boundary so AgenticTask( doesn't
        # match "Task(".
        assert " Task(" in source or "= Task(" in source
        assert " Gateway(" in source or "= Gateway(" in source
        # Exec must succeed.
        ns = _exec_source(source)
        model = _find_model(ns)
        nodes = list(model.all_flow_nodes())
        assert sum(1 for n in nodes if isinstance(n, AgenticTask)) == 1
        assert sum(1 for n in nodes if isinstance(n, Task)
                   and not isinstance(n, AgenticTask)) == 1
        assert sum(1 for n in nodes if isinstance(n, AgenticGateway)) == 1
        assert sum(1 for n in nodes if isinstance(n, Gateway)
                   and not isinstance(n, AgenticGateway)) == 1

    def test_no_agentic_means_no_agentic_imports(self):  # B-10
        source = bpmn_model_to_code(_vanilla_model())
        block = _import_block(source)
        for forbidden in ("AgenticTask", "AgenticGateway", "AgenticLane",
                          "AgentRole", "CollaborationMode", "GatewayRole",
                          "MergingStrategy", "ReflectionMode"):
            assert forbidden not in block, (
                f"vanilla model should not import {forbidden}; got: {block!r}"
            )


# ---------------------------------------------------------------------------
# B-11 -- layout passthrough for agentic classes
# ---------------------------------------------------------------------------

class TestLayoutPassthrough:
    @pytest.mark.parametrize("fixture_fn", [
        _agentic_task_model,
        _agentic_lane_model,
    ])
    def test_layout_round_trips_for_agentic(self, fixture_fn):  # B-11
        model = fixture_fn()
        # Pick the agentic object and attach a layout dict.
        target = None
        if fixture_fn is _agentic_task_model:
            target = next(iter(next(iter(model.processes)).flow_nodes))
        else:
            target = next(iter(next(iter(model.processes)).lanes))
        target.layout = {"id": "x", "bounds": {"x": 1, "y": 2, "width": 30, "height": 40}}
        source = bpmn_model_to_code(model)
        # Layout assignment line present in emitted source.
        assert ".layout =" in source
        ns = _exec_source(source)
        recovered = _find_model(ns)
        if fixture_fn is _agentic_task_model:
            obj = next(iter(next(iter(recovered.processes)).flow_nodes))
        else:
            obj = next(iter(next(iter(recovered.processes)).lanes))
        assert obj.layout is not None
        assert obj.layout["id"] == "x"
        assert obj.layout["bounds"] == {"x": 1, "y": 2, "width": 30, "height": 40}


# ---------------------------------------------------------------------------
# B-12 -- determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("fixture_fn", [
        _agentic_task_model,
        _agentic_gateway_diverging_model,
        _agentic_lane_model,
        _mixed_model,
    ])
    def test_same_agentic_model_produces_same_source(self, fixture_fn):  # B-12
        model = fixture_fn()
        first = bpmn_model_to_code(model)
        second = bpmn_model_to_code(model)
        assert first == second
