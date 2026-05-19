"""Tests for the SEAA'25 agentic-extension wiring of the BPMN converters.

Covers the test matrix in `.claude/bpmn-agentic/03-...md` §6:

* I-1..I-13  Import (JSON -> BUML).
* E-1..E-8   Export (BUML -> JSON).
* R-1..R-5   Round-trip.

Backward compatibility (B-1) is covered transitively by the existing
vanilla-BPMN suite `test_bpmn_converters.py`, which continues to run unchanged.
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    AgentRole,
    BPMNModel,
    Collaboration,
    CollaborationMode,
    Gateway,
    GatewayRole,
    GatewayType,
    Lane,
    MergingStrategy,
    MessageFlow,
    Participant,
    Process,
    ReflectionMode,
    Task,
    TaskType,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.bpmn_diagram_converter import (
    bpmn_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.bpmn_diagram_processor import (
    process_bpmn_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


# ---------------------------------------------------------------------------
# Fixture helpers (mirror test_bpmn_converters.py _envelope / _node / _flow)
# ---------------------------------------------------------------------------

def _envelope(elements, relationships, title="Test BPMN"):
    return {
        "title": title,
        "model": {
            "version": "3.0.0",
            "type": "BPMNDiagram",
            "size": {"width": 1400, "height": 700},
            "elements": elements,
            "relationships": relationships,
            "interactive": {"elements": {}, "relationships": {}},
            "assessments": {},
        },
    }


def _node(elem_id, type_, name, owner=None, **extra):
    base = {
        "id": elem_id,
        "name": name,
        "type": type_,
        "owner": owner,
        "bounds": {"x": 100, "y": 100, "width": 110, "height": 60},
    }
    base.update(extra)
    return base


def _flow(rel_id, source_id, target_id, flow_type="sequence", name="", **extra):
    base = {
        "id": rel_id,
        "name": name,
        "type": "BPMNFlow",
        "owner": None,
        "bounds": {"x": 0, "y": 0, "width": 100, "height": 1},
        "path": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "source": {"element": source_id, "direction": "Right",
                   "bounds": {"x": 0, "y": 0, "width": 0, "height": 0}},
        "target": {"element": target_id, "direction": "Left",
                   "bounds": {"x": 100, "y": 0, "width": 0, "height": 0}},
        "isManuallyLayouted": False,
        "flowType": flow_type,
        "isDefault": False,
    }
    base.update(extra)
    return base


def _process_with_node(node):
    """Wrap a single flow node in a pool-less BPMNModel and export to JSON."""
    process = Process(name="P", flow_nodes={node})
    model = BPMNModel(name="M", processes={process})
    return bpmn_object_to_json(model)


# ===========================================================================
# Import tests (JSON -> BUML)
# ===========================================================================

def test_I1_import_agentic_task():
    """I-1: BPMNTask with isAgentic=true imports as AgenticTask."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Review", taskType="user", marker="none",
            isAgentic=True, reflectionMode="cross", trustScore=85,
            collaborationMode="voting",
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert isinstance(task, AgenticTask)
    assert task.reflection_mode == ReflectionMode.CROSS
    assert task.trust_score == 85
    assert task.task_type == TaskType.USER


def test_I2_import_non_agentic_task_discards_seaa_fields():
    """I-2: isAgentic=false with SEAA fields set in JSON imports as base Task."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Plain", taskType="default", marker="none",
            isAgentic=False, reflectionMode="self", trustScore=99,
            collaborationMode="role",
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert type(task) is Task
    assert not isinstance(task, AgenticTask)


def test_I3_import_agentic_task_clamps_trust_score_high():
    """I-3: trustScore=150 clamps to 100 on import (WME-tolerant bridge)."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Review", taskType="default", marker="none",
            isAgentic=True, reflectionMode="none", trustScore=150,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert task.trust_score == 100


def test_I3b_import_agentic_task_clamps_trust_score_negative():
    """I-3b: trustScore=-5 clamps to 0 on import."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Review", taskType="default", marker="none",
            isAgentic=True, reflectionMode="none", trustScore=-5,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert task.trust_score == 0


def test_I4_import_agentic_task_unknown_reflection_raises():
    """I-4: Unknown reflectionMode raises ConversionError."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Review", taskType="default", marker="none",
            isAgentic=True, reflectionMode="bogus", trustScore=0,
        ),
    }
    with pytest.raises(ConversionError, match="Unknown reflectionMode 'bogus'"):
        process_bpmn_diagram(_envelope(elements, {}))


def test_I5_import_agentic_task_discards_collaboration_mode():
    """I-5: AgenticTask discards collaborationMode (WME extension beyond paper §4.2)."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Review", taskType="default", marker="none",
            isAgentic=True, reflectionMode="cross", trustScore=50,
            collaborationMode="competition",
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert isinstance(task, AgenticTask)
    # Metamodel doesn't store collaboration_mode for tasks.
    assert not hasattr(task, "collaboration_mode")


def test_I6_import_agentic_gateway_diverging():
    """I-6: Diverging gateway: mergingStrategy in JSON is ignored, merging_strategy=None."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Fork", gatewayType="parallel",
            isAgentic=True, gatewayRole="diverging", collaborationMode="voting",
            mergingStrategy="majority", trustScore=85,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [gw] = next(iter(model.processes)).flow_nodes
    assert isinstance(gw, AgenticGateway)
    assert gw.gateway_role == GatewayRole.DIVERGING
    assert gw.collaboration_mode == CollaborationMode.VOTING
    assert gw.merging_strategy is None
    assert gw.trust_score == 85


def test_I7_import_agentic_gateway_merging_voting_absolute():
    """I-7: Full agentic-merging gateway."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Vote", gatewayType="parallel",
            isAgentic=True, gatewayRole="merging", collaborationMode="voting",
            mergingStrategy="absolute-majority", trustScore=70,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [gw] = next(iter(model.processes)).flow_nodes
    assert gw.gateway_role == GatewayRole.MERGING
    assert gw.collaboration_mode == CollaborationMode.VOTING
    assert gw.merging_strategy == MergingStrategy.ABSOLUTE_MAJORITY
    assert gw.trust_score == 70


def test_I8_import_agentic_gateway_rejects_exclusive():
    """I-8: isAgentic=true on gatewayType=exclusive raises ConversionError (paper §4.3)."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "X", gatewayType="exclusive",
            isAgentic=True, gatewayRole="diverging", collaborationMode="voting",
            mergingStrategy="majority", trustScore=0,
        ),
    }
    with pytest.raises(ConversionError, match="Cannot import AgenticGateway"):
        process_bpmn_diagram(_envelope(elements, {}))


def test_I9_import_agentic_gateway_unknown_mode_raises():
    """I-9: Unknown collaborationMode raises ConversionError."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Vote", gatewayType="parallel",
            isAgentic=True, gatewayRole="merging", collaborationMode="bogus",
            mergingStrategy="majority", trustScore=0,
        ),
    }
    with pytest.raises(ConversionError, match="Unknown collaborationMode 'bogus'"):
        process_bpmn_diagram(_envelope(elements, {}))


def test_I10_import_agentic_gateway_illegal_strategy_raises():
    """I-10: (voting, fastest) is illegal per legality table -> ConversionError."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Vote", gatewayType="parallel",
            isAgentic=True, gatewayRole="merging", collaborationMode="voting",
            mergingStrategy="fastest", trustScore=0,
        ),
    }
    with pytest.raises(ConversionError, match="Cannot import AgenticGateway"):
        process_bpmn_diagram(_envelope(elements, {}))


def test_I11_import_agentic_lane():
    """I-11: BPMNSwimlane with isAgentic=true imports as AgenticLane."""
    elements = {
        "p1": _node("p1", "BPMNPool", "Pool"),
        "l1": _node(
            "l1", "BPMNSwimlane", "Reviewer", owner="p1",
            isAgentic=True, role="manager", trustScore=85,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert isinstance(lane, AgenticLane)
    assert lane.role == AgentRole.MANAGER
    assert lane.trust_score == 85


def test_I12_import_non_agentic_lane():
    """I-12: isAgentic=false on swimlane imports as base Lane."""
    elements = {
        "p1": _node("p1", "BPMNPool", "Pool"),
        "l1": _node(
            "l1", "BPMNSwimlane", "Reviewer", owner="p1",
            isAgentic=False, role="worker", trustScore=0,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert type(lane) is Lane
    assert not isinstance(lane, AgenticLane)


def test_I13_import_message_flow_discards_agentic():
    """I-13: BPMNFlow with isAgentic=true imports as base MessageFlow (silent discard)."""
    elements = {
        "p1": _node("p1", "BPMNPool", "P1"),
        "p2": _node("p2", "BPMNPool", "P2"),
        "t1": _node("t1", "BPMNTask", "T1", owner="p1",
                    taskType="default", marker="none"),
        "t2": _node("t2", "BPMNTask", "T2", owner="p2",
                    taskType="default", marker="none"),
    }
    rels = {
        "f1": _flow("f1", "t1", "t2", flow_type="message",
                    isAgentic=True, collaborationMode="debate",
                    mergingStrategy="majority"),
    }
    model = process_bpmn_diagram(_envelope(elements, rels))
    [mf] = model.collaboration.message_flows
    assert type(mf) is MessageFlow


# ===========================================================================
# Export tests (BUML -> JSON)
# ===========================================================================

def test_E1_export_agentic_task():
    """E-1: AgenticTask exports with isAgentic=true and SEAA fields."""
    task = AgenticTask(name="Review", task_type=TaskType.USER,
                       reflection_mode=ReflectionMode.CROSS, trust_score=85)
    out = _process_with_node(task)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is True
    assert entry["reflectionMode"] == "cross"
    assert entry["trustScore"] == 85
    assert entry["collaborationMode"] == "voting"  # WME placeholder
    assert entry["taskType"] == "user"


def test_E2_export_non_agentic_task_emits_wme_defaults():
    """E-2: Base Task exports with isAgentic=false + all four WME-default SEAA fields."""
    task = Task(name="Plain", task_type=TaskType.DEFAULT)
    out = _process_with_node(task)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is False
    assert entry["reflectionMode"] == "none"
    assert entry["trustScore"] == 0
    assert entry["collaborationMode"] == "voting"


def test_E3_export_agentic_gateway_merging():
    """E-3: Agentic merging gateway exports all five SEAA fields."""
    gw = AgenticGateway(
        name="Vote", gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.MERGING,
        collaboration_mode=CollaborationMode.ROLE,
        merging_strategy=MergingStrategy.LEADER_DRIVEN, trust_score=75,
    )
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is True
    assert entry["gatewayRole"] == "merging"
    assert entry["collaborationMode"] == "role"
    assert entry["mergingStrategy"] == "leader-driven"
    assert entry["trustScore"] == 75


def test_E4_export_agentic_gateway_diverging_emits_default_strategy():
    """E-4: Diverging gateway (merging_strategy=None) emits WME placeholder 'majority'."""
    gw = AgenticGateway(
        name="Fork", gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.DIVERGING,
        collaboration_mode=CollaborationMode.VOTING, trust_score=85,
    )
    assert gw.merging_strategy is None  # confirm metamodel state
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is True
    assert entry["gatewayRole"] == "diverging"
    # WME placeholder, not None / null:
    assert entry["mergingStrategy"] == "majority"


def test_E5_export_non_agentic_gateway_emits_wme_defaults():
    """E-5: Base Gateway exports with isAgentic=false + WME defaults."""
    gw = Gateway(name="Plain", gateway_type=GatewayType.EXCLUSIVE)
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is False
    assert entry["gatewayRole"] == "diverging"
    assert entry["collaborationMode"] == "voting"
    assert entry["mergingStrategy"] == "majority"
    assert entry["trustScore"] == 0


def _wrap_lane(lane):
    """Build a Pool+Process+Lane model and export it to JSON."""
    process = Process(name="P", lanes={lane})
    participant = Participant(name="Pool", process=process)
    model = BPMNModel(
        name="M", processes={process},
        collaboration=Collaboration(name="C", participants={participant}),
    )
    return bpmn_object_to_json(model)


def test_E6_export_agentic_lane():
    """E-6: AgenticLane exports with isAgentic=true and SEAA fields."""
    out = _wrap_lane(AgenticLane(name="Reviewer", role=AgentRole.MANAGER,
                                 trust_score=90))
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    assert lane_entry["isAgentic"] is True
    assert lane_entry["role"] == "manager"
    assert lane_entry["trustScore"] == 90


def test_E7_export_non_agentic_lane():
    """E-7: Base Lane exports with isAgentic=false + WME defaults."""
    out = _wrap_lane(Lane(name="Reviewer"))
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    assert lane_entry["isAgentic"] is False
    assert lane_entry["role"] == "worker"
    assert lane_entry["trustScore"] == 0


def test_E8_export_message_flow_emits_isagentic_false():
    """E-8: Base MessageFlow exports with isAgentic=false + WME flow defaults."""
    t1 = Task(name="T1")
    t2 = Task(name="T2")
    p1 = Process(name="P1", flow_nodes={t1})
    p2 = Process(name="P2", flow_nodes={t2})
    part1 = Participant(name="Pool1", process=p1)
    part2 = Participant(name="Pool2", process=p2)
    mf = MessageFlow(source=t1, target=t2, name="msg")
    model = BPMNModel(
        name="M", processes={p1, p2},
        collaboration=Collaboration(name="C", participants={part1, part2},
                                    message_flows={mf}),
    )
    out = bpmn_object_to_json(model)
    [rel] = out["relationships"].values()
    assert rel["isAgentic"] is False
    assert rel["collaborationMode"] == "voting"
    assert rel["mergingStrategy"] == "majority"


# ===========================================================================
# Round-trip tests
# ===========================================================================

def test_R1_roundtrip_agentic_task():
    """R-1: Agentic task JSON -> BUML -> JSON preserves SEAA fields."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Review", taskType="user", marker="none",
            isAgentic=True, reflectionMode="cross", trustScore=85,
            collaborationMode="voting",
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    out = bpmn_object_to_json(model)
    [entry] = (e for e in out["elements"].values() if e["type"] == "BPMNTask")
    assert entry["isAgentic"] is True
    assert entry["reflectionMode"] == "cross"
    assert entry["trustScore"] == 85
    assert entry["taskType"] == "user"


def test_R2_roundtrip_agentic_gateway_merging():
    """R-2: Full agentic merging gateway round-trips lossless."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Vote", gatewayType="inclusive",
            isAgentic=True, gatewayRole="merging",
            collaborationMode="competition", mergingStrategy="most-complete",
            trustScore=60,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    out = bpmn_object_to_json(model)
    [entry] = (e for e in out["elements"].values() if e["type"] == "BPMNGateway")
    assert entry["isAgentic"] is True
    assert entry["gatewayType"] == "inclusive"
    assert entry["gatewayRole"] == "merging"
    assert entry["collaborationMode"] == "competition"
    assert entry["mergingStrategy"] == "most-complete"
    assert entry["trustScore"] == 60


def test_R3_roundtrip_agentic_gateway_diverging():
    """R-3: Diverging round-trips with mergingStrategy='majority' WME placeholder."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Fork", gatewayType="parallel",
            isAgentic=True, gatewayRole="diverging", collaborationMode="role",
            mergingStrategy="majority", trustScore=50,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    out = bpmn_object_to_json(model)
    [entry] = (e for e in out["elements"].values() if e["type"] == "BPMNGateway")
    assert entry["gatewayRole"] == "diverging"
    # Round-tripped via metamodel's None -> WME placeholder "majority":
    assert entry["mergingStrategy"] == "majority"


def test_R4_roundtrip_agentic_lane():
    """R-4: Agentic lane round-trips lossless."""
    elements = {
        "p1": _node("p1", "BPMNPool", "Pool"),
        "l1": _node(
            "l1", "BPMNSwimlane", "Reviewer", owner="p1",
            isAgentic=True, role="manager", trustScore=95,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    out = bpmn_object_to_json(model)
    [lane_entry] = (e for e in out["elements"].values()
                    if e["type"] == "BPMNSwimlane")
    assert lane_entry["isAgentic"] is True
    assert lane_entry["role"] == "manager"
    assert lane_entry["trustScore"] == 95


def test_R5_roundtrip_mixed_diagram():
    """R-5: Mixed agentic + non-agentic elements round-trip correctly."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Plain", taskType="default", marker="none",
            isAgentic=False,
        ),
        "t2": _node(
            "t2", "BPMNTask", "Agent", taskType="user", marker="none",
            isAgentic=True, reflectionMode="self", trustScore=50,
        ),
        "g1": _node(
            "g1", "BPMNGateway", "PlainGw", gatewayType="exclusive",
            isAgentic=False,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    out = bpmn_object_to_json(model)
    by_name = {e["name"]: e for e in out["elements"].values()}
    assert by_name["Plain"]["isAgentic"] is False
    assert by_name["Agent"]["isAgentic"] is True
    assert by_name["Agent"]["reflectionMode"] == "self"
    assert by_name["Agent"]["trustScore"] == 50
    assert by_name["PlainGw"]["isAgentic"] is False
