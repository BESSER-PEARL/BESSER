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
    AgenticMessageFlow,
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


def test_I5_import_agentic_task_stores_collaboration_mode():
    """I-5 (S2): AgenticTask now STORES collaborationMode (04D1 D-D1; was discarded pre-S2)."""
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
    assert task.collaboration_mode == CollaborationMode.COMPETITION


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


def test_I13_import_message_flow_agentic_now_typed():
    """I-13 (S3): BPMNFlow with isAgentic=true now imports as AgenticMessageFlow.

    Pre-S3 this asserted a silent discard to base MessageFlow; S3 introduces the
    AgenticMessageFlow class, so the agentic fields are now preserved.
    """
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
                    mergingStrategy="majority", trustScore=33),
    }
    model = process_bpmn_diagram(_envelope(elements, rels))
    [mf] = model.collaboration.message_flows
    assert isinstance(mf, AgenticMessageFlow)
    assert mf.collaboration_mode == CollaborationMode.DEBATE
    assert mf.merging_strategy == MergingStrategy.MAJORITY
    assert mf.trust_score == 33


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
    assert entry["collaborationMode"] == "voting"  # default VOTING (now a stored value, S2)
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


# ===========================================================================
# R-b — AgenticGateway.governance_dsl (governance-dsl guide 02; opaque CDATA)
# ===========================================================================

_GOV_DSL = "Scopes:\n    Tasks:\n        MergeDecision\nMajorityPolicy P {\n    ratio : 0.5\n}"


def _agentic_merging_gateway_elements(extra):
    """One agentic merging BPMNGateway carrying the given extra fields."""
    return {
        "g1": _node("g1", "BPMNGateway", "Vote", gatewayType="parallel",
                    isAgentic=True, gatewayRole="merging",
                    collaborationMode="role", mergingStrategy="leader-driven",
                    trustScore=75, **extra),
    }


def test_Rb_c1_import_agentic_gateway_with_governance_dsl():
    """R-b-c-1: governanceDsl on a merging gateway imports onto governance_dsl."""
    model = process_bpmn_diagram(
        _envelope(_agentic_merging_gateway_elements({"governanceDsl": _GOV_DSL}), {})
    )
    [gw] = next(iter(model.processes)).flow_nodes
    assert isinstance(gw, AgenticGateway)
    assert gw.governance_dsl == _GOV_DSL


def test_Rb_c2_import_agentic_gateway_blank_governance_is_none():
    """R-b-c-2: a whitespace-only governanceDsl imports as None (matches WME gate)."""
    model = process_bpmn_diagram(
        _envelope(_agentic_merging_gateway_elements({"governanceDsl": "   \n  "}), {})
    )
    [gw] = next(iter(model.processes)).flow_nodes
    assert gw.governance_dsl is None


def test_Rb_c3_import_agentic_gateway_absent_governance_is_none():
    """R-b-c-3: no governanceDsl key → governance_dsl is None."""
    model = process_bpmn_diagram(_envelope(_agentic_merging_gateway_elements({}), {}))
    [gw] = next(iter(model.processes)).flow_nodes
    assert gw.governance_dsl is None


def test_Rb_c4_export_agentic_gateway_with_governance_emits_field():
    """R-b-c-4: an AgenticGateway with governance_dsl emits governanceDsl."""
    gw = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL,
                        gateway_role=GatewayRole.MERGING,
                        collaboration_mode=CollaborationMode.ROLE,
                        merging_strategy=MergingStrategy.LEADER_DRIVEN,
                        governance_dsl=_GOV_DSL)
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["governanceDsl"] == _GOV_DSL


def test_Rb_c5_export_agentic_gateway_without_governance_omits_field():
    """R-b-c-5: an AgenticGateway with no governance_dsl omits governanceDsl."""
    gw = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL,
                        gateway_role=GatewayRole.MERGING,
                        collaboration_mode=CollaborationMode.ROLE,
                        merging_strategy=MergingStrategy.LEADER_DRIVEN)
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert "governanceDsl" not in entry


def test_Rb_c6_export_non_agentic_gateway_no_governance():
    """R-b-c-6: a base Gateway never carries governanceDsl."""
    out = _process_with_node(Gateway(name="Plain", gateway_type=GatewayType.EXCLUSIVE))
    [entry] = out["elements"].values()
    assert "governanceDsl" not in entry


def test_Rb_c7_roundtrip_agentic_gateway_governance_preserved():
    """R-b-c-7: JSON → BUML → JSON preserves a multi-line governanceDsl verbatim."""
    out = bpmn_object_to_json(
        process_bpmn_diagram(
            _envelope(_agentic_merging_gateway_elements({"governanceDsl": _GOV_DSL}), {})
        )
    )
    gw_entry = next(e for e in out["elements"].values() if e["type"] == "BPMNGateway")
    assert gw_entry["governanceDsl"] == _GOV_DSL


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
    assert rel["trustScore"] == 0  # S3: WME always serialises trustScore on flows


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


# ===========================================================================
# S1 — AgenticLane.agent_diagram_ref (WME 08 cross-diagram link)
# ===========================================================================

_REF = "3f0a1c2d-4e5b-4f6a-9012-3456789abcde"


def _lane_elements(extra):
    """Pool + one agentic swimlane carrying the given extra fields."""
    return {
        "p1": _node("p1", "BPMNPool", "Pool"),
        "l1": _node("l1", "BPMNSwimlane", "AgentReviewer", owner="p1",
                    isAgentic=True, role="manager", trustScore=90, **extra),
    }


def test_S1_c1_import_agentic_lane_with_agent_diagram_ref():
    """S1-c-1: agentDiagramRef on the swimlane imports onto the AgenticLane."""
    model = process_bpmn_diagram(_envelope(_lane_elements({"agentDiagramRef": _REF}), {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert isinstance(lane, AgenticLane)
    assert lane.agent_diagram_ref == _REF


def test_S1_c2_import_agentic_lane_empty_ref_is_none():
    """S1-c-2: an empty-string agentDiagramRef imports as None."""
    model = process_bpmn_diagram(_envelope(_lane_elements({"agentDiagramRef": ""}), {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert lane.agent_diagram_ref is None


def test_S1_c3_import_agentic_lane_absent_ref_is_none():
    """S1-c-3: no agentDiagramRef key → agent_diagram_ref is None."""
    model = process_bpmn_diagram(_envelope(_lane_elements({}), {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert lane.agent_diagram_ref is None


def test_S1_c4_export_agentic_lane_with_ref_emits_field():
    """S1-c-4: an AgenticLane with a ref emits agentDiagramRef in the JSON entry."""
    out = _wrap_lane(AgenticLane(name="AgentReviewer", role=AgentRole.MANAGER,
                                 trust_score=90, agent_diagram_ref=_REF))
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    assert lane_entry["agentDiagramRef"] == _REF


def test_S1_c5_export_agentic_lane_without_ref_omits_field():
    """S1-c-5: an AgenticLane with no ref omits agentDiagramRef entirely (WME-08 behaviour)."""
    out = _wrap_lane(AgenticLane(name="AgentReviewer", role=AgentRole.MANAGER,
                                 trust_score=90))
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    assert "agentDiagramRef" not in lane_entry


def test_S1_c6_export_non_agentic_lane_no_ref():
    """S1-c-6: a base Lane never carries agentDiagramRef."""
    out = _wrap_lane(Lane(name="Plain"))
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    assert "agentDiagramRef" not in lane_entry


def test_S1_c7_roundtrip_agentic_lane_with_ref():
    """S1-c-7: JSON → BUML → JSON preserves agentDiagramRef; task does NOT pick it up."""
    elements = _lane_elements({"agentDiagramRef": _REF})
    elements["t1"] = _node("t1", "BPMNTask", "Review", owner="l1",
                           taskType="user", marker="none",
                           isAgentic=True, reflectionMode="cross", trustScore=80)
    out = bpmn_object_to_json(process_bpmn_diagram(_envelope(elements, {})))
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    task_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNTask")
    assert lane_entry["agentDiagramRef"] == _REF
    # The task here has no ref of its own, so it emits none (refs are
    # presence-gated per element; R-a makes the task a first-class carrier).
    assert "agentDiagramRef" not in task_entry


# ===========================================================================
# R-a — AgenticTask.agent_diagram_ref (WME guide 11 canonical task->agent link)
# ===========================================================================

def _agentic_task_elements(extra):
    """One agentic BPMNTask carrying the given extra fields."""
    return {
        "t1": _node("t1", "BPMNTask", "Review", taskType="user", marker="none",
                    isAgentic=True, reflectionMode="cross", trustScore=80,
                    collaborationMode="voting", **extra),
    }


def test_Ra_c1_import_agentic_task_with_agent_diagram_ref():
    """R-a-c-1: agentDiagramRef on the task imports onto the AgenticTask."""
    model = process_bpmn_diagram(_envelope(_agentic_task_elements({"agentDiagramRef": _REF}), {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert isinstance(task, AgenticTask)
    assert task.agent_diagram_ref == _REF


def test_Ra_c2_import_agentic_task_empty_ref_is_none():
    """R-a-c-2: an empty-string agentDiagramRef imports as None."""
    model = process_bpmn_diagram(_envelope(_agentic_task_elements({"agentDiagramRef": ""}), {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert task.agent_diagram_ref is None


def test_Ra_c3_import_agentic_task_absent_ref_is_none():
    """R-a-c-3: no agentDiagramRef key → agent_diagram_ref is None."""
    model = process_bpmn_diagram(_envelope(_agentic_task_elements({}), {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert task.agent_diagram_ref is None


def test_Ra_c4_export_agentic_task_with_ref_emits_field():
    """R-a-c-4: an AgenticTask with a ref emits agentDiagramRef in the JSON entry."""
    out = _process_with_node(AgenticTask(name="Review", task_type=TaskType.USER,
                                         agent_diagram_ref=_REF))
    [entry] = out["elements"].values()
    assert entry["agentDiagramRef"] == _REF


def test_Ra_c5_export_agentic_task_without_ref_omits_field():
    """R-a-c-5: an AgenticTask with no ref omits agentDiagramRef entirely."""
    out = _process_with_node(AgenticTask(name="Review", task_type=TaskType.USER))
    [entry] = out["elements"].values()
    assert "agentDiagramRef" not in entry


def test_Ra_c6_export_non_agentic_task_no_ref():
    """R-a-c-6: a base Task never carries agentDiagramRef."""
    out = _process_with_node(Task(name="Plain", task_type=TaskType.DEFAULT))
    [entry] = out["elements"].values()
    assert "agentDiagramRef" not in entry


def test_Ra_c7_roundtrip_agentic_task_with_ref():
    """R-a-c-7: JSON → BUML → JSON preserves agentDiagramRef on the task."""
    out = bpmn_object_to_json(
        process_bpmn_diagram(_envelope(_agentic_task_elements({"agentDiagramRef": _REF}), {}))
    )
    task_entry = next(e for e in out["elements"].values() if e["type"] == "BPMNTask")
    assert task_entry["agentDiagramRef"] == _REF


# ===========================================================================
# S2 — AgenticTask.collaboration_mode (WME 04D1 D-D1, paper deviation)
# ===========================================================================

def test_S2_c1_import_agentic_task_stores_collaboration_mode():
    """S2-c-1: collaborationMode='debate' imports onto the task (mirror of WME fixture)."""
    elements = {
        "t1": _node("t1", "BPMNTask", "Review", taskType="user", marker="none",
                    isAgentic=True, reflectionMode="cross", trustScore=80,
                    collaborationMode="debate"),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert isinstance(task, AgenticTask)
    assert task.collaboration_mode == CollaborationMode.DEBATE


def test_S2_c2_import_agentic_task_unknown_collaboration_mode_raises():
    """S2-c-2: an unknown collaborationMode on a task raises ConversionError."""
    elements = {
        "t1": _node("t1", "BPMNTask", "Review", taskType="user", marker="none",
                    isAgentic=True, reflectionMode="none", trustScore=0,
                    collaborationMode="telepathy"),
    }
    with pytest.raises(ConversionError, match="Unknown collaborationMode 'telepathy'"):
        process_bpmn_diagram(_envelope(elements, {}))


def test_S2_c3_import_agentic_task_default_collaboration_mode():
    """S2-c-3: absent collaborationMode defaults to VOTING."""
    elements = {
        "t1": _node("t1", "BPMNTask", "Review", taskType="user", marker="none",
                    isAgentic=True, reflectionMode="none", trustScore=0),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert task.collaboration_mode == CollaborationMode.VOTING


def test_S2_c4_export_agentic_task_emits_stored_collaboration_mode():
    """S2-c-4: export emits the stored value, not a fixed placeholder."""
    task = AgenticTask(name="Review", task_type=TaskType.USER,
                       reflection_mode=ReflectionMode.CROSS, trust_score=80,
                       collaboration_mode=CollaborationMode.DEBATE)
    out = _process_with_node(task)
    [entry] = out["elements"].values()
    assert entry["collaborationMode"] == "debate"


def test_S2_c5_roundtrip_agentic_task_collaboration_mode():
    """S2-c-5: a non-voting collaborationMode round-trips JSON -> BUML -> JSON."""
    elements = {
        "t1": _node("t1", "BPMNTask", "Review", taskType="user", marker="none",
                    isAgentic=True, reflectionMode="self", trustScore=30,
                    collaborationMode="role"),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    out = bpmn_object_to_json(model)
    [entry] = (e for e in out["elements"].values() if e["type"] == "BPMNTask")
    assert entry["collaborationMode"] == "role"


# ===========================================================================
# S3 — AgenticMessageFlow (WME 04D1)
# ===========================================================================

def _two_pool_message_flow(rel_extra):
    """Two pools, one task each, and a message flow carrying the given extras."""
    elements = {
        "p1": _node("p1", "BPMNPool", "P1"),
        "p2": _node("p2", "BPMNPool", "P2"),
        "t1": _node("t1", "BPMNTask", "T1", owner="p1",
                    taskType="default", marker="none"),
        "t2": _node("t2", "BPMNTask", "T2", owner="p2",
                    taskType="default", marker="none"),
    }
    rels = {"f1": _flow("f1", "t1", "t2", flow_type="message", **rel_extra)}
    return _envelope(elements, rels)


def test_S3_c1_import_agentic_message_flow():
    """S3-c-1: message flow with isAgentic=true imports as AgenticMessageFlow."""
    env = _two_pool_message_flow({
        "isAgentic": True, "collaborationMode": "voting",
        "mergingStrategy": "majority", "trustScore": 50,
    })
    model = process_bpmn_diagram(env)
    [mf] = model.collaboration.message_flows
    assert isinstance(mf, AgenticMessageFlow)
    assert mf.collaboration_mode == CollaborationMode.VOTING
    assert mf.merging_strategy == MergingStrategy.MAJORITY
    assert mf.trust_score == 50


def test_S3_c2_import_non_agentic_message_flow_still_base():
    """S3-c-2: a flow with no isAgentic imports as base MessageFlow."""
    env = _two_pool_message_flow({})
    model = process_bpmn_diagram(env)
    [mf] = model.collaboration.message_flows
    assert type(mf) is MessageFlow


def test_S3_c3_import_agentic_message_flow_unknown_mode_raises():
    """S3-c-3: an unknown collaborationMode raises ConversionError."""
    env = _two_pool_message_flow({
        "isAgentic": True, "collaborationMode": "telepathy",
        "mergingStrategy": "majority", "trustScore": 0,
    })
    with pytest.raises(ConversionError, match="Unknown collaborationMode 'telepathy'"):
        process_bpmn_diagram(env)


def test_S3_c4_import_agentic_message_flow_illegal_strategy_raises():
    """S3-c-4: (voting, fastest) is illegal per the legality table -> ConversionError."""
    env = _two_pool_message_flow({
        "isAgentic": True, "collaborationMode": "voting",
        "mergingStrategy": "fastest", "trustScore": 0,
    })
    with pytest.raises(ConversionError, match="Cannot import AgenticMessageFlow"):
        process_bpmn_diagram(env)


def _wrap_message_flow(mf, t1, t2):
    """Two-pool model carrying a single (agentic or base) message flow."""
    p1 = Process(name="P1", flow_nodes={t1})
    p2 = Process(name="P2", flow_nodes={t2})
    part1 = Participant(name="Pool1", process=p1)
    part2 = Participant(name="Pool2", process=p2)
    model = BPMNModel(
        name="M", processes={p1, p2},
        collaboration=Collaboration(name="C", participants={part1, part2},
                                    message_flows={mf}),
    )
    return bpmn_object_to_json(model)


def test_S3_c5_export_agentic_message_flow():
    """S3-c-5: AgenticMessageFlow exports with isAgentic=true + the three fields."""
    t1, t2 = Task(name="T1"), Task(name="T2")
    mf = AgenticMessageFlow(source=t1, target=t2, name="msg",
                            collaboration_mode=CollaborationMode.ROLE,
                            merging_strategy=MergingStrategy.COMPOSED,
                            trust_score=70)
    out = _wrap_message_flow(mf, t1, t2)
    [rel] = out["relationships"].values()
    assert rel["isAgentic"] is True
    assert rel["collaborationMode"] == "role"
    assert rel["mergingStrategy"] == "composed"
    assert rel["trustScore"] == 70


def test_S3_c6_export_non_agentic_message_flow_emits_defaults():
    """S3-c-6: a base MessageFlow keeps the WME flow defaults (isAgentic=false)."""
    t1, t2 = Task(name="T1"), Task(name="T2")
    mf = MessageFlow(source=t1, target=t2, name="msg")
    out = _wrap_message_flow(mf, t1, t2)
    [rel] = out["relationships"].values()
    assert rel["isAgentic"] is False
    assert rel["collaborationMode"] == "voting"
    assert rel["mergingStrategy"] == "majority"
    assert rel["trustScore"] == 0


def test_S3_c7_roundtrip_agentic_message_flow():
    """S3-c-7: JSON -> BUML -> JSON preserves all three agentic flow fields."""
    env = _two_pool_message_flow({
        "isAgentic": True, "collaborationMode": "competition",
        "mergingStrategy": "most-complete", "trustScore": 40,
    })
    out = bpmn_object_to_json(process_bpmn_diagram(env))
    [rel] = out["relationships"].values()
    assert rel["isAgentic"] is True
    assert rel["collaborationMode"] == "competition"
    assert rel["mergingStrategy"] == "most-complete"
    assert rel["trustScore"] == 40
