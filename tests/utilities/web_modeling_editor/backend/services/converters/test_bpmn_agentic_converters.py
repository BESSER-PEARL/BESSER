"""Tests for the SEAA'25 agentic-extension wiring of the BPMN converters.

Covers:

* I-1..I-13  Import (JSON -> BUML).
* E-1..E-8   Export (BUML -> JSON).
* R-1..R-5   Round-trip.

P3' rationalization: collaborationMode / mergingStrategy fields are silently
ignored on import; AgenticMessageFlow has been removed. Tests updated accordingly.
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    AgenticGateway,
    AgenticLane,
    AgenticTask,
    AgentRole,
    BPMNModel,
    Collaboration,
    Gateway,
    GatewayRole,
    GatewayType,
    Lane,
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


def test_I5_import_agentic_task_collaboration_mode_silently_ignored():
    """I-5: collaborationMode in JSON is silently ignored (P3' rationalization);
    task still imports as AgenticTask, no error raised."""
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
    # No collaboration_mode attribute exists on P3' AgenticTask.
    assert not hasattr(task, "collaboration_mode")


def test_I6_import_agentic_gateway_diverging():
    """I-6: Diverging gateway imports correctly; collaborationMode+mergingStrategy silently ignored."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Fork", gatewayType="parallel",
            isAgentic=True, gatewayRole="diverging",
            collaborationMode="voting", mergingStrategy="majority", trustScore=85,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [gw] = next(iter(model.processes)).flow_nodes
    assert isinstance(gw, AgenticGateway)
    assert gw.gateway_role == GatewayRole.DIVERGING
    assert gw.trust_score == 85


def test_I7_import_agentic_gateway_merging():
    """I-7: Full agentic-merging gateway; mergingStrategy silently ignored."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Vote", gatewayType="parallel",
            isAgentic=True, gatewayRole="merging",
            collaborationMode="voting", mergingStrategy="absolute-majority", trustScore=70,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [gw] = next(iter(model.processes)).flow_nodes
    assert gw.gateway_role == GatewayRole.MERGING
    assert gw.trust_score == 70


def test_I8_import_agentic_gateway_rejects_exclusive():
    """I-8: isAgentic=true on gatewayType=exclusive raises ConversionError (paper §4.3)."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "X", gatewayType="exclusive",
            isAgentic=True, gatewayRole="diverging", trustScore=0,
        ),
    }
    with pytest.raises(ConversionError, match="Cannot import AgenticGateway"):
        process_bpmn_diagram(_envelope(elements, {}))


def test_I9_import_agentic_gateway_unknown_collaboration_mode_silently_ignored():
    """I-9: Unknown collaborationMode is silently ignored (P3' rationalization)."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Vote", gatewayType="parallel",
            isAgentic=True, gatewayRole="merging",
            collaborationMode="bogus", mergingStrategy="majority", trustScore=0,
        ),
    }
    # No ConversionError — the field is simply not read.
    model = process_bpmn_diagram(_envelope(elements, {}))
    [gw] = next(iter(model.processes)).flow_nodes
    assert isinstance(gw, AgenticGateway)
    assert gw.gateway_role == GatewayRole.MERGING


def test_I10_import_agentic_gateway_illegal_strategy_silently_ignored():
    """I-10: An illegal mergingStrategy is silently ignored (P3' rationalization)."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Vote", gatewayType="parallel",
            isAgentic=True, gatewayRole="merging",
            collaborationMode="voting", mergingStrategy="fastest", trustScore=0,
        ),
    }
    # No ConversionError — the field is simply not read.
    model = process_bpmn_diagram(_envelope(elements, {}))
    [gw] = next(iter(model.processes)).flow_nodes
    assert isinstance(gw, AgenticGateway)


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


def test_I13_import_message_flow_agentic_downgraded_to_base():
    """I-13: BPMNFlow with isAgentic=true is silently downgraded to base MessageFlow
    (P3' rationalization — AgenticMessageFlow removed)."""
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
    # Silently downgraded to base MessageFlow.
    assert type(mf) is MessageFlow


def test_I15_roundtrip_pool_to_pool_message_flow():
    """R-R1: pool-to-pool message flow round-trips JSON -> BUML -> JSON (flowType=message)."""
    elements = {
        "p1": _node("p1", "BPMNPool", "P1"),
        "p2": _node("p2", "BPMNPool", "P2"),
    }
    rels = {
        "f1": _flow("f1", "p1", "p2", flow_type="message", isAgentic=False),
    }
    out = bpmn_object_to_json(process_bpmn_diagram(_envelope(elements, rels)))
    [rel] = out["relationships"].values()
    assert rel["flowType"] == "message"


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
    assert entry["taskType"] == "user"
    # collaborationMode no longer emitted (P3').
    assert "collaborationMode" not in entry


def test_E2_export_non_agentic_task_emits_wme_defaults():
    """E-2: Base Task exports with isAgentic=false + WME-default SEAA fields."""
    task = Task(name="Plain", task_type=TaskType.DEFAULT)
    out = _process_with_node(task)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is False
    assert entry["reflectionMode"] == "none"
    assert entry["trustScore"] == 0
    # collaborationMode no longer emitted (P3').
    assert "collaborationMode" not in entry


def test_E3_export_agentic_gateway_merging():
    """E-3: Agentic merging gateway exports gatewayRole + trustScore."""
    gw = AgenticGateway(
        name="Vote", gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.MERGING,
        trust_score=75,
    )
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is True
    assert entry["gatewayRole"] == "merging"
    assert entry["trustScore"] == 75
    # collaborationMode and mergingStrategy no longer emitted (P3').
    assert "collaborationMode" not in entry
    assert "mergingStrategy" not in entry


def test_E4_export_agentic_gateway_diverging():
    """E-4: Diverging gateway exports correctly."""
    gw = AgenticGateway(
        name="Fork", gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.DIVERGING,
        trust_score=85,
    )
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is True
    assert entry["gatewayRole"] == "diverging"
    assert entry["trustScore"] == 85
    # No merging-related fields (P3').
    assert "mergingStrategy" not in entry
    assert "collaborationMode" not in entry


def test_E5_export_non_agentic_gateway_emits_wme_defaults():
    """E-5: Base Gateway exports with isAgentic=false + WME defaults."""
    gw = Gateway(name="Plain", gateway_type=GatewayType.EXCLUSIVE)
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["isAgentic"] is False
    assert entry["gatewayRole"] == "diverging"
    assert entry["trustScore"] == 0
    # collaborationMode and mergingStrategy no longer emitted (P3').
    assert "collaborationMode" not in entry
    assert "mergingStrategy" not in entry


# ===========================================================================
# R-b — AgenticGateway.governance_dsl (governance-dsl guide 02; opaque CDATA)
# ===========================================================================

_GOV_DSL = "Scopes:\n    Tasks:\n        MergeDecision\nMajorityPolicy P {\n    ratio : 0.5\n}"


def _agentic_merging_gateway_elements(extra):
    """One agentic merging BPMNGateway carrying the given extra fields.

    Legacy fields collaborationMode/mergingStrategy are included to verify
    tolerance (they are silently ignored on import).
    """
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
                        governance_dsl=_GOV_DSL)
    out = _process_with_node(gw)
    [entry] = out["elements"].values()
    assert entry["governanceDsl"] == _GOV_DSL


def test_Rb_c5_export_agentic_gateway_without_governance_omits_field():
    """R-b-c-5: an AgenticGateway with no governance_dsl omits governanceDsl."""
    gw = AgenticGateway(name="Vote", gateway_type=GatewayType.PARALLEL,
                        gateway_role=GatewayRole.MERGING)
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
    assert rel["trustScore"] == 0
    # collaborationMode and mergingStrategy no longer emitted (P3').
    assert "collaborationMode" not in rel
    assert "mergingStrategy" not in rel


# ===========================================================================
# Round-trip tests
# ===========================================================================

def test_R1_roundtrip_agentic_task():
    """R-1: Agentic task JSON -> BUML -> JSON preserves SEAA fields."""
    elements = {
        "t1": _node(
            "t1", "BPMNTask", "Review", taskType="user", marker="none",
            isAgentic=True, reflectionMode="cross", trustScore=85,
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
    assert entry["trustScore"] == 60
    # collaborationMode and mergingStrategy no longer in the output (P3').
    assert "collaborationMode" not in entry
    assert "mergingStrategy" not in entry


def test_R3_roundtrip_agentic_gateway_diverging():
    """R-3: Diverging gateway round-trips correctly."""
    elements = {
        "g1": _node(
            "g1", "BPMNGateway", "Fork", gatewayType="parallel",
            isAgentic=True, gatewayRole="diverging",
            collaborationMode="role", mergingStrategy="majority", trustScore=50,
        ),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    out = bpmn_object_to_json(model)
    [entry] = (e for e in out["elements"].values() if e["type"] == "BPMNGateway")
    assert entry["gatewayRole"] == "diverging"
    assert entry["trustScore"] == 50


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
    assert "agentDiagramRef" not in task_entry


# ===========================================================================
# 3c — AgenticLane.multiplicity (WME 2026-06-08 point #3 swarm size)
# ===========================================================================

def test_3c_roundtrip_agentic_lane_multiplicity():
    """3c: JSON multiplicity=4 → BUML multiplicity=4 → JSON multiplicity=4."""
    model = process_bpmn_diagram(_envelope(_lane_elements({"multiplicity": 4}), {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert lane.multiplicity == 4
    out = bpmn_object_to_json(model)
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    assert lane_entry["multiplicity"] == 4


def test_3c_import_agentic_lane_absent_multiplicity_is_one():
    """3c: no multiplicity key → multiplicity is 1."""
    model = process_bpmn_diagram(_envelope(_lane_elements({}), {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert lane.multiplicity == 1


def test_3c_import_agentic_lane_zero_multiplicity_clamps_to_one():
    """3c: multiplicity=0 clamps to 1 on import (WME-tolerant bridge)."""
    model = process_bpmn_diagram(_envelope(_lane_elements({"multiplicity": 0}), {}))
    [participant] = model.collaboration.participants
    [lane] = participant.process.lanes
    assert lane.multiplicity == 1


def test_3c_export_non_agentic_lane_carries_default_multiplicity():
    """3c: a base Lane emits multiplicity=1 from _WME_LANE_DEFAULTS."""
    out = _wrap_lane(Lane(name="Plain"))
    lane_entry = next(e for e in out["elements"].values()
                      if e["type"] == "BPMNSwimlane")
    assert lane_entry["multiplicity"] == 1


# ===========================================================================
# R-a — AgenticTask.agent_diagram_ref (WME guide 11 canonical task->agent link)
# ===========================================================================

def _agentic_task_elements(extra):
    """One agentic BPMNTask carrying the given extra fields.

    Legacy collaborationMode field included to verify tolerance (silently ignored).
    """
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
# S2 tolerance — collaborationMode silently ignored (P3' rationalization)
# ===========================================================================

def test_S2_collaboration_mode_silently_ignored():
    """S2: collaborationMode in the JSON is silently ignored — task imports OK."""
    elements = {
        "t1": _node("t1", "BPMNTask", "Review", taskType="user", marker="none",
                    isAgentic=True, reflectionMode="cross", trustScore=80,
                    collaborationMode="debate"),
    }
    model = process_bpmn_diagram(_envelope(elements, {}))
    [task] = next(iter(model.processes)).flow_nodes
    assert isinstance(task, AgenticTask)
    # No collaboration_mode attribute on P3' AgenticTask.
    assert not hasattr(task, "collaboration_mode")
