"""Tests for the BPMN converters.

Covers ``process_bpmn_diagram`` (JSON → BUML), ``bpmn_object_to_json`` (BUML → JSON),
their round-trip identity, the full WME ↔ metamodel event mapping (every row of
Appendix A in ``.claude/bpmn/03-bpmn-converters-guide.md``), and error handling.
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    Activity,
    BPMNModel,
    Collaboration,
    EndEvent,
    EventDefinitionType,
    EventDirection,
    Gateway,
    GatewayType,
    IntermediateEvent,
    Lane,
    LoopCharacteristics,
    MessageFlow,
    Participant,
    Process,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TaskType,
)
from besser.utilities.web_modeling_editor.backend.services.converters.bpmn_event_mapping import (
    parse_event_type,
    serialise_event_type,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.bpmn_diagram_converter import (
    bpmn_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.bpmn_diagram_processor import (
    process_bpmn_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


# ---------------------------------------------------------------------------
# JSON envelope helpers — keep test fixtures readable
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


# ---------------------------------------------------------------------------
# Hand-written JSON fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poolless_start_task_end_json():
    """Pool-less: StartEvent -> Task -> EndEvent."""
    elements = {
        "s1": _node("s1", "BPMNStartEvent", "go", eventType="default"),
        "t1": _node("t1", "BPMNTask", "Do work", taskType="user", marker="none"),
        "e1": _node("e1", "BPMNEndEvent", "done", eventType="default"),
    }
    relationships = {
        "f1": _flow("f1", "s1", "t1"),
        "f2": _flow("f2", "t1", "e1"),
    }
    return _envelope(elements, relationships, title="Poolless")


@pytest.fixture
def two_pool_collaboration_json():
    """Two pools, each with one task; one MessageFlow between them."""
    elements = {
        "p1": _node("p1", "BPMNPool", "Buyer"),
        "p2": _node("p2", "BPMNPool", "Seller"),
        "t1": _node("t1", "BPMNTask", "Place order", owner="p1",
                    taskType="default", marker="none"),
        "t2": _node("t2", "BPMNTask", "Ship", owner="p2",
                    taskType="default", marker="none"),
    }
    relationships = {
        "m1": _flow("m1", "t1", "t2", flow_type="message", name="order"),
    }
    return _envelope(elements, relationships, title="Buyer-Seller")


@pytest.fixture
def gateway_with_default_json():
    """ExclusiveGateway with two outgoing sequence flows, one marked default."""
    elements = {
        "s1": _node("s1", "BPMNStartEvent", "", eventType="default"),
        "g1": _node("g1", "BPMNGateway", "?", gatewayType="exclusive"),
        "t1": _node("t1", "BPMNTask", "A", taskType="default", marker="none"),
        "t2": _node("t2", "BPMNTask", "B", taskType="default", marker="none"),
        "e1": _node("e1", "BPMNEndEvent", "", eventType="default"),
        "e2": _node("e2", "BPMNEndEvent", "", eventType="default"),
    }
    relationships = {
        "f0": _flow("f0", "s1", "g1"),
        "f1": _flow("f1", "g1", "t1", name="happy"),
        "f2": _flow("f2", "g1", "t2", name="default", isDefault=True),
        "fa": _flow("fa", "t1", "e1"),
        "fb": _flow("fb", "t2", "e2"),
    }
    return _envelope(elements, relationships, title="Gateway")


@pytest.fixture
def subprocess_json():
    """Subprocess containing two flow nodes and one sequence flow inside it."""
    elements = {
        "s1": _node("s1", "BPMNStartEvent", "", eventType="default"),
        "sp": _node("sp", "BPMNSubprocess", "Sub", marker="none"),
        "e1": _node("e1", "BPMNEndEvent", "", eventType="default"),
        "ti": _node("ti", "BPMNTask", "inner", owner="sp",
                    taskType="default", marker="none"),
        "ei": _node("ei", "BPMNEndEvent", "inner end", owner="sp",
                    eventType="default"),
    }
    relationships = {
        "f1": _flow("f1", "s1", "sp"),
        "f2": _flow("f2", "sp", "e1"),
        "fi": _flow("fi", "ti", "ei"),  # inside subprocess
    }
    return _envelope(elements, relationships, title="Subprocess")


# ---------------------------------------------------------------------------
# 1. process_bpmn_diagram — JSON → BUML
# ---------------------------------------------------------------------------

class TestProcessBpmnDiagram:
    def test_poolless_builds_one_process_with_three_nodes(self, poolless_start_task_end_json):
        model = process_bpmn_diagram(poolless_start_task_end_json)
        assert isinstance(model, BPMNModel)
        assert model.collaboration is None
        assert len(model.processes) == 1
        process = next(iter(model.processes))
        assert len(process.flow_nodes) == 3
        assert len(process.sequence_flows) == 2

    def test_poolless_node_classes_and_subtype_attrs(self, poolless_start_task_end_json):
        model = process_bpmn_diagram(poolless_start_task_end_json)
        nodes = {type(n).__name__ for n in model.all_flow_nodes()}
        assert nodes == {"StartEvent", "Task", "EndEvent"}
        task = next(n for n in model.all_flow_nodes() if isinstance(n, Task))
        assert task.task_type is TaskType.USER
        assert task.loop_characteristics is LoopCharacteristics.NONE

    def test_poolless_layout_round_trip_data_stashed(self, poolless_start_task_end_json):
        model = process_bpmn_diagram(poolless_start_task_end_json)
        ids = {(n.layout or {}).get("id") for n in model.all_flow_nodes()}
        assert ids == {"s1", "t1", "e1"}

    def test_poolless_validate_succeeds(self, poolless_start_task_end_json):
        model = process_bpmn_diagram(poolless_start_task_end_json)
        result = model.validate(raise_exception=False)
        assert result["success"] is True, result["errors"]

    def test_two_pool_builds_collaboration(self, two_pool_collaboration_json):
        model = process_bpmn_diagram(two_pool_collaboration_json)
        assert isinstance(model.collaboration, Collaboration)
        assert len(model.collaboration.participants) == 2
        # Each pool has its own Process; collaboration has 1 message flow.
        assert len(model.processes) == 2
        assert len(model.collaboration.message_flows) == 1
        msg = next(iter(model.collaboration.message_flows))
        assert isinstance(msg, MessageFlow)
        assert msg.name == "order"

    def test_gateway_default_flow_is_marked(self, gateway_with_default_json):
        model = process_bpmn_diagram(gateway_with_default_json)
        gateway = next(n for n in model.all_flow_nodes() if isinstance(n, Gateway))
        assert gateway.gateway_type is GatewayType.EXCLUSIVE
        defaults = [f for f in gateway.outgoing() if f.is_default]
        assert len(defaults) == 1
        assert defaults[0].name == "default"
        assert gateway.default_flow is defaults[0]

    def test_subprocess_contains_inner_nodes(self, subprocess_json):
        model = process_bpmn_diagram(subprocess_json)
        sub = next(n for n in model.all_flow_nodes() if isinstance(n, SubProcess))
        # The subprocess holds its inner nodes; outer process holds it + start + end.
        assert len(sub.flow_nodes) == 2
        assert len(sub.sequence_flows) == 1
        outer = next(iter(model.processes))
        assert sub in outer.flow_nodes
        # The inner sequence flow is in the subprocess, not in the outer process.
        assert all(f.source.container is sub for f in sub.sequence_flows)

    def test_dangling_endpoint_logs_and_skips(self, caplog):
        elements = {"t1": _node("t1", "BPMNTask", "A", taskType="default", marker="none")}
        relationships = {
            "f1": _flow("f1", "t1", "missing"),  # target doesn't exist
        }
        json_data = _envelope(elements, relationships)
        with caplog.at_level("WARNING"):
            model = process_bpmn_diagram(json_data)
        assert len(next(iter(model.processes)).sequence_flows) == 0
        assert any("dangling endpoint" in rec.message for rec in caplog.records)

    def test_illegal_is_default_logs_and_downgrades(self, caplog):
        # A Task -> Task sequence flow with isDefault=True. The first task is the
        # source — Activity sources CAN carry default, so use a parallel gateway
        # source instead (parallel cannot carry defaults per BPMN §8.3.13).
        elements = {
            "g1": _node("g1", "BPMNGateway", "", gatewayType="parallel"),
            "t1": _node("t1", "BPMNTask", "A", taskType="default", marker="none"),
            "t2": _node("t2", "BPMNTask", "B", taskType="default", marker="none"),
        }
        relationships = {
            "fa": _flow("fa", "g1", "t1"),
            "fb": _flow("fb", "g1", "t2", isDefault=True),
        }
        json_data = _envelope(elements, relationships)
        with caplog.at_level("WARNING"):
            model = process_bpmn_diagram(json_data)
        flows = next(iter(model.processes)).sequence_flows
        assert all(not f.is_default for f in flows)
        assert any("downgrading to is_default=False" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 2. bpmn_object_to_json — BUML → JSON
# ---------------------------------------------------------------------------

class TestBpmnObjectToJson:
    def test_envelope_keys_present(self, poolless_start_task_end_json):
        model = process_bpmn_diagram(poolless_start_task_end_json)
        out = bpmn_object_to_json(model)
        assert out["version"] == "3.0.0"
        assert out["type"] == "BPMNDiagram"
        assert set(out.keys()) >= {
            "version", "type", "size", "interactive", "elements", "relationships",
            "assessments",
        }

    def test_emits_three_nodes_two_flows(self, poolless_start_task_end_json):
        model = process_bpmn_diagram(poolless_start_task_end_json)
        out = bpmn_object_to_json(model)
        assert len(out["elements"]) == 3
        assert len(out["relationships"]) == 2
        types = {e["type"] for e in out["elements"].values()}
        assert types == {"BPMNStartEvent", "BPMNTask", "BPMNEndEvent"}

    def test_pool_owner_pointer(self, two_pool_collaboration_json):
        model = process_bpmn_diagram(two_pool_collaboration_json)
        out = bpmn_object_to_json(model)
        pool_ids = {eid for eid, e in out["elements"].items() if e["type"] == "BPMNPool"}
        # Each task's owner is one of the pools.
        task_owners = {e["owner"] for e in out["elements"].values() if e["type"] == "BPMNTask"}
        assert task_owners.issubset(pool_ids)

    def test_message_flow_emitted(self, two_pool_collaboration_json):
        model = process_bpmn_diagram(two_pool_collaboration_json)
        out = bpmn_object_to_json(model)
        flow_types = {r["flowType"] for r in out["relationships"].values()}
        assert flow_types == {"message"}

    def test_gateway_default_flag_round_trips(self, gateway_with_default_json):
        model = process_bpmn_diagram(gateway_with_default_json)
        out = bpmn_object_to_json(model)
        defaults = [r for r in out["relationships"].values()
                    if r.get("flowType") == "sequence" and r.get("isDefault")]
        assert len(defaults) == 1
        assert defaults[0]["name"] == "default"

    def test_subprocess_owner_pointer(self, subprocess_json):
        model = process_bpmn_diagram(subprocess_json)
        out = bpmn_object_to_json(model)
        sub_id = next(
            eid for eid, e in out["elements"].items() if e["type"] == "BPMNSubprocess"
        )
        # The two inner nodes have the subprocess as owner.
        inner = [e for e in out["elements"].values() if e["owner"] == sub_id]
        assert {e["type"] for e in inner} == {"BPMNTask", "BPMNEndEvent"}

    def test_ids_are_reused_from_layout(self, poolless_start_task_end_json):
        model = process_bpmn_diagram(poolless_start_task_end_json)
        out = bpmn_object_to_json(model)
        # Since process_ stashed the original ids in layout, the converter reuses them.
        assert set(out["elements"].keys()) == {"s1", "t1", "e1"}
        assert set(out["relationships"].keys()) == {"f1", "f2"}


# ---------------------------------------------------------------------------
# 3. Round-trip identity (JSON → BUML → JSON)
# ---------------------------------------------------------------------------

def _signature(elements, relationships):
    """Compact comparable summary of the load-bearing fields. ``isDefault`` is only
    meaningful for sequence flows — the converter omits it on other flow types."""
    el_sig = sorted(
        (eid, e["type"], e["name"], e.get("owner"),
         e.get("taskType"), e.get("marker"),
         e.get("eventType"), e.get("gatewayType"))
        for eid, e in elements.items()
    )
    rel_sig = sorted(
        (rid, r["type"], r["name"], r.get("flowType"),
         r.get("isDefault") if r.get("flowType") == "sequence" else None,
         (r.get("source") or {}).get("element"),
         (r.get("target") or {}).get("element"))
        for rid, r in relationships.items()
    )
    return el_sig, rel_sig


class TestRoundTripIdentity:
    @pytest.mark.parametrize(
        "fixture_name",
        ["poolless_start_task_end_json", "two_pool_collaboration_json",
         "gateway_with_default_json", "subprocess_json"],
    )
    def test_load_bearing_fields_preserved(self, request, fixture_name):
        json_in = request.getfixturevalue(fixture_name)
        model = process_bpmn_diagram(json_in)
        json_out = bpmn_object_to_json(model)

        in_el, in_rel = _signature(
            json_in["model"]["elements"], json_in["model"]["relationships"],
        )
        out_el, out_rel = _signature(json_out["elements"], json_out["relationships"])
        assert in_el == out_el
        assert in_rel == out_rel


# ---------------------------------------------------------------------------
# 4. Event mapping — every row of Appendix A, both directions
# ---------------------------------------------------------------------------

# (event_class, wme_event_type, expected_direction, expected_definition)
_APPENDIX_A_ROWS = [
    # StartEvent
    (StartEvent, "default", EventDirection.CATCH, EventDefinitionType.NONE),
    (StartEvent, "message", EventDirection.CATCH, EventDefinitionType.MESSAGE),
    (StartEvent, "timer", EventDirection.CATCH, EventDefinitionType.TIMER),
    (StartEvent, "conditional", EventDirection.CATCH, EventDefinitionType.CONDITIONAL),
    (StartEvent, "signal", EventDirection.CATCH, EventDefinitionType.SIGNAL),
    (StartEvent, "escalation", EventDirection.CATCH, EventDefinitionType.ESCALATION),
    (StartEvent, "error", EventDirection.CATCH, EventDefinitionType.ERROR),
    (StartEvent, "compensation", EventDirection.CATCH, EventDefinitionType.COMPENSATION),
    (StartEvent, "link", EventDirection.CATCH, EventDefinitionType.LINK),
    # EndEvent
    (EndEvent, "default", EventDirection.THROW, EventDefinitionType.NONE),
    (EndEvent, "message", EventDirection.THROW, EventDefinitionType.MESSAGE),
    (EndEvent, "escalation", EventDirection.THROW, EventDefinitionType.ESCALATION),
    (EndEvent, "error", EventDirection.THROW, EventDefinitionType.ERROR),
    (EndEvent, "compensation", EventDirection.THROW, EventDefinitionType.COMPENSATION),
    (EndEvent, "signal", EventDirection.THROW, EventDefinitionType.SIGNAL),
    (EndEvent, "terminate", EventDirection.THROW, EventDefinitionType.TERMINATE),
    # IntermediateEvent
    (IntermediateEvent, "default", EventDirection.CATCH, EventDefinitionType.NONE),
    (IntermediateEvent, "message-catch", EventDirection.CATCH, EventDefinitionType.MESSAGE),
    (IntermediateEvent, "message-throw", EventDirection.THROW, EventDefinitionType.MESSAGE),
    (IntermediateEvent, "timer-catch", EventDirection.CATCH, EventDefinitionType.TIMER),
    (IntermediateEvent, "timer-throw", EventDirection.THROW, EventDefinitionType.TIMER),
    (IntermediateEvent, "conditional-catch", EventDirection.CATCH, EventDefinitionType.CONDITIONAL),
    (IntermediateEvent, "escalation-throw", EventDirection.THROW, EventDefinitionType.ESCALATION),
    (IntermediateEvent, "link-catch", EventDirection.CATCH, EventDefinitionType.LINK),
    (IntermediateEvent, "link-throw", EventDirection.THROW, EventDefinitionType.LINK),
    (IntermediateEvent, "compensation-throw", EventDirection.THROW, EventDefinitionType.COMPENSATION),
    (IntermediateEvent, "signal-catch", EventDirection.CATCH, EventDefinitionType.SIGNAL),
    (IntermediateEvent, "signal-throw", EventDirection.THROW, EventDefinitionType.SIGNAL),
]


@pytest.mark.parametrize("event_class,event_type,direction,definition", _APPENDIX_A_ROWS)
def test_parse_event_type_table(event_class, event_type, direction, definition):
    got_dir, got_def = parse_event_type(event_class, event_type)
    assert got_dir is direction
    assert got_def is definition


@pytest.mark.parametrize("event_class,event_type,direction,definition", _APPENDIX_A_ROWS)
def test_serialise_event_type_round_trip(event_class, event_type, direction, definition):
    event = event_class(direction=direction, event_definition=definition)
    assert serialise_event_type(event) == event_type


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_model_key_raises(self):
        with pytest.raises(ConversionError, match="missing the 'model' key"):
            process_bpmn_diagram({"title": "x"})

    def test_unknown_element_type_logs_skip(self, caplog):
        elements = {
            "u1": _node("u1", "BPMNNonsense", "?"),
            "t1": _node("t1", "BPMNTask", "A", taskType="default", marker="none"),
        }
        json_data = _envelope(elements, {})
        with caplog.at_level("WARNING"):
            model = process_bpmn_diagram(json_data)
        assert {type(n).__name__ for n in model.all_flow_nodes()} == {"Task"}
        assert any("unknown type 'BPMNNonsense'" in rec.message for rec in caplog.records)

    def test_unknown_task_type_raises_conversion_error(self):
        elements = {
            "t1": _node("t1", "BPMNTask", "A", taskType="bogus", marker="none"),
        }
        with pytest.raises(ConversionError, match="Unknown BPMN task type"):
            process_bpmn_diagram(_envelope(elements, {}))

    def test_unknown_flow_type_raises_conversion_error(self):
        elements = {
            "t1": _node("t1", "BPMNTask", "A", taskType="default", marker="none"),
            "t2": _node("t2", "BPMNTask", "B", taskType="default", marker="none"),
        }
        relationships = {"f1": _flow("f1", "t1", "t2", flow_type="bogus")}
        with pytest.raises(ConversionError, match="Unknown BPMN flowType"):
            process_bpmn_diagram(_envelope(elements, relationships))


# ---------------------------------------------------------------------------
# 6. Direct BUML → JSON tests (no JSON input — exercises the layout fallback)
# ---------------------------------------------------------------------------

def _simple_buml_model():
    """Build a minimal BPMNModel programmatically (no `layout` anywhere)."""
    s = StartEvent(name="start")
    t = Task(name="work", task_type=TaskType.SERVICE)
    e = EndEvent(name="end")
    p = Process(name="P", flow_nodes={s, t, e},
                sequence_flows={SequenceFlow(s, t), SequenceFlow(t, e)})
    return BPMNModel(name="Programmatic", processes={p})


def test_layout_fallback_emits_bounds_for_every_node():
    model = _simple_buml_model()
    out = bpmn_object_to_json(model)
    for entry in out["elements"].values():
        bounds = entry["bounds"]
        assert {"x", "y", "width", "height"} <= set(bounds.keys())
        assert bounds["width"] > 0 and bounds["height"] > 0


def test_layout_fallback_envelope_size_min_800x600():
    model = _simple_buml_model()
    out = bpmn_object_to_json(model)
    assert out["size"]["width"] >= 800
    assert out["size"]["height"] >= 600
