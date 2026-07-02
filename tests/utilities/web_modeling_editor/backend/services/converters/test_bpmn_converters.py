"""Tests for the v4 BPMN JSON <-> BUML converters.

All fixtures use the v4 wire shape (flat ``{nodes, edges}`` lists, node ``type``
in lowerCamelCase, containment via top-level ``parentId``, and one edge ``type``
per BPMN flow kind). They are NOT ports of the v3 ``{elements, relationships}``
fixtures. Helper idioms mirror
``tests/utilities/web_modeling_editor/backend/services/converters/test_converter_roundtrip.py``.
"""

from collections import Counter

import pytest

from besser.BUML.metamodel.bpmn import (
    BPMNModel,
    EndEvent,
    EventDefinitionType,
    EventDirection,
    IntermediateEvent,
    StartEvent,
)
from besser.utilities.buml_code_builder.bpmn_model_builder import bpmn_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.bpmn_event_mapping import (
    parse_event_type,
    serialise_event_type,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.bpmn_diagram_processor import (
    process_bpmn_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.bpmn_diagram_converter import (
    bpmn_object_to_json,
    bpmn_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


# ---------------------------------------------------------------------------
# v4 helpers (mirror test_converter_roundtrip.py)
# ---------------------------------------------------------------------------

def _model(payload):
    if isinstance(payload, dict) and "nodes" in payload and "edges" in payload:
        return payload
    return (payload or {}).get("model") or {}


def _nodes(payload):
    return _model(payload).get("nodes") or []


def _edges(payload):
    return _model(payload).get("edges") or []


def _nodes_by_type(payload, node_type):
    return [n for n in _nodes(payload) if n.get("type") == node_type]


def _edges_by_type(payload, edge_type):
    return [e for e in _edges(payload) if e.get("type") == edge_type]


def _data(node):
    return node.get("data") or {}


def _node_by_id(payload, node_id):
    for n in _nodes(payload):
        if n.get("id") == node_id:
            return n
    return None


def _node_by_name(payload, name):
    for n in _nodes(payload):
        if (_data(n).get("name") or "") == name:
            return n
    return None


def _type_name_pairs(payload):
    """Multiset of ``(type, data.name)`` over every node — a shape fingerprint."""
    return Counter((n.get("type"), _data(n).get("name") or "") for n in _nodes(payload))


def _reprocess(emitted):
    """Wrap a ``bpmn_object_to_json`` payload back into the ``{title, model}``
    envelope that ``process_bpmn_diagram`` consumes (same contract as
    ``process_class_diagram``)."""
    return process_bpmn_diagram({"title": emitted.get("title"), "model": emitted})


def _containment_by_name(payload):
    """Map each node's ``data.name`` -> its parent node's ``data.name`` (or None)."""
    id_to_name = {n.get("id"): _data(n).get("name") or "" for n in _nodes(payload)}
    out = {}
    for n in _nodes(payload):
        parent_id = n.get("parentId")
        out[_data(n).get("name") or ""] = id_to_name.get(parent_id) if parent_id else None
    return out


# ---------------------------------------------------------------------------
# v4 fixtures — plain dicts, {nodes, edges}
# ---------------------------------------------------------------------------

def _node(node_id, node_type, name, *, parent_id=None, **data):
    payload = {"name": name}
    payload.update(data)
    node = {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "width": 100,
        "height": 60,
        "data": payload,
    }
    if parent_id:
        node["parentId"] = parent_id
    return node


def _edge(edge_id, edge_type, source, target, name="", **data):
    edge_data = {"name": name}
    edge_data.update(data)
    return {
        "id": edge_id,
        "type": edge_type,
        "source": source,
        "target": target,
        "sourceHandle": "Right",
        "targetHandle": "Left",
        "data": edge_data,
    }


def simple_process_fixture():
    """start(default) -> task(user) -> gateway(exclusive) -> end(terminate).
    No pools; everything lands in one synthetic process. The gateway->end flow
    is the default flow."""
    return {
        "title": "Order Handling",
        "model": {
            "type": "BPMN",
            "nodes": [
                _node("start1", "bpmnStartEvent", "Start", eventType="default"),
                _node("task1", "bpmnTask", "Review", taskType="user", marker="none"),
                _node("gw1", "bpmnGateway", "Approved?", gatewayType="exclusive"),
                _node("end1", "bpmnEndEvent", "Done", eventType="terminate"),
            ],
            "edges": [
                _edge("f1", "BPMNSequenceFlow", "start1", "task1"),
                _edge("f2", "BPMNSequenceFlow", "task1", "gw1"),
                _edge("f3", "BPMNSequenceFlow", "gw1", "end1", name="yes", isDefault=True),
            ],
        },
    }


def pool_lane_fixture():
    """A pool containing one lane, which contains a Task. Exercises the
    forward-compat ``bpmnSwimlane`` -> Lane mapping and lane containment."""
    return {
        "title": "Sales",
        "model": {
            "type": "BPMNDiagram",  # exercise the alternate accepted spelling
            "nodes": [
                _node("pool1", "bpmnPool", "Customer"),
                _node("lane1", "bpmnSwimlane", "Agent", parent_id="pool1"),
                _node("t1", "bpmnTask", "Call", parent_id="lane1", taskType="default", marker="none"),
                _node("e1", "bpmnEndEvent", "Hang up", parent_id="lane1", eventType="default"),
            ],
            "edges": [
                _edge("sf1", "BPMNSequenceFlow", "t1", "e1"),
            ],
        },
    }


def subprocess_fixture():
    """A top-level SubProcess with a child Task (via parentId) and an internal
    sequence flow between two children."""
    return {
        "title": "Nested",
        "model": {
            "type": "BPMN",
            "nodes": [
                _node("sub1", "bpmnSubprocess", "Fulfil", marker="none"),
                _node("cs", "bpmnStartEvent", "SubStart", parent_id="sub1", eventType="default"),
                _node("ct", "bpmnTask", "Pack", parent_id="sub1", taskType="default", marker="none"),
            ],
            "edges": [
                _edge("csf", "BPMNSequenceFlow", "cs", "ct"),
            ],
        },
    }


def artifact_fixture():
    """A Task, a TextAnnotation, and a BPMNAssociationFlow between them."""
    return {
        "title": "Annotated",
        "model": {
            "type": "BPMN",
            "nodes": [
                _node("task1", "bpmnTask", "Do it", taskType="default", marker="none"),
                _node("note1", "bpmnAnnotation", "please hurry"),
            ],
            "edges": [
                _edge("a1", "BPMNAssociationFlow", "task1", "note1"),
            ],
        },
    }


# ---------------------------------------------------------------------------
# TestProcessBpmnDiagram  (json -> buml)
# ---------------------------------------------------------------------------

class TestProcessBpmnDiagram:
    def test_simple_process_builds_all_nodes_and_flows(self):
        model = process_bpmn_diagram(simple_process_fixture())
        assert isinstance(model, BPMNModel)
        assert model.name == "Order Handling"
        assert len(model.processes) == 1
        process = next(iter(model.processes))
        names = {n.name for n in process.flow_nodes}
        assert names == {"Start", "Review", "Approved?", "Done"}
        assert len(process.sequence_flows) == 3

    def test_task_type_and_gateway_type_parsed(self):
        model = process_bpmn_diagram(simple_process_fixture())
        process = next(iter(model.processes))
        task = next(n for n in process.flow_nodes if n.name == "Review")
        gateway = next(n for n in process.flow_nodes if n.name == "Approved?")
        assert task.task_type.value == "user"
        assert gateway.gateway_type.value == "exclusive"

    def test_event_direction_and_definition_parsed(self):
        model = process_bpmn_diagram(simple_process_fixture())
        process = next(iter(model.processes))
        start = next(n for n in process.flow_nodes if n.name == "Start")
        end = next(n for n in process.flow_nodes if n.name == "Done")
        assert isinstance(start, StartEvent)
        assert start.direction is EventDirection.CATCH
        assert start.event_definition is EventDefinitionType.NONE
        assert isinstance(end, EndEvent)
        assert end.direction is EventDirection.THROW
        assert end.event_definition is EventDefinitionType.TERMINATE

    def test_default_sequence_flow_marked(self):
        model = process_bpmn_diagram(simple_process_fixture())
        process = next(iter(model.processes))
        defaults = [f for f in process.sequence_flows if f.is_default]
        assert len(defaults) == 1
        assert defaults[0].name == "yes"

    def test_pool_and_lane_containment(self):
        model = process_bpmn_diagram(pool_lane_fixture())
        assert model.collaboration is not None
        assert len(model.collaboration.participants) == 1
        participant = next(iter(model.collaboration.participants))
        assert participant.name == "Customer"
        assert participant.process is not None
        assert len(participant.process.lanes) == 1
        lane = next(iter(participant.process.lanes))
        assert lane.name == "Agent"
        # The Task and EndEvent are members of the lane.
        lane_member_names = {n.name for n in lane.flow_nodes}
        assert lane_member_names == {"Call", "Hang up"}

    def test_subprocess_containment(self):
        model = process_bpmn_diagram(subprocess_fixture())
        process = next(iter(model.processes))
        subs = [n for n in process.flow_nodes if n.name == "Fulfil"]
        assert len(subs) == 1
        sub = subs[0]
        child_names = {n.name for n in sub.flow_nodes}
        assert child_names == {"SubStart", "Pack"}
        assert len(sub.sequence_flows) == 1

    def test_dangling_flow_is_skipped(self):
        fixture = simple_process_fixture()
        fixture["model"]["edges"].append(
            _edge("bad", "BPMNSequenceFlow", "start1", "does-not-exist")
        )
        model = process_bpmn_diagram(fixture)
        process = next(iter(model.processes))
        # The dangling flow is dropped; the 3 valid flows remain.
        assert len(process.sequence_flows) == 3

    def test_unknown_node_type_is_skipped(self):
        fixture = simple_process_fixture()
        fixture["model"]["nodes"].append(_node("weird", "bpmnUnicorn", "Sparkle"))
        model = process_bpmn_diagram(fixture)
        process = next(iter(model.processes))
        assert "Sparkle" not in {n.name for n in process.flow_nodes}

    def test_non_bpmn_edge_type_is_ignored(self):
        fixture = simple_process_fixture()
        fixture["model"]["edges"].append(
            _edge("cl", "CommentLink", "start1", "task1")
        )
        # Should not raise; the CommentLink is simply not a BPMN flow.
        model = process_bpmn_diagram(fixture)
        process = next(iter(model.processes))
        assert len(process.sequence_flows) == 3


# ---------------------------------------------------------------------------
# TestBpmnObjectToJson  (buml -> json)
# ---------------------------------------------------------------------------

class TestBpmnObjectToJson:
    def test_node_type_strings(self):
        model = process_bpmn_diagram(simple_process_fixture())
        payload = bpmn_object_to_json(model)
        assert payload["type"] == "BPMN"
        assert payload["version"] == "4.0.0"
        types = {n.get("type") for n in _nodes(payload)}
        assert types == {"bpmnStartEvent", "bpmnTask", "bpmnGateway", "bpmnEndEvent"}

    def test_edge_type_strings(self):
        model = process_bpmn_diagram(simple_process_fixture())
        payload = bpmn_object_to_json(model)
        assert len(_edges_by_type(payload, "BPMNSequenceFlow")) == 3

    def test_data_carries_task_gateway_event_fields(self):
        model = process_bpmn_diagram(simple_process_fixture())
        payload = bpmn_object_to_json(model)
        task = _node_by_name(payload, "Review")
        gateway = _node_by_name(payload, "Approved?")
        end = _node_by_name(payload, "Done")
        assert _data(task)["taskType"] == "user"
        assert _data(task)["marker"] == "none"
        assert _data(gateway)["gatewayType"] == "exclusive"
        assert _data(end)["eventType"] == "terminate"

    def test_parent_id_propagation_pool_lane(self):
        model = process_bpmn_diagram(pool_lane_fixture())
        payload = bpmn_object_to_json(model)
        pool = _node_by_name(payload, "Customer")
        lane = _node_by_name(payload, "Agent")
        task = _node_by_name(payload, "Call")
        assert pool.get("parentId") is None
        assert lane.get("parentId") == pool.get("id")
        assert task.get("parentId") == lane.get("id")

    def test_subprocess_child_parent_id(self):
        model = process_bpmn_diagram(subprocess_fixture())
        payload = bpmn_object_to_json(model)
        sub = _node_by_name(payload, "Fulfil")
        child = _node_by_name(payload, "Pack")
        assert child.get("parentId") == sub.get("id")

    def test_default_flow_emits_is_default(self):
        model = process_bpmn_diagram(simple_process_fixture())
        payload = bpmn_object_to_json(model)
        default_edges = [e for e in _edges(payload) if _data(e).get("isDefault")]
        assert len(default_edges) == 1

    def test_sequence_flow_edge_data_has_both_name_and_label(self):
        """Regression guard: BPMNDiagramEdge.tsx reads ``data.label`` while every
        other v4 edge type reads ``data.name`` — the converter must emit both."""
        model = process_bpmn_diagram(simple_process_fixture())
        payload = bpmn_object_to_json(model)
        named = next(e for e in _edges(payload) if _data(e).get("name") == "yes")
        assert _data(named)["name"] == "yes"
        assert _data(named)["label"] == "yes"

    def test_association_edge_type(self):
        model = process_bpmn_diagram(artifact_fixture())
        payload = bpmn_object_to_json(model)
        assert len(_edges_by_type(payload, "BPMNAssociationFlow")) == 1
        assert _node_by_name(payload, "please hurry")["type"] == "bpmnAnnotation"


# ---------------------------------------------------------------------------
# TestRoundTripIdentity
# ---------------------------------------------------------------------------

class TestRoundTripIdentity:
    @pytest.mark.parametrize(
        "fixture_fn",
        [simple_process_fixture, pool_lane_fixture, subprocess_fixture, artifact_fixture],
    )
    def test_process_emit_process_is_stable(self, fixture_fn):
        json1 = bpmn_object_to_json(process_bpmn_diagram(fixture_fn()))
        json2 = bpmn_object_to_json(_reprocess(json1))

        assert _type_name_pairs(json1) == _type_name_pairs(json2)
        assert _containment_by_name(json1) == _containment_by_name(json2)
        assert Counter(e.get("type") for e in _edges(json1)) == \
            Counter(e.get("type") for e in _edges(json2))

    def test_ids_are_preserved_across_round_trip(self):
        json1 = bpmn_object_to_json(process_bpmn_diagram(simple_process_fixture()))
        json2 = bpmn_object_to_json(_reprocess(json1))
        assert {n["id"] for n in _nodes(json1)} == {n["id"] for n in _nodes(json2)}


# ---------------------------------------------------------------------------
# Event mapping table  (bpmn_event_mapping.py is unmodified)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "event_cls, wire, direction, definition",
    [
        (StartEvent, "default", EventDirection.CATCH, EventDefinitionType.NONE),
        (StartEvent, "message", EventDirection.CATCH, EventDefinitionType.MESSAGE),
        (StartEvent, "timer", EventDirection.CATCH, EventDefinitionType.TIMER),
        (EndEvent, "default", EventDirection.THROW, EventDefinitionType.NONE),
        (EndEvent, "terminate", EventDirection.THROW, EventDefinitionType.TERMINATE),
        (IntermediateEvent, "default", EventDirection.CATCH, EventDefinitionType.NONE),
        (IntermediateEvent, "message-catch", EventDirection.CATCH, EventDefinitionType.MESSAGE),
        (IntermediateEvent, "timer-throw", EventDirection.THROW, EventDefinitionType.TIMER),
    ],
)
def test_parse_event_type_table(event_cls, wire, direction, definition):
    assert parse_event_type(event_cls, wire) == (direction, definition)


@pytest.mark.parametrize(
    "event_cls, wire",
    [
        (StartEvent, "default"),
        (StartEvent, "message"),
        (EndEvent, "terminate"),
        (IntermediateEvent, "message-catch"),
        (IntermediateEvent, "timer-throw"),
    ],
)
def test_serialise_event_type_round_trip(event_cls, wire):
    direction, definition = parse_event_type(event_cls, wire)
    event = event_cls(name="e", direction=direction, event_definition=definition)
    assert serialise_event_type(event) == wire


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def _with_node(self, node):
        return {"title": "Bad", "model": {"type": "BPMN", "nodes": [node], "edges": []}}

    def test_bad_task_type_raises(self):
        with pytest.raises(ConversionError):
            process_bpmn_diagram(self._with_node(
                _node("t", "bpmnTask", "X", taskType="bogus", marker="none")
            ))

    def test_bad_gateway_type_raises(self):
        with pytest.raises(ConversionError):
            process_bpmn_diagram(self._with_node(
                _node("g", "bpmnGateway", "X", gatewayType="bogus")
            ))

    def test_bad_event_type_raises(self):
        with pytest.raises(ConversionError):
            process_bpmn_diagram(self._with_node(
                _node("s", "bpmnStartEvent", "X", eventType="bogus")
            ))

    def test_bad_loop_marker_raises(self):
        with pytest.raises(ConversionError):
            process_bpmn_diagram(self._with_node(
                _node("t", "bpmnTask", "X", taskType="default", marker="bogus")
            ))

    def test_bad_edge_type_raises(self):
        # A BPMN-flavoured but unrecognised edge ``type`` that reaches
        # ``_build_flow`` raises ConversionError. (A wholly foreign type like
        # ``CommentLink`` is skipped earlier by process_bpmn_diagram; this
        # exercises _build_flow's guard directly.)
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.bpmn_diagram_processor import (
            _build_flow,
        )
        with pytest.raises(ConversionError):
            _build_flow({"type": "BPMNBogusFlow", "data": {}}, object(), object())


# ---------------------------------------------------------------------------
# TestBpmnToJsonWrapper  (bpmn_buml_to_json from generated .py source)
# ---------------------------------------------------------------------------

class TestBpmnToJsonWrapper:
    def test_buml_source_round_trips_to_json(self, tmp_path):
        model = process_bpmn_diagram(simple_process_fixture())
        py_path = tmp_path / "bpmn_model.py"
        bpmn_model_to_code(model=model, file_path=str(py_path))
        source = py_path.read_text(encoding="utf-8")

        payload = bpmn_buml_to_json(source)
        assert payload["type"] == "BPMN"
        types = Counter(n.get("type") for n in _nodes(payload))
        assert types["bpmnStartEvent"] == 1
        assert types["bpmnTask"] == 1
        assert types["bpmnGateway"] == 1
        assert types["bpmnEndEvent"] == 1
        assert len(_edges_by_type(payload, "BPMNSequenceFlow")) == 3

    def test_missing_model_raises_conversion_error(self):
        with pytest.raises(ConversionError):
            bpmn_buml_to_json("x = 1\n")
