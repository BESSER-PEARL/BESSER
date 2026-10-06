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
            "type": "BPMNDiagram",  # canonical spelling (other fixtures use the legacy "BPMN")
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
        assert payload["type"] == "BPMNDiagram"
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
        assert payload["type"] == "BPMNDiagram"
        types = Counter(n.get("type") for n in _nodes(payload))
        assert types["bpmnStartEvent"] == 1
        assert types["bpmnTask"] == 1
        assert types["bpmnGateway"] == 1
        assert types["bpmnEndEvent"] == 1
        assert len(_edges_by_type(payload, "BPMNSequenceFlow")) == 3

    def test_missing_model_raises_conversion_error(self):
        with pytest.raises(ConversionError):
            bpmn_buml_to_json("x = 1\n")


# ---------------------------------------------------------------------------
# Ported from feature/smart-generator's (v3) suite — same assertions, v4 shape
# ---------------------------------------------------------------------------

from besser.BUML.metamodel.bpmn import (  # noqa: E402
    Collaboration,
    Gateway,
    GatewayType,
    MessageFlow,
    Process,
    SequenceFlow,
    SubProcess,
    Task,
    TaskType,
)


def two_pool_collaboration_fixture():
    """Two pools, each with one task; one MessageFlow between them."""
    return {
        "title": "Buyer-Seller",
        "model": {
            "type": "BPMNDiagram",
            "nodes": [
                _node("p1", "bpmnPool", "Buyer"),
                _node("p2", "bpmnPool", "Seller"),
                _node("t1", "bpmnTask", "Place order", parent_id="p1", taskType="default", marker="none"),
                _node("t2", "bpmnTask", "Ship", parent_id="p2", taskType="default", marker="none"),
            ],
            "edges": [
                _edge("m1", "BPMNMessageFlow", "t1", "t2", name="order"),
            ],
        },
    }


class TestPortedProcessBehaviour:
    def test_layout_ids_stashed(self):
        model = process_bpmn_diagram(simple_process_fixture())
        ids = {(n.layout or {}).get("id") for n in model.all_flow_nodes()}
        assert ids == {"start1", "task1", "gw1", "end1"}

    def test_simple_process_validates(self):
        model = process_bpmn_diagram(simple_process_fixture())
        result = model.validate(raise_exception=False)
        assert result["success"] is True, result["errors"]

    def test_two_pool_builds_collaboration(self):
        model = process_bpmn_diagram(two_pool_collaboration_fixture())
        assert isinstance(model.collaboration, Collaboration)
        assert len(model.collaboration.participants) == 2
        assert len(model.processes) == 2
        assert len(model.collaboration.message_flows) == 1
        msg = next(iter(model.collaboration.message_flows))
        assert isinstance(msg, MessageFlow)
        assert msg.name == "order"

    def test_gateway_default_flow_is_the_gateway_default(self):
        model = process_bpmn_diagram(simple_process_fixture())
        gateway = next(n for n in model.all_flow_nodes() if isinstance(n, Gateway))
        assert gateway.gateway_type is GatewayType.EXCLUSIVE
        defaults = [f for f in gateway.outgoing() if f.is_default]
        assert len(defaults) == 1
        assert gateway.default_flow is defaults[0]

    def test_inner_sequence_flow_lives_in_the_subprocess(self):
        model = process_bpmn_diagram(subprocess_fixture())
        sub = next(n for n in model.all_flow_nodes() if isinstance(n, SubProcess))
        outer = next(iter(model.processes))
        assert sub in outer.flow_nodes
        assert all(f.source.container is sub for f in sub.sequence_flows)

    def test_dangling_endpoint_logs(self, caplog):
        fixture = simple_process_fixture()
        fixture["model"]["edges"].append(_edge("bad", "BPMNSequenceFlow", "start1", "missing"))
        with caplog.at_level("WARNING"):
            process_bpmn_diagram(fixture)
        assert any("dangling endpoint" in rec.message for rec in caplog.records)

    def test_illegal_is_default_logs_and_downgrades(self, caplog):
        # A parallel gateway cannot carry a default flow (BPMN 8.3.13).
        fixture = {
            "title": "Parallel",
            "model": {
                "type": "BPMNDiagram",
                "nodes": [
                    _node("g1", "bpmnGateway", "", gatewayType="parallel"),
                    _node("t1", "bpmnTask", "A", taskType="default", marker="none"),
                    _node("t2", "bpmnTask", "B", taskType="default", marker="none"),
                ],
                "edges": [
                    _edge("fa", "BPMNSequenceFlow", "g1", "t1"),
                    _edge("fb", "BPMNSequenceFlow", "g1", "t2", isDefault=True),
                ],
            },
        }
        with caplog.at_level("WARNING"):
            model = process_bpmn_diagram(fixture)
        flows = next(iter(model.processes)).sequence_flows
        assert all(not f.is_default for f in flows)
        assert any("downgrading to is_default=False" in rec.message for rec in caplog.records)

    def test_unknown_node_type_logs(self, caplog):
        fixture = simple_process_fixture()
        fixture["model"]["nodes"].append(_node("u1", "bpmnNonsense", "?"))
        with caplog.at_level("WARNING"):
            process_bpmn_diagram(fixture)
        assert any("unknown type 'bpmnNonsense'" in rec.message for rec in caplog.records)

    def test_missing_model_key_raises(self):
        with pytest.raises(ConversionError, match="missing the 'model' key"):
            process_bpmn_diagram({"title": "x"})

    def test_highlight_style_round_trips(self):
        fixture = simple_process_fixture()
        fixture["model"]["nodes"][1]["data"]["highlight"] = "#ff0"
        payload = bpmn_object_to_json(process_bpmn_diagram(fixture))
        assert _data(_node_by_name(payload, "Review"))["highlight"] == "#ff0"


class TestPortedObjectToJson:
    def test_envelope_keys_present(self):
        out = bpmn_object_to_json(process_bpmn_diagram(simple_process_fixture()))
        assert out["version"] == "4.0.0"
        assert out["type"] == "BPMNDiagram"
        assert set(out.keys()) >= {
            "version", "type", "title", "size", "interactive", "nodes", "edges", "assessments",
        }

    def test_pool_parent_pointer(self):
        out = bpmn_object_to_json(process_bpmn_diagram(two_pool_collaboration_fixture()))
        pool_ids = {n["id"] for n in _nodes_by_type(out, "bpmnPool")}
        task_parents = {n.get("parentId") for n in _nodes_by_type(out, "bpmnTask")}
        assert task_parents.issubset(pool_ids)

    def test_message_flow_emitted(self):
        out = bpmn_object_to_json(process_bpmn_diagram(two_pool_collaboration_fixture()))
        assert {e["type"] for e in _edges(out)} == {"BPMNMessageFlow"}

    def test_ids_are_reused_from_layout(self):
        out = bpmn_object_to_json(process_bpmn_diagram(simple_process_fixture()))
        assert {n["id"] for n in _nodes(out)} == {"start1", "task1", "gw1", "end1"}
        assert {e["id"] for e in _edges(out)} == {"f1", "f2", "f3"}


def _simple_buml_model():
    """A minimal BPMNModel built programmatically (no ``layout`` anywhere)."""
    s = StartEvent(name="start")
    t = Task(name="work", task_type=TaskType.SERVICE)
    e = EndEvent(name="end")
    p = Process(name="P", flow_nodes={s, t, e},
                sequence_flows={SequenceFlow(s, t), SequenceFlow(t, e)})
    return BPMNModel(name="Programmatic", processes={p})


def test_layout_fallback_emits_geometry_for_every_node():
    out = bpmn_object_to_json(_simple_buml_model())
    assert len(_nodes(out)) == 3
    for node in _nodes(out):
        assert {"x", "y"} <= set(node["position"].keys())
        assert node["width"] > 0 and node["height"] > 0
    assert len(_edges(out)) == 2


def test_layout_fallback_envelope_size_min_800x600():
    out = bpmn_object_to_json(_simple_buml_model())
    assert out["size"]["width"] >= 800
    assert out["size"]["height"] >= 600


class TestPortedBumlWrapper:
    @pytest.mark.parametrize(
        "fixture_fn",
        [simple_process_fixture, pool_lane_fixture, subprocess_fixture, artifact_fixture,
         two_pool_collaboration_fixture],
    )
    def test_full_round_trip_via_builder(self, fixture_fn):
        # JSON -> BUML model -> emitted .py -> safe loader -> BPMNModel -> JSON must agree
        # with bpmn_object_to_json on the model for the load-bearing fields.
        model = process_bpmn_diagram(fixture_fn())
        json_out = bpmn_buml_to_json(bpmn_model_to_code(model))
        direct = bpmn_object_to_json(model)
        assert _type_name_pairs(json_out) == _type_name_pairs(direct)
        assert _containment_by_name(json_out) == _containment_by_name(direct)
        assert Counter(e["type"] for e in _edges(json_out)) == Counter(e["type"] for e in _edges(direct))

    def test_exec_failure_raises_conversion_error(self):
        with pytest.raises(ConversionError, match="failed to execute"):
            bpmn_buml_to_json("undefined_symbol\n")

    def test_no_bpmn_model_raises_conversion_error(self):
        with pytest.raises(ConversionError, match="produced no BPMNModel"):
            bpmn_buml_to_json("x = 1\n")

    def test_finds_model_under_any_variable_name(self):
        source = (
            "from besser.BUML.metamodel.bpmn import BPMNModel, Process, Task\n"
            "task_x = Task(name='x')\n"
            "p = Process(name='P', flow_nodes={task_x})\n"
            "weird_var_name = BPMNModel(name='X', processes={p})\n"
        )
        out = bpmn_buml_to_json(source)
        assert out["type"] == "BPMNDiagram"
        assert len(_nodes(out)) == 1

    def test_syntax_error_raises_conversion_error(self):
        with pytest.raises(ConversionError, match="failed to execute"):
            bpmn_buml_to_json("def broken(:\n    pass\n")


# ---------------------------------------------------------------------------
# Diagram interchange: the .bpmn export must carry the editor's geometry the
# way development's v3 converter did (regression: BPMNEdge DI was dropped and
# shapes inside a pool were placed parent-relative).
# ---------------------------------------------------------------------------

def _positioned_pool_fixture():
    def at(node, x, y):
        node["position"] = {"x": x, "y": y}
        return node

    return {
        "title": "Positioned",
        "model": {
            "type": "BPMNDiagram",
            "nodes": [
                at(_node("pool1", "bpmnPool", "Customer"), 100, 50),
                at(_node("lane1", "bpmnSwimlane", "Agent", parent_id="pool1"), 30, 0),
                at(_node("t1", "bpmnTask", "Call", parent_id="lane1", taskType="default", marker="none"), 40, 20),
                at(_node("e1", "bpmnEndEvent", "Hang up", parent_id="lane1", eventType="default"), 200, 20),
            ],
            "edges": [
                _edge("sf1", "BPMNSequenceFlow", "t1", "e1",
                      points=[{"x": 270, "y": 120}, {"x": 400, "y": 120}, {"x": 400, "y": 140}]),
            ],
        },
    }


def _generated_bpmn_root(tmp_path):
    import xml.etree.ElementTree as ET
    from besser.generators.bpmn.bpmn_generator import BPMNGenerator

    BPMNGenerator(process_bpmn_diagram(_positioned_pool_fixture()), output_dir=str(tmp_path)).generate()
    (bpmn_file,) = tmp_path.glob("*.bpmn")
    return ET.parse(bpmn_file).getroot()


_DI = "{http://www.omg.org/spec/BPMN/20100524/DI}"
_DC = "{http://www.omg.org/spec/DD/20100524/DC}"
_DD_DI = "{http://www.omg.org/spec/DD/20100524/DI}"


class TestDiagramInterchange:
    def test_generated_bpmn_keeps_edge_waypoints(self, tmp_path):
        root = _generated_bpmn_root(tmp_path)
        (edge,) = root.iter(f"{_DI}BPMNEdge")
        waypoints = [(float(w.get("x")), float(w.get("y"))) for w in edge.iter(f"{_DD_DI}waypoint")]
        # v3 ``path`` shape: the points relative to their bounding box.
        assert waypoints == [(0, 0), (130, 0), (130, 20)]

    def test_generated_bpmn_shape_bounds_are_absolute(self, tmp_path):
        root = _generated_bpmn_root(tmp_path)
        bounds = {}
        for shape in root.iter(f"{_DI}BPMNShape"):
            b = shape.find(f"{_DC}Bounds")
            bounds[shape.get("bpmnElement")] = (float(b.get("x")), float(b.get("y")))
        assert bounds["pool1"] == (100, 50)
        assert bounds["lane1"] == (130, 50)
        assert bounds["t1"] == (170, 70)

    def test_round_trip_keeps_relative_positions_and_points(self):
        fixture = _positioned_pool_fixture()
        out = bpmn_object_to_json(process_bpmn_diagram(fixture))
        expected = {n["id"]: (n["position"], n.get("parentId")) for n in _nodes(fixture)}
        assert {n["id"]: (n["position"], n.get("parentId")) for n in _nodes(out)} == expected
        assert _edges(out)[0]["data"]["points"] == _edges(fixture)[0]["data"]["points"]
