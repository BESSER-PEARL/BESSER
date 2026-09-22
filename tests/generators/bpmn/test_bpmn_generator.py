"""Tests for the BPMN generator (vendor-neutral BPMN 2.0 XML).

Covers per-fixture XML generation, id strategy, DI emission, and determinism.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from besser.BUML.metamodel.bpmn import (
    BPMNModel,
    Collaboration,
    EndEvent,
    EventDefinitionType,
    EventDirection,
    Gateway,
    GatewayType,
    IntermediateEvent,
    LoopCharacteristics,
    Participant,
    Process,
    StartEvent,
    Task,
    TaskType,
)
from besser.generators.bpmn import BPMNGenerator
from tests.bpmn_models import (
    _gateway_model,
    _lane_model,
    _poolless_model,
    _subprocess_model,
    _two_pool_model,
)


# ---------------------------------------------------------------------------
# XML helpers — keep the assertions readable.
# ---------------------------------------------------------------------------

_NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "dc": "http://www.omg.org/spec/DD/20100524/DC",
    "di": "http://www.omg.org/spec/DD/20100524/DI",
}


def _generate(model: BPMNModel, tmp_path) -> Path:
    gen = BPMNGenerator(model, output_dir=str(tmp_path))
    out = gen.generate()
    return Path(out)


def _parse(path: Path) -> ET.Element:
    tree = ET.parse(str(path))
    return tree.getroot()


def _findall(root, prefix_local: str):
    prefix, local = prefix_local.split(":", 1)
    return root.findall(f".//{{{_NS[prefix]}}}{local}")


def _find(root, prefix_local: str):
    prefix, local = prefix_local.split(":", 1)
    return root.find(f".//{{{_NS[prefix]}}}{local}")


# ---------------------------------------------------------------------------
# A. Per-fixture XML generation
# ---------------------------------------------------------------------------

class TestPerFixtureXmlGeneration:
    def test_poolless_three_nodes_two_flows(self, tmp_path):
        root = _parse(_generate(_poolless_model(), tmp_path))
        assert len(_findall(root, "bpmn:process")) == 1
        assert len(_findall(root, "bpmn:startEvent")) == 1
        assert len(_findall(root, "bpmn:userTask")) == 1
        assert len(_findall(root, "bpmn:endEvent")) == 1
        assert len(_findall(root, "bpmn:sequenceFlow")) == 2

    def test_poolless_incoming_outgoing_counts_match_metamodel(self, tmp_path):
        model = _poolless_model()
        root = _parse(_generate(model, tmp_path))
        # Every metamodel flow node's <incoming>/<outgoing> count should match
        # the metamodel's incoming()/outgoing() count.
        for node in model.all_flow_nodes():
            # Find the emitted element by tag-and-name match (names are unique here).
            matching = [e for e in root.iter()
                        if e.tag.endswith("}" + _wme_xml_tag(node))
                        and e.attrib.get("name", "") == (node.name or "")]
            assert len(matching) == 1, f"expected exactly one element for {node!r}"
            el = matching[0]
            assert len(el.findall(f"{{{_NS['bpmn']}}}incoming")) == len(node.incoming())
            assert len(el.findall(f"{{{_NS['bpmn']}}}outgoing")) == len(node.outgoing())

    def test_two_pool_collaboration(self, tmp_path):
        root = _parse(_generate(_two_pool_model(), tmp_path))
        assert len(_findall(root, "bpmn:collaboration")) == 1
        assert len(_findall(root, "bpmn:participant")) == 2
        assert len(_findall(root, "bpmn:messageFlow")) == 1
        assert len(_findall(root, "bpmn:process")) == 2

    def test_gateway_default_attribute(self, tmp_path):
        root = _parse(_generate(_gateway_model(), tmp_path))
        gateway = _find(root, "bpmn:exclusiveGateway")
        assert gateway is not None
        default_flow_id = gateway.attrib.get("default")
        assert default_flow_id is not None
        # The referenced flow exists and has name="default".
        default_flow = next(
            (f for f in _findall(root, "bpmn:sequenceFlow")
             if f.attrib.get("id") == default_flow_id),
            None,
        )
        assert default_flow is not None
        assert default_flow.attrib.get("name") == "default"

    def test_subprocess_inner_nodes_nested(self, tmp_path):
        root = _parse(_generate(_subprocess_model(), tmp_path))
        sub = _find(root, "bpmn:subProcess")
        assert sub is not None
        # Inner task is a CHILD of subProcess, not of the outer process.
        sub_tasks = sub.findall(f"{{{_NS['bpmn']}}}task") + \
                    sub.findall(f"{{{_NS['bpmn']}}}userTask")
        assert len(sub_tasks) == 1
        # Inner sequence flow also nested.
        sub_flows = sub.findall(f"{{{_NS['bpmn']}}}sequenceFlow")
        assert len(sub_flows) == 1

    def test_lanes_emit_laneset_with_refs(self, tmp_path):
        root = _parse(_generate(_lane_model(), tmp_path))
        lane_set = _find(root, "bpmn:laneSet")
        assert lane_set is not None
        lane = lane_set.find(f"{{{_NS['bpmn']}}}lane")
        assert lane is not None
        assert lane.attrib.get("name") == "Reviewer"
        refs = lane.findall(f"{{{_NS['bpmn']}}}flowNodeRef")
        assert len(refs) == 1

    @pytest.mark.parametrize("loop,marker_tag,expected_isseq", [
        (LoopCharacteristics.STANDARD_LOOP, "standardLoopCharacteristics", None),
        (LoopCharacteristics.PARALLEL_MI, "multiInstanceLoopCharacteristics", "false"),
        (LoopCharacteristics.SEQUENTIAL_MI, "multiInstanceLoopCharacteristics", "true"),
    ])
    def test_loop_characteristics_marker(self, tmp_path, loop, marker_tag, expected_isseq):
        t = Task(name="x", loop_characteristics=loop)
        model = BPMNModel(name="loops", processes={Process(name="P", flow_nodes={t})})
        root = _parse(_generate(model, tmp_path))
        marker = _find(root, f"bpmn:{marker_tag}")
        assert marker is not None
        if expected_isseq is not None:
            assert marker.attrib.get("isSequential") == expected_isseq

    @pytest.mark.parametrize("event_class,event_def,direction,expected_tag,expected_def_tag", [
        (StartEvent, EventDefinitionType.MESSAGE, EventDirection.CATCH,
         "startEvent", "messageEventDefinition"),
        (EndEvent, EventDefinitionType.TERMINATE, EventDirection.THROW,
         "endEvent", "terminateEventDefinition"),
        (IntermediateEvent, EventDefinitionType.TIMER, EventDirection.CATCH,
         "intermediateCatchEvent", "timerEventDefinition"),
        (IntermediateEvent, EventDefinitionType.SIGNAL, EventDirection.THROW,
         "intermediateThrowEvent", "signalEventDefinition"),
    ])
    def test_event_definition_nested(self, tmp_path, event_class, event_def, direction,
                                     expected_tag, expected_def_tag):
        kwargs = {"event_definition": event_def}
        if event_class is IntermediateEvent:
            kwargs["direction"] = direction
        ev = event_class(name="e", **kwargs)
        # Need at least one other node to make a valid process (E2 needs flows
        # to share a container; we keep it minimal — just the event).
        model = BPMNModel(name="ev", processes={Process(name="P", flow_nodes={ev})})
        root = _parse(_generate(model, tmp_path))
        assert _find(root, f"bpmn:{expected_tag}") is not None
        assert _find(root, f"bpmn:{expected_def_tag}") is not None


# ---------------------------------------------------------------------------
# B. Identifier strategy
# ---------------------------------------------------------------------------

class TestIdStrategy:
    def test_valid_ncname_layout_id_reused(self, tmp_path):
        t = Task(name="x", layout={"id": "Task_X"})
        model = BPMNModel(name="m", processes={Process(name="P", flow_nodes={t})})
        root = _parse(_generate(model, tmp_path))
        # The userTask carries the reused id.
        task_el = _find(root, "bpmn:task")
        assert task_el is not None
        assert task_el.attrib.get("id") == "Task_X"

    def test_invalid_ncname_layout_id_replaced(self, tmp_path):
        # Starts with a digit and contains a space — NCName-invalid.
        t = Task(name="x", layout={"id": "5f3a-bad space"})
        model = BPMNModel(name="m", processes={Process(name="P", flow_nodes={t})})
        root = _parse(_generate(model, tmp_path))
        task_el = _find(root, "bpmn:task")
        assert task_el is not None
        new_id = task_el.attrib.get("id")
        assert new_id != "5f3a-bad space"
        assert new_id.startswith("Task_")

    def test_duplicate_stashed_ids_get_suffixed(self, tmp_path):
        t1 = Task(name="a", layout={"id": "Task_X"})
        t2 = Task(name="b", layout={"id": "Task_X"})
        model = BPMNModel(name="m",
                          processes={Process(name="P", flow_nodes={t1, t2})})
        root = _parse(_generate(model, tmp_path))
        ids = sorted(e.attrib["id"] for e in _findall(root, "bpmn:task"))
        # One keeps "Task_X", the other gets a "_1" suffix.
        assert "Task_X" in ids
        assert "Task_X_1" in ids
        assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# C. Diagram Interchange emission
# ---------------------------------------------------------------------------

class TestDiagramInterchange:
    def test_emit_when_layout_present(self, tmp_path):
        t = Task(name="x", layout={"id": "Task_X",
                                   "bounds": {"x": 100, "y": 100,
                                              "width": 110, "height": 60}})
        model = BPMNModel(name="m", processes={Process(name="P", flow_nodes={t})})
        root = _parse(_generate(model, tmp_path))
        assert _find(root, "bpmndi:BPMNDiagram") is not None
        shape = _find(root, "bpmndi:BPMNShape")
        assert shape is not None
        assert shape.attrib.get("bpmnElement") == "Task_X"
        bounds = shape.find(f"{{{_NS['dc']}}}Bounds")
        assert bounds is not None
        assert bounds.attrib == {"x": "100", "y": "100",
                                 "width": "110", "height": "60"}

    def test_skip_when_layout_absent(self, tmp_path):
        root = _parse(_generate(_poolless_model(), tmp_path))
        assert _find(root, "bpmndi:BPMNDiagram") is None

    def test_plane_targets_collaboration_in_poolful_diagrams(self, tmp_path):
        # Build a two-pool model with layout on each pool so DI emits.
        t1 = Task(name="t1", layout={"bounds": {"x": 100, "y": 100,
                                                "width": 110, "height": 60}})
        t2 = Task(name="t2", layout={"bounds": {"x": 100, "y": 300,
                                                "width": 110, "height": 60}})
        p1 = Process(name="P1", flow_nodes={t1})
        p2 = Process(name="P2", flow_nodes={t2})
        part1 = Participant(name="A", process=p1,
                            layout={"id": "Pool_A",
                                    "bounds": {"x": 0, "y": 0,
                                               "width": 800, "height": 200}})
        part2 = Participant(name="B", process=p2,
                            layout={"id": "Pool_B",
                                    "bounds": {"x": 0, "y": 200,
                                               "width": 800, "height": 200}})
        coll = Collaboration(name="C", participants={part1, part2})
        model = BPMNModel(name="m", processes={p1, p2}, collaboration=coll)
        root = _parse(_generate(model, tmp_path))
        plane = _find(root, "bpmndi:BPMNPlane")
        assert plane is not None
        # The plane references the collaboration's id, not a process's.
        coll_el = _find(root, "bpmn:collaboration")
        assert plane.attrib.get("bpmnElement") == coll_el.attrib.get("id")


# ---------------------------------------------------------------------------
# D. Determinism + edge cases
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_model_same_bytes(self, tmp_path):
        # The model must be the SAME instance — sort_by_timestamp keys off the
        # NamedElement timestamp; building a fresh model would give fresh timestamps.
        model = _poolless_model()
        out1 = tmp_path / "first.bpmn"
        out2 = tmp_path / "second.bpmn"
        BPMNGenerator(model, output_dir=str(tmp_path), file_name="first.bpmn").generate()
        BPMNGenerator(model, output_dir=str(tmp_path), file_name="second.bpmn").generate()
        assert out1.read_bytes() == out2.read_bytes()

    def test_empty_model_produces_valid_minimal_xml(self, tmp_path):
        model = BPMNModel(name="empty")
        root = _parse(_generate(model, tmp_path))
        # Just <bpmn:definitions> with namespaces and the targetNamespace.
        assert root.tag == f"{{{_NS['bpmn']}}}definitions"
        assert root.attrib.get("targetNamespace") == "http://besser-pearl.org/bpmn"
        # No processes, no collaboration, no DI section.
        assert _find(root, "bpmn:process") is None
        assert _find(root, "bpmndi:BPMNDiagram") is None


# ---------------------------------------------------------------------------
# E. Smoke: file is parseable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_fn", [
    _poolless_model, _two_pool_model, _gateway_model, _subprocess_model, _lane_model,
])
def test_emitted_file_parses_without_error(fixture_fn, tmp_path):
    path = _generate(fixture_fn(), tmp_path)
    # Round-trip through ElementTree — any malformed XML would raise.
    tree = ET.parse(str(path))
    assert tree.getroot().tag == f"{{{_NS['bpmn']}}}definitions"


# ---------------------------------------------------------------------------
# Internal helpers used by the assertion helpers above
# ---------------------------------------------------------------------------

def _wme_xml_tag(node) -> str:
    """Return the BPMN XML tag name for a metamodel flow node — duplicates the
    generator's dispatch but kept here so tests don't depend on private helpers."""
    if isinstance(node, Task):
        return {
            TaskType.DEFAULT: "task",
            TaskType.USER: "userTask",
            TaskType.SERVICE: "serviceTask",
            TaskType.SEND: "sendTask",
            TaskType.RECEIVE: "receiveTask",
            TaskType.MANUAL: "manualTask",
            TaskType.BUSINESS_RULE: "businessRuleTask",
            TaskType.SCRIPT: "scriptTask",
        }[node.task_type]
    if isinstance(node, StartEvent):
        return "startEvent"
    if isinstance(node, EndEvent):
        return "endEvent"
    if isinstance(node, IntermediateEvent):
        return ("intermediateCatchEvent" if node.direction is EventDirection.CATCH
                else "intermediateThrowEvent")
    if isinstance(node, Gateway):
        return {
            GatewayType.EXCLUSIVE: "exclusiveGateway",
            GatewayType.INCLUSIVE: "inclusiveGateway",
            GatewayType.PARALLEL: "parallelGateway",
            GatewayType.COMPLEX: "complexGateway",
            GatewayType.EVENT_BASED: "eventBasedGateway",
        }[node.gateway_type]
    return type(node).__name__
