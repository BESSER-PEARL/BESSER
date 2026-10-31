"""BPMN generator: data elements and Diagram Interchange geometry.

Regression tests for issue #627: an exported BPMN file opened in Camunda Modeler
showed sequence-flow arrows "at fixed locations, not connected between steps", and
data objects were not drawn. Causes:

* the Web Modeling Editor stores a relationship ``path`` relative to the
  relationship's ``bounds``, but the generator wrote it verbatim as absolute
  waypoints (and flows created by the modeling assistant carry a placeholder
  ``(0,0)->(100,0)`` path until the editor re-routes them);
* DI shapes and data associations pointed at ``<bpmn:dataObject>`` /
  ``<bpmn:dataStore>`` instead of the visual ``*Reference`` elements, which are
  the only ones bpmn-js / Camunda Modeler render.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from besser.BUML.metamodel.bpmn import (
    BPMNModel,
    DataAssociation,
    DataObject,
    DataStore,
    Process,
    SequenceFlow,
    Task,
)
from besser.generators.bpmn import BPMNGenerator
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.bpmn_diagram_processor import (
    process_bpmn_diagram,
)

_NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "dc": "http://www.omg.org/spec/DD/20100524/DC",
    "di": "http://www.omg.org/spec/DD/20100524/DI",
}


def _q(prefix_local: str) -> str:
    prefix, local = prefix_local.split(":", 1)
    return f"{{{_NS[prefix]}}}{local}"


def _generate(model: BPMNModel, tmp_path) -> ET.Element:
    out = BPMNGenerator(model, output_dir=str(tmp_path)).generate()
    return ET.parse(str(Path(out))).getroot()


def _by_id(root: ET.Element) -> dict:
    return {el.attrib["id"]: el for el in root.iter() if "id" in el.attrib}


def _shape_bounds(root: ET.Element) -> dict:
    """bpmnElement id -> (x, y, w, h) for every BPMNShape."""
    result = {}
    for shape in root.iter(_q("bpmndi:BPMNShape")):
        b = shape.find(_q("dc:Bounds"))
        result[shape.attrib["bpmnElement"]] = tuple(
            float(b.attrib[k]) for k in ("x", "y", "width", "height"))
    return result


def _edge_points(root: ET.Element) -> dict:
    """bpmnElement id -> list of (x, y) waypoints for every BPMNEdge."""
    return {
        edge.attrib["bpmnElement"]: [
            (float(wp.attrib["x"]), float(wp.attrib["y"]))
            for wp in edge.findall(_q("di:waypoint"))
        ]
        for edge in root.iter(_q("bpmndi:BPMNEdge"))
    }


def _on_bounds(point, bounds, tol=0.5) -> bool:
    """Point lies on (within ``tol`` of) the rectangle's border."""
    x, y, w, h = bounds
    px, py = point
    inside_x = x - tol <= px <= x + w + tol
    inside_y = y - tol <= py <= y + h + tol
    on_vertical = inside_y and (abs(px - x) <= tol or abs(px - (x + w)) <= tol)
    on_horizontal = inside_x and (abs(py - y) <= tol or abs(py - (y + h)) <= tol)
    return on_vertical or on_horizontal


def _assert_edges_attached(root: ET.Element) -> None:
    """Every edge starts on its source shape and ends on its target shape."""
    ids = _by_id(root)
    shapes = _shape_bounds(root)
    edges = _edge_points(root)
    assert edges, "expected BPMNEdge elements"
    for flow_id, points in edges.items():
        flow_el = ids[flow_id]
        if flow_el.tag == _q("bpmn:sequenceFlow"):
            source_id, target_id = flow_el.attrib["sourceRef"], flow_el.attrib["targetRef"]
        else:  # data input/output association: host node + child ref
            host = next(el for el in root.iter() if flow_el in list(el))
            if flow_el.tag == _q("bpmn:dataInputAssociation"):
                source_id = flow_el.find(_q("bpmn:sourceRef")).text
                target_id = host.attrib["id"]
            else:
                source_id = host.attrib["id"]
                target_id = flow_el.find(_q("bpmn:targetRef")).text
        assert len(points) >= 2
        assert _on_bounds(points[0], shapes[source_id]), (flow_id, points, shapes[source_id])
        assert _on_bounds(points[-1], shapes[target_id]), (flow_id, points, shapes[target_id])


def _bounds(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


# ---------------------------------------------------------------------------
# Data objects / data stores
# ---------------------------------------------------------------------------

class TestDataElements:
    def _model(self):
        task = Task(name="Assemble", layout={"id": "t1", "bounds": _bounds(200, 100, 110, 60)})
        doc = DataObject(name="Spec sheet", layout={"id": "d1", "bounds": _bounds(220, 250, 40, 50)})
        store = DataStore(name="MES", layout={"id": "ds1", "bounds": _bounds(400, 250, 50, 50)})
        process = Process(name="P", flow_nodes={task}, data_objects={doc})
        process.add_data_association(DataAssociation(source=doc, target=task,
                                                     layout={"id": "a_in"}))
        process.add_data_association(DataAssociation(source=task, target=store,
                                                     layout={"id": "a_out"}))
        return BPMNModel(name="m", processes={process}, data_stores={store})

    def test_data_object_reference_carries_the_element_id(self, tmp_path):
        root = _generate(self._model(), tmp_path)
        ref = root.find(f".//{_q('bpmn:dataObjectReference')}")
        assert ref.attrib["id"] == "d1"
        assert ref.attrib["name"] == "Spec sheet"
        data_object = _by_id(root)[ref.attrib["dataObjectRef"]]
        assert data_object.tag == _q("bpmn:dataObject")
        assert data_object.attrib["id"] == "DataObject_d1"

    def test_data_store_reference_carries_the_element_id(self, tmp_path):
        root = _generate(self._model(), tmp_path)
        ref = root.find(f".//{_q('bpmn:dataStoreReference')}")
        assert ref.attrib["id"] == "ds1"
        store = _by_id(root)[ref.attrib["dataStoreRef"]]
        assert store.tag == _q("bpmn:dataStore")
        assert store in list(root)  # definitions-level

    def test_shapes_and_associations_point_at_references(self, tmp_path):
        root = _generate(self._model(), tmp_path)
        ids = _by_id(root)
        for shape in root.iter(_q("bpmndi:BPMNShape")):
            assert ids[shape.attrib["bpmnElement"]].tag not in (
                _q("bpmn:dataObject"), _q("bpmn:dataStore"))
        data_in = root.find(f".//{_q('bpmn:dataInputAssociation')}")
        data_out = root.find(f".//{_q('bpmn:dataOutputAssociation')}")
        assert ids[data_in.find(_q("bpmn:sourceRef")).text].tag == _q("bpmn:dataObjectReference")
        assert ids[data_out.find(_q("bpmn:targetRef")).text].tag == _q("bpmn:dataStoreReference")

    def test_data_association_edges_attach_to_shapes(self, tmp_path):
        _assert_edges_attached(_generate(self._model(), tmp_path))

    def test_ids_unique_without_layout_ids(self, tmp_path):
        task = Task(name="t", layout={"bounds": _bounds(0, 0, 100, 60)})
        doc = DataObject(name="d")
        process = Process(name="P", flow_nodes={task}, data_objects={doc})
        process.add_data_association(DataAssociation(source=doc, target=task))
        root = _generate(BPMNModel(name="m", processes={process},
                                   data_stores={DataStore(name="s")}), tmp_path)
        all_ids = [el.attrib["id"] for el in root.iter() if "id" in el.attrib]
        assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Edge waypoints
# ---------------------------------------------------------------------------

def _two_task_model(flow_layout):
    a = Task(name="A", layout={"id": "A", "bounds": _bounds(100, 100, 100, 60)})
    b = Task(name="B", layout={"id": "B", "bounds": _bounds(300, 100, 100, 60)})
    flow = SequenceFlow(source=a, target=b, layout={"id": "f", **flow_layout})
    return BPMNModel(name="m", processes={Process(name="P", flow_nodes={a, b},
                                                  sequence_flows={flow})})


class TestEdgeWaypoints:
    def test_sequence_flow_has_source_and_target_refs(self, tmp_path):
        root = _generate(_two_task_model({}), tmp_path)
        flow = _by_id(root)["f"]
        assert flow.attrib["sourceRef"] == "A"
        assert flow.attrib["targetRef"] == "B"

    def test_relative_path_is_offset_by_relationship_bounds(self, tmp_path):
        # The WME stores path points relative to the relationship's bounds.
        layout = {"bounds": _bounds(200, 125, 100, 10),
                  "path": [{"x": 0, "y": 5}, {"x": 100, "y": 5}]}
        root = _generate(_two_task_model(layout), tmp_path)
        assert _edge_points(root)["f"] == [(200, 130), (300, 130)]

    def test_absolute_path_is_kept(self, tmp_path):
        layout = {"bounds": _bounds(190, 120, 120, 20),
                  "path": [{"x": 200, "y": 130}, {"x": 300, "y": 130}]}
        root = _generate(_two_task_model(layout), tmp_path)
        assert _edge_points(root)["f"] == [(200, 130), (300, 130)]

    def test_placeholder_path_is_rerouted_between_shapes(self, tmp_path):
        # What the modeling assistant emits before the editor re-routes the flow.
        layout = {"bounds": _bounds(0, 0, 100, 1),
                  "path": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]}
        root = _generate(_two_task_model(layout), tmp_path)
        assert _edge_points(root)["f"] == [(200, 130), (300, 130)]

    def test_missing_path_is_routed_when_shapes_known(self, tmp_path):
        root = _generate(_two_task_model({}), tmp_path)
        assert _edge_points(root)["f"] == [(200, 130), (300, 130)]

    def test_elbow_route_for_unaligned_shapes(self, tmp_path):
        a = Task(name="A", layout={"bounds": _bounds(0, 0, 100, 60)})
        b = Task(name="B", layout={"bounds": _bounds(300, 100, 100, 60)})
        flow = SequenceFlow(source=a, target=b, layout={"id": "f"})
        model = BPMNModel(name="m", processes={Process(name="P", flow_nodes={a, b},
                                                       sequence_flows={flow})})
        points = _edge_points(_generate(model, tmp_path))["f"]
        assert points == [(100, 30), (200, 30), (200, 130), (300, 130)]

    def test_path_kept_when_endpoint_shapes_unknown(self, tmp_path):
        a = Task(name="A", layout={"bounds": _bounds(0, 0, 100, 60)})
        b = Task(name="B")  # no layout
        flow = SequenceFlow(source=a, target=b, layout={
            "bounds": _bounds(10, 20, 50, 1), "path": [{"x": 0, "y": 0}, {"x": 50, "y": 0}]})
        model = BPMNModel(name="m", processes={Process(name="P", flow_nodes={a, b},
                                                       sequence_flows={flow})})
        root = _generate(model, tmp_path)
        assert list(_edge_points(root).values()) == [[(10, 20), (60, 20)]]


# ---------------------------------------------------------------------------
# End to end: assistant-style WME JSON (issue #627 scenario)
# ---------------------------------------------------------------------------

def _assistant_style_json(n_tasks=9):
    """A start event, ``n_tasks`` sequential tasks each fed by a data object, and an
    end event -- with the placeholder flow geometry the modeling assistant emits."""
    nodes, edges = [], []

    def node(node_id, type_, name, x, y, w, h, **extra):
        nodes.append({"id": node_id, "type": type_, "position": {"x": x, "y": y},
                      "width": w, "height": h, "data": {"name": name, **extra}})

    def edge(edge_id, source, target, edge_type):
        edges.append({
            "id": edge_id, "type": edge_type, "source": source, "target": target,
            "sourceHandle": "Right", "targetHandle": "Left",
            "data": {"name": "", "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
                     "isManuallyLayouted": False},
        })

    node("start", "bpmnStartEvent", "Cell production started", -1000, 10, 40, 40,
         eventType="default")
    previous = "start"
    for i in range(n_tasks):
        node(f"t{i}", "bpmnTask", f"Step {i + 1}", -780 + i * 220, 0, 140, 60,
             taskType="default", marker="none")
        node(f"d{i}", "bpmnDataObject", f"Input {i + 1}", -730 + i * 220, 120, 40, 50)
        edge(f"f{i}", previous, f"t{i}", "BPMNSequenceFlow")
        edge(f"a{i}", f"d{i}", f"t{i}", "BPMNDataAssociationFlow")
        previous = f"t{i}"
    node("end", "bpmnEndEvent", "Cell ready", -780 + n_tasks * 220, 10, 40, 40,
         eventType="default")
    edge("f_end", previous, "end", "BPMNSequenceFlow")
    return {"title": "Cell production", "model": {
        "version": "4.0.0", "type": "BPMNDiagram", "nodes": nodes, "edges": edges,
        "assessments": {}}}


def test_issue_627_export_is_connected_and_keeps_data_objects(tmp_path):
    model = process_bpmn_diagram(_assistant_style_json())
    root = _generate(model, tmp_path)

    flows = root.findall(f".//{_q('bpmn:sequenceFlow')}")
    assert len(flows) == 10
    ids = _by_id(root)
    for flow in flows:
        assert flow.attrib["sourceRef"] in ids
        assert flow.attrib["targetRef"] in ids

    refs = root.findall(f".//{_q('bpmn:dataObjectReference')}")
    assert sorted(r.attrib["name"] for r in refs) == sorted(f"Input {i + 1}" for i in range(9))
    assert len(root.findall(f".//{_q('bpmn:dataInputAssociation')}")) == 9
    shapes = _shape_bounds(root)
    assert all(r.attrib["id"] in shapes for r in refs)

    _assert_edges_attached(root)
    # No two sequence-flow edges collapse onto the same placeholder segment.
    edges = _edge_points(root)
    seq_edges = [tuple(edges[f.attrib["id"]]) for f in flows]
    assert len(set(seq_edges)) == len(seq_edges)


@pytest.mark.parametrize("n_tasks", [1, 3])
def test_generation_is_deterministic_with_rerouting(n_tasks, tmp_path):
    model = process_bpmn_diagram(_assistant_style_json(n_tasks))
    first = BPMNGenerator(model, output_dir=str(tmp_path), file_name="a.bpmn").generate()
    second = BPMNGenerator(model, output_dir=str(tmp_path), file_name="b.bpmn").generate()
    assert Path(first).read_bytes() == Path(second).read_bytes()
