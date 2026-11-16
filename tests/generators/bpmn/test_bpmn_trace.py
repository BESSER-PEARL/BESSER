"""BPMNGenerator records which lines of the .bpmn file serialise which model element, by the ids the
elements have (``BPMNElement.id``; the web editor's processor gives each element its editor id)."""

from __future__ import annotations

from besser.BUML.metamodel.bpmn import (
    BPMNModel, EndEvent, Lane, MessageFlow, Collaboration, Participant, Process, SequenceFlow, StartEvent, SubProcess,
    Task, TextAnnotation,
)
from besser.generators.bpmn import BPMNGenerator
from besser.utilities.web_modeling_editor.backend.services.converters import process_bpmn_diagram
from tests.bpmn_models import _subprocess_model, _two_pool_model


def _bounds(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


def _editor_diagram() -> dict:
    """A start event, two tasks each fed by a data object, and an end event, as the web editor sends them."""
    elements, relationships = {}, {}

    def node(elem_id, type_, name, x, y, w, h, **extra):
        elements[elem_id] = {"id": elem_id, "name": name, "type": type_, "owner": None,
                             "bounds": _bounds(x, y, w, h), **extra}

    def rel(rel_id, source, target, flow_type):
        relationships[rel_id] = {
            "id": rel_id, "name": "", "type": "BPMNFlow", "owner": None, "bounds": _bounds(0, 0, 100, 1),
            "path": [{"x": 0, "y": 0}, {"x": 100, "y": 0}], "source": {"element": source, "direction": "Right"},
            "target": {"element": target, "direction": "Left"}, "isManuallyLayouted": False, "flowType": flow_type,
            "isDefault": False}

    node("start", "BPMNStartEvent", "Started", -300, 10, 40, 40, eventType="default")
    previous = "start"
    for i in range(2):
        node(f"t{i}", "BPMNTask", f"Step {i + 1}", -200 + i * 220, 0, 140, 60, taskType="default", marker="none")
        node(f"d{i}", "BPMNDataObject", f"Input {i + 1}", -150 + i * 220, 120, 40, 50)
        rel(f"f{i}", previous, f"t{i}", "sequence")
        rel(f"a{i}", f"d{i}", f"t{i}", "data association")
        previous = f"t{i}"
    node("end", "BPMNEndEvent", "Ready", 300, 10, 40, 40, eventType="default")
    rel("f_end", previous, "end", "sequence")
    return {"title": "Cell", "model": {"version": "3.0.0", "type": "BPMNDiagram", "size": {"width": 800, "height": 400},
                                       "elements": elements, "relationships": relationships,
                                       "interactive": {"elements": {}, "relationships": {}}, "assessments": {}}}


def _generate(model, out, file_name="bpmn_diagram.bpmn"):
    generator = BPMNGenerator(model, output_dir=str(out), file_name=file_name)
    generator.generate()
    return generator, (out / file_name).read_text(encoding="utf-8")


def _regions(generator, file_name="bpmn_diagram.bpmn"):
    result = {}
    for region in generator.traced_files[file_name].regions:
        result.setdefault((region.element_id, region.rule.split("#")[1]), []).append(region)
    return result


def _lines(text, region):
    return text.split("\n")[region.start_line - 1:region.end_line]


def test_a_model_converted_from_the_editor_is_traced_by_its_editor_ids(tmp_path):
    generator, text = _generate(process_bpmn_diagram(_editor_diagram()), tmp_path)
    regions = _regions(generator)
    assert set(generator.traced_files) == {"bpmn_diagram.bpmn"}

    task = regions[("t0", "task")][0]
    assert _lines(text, task)[0].strip() == '<bpmn:task id="t0" name="Step 1">'
    assert _lines(text, task)[-1].strip() == "</bpmn:task>"
    assert (task.element_type, task.element_name) == ("Task", "Step 1")
    assert _lines(text, regions[("start", "startEvent")][0])[0].strip().startswith('<bpmn:startEvent id="start"')
    assert _lines(text, regions[("end", "endEvent")][0])[0].strip().startswith('<bpmn:endEvent id="end"')
    flow = regions[("f1", "sequenceFlow")][0]
    assert len(_lines(text, flow)) == 1
    assert _lines(text, flow)[0].strip().startswith('<bpmn:sequenceFlow id="f1" sourceRef="t0" targetRef="t1"')
    assert _lines(text, regions[("d1", "dataObjectReference")][0])[0].strip().startswith(
        '<bpmn:dataObjectReference id="d1"')
    association = regions[("a0", "dataInputAssociation")][0]
    assert task.start_line < association.start_line <= association.end_line < task.end_line  # nested in its host
    shape, edge = regions[("t0", "shape")][0], regions[("f0", "edge")][0]
    assert _lines(text, shape)[0].strip().startswith('<bpmndi:BPMNShape id="Shape_') and 'bpmnElement="t0"' in _lines(text, shape)[0]
    assert _lines(text, edge)[0].strip().startswith('<bpmndi:BPMNEdge id="Edge_') and 'bpmnElement="f0"' in _lines(text, edge)[0]
    # The process has no editor id: no region of its own.
    assert not any(rule == "process" for _, rule in regions)


def test_every_region_is_a_well_formed_range_that_holds_only_its_own_element(tmp_path):
    generator, text = _generate(process_bpmn_diagram(_editor_diagram()), tmp_path)
    lines = text.split("\n")
    for region in generator.traced_files["bpmn_diagram.bpmn"].regions:
        assert 1 <= region.start_line <= region.end_line <= len(lines)
        if region.rule.endswith(("#task", "#startEvent", "#endEvent", "#sequenceFlow", "#dataObjectReference")):
            assert f'id="{region.element_id}"' in lines[region.start_line - 1]


def test_python_built_elements_are_traced_by_their_own_ids_and_nest_in_a_subprocess(tmp_path):
    model = _subprocess_model()
    ids = {}
    for index, node in enumerate(model.all_flow_nodes()):
        node.id = ids[node] = f"node-{index}"
    for index, flow in enumerate(model.all_connecting_objects()):
        flow.id = f"flow-{index}"
    model.id = "bpmn-model"
    generator, text = _generate(model, tmp_path)
    regions = _regions(generator)

    sub = next(node for node in ids if isinstance(node, SubProcess))
    inner = next(node for node in ids if isinstance(node, Task))
    outer = regions[(ids[sub], "subProcess")][0]
    inside = regions[(ids[inner], "task")][0]
    assert outer.start_line < inside.start_line <= inside.end_line < outer.end_line
    assert [s.element.element_id for s in generator.traced_files["bpmn_diagram.bpmn"].sources] == ["bpmn-model"]
    assert len([key for key in regions if key[1] == "sequenceFlow"]) == 3


def test_collaboration_participants_message_flows_lanes_and_annotations_are_traced(tmp_path):
    model = _two_pool_model()
    collaboration = model.collaboration
    collaboration.id = "collab"
    for index, participant in enumerate(collaboration.participants):
        participant.id = f"participant-{index}"
    for index, message_flow in enumerate(collaboration.message_flows):
        message_flow.id = f"message-{index}"
    generator, text = _generate(model, tmp_path)
    regions = _regions(generator)
    assert _lines(text, regions[("collab", "collaboration")][0])[0].strip().startswith("<bpmn:collaboration")
    assert {key for key in regions if key[1] in ("participant", "messageFlow")} == {
        ("participant-0", "participant"), ("participant-1", "participant"), ("message-0", "messageFlow")}

    note = TextAnnotation(name="n", text="remember")
    lane_task = Task(name="Review")
    lane = Lane(name="Reviewer", flow_nodes={lane_task})
    process = Process(name="P", flow_nodes={lane_task}, lanes={lane}, artifacts={note})
    lane.id, note.id = "lane-1", "note-1"
    generator, text = _generate(BPMNModel(name="Lanes", processes={process}), tmp_path / "lanes")
    regions = _regions(generator)
    assert _lines(text, regions[("lane-1", "lane")][0])[0].strip().startswith('<bpmn:lane id="Lane_1" name="Reviewer"')
    assert "remember" in "\n".join(_lines(text, regions[("note-1", "textAnnotation")][0]))


def test_an_id_that_is_not_a_valid_xml_id_is_still_matched_to_its_element(tmp_path):
    task = Task(name="odd")
    task.id = "1-not-an-NCName"
    task.layout = {"id": "1-not-an-NCName"}
    generator, text = _generate(BPMNModel(name="M", processes={Process(name="P", flow_nodes={task})}), tmp_path)
    (region,) = _regions(generator)[("1-not-an-NCName", "task")]
    assert 'id="Task_1"' in _lines(text, region)[0]


def test_elements_without_ids_are_untraced_and_the_file_is_the_same(tmp_path):
    model = _subprocess_model()
    untraced, untraced_text = _generate(model, tmp_path / "untraced")
    assert untraced.traced_files["bpmn_diagram.bpmn"].regions == ()
    assert untraced.traced_files["bpmn_diagram.bpmn"].sources == ()

    for index, element in enumerate((*model.all_flow_nodes(), *model.all_connecting_objects())):
        element.id = f"element-{index}"
    traced, traced_text = _generate(model, tmp_path / "traced")
    assert len(traced.traced_files["bpmn_diagram.bpmn"].regions) == 5 + 3  # five flow nodes and three flows
    assert traced_text == untraced_text
    assert "\x02" not in traced_text


def test_the_processor_gives_each_element_its_editor_id():
    model = process_bpmn_diagram(_editor_diagram())
    (process,) = model.processes
    elements = (*process.flow_nodes, *process.sequence_flows, *process.data_objects, *process.data_associations)
    assert {element.id for element in elements} == {element.layout["id"] for element in elements}
    assert {element.id for element in elements} == {"start", "t0", "t1", "d0", "d1", "end",
                                                    "f0", "f1", "f_end", "a0", "a1"}


def test_a_layout_id_alone_does_not_trace_an_element(tmp_path):
    """The XML id comes from ``layout["id"]``; only ``Element.id`` makes an element traceable."""
    model = process_bpmn_diagram(_editor_diagram())
    for process in model.processes:
        for element in (*process.flow_nodes, *process.sequence_flows, *process.data_objects,
                        *process.data_associations):
            element.id = None
    generator, text = _generate(model, tmp_path)
    assert generator.traced_files["bpmn_diagram.bpmn"].regions == ()
    assert '<bpmn:task id="t0" name="Step 1">' in text


def test_the_trace_key_is_the_name_of_the_file_written(tmp_path):
    generator, _ = _generate(process_bpmn_diagram(_editor_diagram()), tmp_path, file_name="order.bpmn")
    assert set(generator.traced_files) == {"order.bpmn"}
    assert (tmp_path / "order.bpmn").is_file()
