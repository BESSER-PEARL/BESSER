"""Every editor processor gives the elements it builds their editor id (``Element.id``).

A trace refers to a model element by the id the editor recorded it by, so the processors give each
element the id of the editor element it is built from (a quantum gate, whose cell has no id, the
editor's ``<diagram id>:gate:<column>:<wire>``). These tests check, on the editor's own templates,
that the ids are the editor's and unique within a model, that the elements the editor's trace records
get them, that the ids change nothing a generator writes, and that a model converted back to JSON keeps
them.
"""

import json
import pathlib

import pytest

from besser.BUML.metamodel.structural import Element
from besser.BUML.metamodel.traceability.traceability import model_elements
from besser.generators.agents.baf_generator import BAFGenerator
from besser.generators.bpmn import BPMNGenerator
from besser.generators.docker_compose import DockerComposeGenerator
from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator
from besser.generators.qiskit import QiskitGenerator
from besser.generators.react import ReactGenerator
from besser.utilities.web_modeling_editor.backend.services.converters import (
    bpmn_object_to_json,
    component_object_to_json,
    deployment_object_to_json,
    nn_model_to_json,
    process_agent_diagram,
    process_bpmn_diagram,
    process_class_diagram,
    process_component_diagram,
    process_deployment_diagram,
    process_gui_diagram,
    process_nn_diagram,
    process_quantum_diagram,
    process_state_machine,
    state_machine_object_to_json,
)

TESTS = pathlib.Path(__file__).parents[5]
FIXTURES = TESTS / "utilities" / "web_modeling_editor" / "backend" / "fixtures"
GENERATOR_FIXTURES = TESTS / "generators" / "fixtures"


def _json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _diagram(model: dict, diagram_id: str, title: str = "Model") -> dict:
    return {"id": diagram_id, "title": title, "model": model}


CLASS = _json(GENERATOR_FIXTURES / "library_template_class_diagram.json")
AGENT = _json(GENERATOR_FIXTURES / "library_template_agent_diagram.json")
GUI = _json(FIXTURES / "library_template_gui_diagram.json")
BPMN = _diagram(_json(FIXTURES / "template_bpmn_pizza_store.json"), "bpmn-diagram", "Pizza store")
NN = _diagram(_json(FIXTURES / "template_nn_tutorial_example.json"), "nn-diagram", "Tutorial")
STATE_MACHINE = _diagram(_json(FIXTURES / "template_state_machine_traffic_light.json"), "sm-diagram", "Lights")
QUANTUM = _diagram({"cols": [["H", 1], ["•", "X"], ["Swap", "Swap"]], "gateMetadata": {}}, "q-diagram", "Bell")


def _element(element_id, element_type, name, owner=None, **extra):
    return {"id": element_id, "type": element_type, "name": name, "owner": owner,
            "bounds": {"x": 0, "y": 0, "width": 100, "height": 100}, **extra}


def _relationship(rel_id, rel_type, source, target):
    return {"id": rel_id, "type": rel_type, "name": "", "owner": None,
            "bounds": {"x": 0, "y": 0, "width": 1, "height": 1}, "path": [],
            "source": {"element": source, "direction": "Right"}, "target": {"element": target, "direction": "Left"}}


DEPLOYMENT = _diagram({"type": "DeploymentDiagram", "elements": {
    "node-1": _element("node-1", "DeploymentNode", "server"),
    "node-2": _element("node-2", "DeploymentNode", "db"),
    "artifact-1": _element("artifact-1", "DeploymentArtifact", "api", "node-1"),
}, "relationships": {"path-1": _relationship("path-1", "DeploymentAssociation", "node-1", "node-2")}},
    "deployment-diagram", "Cloud")
COMPONENT = _diagram({"type": "ComponentDiagram", "elements": {
    "component-1": _element("component-1", "Component", "Orders"),
    "component-2": _element("component-2", "Component", "Billing"),
}, "relationships": {"dep-1": _relationship("dep-1", "ComponentDependency", "component-1", "component-2")}},
    "component-diagram", "Parts")


def _domain():
    return process_class_diagram(CLASS)


# Each processor on an editor diagram: (model, the editor element ids its traced elements may have).
def _models():
    domain = _domain()
    return {
        "agent": (process_agent_diagram(AGENT), AGENT["model"]),
        "gui": (process_gui_diagram(GUI["model"], CLASS["model"], domain), GUI["model"]),
        "bpmn": (process_bpmn_diagram(BPMN), BPMN["model"]),
        "nn": (process_nn_diagram(NN), NN["model"]),
        "state_machine": (process_state_machine(STATE_MACHINE), STATE_MACHINE["model"]),
        "deployment": (process_deployment_diagram(DEPLOYMENT), DEPLOYMENT["model"]),
        "component": (process_component_diagram(COMPONENT), COMPONENT["model"]),
        "quantum": (process_quantum_diagram(QUANTUM), None),
    }


def _editor_ids(model_json: dict) -> set:
    ids, pending = set(), [model_json]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key in ("elements", "relationships", "components"):
                if isinstance(value.get(key), dict):
                    ids.update(value[key])
            if isinstance(value.get("id"), str):
                ids.add(value["id"])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return ids


def _own_elements(model):
    """The model's own elements: not the classes a GUI shows, nor the shared primitive types."""
    return [e for e in model_elements(model) if not type(e).__module__.startswith("besser.BUML.metamodel.structural")]


@pytest.mark.parametrize("kind", ["agent", "gui", "bpmn", "nn", "state_machine", "deployment", "component"])
def test_the_ids_are_editor_ids_unique_in_the_model(kind):
    model, model_json = _models()[kind]
    ids = [e.id for e in _own_elements(model) if e.id]
    assert ids, f"{kind}: no element has an id"
    assert len(ids) == len(set(ids))
    assert set(ids) <= _editor_ids(model_json)


def _with_id(model, metaclass_name):
    return {e.name: e.id for e in _own_elements(model) if type(e).__name__ == metaclass_name}


def test_the_elements_the_editor_traces_have_their_ids():
    models = _models()
    agent, agent_json = models["agent"]
    states = {e["name"]: e_id for e_id, e in agent_json["elements"].items() if e["type"] == "AgentState"}
    assert _with_id(agent, "AgentState") == states
    intents = {e["name"]: e_id for e_id, e in {**agent_json["elements"], **agent_json.get("components", {})}.items()
               if e["type"] == "AgentIntent"}
    assert _with_id(agent, "Intent") == intents
    transitions = [e for e in _own_elements(agent) if type(e).__name__ == "Transition"]
    assert transitions and all(t.id in agent_json["relationships"] for t in transitions)

    gui, gui_json = models["gui"]
    pages = {page["id"] for page in gui_json["pages"]}
    assert {e.id for e in _own_elements(gui) if type(e).__name__ == "Screen"} == pages
    # Tables, buttons and agent components (traced by the editor) have their ids; text and containers do not.
    assert all(e.id for e in _own_elements(gui) if type(e).__name__ in ("Table", "Button", "AgentComponent"))
    assert not any(e.id for e in _own_elements(gui) if type(e).__name__ in ("Text", "ViewContainer"))

    nn, nn_json = models["nn"]
    layers = {e_id for e_id, e in nn_json["elements"].items() if e["type"].endswith("Layer")}
    assert {e.id for e in _own_elements(nn) if type(e).__name__ in
            ("Conv2D", "PoolingLayer", "FlattenLayer", "LinearLayer")} == layers
    assert {e.id for e in _own_elements(nn) if type(e).__name__ == "Dataset"} == {
        e_id for e_id, e in nn_json["elements"].items() if e["type"] in ("TrainingDataset", "TestDataset")}
    (container,) = [e_id for e_id, e in nn_json["elements"].items() if e["type"] == "NNContainer"]
    assert nn.id == container

    quantum, _ = models["quantum"]
    assert [(type(g).__name__, g.id) for g in quantum.operations] == [
        ("HadamardGate", "q-diagram:gate:0:0"), ("PauliXGate", "q-diagram:gate:1:1"),
        ("SwapGate", "q-diagram:gate:2:0")]

    deployment, _ = models["deployment"]
    assert {(type(e).__name__, e.id) for e in _own_elements(deployment)} == {
        ("Node", "node-1"), ("Node", "node-2"), ("Artifact", "artifact-1"), ("CommunicationPath", "path-1"),
        ("DeploymentRelation", None)}  # the relation an artifact's owner implies has no editor element


def test_a_quantum_circuit_without_a_diagram_id_has_no_gate_ids():
    circuit = process_quantum_diagram({"model": QUANTUM["model"]})
    assert all(g.id is None for g in circuit.operations)


def _strip_ids(model) -> None:
    for element in model_elements(model):
        if isinstance(element, Element) and not type(element).__module__.startswith("besser.BUML.metamodel.structural"):
            element.id = None
    model.id = None


def _read_all(folder: pathlib.Path) -> dict:
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


def _generate(make_generator, model, out: pathlib.Path) -> dict:
    make_generator(model, str(out)).generate()
    return _read_all(out)


_GENERATIONS = {
    "agent": (lambda: process_agent_diagram(AGENT), lambda model, out: BAFGenerator(model, output_dir=out)),
    "bpmn": (lambda: process_bpmn_diagram(BPMN),
             lambda model, out: BPMNGenerator(model, output_dir=out)),
    "nn": (lambda: process_nn_diagram(NN), lambda model, out: PytorchGenerator(model, output_dir=out)),
    "quantum": (lambda: process_quantum_diagram(QUANTUM), lambda model, out: QiskitGenerator(model, output_dir=out)),
    "deployment": (lambda: process_deployment_diagram(DEPLOYMENT),
                   lambda model, out: DockerComposeGenerator(model, output_dir=out)),
}


@pytest.mark.parametrize("kind", sorted(_GENERATIONS))
def test_the_ids_change_nothing_a_generator_writes(kind, tmp_path):
    convert, make_generator = _GENERATIONS[kind]
    with_ids = _generate(make_generator, convert(), tmp_path / "with_ids")
    model = convert()
    _strip_ids(model)
    assert _generate(make_generator, model, tmp_path / "without") == with_ids


def test_the_gui_ids_change_nothing_the_react_generator_writes(tmp_path):
    def generate(strip: bool, out: pathlib.Path) -> dict:
        domain = _domain()
        gui = process_gui_diagram(GUI["model"], CLASS["model"], domain)
        if strip:
            _strip_ids(gui)
        ReactGenerator(domain, gui, output_dir=str(out)).generate()
        return _read_all(out)
    assert generate(False, tmp_path / "with_ids") == generate(True, tmp_path / "without")


_BACK_TO_JSON = {
    "bpmn": (lambda: process_bpmn_diagram(BPMN), bpmn_object_to_json, BPMN["model"]),
    "nn": (lambda: process_nn_diagram(NN), nn_model_to_json, NN["model"]),
    "state_machine": (lambda: process_state_machine(STATE_MACHINE), state_machine_object_to_json,
                      STATE_MACHINE["model"]),
    "deployment": (lambda: process_deployment_diagram(DEPLOYMENT), deployment_object_to_json, DEPLOYMENT["model"]),
    "component": (lambda: process_component_diagram(COMPONENT), component_object_to_json, COMPONENT["model"]),
}


@pytest.mark.parametrize("kind", sorted(_BACK_TO_JSON))
def test_a_model_converted_back_to_json_keeps_the_editor_ids(kind):
    convert, to_json, model_json = _BACK_TO_JSON[kind]
    model = convert()
    back = to_json(model)
    back = back.get("model", back)
    kept = {e.id for e in _own_elements(model) if e.id} | ({model.id} if getattr(model, "id", None) else set())
    assert kept <= set(back["elements"]) | set(back["relationships"])
    assert kept <= _editor_ids(model_json)
