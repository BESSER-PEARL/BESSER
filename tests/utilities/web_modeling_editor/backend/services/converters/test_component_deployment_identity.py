"""JSON -> BUML -> JSON identity for editor-shaped Component and Deployment diagrams.

The fixtures carry exactly the keys the web editor serialises for each element
and relationship, so every field the editor writes (cross-diagram links,
stereotypes, owners, layout) must come back unchanged -- both straight through
the metamodel and through a generated BUML file.
"""

import copy

import pytest

from besser.utilities.buml_code_builder.component_model_builder import component_model_to_code
from besser.utilities.buml_code_builder.deployment_model_builder import deployment_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.component_diagram_converter import (
    component_buml_to_json,
    component_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.deployment_diagram_converter import (
    deployment_buml_to_json,
    deployment_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.component_diagram_processor import (
    process_component_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.deployment_diagram_processor import (
    process_deployment_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


def _bounds(x, y, width=160, height=100):
    return {"x": x, "y": y, "width": width, "height": height}


def _edge(rel_id, rel_type, source, target, **extra):
    edge = {
        "id": rel_id, "name": "", "type": rel_type, "owner": None,
        "bounds": _bounds(0, 0, 100, 1),
        "path": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "source": {"element": source, "direction": "Right"},
        "target": {"element": target, "direction": "Left"},
        "isManuallyLayouted": False,
    }
    edge.update(extra)
    return edge


def _component(elem_id, name, stereotype, owner=None, **refs):
    element = {
        "id": elem_id, "name": name, "type": "Component", "owner": owner,
        "bounds": _bounds(0, 0), "stereotype": stereotype, "displayStereotype": True,
        "realizes": refs.get("realizes", []),
        "processModelRefs": refs.get("processModelRefs", []),
    }
    if "agentModelRef" in refs:
        element["agentModelRef"] = refs["agentModelRef"]
    return element


@pytest.fixture
def component_diagram():
    elements = {
        "sub": {"id": "sub", "name": "Swarm", "type": "Subsystem", "owner": None,
                "bounds": _bounds(0, 0, 600, 400), "stereotype": "subsystem",
                "displayStereotype": True},
        "coder": _component("coder", "Coder", "solution", owner="sub",
                            realizes=["class-1"], processModelRefs=["bpmn-1"],
                            agentModelRef="agent-coder"),
        "reviewer": _component("reviewer", "Reviewer", "supervision", owner="sub",
                               processModelRefs=["bpmn-1"], agentModelRef="agent-reviewer"),
        "human": _component("human", "Product Owner", "human",
                            processModelRefs=["bpmn-1"], agentModelRef="agent-human"),
        "frontend": _component("frontend", "Frontend", "component", realizes=["class-2"]),
        "llm": _component("llm", "GPT", "llm"),
        "api": {"id": "api", "name": "TaskApi", "type": "ComponentInterface", "owner": None,
                "bounds": _bounds(0, 0, 20, 20)},
    }
    relationships = {
        "r-del": _edge("r-del", "ComponentDependency", "reviewer", "coder",
                       stereotype="delegates {permission: repo:write}"),
        "r-uses": _edge("r-uses", "ComponentDependency", "coder", "llm", stereotype="uses"),
        "r-plain": _edge("r-plain", "ComponentDependency", "frontend", "human", stereotype="«use»"),
        "r-prov": _edge("r-prov", "ComponentInterfaceProvided", "coder", "api"),
        "r-req": _edge("r-req", "ComponentInterfaceRequired", "frontend", "api"),
    }
    return {"title": "Components", "model": {
        "version": "3.0.0", "type": "ComponentDiagram", "size": {"width": 800, "height": 600},
        "elements": elements, "relationships": relationships,
        "interactive": {"elements": {}, "relationships": {}}, "assessments": {},
    }}


@pytest.fixture
def deployment_diagram():
    elements = {
        "host": {"id": "host", "name": "Swarm Host", "type": "DeploymentNode", "owner": None,
                 "bounds": _bounds(0, 0, 700, 400), "stereotype": "docker host",
                 "displayStereotype": True},
        "plain": {"id": "plain", "name": "Edge", "type": "DeploymentNode", "owner": None,
                  "bounds": _bounds(800, 0)},
        "generic": {"id": "generic", "name": "Cache", "type": "DeploymentNode", "owner": None,
                    "bounds": _bounds(800, 200), "stereotype": "node", "displayStereotype": True},
        "ee": {"id": "ee", "name": "Coder", "type": "DeploymentNode", "owner": "host",
               "bounds": _bounds(20, 40, 200, 150), "stereotype": "executionEnvironment",
               "displayStereotype": True},
        "art": {"id": "art", "name": "Coder [3]", "type": "DeploymentArtifact", "owner": "ee",
                "bounds": _bounds(30, 80, 160, 40), "manifests": ["coder"],
                "agentModelRef": "agent-coder"},
        "lib": {"id": "lib", "name": "Library", "type": "DeploymentArtifact", "owner": None,
                "bounds": _bounds(300, 500, 160, 40), "manifests": []},
        "dc": {"id": "dc", "name": "Coder", "type": "DeploymentComponent", "owner": None,
               "bounds": _bounds(20, 450), "stereotype": "component", "displayStereotype": True},
        "iface": {"id": "iface", "name": "Port", "type": "DeploymentInterface", "owner": None,
                  "bounds": _bounds(500, 500, 20, 20)},
    }
    relationships = {
        "r-manifest": _edge("r-manifest", "DeploymentDependency", "art", "dc"),
        "r-comm": _edge("r-comm", "DeploymentAssociation", "host", "plain"),
        "r-deploy": _edge("r-deploy", "DeploymentAssociation", "lib", "generic"),
        "r-prov": _edge("r-prov", "DeploymentInterfaceProvided", "lib", "iface"),
    }
    return {"title": "Deployment", "model": {
        "version": "3.0.0", "type": "DeploymentDiagram", "size": {"width": 1000, "height": 700},
        "elements": elements, "relationships": relationships,
        "interactive": {"elements": {}, "relationships": {}}, "assessments": {},
    }}


def _assert_identity(original: dict, round_tripped: dict):
    assert round_tripped["elements"] == original["model"]["elements"]
    assert round_tripped["relationships"] == original["model"]["relationships"]


class TestComponentIdentity:
    def test_json_buml_json_is_identity(self, component_diagram):
        original = copy.deepcopy(component_diagram)
        _assert_identity(original, component_object_to_json(process_component_diagram(component_diagram)))

    def test_json_buml_file_json_is_identity(self, component_diagram):
        original = copy.deepcopy(component_diagram)
        source = component_model_to_code(process_component_diagram(component_diagram))
        _assert_identity(original, component_buml_to_json(source))

    def test_invalid_edge_raises_instead_of_being_dropped(self, component_diagram):
        component_diagram["model"]["relationships"]["r-bad"] = _edge(
            "r-bad", "ComponentDependency", "frontend", "llm", stereotype="has")
        with pytest.raises(ConversionError, match="r-bad"):
            process_component_diagram(component_diagram)


class TestDeploymentIdentity:
    def test_json_buml_json_is_identity(self, deployment_diagram):
        original = copy.deepcopy(deployment_diagram)
        _assert_identity(original, deployment_object_to_json(process_deployment_diagram(deployment_diagram)))

    def test_json_buml_file_json_is_identity(self, deployment_diagram):
        original = copy.deepcopy(deployment_diagram)
        source = deployment_model_to_code(process_deployment_diagram(deployment_diagram))
        _assert_identity(original, deployment_buml_to_json(source))

    def test_invalid_association_raises_instead_of_being_dropped(self, deployment_diagram):
        deployment_diagram["model"]["relationships"]["r-bad"] = _edge(
            "r-bad", "DeploymentAssociation", "plain", "lib")
        with pytest.raises(ConversionError, match="r-bad"):
            process_deployment_diagram(deployment_diagram)

    def test_node_nested_in_its_own_child_raises(self, deployment_diagram):
        deployment_diagram["model"]["elements"]["host"]["owner"] = "ee"
        with pytest.raises(ConversionError):
            process_deployment_diagram(deployment_diagram)
