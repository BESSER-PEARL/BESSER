"""JSON -> BUML -> JSON identity for editor-shaped v4 Component and Deployment diagrams.

The fixtures carry exactly the keys the web editor serialises for each node and
edge, so every field the editor writes (cross-diagram links, stereotypes,
parents, layout) must come back unchanged -- both straight through the
metamodel and through a generated BUML file.
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


def _node(node_id, node_type, x, y, width, height, data, parent=None):
    node = {
        "id": node_id, "type": node_type,
        "position": {"x": x, "y": y},
        "width": width, "height": height,
        "measured": {"width": width, "height": height},
        "data": data,
    }
    if parent is not None:
        node["parentId"] = parent
    return node


def _edge(edge_id, edge_type, source, target, stereotype=None):
    data = {
        "name": "", "label": "",
        "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "isManuallyLayouted": False,
    }
    if stereotype is not None:
        data["stereotype"] = stereotype
    return {
        "id": edge_id, "source": source, "target": target, "type": edge_type,
        "sourceHandle": "right", "targetHandle": "left", "data": data,
    }


def _component(node_id, name, stereotype, x=0, y=0, parent=None, **refs):
    data = {
        "name": name, "stereotype": stereotype, "isComponentHeaderShown": True,
        "realizes": refs.get("realizes", []),
        "processModelRefs": refs.get("processModelRefs", []),
    }
    if "agentModelRef" in refs:
        data["agentModelRef"] = refs["agentModelRef"]
    return _node(node_id, "component", x, y, 160, 100, data, parent)


@pytest.fixture
def component_diagram():
    nodes = [
        _node("sub", "componentSubsystem", 0, 0, 600, 400,
              {"name": "Swarm", "stereotype": "subsystem",
               "isComponentSubsystemHeaderShown": True}),
        _component("coder", "Coder", "solution", 20, 60, parent="sub",
                   realizes=["class-1"], processModelRefs=["bpmn-1"],
                   agentModelRef="agent-coder"),
        _component("reviewer", "Reviewer", "supervision", 300, 60, parent="sub",
                   processModelRefs=["bpmn-1"], agentModelRef="agent-reviewer"),
        _component("human", "Product Owner", "human", 700, 0,
                   processModelRefs=["bpmn-1"], agentModelRef="agent-human"),
        _component("frontend", "Frontend", "component", 700, 200, realizes=["class-2"]),
        _component("llm", "GPT", "llm", 700, 400),
        _node("api", "componentInterface", 900, 100, 20, 20, {"name": "TaskApi"}),
    ]
    edges = [
        _edge("r-del", "ComponentDependency", "reviewer", "coder",
              stereotype="delegates {permission: repo:write}"),
        _edge("r-uses", "ComponentDependency", "coder", "llm", stereotype="uses"),
        _edge("r-plain", "ComponentDependency", "frontend", "human", stereotype="«use»"),
        _edge("r-prov", "ComponentProvidedInterface", "coder", "api"),
        _edge("r-req", "ComponentRequiredInterface", "frontend", "api"),
    ]
    return {"title": "Components", "model": {
        "version": "4.0.0", "type": "ComponentDiagram", "title": "Components",
        "nodes": nodes, "edges": edges,
        "interactive": {"elements": {}, "relationships": {}}, "assessments": {},
    }}


@pytest.fixture
def deployment_diagram():
    nodes = [
        _node("host", "deploymentNode", 0, 0, 700, 400,
              {"name": "Swarm Host", "stereotype": "docker host", "isComponentHeaderShown": True}),
        _node("plain", "deploymentNode", 800, 0, 160, 100,
              {"name": "Edge", "stereotype": "", "isComponentHeaderShown": True}),
        _node("generic", "deploymentNode", 800, 200, 160, 100,
              {"name": "Cache", "stereotype": "node", "isComponentHeaderShown": True}),
        _node("ee", "deploymentNode", 20, 40, 200, 150,
              {"name": "Coder", "stereotype": "executionEnvironment",
               "isComponentHeaderShown": True}, parent="host"),
        _node("art", "deploymentArtifact", 10, 40, 160, 40,
              {"name": "Coder [3]", "manifests": ["coder"], "agentModelRef": "agent-coder"},
              parent="ee"),
        _node("lib", "deploymentArtifact", 300, 500, 160, 40,
              {"name": "Library", "manifests": []}),
        _node("dc", "deploymentComponent", 20, 450, 160, 100,
              {"name": "Coder", "stereotype": "component", "isComponentHeaderShown": True}),
        _node("iface", "deploymentInterface", 500, 500, 20, 20, {"name": "Port"}),
    ]
    edges = [
        _edge("r-manifest", "DeploymentDependency", "art", "dc"),
        _edge("r-comm", "DeploymentAssociation", "host", "plain"),
        _edge("r-deploy", "DeploymentAssociation", "lib", "generic"),
        _edge("r-prov", "DeploymentProvidedInterface", "lib", "iface"),
    ]
    return {"title": "Deployment", "model": {
        "version": "4.0.0", "type": "DeploymentDiagram", "title": "Deployment",
        "nodes": nodes, "edges": edges,
        "interactive": {"elements": {}, "relationships": {}}, "assessments": {},
    }}


def _by_id(items):
    return {item["id"]: item for item in items}


def _assert_identity(original: dict, round_tripped: dict):
    assert round_tripped["version"] == "4.0.0"
    assert _by_id(round_tripped["nodes"]) == _by_id(original["model"]["nodes"])
    assert _by_id(round_tripped["edges"]) == _by_id(original["model"]["edges"])


def _node_named(diagram, node_id):
    return next(n for n in diagram["model"]["nodes"] if n["id"] == node_id)


class TestComponentIdentity:
    def test_json_buml_json_is_identity(self, component_diagram):
        original = copy.deepcopy(component_diagram)
        _assert_identity(original, component_object_to_json(process_component_diagram(component_diagram)))

    def test_json_buml_file_json_is_identity(self, component_diagram):
        original = copy.deepcopy(component_diagram)
        source = component_model_to_code(process_component_diagram(component_diagram))
        _assert_identity(original, component_buml_to_json(source))

    def test_invalid_edge_raises_instead_of_being_dropped(self, component_diagram):
        component_diagram["model"]["edges"].append(_edge(
            "r-bad", "ComponentDependency", "frontend", "llm", stereotype="has"))
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
        deployment_diagram["model"]["edges"].append(_edge(
            "r-bad", "DeploymentAssociation", "plain", "lib"))
        with pytest.raises(ConversionError, match="r-bad"):
            process_deployment_diagram(deployment_diagram)

    def test_node_nested_in_its_own_child_raises(self, deployment_diagram):
        _node_named(deployment_diagram, "host")["parentId"] = "ee"
        with pytest.raises(ConversionError):
            process_deployment_diagram(deployment_diagram)
