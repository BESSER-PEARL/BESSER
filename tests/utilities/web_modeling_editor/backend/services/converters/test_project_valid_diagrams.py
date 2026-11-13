"""Which project diagrams ``json_to_buml_project`` keeps.

An agent diagram whose only content is off-canvas ``components`` (intents,
LLMs, ...) and no canvas ``nodes`` is still a model and must not be dropped.
"""

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.utilities.web_modeling_editor.backend.models import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
    _is_valid_diagram,
    json_to_buml_project,
)


def _agent_model(nodes=None, components=None):
    model = {"version": "4.0.0", "type": "AgentDiagram", "title": "Bot", "nodes": nodes or [], "edges": []}
    if components is not None:
        model["components"] = components
    return model


INTENT_COMPONENTS = {
    "i1": {"id": "i1", "type": "AgentIntent", "name": "Greeting", "intent_description": "", "bodies": ["b1"]},
    "b1": {"id": "b1", "type": "AgentIntentBody", "name": "hello", "owner": "i1"},
}


def _project(agent_model):
    return ProjectInput(**{
        "id": "p1",
        "type": "Project",
        "name": "ComponentsOnly",
        "createdAt": "2025-01-01T00:00:00Z",
        "currentDiagramType": "AgentDiagram",
        "currentDiagramIndices": {"AgentDiagram": 0},
        "diagrams": {"AgentDiagram": [{"id": "a1", "title": "Bot", "model": agent_model}]},
    })


def test_agent_diagram_with_only_components_is_valid():
    project = _project(_agent_model(components=INTENT_COMPONENTS))
    assert _is_valid_diagram(project.diagrams["AgentDiagram"][0], "AgentDiagram")


def test_empty_agent_diagram_is_still_dropped():
    project = _project(_agent_model(components={}))
    assert not _is_valid_diagram(project.diagrams["AgentDiagram"][0], "AgentDiagram")


def test_components_do_not_validate_other_diagram_types():
    project = _project(_agent_model(components=INTENT_COMPONENTS))
    assert not _is_valid_diagram(project.diagrams["AgentDiagram"][0], "StateMachineDiagram")


def test_components_only_agent_is_kept_in_project():
    buml_project = json_to_buml_project(_project(_agent_model(components=INTENT_COMPONENTS)))
    agents = [m for m in buml_project.models if isinstance(m, Agent)]
    assert len(agents) == 1
    assert [intent.name for intent in agents[0].intents] == ["Greeting"]
