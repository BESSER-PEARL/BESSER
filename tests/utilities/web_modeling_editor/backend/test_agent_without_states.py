"""An agent diagram with intents or LLMs but no states is reported, not skipped.

Such a diagram used to be skipped silently by the paths that build every
agent of a project (web app / Docker Compose generation, GitHub deploy), so
the agent simply went missing from the output.
"""

import asyncio

import httpx
import pytest
from fastapi import HTTPException
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    collect_agents_from_diagrams,
)

_COMPONENTS_ONLY = {
    "version": "4.0.0", "type": "AgentDiagram", "nodes": [], "edges": [],
    "components": {
        "i1": {"id": "i1", "type": "AgentIntent", "name": "Greet", "bodies": ["b1"]},
        "b1": {"id": "b1", "type": "AgentIntentBody", "name": "hi", "owner": "i1"},
    },
}
_EMPTY = {"version": "4.0.0", "type": "AgentDiagram", "nodes": [], "edges": []}


def _entry(model, title="Helper"):
    return {"id": "agent-1", "title": title, "model": model}


def _post(url, **kwargs):
    async def _go():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            return await ac.post(url, **kwargs)
    return asyncio.run(_go())


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_collect_agents_reports_an_agent_without_states():
    with pytest.raises(HTTPException) as exc_info:
        collect_agents_from_diagrams([_entry(_COMPONENTS_ONLY)])
    assert exc_info.value.status_code == 400
    assert "'Helper' has no states" in exc_info.value.detail


def test_collect_agents_still_skips_an_empty_placeholder():
    assert collect_agents_from_diagrams([_entry(_EMPTY)])[0] == []


def test_docker_compose_reports_an_agent_without_states():
    deployment = {"version": "4.0.0", "type": "DeploymentDiagram", "edges": [], "nodes": [
        {"id": "host", "type": "deploymentNode", "position": {"x": 0, "y": 0}, "width": 300,
         "height": 200, "data": {"name": "Host", "stereotype": "node", "isComponentHeaderShown": True}},
    ]}
    project = {
        "id": "p", "type": "Project", "name": "P", "createdAt": "2026-01-01T00:00:00Z",
        "currentDiagramType": "DeploymentDiagram", "currentDiagramIndices": {},
        "settings": {"generator": "docker_compose"},
        "diagrams": {"DeploymentDiagram": [{"id": "d", "title": "Infra", "model": deployment}],
                     "AgentDiagram": [_entry(_COMPONENTS_ONLY)]},
    }
    response = _post("/besser_api/generate-output-from-project", json=project)
    assert response.status_code == 400, response.text
    assert "'Helper' has no states" in response.json()["detail"]


def test_saving_to_github_keeps_the_model_files_of_an_agent_without_states():
    """Saving is storage, not generation: an agent still being drawn does not block the save."""
    from besser.utilities.web_modeling_editor.backend.services.deployment import repo_model

    project = {"id": "p", "type": "Project", "name": "P", "createdAt": "2026-01-01T00:00:00Z",
               "currentDiagramIndices": {}, "diagrams": {"AgentDiagram": [_entry(_COMPONENTS_ONLY)]}}
    saved = repo_model.build_model_files(project)

    assert saved.sources_built
    assert repo_model.MODEL_FILE in saved.files


def test_standalone_agent_deploy_reports_an_agent_without_states(monkeypatch):
    from besser.utilities.web_modeling_editor.backend.services.deployment import github_deploy_api as gda

    class _FakeGitHub:
        async def get_authenticated_user(self):
            return {"login": "someone"}

    monkeypatch.setattr(gda, "get_user_token", lambda _session: "fake-token")
    monkeypatch.setattr(gda, "create_github_service", lambda _token: _FakeGitHub())
    body = {"id": "p", "type": "Project", "name": "P", "createdAt": "2026-01-01T00:00:00Z",
            "diagrams": {"AgentDiagram": [_entry(_COMPONENTS_ONLY)]},
            "deploy_config": {"repo_name": "bot", "target": "agent"}}
    response = _post("/besser_api/github/deploy-webapp", json=body,
                     headers={"X-GitHub-Session": "test-session"})
    assert response.status_code == 400, response.text
    assert "'Helper' has no states" in response.json()["detail"]


def test_single_diagram_agent_generation_reports_an_agent_without_states():
    body = {"title": "Helper", "model": _COMPONENTS_ONLY, "generator": "agent"}
    response = _post("/besser_api/generate-output", json=body)
    assert response.status_code == 400, response.text
    assert "'Helper' has no states" in response.json()["detail"]
