"""An agent diagram with intents or LLMs but no states cannot become an agent.

Asking for exactly that agent (single-diagram generation, standalone chatbot
deploy) is an error. The paths that build every agent of a project (web app /
Docker Compose generation, web-app deploy) skip it with a warning in the log,
so one agent still being drawn does not block the rest of the project.
"""

import asyncio
import io
import logging
import zipfile

import json
import os

import httpx
import pytest
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
FIXTURES_V4 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "fixtures", "v4")
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


def _skip_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "Skipping agent diagram 'Helper': it has no states" in r.getMessage()]


def test_collect_agents_skips_an_agent_without_states_with_a_warning(caplog):
    with open(os.path.join(FIXTURES_V4, "agent_diagram_basic.json"), encoding="utf-8") as f:
        drawn = json.load(f)

    with caplog.at_level(logging.WARNING):
        agents, configs, _ = collect_agents_from_diagrams([_entry(_COMPONENTS_ONLY), drawn])

    assert [agent.name for agent in agents] == ["GreetBot"]
    assert list(configs) == ["GreetBot"]
    assert len(_skip_warnings(caplog)) == 1


def test_collect_agents_still_skips_an_empty_placeholder():
    assert collect_agents_from_diagrams([_entry(_EMPTY)])[0] == []


def test_docker_compose_skips_an_agent_without_states_with_a_warning(caplog):
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
    with caplog.at_level(logging.WARNING):
        response = _post("/besser_api/generate-output-from-project", json=project)

    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "docker-compose.yml" in archive.namelist()
    assert len(_skip_warnings(caplog)) == 1


def test_saving_to_github_keeps_the_model_files_of_an_agent_without_states():
    """Saving is storage, not generation: an agent still being drawn does not block the save."""
    from besser.utilities.web_modeling_editor.backend.services.deployment import repo_model

    project = {"id": "p", "type": "Project", "name": "P", "createdAt": "2026-01-01T00:00:00Z",
               "currentDiagramIndices": {}, "diagrams": {"AgentDiagram": [_entry(_COMPONENTS_ONLY)]}}
    saved = repo_model.build_model_files(project)

    assert saved.sources_built
    assert repo_model.MODEL_FILE in saved.files


def test_standalone_agent_deploy_reports_an_agent_without_states(monkeypatch):
    from besser.utilities.web_modeling_editor.backend.routers import github_deploy_api as gda

    class _FakeGitHub:
        async def get_authenticated_user(self):
            return {"login": "someone"}

    monkeypatch.setattr(gda, "get_user_token", lambda _sessions, _session: "fake-token")
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
