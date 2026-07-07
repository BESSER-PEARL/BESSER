"""HTTP-level test: push prefers the run's model-synced export.

A vibe-MODIFY run whose instruction implied new domain entities stores
an ``updated_project_export`` on its ``SmartRunEntry``. The push endpoint
must write ``buml/diagrams.json`` from THAT export (so the pushed model
matches the code), falling back to the request's ``projectExport`` only
when the entry has none.

Mocks ``GitHubService`` and captures the ``buml/diagrams.json`` content at
push time, so nothing touches the real GitHub API.
"""

import asyncio
import json
import os
import tempfile
import time

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import (
    smart_generation_router as router_module,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_runner import (
    _clear_registry,
)


BASE_URL = "http://testserver"
PUSH_URL = "/besser_api/push-smart-to-github"


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


class _CapturingGitHubService:
    """Reads buml/diagrams.json out of the pushed workdir at call time."""

    def __init__(self):
        self.pushed_diagrams_json = None

    async def get_authenticated_user(self):
        return {"login": "test-owner"}

    async def create_repository(
        self, repo_name, description="", is_private=False, auto_init=False
    ):
        return {"default_branch": "main"}

    async def get_branches(self, owner, repo_name, per_page=100):
        return ["main"]

    async def push_directory_to_repo(
        self,
        owner,
        repo_name,
        directory_path,
        commit_message="",
        branch="main",
        preserve_existing_files=False,
    ):
        path = os.path.join(directory_path, "buml", "diagrams.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                self.pushed_diagrams_json = json.load(fh)
        return {"commit_sha": "deadbeef", "total_files": 1, "uploaded_files": []}


def _install(monkeypatch, fake):
    monkeypatch.setattr(router_module, "create_github_service", lambda token: fake)
    monkeypatch.setattr(router_module, "get_user_token", lambda sid: "fake-token")


def _export(name: str) -> dict:
    """A re-importable ProjectInput-shaped export tagged by ``name``."""
    return {
        "id": "test-project",
        "type": "BesserProject",
        "name": name,
        "createdAt": "2026-04-15T00:00:00Z",
        "diagrams": {
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        },
        "currentDiagramIndices": {"ClassDiagram": 0},
    }


def _seed_run(run_id: str, *, updated_export: dict | None) -> str:
    tmp = tempfile.mkdtemp(prefix="besser_smart_pushsync_")
    with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as fh:
        fh.write("print('hi')\n")
    entry = SmartRunEntry(
        file_path=os.path.join(tmp, "main.py"),
        file_name="main.py",
        is_zip=False,
        temp_dir=tmp,
        created_at=time.time(),
        updated_project_export=updated_export,
    )
    asyncio.run(SMART_RUN_REGISTRY.put(run_id, entry))
    return tmp


def _post(body: dict, headers: dict | None = None) -> httpx.Response:
    transport = ASGITransport(app=app)

    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.post(PUSH_URL, json=body, headers=headers or {})

    return asyncio.run(_run())


def test_push_prefers_updated_project_export(monkeypatch):
    """Entry carries a model-synced export → diagrams.json is written from
    it, NOT from the request's (stale) projectExport."""
    run_id = "a" * 32
    _seed_run(run_id, updated_export=_export("UPDATED_PROJECT"))
    fake = _CapturingGitHubService()
    _install(monkeypatch, fake)

    body = {
        "run_id": run_id,
        "projectExport": _export("REQUEST_PROJECT"),
        "deploy_config": {"repo_name": "my-vibe-app"},
    }
    r = _post(body, headers={"X-GitHub-Session": "sess"})
    assert r.status_code == 200
    assert fake.pushed_diagrams_json is not None
    # The model-synced export won — not the request's.
    assert fake.pushed_diagrams_json["name"] == "UPDATED_PROJECT"


def test_push_falls_back_to_request_export_when_no_delta(monkeypatch):
    """No stored delta (generate/resume, or a no-op modify) → diagrams.json
    is written from the request's projectExport."""
    run_id = "b" * 32
    _seed_run(run_id, updated_export=None)
    fake = _CapturingGitHubService()
    _install(monkeypatch, fake)

    body = {
        "run_id": run_id,
        "projectExport": _export("REQUEST_PROJECT"),
        "deploy_config": {"repo_name": "my-vibe-app"},
    }
    r = _post(body, headers={"X-GitHub-Session": "sess"})
    assert r.status_code == 200
    assert fake.pushed_diagrams_json is not None
    assert fake.pushed_diagrams_json["name"] == "REQUEST_PROJECT"
