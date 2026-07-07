"""HTTP-level tests for POST /besser_api/push-smart-to-github.

The endpoint pushes the *stored* artifact of a finished vibe/smart-gen
run (by ``run_id``) plus the re-importable model source to GitHub. These
tests mock ``GitHubService`` and seed ``SMART_RUN_REGISTRY`` with a fake
``SmartRunEntry`` pointing at a temp dir we build, so nothing touches the
real GitHub API.
"""

import asyncio
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


# ---------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------


class _FakeGitHubService:
    """Records calls and snapshots the pushed directory tree.

    ``push_directory_to_repo`` walks the directory it is handed *at call
    time* — the real temp workdir is deleted when the endpoint's
    ``TemporaryDirectory`` context exits, so the snapshot must happen
    during the call, which is exactly what the endpoint does in practice.
    """

    def __init__(self, *, branches=None, get_branches_404=False):
        self._branches = ["main"] if branches is None else branches
        self._get_branches_404 = get_branches_404
        self.created = False
        self.created_kwargs = None
        self.push = None

    async def get_authenticated_user(self):
        return {"login": "test-owner"}

    async def create_repository(
        self, repo_name, description="", is_private=False, auto_init=False
    ):
        self.created = True
        self.created_kwargs = dict(
            repo_name=repo_name,
            description=description,
            is_private=is_private,
            auto_init=auto_init,
        )
        return {"default_branch": "main"}

    async def get_branches(self, owner, repo_name, per_page=100):
        if self._get_branches_404:
            request = httpx.Request(
                "GET", "https://api.github.com/repos/test-owner/x/branches"
            )
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError(
                "Not Found", request=request, response=response
            )
        return self._branches

    async def push_directory_to_repo(
        self,
        owner,
        repo_name,
        directory_path,
        commit_message="",
        branch="main",
        preserve_existing_files=False,
    ):
        files = []
        for root, _dirs, names in os.walk(directory_path):
            for name in names:
                rel = os.path.relpath(
                    os.path.join(root, name), directory_path
                ).replace("\\", "/")
                files.append(rel)
        self.push = dict(
            owner=owner,
            repo_name=repo_name,
            directory_path=directory_path,
            files=sorted(files),
            commit_message=commit_message,
            branch=branch,
            preserve_existing_files=preserve_existing_files,
        )
        return {
            "commit_sha": "deadbeef",
            "total_files": len(files),
            "uploaded_files": files,
        }


def _install(monkeypatch, fake):
    """Patch the GitHub factory + token lookup in the router namespace."""
    monkeypatch.setattr(router_module, "create_github_service", lambda token: fake)
    monkeypatch.setattr(router_module, "get_user_token", lambda sid: "fake-token")


def _project_export() -> dict:
    """A re-importable V2 project-export envelope (ProjectInput shape)."""
    return {
        "id": "test-project",
        "type": "BesserProject",
        "name": "TestProject",
        "createdAt": "2026-04-15T00:00:00Z",
        "diagrams": {
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        },
        "currentDiagramIndices": {"ClassDiagram": 0},
    }


def _seed_run(run_id: str, *, with_secret_env: bool = True) -> str:
    """Create a fake generated file tree and register it under ``run_id``."""
    tmp = tempfile.mkdtemp(prefix="besser_smart_push_test_")

    with open(os.path.join(tmp, "main.py"), "w", encoding="utf-8") as fh:
        fh.write("# vibe generated\nprint('hi')\n")

    # Internal artifact that must NOT be pushed.
    with open(os.path.join(tmp, ".besser_recipe.json"), "w", encoding="utf-8") as fh:
        fh.write('{"instructions": "x"}')

    # Build/dependency dir that must be skipped.
    nm = os.path.join(tmp, "node_modules", "left-pad")
    os.makedirs(nm, exist_ok=True)
    with open(os.path.join(nm, "index.js"), "w", encoding="utf-8") as fh:
        fh.write("module.exports = 1;\n")

    if with_secret_env:
        # Real-looking secret → must be scrubbed.
        with open(os.path.join(tmp, ".env"), "w", encoding="utf-8") as fh:
            fh.write("OPENAI_API_KEY=sk-ant-REALSECRET0123456789abcdefghij\n")
        # Template → must be preserved.
        with open(os.path.join(tmp, ".env.example"), "w", encoding="utf-8") as fh:
            fh.write("OPENAI_API_KEY=your-key-here\n")

    entry = SmartRunEntry(
        file_path=os.path.join(tmp, "main.py"),
        file_name="main.py",
        is_zip=False,
        temp_dir=tmp,
        created_at=time.time(),
    )
    asyncio.run(SMART_RUN_REGISTRY.put(run_id, entry))
    return tmp


def _post(body: dict, headers: dict | None = None) -> httpx.Response:
    transport = ASGITransport(app=app)

    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.post(PUSH_URL, json=body, headers=headers or {})

    return asyncio.run(_run())


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestPushSmartToGitHub:
    def test_valid_run_pushes_and_returns_repo_url(self, monkeypatch):
        run_id = "a" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-vibe-app", "is_private": True},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["repo_url"] == "https://github.com/test-owner/my-vibe-app"
        assert data["owner"] == "test-owner"
        assert data["files_uploaded"] >= 1
        assert data["is_first_push"] is True
        # A fresh (default-private) repo was created.
        assert fake.created is True
        assert fake.created_kwargs["is_private"] is True

    def test_unknown_run_id_returns_404_run_expired(self, monkeypatch):
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": "0" * 32,  # never seeded
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-vibe-app"},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 404
        assert r.json()["detail"] == "run_expired"

    def test_missing_github_session_returns_401(self, monkeypatch):
        run_id = "b" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-vibe-app"},
        }
        r = _post(body)  # no X-GitHub-Session header
        assert r.status_code == 401

    def test_model_and_diagrams_land_in_pushed_dir(self, monkeypatch):
        run_id = "c" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-vibe-app"},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200

        files = fake.push["files"]
        # Model source injected under buml/.
        assert "buml/diagrams.json" in files
        assert "buml/domain_model.py" in files
        # The stored vibe code is present.
        assert "main.py" in files
        # Internal artifact + build dir excluded.
        assert ".besser_recipe.json" not in files
        assert not any(f.startswith("node_modules/") for f in files)
        # Secret .env scrubbed; template preserved.
        assert ".env" not in files
        assert ".env.example" in files

    def test_use_existing_pushes_requested_branch_replacing_tree(self, monkeypatch):
        run_id = "d" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService(branches=["main", "dev"])
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {
                "repo_name": "my-vibe-app",
                "use_existing": True,
                "branch": "dev",
            },
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200
        data = r.json()
        assert data["is_first_push"] is False
        # Reused, not created.
        assert fake.created is False
        assert fake.push["branch"] == "dev"
        # Each vibe run is a full app → replace the tree.
        assert fake.push["preserve_existing_files"] is False

    def test_use_existing_missing_repo_returns_404_repo_missing(self, monkeypatch):
        run_id = "e" * 32
        _seed_run(run_id)
        fake = _FakeGitHubService(get_branches_404=True)
        _install(monkeypatch, fake)

        body = {
            "run_id": run_id,
            "projectExport": _project_export(),
            "deploy_config": {"repo_name": "my-vibe-app", "use_existing": True},
        }
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 404
        assert r.json()["detail"] == "repo_missing"
        # Must NOT silently create a repo when the user asked to reuse one.
        assert fake.created is False
