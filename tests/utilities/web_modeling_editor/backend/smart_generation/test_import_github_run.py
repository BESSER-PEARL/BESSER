"""HTTP-level tests for POST /besser_api/import-github-run.

The endpoint downloads an existing BESSER-created repo, registers its
code tree as a run (usable as a modify seed) and returns the repo's
re-importable model. These tests mock ``GitHubService`` (patching the
router's factory + token lookup) so nothing touches the real GitHub API.
"""

import asyncio
import json
import os
import tempfile

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import (
    smart_generation_router as router_module,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_runner import (
    _clear_registry,
)


BASE_URL = "http://testserver"
IMPORT_URL = "/besser_api/import-github-run"


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


# ---------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------


def _project_export() -> dict:
    """A minimal re-importable V2 project-export envelope."""
    return {
        "id": "test-project",
        "type": "BesserProject",
        "name": "TestProject",
        "diagrams": {
            "ClassDiagram": [{"id": "cd1", "title": "Library", "model": {}}]
        },
        "currentDiagramIndices": {"ClassDiagram": 0},
    }


class _FakeGitHubService:
    """Records calls and materialises a fake extracted repo tree.

    ``download_repo_tarball`` builds a temp directory (as the real method
    would after extraction) and remembers its path so the test can assert
    that the registered run's ``temp_dir`` points at exactly that tree.
    """

    def __init__(self, *, with_model=True, branches=None, get_branches_404=False):
        self._with_model = with_model
        self._branches = ["main"] if branches is None else branches
        self._get_branches_404 = get_branches_404
        self.extract_dir = None
        self.tarball_ref = None

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

    async def download_repo_tarball(self, owner, repo, ref):
        self.tarball_ref = ref
        d = tempfile.mkdtemp(prefix="besser_gh_import_test_")
        # A code file — the modify seed reads this via temp_dir.
        os.makedirs(os.path.join(d, "backend"), exist_ok=True)
        with open(os.path.join(d, "backend", "main.py"), "w", encoding="utf-8") as fh:
            fh.write("print('hi')\n")
        if self._with_model:
            os.makedirs(os.path.join(d, "buml"), exist_ok=True)
            with open(
                os.path.join(d, "buml", "diagrams.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(_project_export(), fh)
        self.extract_dir = d
        return d


def _install(monkeypatch, fake):
    """Patch the GitHub factory + token lookup in the router namespace."""
    monkeypatch.setattr(router_module, "create_github_service", lambda token: fake)
    monkeypatch.setattr(router_module, "get_user_token", lambda sid: "fake-token")


def _post(body: dict, headers: dict | None = None) -> httpx.Response:
    transport = ASGITransport(app=app)

    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.post(IMPORT_URL, json=body, headers=headers or {})

    return asyncio.run(_run())


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestImportGitHubRun:
    def test_repo_with_model_returns_run_and_project(self, monkeypatch):
        fake = _FakeGitHubService(with_model=True)
        _install(monkeypatch, fake)

        body = {"owner": "test-owner", "repo": "my-app"}
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200
        data = r.json()

        assert data["has_model"] is True
        assert data["project"] == _project_export()
        assert data["owner"] == "test-owner"
        assert data["repo"] == "my-app"
        assert data["branch"] == "main"

        # The run is registered and its temp_dir is the extracted repo tree.
        run_id = data["run_id"]
        entry = asyncio.run(SMART_RUN_REGISTRY.get(run_id))
        assert entry is not None
        assert entry.temp_dir == fake.extract_dir

    def test_repo_without_model_has_model_false(self, monkeypatch):
        fake = _FakeGitHubService(with_model=False)
        _install(monkeypatch, fake)

        body = {"owner": "test-owner", "repo": "my-app"}
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200
        data = r.json()

        assert data["has_model"] is False
        assert data["project"] is None
        assert data["message"]  # a human-readable hint is present

        # A run is still registered so the code can be used as a modify seed.
        run_id = data["run_id"]
        entry = asyncio.run(SMART_RUN_REGISTRY.get(run_id))
        assert entry is not None
        assert entry.temp_dir == fake.extract_dir

    def test_requested_branch_is_honoured(self, monkeypatch):
        fake = _FakeGitHubService(branches=["main", "dev"])
        _install(monkeypatch, fake)

        body = {"owner": "test-owner", "repo": "my-app", "branch": "dev"}
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 200
        assert r.json()["branch"] == "dev"
        # The resolved branch is the ref actually downloaded.
        assert fake.tarball_ref == "dev"

    def test_missing_github_session_returns_401(self, monkeypatch):
        fake = _FakeGitHubService()
        _install(monkeypatch, fake)

        body = {"owner": "test-owner", "repo": "my-app"}
        r = _post(body)  # no X-GitHub-Session header
        assert r.status_code == 401

    def test_missing_repo_returns_404_repo_missing(self, monkeypatch):
        fake = _FakeGitHubService(get_branches_404=True)
        _install(monkeypatch, fake)

        body = {"owner": "test-owner", "repo": "does-not-exist"}
        r = _post(body, headers={"X-GitHub-Session": "sess"})
        assert r.status_code == 404
        assert r.json()["detail"] == "repo_missing"
