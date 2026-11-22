"""The HTTP answers of the GitHub push and import when GitHub refuses or fails.

The work is in ``services/spec_driven/github_sync.py``; the routes map its
``UpstreamError`` and GitHub's ``HTTPStatusError`` to these statuses.
"""

import httpx
import pytest

from tests.utilities.web_modeling_editor.backend.spec_driven import test_import_github_run as importing
from tests.utilities.web_modeling_editor.backend.spec_driven import test_push_smart_to_github as pushing
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _clear_registry

SESSION = {"X-GitHub-Session": "sess"}
RUN_ID = "b" * 32


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    import asyncio

    from besser.utilities.web_modeling_editor.backend.settings import get_settings

    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path))
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


def _github_error(status: int, message: str = "Nope") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.github.com/user/repos")
    response = httpx.Response(status, json={"message": message}, request=request)
    return httpx.HTTPStatusError(message, request=request, response=response)


def _push(monkeypatch, fake):
    pushing._seed_run(RUN_ID, with_secret_env=False)
    pushing._install(monkeypatch, fake)
    return pushing._post({
        "run_id": RUN_ID,
        "projectExport": pushing._project_export(),
        "deploy_config": {"repo_name": "app"},
    }, headers=SESSION)


class _NoLogin(pushing._FakeGitHubService):
    async def get_authenticated_user(self):
        return {}


class _NoCommit(pushing._FakeGitHubService):
    async def push_directory_to_repo(self, *args, **kwargs):
        return {"commit_sha": None}


class _RefusedCreate(pushing._FakeGitHubService):
    def __init__(self, status):
        super().__init__()
        self.status = status

    async def create_repository(self, *args, **kwargs):
        raise _github_error(self.status, "name already exists on this account")


def test_a_push_for_a_user_github_does_not_name_is_401(monkeypatch):
    r = _push(monkeypatch, _NoLogin())
    assert (r.status_code, r.json()) == (401, {"detail": "Could not resolve GitHub user from session."})


def test_a_push_github_made_no_commit_for_is_500(monkeypatch):
    r = _push(monkeypatch, _NoCommit())
    assert (r.status_code, r.json()) == (500, {"detail": "Failed to create GitHub commit for pushed files."})


@pytest.mark.parametrize("github_status, status, detail", [
    (422, 409, "name already exists on this account"),
    (403, 403, "name already exists on this account"),
    (404, 404, "name already exists on this account"),
    (503, 502, "GitHub upstream error: name already exists on this account"),
])
def test_a_push_github_refuses_maps_its_status(monkeypatch, github_status, status, detail):
    r = _push(monkeypatch, _RefusedCreate(github_status))
    assert (r.status_code, r.json()) == (status, {"detail": detail})


def _import(monkeypatch, fake):
    importing._install(monkeypatch, fake)
    return importing._post({"owner": "ana", "repo": "shop"}, headers=SESSION)


class _NoHead(importing._FakeGitHubService):
    async def get_branch_head(self, owner, repo_name, branch):
        return None


class _RefusedDownload(importing._FakeGitHubService):
    def __init__(self, error):
        super().__init__()
        self.error = error

    async def download_repo_tarball(self, *args, **kwargs):
        raise self.error


def test_an_import_of_a_branch_without_a_head_is_repo_missing(monkeypatch):
    r = _import(monkeypatch, _NoHead())
    assert (r.status_code, r.json()) == (404, {"detail": "repo_missing"})


@pytest.mark.parametrize("error, status, detail", [
    (_github_error(404), 404, "repo_missing"),
    (_github_error(422, "Too big"), 409, "Too big"),
    (_github_error(500, "Down"), 502, "GitHub upstream error: Down"),
    (ValueError("Repository archive exceeds the 100 MB size cap"), 400,
     "Repository archive exceeds the 100 MB size cap"),
    (RuntimeError("boom"), 500, "An internal error occurred during GitHub import."),
])
def test_an_import_github_refuses_maps_its_status(monkeypatch, error, status, detail):
    r = _import(monkeypatch, _RefusedDownload(error))
    assert (r.status_code, r.json()) == (status, {"detail": detail})


@pytest.mark.parametrize("github_status, status, detail", [
    (401, 401, "Nope"),
    (403, 403, "Nope"),
    (404, 404, "Nope"),
    (422, 409, "Nope"),
    (429, 429, "Nope"),
    (503, 502, "GitHub upstream error: Nope"),
])
def test_deploy_webapp_shares_the_github_status_mapping(github_status, status, detail):
    # /deploy-webapp mapped GitHub's statuses inline, the same way, before it used the helper.
    from besser.utilities.web_modeling_editor.backend.routers.github_deploy_api import _github_status_error

    error = _github_status_error(_github_error(github_status), "deploy_webapp_to_github")
    assert (error.status_code, error.detail) == (status, detail)
