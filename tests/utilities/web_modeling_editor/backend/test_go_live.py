"""Go live: deploy the app that is in the project's repo, which is the agent's app.

Until now Deploy regenerated a second app from the diagrams into its own repo,
so the app the Spec-Driven Agent built never went live. These tests pin the
pieces that replace it: reading the repo's real layout, writing Render files
for it, committing them into the project repo, and keeping them through later
agent pushes.
"""

import asyncio
import os
import shutil
import time

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.constants import constants
from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
from besser.utilities.web_modeling_editor.backend.services.deployment import github_deploy_api, render_blueprint
from besser.utilities.web_modeling_editor.backend.services.deployment.github_deploy_api import _add_deployment_configs
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SMART_RUN_REGISTRY,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _clear_registry

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HEAD = "b" * 40
OURS = "c" * 40
SESSION = {"X-GitHub-Session": "sess"}


def _write(root, path, content=""):
    target = os.path.join(root, *path.split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(content)


def _web_app(root, prefix="web_app/", out_dir='outDir: "build",', app_var="app", reads_api_url=True):
    """An app as the agent lays it out: FastAPI backend + Vite React frontend."""
    _write(root, f"{prefix}backend/main_api.py", f"from fastapi import FastAPI\n{app_var} = FastAPI()\n")
    _write(root, f"{prefix}backend/requirements.txt", "fastapi\n")
    _write(root, f"{prefix}frontend/package.json", "{}")
    _write(root, f"{prefix}frontend/vite.config.ts", f"export default defineConfig({{ build: {{ {out_dir} }} }})\n")
    api = "import.meta.env.VITE_API_URL" if reads_api_url else "'http://localhost:8000'"
    _write(root, f"{prefix}frontend/src/api.ts", f"export const API = {api};\n")
    _write(root, "buml/diagrams.json", "{}")


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setattr(constants, "DEPLOY_FROM_REPO", True)
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


# ---------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------


def test_the_agents_web_app_layout_is_found_under_web_app(tmp_path):
    _web_app(str(tmp_path))
    layout = render_blueprint.detect_layout(str(tmp_path))
    assert (layout.backend_dir, layout.frontend_dir, layout.frontend_out) == ("web_app/backend", "web_app/frontend", "build")
    assert layout.warnings == ()


def test_a_flat_app_with_vites_default_output_is_found(tmp_path):
    _web_app(str(tmp_path), prefix="", out_dir="")
    layout = render_blueprint.detect_layout(str(tmp_path))
    assert (layout.backend_dir, layout.frontend_dir, layout.frontend_out) == ("backend", "frontend", "dist")


def test_the_fastapi_instance_name_is_read_from_main_api(tmp_path):
    _web_app(str(tmp_path), app_var="api")
    files, _ = render_blueprint.deploy_files(render_blueprint.detect_layout(str(tmp_path)), "library", "abc123")
    assert "uvicorn main_api:api" in files["render.yaml"]


def test_a_frontend_that_ignores_the_api_url_is_flagged(tmp_path):
    _web_app(str(tmp_path), reads_api_url=False)
    assert "VITE_API_URL" in render_blueprint.detect_layout(str(tmp_path)).warnings[0]


@pytest.mark.parametrize("build, reason", [
    (lambda r: _write(r, "django/manage.py"), "Django"),
    (lambda r: (_web_app(r), _write(r, "web_app/agents/helper/helper.py")), "built-in agents"),
    (lambda r: (_web_app(r), _web_app(r, prefix="old/")), "more than one backend"),
    (lambda r: (_write(r, "backend/main_api.py", "print('no app')\n"), _write(r, "backend/requirements.txt")),
     "does not create a FastAPI app"),
    (lambda r: _write(r, "README.md"), "No FastAPI backend"),
    # Seen on a real app tree: main_api.py with no dependency list beside it.
    (lambda r: _write(r, "app/backend/main_api.py", "app = FastAPI()\n"), "no requirements.txt"),
])
def test_an_app_besser_cannot_deploy_is_refused_with_the_reason(tmp_path, build, reason):
    build(str(tmp_path))
    with pytest.raises(render_blueprint.LayoutRefused, match=reason):
        render_blueprint.detect_layout(str(tmp_path))


def test_render_files_follow_the_layout_and_skip_model_only_saves(tmp_path):
    _web_app(str(tmp_path))
    files, names = render_blueprint.deploy_files(render_blueprint.detect_layout(str(tmp_path)), "room_booking", "abc123")

    render_yaml = files["render.yaml"]
    assert render_blueprint.is_managed(render_yaml)
    assert "pip install -r web_app/backend/requirements.txt" in render_yaml
    assert "cd web_app/frontend && npm install && npm run build" in render_yaml
    assert "staticPublishPath: web_app/frontend/build" in render_yaml
    assert render_yaml.count("- buml/**") == 2  # one build filter per service
    assert names.backend == "room-booking-backend-abc123"
    assert files["web_app/frontend/.env.production"].endswith(
        "VITE_API_URL=https://room-booking-backend-abc123.onrender.com\n")


def test_the_classic_deploy_writes_exactly_what_it_wrote_before(tmp_path):
    # Byte-for-byte output of _add_deployment_configs before it moved onto
    # the shared service writer.
    os.makedirs(tmp_path / "frontend")
    _add_deployment_configs(str(tmp_path), "library_app", service_suffix="a1b2c3")

    for produced, golden in (("render.yaml", "render_webapp_golden.yaml"),
                             ("frontend/.env.production", "env_production_webapp_golden.txt")):
        with open(tmp_path / produced, encoding="utf-8") as a, open(os.path.join(FIXTURES, golden), encoding="utf-8") as b:
            assert a.read() == b.read()


# ---------------------------------------------------------------------
# POST /github/app/go-live
# ---------------------------------------------------------------------


class _RepoGitHub:
    """A repo at HEAD whose files are a local folder."""

    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.commit = None

    async def get_branch_head(self, owner, repo, branch):
        return HEAD

    async def download_repo_tarball(self, owner, repo, ref, dest_dir=None):
        shutil.copytree(self.source_dir, dest_dir, dirs_exist_ok=True)
        return dest_dir

    async def create_blob(self, owner, repo, content):
        return content.decode("utf-8")

    async def commit_tree(self, owner, repo, branch, entries, message, layer_on_parent, base_commit_sha=None):
        self.commit = dict(files={e["path"]: e["sha"] for e in entries}, layer=layer_on_parent, base=base_commit_sha)
        return {"commit_sha": OURS, "status": "pushed", "conflict_branch": None, "own_commit_sha": OURS}


def _call(method, url, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(run())


def _go_live(monkeypatch, fake, repo="room-booking"):
    monkeypatch.setattr(github_deploy_api, "create_github_service", lambda token: fake)
    monkeypatch.setattr(github_deploy_api, "get_user_token", lambda sid: "token")
    return _call("POST", "/besser_api/github/app/go-live", headers=SESSION,
                 json={"owner": "me", "repo": repo, "branch": "main"})


def test_going_live_commits_render_files_into_the_projects_own_repo(monkeypatch, tmp_path):
    _web_app(str(tmp_path))
    fake = _RepoGitHub(str(tmp_path))

    r = _go_live(monkeypatch, fake)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True and body["commit_sha"] == OURS
    assert body["render_deploy_url"] == "https://render.com/deploy?repo=https://github.com/me/room-booking"
    assert body["frontend_url"].startswith("https://room-booking-frontend-")
    assert set(fake.commit["files"]) == {"render.yaml", "web_app/frontend/.env.production"}
    assert fake.commit["layer"] is True and fake.commit["base"] == HEAD  # every other file stays


def test_going_live_again_keeps_the_same_services(monkeypatch, tmp_path):
    _web_app(str(tmp_path))
    files, _ = render_blueprint.deploy_files(render_blueprint.detect_layout(str(tmp_path)), "room-booking", "fed456")
    for path, content in files.items():
        _write(str(tmp_path), path, content)
    fake = _RepoGitHub(str(tmp_path))

    r = _go_live(monkeypatch, fake)

    assert r.json()["changed"] is False and fake.commit is None
    assert r.json()["backend_url"] == "https://room-booking-backend-fed456.onrender.com"


def test_a_render_yaml_besser_did_not_write_is_not_overwritten(monkeypatch, tmp_path):
    _web_app(str(tmp_path))
    _write(str(tmp_path), "render.yaml", "services: []\n")

    r = _go_live(monkeypatch, _RepoGitHub(str(tmp_path)))

    assert r.status_code == 409


def test_a_repo_without_a_deployable_app_says_why(monkeypatch, tmp_path):
    _write(str(tmp_path), "django/manage.py")

    r = _go_live(monkeypatch, _RepoGitHub(str(tmp_path)))

    assert r.status_code == 422 and "Django" in r.json()["detail"]


def test_go_live_stays_off_until_the_server_turns_it_on(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "DEPLOY_FROM_REPO", False)

    r = _go_live(monkeypatch, _RepoGitHub(str(tmp_path)))

    assert r.status_code == 404 and r.json()["detail"] == "go_live_disabled"
    assert _call("GET", "/besser_api/github/app/capabilities").json() == {"go_live": False}


# ---------------------------------------------------------------------
# Agent pushes keep the deploy files
# ---------------------------------------------------------------------


class _PushGitHub:
    def __init__(self, render_yaml):
        self.render_yaml = render_yaml
        self.files = None

    async def get_authenticated_user(self):
        return {"login": "me"}

    async def get_branches(self, owner, repo_name, per_page=100):
        return ["main"]

    async def get_default_branch(self, owner, repo_name):
        return "main"

    async def get_file_content(self, owner, repo_name, file_path, branch="main", ref=None):
        return {"content": self.render_yaml} if file_path == "render.yaml" and self.render_yaml else None

    async def push_directory_to_repo(self, owner, repo_name, directory_path, commit_message="", branch="main",
                                     preserve_existing_files=False, base_commit_sha=None):
        self.files = {}
        for root, _dirs, names in os.walk(directory_path):
            for name in names:
                path = os.path.join(root, name)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    self.files[os.path.relpath(path, directory_path).replace("\\", "/")] = fh.read()
        return {"commit_sha": OURS, "total_files": len(self.files), "status": "pushed",
                "conflict_branch": None, "own_commit_sha": OURS}

    async def add_topic(self, owner, repo_name, topic):
        pass


def _push_run(monkeypatch, tmp_path, fake):
    run_dir = tmp_path / "run"
    _web_app(str(run_dir))
    asyncio.run(SMART_RUN_REGISTRY.put("f" * 32, SmartRunEntry(
        file_path=str(run_dir / "buml" / "diagrams.json"), file_name="diagrams.json", is_zip=False,
        temp_dir=str(run_dir), created_at=time.time(),
    )))
    monkeypatch.setattr(spec_driven_router, "create_github_service", lambda token: fake)
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda sid: "token")
    return _call("POST", "/besser_api/spec-driven/push-to-github", headers=SESSION, json={
        "run_id": "f" * 32, "deploy_config": {"repo_name": "room-booking", "use_existing": True}})


def test_an_agent_push_keeps_the_live_apps_deploy_files(monkeypatch, tmp_path):
    # The push replaces the repo's tree with the run's, which has no render.yaml:
    # without regeneration every agent push after Go live took the app offline.
    managed = render_blueprint.MANAGED_MARKER + "\nservices:\n  - type: web\n    name: room-booking-backend-fed456\n"
    fake = _PushGitHub(render_yaml=managed)

    r = _push_run(monkeypatch, tmp_path, fake)

    assert r.status_code == 200, r.text
    assert "staticPublishPath: web_app/frontend/build" in fake.files["render.yaml"]
    assert "room-booking-backend-fed456" in fake.files["render.yaml"]  # same services, same URLs
    assert "room-booking-backend-fed456" in fake.files["web_app/frontend/.env.production"]


def test_an_agent_push_leaves_a_repo_that_never_went_live_alone(monkeypatch, tmp_path):
    fake = _PushGitHub(render_yaml="services: []\n")  # not BESSER's

    _push_run(monkeypatch, tmp_path, fake)

    assert "render.yaml" not in fake.files
