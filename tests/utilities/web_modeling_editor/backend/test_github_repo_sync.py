"""Keeping a project's GitHub repo and the editor in step.

Covers the failures behind "BESSER can't keep working on an existing repo":

- a repo imported on the backend was invisible to the Spec-Driven worker, so
  a modify run from it rebuilt the app from scratch;
- a push replaced the branch with its own tree, reverting commits other
  people made since the editor's last sync;
- the push validated the V2 export envelope as a bare project, so the B-UML
  ``.py`` files were never written;
- the sidebar and the agent wrote the model to two different files.
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
from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
from besser.utilities.web_modeling_editor.backend.services.deployment import (
    github_deploy_api,
    repo_model,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    SYNC_BRANCH_PREFIX,
    GitHubService,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SMART_RUN_REGISTRY,
    SmartRunEntry,
    SmartRunRegistry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _clear_registry

BASE = "a" * 40
HEAD = "b" * 40
OURS = "c" * 40
MERGED = "e" * 40
SESSION = {"X-GitHub-Session": "sess"}


@pytest.fixture(autouse=True)
def shared_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path))
    asyncio.run(_clear_registry())
    yield tmp_path
    asyncio.run(_clear_registry())


def _project(class_model=CLASS_DIAGRAM_MODEL) -> dict:
    return {
        "id": "proj-1",
        "type": "BesserProject",
        "name": "Library",
        "createdAt": "2026-04-15T00:00:00Z",
        "diagrams": {"ClassDiagram": [{"id": "cd1", "title": "Library", "model": class_model}]},
        "currentDiagramIndices": {"ClassDiagram": 0},
    }


def _envelope(class_model=CLASS_DIAGRAM_MODEL) -> dict:
    """The V2 export envelope the editor really sends (buildProjectExportEnvelope)."""
    return {"project": _project(class_model), "exportedAt": "2026-10-26T00:00:00Z", "version": "2.0.0"}


def _call(method: str, url: str, **kwargs) -> httpx.Response:
    async def run():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(run())


# ---------------------------------------------------------------------
# GitHubService.commit_tree: never revert newer commits
# ---------------------------------------------------------------------


class _RecordingGitHub(GitHubService):
    """GitHubService with the HTTP calls replaced by a record of what was asked."""

    def __init__(self, head=HEAD, merge_result=MERGED):
        super().__init__("token")
        self.head = head
        self.merge_result = merge_result
        self.commits = []
        self.refs = []
        self.merges = []
        self.deleted = []

    async def _get_ref_sha(self, owner, repo_name, branch):
        return self.head if branch == "main" else None

    async def _get_commit_tree_sha(self, owner, repo_name, commit_sha):
        return f"tree-of-{commit_sha[:4]}"

    async def _create_tree(self, owner, repo_name, tree, base_tree=None):
        return "new-tree"

    async def _create_commit(self, owner, repo_name, message, tree_sha, parent_sha):
        self.commits.append({"parent": parent_sha, "tree": tree_sha})
        return OURS

    async def _update_or_create_ref(self, owner, repo_name, branch, commit_sha, parent_sha):
        self.refs.append((branch, commit_sha))

    async def merge_branch(self, owner, repo_name, base, head, commit_message):
        self.merges.append((base, head))
        return self.merge_result

    async def delete_branch(self, owner, repo_name, branch):
        self.deleted.append(branch)


def _commit(github, base_commit_sha):
    return asyncio.run(github.commit_tree(
        "me", "app", "main", [{"path": "a.py"}], "msg",
        layer_on_parent=False, base_commit_sha=base_commit_sha,
    ))


def test_a_push_after_someone_else_committed_is_merged_not_reverted():
    github = _RecordingGitHub(head=HEAD)

    outcome = _commit(github, BASE)

    # Our commit sits on the commit we started from, not on their newer one...
    assert github.commits == [{"parent": BASE, "tree": "new-tree"}]
    # ...and reaches main only through a merge, which keeps both sides.
    side = f"{SYNC_BRANCH_PREFIX}{OURS[:7]}"
    assert github.merges == [("main", side)]
    assert ("main", OURS) not in github.refs
    assert outcome == {"commit_sha": MERGED, "status": "merged", "conflict_branch": None, "own_commit_sha": OURS}
    assert github.deleted == [side]


def test_a_push_on_an_unchanged_branch_fast_forwards():
    github = _RecordingGitHub(head=BASE)

    outcome = _commit(github, BASE)

    assert github.refs == [("main", OURS)]
    assert github.merges == []
    assert outcome["status"] == "pushed"


def test_a_merge_github_cannot_resolve_is_kept_on_its_branch():
    github = _RecordingGitHub(head=HEAD, merge_result=None)

    outcome = _commit(github, BASE)

    side = f"{SYNC_BRANCH_PREFIX}{OURS[:7]}"
    assert outcome == {"commit_sha": OURS, "status": "conflict", "conflict_branch": side, "own_commit_sha": OURS}
    assert github.deleted == []
    assert ("main", OURS) not in github.refs


def test_without_a_base_the_push_builds_on_the_head():
    github = _RecordingGitHub(head=HEAD)

    _commit(github, None)

    assert github.commits[0]["parent"] == HEAD
    assert github.refs == [("main", OURS)]


# ---------------------------------------------------------------------
# Import: the worker must see the imported repo
# ---------------------------------------------------------------------


class _ImportGitHub:
    async def get_branches(self, owner, repo_name, per_page=100):
        return ["main"]

    async def get_default_branch(self, owner, repo_name):
        return "main"

    async def get_branch_head(self, owner, repo_name, branch):
        return HEAD

    async def download_repo_tarball(self, owner, repo, ref, dest_dir=None):
        dest_dir = dest_dir or tempfile.mkdtemp(prefix="besser_gh_import_")
        os.makedirs(os.path.join(dest_dir, "backend"), exist_ok=True)
        with open(os.path.join(dest_dir, "backend", "main.py"), "w", encoding="utf-8") as fh:
            fh.write("print('hi')\n")
        os.makedirs(os.path.join(dest_dir, "buml"), exist_ok=True)
        with open(os.path.join(dest_dir, "buml", "diagrams.json"), "w", encoding="utf-8") as fh:
            json.dump(_envelope(), fh)
        return dest_dir


def test_a_repo_imported_by_the_backend_is_found_by_the_worker(monkeypatch, shared_workspace):
    monkeypatch.setattr(spec_driven_router, "create_github_service", lambda token: _ImportGitHub())
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda sid: "token")

    r = _call("POST", "/besser_api/spec-driven/import-github-run",
              json={"owner": "me", "repo": "app"}, headers=SESSION)
    assert r.status_code == 200, r.text
    assert r.json()["commit_sha"] == HEAD

    # The worker is another process: its registry starts empty and can only
    # find the run on the shared workspace volume.
    worker_registry = SmartRunRegistry()
    entry = asyncio.run(worker_registry.get(r.json()["run_id"]))
    assert entry is not None
    assert os.path.commonpath([str(shared_workspace), entry.temp_dir]) == str(shared_workspace)
    assert os.path.isfile(os.path.join(entry.temp_dir, "backend", "main.py"))


# ---------------------------------------------------------------------
# Push endpoint
# ---------------------------------------------------------------------


class _PushGitHub(_ImportGitHub):
    def __init__(self, status="pushed", render_yaml=None):
        self.status = status
        self.render_yaml = render_yaml
        self.push = None
        self.topics = []
        self.pull_request = None

    async def get_authenticated_user(self):
        return {"login": "me"}

    async def push_directory_to_repo(self, owner, repo_name, directory_path, commit_message="",
                                     branch="main", preserve_existing_files=False, base_commit_sha=None,
                                     keep_path=None, model_merge=None):
        files = sorted(
            os.path.relpath(os.path.join(root, name), directory_path).replace("\\", "/")
            for root, _dirs, names in os.walk(directory_path) for name in names
        )
        self.push = dict(branch=branch, base_commit_sha=base_commit_sha, files=files,
                         message=commit_message)
        sha = MERGED if self.status == "merged" else OURS
        return {"commit_sha": sha, "total_files": len(files), "status": self.status, "conflict_branch": None,
                "own_commit_sha": OURS, "uploaded_files": files, "deleted_files": []}

    async def add_topic(self, owner, repo_name, topic):
        self.topics.append(topic)

    async def get_file_content(self, owner, repo_name, file_path, branch="main", ref=None):
        return {"content": self.render_yaml} if file_path == "render.yaml" and self.render_yaml else None

    async def create_pull_request(self, owner, repo_name, head, base, title, body):
        self.pull_request = dict(head=head, base=base)
        return {"number": 3, "html_url": "https://github.com/me/app/pull/3"}


def _seed_run(tmp_path) -> str:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "main.py").write_text("print('hi')\n", encoding="utf-8")
    run_id = "f" * 32
    asyncio.run(SMART_RUN_REGISTRY.put(run_id, SmartRunEntry(
        file_path=str(run_dir / "main.py"), file_name="main.py", is_zip=False,
        temp_dir=str(run_dir), created_at=time.time(),
    )))
    return run_id


def _push(monkeypatch, fake, run_id, **deploy):
    monkeypatch.setattr(spec_driven_router, "create_github_service", lambda token: fake)
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda sid: "token")
    body = {"run_id": run_id, "projectExport": _envelope(),
            "deploy_config": {"repo_name": "app", "use_existing": True, **deploy}}
    return _call("POST", "/besser_api/spec-driven/push-to-github", json=body, headers=SESSION)


def test_the_push_writes_every_model_file_from_the_real_export_envelope(monkeypatch, tmp_path):
    fake = _PushGitHub()
    r = _push(monkeypatch, fake, _seed_run(tmp_path))

    assert r.status_code == 200, r.text
    assert {"buml/diagrams.json", "buml/domain_model.py", ".besser/project.json"} <= set(fake.push["files"])
    assert fake.topics == [repo_model.BESSER_TOPIC]
    assert repo_model.commit_origin(fake.push["message"]) == repo_model.ORIGIN_AGENT


def test_the_push_builds_on_the_commit_the_editor_last_synced(monkeypatch, tmp_path):
    fake = _PushGitHub()
    r = _push(monkeypatch, fake, _seed_run(tmp_path), base_commit_sha=BASE)

    assert r.status_code == 200, r.text
    assert fake.push["base_commit_sha"] == BASE
    assert r.json()["commit_sha"] == OURS
    assert r.json()["status"] == "pushed"
    assert r.json()["base_run_id"] is None


def _record_seed(run_dir, owner, repo, sha):
    """What import-github-run writes into a run, and seeding copies into every modify run after it."""
    with open(os.path.join(run_dir, runner_module.GITHUB_SOURCE_FILENAME), "w", encoding="utf-8") as fh:
        json.dump({"owner": owner, "repo": repo, "commitSha": sha}, fh)


def test_a_run_pushes_on_the_commit_it_started_from_not_the_links_latest(monkeypatch, tmp_path):
    # The run was seeded from commit BASE. Afterwards the user pulled HEAD, which
    # moved the link's base to HEAD. Pushing that run with base HEAD would treat
    # the pulled commits as replaced by the run's older tree and revert them.
    run_id = _seed_run(tmp_path)
    _record_seed(str(tmp_path / "run"), "me", "app", BASE)
    fake = _PushGitHub()

    r = _push(monkeypatch, fake, run_id, base_commit_sha=HEAD)

    assert r.status_code == 200, r.text
    assert fake.push["base_commit_sha"] == BASE


def test_a_seed_from_another_repo_does_not_set_the_base(monkeypatch, tmp_path):
    run_id = _seed_run(tmp_path)
    _record_seed(str(tmp_path / "run"), "me", "other-repo", BASE)
    fake = _PushGitHub()

    _push(monkeypatch, fake, run_id, base_commit_sha=HEAD)

    assert fake.push["base_commit_sha"] == HEAD


def test_a_pushed_run_records_its_commit_for_the_runs_seeded_from_it(monkeypatch, tmp_path):
    run_id = _seed_run(tmp_path)
    _record_seed(str(tmp_path / "run"), "me", "app", BASE)

    _push(monkeypatch, _PushGitHub(), run_id, base_commit_sha=BASE)

    with open(tmp_path / "run" / runner_module.GITHUB_SOURCE_FILENAME, encoding="utf-8") as fh:
        assert json.load(fh) == {"owner": "me", "repo": "app", "commitSha": OURS}


def test_a_merged_push_hands_back_the_merged_repo_as_the_next_base(monkeypatch, tmp_path):
    fake = _PushGitHub(status="merged")
    r = _push(monkeypatch, fake, _seed_run(tmp_path), base_commit_sha=BASE)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged"
    base_run = asyncio.run(SmartRunRegistry().get(r.json()["base_run_id"]))
    assert base_run is not None


def test_a_pull_request_push_lands_on_its_own_branch(monkeypatch, tmp_path):
    fake = _PushGitHub()
    r = _push(monkeypatch, fake, _seed_run(tmp_path), base_commit_sha=BASE, push_mode="pull_request")

    assert r.status_code == 200, r.text
    assert fake.push["branch"].startswith("besser/update-")
    assert fake.pull_request == {"head": fake.push["branch"], "base": "main"}
    assert r.json()["status"] == "pull_request"
    assert r.json()["pull_request_url"] == "https://github.com/me/app/pull/3"
    assert r.json()["pull_request_branch"] == fake.push["branch"]


# ---------------------------------------------------------------------
# Sidebar save and model history
# ---------------------------------------------------------------------


class _SidebarGitHub:
    def __init__(self, files_at=None, legacy_exists=True):
        self.files_at = files_at or {}
        self.legacy_exists = legacy_exists
        self.commit = None

    async def create_blob(self, owner, repo, content):
        return f"blob-{len(content)}"

    async def file_exists(self, owner, repo, path, branch):
        return self.legacy_exists and path == repo_model.LEGACY_PROJECT_FILE

    async def commit_tree(self, owner, repo, branch, entries, message, layer_on_parent, base_commit_sha=None,
                          model_merge=None):
        self.commit = dict(paths={e["path"]: e["sha"] for e in entries}, message=message,
                           layer_on_parent=layer_on_parent, base=base_commit_sha)
        return {"commit_sha": OURS, "status": "pushed", "conflict_branch": None}

    async def add_topic(self, owner, repo, topic):
        pass

    async def get_commits(self, owner, repo, path=None, per_page=30, ref=None):
        return [{"sha": sha, "message": message, "author": "Ana", "date": "", "html_url": ""}
                for sha, message in [("2" * 40, "Add Loan"), ("1" * 40, f"First\n\nBESSER-Origin: editor")]]

    async def get_file_content(self, owner, repo, path, branch="main", ref=None):
        content = self.files_at.get((path, ref))
        return {"content": content} if content else None


def _sidebar(monkeypatch, fake):
    monkeypatch.setattr(github_deploy_api, "create_github_service", lambda token: fake)
    monkeypatch.setattr(github_deploy_api, "get_user_token", lambda sid: "token")


def test_the_sidebar_saves_the_one_model_file_and_drops_the_legacy_one(monkeypatch):
    fake = _SidebarGitHub()
    _sidebar(monkeypatch, fake)

    r = _call("POST", "/besser_api/github/project/save", headers=SESSION, json={
        "owner": "me", "repo": "app", "commit_message": "Save", "project_data": _project(),
        "project_export": _envelope(), "base_commit_sha": BASE,
    })

    assert r.status_code == 200, r.text
    paths = fake.commit["paths"]
    assert paths[repo_model.LEGACY_PROJECT_FILE] is None  # deleted in the same commit
    assert {repo_model.MODEL_FILE, repo_model.LINK_FILE, "buml/domain_model.py"} <= set(paths)
    assert fake.commit["layer_on_parent"] is True
    assert fake.commit["base"] == BASE
    assert repo_model.commit_origin(fake.commit["message"]) == repo_model.ORIGIN_EDITOR


def test_the_model_history_says_what_each_commit_changed_and_who_made_it(monkeypatch):
    with_loan = json.loads(json.dumps(CLASS_DIAGRAM_MODEL))
    loan_id = "loan-class"
    with_loan["elements"][loan_id] = {
        "id": loan_id, "name": "Loan", "type": "Class", "owner": None,
        "bounds": {"x": 0, "y": 0, "width": 160, "height": 100}, "attributes": [], "methods": [],
    }
    fake = _SidebarGitHub(files_at={
        (repo_model.MODEL_FILE, "2" * 40): json.dumps(_envelope(with_loan)),
        (repo_model.MODEL_FILE, "1" * 40): json.dumps(_envelope()),
    })
    _sidebar(monkeypatch, fake)

    r = _call("GET", "/besser_api/github/model-history", headers=SESSION,
              params={"owner": "me", "repo": "app"})

    assert r.status_code == 200, r.text
    newest, first = r.json()["commits"]
    assert newest["changes"] == ["class Loan added"]
    assert newest["origin"] is None  # a person's commit
    assert first["origin"] == repo_model.ORIGIN_EDITOR


def test_the_origin_trailer_round_trips():
    message = repo_model.with_origin("Add the model", repo_model.ORIGIN_AGENT)
    assert repo_model.commit_origin(message) == repo_model.ORIGIN_AGENT
    assert repo_model.commit_origin("Fix a typo") is None


def test_agent_credentials_never_reach_the_repo_model_file():
    """The agent config (LLM keys, bot tokens, DB passwords) was written verbatim into buml/diagrams.json."""
    key = "sk-proj-" + "a1B2c3D4" * 4
    envelope = _envelope()
    envelope["project"]["diagrams"]["AgentDiagram"] = [{
        "id": "ag1", "title": "Helper", "model": {},
        "config": {"llm": {"provider": "openai", "api_key": key, "max_tokens": "1024"}},
        "agentConfigForm": {
            "nlp": {"openai_api_key": key, "mistral_api_key": "YOUR-API-KEY", "language": "en"},
            "platforms": {"telegram": {"enabled": True, "token": "123456:bot-token-value"}},
            "db": {"sqlDatabases": [{"name": "shop", "username": "admin", "password": "hunter2-db"}]},
        },
        "configYaml": f"nlp:\n  openai:\n    api_key: {key}\n  language: en\n"
                      "platforms:\n  telegram:\n    token: '123456:bot-token-value'\n",
        "agentConfigCustomYaml": "custom:\n  webhook_secret: s3cr3t-value\n  retries: 3\n",
    }]

    files = repo_model.model_files(envelope)
    text = files[repo_model.MODEL_FILE].decode("utf-8")
    saved = json.loads(text)["project"]["diagrams"]["AgentDiagram"][0]

    for secret in (key, "bot-token-value", "hunter2-db", "s3cr3t-value"):
        assert secret not in text
    assert saved["agentConfigForm"]["nlp"]["mistral_api_key"] == "YOUR-API-KEY"  # placeholders stay
    assert saved["agentConfigForm"]["nlp"]["language"] == "en"
    assert saved["config"]["llm"]["max_tokens"] == "1024"
    assert "language: en" in saved["configYaml"] and "retries: 3" in saved["agentConfigCustomYaml"]
    assert saved["agentConfigForm"]["db"]["sqlDatabases"][0]["username"] == "admin"
    # The class diagram is untouched.
    assert json.loads(text)["project"]["diagrams"]["ClassDiagram"] == envelope["project"]["diagrams"]["ClassDiagram"]


def test_a_new_project_repo_is_private_unless_the_user_asks_for_public(monkeypatch):
    """Repos defaulted to public, and a model can describe private work."""
    class _CreatingGitHub(_SidebarGitHub):
        created = []

        async def get_authenticated_user(self):
            return {"login": "me"}

        async def create_repository(self, repo_name, description, is_private, auto_init):
            self.created.append(is_private)
            return {"default_branch": "main", "html_url": f"https://github.com/me/{repo_name}"}

    fake = _CreatingGitHub(legacy_exists=False)
    _sidebar(monkeypatch, fake)
    body = {"repo_name": "library", "project_data": _project(), "project_export": _envelope()}

    for extra in ({}, {"is_private": False}):
        r = _call("POST", "/besser_api/github/project/create-repo", headers=SESSION, json={**body, **extra})
        assert r.status_code == 200, r.text
    assert fake.created == [True, False]
