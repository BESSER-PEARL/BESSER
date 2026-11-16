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
import base64
import copy
import json
import os
import re
import tempfile
import time

import httpx
import pytest
import yaml
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import github_deploy_api, spec_driven_router
from besser.utilities.web_modeling_editor.backend.services.spec_driven import github_sync
from besser.utilities.web_modeling_editor.backend.services.deployment import repo_model
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    SYNC_BRANCH_PREFIX,
    GitHubService,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartRunEntry,
    SmartRunRegistry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _clear_registry
from besser.utilities.web_modeling_editor.backend.settings import get_settings
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime

BASE = "a" * 40
HEAD = "b" * 40
OURS = "c" * 40
MERGED = "e" * 40
SESSION = {"X-GitHub-Session": "sess"}


@pytest.fixture(autouse=True)
def shared_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_run_workspace_root", str(tmp_path))
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
    assert outcome == {"commit_sha": MERGED, "status": "merged", "conflict_branch": None, "own_commit_sha": OURS,
                       "warnings": []}
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
    assert outcome == {"commit_sha": OURS, "status": "conflict", "conflict_branch": side, "own_commit_sha": OURS,
                       "warnings": []}
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
    monkeypatch.setattr(github_sync, "create_github_service", lambda token: _ImportGitHub())
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda _sessions, sid: "token")

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
                                     keep_path=None, model_merge=None, base_hashes=None):
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

    async def get_tree_entries(self, owner, repo_name, commit_sha):
        return {}

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
    asyncio.run(current_runtime().registry.put(run_id, SmartRunEntry(
        file_path=str(run_dir / "main.py"), file_name="main.py", is_zip=False,
        temp_dir=str(run_dir), created_at=time.time(),
    )))
    return run_id


def _push(monkeypatch, fake, run_id, **deploy):
    monkeypatch.setattr(github_sync, "create_github_service", lambda token: fake)
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda _sessions, sid: "token")
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
        record = json.load(fh)
    assert {key: record[key] for key in ("owner", "repo", "commitSha")} == {"owner": "me", "repo": "app", "commitSha": OURS}
    # The files the run now holds, so the next push deletes only what the run removed.
    assert "main.py" in record["paths"]


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

    async def get_authenticated_user(self):
        return {"login": "ana"}

    async def create_blob(self, owner, repo, content):
        return f"blob-{len(content)}"

    async def ensure_first_commit(self, owner, repo, branch, path, content, message):
        pass

    async def commit_tree(self, owner, repo, branch, entries, message, layer_on_parent, base_commit_sha=None,
                          model_merge=None):
        self.commit = dict(paths={e["path"]: e["sha"] for e in entries}, message=message,
                           layer_on_parent=layer_on_parent, base=base_commit_sha)
        return {"commit_sha": OURS, "status": "pushed", "conflict_branch": None}

    async def add_topic(self, owner, repo, topic):
        pass

    async def get_commits(self, owner, repo, path=None, per_page=30, ref=None):
        return [{"sha": sha, "message": message, "author": "Ana", "date": "", "html_url": "", "parents": parents}
                for sha, message, parents in [("2" * 40, "Add Loan", ["1" * 40]),
                                              ("1" * 40, "First\n\nBESSER-Origin: editor", [])]]

    async def get_branch_head(self, owner, repo, branch):
        return "2" * 40

    async def get_tree_entries(self, owner, repo_name, commit_sha):
        return {}

    async def get_file_content(self, owner, repo, path, branch="main", ref=None):
        if ref is None and self.legacy_exists and path == repo_model.LEGACY_PROJECT_FILE:
            return {"content": json.dumps(_project())}
        content = self.files_at.get((path, ref))
        return {"content": content} if content else None


def _sidebar(monkeypatch, fake):
    monkeypatch.setattr(github_deploy_api, "create_github_service", lambda token: fake)
    monkeypatch.setattr(github_deploy_api, "get_user_token", lambda _sessions, sid: "token")


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


def _model_with_an_unknown_type() -> dict:
    """An attribute typed by an enumeration the model does not have."""
    model = copy.deepcopy(CLASS_DIAGRAM_MODEL)
    model["elements"]["attr-1"]["name"] = "+ kind: RoomType"
    return model


def _model_with_two_enumerations_of_one_name() -> dict:
    """Two enumerations left at the palette's default name, as in a repository of a live two-person test."""
    model = copy.deepcopy(CLASS_DIAGRAM_MODEL)
    for element_id in ("enum-1", "enum-2"):
        model["elements"][element_id] = {"type": "Enumeration", "name": "Enumeration", "attributes": []}
    return model


def _as_the_editor_saves_it(class_model) -> dict:
    """A class diagram beside another diagram, as in every project: that one survives a class diagram that fails."""
    envelope = _envelope(class_model)
    envelope["project"]["diagrams"]["StateMachineDiagram"] = [{
        "id": "sm1", "title": "Empty",
        "model": {"type": "StateMachineDiagram", "elements": {}, "relationships": {}},
    }]
    envelope["project"]["currentDiagramIndices"]["StateMachineDiagram"] = 0
    return envelope


def _history_changes(monkeypatch, newer: str, older: str) -> list[list[str]]:
    fake = _SidebarGitHub(files_at={
        (repo_model.MODEL_FILE, "2" * 40): newer,
        (repo_model.MODEL_FILE, "1" * 40): older,
    })
    _sidebar(monkeypatch, fake)
    r = _call("GET", "/besser_api/github/model-history", headers=SESSION,
              params={"owner": "me", "repo": "app"})
    assert r.status_code == 200, r.text
    return [commit["changes"] for commit in r.json()["commits"]]


_UNREADABLE_VERSIONS = {
    "unknown_type": json.dumps(_as_the_editor_saves_it(_model_with_an_unknown_type())),
    "duplicate_enumerations": json.dumps(_as_the_editor_saves_it(_model_with_two_enumerations_of_one_name())),
    "class_diagram_only": json.dumps(_envelope(_model_with_an_unknown_type())),
    "not_json": "<<<<<<< HEAD {}",
}


@pytest.mark.parametrize("unreadable", _UNREADABLE_VERSIONS.values(), ids=_UNREADABLE_VERSIONS.keys())
def test_the_history_does_not_call_an_unreadable_newer_version_a_change(monkeypatch, unreadable):
    """Its row said every class was removed."""
    newest, first = _history_changes(monkeypatch, unreadable, json.dumps(_as_the_editor_saves_it(CLASS_DIAGRAM_MODEL)))
    assert newest == [repo_model.VERSION_NOT_COMPARED]
    assert "class Author added" in first


@pytest.mark.parametrize("unreadable", _UNREADABLE_VERSIONS.values(), ids=_UNREADABLE_VERSIONS.keys())
def test_the_history_does_not_call_a_version_after_an_unreadable_one_a_change(monkeypatch, unreadable):
    """Its row said every class was added."""
    newest, first = _history_changes(monkeypatch, json.dumps(_as_the_editor_saves_it(CLASS_DIAGRAM_MODEL)), unreadable)
    assert newest == [repo_model.VERSION_NOT_COMPARED]
    assert first == [repo_model.VERSION_NOT_COMPARED]


def test_the_history_still_lists_changes_beside_other_diagrams(monkeypatch):
    with_loan = copy.deepcopy(CLASS_DIAGRAM_MODEL)
    with_loan["elements"]["loan-class"] = {"type": "Class", "name": "Loan", "attributes": [], "methods": []}
    newest, first = _history_changes(
        monkeypatch,
        json.dumps(_as_the_editor_saves_it(with_loan)),
        json.dumps(_as_the_editor_saves_it(CLASS_DIAGRAM_MODEL)),
    )
    assert newest == ["class Loan added"]
    assert "class Author added" in first


def test_a_pull_request_does_not_list_an_unreadable_head_as_removals():
    changes = repo_model.describe_change(
        _as_the_editor_saves_it(CLASS_DIAGRAM_MODEL), _as_the_editor_saves_it(_model_with_an_unknown_type())
    )
    body = repo_model.pull_request_body(changes)
    assert repo_model.VERSION_NOT_COMPARED in body
    assert "removed" not in body


def test_deleting_every_class_is_still_listed_as_removals():
    empty = {"type": "ClassDiagram", "elements": {}, "relationships": {}}
    assert "class Author removed" in repo_model.describe_change(
        _as_the_editor_saves_it(CLASS_DIAGRAM_MODEL), _as_the_editor_saves_it(empty)
    )


def test_a_project_without_a_class_diagram_has_nothing_to_compare():
    no_class_diagram = _envelope()
    no_class_diagram["project"]["diagrams"] = {}
    assert repo_model.describe_change(None, no_class_diagram) == []
    assert repo_model.describe_change(no_class_diagram, _envelope()) == ["class Author added", "class Book added"]


def test_the_origin_trailer_round_trips():
    message = repo_model.with_origin("Add the model", repo_model.ORIGIN_AGENT)
    assert repo_model.commit_origin(message) == repo_model.ORIGIN_AGENT
    assert repo_model.commit_origin("Fix a typo") is None


def _trailers(message):
    return message.rstrip().split("\n\n")[-1].splitlines()


def test_ai_headers_come_before_the_origin_and_are_sanitised():
    message = repo_model.with_origin(
        "Add the model", repo_model.ORIGIN_AGENT,
        {"AI-Tool": "besser-spec-driven-agent/9.9", "AI-Model": "gateway\n\nBESSER-Origin: editor\x00" + "x" * 300,
         "AI-Role": "code-generation"},
    )
    lines = _trailers(message)
    assert [line.split(":")[0] for line in lines] == ["AI-Tool", "AI-Model", "AI-Role", "BESSER-Origin"]
    assert lines[0] == "AI-Tool: besser-spec-driven-agent/9.9"
    assert lines[-1] == "BESSER-Origin: spec-driven-agent"
    assert len(lines[1]) <= 120 and "\x00" not in message
    assert repo_model.commit_origin(message) == repo_model.ORIGIN_AGENT


def test_a_fake_origin_earlier_in_the_message_cannot_win():
    body = "Add the model\n\nBESSER-Origin: editor\n\nMore text"
    assert repo_model.commit_origin(body) is None
    message = repo_model.with_origin(body, repo_model.ORIGIN_AGENT, {"AI-Role": "code-generation"})
    assert repo_model.commit_origin(message) == repo_model.ORIGIN_AGENT
    assert repo_model.commit_origin("Fix\n\nBESSER-Origin: editor\n\nSigned-off-by: a") is None


def test_the_agent_push_commit_carries_ai_headers_with_the_public_model_label(monkeypatch, tmp_path):
    fake = _PushGitHub()
    run_id = _seed_run(tmp_path)
    (tmp_path / "run" / ".besser_recipe.json").write_text(json.dumps(
        {"llm_model": "gateway", "history": [{"mode": "modify"}]}), encoding="utf-8")
    r = _push(monkeypatch, fake, run_id)

    assert r.status_code == 200, r.text
    lines = _trailers(fake.push["message"])
    assert lines[0].startswith("AI-Tool: besser-spec-driven-agent/")
    assert lines[1:] == ["AI-Model: gateway", "AI-Role: feature-modification", "BESSER-Origin: spec-driven-agent"]


def test_a_push_without_a_recorded_model_omits_ai_model(monkeypatch, tmp_path):
    fake = _PushGitHub()
    r = _push(monkeypatch, fake, _seed_run(tmp_path))

    assert r.status_code == 200, r.text
    lines = _trailers(fake.push["message"])
    assert not any(line.startswith("AI-Model") for line in lines)
    assert lines[-2:] == ["AI-Role: code-generation", "BESSER-Origin: spec-driven-agent"]


def test_a_merge_commit_carries_only_the_origin():
    from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import _merge_message
    pushed = repo_model.with_origin("x", repo_model.ORIGIN_AGENT, {"AI-Model": "gateway"})
    assert _trailers(_merge_message("main", pushed)) == ["BESSER-Origin: spec-driven-agent"]


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


def test_agent_credentials_in_any_yaml_shape_never_reach_the_repo_model_file():
    """The line rule saw only ``key: value`` lines: a list item, a flow map and a block scalar kept their secrets."""
    envelope = _envelope()
    envelope["project"]["diagrams"]["AgentDiagram"] = [{
        "id": "ag1", "title": "Helper", "model": {},
        "configYaml": "databases:\n  - name: shop\n    password: list-item-pass\n"
                      "cache: {host: redis, password: flow-map-pass}\n"
                      "openai:\n  api_key: |\n    sk-block-scalar-value\n  language: en\n",
        "agentConfigCustomYaml": "custom:\n  - token: list-token-value\n  - retries: 3\n",
    }]

    text = repo_model.model_files(envelope)[repo_model.MODEL_FILE].decode("utf-8")
    saved = json.loads(text)["project"]["diagrams"]["AgentDiagram"][0]

    for secret in ("list-item-pass", "flow-map-pass", "sk-block-scalar-value", "list-token-value"):
        assert secret not in text
    config = yaml.safe_load(saved["configYaml"])
    assert config["databases"][0]["name"] == "shop" and config["cache"]["host"] == "redis"
    assert config["openai"]["language"] == "en"
    assert yaml.safe_load(saved["agentConfigCustomYaml"])["custom"][1] == {"retries": 3}


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


def test_a_new_project_repo_readme_holds_the_project_description_and_date(monkeypatch):
    """#750: the README said only "A BESSER project"; it now says what the project is and when it was made."""
    class _ReadmeGitHub(_SidebarGitHub):
        blobs = {}

        async def get_authenticated_user(self):
            return {"login": "me"}

        async def create_repository(self, repo_name, description, is_private, auto_init):
            return {"default_branch": "main", "html_url": f"https://github.com/me/{repo_name}"}

        async def create_blob(self, owner, repo, content):
            sha = f"blob-{len(self.blobs)}"
            self.blobs[sha] = content
            return sha

    def readme_for(project):
        fake = _ReadmeGitHub(legacy_exists=False)
        _sidebar(monkeypatch, fake)
        r = _call("POST", "/besser_api/github/project/create-repo", headers=SESSION, json={
            "repo_name": "library", "project_data": project, "project_export": _envelope(),
        })
        assert r.status_code == 200, r.text
        blob = fake.blobs[fake.commit["paths"]["README.md"]]
        return (blob if isinstance(blob, bytes) else base64.b64decode(blob)).decode("utf-8")

    described = readme_for({**_project(), "description": "  Loans and members of a city library.\n"})
    assert "\nLoans and members of a city library.\n\nCreated with [BESSER]" in described
    assert re.search(r"Created with \[BESSER\]\([^)]+\) on \d{4}-\d{2}-\d{2}\.", described)
    assert described.index("Loans and members") < described.index("## Files")

    for project in (_project(), {**_project(), "description": "   "}, {**_project(), "description": None}):
        readme = readme_for(project)
        assert "Created with [BESSER]" in readme
        assert "\n\n\n" not in readme


def test_the_deploy_readme_holds_the_description_and_the_generation_date():
    """#750: a generated app's README says what the project is and when BESSER generated it."""
    github = GitHubService("token")
    generated = re.compile(r"Generated with \[BESSER\]\(https://github.com/BESSER-PEARL/BESSER\) "
                           r"on \d{4}-\d{2}-\d{2}\.")

    for kind in ("webapp", "chatbot"):
        readme = github.generate_readme_content(
            "Library", owner="me", repo_name="library", deployment_type=kind, description="Loans and members.",
        )
        assert "</div>\n\nLoans and members.\n\nGenerated with [BESSER]" in readme
        assert generated.search(readme)
        bare = github.generate_readme_content("Library", deployment_type=kind)
        assert generated.search(bare)
        assert "\n\n\n" not in bare
        blank = github.generate_readme_content("Library", deployment_type=kind, description="  \n ")
        assert blank == bare


def test_a_description_holding_the_besser_block_markers_survives_a_push():
    """The README block markers in a description would make a later push strip or replace the user's text."""
    from besser.utilities.provenance import (
        BESSER_BLOCK_BEGIN, BESSER_BLOCK_END, readme_about, strip_besser_block, with_besser_block,
    )

    description = f"Loans. {BESSER_BLOCK_BEGIN} keep me {BESSER_BLOCK_END} and members."
    readme = "# Library\n\n" + readme_about(description, "Created")
    assert BESSER_BLOCK_BEGIN not in readme and BESSER_BLOCK_END not in readme
    assert strip_besser_block(readme) == readme
    block = f"{BESSER_BLOCK_BEGIN}\nBuilt with BESSER\n{BESSER_BLOCK_END}"
    pushed = with_besser_block(readme, block)
    assert "keep me" in pushed and pushed.rstrip().endswith(BESSER_BLOCK_END)


class _FailingGitHub:
    """Every GitHub call answers ``status``."""

    def __init__(self, status: int):
        self.status = status

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            request = httpx.Request("GET", "https://api.github.com/x")
            raise httpx.HTTPStatusError(
                "failed", request=request,
                response=httpx.Response(self.status, json={"message": "GitHub said no"}, request=request))
        return fail


@pytest.mark.parametrize("upstream, expected", [(401, 401), (404, 404), (422, 409), (503, 502)])
def test_the_sidebar_endpoints_map_github_errors_like_the_spec_driven_ones(monkeypatch, upstream, expected):
    """``/project/save`` and the rest answered GitHub's 422 as 422; the push endpoints answer 409."""
    _sidebar(monkeypatch, _FailingGitHub(upstream))

    save = _call("POST", "/besser_api/github/project/save", headers=SESSION, json={
        "owner": "me", "repo": "app", "commit_message": "Save", "project_data": _project(),
        "project_export": _envelope(), "base_commit_sha": BASE,
    })
    pull = _call("POST", "/besser_api/github/pull-request", headers=SESSION, json={
        "owner": "me", "repo": "app", "head": "besser/sync-1", "base": "main", "title": "Sync", "body": "b",
    })

    assert (save.status_code, pull.status_code) == (expected, expected), (save.text, pull.text)
    assert "GitHub said no" in save.json()["detail"]


@pytest.mark.parametrize("upstream, expected", [(401, 401), (422, 409), (503, 502)])
@pytest.mark.parametrize("method, url, body", [
    ("GET", "/besser_api/github/contents?owner=me&repo=app", None),
    ("GET", "/besser_api/github/commits?owner=me&repo=app", None),
    ("GET", "/besser_api/github/project/load?owner=me&repo=app", None),
    ("GET", f"/besser_api/github/project/load-commit?owner=me&repo=app&commit_sha={BASE}", None),
    ("POST", "/besser_api/github/project/create-repo", {"repo_name": "app", "project_data": _project()}),
])
def test_the_browse_load_and_create_endpoints_map_github_errors_too(monkeypatch, method, url, body, upstream, expected):
    """These answered every GitHub error with 500, so a revoked token never offered Reconnect."""
    _sidebar(monkeypatch, _FailingGitHub(upstream))

    r = _call(method, url, headers=SESSION, json=body)

    assert r.status_code == expected, r.text
    assert "GitHub said no" in r.json()["detail"]


def test_a_taken_repo_name_answers_409_with_githubs_reason(monkeypatch):
    class _NameTaken(_FailingGitHub):
        async def get_authenticated_user(self):
            return {"login": "me"}

    _sidebar(monkeypatch, _NameTaken(422))

    r = _call("POST", "/besser_api/github/project/create-repo", headers=SESSION,
              json={"repo_name": "app", "project_data": _project()})

    assert r.status_code == 409, r.text


def _github_answers(monkeypatch, status: int, body=None):
    """GitHubService's requests all get ``status``; the test client keeps its own transport."""
    real_client = httpx.AsyncClient
    github = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(**{"transport": github, **kw}))


@pytest.mark.parametrize("upstream, expected", [(401, 401), (503, 502)])
def test_file_exists_reports_a_github_error_instead_of_a_missing_file(monkeypatch, upstream, expected):
    """``file_exists`` read any non-200 as "missing": a revoked token answered 200 ``exists: false``."""
    _github_answers(monkeypatch, upstream, {"message": "Bad credentials"})
    monkeypatch.setattr(github_deploy_api, "get_user_token", lambda _sessions, sid: "token")

    r = _call("GET", "/besser_api/github/file/exists?owner=me&repo=app&file_path=a.json", headers=SESSION)

    assert r.status_code == expected, r.text
    assert "Bad credentials" in r.json()["detail"]


def test_file_exists_still_reads_404_as_missing(monkeypatch):
    _github_answers(monkeypatch, 404)

    assert asyncio.run(GitHubService("token").file_exists("me", "app", "a.json")) is False
