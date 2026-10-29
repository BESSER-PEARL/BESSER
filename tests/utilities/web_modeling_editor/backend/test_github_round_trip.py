"""Editor and Spec-Driven Agent round trips against a git-backed GitHub stand-in.

``fake_github.FakeGitHub`` answers GitHub's REST API with real git objects, so
these tests run the real ``GitHubService`` requests (blobs, trees, merges,
paging) and assert on the files GitHub would hold afterwards.
"""

import asyncio
import json
import os
import time
import uuid

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
from besser.utilities.web_modeling_editor.backend.services.deployment import github_deploy_api, repo_model
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SMART_RUN_REGISTRY,
    SmartRunEntry,
    _seed_workspace_from_base,
    create_run_workspace,
)
from tests.utilities.web_modeling_editor.backend.fake_github import FakeGitHub, use_fake_github
from tests.utilities.web_modeling_editor.backend.spec_driven.test_model_assembly import CLASS_DIAGRAM_MODEL
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _clear_registry

SESSION = {"X-GitHub-Session": "sess"}


@pytest.fixture
def github(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path / "runs"))
    asyncio.run(_clear_registry())
    fake = FakeGitHub(root=str(tmp_path / "github"))
    use_fake_github(monkeypatch, fake)
    for module in (spec_driven_router, github_deploy_api):
        monkeypatch.setattr(module, "get_user_token", lambda sid: fake.token if sid == "sess" else None)
    yield fake
    asyncio.run(_clear_registry())


def _call(method: str, url: str, **kwargs) -> httpx.Response:
    async def run():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, url, timeout=60, **kwargs)
    return asyncio.run(run())


def _envelope(class_model=CLASS_DIAGRAM_MODEL) -> dict:
    return {
        "project": {
            "id": "proj-1", "type": "BesserProject", "name": "Library", "createdAt": "2026-04-15T00:00:00Z",
            "diagrams": {"ClassDiagram": [{"id": "cd1", "title": "Library", "model": class_model}]},
            "currentDiagramIndices": {"ClassDiagram": 0},
        },
        "exportedAt": "2026-10-26T00:00:00Z", "version": "2.0.0",
    }


def _import(owner: str, repo: str) -> dict:
    r = _call("POST", "/besser_api/spec-driven/import-github-run", json={"owner": owner, "repo": repo},
              headers=SESSION)
    assert r.status_code == 200, r.text
    return r.json()


def _modify_run(base_run_id: str, edit) -> str:
    """A finished modify run seeded from ``base_run_id`` (as the runner seeds it), after ``edit(workdir)``."""
    base = asyncio.run(SMART_RUN_REGISTRY.get(base_run_id))
    run_id = uuid.uuid4().hex
    workdir = create_run_workspace(run_id)
    _seed_workspace_from_base(base.temp_dir, workdir)
    edit(workdir)
    marker = os.path.join(workdir, "README.md")
    asyncio.run(SMART_RUN_REGISTRY.put(run_id, SmartRunEntry(
        file_path=marker, file_name="README.md", is_zip=False, temp_dir=workdir, created_at=time.time(),
    )))
    return run_id


def _push(run_id: str, repo: str, **deploy) -> httpx.Response:
    body = {"run_id": run_id, "projectExport": _envelope(),
            "deploy_config": {"repo_name": repo, "use_existing": True, **deploy}}
    return _call("POST", "/besser_api/spec-driven/push-to-github", json=body, headers=SESSION)


def _write(workdir: str, path: str, text: str) -> None:
    target = os.path.join(workdir, *path.split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as fh:
        fh.write(text.encode("utf-8"))


# ---------------------------------------------------------------------
# A Spec-Driven push changes only what the run changed
# ---------------------------------------------------------------------


IMPORTED_REPO = {
    "README.md": "# shop\n",
    "src/app.py": "print('v1')\n",
    "src/old.py": "OLD = 1\n",
    "scripts/run.sh": ("#!/bin/sh\necho run\n", "100755"),
    "current": ("src/app.py", "120000"),
    "build/bundle.js": "console.log('built');\n",
    "dist/app.tar": "binary-ish\n",
    "data/shop.db": "SQLite format 3\0",
    "frontend/node_modules/left-pad/index.js": "module.exports = 1;\n",
}


def test_a_modify_push_keeps_the_files_the_run_never_saw(github):
    """Pushing a run made from an imported repo replaced the whole tree.

    The import skips symlinks, and the run seed and the push skip build output
    and ``*.db``, so the replaced tree lost all of them, and every file came back
    as ``100644``, dropping executable bits.
    """
    repo = github.create_repo("shop", files=IMPORTED_REPO)
    imported = _import("ana", "shop")

    def edit(workdir):
        _write(workdir, "src/app.py", "print('v2')\n")
        _write(workdir, "src/new.py", "NEW = 2\n")
        os.remove(os.path.join(workdir, "src", "old.py"))

    r = _push(_modify_run(imported["run_id"], edit), "shop", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    files = github.files(repo)
    # What the run changed:
    assert files["src/app.py"][1] == b"print('v2')\n"
    assert files["src/new.py"][1] == b"NEW = 2\n"
    assert "src/old.py" not in files
    # What it never saw, kept as it was:
    assert files["scripts/run.sh"][0] == "100755"
    assert files["current"] == ("120000", b"src/app.py")
    for kept in ("build/bundle.js", "dist/app.tar", "data/shop.db", "frontend/node_modules/left-pad/index.js"):
        assert files[kept][1] == IMPORTED_REPO[kept].encode()
    # The model rides along.
    assert repo_model.MODEL_FILE in files


def test_a_push_sends_only_the_files_that_changed(github):
    github.create_repo("shop", files=IMPORTED_REPO)
    imported = _import("ana", "shop")
    run_id = _modify_run(imported["run_id"], lambda workdir: _write(workdir, "src/app.py", "print('v2')\n"))

    r = _push(run_id, "shop", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    # src/app.py and the three model files the push adds; nothing else of the repo.
    assert r.json()["files_uploaded"] == 4


def test_a_push_with_nothing_changed_makes_no_commit(github):
    repo = github.create_repo("shop", files=IMPORTED_REPO)
    imported = _import("ana", "shop")
    first_run = _modify_run(imported["run_id"], lambda workdir: None)
    first = _push(first_run, "shop", base_commit_sha=imported["commit_sha"])
    assert first.status_code == 200, first.text
    head = github.ref(repo, "main")

    again = _push(_modify_run(first_run, lambda workdir: None), "shop", base_commit_sha=head)

    assert again.status_code == 200, again.text
    assert github.ref(repo, "main") == head
    assert again.json()["commit_sha"] == head
    assert again.json()["files_uploaded"] == 0


# ---------------------------------------------------------------------
# Repositories the user does not own
# ---------------------------------------------------------------------


def test_a_run_from_an_organization_repo_pushes_back_to_it(github):
    """The push resolved the owner as the signed-in user, so an org repo's run went to ana/<repo>."""
    repo = github.create_repo("crm", owner="acme", files={"src/app.py": "print('v1')\n"})
    imported = _import("acme", "crm")
    run_id = _modify_run(imported["run_id"], lambda workdir: _write(workdir, "src/app.py", "print('v2')\n"))

    r = _push(run_id, "crm", owner="acme", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    assert r.json()["owner"] == "acme"
    assert github.files(repo)["src/app.py"][1] == b"print('v2')\n"
    assert "ana/crm" not in github.repos


def test_the_repo_list_holds_every_page_and_other_owners_repos(github):
    """The list asked for the owner's repos only, and only the first 100."""
    for index in range(130):
        github.create_repo(f"repo-{index:03d}")
    github.create_repo("crm", owner="acme", can_push=False)

    r = _call("GET", "/besser_api/github/repos", headers=SESSION)

    assert r.status_code == 200, r.text
    listing = {repo["full_name"]: repo for repo in r.json()["repositories"]}
    assert len(listing) == 131
    assert listing["acme/crm"]["can_push"] is False
    assert listing["ana/repo-129"]["can_push"] is True
    assert r.json()["truncated"] is False


def test_a_branch_past_the_first_page_is_pushed_to(github):
    """Branch checks read one page of 100, so a later branch silently fell back to the default."""
    repo = github.create_repo("shop", files={"src/app.py": "print('v1')\n"})
    head = github.ref(repo, "main")
    for index in range(120):
        github.set_ref(repo, f"feature/b{index:03d}", head)
    imported = _import("ana", "shop")
    run_id = _modify_run(imported["run_id"], lambda workdir: _write(workdir, "src/app.py", "print('v2')\n"))

    r = _push(run_id, "shop", branch="feature/b119", base_commit_sha=head)

    assert r.status_code == 200, r.text
    assert r.json()["branch"] == "feature/b119"
    assert github.files(repo, "feature/b119")["src/app.py"][1] == b"print('v2')\n"
    assert github.files(repo)["src/app.py"][1] == b"print('v1')\n"


# ---------------------------------------------------------------------
# Saving the model while someone else changed it
# ---------------------------------------------------------------------


def _with_class(model: dict, name: str, attribute: str = "id: int") -> dict:
    """``model`` plus class ``name``, as the editor adds one."""
    model = json.loads(json.dumps(model))
    class_id, attr_id = f"class-{name.lower()}", f"attr-{name.lower()}"
    model["elements"][class_id] = {"type": "Class", "name": name, "attributes": [attr_id], "methods": []}
    model["elements"][attr_id] = {"type": "Attribute", "name": f"+ {attribute}"}
    return model


def _model_repo(github, model=CLASS_DIAGRAM_MODEL):
    """A BESSER repo holding ``model``, as "Create a repository" leaves it."""
    files = {path: content for path, content in repo_model.model_files(_envelope(model)).items()}
    repo = github.create_repo("library", files=files, topics=["besser"])
    return repo, github.ref(repo, "main")


def _teammate_saves(github, repo, model) -> str:
    """Another BESSER user saves ``model`` (every model file, as the editor writes them)."""
    files = repo_model.model_files(_envelope(model))
    return github.commit_files(repo, files, repo_model.with_origin("Add a class", repo_model.ORIGIN_EDITOR),
                               author="ben")


def _save(model, base_sha: str) -> httpx.Response:
    envelope = _envelope(model)
    return _call("POST", "/besser_api/github/project/save", headers=SESSION, json={
        "owner": "ana", "repo": "library", "commit_message": "Add Loan",
        "project_data": envelope["project"], "project_export": envelope, "base_commit_sha": base_sha,
    })


def _saved_classes(github, repo) -> set[str]:
    saved = json.loads(github.files(repo)[repo_model.MODEL_FILE][1])
    elements = saved["project"]["diagrams"]["ClassDiagram"][0]["model"]["elements"]
    return {e["name"] for e in elements.values() if e["type"] == "Class"}


def test_two_people_adding_different_classes_both_keep_theirs(github):
    """Both adds touch the same lines of diagrams.json and of the B-UML sources, so a text merge conflicts."""
    repo, base = _model_repo(github)
    _teammate_saves(github, repo, _with_class(CLASS_DIAGRAM_MODEL, "Member"))

    r = _save(_with_class(CLASS_DIAGRAM_MODEL, "Loan"), base)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged", r.json()
    assert _saved_classes(github, repo) == {"Author", "Book", "Member", "Loan"}
    # The B-UML source is rebuilt from the merged model, not text-merged.
    domain = github.files(repo)["buml/domain_model.py"][1].decode()
    assert "Member" in domain and "Loan" in domain


def _renamed(model: dict, element_id: str, name: str) -> dict:
    model = json.loads(json.dumps(model))
    model["elements"][element_id]["name"] = name
    return model


def test_two_people_renaming_the_same_class_is_a_conflict_kept_for_a_pull_request(github):
    repo, base = _model_repo(github)
    theirs = _teammate_saves(github, repo, _renamed(CLASS_DIAGRAM_MODEL, "class-1", "Writer"))

    r = _save(_renamed(CLASS_DIAGRAM_MODEL, "class-1", "Creator"), base)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "conflict"
    assert github.ref(repo, "main") == theirs  # nothing landed on the branch
    side = r.json()["conflict_branch"]
    on_side = json.loads(github.files(repo, side)[repo_model.MODEL_FILE][1])
    assert on_side["project"]["diagrams"]["ClassDiagram"][0]["model"]["elements"]["class-1"]["name"] == "Creator"
    assert not any(b.startswith("besser/sync-merge-") or b.startswith("besser/sync-code-")
                   for b in github.out(repo, "for-each-ref", "--format=%(refname:lstrip=2)").split())


def test_an_agent_push_merges_a_teammates_code_and_model_changes(github):
    """The run changed code and the model; meanwhile a teammate changed other code and added a class."""
    files = {**repo_model.model_files(_envelope()), "src/app.py": "print('v1')\n", "src/util.py": "X = 1\n"}
    repo = github.create_repo("library", files=files)
    imported = _import("ana", "library")
    _teammate_saves(github, repo, _with_class(CLASS_DIAGRAM_MODEL, "Member"))
    github.commit_files(repo, {"src/util.py": "X = 2\n"}, "Tweak util", author="ben")

    def edit(workdir):
        _write(workdir, "src/app.py", "print('v2')\n")

    run_id = _modify_run(imported["run_id"], edit)
    body = {"run_id": run_id, "projectExport": _envelope(_with_class(CLASS_DIAGRAM_MODEL, "Loan")),
            "deploy_config": {"repo_name": "library", "use_existing": True,
                              "base_commit_sha": imported["commit_sha"]}}
    r = _call("POST", "/besser_api/spec-driven/push-to-github", json=body, headers=SESSION)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged"
    after = github.files(repo)
    assert after["src/app.py"][1] == b"print('v2')\n"
    assert after["src/util.py"][1] == b"X = 2\n"
    assert _saved_classes(github, repo) == {"Author", "Book", "Member", "Loan"}
    # The next modify run starts from the merged repo.
    assert r.json()["base_run_id"]
    # The scratch and code branches of the merge are gone.
    assert github.out(repo, "for-each-ref", "--format=%(refname:lstrip=2)").split() == ["main"]


def test_the_same_code_lines_changed_on_both_sides_is_a_conflict(github):
    files = {**repo_model.model_files(_envelope()), "src/app.py": "print('v1')\n"}
    repo = github.create_repo("library", files=files)
    imported = _import("ana", "library")
    theirs = github.commit_files(repo, {"src/app.py": "print('theirs')\n"}, "Their change", author="ben")

    run_id = _modify_run(imported["run_id"], lambda workdir: _write(workdir, "src/app.py", "print('ours')\n"))
    r = _push(run_id, "library", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "conflict"
    assert github.ref(repo, "main") == theirs
    assert github.files(repo, r.json()["conflict_branch"])["src/app.py"][1] == b"print('ours')\n"


# ---------------------------------------------------------------------
# A change kept for a pull request
# ---------------------------------------------------------------------


def _conflicting_save(github):
    repo, base = _model_repo(github)
    _teammate_saves(github, repo, _renamed(CLASS_DIAGRAM_MODEL, "class-1", "Writer"))
    r = _save(_renamed(CLASS_DIAGRAM_MODEL, "class-1", "Creator"), base)
    assert r.json()["status"] == "conflict", r.text
    return repo, r.json()["conflict_branch"]


def _sync_status(pending_branch: str) -> dict:
    r = _call("GET", "/besser_api/github/sync-status", headers=SESSION,
              params={"owner": "ana", "repo": "library", "branch": "main", "pending_branch": pending_branch})
    assert r.status_code == 200, r.text
    return r.json()["pending"]


def test_a_conflict_pull_request_says_what_it_changes_and_is_opened_once(github):
    repo, branch = _conflicting_save(github)
    body = {"owner": "ana", "repo": "library", "head": branch, "base": "main", "title": "Rename Author"}

    first = _call("POST", "/besser_api/github/pull-request", headers=SESSION, json=body)
    again = _call("POST", "/besser_api/github/pull-request", headers=SESSION, json=body)

    assert first.status_code == 200, first.text
    assert again.status_code == 200, again.text
    assert again.json()["number"] == first.json()["number"]
    assert len(repo.pulls) == 1
    assert "- class Writer renamed to Creator" in repo.pulls[0]["body"]


def test_the_sync_status_follows_a_pending_branch_until_its_pull_request_is_merged(github):
    repo, branch = _conflicting_save(github)
    assert _sync_status(branch) == {"branch": branch, "branch_exists": True, "pull_request_number": None,
                                    "pull_request_url": None, "pull_request_state": None}

    number = _call("POST", "/besser_api/github/pull-request", headers=SESSION, json={
        "owner": "ana", "repo": "library", "head": branch, "base": "main", "title": "Rename"}).json()["number"]
    assert _sync_status(branch)["pull_request_state"] == "open"

    # The person resolves it on GitHub: keep theirs, then merge.
    github.commit_files(repo, repo_model.model_files(_envelope(_renamed(CLASS_DIAGRAM_MODEL, "class-1", "Writer"))),
                        "Resolve", branch=branch)
    github.merge_pull(repo, number)

    assert _sync_status(branch)["pull_request_state"] == "merged"


# ---------------------------------------------------------------------
# Signing in and out
# ---------------------------------------------------------------------


def test_sign_in_then_out_revokes_the_token(github, monkeypatch):
    from besser.utilities.web_modeling_editor.backend.services.deployment import github_oauth

    monkeypatch.setattr(github_oauth, "GITHUB_CLIENT_ID", "client")
    monkeypatch.setattr(github_oauth, "GITHUB_CLIENT_SECRET", "secret")
    monkeypatch.setattr(github_oauth, "DEPLOYMENT_URL", "http://editor")
    for module in (spec_driven_router, github_deploy_api):
        monkeypatch.setattr(module, "get_user_token", github_oauth.get_user_token)
    state = "s" * 32
    github_oauth._oauth_sessions.set(state, {"client_ip": "test"})

    callback = _call("GET", "/besser_api/github/auth/callback", params={"code": "c", "state": state})
    assert callback.status_code == 302, callback.text
    fragment = dict(part.split("=", 1) for part in callback.headers["location"].split("#", 1)[1].split("&"))
    session = {"X-GitHub-Session": fragment["github_session"]}
    assert fragment["username"] == "ana"
    assert _call("GET", "/besser_api/github/repos", headers=session).status_code == 200

    out = _call("POST", "/besser_api/github/auth/logout", json={"session_id": fragment["github_session"]})

    assert out.status_code == 200, out.text
    assert github.revoked == [github.token]
    assert _call("GET", "/besser_api/github/repos", headers=session).status_code == 401


def test_a_merged_push_is_reported_as_done_even_if_reading_it_back_fails(github, monkeypatch):
    """The push had landed, then reading the merged repo back failed: the user saw a failed push."""
    files = {**repo_model.model_files(_envelope()), "src/app.py": "print('v1')\n"}
    repo = github.create_repo("library", files=files)
    imported = _import("ana", "library")
    github.commit_files(repo, {"src/util.py": "X = 2\n"}, "Their change", author="ben")
    run_id = _modify_run(imported["run_id"], lambda workdir: _write(workdir, "src/app.py", "print('v2')\n"))

    async def unreadable(*args, **kwargs):
        raise ValueError("Repository archive exceeds the 100 MB size cap")
    monkeypatch.setattr(spec_driven_router, "_import_repo_as_run", unreadable)
    r = _push(run_id, "library", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged"
    assert r.json()["base_run_id"] is None
    assert any("Get latest" in warning for warning in r.json()["warnings"])
    assert github.files(repo)["src/app.py"][1] == b"print('v2')\n"
