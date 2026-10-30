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
    # src/app.py, the three model files the push adds and the README's BESSER block
    # (this repo had none); nothing else of the repo.
    assert r.json()["files_uploaded"] == 5


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
    attr_name, attr_type = (part.strip() for part in attribute.split(":"))
    model["nodes"].append({
        "id": f"class-{name.lower()}", "type": "class", "position": {"x": 0, "y": 200}, "width": 160, "height": 100,
        "data": {"name": name, "methods": [], "attributes": [{
            "id": f"attr-{name.lower()}", "name": attr_name, "attributeType": attr_type, "visibility": "public",
        }]},
    })
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
    nodes = saved["project"]["diagrams"]["ClassDiagram"][0]["model"]["nodes"]
    return {n["data"]["name"] for n in nodes if n["type"] == "class"}


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
    next(n for n in model["nodes"] if n["id"] == element_id)["data"]["name"] = name
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
    side_nodes = on_side["project"]["diagrams"]["ClassDiagram"][0]["model"]["nodes"]
    assert next(n for n in side_nodes if n["id"] == "class-1")["data"]["name"] == "Creator"
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


def test_the_model_history_describes_each_commit_against_its_parent(github):
    """History compared each commit with the one listed next by date, so after a merge
    "Add Loan" read "class Member removed" (Member came from the other branch)."""
    repo, base = _model_repo(github)
    _teammate_saves(github, repo, _with_class(CLASS_DIAGRAM_MODEL, "Member"))
    assert _save(_with_class(CLASS_DIAGRAM_MODEL, "Loan"), base).json()["status"] == "merged"

    r = _call("GET", "/besser_api/github/model-history", headers=SESSION,
              params={"owner": "ana", "repo": "library", "branch": "main"})

    assert r.status_code == 200, r.text
    changes = {c["message"].splitlines()[0]: c["changes"] for c in r.json()["commits"]}
    assert changes["Add Loan"] == ["class Loan added"]
    assert changes["Add a class"] == ["class Member added"]
    # The merge brings Loan into the branch that had Member.
    assert changes["Merge BESSER changes into main"] == ["class Loan added"]


def test_a_fresh_run_pushed_into_a_repo_deletes_nothing_it_never_saw(github, tmp_path):
    """Found on experimental: a first generation pushed into the project's repo deleted its
    README, because the run was generated from the model, not seeded from the repo."""
    repo = github.create_repo("library", files={**repo_model.model_files(_envelope()), "README.md": "# Library\n"})
    run_id = uuid.uuid4().hex
    workdir = create_run_workspace(run_id)
    _write(workdir, "backend/main_api.py", "app = None\n")
    asyncio.run(SMART_RUN_REGISTRY.put(run_id, SmartRunEntry(
        file_path=os.path.join(workdir, "backend", "main_api.py"), file_name="app.zip", is_zip=True,
        temp_dir=workdir, created_at=time.time(),
    )))

    r = _push(run_id, "library", base_commit_sha=github.ref(repo, "main"))

    assert r.status_code == 200, r.text
    files = github.files(repo)
    assert files["README.md"][1] == b"# Library\n"
    assert files["backend/main_api.py"][1] == b"app = None\n"


# ---------------------------------------------------------------------
# What a push must never touch or leak
# ---------------------------------------------------------------------


def test_a_push_keeps_paths_the_import_did_not_extract(github):
    """An import is a git archive: export-ignore paths are not in it, so a push read them as deleted."""
    repo = github.create_repo("shop", files={
        ".gitattributes": "tests/ export-ignore\n",
        "src/app.py": "print('v1')\n",
        "tests/test_app.py": "def test_app(): pass\n",
    })
    imported = _import("ana", "shop")
    run_id = _modify_run(imported["run_id"], lambda workdir: _write(workdir, "src/app.py", "print('v2')\n"))

    r = _push(run_id, "shop", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    files = github.files(repo)
    assert files["src/app.py"][1] == b"print('v2')\n"
    assert files["tests/test_app.py"][1] == b"def test_app(): pass\n"


def test_a_link_planted_at_the_runs_github_record_is_replaced_not_written_through(github, tmp_path):
    """The backend updates the run's GitHub record after a push; the run's sandboxed code can make it a link."""
    victim = tmp_path / "backend_file.py"
    victim.write_text("ORIGINAL = True\n", encoding="utf-8")
    github.create_repo("shop", files={"src/app.py": "print('v1')\n"})
    imported = _import("ana", "shop")

    def plant(workdir):
        _write(workdir, "src/app.py", "print('v2')\n")
        record = os.path.join(workdir, runner_module.GITHUB_SOURCE_FILENAME)
        os.remove(record)
        try:
            os.symlink(victim, record)
        except (OSError, NotImplementedError):
            pytest.skip("this platform cannot create symlinks")

    run_id = _modify_run(imported["run_id"], plant)
    r = _push(run_id, "shop", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    assert victim.read_text(encoding="utf-8") == "ORIGINAL = True\n"


def test_a_shared_gist_carries_no_agent_credentials(github):
    """Repo saves redacted the agent's keys; sharing the project as a (possibly public) gist did not."""
    key = "sk-proj-" + "a1B2c3D4" * 4
    project = _envelope()["project"]
    project["diagrams"]["AgentDiagram"] = [{
        "id": "ag1", "title": "Helper", "model": {},
        "agentConfigForm": {"nlp": {"openai_api_key": key, "language": "en"}},
        "configYaml": f"nlp:\n  openai:\n    api_key: {key}\n",
    }]

    r = _call("POST", "/besser_api/github/gist/create", headers=SESSION,
              json={"project_data": project, "description": "Library", "is_public": True})

    assert r.status_code == 200, r.text
    (gist,) = github.gists.values()
    (content,) = [f["content"] for f in gist["files"].values()]
    assert key not in content
    assert '"language": "en"' in content


def test_the_fake_refuses_a_repo_name_that_leaves_its_root(tmp_path):
    """CodeQL py/path-injection: a created repo's name was joined into a path unchecked."""
    fake = FakeGitHub(root=str(tmp_path / "github"))

    async def create(name):
        transport = ASGITransport(app=fake.app)
        async with httpx.AsyncClient(transport=transport, base_url="https://api.github.com") as client:
            return await client.post("/user/repos", json={"name": name},
                                     headers={"authorization": f"Bearer {fake.token}"})

    for name in ("../escape", "..", "a/b"):
        assert asyncio.run(create(name)).status_code == 422
    assert not (tmp_path / "escape.git").exists()
    assert asyncio.run(create("shop")).status_code == 201


# ---------------------------------------------------------------------
# Review of #693
# ---------------------------------------------------------------------


GITATTRIBUTES_REPO = {
    ".gitattributes": "*.bat text eol=crlf\nversion.txt export-subst\nid.txt ident\n",
    "run.bat": "@echo off\necho hi\n",
    "version.txt": "$Format:%H$\n",
    "id.txt": "$Id$\n",
    "src/app.py": "print('v1')\n",
}


def test_a_push_leaves_alone_the_files_gitattributes_rewrote_on_import(github):
    """The import is a git archive, which applies ``eol``, ``export-subst`` and ``ident``: the run's
    bytes differed from the blobs, so every push rewrote files the run never touched (run.bat came
    back CRLF, ``$Format:%H$`` became a literal commit id)."""
    repo = github.create_repo("win", files=GITATTRIBUTES_REPO)
    before = github.files(repo)
    imported = _import("ana", "win")
    first_run = _modify_run(imported["run_id"], lambda w: _write(w, "src/app.py", "print('v2')\n"))

    r = _push(first_run, "win", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    after = github.files(repo)
    assert after["src/app.py"][1] == b"print('v2')\n"
    for untouched in ("run.bat", "version.txt", "id.txt"):
        assert after[untouched] == before[untouched], untouched

    # The record follows the pushed commit: the next push from this run leaves them alone too.
    again = _push(_modify_run(first_run, lambda w: _write(w, "src/app.py", "print('v3')\n")), "win",
                  base_commit_sha=github.ref(repo, "main"))

    assert again.status_code == 200, again.text
    latest = github.files(repo)
    assert latest["src/app.py"][1] == b"print('v3')\n"
    for untouched in ("run.bat", "version.txt", "id.txt"):
        assert latest[untouched] == before[untouched], untouched


def test_a_run_record_without_hashes_is_still_read(github, tmp_path):
    """Records written before the hashes held a list of paths."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / runner_module.GITHUB_SOURCE_FILENAME).write_text(json.dumps(
        {"owner": "ana", "repo": "shop", "commitSha": "a" * 40, "paths": ["src/app.py"]}), encoding="utf-8")

    assert spec_driven_router._run_seed(str(run_dir), "ana", "shop") == ("a" * 40, {"src/app.py": None})


def test_a_teammates_buml_helper_survives_a_model_merge(github):
    """Every ``buml/*.py`` counted as a model file, so a merged save deleted a hand-written one."""
    repo, base = _model_repo(github)
    _teammate_saves(github, repo, _with_class(CLASS_DIAGRAM_MODEL, "Member"))
    github.commit_files(repo, {"buml/seed_data.py": "ROWS = [1, 2, 3]\n"}, "Add seed data", author="ben")

    r = _save(_with_class(CLASS_DIAGRAM_MODEL, "Loan"), base)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged"
    assert github.files(repo)["buml/seed_data.py"][1] == b"ROWS = [1, 2, 3]\n"


def test_a_failed_buml_export_keeps_the_generated_sources(github, monkeypatch):
    """The B-UML export failure was swallowed, so the merge deleted ``domain_model.py`` as no longer produced."""
    repo, base = _model_repo(github)
    _teammate_saves(github, repo, _with_class(CLASS_DIAGRAM_MODEL, "Member"))

    def broken(_export):
        raise RuntimeError("export failed")
    monkeypatch.setattr(repo_model, "_buml_sources", broken)
    r = _save(_with_class(CLASS_DIAGRAM_MODEL, "Loan"), base)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged"
    assert "buml/domain_model.py" in github.files(repo)
    assert any("buml/domain_model.py" in warning for warning in r.json()["warnings"])
    assert _saved_classes(github, repo) == {"Author", "Book", "Member", "Loan"}


def test_deploying_an_agent_commits_no_agent_credentials(github, monkeypatch):
    """/deploy-webapp wrote the raw ``projectExport`` to ``buml/diagrams.json``, agent API keys included."""
    from types import SimpleNamespace

    key = "sk-proj-" + "a1B2c3D4" * 4

    class _Generator:
        def __init__(self, agent_model, output_dir, **_kwargs):
            self.output_dir = output_dir

        def generate(self):
            _write(self.output_dir, "agent.py", "# agent\n")

    monkeypatch.setattr(github_deploy_api, "get_generator_info", lambda _name: SimpleNamespace(generator_class=_Generator))
    agent = {
        "id": "ag1", "title": "Helper", "model": {
            "version": "4.0.0", "type": "AgentDiagram", "edges": [], "components": {},
            "nodes": [{"id": "s1", "type": "AgentState", "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                       "data": {"name": "Greet", "initial": True, "bodies": [], "fallbackBodies": []}}],
        },
        "configYaml": f"nlp:\n  openai:\n    api_key: {key}\n",
    }
    export = _envelope()
    export["project"]["diagrams"]["AgentDiagram"] = [agent]

    r = _call("POST", "/besser_api/github/deploy-webapp", headers=SESSION, json={
        "name": "Helper", "diagrams": {"AgentDiagram": [agent]}, "currentDiagramIndices": {"AgentDiagram": 0},
        "deploy_config": {"target": "agent", "repo_name": "helper"}, "projectExport": export,
    })

    assert r.status_code == 200, r.text
    files = github.files(github.repos["ana/helper"])
    model = files[repo_model.MODEL_FILE][1].decode()
    assert key not in model
    assert json.loads(model)["project"]["diagrams"]["AgentDiagram"][0]["title"] == "Helper"


def test_an_import_takes_a_run_slot_and_gives_it_back(github, monkeypatch):
    """An import downloads up to 100 MB, unpacks up to 500 MB and may call an LLM, with no cap at all."""
    github.create_repo("shop", files={"src/app.py": "print('v1')\n"})
    slots = {"taken": 0, "returned": 0}

    def take():
        slots["taken"] += 1
        return True

    def give_back():
        slots["returned"] += 1

    monkeypatch.setattr(spec_driven_router, "try_acquire_run_slot", take)
    monkeypatch.setattr(spec_driven_router, "release_run_slot", give_back)

    assert _import("ana", "shop")["run_id"]
    missing = _call("POST", "/besser_api/spec-driven/import-github-run", json={"owner": "ana", "repo": "nope"},
                    headers=SESSION)

    assert missing.status_code == 404
    assert slots == {"taken": 2, "returned": 2}


def test_an_import_is_refused_while_every_run_slot_is_taken(github, monkeypatch):
    github.create_repo("shop", files={"src/app.py": "print('v1')\n"})
    monkeypatch.setattr(spec_driven_router, "try_acquire_run_slot", lambda: False)

    def no_download(*_args, **_kwargs):
        raise AssertionError("the repo was downloaded")
    monkeypatch.setattr(spec_driven_router, "_import_repo_as_run", no_download)

    r = _call("POST", "/besser_api/spec-driven/import-github-run", json={"owner": "ana", "repo": "shop"},
              headers=SESSION)

    assert r.status_code == 429, r.text


def test_saving_into_an_empty_repository_makes_its_first_commit(github):
    """GitHub's git data API answers 409 "Git Repository is empty" until a repository has a commit."""
    repo = github.create_repo("library")

    r = _save(CLASS_DIAGRAM_MODEL, None)

    assert r.status_code == 200, r.text
    files = github.files(repo)
    assert {repo_model.MODEL_FILE, repo_model.LINK_FILE, "buml/domain_model.py"} <= set(files)
    assert r.json()["commit_sha"] == github.ref(repo, "main")


def test_pushing_a_run_into_an_empty_repository_makes_its_first_commit(github):
    repo = github.create_repo("library")
    run_id = uuid.uuid4().hex
    workdir = create_run_workspace(run_id)
    _write(workdir, "backend/main_api.py", "app = None\n")
    asyncio.run(SMART_RUN_REGISTRY.put(run_id, SmartRunEntry(
        file_path=os.path.join(workdir, "backend", "main_api.py"), file_name="app.zip", is_zip=True,
        temp_dir=workdir, created_at=time.time(),
    )))

    r = _push(run_id, "library")

    assert r.status_code == 200, r.text
    files = github.files(repo)
    assert files["backend/main_api.py"][1] == b"app = None\n"
    assert repo_model.MODEL_FILE in files


def _teammate_pushes_before_our_ref_update(github, repo):
    """Land ben's commit on main once, just before BESSER's first update of main is checked."""
    def land(target, branch):
        if target is repo and branch == "main" and github.before_ref_update is not None:
            github.before_ref_update = None
            github.commit_files(repo, {"src/util.py": "X = 2\n"}, "Their change", author="ben")
    github.before_ref_update = land


def test_a_save_racing_a_teammates_push_is_merged_not_refused(github):
    """The ref update was a fast-forward of the head read moments before: a push landing in between made
    GitHub answer 422 "Update is not a fast forward" and the save failed."""
    repo, base = _model_repo(github)
    _teammate_pushes_before_our_ref_update(github, repo)

    r = _save(_with_class(CLASS_DIAGRAM_MODEL, "Loan"), base)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged"
    files = github.files(repo)
    assert files["src/util.py"][1] == b"X = 2\n"
    assert _saved_classes(github, repo) == {"Author", "Book", "Loan"}


def test_a_push_racing_a_teammates_push_is_merged_not_refused(github):
    """The user saw a 409 for a push GitHub could have merged."""
    repo = github.create_repo("shop", files={"src/app.py": "print('v1')\n"})
    imported = _import("ana", "shop")
    run_id = _modify_run(imported["run_id"], lambda w: _write(w, "src/app.py", "print('v2')\n"))
    _teammate_pushes_before_our_ref_update(github, repo)

    r = _push(run_id, "shop", base_commit_sha=imported["commit_sha"])

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "merged"
    files = github.files(repo)
    assert files["src/app.py"][1] == b"print('v2')\n"
    assert files["src/util.py"][1] == b"X = 2\n"


def _save_with_legacy(base_sha: str, legacy_file_path) -> httpx.Response:
    envelope = _envelope()
    return _call("POST", "/besser_api/github/project/save", headers=SESSION, json={
        "owner": "ana", "repo": "library", "commit_message": "Save", "project_data": envelope["project"],
        "project_export": envelope, "base_commit_sha": base_sha, "legacy_file_path": legacy_file_path,
    })


def test_a_save_removes_the_project_file_the_old_sidebar_saved(github):
    """The old sidebar saved the model at ``<dir>/<project>.json``; after a save the repo held two models."""
    repo = github.create_repo("library", files={
        "models/Library.json": json.dumps(_envelope()["project"]),
        "config/settings.json": json.dumps({"theme": "dark"}),
    })
    head = github.ref(repo, "main")

    r = _save_with_legacy(head, "models/Library.json")

    assert r.status_code == 200, r.text
    files = github.files(repo)
    assert "models/Library.json" not in files
    assert repo_model.MODEL_FILE in files

    # A JSON file that is not a BESSER project is never removed.
    kept = _save_with_legacy(github.ref(repo, "main"), "config/settings.json")
    assert kept.status_code == 200, kept.text
    assert "config/settings.json" in github.files(repo)


def test_the_legacy_project_path_must_stay_inside_the_repo(github):
    repo, base = _model_repo(github)
    for bad in ("../outside.json", "/etc/passwd.json", "models/notes.txt", "models//Library.json"):
        r = _save_with_legacy(base, bad)
        assert r.status_code == 422, (bad, r.text)
    assert github.ref(repo, "main") == base
