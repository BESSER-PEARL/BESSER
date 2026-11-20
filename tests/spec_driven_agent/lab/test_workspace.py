"""LabWorkspace against real git repositories: settings, readiness, state files and the git operations of one change.

Several changes at once are in ``test_workspace_changes.py``; git directories outside the trees in
``test_workspace_git_dir.py``. The layout comes from the ``lab_layout`` fixture (``conftest.py``).
"""

import json
import os
import shutil
from dataclasses import fields

import pytest

from besser.spec_driven_agent.lab.contracts import (
    DEFAULT_UPSTREAMS,
    FRONTEND_SUBMODULE_PATH,
    LAB_BASE_DIRNAME,
    LAB_STATE_VERSION,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    ChangeKind,
    FileChange,
    LabChange,
    RepoBase,
)
from besser.spec_driven_agent.lab.workspace import (
    LabWorkspace,
    LabWorkspaceError,
    _slug,
    parse_upstream,
    repo_specs_from_settings,
    upstream_url,
)
from tests.spec_driven_agent.lab.conftest import SUBMODULE_PARTS, git, read, write


def open_new(ws, owner="Jordi", title="Note element", kind=ChangeKind.EDITOR_ELEMENT):
    return ws.open_change(ws.create_draft(owner=owner), title=title, kind=kind)


def tree(ws, change, repo, *parts):
    return os.path.join(ws.change_repos(change)[repo].path, *parts)


def state_file(layout, change_id):
    return os.path.join(layout.state_dir, "changes", change_id, "state.json")


# -- settings ------------------------------------------------------------------------------------------

def test_parse_upstream_splits_owner_name_and_branch():
    assert parse_upstream("BESSER-PEARL/BESSER@development") == ("BESSER-PEARL", "BESSER", "development")
    assert parse_upstream("me/BESSER-Web-Modeling-Editor@feature/x.y") == (
        "me", "BESSER-Web-Modeling-Editor", "feature/x.y")


@pytest.mark.parametrize("value", ["BESSER", "BESSER-PEARL/BESSER", "a/b@", "a b/c@d", "a/b@c d", "a/b/c@d", ""])
def test_parse_upstream_rejects_anything_else(value):
    with pytest.raises(ValueError, match="owner/name@branch"):
        parse_upstream(value)


def test_repo_specs_from_settings_places_base_trees_under_base_and_git_dirs_in_the_git_root(tmp_path):
    root, git_root = str(tmp_path / "work"), str(tmp_path / "git")

    specs = repo_specs_from_settings(root, git_root, {**DEFAULT_UPSTREAMS, REPO_BESSER: "fork/BESSER@lab-test"})

    assert tuple(spec.key for spec in specs) == REPO_ORDER
    by_key = {spec.key: spec for spec in specs}
    base = os.path.join(root, LAB_BASE_DIRNAME)
    assert by_key[REPO_BESSER].path == os.path.join(base, "besser")
    assert by_key[REPO_FRONTEND].path == os.path.join(base, "besser", *SUBMODULE_PARTS)
    assert by_key[REPO_MODELING_AGENT].path == os.path.join(base, "modeling-agent")
    assert by_key[REPO_BESSER].git_dir == os.path.join(git_root, "besser")
    assert by_key[REPO_FRONTEND].git_dir == os.path.join(git_root, "besser", "modules", *SUBMODULE_PARTS)
    assert by_key[REPO_MODELING_AGENT].git_dir == os.path.join(git_root, "modeling-agent")
    assert [(s.base_branch, s.github_owner, s.github_name) for s in specs] == [
        ("develop", "BESSER-PEARL", "BESSER-Web-Modeling-Editor"),
        ("lab-test", "fork", "BESSER"),
        ("develop", "BESSER-PEARL", "modeling-agent"),
    ]


def test_upstream_url_is_the_public_github_repository(tmp_path):
    spec = repo_specs_from_settings(str(tmp_path), str(tmp_path / "git"), DEFAULT_UPSTREAMS)[1]
    assert upstream_url(spec) == "https://github.com/BESSER-PEARL/BESSER.git"


# -- readiness -----------------------------------------------------------------------------------------

def test_ensure_ready_accepts_the_layout_of_the_setup_script(lab_layout):
    lab_layout.workspace().ensure_ready()


def test_ensure_ready_refuses_a_missing_base_clone_naming_the_setup_script(lab_layout):
    shutil.rmtree(lab_layout.base_path(REPO_MODELING_AGENT))

    with pytest.raises(LabWorkspaceError, match="modeling_agent.*scripts/lab/setup_workspace.sh"):
        lab_layout.workspace().ensure_ready()


def test_ensure_ready_refuses_a_base_branch_the_clone_does_not_have(lab_layout):
    specs = repo_specs_from_settings(lab_layout.root, lab_layout.git_root, {**DEFAULT_UPSTREAMS,
                                                                            REPO_BESSER: "BESSER-PEARL/BESSER@nope"})
    workspace = LabWorkspace(lab_layout.root, specs, state_dir=lab_layout.state_dir, git_root=lab_layout.git_root)

    with pytest.raises(LabWorkspaceError, match="besser: base branch nope"):
        workspace.ensure_ready()


# -- state ---------------------------------------------------------------------------------------------

def test_state_json_has_the_version_and_every_field_and_reloads(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    change.issues = [{"repo": "besser", "number": 7, "title": "Notes", "closes": True}]
    change.pull_requests = {REPO_BESSER: 812}
    lab_workspace.save_change(change)

    state = json.loads(read(state_file(lab_layout, change.id)))

    assert state["version"] == LAB_STATE_VERSION == 2
    assert set(state) == {"version"} | {f.name for f in fields(LabChange)}
    assert state["kind"] == "editor_element" and state["owner"] == "Jordi" and state["status"] == "open"
    assert state["bases"][REPO_BESSER]["branch"] == "development"
    assert state["updated_at"] >= state["created_at"] > 0
    reloaded = LabWorkspace(lab_layout.root, lab_layout.specs(), state_dir=lab_layout.state_dir,
                            git_root=lab_layout.git_root).get_change(change.id)
    assert reloaded == change
    assert isinstance(reloaded.bases[REPO_BESSER], RepoBase)


def test_a_phase_1_state_file_loads_with_defaults(lab_workspace, lab_layout):
    write(state_file(lab_layout, "0123456789ab"), json.dumps({
        "id": "0123456789ab", "slug": "note", "title": "Note", "user": "jordi", "kind": "generator",
        "status": "parked", "created_at": 5.0, "branches": {"besser": "lab/jordi/note"}, "plan_approved": True}))

    change = lab_workspace.get_change("0123456789ab")

    assert change == LabChange(id="0123456789ab", slug="note", title="Note", owner="jordi", kind=ChangeKind.GENERATOR,
                               status="open", created_at=5.0, branches={"besser": "lab/jordi/note"},
                               plan_approved=True)


@pytest.mark.parametrize("state", ['{"kind": "nope"}', "{not json", "[]"])
def test_an_unreadable_change_state_is_a_refusal_naming_the_file(lab_workspace, lab_layout, state):
    change = lab_workspace.create_draft(owner="jordi")
    write(state_file(lab_layout, change.id), state)

    with pytest.raises(LabWorkspaceError, match=f"state of change {change.id}.*cannot be read"):
        lab_workspace.get_change(change.id)
    with pytest.raises(LabWorkspaceError, match=f"state of change {change.id}.*cannot be read"):
        lab_workspace.changes()


def test_a_change_id_must_have_the_shape_of_one(lab_workspace):
    with pytest.raises(LabWorkspaceError, match="invalid change id"):
        lab_workspace.get_change("../../etc")
    with pytest.raises(LabWorkspaceError, match="no change 0123456789ab"):
        lab_workspace.get_change("0123456789ab")


def test_changes_lists_drafts_and_open_changes_oldest_first_unless_asked(lab_workspace):
    draft = lab_workspace.create_draft(owner="ana")
    opened = open_new(lab_workspace)
    closed = lab_workspace.create_draft(owner="ben")
    closed.status = "closed"
    lab_workspace.save_change(closed)

    assert [c.id for c in lab_workspace.changes()] == [draft.id, opened.id]
    assert [c.id for c in lab_workspace.changes(statuses=("closed",))] == [closed.id]


def test_notes_accumulate_per_change(lab_workspace, lab_layout):
    change = lab_workspace.create_draft(owner="jordi")
    assert lab_workspace.notes(change) == ""

    lab_workspace.append_note(change, "Asked: add a note element.")
    lab_layout.workspace().append_note(change, "Done: added Note.\n")

    assert lab_workspace.notes(change) == "Asked: add a note element.\n\nDone: added Note.\n\n"
    assert not os.path.exists(os.path.join(lab_layout.root, ".lab"))


# -- opening a change ----------------------------------------------------------------------------------

def test_open_change_takes_a_slug_no_local_upstream_or_pushed_branch_has(lab_workspace, lab_layout):
    lab_layout.git_in(REPO_MODELING_AGENT, "branch", "lab/jordi-c/note-element", "develop")
    first = open_new(lab_workspace, owner="Jordi.C", title="Note element!")
    lab_workspace.record_pushed(REPO_BESSER, "lab/jordi-c/pushed", first.bases[REPO_BESSER].sha)

    pushed = open_new(lab_workspace, owner="Jordi.C", title="Pushed")

    assert first.slug == "note-element-2"
    assert first.branches == {key: "lab/jordi-c/note-element-2" for key in REPO_ORDER}
    assert first.title == "Note element!" and first.kind == ChangeKind.EDITOR_ELEMENT
    assert pushed.slug == "pushed-2"


def test_open_change_refuses_a_change_that_is_not_a_draft(lab_workspace):
    change = open_new(lab_workspace)

    with pytest.raises(LabWorkspaceError, match="is not a draft"):
        lab_workspace.open_change(change, title="Again", kind=ChangeKind.OTHER)


def test_slug_falls_back_and_is_reduced_to_branch_safe_characters(lab_workspace):
    change = open_new(lab_workspace, owner="Ünïcode_User", title="!!!", kind=ChangeKind.OTHER)

    assert change.branches[REPO_BESSER] == "lab/unicode-user/change"


@pytest.mark.parametrize("title, slug", [
    # Titles of the live test (C-D5): cut mid-word to "...get-bui" and "...not-stam" before.
    ("Comment why 'unknown' means not stamped in get_build_sha", "comment-why-unknown-means-not-stamped-in-get"),
    ("Comment why 'unknown' build SHA counts as not stamped", "comment-why-unknown-build-sha-counts-as-not"),
    ("Add the forty eight character title right here x", "add-the-forty-eight-character-title-right-here-x"),
    ("a" * 60, "a" * 48),
])
def test_a_long_title_is_cut_at_a_word_boundary(title, slug):
    assert _slug(title, "change") == slug


def test_run_git_runs_in_the_base_or_in_the_change_tree(lab_workspace):
    change = open_new(lab_workspace)

    assert lab_workspace.run_git(REPO_BESSER, "rev-parse", "--abbrev-ref", "HEAD").strip() == "development"
    assert lab_workspace.run_git(REPO_BESSER, "rev-parse", "--abbrev-ref", "HEAD", change=change).strip() == (
        "lab/jordi/note-element")
    assert lab_workspace.run_git(REPO_FRONTEND, "rev-parse", "--show-toplevel", change=change).strip().replace(
        "\\", "/").lower() == tree(lab_workspace, change, REPO_FRONTEND).replace("\\", "/").lower()


# -- the work of a change ------------------------------------------------------------------------------

def test_repo_changes_reports_each_repository_without_artefacts_or_the_submodule_pointer(lab_workspace):
    change = open_new(lab_workspace)
    besser = tree(lab_workspace, change, REPO_BESSER)
    editor = tree(lab_workspace, change, REPO_FRONTEND)
    write(os.path.join(besser, "README.md"), "BESSER Lab\n")
    os.remove(os.path.join(besser, "docs", "old.rst"))
    write(os.path.join(besser, "besser", "new_gen.py"), "print('hi')\n")
    write(os.path.join(besser, "café.md"), "non-ASCII name\n")
    git(besser, "mv", "besser/old_name.py", "besser/new_name.py")
    write(os.path.join(besser, ".besser_trace.jsonl"), "{}\n")
    write(os.path.join(besser, ".besser_command_output", "out.txt"), "log\n")
    write(os.path.join(besser, "run.sandbox-home", "cache"), "x\n")
    write(os.path.join(editor, "packages", "webapp", "src", "note.ts"), "export const note = 1;\n")
    git(editor, "add", "-A")
    git(editor, "commit", "-q", "-m", "note")  # moves the editor away from BESSER's submodule pointer
    write(os.path.join(editor, "packages", "webapp", "src", "app.ts"), "export const app = 2;\n")

    changes = lab_workspace.repo_changes(change)

    assert [(c.repo, c.branch, c.base) for c in changes] == [
        (REPO_FRONTEND, "lab/jordi/note-element", "develop"),
        (REPO_BESSER, "lab/jordi/note-element", "development"),
    ]
    assert set(changes[0].files) == {
        FileChange("packages/webapp/src/app.ts", "modified"),
        FileChange("packages/webapp/src/note.ts", "added"),
    }
    assert set(changes[1].files) == {
        FileChange("README.md", "modified"),
        FileChange("docs/old.rst", "deleted"),
        FileChange("besser/new_gen.py", "added"),
        FileChange("besser/new_name.py", "renamed"),
        FileChange("café.md", "added"),
    }
    assert lab_workspace.touched_paths(change, REPO_BESSER) == {
        "README.md", "docs/old.rst", "besser/new_gen.py", "besser/old_name.py", "besser/new_name.py", "café.md"}


def test_commit_all_commits_only_the_changed_paths_as_the_login(lab_workspace):
    change = open_new(lab_workspace)
    besser = tree(lab_workspace, change, REPO_BESSER)
    editor = tree(lab_workspace, change, REPO_FRONTEND)
    write(os.path.join(besser, "README.md"), "BESSER Lab\n")
    write(os.path.join(besser, "café.md"), "non-ASCII name\n")
    write(os.path.join(besser, ".besser_trace.jsonl"), "{}\n")
    write(os.path.join(editor, "packages", "webapp", "src", "note.ts"), "export const note = 1;\n")
    git(editor, "add", "-A")
    git(editor, "commit", "-q", "-m", "note")
    git(besser, "add", FRONTEND_SUBMODULE_PATH)  # a staged pointer must stay out of the lab's commit
    write(os.path.join(editor, "packages", "webapp", "src", "app.ts"), "export const app = 2;\n")
    besser_before = git(besser, "rev-parse", "HEAD")

    shas = lab_workspace.commit_all(change, "feat: note element", login="Jordi")

    assert set(shas) == {REPO_FRONTEND, REPO_BESSER}
    assert shas[REPO_BESSER] == git(besser, "rev-parse", "HEAD") != besser_before
    assert shas[REPO_FRONTEND] == git(editor, "rev-parse", "HEAD")
    assert git(besser, "log", "-1", "--format=%an <%ae>|%cn|%s") == (
        "Jordi <Jordi@users.noreply.github.com>|Jordi|feat: note element")
    committed = git(besser, "-c", "core.quotepath=false", "show", "--name-only", "--format=", "HEAD").splitlines()
    assert sorted(committed) == ["README.md", "café.md"]
    assert ".besser_trace.jsonl" in git(besser, "ls-files", "--others")
    assert git(editor, "show", "--name-only", "--format=", "HEAD") == "packages/webapp/src/app.ts"
    assert lab_workspace.commit_all(change, "feat: nothing new", login="Jordi") == {}
    # Committed work still counts against the base.
    assert {c.repo for c in lab_workspace.repo_changes(change)} == {REPO_FRONTEND, REPO_BESSER}


def test_git_runs_no_hook_or_fsmonitor_command_the_agent_could_have_planted(lab_workspace, lab_layout, tmp_path):
    change = open_new(lab_workspace)
    marker = tmp_path / "planted-ran"
    script = f'#!/bin/sh\necho ran >> "{marker.as_posix()}"\n'
    for hook in ("pre-commit", "commit-msg", "post-commit", "post-checkout", "reference-transaction"):
        path = os.path.join(lab_layout.git_dir(REPO_BESSER), "hooks", hook)
        write(path, script)
        os.chmod(path, 0o755)
    fsmonitor = tmp_path / "fsmonitor.sh"
    write(str(fsmonitor), script)
    os.chmod(fsmonitor, 0o755)
    lab_layout.git_in(REPO_BESSER, "config", "core.fsmonitor", fsmonitor.as_posix())
    write(tree(lab_workspace, change, REPO_BESSER, "README.md"), "BESSER Lab\n")

    lab_workspace.repo_changes(change)
    lab_workspace.commit_all(change, "feat: note element", login="Jordi")
    open_new(lab_workspace, title="Second")
    lab_workspace.drop(change)

    assert not marker.exists()


def test_a_repository_made_inside_a_tree_is_named_and_removed_by_drop(lab_workspace):
    change = open_new(lab_workspace)
    nested = tree(lab_workspace, change, REPO_BESSER, "besser", "generators", "rusty")
    write(os.path.join(nested, "Cargo.toml"), "[package]\n")
    git(nested, "init", "-q")
    write(tree(lab_workspace, change, REPO_BESSER, "README.md"), "BESSER Lab\n")
    before = lab_workspace.run_git(REPO_BESSER, "rev-parse", "HEAD", change=change)

    with pytest.raises(LabWorkspaceError, match=r"besser/generators/rusty/ is a git repository of its own"):
        lab_workspace.commit_all(change, "feat: rusty", login="Jordi")
    assert lab_workspace.run_git(REPO_BESSER, "rev-parse", "HEAD", change=change) == before

    lab_workspace.drop(change)

    assert not os.path.exists(nested)
    open_new(lab_workspace, title="Next")


def test_merge_github_work_merges_commits_added_on_github(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    besser = tree(lab_workspace, change, REPO_BESSER)
    write(os.path.join(besser, "README.md"), "BESSER Lab\n")
    lab_workspace.commit_all(change, "feat: x", login="Jordi")
    # A suggestion accepted on GitHub: one more commit on the change's branch, held by another git directory.
    github = os.path.join(lab_layout.home, "github")
    git(lab_layout.home, "clone", "-q", "--bare", lab_layout.git_dir(REPO_BESSER), github)
    work = os.path.join(lab_layout.home, "github-work")
    git(lab_layout.home, "clone", "-q", "-b", change.branches[REPO_BESSER], github, work)
    write(os.path.join(work, "README.md"), "BESSER Lab, reviewed\n")
    git(work, "commit", "-q", "-am", "Apply suggestion")
    git(work, "push", "-q", "origin", change.branches[REPO_BESSER])

    lab_workspace.merge_github_work(change, REPO_BESSER, github, f"refs/heads/{change.branches[REPO_BESSER]}")

    assert read(os.path.join(besser, "README.md")) == "BESSER Lab, reviewed\n"
    assert lab_workspace.run_git(REPO_BESSER, "log", "-1", "--format=%s", change=change).strip() == "Apply suggestion"
    # GitHub's head is recorded as pushed: taking it does not make the change look unpushed.
    assert lab_workspace.unpushed_repos(change) == []


def test_git_errors_are_raised_with_secrets_redacted(lab_workspace):
    token = "ghp_" + "a1B2c3D4" * 5
    with pytest.raises(LabWorkspaceError) as error:
        lab_workspace.run_git(REPO_BESSER, "switch", token)
    assert "git switch failed in besser" in str(error.value)
    assert token not in str(error.value)

    own_secret = "plain-secret-without-a-known-shape"
    with pytest.raises(LabWorkspaceError) as error:
        lab_workspace.run_git(REPO_BESSER, "switch", own_secret, env_extra={"LAB_EXTRA": own_secret})
    assert own_secret not in str(error.value)
