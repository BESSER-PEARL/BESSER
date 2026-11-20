"""LabWorkspace against real git repositories: BESSER with the editor as a submodule, and the modeling agent."""

import json
import os
import shutil
import subprocess

import pytest

from besser.spec_driven_agent.lab.contracts import (
    FRONTEND_SUBMODULE_PATH,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    ChangeKind,
    FileChange,
    RepoSpec,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError, default_repos


def git(cwd, *args):
    result = subprocess.run(
        ["git", "-c", "user.name=setup", "-c", "user.email=setup@example.com", "-c", "core.autocrlf=false",
         "-c", "protocol.file.allow=always", *args],
        cwd=cwd, check=True, capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def init_repo(path, branch, files):
    os.makedirs(path)
    git(path, "init", "-q", "-b", branch)
    for name, text in files.items():
        write(os.path.join(path, name), text)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A lab root as the server has it: the editor submodule on a detached HEAD, its base only on origin."""
    base = tmp_path_factory.mktemp("lab_template")
    editor_origin = str(base / "editor-origin")
    init_repo(editor_origin, "develop", {"packages/webapp/src/app.ts": "export const app = 1;\n"})
    root = base / "root"
    besser = str(root / "besser")
    init_repo(besser, "development", {
        "README.md": "BESSER\n",
        "docs/old.rst": "Old page\n",
        "besser/old_name.py": "VALUE = 1\n",
    })
    git(besser, "submodule", "add", "-q", editor_origin, FRONTEND_SUBMODULE_PATH)
    git(besser, "commit", "-q", "-m", "add the editor")
    editor = os.path.join(besser, *FRONTEND_SUBMODULE_PATH.split("/"))
    git(editor, "checkout", "-q", "--detach")
    git(editor, "branch", "-q", "-D", "develop")
    init_repo(str(root / "modeling-agent"), "develop", {"agent.py": "AGENT = 1\n"})
    return root


@pytest.fixture
def root(template, tmp_path):
    target = tmp_path / "root"
    shutil.copytree(template, target)
    return str(target)


@pytest.fixture
def ws(root):
    return LabWorkspace(root)


def path_of(ws, repo, *parts):
    return os.path.join(ws.repos[repo].path, *parts)


def branch_of(ws, repo):
    return git(ws.repos[repo].path, "rev-parse", "--abbrev-ref", "HEAD")


def start(ws, title="Note element"):
    return ws.start_change(user="Jordi", title=title, kind=ChangeKind.EDITOR_ELEMENT)


def test_default_repos_lists_the_three_checkouts_in_pull_request_order(tmp_path):
    repos = default_repos(str(tmp_path))

    assert tuple(spec.key for spec in repos) == REPO_ORDER
    by_key = {spec.key: spec for spec in repos}
    assert by_key[REPO_BESSER].path == os.path.join(str(tmp_path), "besser")
    assert by_key[REPO_FRONTEND].path == os.path.join(str(tmp_path), "besser", *FRONTEND_SUBMODULE_PATH.split("/"))
    assert by_key[REPO_MODELING_AGENT].path == os.path.join(str(tmp_path), "modeling-agent")
    assert [(s.base_branch, s.github_owner, s.github_name) for s in repos] == [
        ("develop", "BESSER-PEARL", "BESSER-Web-Modeling-Editor"),
        ("development", "BESSER-PEARL", "BESSER"),
        ("develop", "BESSER-PEARL", "modeling-agent"),
    ]


def test_ensure_ready_accepts_the_checkouts_and_rejects_a_folder_inside_another_repository(ws, root):
    ws.ensure_ready()

    # An uninitialised submodule is an empty folder inside BESSER: git would act on BESSER.
    inner = os.path.join(root, "besser", "docs")
    with pytest.raises(LabWorkspaceError, match="not a git checkout of its own"):
        LabWorkspace(root, [RepoSpec(REPO_FRONTEND, inner, "develop", "o", "n")]).ensure_ready()
    with pytest.raises(LabWorkspaceError, match="does not exist"):
        LabWorkspace(root, [RepoSpec(REPO_FRONTEND, os.path.join(root, "missing"), "develop", "o", "n")]).ensure_ready()
    with pytest.raises(LabWorkspaceError, match="base branch nope not found"):
        LabWorkspace(root, [RepoSpec(REPO_BESSER, os.path.join(root, "besser"), "nope", "o", "n")]).ensure_ready()


def test_start_change_checks_out_a_unique_lab_branch_everywhere_and_saves_its_state(ws, root):
    git(path_of(ws, REPO_MODELING_AGENT), "branch", "lab/jordi-c/note-element")

    change = ws.start_change(user="Jordi.C", title="Note element!", kind=ChangeKind.EDITOR_ELEMENT)

    assert change.slug == "note-element-2"
    assert change.branches == {key: "lab/jordi-c/note-element-2" for key in REPO_ORDER}
    assert {key: branch_of(ws, key) for key in REPO_ORDER} == change.branches
    with open(os.path.join(root, ".lab", "changes", change.id, "state.json"), encoding="utf-8") as handle:
        state = json.load(handle)
    assert state["kind"] == "editor_element" and state["user"] == "Jordi.C" and state["status"] == "active"
    assert LabWorkspace(root).active_change() == change
    with pytest.raises(LabWorkspaceError, match="is active"):
        start(ws, "Another")


def test_a_state_folder_outside_the_root_holds_every_change_and_nothing_lands_under_the_root(root, tmp_path):
    # Outside root, the lab agent's sandboxed shell (which can write all of root) cannot reach the state.
    state_dir = str(tmp_path / "state")
    ws = LabWorkspace(root, state_dir=state_dir)
    assert ws.state_dir == os.path.abspath(state_dir)
    assert LabWorkspace(root).state_dir == os.path.join(os.path.abspath(root), ".lab")

    change = start(ws)
    ws.append_note(change, "## Plan\nAdd a note.")

    assert os.path.isfile(os.path.join(state_dir, "changes", change.id, "state.json"))
    assert read(os.path.join(state_dir, "changes", change.id, "notes.md")) == "## Plan\nAdd a note.\n\n"
    assert not os.path.exists(os.path.join(root, ".lab"))
    assert LabWorkspace(root, state_dir=state_dir).active_change() == change
    assert LabWorkspace(root).active_change() is None
    ws.park(change)
    assert [c.id for c in LabWorkspace(root, state_dir=state_dir).parked_changes()] == [change.id]
    ws.drop(ws.resume(change.id))
    assert not os.path.exists(os.path.join(state_dir, "changes", change.id))


def test_slug_falls_back_and_is_reduced_to_branch_safe_characters(ws):
    change = ws.start_change(user="Ünïcode_User", title="!!!", kind=ChangeKind.OTHER)

    assert change.branches[REPO_BESSER] == "lab/unicode-user/change"


def test_start_change_refuses_uncommitted_work_but_ignores_run_artefacts(ws, root):
    write(path_of(ws, REPO_BESSER, ".besser_trace.jsonl"), "{}\n")
    write(path_of(ws, REPO_BESSER, ".besser_command_output", "out.txt"), "log\n")
    ws.drop(start(ws))

    write(path_of(ws, REPO_MODELING_AGENT, "stray.py"), "x = 1\n")
    with pytest.raises(LabWorkspaceError, match="modeling_agent has uncommitted changes"):
        start(ws)
    assert ws.active_change() is None
    assert branch_of(ws, REPO_BESSER) == "development"


def test_repo_changes_reports_each_repository_without_artefacts_or_the_submodule_pointer(ws):
    change = start(ws)
    besser = ws.repos[REPO_BESSER].path
    write(path_of(ws, REPO_BESSER, "README.md"), "BESSER Lab\n")
    os.remove(path_of(ws, REPO_BESSER, "docs", "old.rst"))
    write(path_of(ws, REPO_BESSER, "besser", "new_gen.py"), "print('hi')\n")
    write(path_of(ws, REPO_BESSER, "café.md"), "non-ASCII name\n")
    git(besser, "mv", "besser/old_name.py", "besser/new_name.py")
    write(path_of(ws, REPO_BESSER, ".besser_trace.jsonl"), "{}\n")
    write(path_of(ws, REPO_BESSER, ".besser_command_output", "out.txt"), "log\n")
    write(path_of(ws, REPO_BESSER, "run.sandbox-home", "cache"), "x\n")
    editor = ws.repos[REPO_FRONTEND].path
    write(os.path.join(editor, "packages", "webapp", "src", "note.ts"), "export const note = 1;\n")
    git(editor, "add", "-A")
    git(editor, "commit", "-q", "-m", "note")  # moves BESSER's submodule pointer
    write(os.path.join(editor, "packages", "webapp", "src", "app.ts"), "export const app = 2;\n")

    changes = ws.repo_changes(change)

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


def test_commit_all_commits_only_the_changed_paths_as_the_user(ws):
    change = start(ws)
    besser = ws.repos[REPO_BESSER].path
    editor = ws.repos[REPO_FRONTEND].path
    write(path_of(ws, REPO_BESSER, "README.md"), "BESSER Lab\n")
    write(path_of(ws, REPO_BESSER, "café.md"), "non-ASCII name\n")
    write(path_of(ws, REPO_BESSER, ".besser_trace.jsonl"), "{}\n")
    write(os.path.join(editor, "packages", "webapp", "src", "note.ts"), "export const note = 1;\n")
    git(editor, "add", "-A")
    git(editor, "commit", "-q", "-m", "note")
    git(besser, "add", FRONTEND_SUBMODULE_PATH)  # a staged pointer must stay out of the lab's commit
    write(os.path.join(editor, "packages", "webapp", "src", "app.ts"), "export const app = 2;\n")
    besser_before = git(besser, "rev-parse", "HEAD")

    shas = ws.commit_all(change, "feat: note element")

    assert set(shas) == {REPO_FRONTEND, REPO_BESSER}
    assert shas[REPO_BESSER] == git(besser, "rev-parse", "HEAD") != besser_before
    assert shas[REPO_FRONTEND] == git(editor, "rev-parse", "HEAD")
    assert git(besser, "log", "-1", "--format=%an <%ae>|%cn|%s") == (
        "Jordi <Jordi@users.noreply.github.com>|Jordi|feat: note element")
    committed = git(besser, "-c", "core.quotepath=false", "show", "--name-only", "--format=", "HEAD").splitlines()
    assert sorted(committed) == ["README.md", "café.md"]
    assert ".besser_trace.jsonl" in git(besser, "ls-files", "--others")
    assert git(editor, "show", "--name-only", "--format=", "HEAD") == "packages/webapp/src/app.ts"
    assert ws.commit_all(change, "feat: nothing new") == {}
    # Committed work still counts against the merge-base.
    assert {c.repo for c in ws.repo_changes(change)} == {REPO_FRONTEND, REPO_BESSER}


def test_git_runs_no_hook_or_fsmonitor_command_the_agent_could_have_planted(ws, tmp_path):
    # The lab agent's file tools reach .git/: anything git ran from there would run outside the sandbox.
    change = start(ws)
    besser = ws.repos[REPO_BESSER].path
    marker = tmp_path / "planted-ran"
    script = f'#!/bin/sh\necho ran >> "{marker.as_posix()}"\n'
    hooks = os.path.join(besser, git(besser, "rev-parse", "--git-path", "hooks"))
    for hook in ("pre-commit", "commit-msg", "post-commit", "post-checkout"):
        write(os.path.join(hooks, hook), script)
        os.chmod(os.path.join(hooks, hook), 0o755)
    fsmonitor = tmp_path / "fsmonitor.sh"
    write(str(fsmonitor), script)
    os.chmod(fsmonitor, 0o755)
    git(besser, "config", "core.fsmonitor", fsmonitor.as_posix())
    write(path_of(ws, REPO_BESSER, "README.md"), "BESSER Lab\n")

    ws.repo_changes(change)
    ws.commit_all(change, "feat: note element")
    ws.park(change)

    assert not marker.exists()


def test_commit_all_refuses_a_repository_that_is_not_on_the_change_branch(ws):
    change = start(ws)
    besser = ws.repos[REPO_BESSER].path
    git(besser, "switch", "-q", "development")
    write(path_of(ws, REPO_BESSER, "README.md"), "edited on the base branch\n")
    before = git(besser, "rev-parse", "HEAD")

    with pytest.raises(LabWorkspaceError, match="not on the change's branch"):
        ws.commit_all(change, "feat: x")
    assert git(besser, "rev-parse", "HEAD") == before


def test_park_commits_work_in_progress_and_resume_brings_it_back(ws, root):
    change = start(ws)
    write(path_of(ws, REPO_BESSER, "README.md"), "BESSER Lab\n")
    write(os.path.join(ws.repos[REPO_FRONTEND].path, "packages", "webapp", "src", "app.ts"), "export const app = 2;\n")

    ws.park(change)

    assert change.status == "parked"
    assert ws.active_change() is None
    assert [c.id for c in LabWorkspace(root).parked_changes()] == [change.id]
    assert {key: branch_of(ws, key) for key in REPO_ORDER} == {
        REPO_FRONTEND: "develop", REPO_BESSER: "development", REPO_MODELING_AGENT: "develop"}
    assert read(path_of(ws, REPO_BESSER, "README.md")) == "BESSER\n"
    assert git(ws.repos[REPO_BESSER].path, "log", "-1", "--format=%s", change.branches[REPO_BESSER]) == (
        "wip: Note element")
    with pytest.raises(LabWorkspaceError, match="not active"):
        ws.park(change)

    resumed = LabWorkspace(root).resume(change.id)

    assert resumed.status == "active" and resumed.id == change.id
    assert {key: branch_of(ws, key) for key in REPO_ORDER} == change.branches
    assert read(path_of(ws, REPO_BESSER, "README.md")) == "BESSER Lab\n"
    assert ws.parked_changes() == []
    with pytest.raises(LabWorkspaceError, match="is active"):
        ws.resume(change.id)


def test_resume_rejects_an_id_that_is_not_a_change_id(ws):
    with pytest.raises(LabWorkspaceError, match="invalid change id"):
        ws.resume("../../etc")
    with pytest.raises(LabWorkspaceError, match="no change"):
        ws.resume("0123456789ab")


def test_drop_discards_the_work_the_branches_and_the_state(ws, root):
    change = start(ws)
    write(path_of(ws, REPO_BESSER, "README.md"), "committed\n")
    ws.commit_all(change, "feat: x")
    write(path_of(ws, REPO_BESSER, "README.md"), "uncommitted\n")
    write(path_of(ws, REPO_BESSER, "besser", "new_gen.py"), "print('hi')\n")

    ws.drop(change)

    assert ws.active_change() is None
    assert branch_of(ws, REPO_BESSER) == "development"
    assert read(path_of(ws, REPO_BESSER, "README.md")) == "BESSER\n"
    assert not os.path.exists(path_of(ws, REPO_BESSER, "besser", "new_gen.py"))
    for key in REPO_ORDER:
        assert git(ws.repos[key].path, "branch", "--list", "lab/*") == ""
    assert not os.path.exists(os.path.join(root, ".lab", "changes", change.id))


def test_drop_of_a_parked_change_leaves_the_active_one_checked_out(ws):
    parked = start(ws, "First")
    write(path_of(ws, REPO_BESSER, "README.md"), "first\n")
    ws.park(parked)
    active = start(ws, "Second")
    write(path_of(ws, REPO_BESSER, "README.md"), "second\n")

    ws.drop(parked)

    assert ws.parked_changes() == []
    assert ws.active_change().id == active.id
    assert branch_of(ws, REPO_BESSER) == "lab/jordi/second"
    assert read(path_of(ws, REPO_BESSER, "README.md")) == "second\n"
    assert git(ws.repos[REPO_BESSER].path, "branch", "--list", "lab/jordi/first") == ""


def test_files_only_the_change_ignores_do_not_block_the_base_after_park_or_drop(ws):
    change = start(ws)
    write(path_of(ws, REPO_BESSER, ".gitignore"), "gen_out/\n*.tmp\n")
    write(path_of(ws, REPO_BESSER, "gen_out", "result.txt"), "generated\n")
    write(path_of(ws, REPO_BESSER, "besser", "x.tmp"), "scratch\n")
    write(path_of(ws, REPO_BESSER, "venv", "lib.py"), "kept\n")
    write(path_of(ws, REPO_BESSER, ".git", "info", "exclude"), "venv/\n")  # ignored on every branch

    ws.park(change)

    assert not os.path.exists(path_of(ws, REPO_BESSER, "gen_out", "result.txt"))
    assert not os.path.exists(path_of(ws, REPO_BESSER, "besser", "x.tmp"))
    assert read(path_of(ws, REPO_BESSER, "venv", "lib.py")) == "kept\n"
    ws.ensure_clean()
    change = ws.resume(change.id)
    write(path_of(ws, REPO_BESSER, "gen_out", "result.txt"), "generated\n")

    ws.drop(change)

    ws.ensure_clean()
    assert read(path_of(ws, REPO_BESSER, "venv", "lib.py")) == "kept\n"


def test_a_repository_made_inside_a_checkout_is_named_and_removed_by_drop(ws):
    change = start(ws)
    nested = path_of(ws, REPO_BESSER, "besser", "generators", "rusty")
    write(os.path.join(nested, "Cargo.toml"), "[package]\n")
    git(nested, "init", "-q")
    write(path_of(ws, REPO_BESSER, "README.md"), "BESSER Lab\n")
    before = git(ws.repos[REPO_BESSER].path, "rev-parse", "HEAD")

    with pytest.raises(LabWorkspaceError, match=r"besser/generators/rusty/ is a git repository of its own"):
        ws.commit_all(change, "feat: rusty")
    assert git(ws.repos[REPO_BESSER].path, "rev-parse", "HEAD") == before

    ws.drop(change)

    assert not os.path.exists(nested)
    start(ws, "Next")


def test_drop_refuses_a_repository_switched_to_another_branch_and_discards_nothing(ws):
    change = start(ws)
    besser = ws.repos[REPO_BESSER].path
    write(path_of(ws, REPO_FRONTEND, "packages", "webapp", "src", "app.ts"), "export const app = 2;\n")
    git(besser, "switch", "-q", "-c", "operator-hotfix")
    write(path_of(ws, REPO_BESSER, "README.md"), "operator edit\n")
    write(path_of(ws, REPO_BESSER, "operator_notes.txt"), "notes\n")

    with pytest.raises(LabWorkspaceError, match="besser is on operator-hotfix"):
        ws.drop(change)

    assert read(path_of(ws, REPO_BESSER, "README.md")) == "operator edit\n"
    assert read(path_of(ws, REPO_BESSER, "operator_notes.txt")) == "notes\n"
    assert read(path_of(ws, REPO_FRONTEND, "packages", "webapp", "src", "app.ts")) == "export const app = 2;\n"
    assert ws.active_change() == change


def test_drop_of_a_change_whose_start_failed_part_way_leaves_repositories_without_its_branch(ws):
    change = start(ws)
    agent = ws.repos[REPO_MODELING_AGENT].path
    git(agent, "switch", "-q", "develop")
    git(agent, "branch", "-q", "-D", change.branches[REPO_MODELING_AGENT])
    write(os.path.join(agent, "agent.py"), "AGENT = 2\n")

    ws.drop(change)

    assert ws.active_change() is None
    assert branch_of(ws, REPO_BESSER) == "development"
    assert read(os.path.join(agent, "agent.py")) == "AGENT = 2\n"


@pytest.mark.parametrize("state", ['{"kind": "nope"}', "{not json", "[]"])
def test_an_unreadable_change_state_is_a_refusal_naming_the_file(ws, root, state):
    parked = start(ws)
    ws.park(parked)
    write(os.path.join(root, ".lab", "changes", parked.id, "state.json"), state)

    for read_state in (ws.active_change, ws.parked_changes):
        with pytest.raises(LabWorkspaceError, match=f"state of change {parked.id}.*cannot be read"):
            read_state()


def test_notes_accumulate_per_change(ws, root):
    change = start(ws)
    assert ws.notes(change) == ""

    ws.append_note(change, "Asked: add a note element.")
    LabWorkspace(root).append_note(change, "Done: added Note.\n")

    assert ws.notes(change) == "Asked: add a note element.\n\nDone: added Note.\n\n"


def test_diff_shows_new_files_without_touching_the_index_and_is_bounded(ws):
    start(ws)
    besser = ws.repos[REPO_BESSER].path
    write(path_of(ws, REPO_BESSER, "README.md"), "BESSER Lab\n")
    write(path_of(ws, REPO_BESSER, "besser", "new_gen.py"), "print('hi')\n")
    write(path_of(ws, REPO_BESSER, ".besser_trace.jsonl"), "{}\n")

    text = ws.diff(REPO_BESSER)

    assert "+BESSER Lab" in text and "+print('hi')" in text
    assert ".besser_trace" not in text
    assert git(besser, "diff", "--cached", "--name-only") == ""
    assert "besser/new_gen.py" in git(besser, "ls-files", "--others")
    short = ws.diff(REPO_BESSER, max_bytes=40)
    assert short.endswith("bytes shown; read the files for the rest]\n")
    assert "[diff truncated: 40 of" in short


def test_git_errors_are_raised_with_secrets_redacted(ws):
    token = "ghp_" + "a1B2c3D4" * 5
    with pytest.raises(LabWorkspaceError) as error:
        ws.run_git(REPO_BESSER, "switch", token)
    assert "git switch failed in besser" in str(error.value)
    assert token not in str(error.value)

    own_secret = "plain-secret-without-a-known-shape"
    with pytest.raises(LabWorkspaceError) as error:
        ws.run_git(REPO_BESSER, "switch", own_secret, env_extra={"LAB_EXTRA": own_secret})
    assert own_secret not in str(error.value)
