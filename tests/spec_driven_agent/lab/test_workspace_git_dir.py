"""Git directories live outside the working trees, and the lab never reads a ``.git`` file the sandbox can rewrite."""

import os
import shutil
from dataclasses import replace

import pytest

from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_MODELING_AGENT, ChangeKind
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError
from tests.spec_driven_agent.lab.conftest import git, write


def test_work_tree_git_file_is_never_read(lab_workspace, lab_layout, tmp_path):
    change = lab_workspace.open_change(lab_workspace.create_draft(owner="ana"), title="Note", kind=ChangeKind.OTHER)
    tree = lab_workspace.change_repos(change)[REPO_BESSER].path
    # What sandboxed code can do to its own tree: point .git at a repository whose config runs a command.
    pwned = tmp_path / "PWNED"
    evil = tmp_path / "evil"
    git(str(tmp_path), "init", "-q", str(evil))
    evil_git_dir = evil / ".git"
    git(str(tmp_path), "config", "--file", str(evil_git_dir / "config"), "filter.x.clean",
        f"touch '{pwned.as_posix()}'; cat")
    os.remove(os.path.join(tree, ".git"))  # Git for Windows marks it hidden, which open(..., "w") refuses
    write(os.path.join(tree, ".git"), f"gitdir: {evil_git_dir.as_posix()}\n")
    write(os.path.join(tree, ".gitattributes"), "* filter=x\n")
    write(os.path.join(tree, "README.md"), "BESSER Lab\n")

    shas = lab_workspace.commit_all(change, "feat: x", login="ana")

    assert not pwned.exists()
    assert lab_layout.git_in(REPO_BESSER, "rev-parse", change.branches[REPO_BESSER]) == shas[REPO_BESSER]
    assert sorted(lab_layout.git_in(REPO_BESSER, "show", "--name-only", "--format=", shas[REPO_BESSER]).split()) == [
        ".gitattributes", "README.md"]
    # The planted repository is armed: a git that follows the .git file runs its filter.
    write(os.path.join(tree, "other.md"), "x\n")
    git(tree, "add", "other.md")
    assert pwned.exists()


def test_in_tree_git_dir_is_refused(lab_layout):
    # A checkout made by a plain clone: its git directory is inside the workspace, writable by sandboxed code.
    agent = lab_layout.base_path(REPO_MODELING_AGENT)
    os.remove(os.path.join(agent, ".git"))
    shutil.move(lab_layout.git_dir(REPO_MODELING_AGENT), os.path.join(agent, ".git"))

    with pytest.raises(LabWorkspaceError, match="BESSER_LAB_GIT_ROOT") as error:
        lab_layout.workspace().ensure_ready()
    assert "modeling_agent" in str(error.value) and "scripts/lab/setup_workspace.sh" in str(error.value)


def test_a_git_dir_configured_inside_the_workspace_root_is_refused(lab_layout):
    inside = os.path.join(lab_layout.root, "agent-git")
    shutil.move(lab_layout.git_dir(REPO_MODELING_AGENT), inside)
    specs = [replace(spec, git_dir=inside) if spec.key == REPO_MODELING_AGENT else spec for spec in lab_layout.specs()]

    workspace = LabWorkspace(lab_layout.root, specs, state_dir=lab_layout.state_dir, git_root=lab_layout.git_root)
    with pytest.raises(LabWorkspaceError, match="modeling_agent.*BESSER_LAB_GIT_ROOT"):
        workspace.ensure_ready()
