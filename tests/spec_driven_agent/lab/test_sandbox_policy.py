"""The lab's mount policy: what a change's sandbox binds read-write, read-only and hides (D1 test 5)."""

import os

import pytest

from besser.spec_driven_agent.lab.contracts import FRONTEND_SUBMODULE_PATH, RepoSpec
from besser.spec_driven_agent.lab.sandbox_policy import LabSandboxPolicyError, lab_sandbox_mounts


def _spec(key: str, path, git_dir) -> RepoSpec:
    return RepoSpec(key=key, path=str(path), base_branch="develop", github_owner="o", github_name=key,
                    git_dir=str(git_dir))


@pytest.fixture
def layout(tmp_path):
    """A change's trees under ``<root>/changes/<id>``, git dirs in ``<root>-git``, a state folder beside them."""
    root = tmp_path / "work"
    git_root = tmp_path / "work-git"
    change = root / "changes" / "abcdef012345"
    besser = change / "besser"
    frontend = besser / FRONTEND_SUBMODULE_PATH
    agent = change / "modeling-agent"
    state = tmp_path / "work-state"
    for folder in (frontend, agent, git_root, state):
        folder.mkdir(parents=True)
    repos = {
        "frontend": _spec("frontend", frontend, git_root / "besser" / "modules" / "frontend"),
        "besser": _spec("besser", besser, git_root / "besser"),
        "modeling_agent": _spec("modeling_agent", agent, git_root / "modeling-agent"),
    }
    return {"root": root, "git_root": git_root, "besser": besser, "agent": agent, "state": state,
            "repos": repos}


def _real(path) -> str:
    return os.path.realpath(str(path))


def test_the_editor_nested_in_besser_is_bound_with_besser(layout):
    mounts = lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]),
                                git_root=str(layout["git_root"]), hidden=[str(layout["state"])])

    assert mounts.writable == (_real(layout["besser"]), _real(layout["agent"]))
    assert mounts.read_only == (_real(layout["git_root"]),)
    assert mounts.masked == (_real(layout["root"]), _real(layout["state"]))


def test_a_read_only_sandbox_binds_no_tree_read_write(layout):
    mounts = lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]),
                                git_root=str(layout["git_root"]), hidden=[], writable=False)

    assert mounts.writable == ()
    assert mounts.read_only == (_real(layout["git_root"]), _real(layout["besser"]), _real(layout["agent"]))
    assert mounts.masked == (_real(layout["root"]),)


def test_missing_hidden_paths_are_dropped_and_the_workspace_root_is_always_masked(layout, tmp_path):
    mounts = lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]),
                                git_root=str(layout["git_root"]),
                                hidden=[str(tmp_path / "no-such-folder"), str(layout["state"])])

    assert mounts.masked == (_real(layout["root"]), _real(layout["state"]))
    assert lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]), git_root=str(layout["git_root"]),
                              hidden=[]).masked == (_real(layout["root"]),)


def test_a_state_folder_inside_a_tree_is_refused(layout):
    state = layout["besser"] / ".lab"
    state.mkdir()

    with pytest.raises(LabSandboxPolicyError, match="BESSER_LAB_WORKSPACE_ROOT") as refused:
        lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]), git_root=str(layout["git_root"]),
                           hidden=[str(state)])
    assert _real(state) in str(refused.value)


@pytest.mark.parametrize("where", ["in_tree", "holds_tree"])
def test_a_git_root_overlapping_a_tree_is_refused(layout, where):
    git_root = layout["agent"] / "git" if where == "in_tree" else layout["root"]

    with pytest.raises(LabSandboxPolicyError, match="BESSER_LAB_GIT_ROOT"):
        lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]), git_root=str(git_root), hidden=[])


def test_a_hidden_folder_inside_the_git_root_is_refused(layout):
    # The read-only bind of the git root comes after the masks and would show it again.
    sessions = layout["git_root"] / "sessions"
    sessions.mkdir()

    with pytest.raises(LabSandboxPolicyError, match="BESSER_LAB_GIT_ROOT"):
        lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]), git_root=str(layout["git_root"]),
                           hidden=[str(sessions)])


def test_a_hidden_file_is_refused(layout, tmp_path):
    # The sandbox masks folders only; a file would stay in view.
    session_file = tmp_path / "sessions.json"
    session_file.write_text("{}", encoding="utf-8")

    with pytest.raises(LabSandboxPolicyError, match="folder"):
        lab_sandbox_mounts(layout["repos"], workspace_root=str(layout["root"]), git_root=str(layout["git_root"]),
                           hidden=[str(session_file)])
