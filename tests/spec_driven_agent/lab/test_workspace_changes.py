"""Several changes at once: one worktree per repository per change, opened, committed, refreshed and dropped."""

import os
import shutil
import threading

import pytest

from besser.spec_driven_agent.lab.contracts import (
    LAB_BASE_DIRNAME,
    LAB_CHANGES_DIRNAME,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    ChangeKind,
    FileChange,
    RepoBase,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from tests.spec_driven_agent.lab.conftest import SUBMODULE_PARTS, git, read, write


def open_new(ws, owner="ana", title="Note element"):
    return ws.open_change(ws.create_draft(owner=owner), title=title, kind=ChangeKind.EDITOR_ELEMENT)


def tree(ws, change, repo, *parts):
    return os.path.join(ws.change_repos(change)[repo].path, *parts)


def branches(layout, repo):
    return layout.git_in(repo, "branch", "--list", "--format=%(refname:short)", "lab/*").split()


def test_a_draft_has_no_branches_and_works_in_the_base(lab_workspace, lab_layout):
    draft = lab_workspace.create_draft(owner="ana")

    assert draft.status == "draft" and draft.owner == "ana" and draft.branches == {} and draft.bases == {}
    assert lab_workspace.change_root(draft) == os.path.join(lab_layout.root, LAB_BASE_DIRNAME)
    assert lab_workspace.change_repos(draft) == lab_workspace.base_repos
    assert lab_workspace.repo_changes(draft) == []
    assert lab_workspace.base_status(draft) == {}
    assert lab_workspace.get_change(draft.id) == draft
    for repo in REPO_ORDER:
        assert branches(lab_layout, repo) == []


def test_two_changes_opened_by_two_logins_have_their_own_trees(lab_workspace, lab_layout):
    ana = open_new(lab_workspace, "ana", "Note element")
    ben = open_new(lab_workspace, "ben", "Faster export")

    assert ana.status == ben.status == "open"
    assert ana.branches == {key: "lab/ana/note-element" for key in REPO_ORDER}
    assert ben.branches == {key: "lab/ben/faster-export" for key in REPO_ORDER}
    assert lab_workspace.change_root(ana) == os.path.join(lab_layout.root, LAB_CHANGES_DIRNAME, ana.id)
    for repo in REPO_ORDER:
        assert git(tree(lab_workspace, ana, repo), "rev-parse", "--abbrev-ref", "HEAD") == "lab/ana/note-element"
        assert git(tree(lab_workspace, ben, repo), "rev-parse", "--abbrev-ref", "HEAD") == "lab/ben/faster-export"

    write(tree(lab_workspace, ana, REPO_BESSER, "besser", "note.py"), "NOTE = 1\n")

    assert lab_workspace.repo_changes(ana)[0].files == (FileChange("besser/note.py", "added"),)
    assert lab_workspace.repo_changes(ben) == []
    assert not os.path.exists(tree(lab_workspace, ben, REPO_BESSER, "besser", "note.py"))
    assert not os.path.exists(lab_layout.base_path(REPO_BESSER, "besser", "note.py"))
    assert [c.id for c in lab_workspace.changes()] == [ana.id, ben.id]


def test_second_open_change_succeeds_while_the_first_is_open(lab_workspace):
    # Phase 1 refused it: "change ... is active; park or drop it first".
    first = open_new(lab_workspace, "ana", "First")
    write(tree(lab_workspace, first, REPO_BESSER, "README.md"), "first\n")

    second = open_new(lab_workspace, "ana", "Second")

    assert second.status == "open"
    assert read(tree(lab_workspace, first, REPO_BESSER, "README.md")) == "first\n"
    assert read(tree(lab_workspace, second, REPO_BESSER, "README.md")) == "BESSER\n"


def test_commit_all_authors_as_the_login_that_ran_the_build(lab_workspace, lab_layout):
    change = open_new(lab_workspace, "ana")
    write(tree(lab_workspace, change, REPO_BESSER, "README.md"), "BESSER Lab\n")

    shas = lab_workspace.commit_all(change, "feat: note element", login="ben")

    assert set(shas) == {REPO_BESSER}
    log = lab_layout.git_in(REPO_BESSER, "log", "-1", "--format=%H|%an <%ae>|%cn <%ce>|%s",
                            change.branches[REPO_BESSER])
    assert log == (f"{shas[REPO_BESSER]}|ben <ben@users.noreply.github.com>|"
                   "ben <ben@users.noreply.github.com>|feat: note element")


def test_the_editor_worktree_sits_at_the_submodule_path_and_besser_never_lists_the_pointer(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    repos = lab_workspace.change_repos(change)
    root = lab_workspace.change_root(change)

    assert repos[REPO_BESSER].path == os.path.join(root, "besser")
    assert repos[REPO_FRONTEND].path == os.path.join(root, "besser", *SUBMODULE_PARTS)
    assert repos[REPO_MODELING_AGENT].path == os.path.join(root, "modeling-agent")
    editor_worktrees = os.path.join(lab_layout.git_dir(REPO_FRONTEND), "worktrees")
    assert os.path.dirname(repos[REPO_FRONTEND].git_dir) == editor_worktrees
    assert os.path.dirname(repos[REPO_BESSER].git_dir) == os.path.join(lab_layout.git_dir(REPO_BESSER), "worktrees")
    assert read(tree(lab_workspace, change, REPO_FRONTEND, "packages", "webapp", "src", "app.ts")) == (
        "export const app = 1;\n")

    write(tree(lab_workspace, change, REPO_FRONTEND, "packages", "webapp", "src", "note.ts"), "export const n = 1;\n")
    lab_workspace.commit_all(change, "feat: note", login="ana")  # moves the editor's HEAD away from BESSER's pointer
    write(tree(lab_workspace, change, REPO_BESSER, "README.md"), "BESSER Lab\n")

    changes = {c.repo: c for c in lab_workspace.repo_changes(change)}
    assert changes[REPO_FRONTEND].files == (FileChange("packages/webapp/src/note.ts", "added"),)
    assert changes[REPO_BESSER].files == (FileChange("README.md", "modified"),)


def test_two_threads_opening_the_same_title_get_slug_and_slug_2(lab_workspace):
    drafts = [lab_workspace.create_draft(owner="ana") for _ in range(2)]
    barrier = threading.Barrier(2)
    opened, errors = [], []

    def open_one(draft):
        barrier.wait()
        try:
            opened.append(lab_workspace.open_change(draft, title="Note element", kind=ChangeKind.OTHER))
        except Exception as exc:  # reported below, a thread's exception is otherwise lost
            errors.append(exc)

    threads = [threading.Thread(target=open_one, args=(draft,)) for draft in drafts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(change.slug for change in opened) == ["note-element", "note-element-2"]


def test_drop_removes_the_worktrees_the_branches_and_the_state(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    other = open_new(lab_workspace, "ben", "Other")
    write(tree(lab_workspace, change, REPO_BESSER, "README.md"), "committed\n")
    lab_workspace.commit_all(change, "feat: x", login="ana")
    write(tree(lab_workspace, change, REPO_BESSER, "besser", "new.py"), "uncommitted\n")
    write(tree(lab_workspace, change, REPO_FRONTEND, "packages", "webapp", "src", "x.ts"), "uncommitted\n")

    lab_workspace.drop(change)

    assert not os.path.exists(lab_workspace.change_root(change))
    assert not os.path.exists(os.path.join(lab_layout.state_dir, "changes", change.id))
    for repo in REPO_ORDER:
        assert branches(lab_layout, repo) == ["lab/ben/other"]
        listed = lab_layout.git_in(repo, "worktree", "list", "--porcelain")
        assert change.id not in listed and other.id in listed
    with pytest.raises(LabWorkspaceError, match=f"no change {change.id}"):
        lab_workspace.get_change(change.id)
    assert [c.id for c in lab_workspace.changes()] == [other.id]


def test_remove_trees_keeps_the_state(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    lab_workspace.append_note(change, "Asked: a note element.")

    lab_workspace.remove_trees(change)

    assert not os.path.exists(lab_workspace.change_root(change))
    for repo in REPO_ORDER:
        assert branches(lab_layout, repo) == []
    assert lab_workspace.get_change(change.id) == change
    assert lab_workspace.notes(change) == "Asked: a note element.\n\n"


def test_refresh_bases_fast_forwards_from_the_upstream_and_prunes(lab_workspace, lab_layout):
    gone = open_new(lab_workspace)
    shutil.rmtree(lab_workspace.change_root(gone))  # a tree deleted by hand leaves a stale worktree entry
    new_sha = lab_layout.push_upstream(REPO_BESSER, "README.md", "BESSER, newer\n")

    bases = lab_workspace.refresh_bases()

    assert bases[REPO_BESSER] == RepoBase("BESSER-PEARL/BESSER", "development", new_sha)
    assert set(bases) == set(REPO_ORDER)
    assert read(lab_layout.base_path(REPO_BESSER, "README.md")) == "BESSER, newer\n"
    assert lab_layout.git_in(REPO_BESSER, "rev-parse", "refs/remotes/upstream/development") == new_sha
    assert lab_layout.git_in(REPO_BESSER, "rev-parse", "refs/heads/development") == new_sha
    for repo in REPO_ORDER:
        assert gone.id not in lab_layout.git_in(repo, "worktree", "list", "--porcelain")


def test_open_change_starts_from_the_refreshed_base_and_base_status_counts_commits(lab_workspace, lab_layout):
    started = lab_layout.push_upstream(REPO_BESSER, "README.md", "BESSER 2\n")

    change = open_new(lab_workspace)

    assert change.bases[REPO_BESSER] == RepoBase("BESSER-PEARL/BESSER", "development", started)
    assert change.bases[REPO_MODELING_AGENT].upstream == "BESSER-PEARL/modeling-agent"
    assert read(tree(lab_workspace, change, REPO_BESSER, "README.md")) == "BESSER 2\n"
    assert lab_workspace.base_status(change) == {repo: 0 for repo in REPO_ORDER}

    lab_layout.push_upstream(REPO_BESSER, "README.md", "BESSER 3\n")
    lab_layout.push_upstream(REPO_BESSER, "docs/new.rst", "New\n")
    lab_workspace.refresh_bases()

    assert lab_workspace.base_status(change) == {REPO_FRONTEND: 0, REPO_BESSER: 2, REPO_MODELING_AGENT: 0}
    assert read(tree(lab_workspace, change, REPO_BESSER, "README.md")) == "BESSER 2\n"  # never moved by itself


def test_an_open_that_fails_part_way_removes_what_it_made(lab_workspace, lab_layout):
    draft = lab_workspace.create_draft(owner="ana")
    # The editor's worktree is made last (it goes inside BESSER's): a file where its admin folder goes fails it.
    write(os.path.join(lab_layout.git_dir(REPO_FRONTEND), "worktrees"), "not a folder\n")

    with pytest.raises(LabWorkspaceError, match="failed in frontend"):
        lab_workspace.open_change(draft, title="Note element", kind=ChangeKind.OTHER)

    for repo in (REPO_BESSER, REPO_MODELING_AGENT):
        assert draft.id not in lab_layout.git_in(repo, "worktree", "list", "--porcelain")
    for repo in REPO_ORDER:
        assert branches(lab_layout, repo) == []
    assert not os.path.exists(os.path.join(lab_layout.root, LAB_CHANGES_DIRNAME, draft.id))
    reloaded = lab_workspace.get_change(draft.id)
    assert reloaded.status == "draft" and reloaded.branches == {}
    assert draft.status == "draft" and draft.branches == {}
