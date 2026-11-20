"""The workspace's review operations: per-file diffs, closing a change and merging its base in."""

import os

import pytest

from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    REPO_ORDER,
    ChangeKind,
)
from besser.spec_driven_agent.lab import workspace as workspace_module
from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from tests.spec_driven_agent.lab.conftest import git, read, write

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x01"


def open_new(ws, owner="ana", title="Note element"):
    return ws.open_change(ws.create_draft(owner=owner), title=title, kind=ChangeKind.EDITOR_ELEMENT)


def tree(ws, change, repo, *parts):
    return os.path.join(ws.change_repos(change)[repo].path, *parts)


def branches(layout, repo):
    return layout.git_in(repo, "branch", "--list", "--format=%(refname:short)", "lab/*").split()


def push(ws, change, repo):
    """What the pull-request service does after a push: GitHub now holds the branch's head."""
    ws.record_pushed(repo, change.branches[repo], ws.run_git(repo, "rev-parse", "HEAD", change=change).strip())


# -- file_diffs ------------------------------------------------------------------------------------------

def test_file_diffs_one_entry_per_file_in_numstat_order(lab_workspace):
    change = open_new(lab_workspace)
    besser = tree(lab_workspace, change, REPO_BESSER)
    # Committed: a rename (same content, new path) and a modification.
    os.remove(os.path.join(besser, "besser", "old_name.py"))
    write(os.path.join(besser, "besser", "new_name.py"), "VALUE = 1\n")
    write(os.path.join(besser, "README.md"), "BESSER\nLab\n")
    lab_workspace.commit_all(change, "feat: rename", login="ana")
    # Uncommitted: a deletion, an untracked text file (space and non-ASCII), an untracked binary, run artefacts.
    os.remove(os.path.join(besser, "docs", "old.rst"))
    write(os.path.join(besser, "docs", "new pagé.rst"), "New\npage\n")
    with open(os.path.join(besser, "logo.png"), "wb") as handle:
        handle.write(PNG)
    write(os.path.join(besser, ".besser_trace.jsonl"), "{}\n")
    write(os.path.join(besser, "run.sandbox-home", "cache"), "x\n")

    diffs = lab_workspace.file_diffs(change, REPO_BESSER)

    assert [(d.path, d.old_path, d.status, d.additions, d.deletions, d.binary) for d in diffs] == [
        ("README.md", None, "modified", 1, 0, False),
        ("besser/new_name.py", "besser/old_name.py", "renamed", 0, 0, False),
        ("docs/new pagé.rst", None, "added", 2, 0, False),
        ("docs/old.rst", None, "deleted", 0, 1, False),
        ("logo.png", None, "added", None, None, True),
    ]
    by_path = {d.path: d for d in diffs}
    assert by_path["README.md"].patch.startswith("diff --git a/README.md b/README.md\n")
    assert "\n+Lab\n" in by_path["README.md"].patch and "docs/" not in by_path["README.md"].patch
    assert "rename from besser/old_name.py\nrename to besser/new_name.py" in by_path["besser/new_name.py"].patch
    assert by_path["docs/new pagé.rst"].patch.endswith("@@ -0,0 +1,2 @@\n+New\n+page\n")
    assert "\n-Old page\n" in by_path["docs/old.rst"].patch
    assert by_path["logo.png"].patch == ""
    # The real index is left as it was: the untracked files are still untracked.
    assert "?? logo.png" in git(besser, "status", "--porcelain")


def test_file_diffs_of_one_path_keeps_its_rename(lab_workspace):
    change = open_new(lab_workspace)
    besser = tree(lab_workspace, change, REPO_BESSER)
    os.remove(os.path.join(besser, "besser", "old_name.py"))
    write(os.path.join(besser, "besser", "new_name.py"), "VALUE = 1\nMORE = 2\n")
    write(os.path.join(besser, "README.md"), "BESSER Lab\n")
    lab_workspace.commit_all(change, "feat: rename", login="ana")

    diffs = lab_workspace.file_diffs(change, REPO_BESSER, path="besser/new_name.py")

    assert [(d.path, d.old_path, d.status, d.additions, d.deletions) for d in diffs] == [
        ("besser/new_name.py", "besser/old_name.py", "renamed", 1, 0)]
    assert "README" not in diffs[0].patch
    assert lab_workspace.file_diffs(change, REPO_BESSER, path="no/such.py") == []


def test_file_diffs_of_a_draft_are_empty_and_of_a_closed_change_refused(lab_workspace):
    draft = lab_workspace.create_draft(owner="ana")
    assert lab_workspace.file_diffs(draft, REPO_BESSER) == []

    change = open_new(lab_workspace)
    lab_workspace.close(change, outcome="closed")

    with pytest.raises(LabWorkspaceError,
                       match="its branches were deleted when it closed; the diff is on the pull request"):
        lab_workspace.file_diffs(change, REPO_BESSER)


def test_file_diffs_refuse_output_over_the_cap(lab_workspace, monkeypatch):
    change = open_new(lab_workspace)
    write(tree(lab_workspace, change, REPO_BESSER, "dist", "bundle.js"), "x = 1;\n" * 200)
    monkeypatch.setattr(workspace_module, "_MAX_DIFF_BYTES", 1000)

    with pytest.raises(LabWorkspaceError, match="besser: the diff is .* MB the lab reads"):
        lab_workspace.file_diffs(change, REPO_BESSER)


def test_file_diffs_in_the_editor_list_its_own_files(lab_workspace):
    change = open_new(lab_workspace)
    write(tree(lab_workspace, change, REPO_FRONTEND, "packages", "webapp", "src", "note.ts"), "export const n = 1;\n")
    lab_workspace.commit_all(change, "feat: note", login="ana")

    assert [d.path for d in lab_workspace.file_diffs(change, REPO_FRONTEND)] == ["packages/webapp/src/note.ts"]
    assert lab_workspace.file_diffs(change, REPO_BESSER) == []  # never the moved submodule pointer


# -- close -----------------------------------------------------------------------------------------------

def test_close_keeps_notes_and_state_and_deletes_trees_and_branches(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    other = open_new(lab_workspace, "ben", "Other")
    write(tree(lab_workspace, change, REPO_BESSER, "README.md"), "BESSER Lab\n")
    lab_workspace.commit_all(change, "feat: x", login="ana")
    push(lab_workspace, change, REPO_BESSER)
    lab_workspace.append_note(change, "Asked: a note element.")
    state = os.path.join(lab_layout.state_dir, "changes", change.id)
    kept = ("plan.json", "last_run.json", "runs.jsonl", "thread.jsonl")
    for name in kept:
        write(os.path.join(state, name), "{}\n")

    lab_workspace.close(change, outcome="merged")

    assert change.status == "closed" and change.outcome == "merged" and change.closed_at > 0
    reloaded = lab_workspace.get_change(change.id)
    assert (reloaded.status, reloaded.outcome, reloaded.closed_at) == ("closed", "merged", change.closed_at)
    assert reloaded.branches == change.branches  # the names stay on record
    assert lab_workspace.notes(change) == "Asked: a note element.\n\n"
    for name in kept:
        assert read(os.path.join(state, name)) == "{}\n"
    assert not os.path.exists(os.path.join(lab_layout.root, "changes", change.id))
    for repo in REPO_ORDER:
        assert branches(lab_layout, repo) == ["lab/ben/other"]
        assert change.id not in lab_layout.git_in(repo, "worktree", "list", "--porcelain")
    assert [c.id for c in lab_workspace.changes()] == [other.id]
    assert [c.id for c in lab_workspace.changes(statuses=("closed",))] == [change.id]
    assert lab_workspace.unpushed_repos(change) == []


def test_close_refuses_unpushed_work(lab_workspace):
    change = open_new(lab_workspace)
    assert lab_workspace.unpushed_repos(change) == []
    write(tree(lab_workspace, change, REPO_BESSER, "README.md"), "BESSER Lab\n")
    write(tree(lab_workspace, change, REPO_MODELING_AGENT, "agent.py"), "AGENT = 2\n")

    with pytest.raises(LabWorkspaceError, match="besser, modeling_agent"):
        lab_workspace.close(change, outcome="closed")
    lab_workspace.commit_all(change, "feat: x", login="ana")
    assert lab_workspace.unpushed_repos(change) == [REPO_BESSER, REPO_MODELING_AGENT]  # committed, never pushed
    push(lab_workspace, change, REPO_BESSER)
    assert lab_workspace.unpushed_repos(change) == [REPO_MODELING_AGENT]
    with pytest.raises(LabWorkspaceError, match="modeling_agent"):
        lab_workspace.close(change, outcome="closed")

    assert lab_workspace.get_change(change.id).status == "open"
    assert read(tree(lab_workspace, change, REPO_MODELING_AGENT, "agent.py")) == "AGENT = 2\n"
    push(lab_workspace, change, REPO_MODELING_AGENT)
    lab_workspace.close(change, outcome="closed")
    assert lab_workspace.get_change(change.id).outcome == "closed"


def test_close_refuses_a_draft_and_an_unknown_outcome(lab_workspace):
    with pytest.raises(LabWorkspaceError, match="is a draft"):
        lab_workspace.close(lab_workspace.create_draft(owner="ana"), outcome="closed")
    change = open_new(lab_workspace)
    with pytest.raises(LabWorkspaceError, match="outcome"):
        lab_workspace.close(change, outcome="abandoned")
    lab_workspace.close(change, outcome="merged")
    with pytest.raises(LabWorkspaceError, match="already closed"):
        lab_workspace.close(change, outcome="merged")


# -- merging the base in ---------------------------------------------------------------------------------

def test_base_merge_conflict_is_left_in_progress_and_concluded_by_commit_all(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    readme = tree(lab_workspace, change, REPO_BESSER, "README.md")
    write(readme, "BESSER, from the lab\n")
    lab_workspace.commit_all(change, "feat: readme", login="ana")
    old_head = lab_workspace.run_git(REPO_BESSER, "rev-parse", "HEAD", change=change).strip()
    upstream = lab_layout.push_upstream(REPO_BESSER, "README.md", "BESSER, upstream\n")
    lab_workspace.refresh_bases()

    assert lab_workspace.begin_base_merge(change, REPO_BESSER) == ["README.md"]

    assert lab_workspace.merge_in_progress(change, REPO_BESSER)
    assert not lab_workspace.merge_in_progress(change, REPO_MODELING_AGENT)
    assert os.path.isfile(os.path.join(lab_workspace.change_repos(change)[REPO_BESSER].git_dir, "MERGE_HEAD"))
    assert "<<<<<<<" in read(readme)
    assert lab_workspace.unpushed_repos(change) == [REPO_BESSER]
    with pytest.raises(LabWorkspaceError, match=r"conflict markers.*README\.md"):
        lab_workspace.commit_all(change, "fix: resolve", login="ben")
    assert lab_workspace.merge_in_progress(change, REPO_BESSER)

    write(readme, "BESSER, from the lab and upstream\n")
    shas = lab_workspace.commit_all(change, "fix: resolve", login="ben")

    head = shas[REPO_BESSER]
    parents = lab_workspace.run_git(REPO_BESSER, "rev-list", "--parents", "-n", "1", head, change=change).split()
    assert parents == [head, old_head, upstream]
    assert lab_layout.git_in(REPO_BESSER, "rev-parse", f"refs/heads/{change.branches[REPO_BESSER]}") == head
    assert not lab_workspace.merge_in_progress(change, REPO_BESSER)
    assert change.bases[REPO_BESSER].sha == upstream
    assert lab_workspace.get_change(change.id).bases[REPO_BESSER].sha == upstream
    assert lab_workspace.base_status(change)[REPO_BESSER] == 0
    assert [d.path for d in lab_workspace.file_diffs(change, REPO_BESSER)] == ["README.md"]


def test_a_conflict_resolved_to_the_change_side_still_concludes_the_merge(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    readme = tree(lab_workspace, change, REPO_BESSER, "README.md")
    write(readme, "BESSER, from the lab\n")
    lab_workspace.commit_all(change, "feat: readme", login="ana")
    lab_layout.push_upstream(REPO_BESSER, "README.md", "BESSER, upstream\n")
    lab_workspace.refresh_bases()
    lab_workspace.begin_base_merge(change, REPO_BESSER)

    write(readme, "BESSER, from the lab\n")  # the same content as HEAD, but still unmerged in the index
    shas = lab_workspace.commit_all(change, "fix: resolve", login="ana")

    assert set(shas) == {REPO_BESSER}
    assert not lab_workspace.merge_in_progress(change, REPO_BESSER)


def test_a_clean_base_merge_commits_and_moves_the_base(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    write(tree(lab_workspace, change, REPO_BESSER, "docs", "note.rst"), "Note\n")
    lab_workspace.commit_all(change, "docs: note", login="ana")
    old_head = lab_workspace.run_git(REPO_BESSER, "rev-parse", "HEAD", change=change).strip()
    upstream = lab_layout.push_upstream(REPO_BESSER, "README.md", "BESSER, upstream\n")
    lab_workspace.refresh_bases()
    assert lab_workspace.base_status(change)[REPO_BESSER] == 1

    assert lab_workspace.begin_base_merge(change, REPO_BESSER) == []

    assert change.bases[REPO_BESSER].sha == upstream
    assert lab_workspace.get_change(change.id).bases[REPO_BESSER].sha == upstream
    assert lab_workspace.base_status(change)[REPO_BESSER] == 0
    assert not lab_workspace.merge_in_progress(change, REPO_BESSER)
    parents = lab_workspace.run_git(REPO_BESSER, "rev-list", "--parents", "-n", "1", "HEAD", change=change).split()
    assert parents[1:] == [old_head, upstream]
    assert read(tree(lab_workspace, change, REPO_BESSER, "README.md")) == "BESSER, upstream\n"
    assert [d.path for d in lab_workspace.file_diffs(change, REPO_BESSER)] == ["docs/note.rst"]


def test_base_merge_refuses_uncommitted_work_and_a_draft(lab_workspace, lab_layout):
    change = open_new(lab_workspace)
    write(tree(lab_workspace, change, REPO_BESSER, "docs", "note.rst"), "Note\n")
    lab_layout.push_upstream(REPO_BESSER, "README.md", "BESSER, upstream\n")
    lab_workspace.refresh_bases()
    before = lab_workspace.run_git(REPO_BESSER, "rev-parse", "HEAD", change=change)

    with pytest.raises(LabWorkspaceError, match="uncommitted"):
        lab_workspace.begin_base_merge(change, REPO_BESSER)
    assert lab_workspace.run_git(REPO_BESSER, "rev-parse", "HEAD", change=change) == before
    with pytest.raises(LabWorkspaceError, match="is a draft"):
        lab_workspace.begin_base_merge(lab_workspace.create_draft(owner="ana"), REPO_BESSER)
    assert not lab_workspace.merge_in_progress(lab_workspace.create_draft(owner="ana"), REPO_BESSER)
