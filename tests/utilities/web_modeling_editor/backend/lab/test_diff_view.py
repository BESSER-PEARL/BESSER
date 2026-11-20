"""The diff of a lab change as ``GET /lab/diff`` sends it: redacted, capped, per file."""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT
from besser.spec_driven_agent.lab.workspace import LabWorkspaceError
from besser.utilities.web_modeling_editor.backend.services.lab import diff_view
from besser.utilities.web_modeling_editor.backend.services.lab.diff_view import (
    MAX_FILE_PATCH_BYTES,
    diff_response,
)
from tests.utilities.web_modeling_editor.backend.lab.conftest import BRANCH, LOGIN

GITHUB_TOKEN = "ghp_" + "R7tY2uI9oP4aS6dF1gH3jK5lZ8xC0vB2nM4q"


def _files(response, repo: str) -> dict:
    return {f.path: f for f in next(r for r in response.repos if r.repo == repo).files}


def test_diff_redacts_a_github_token_in_an_added_line(lab):
    lab.write(REPO_BESSER, "besser/client.py", f'TOKEN = "{GITHUB_TOKEN}"\nTIMEOUT = 5\n')

    response = diff_response(lab.workspace, lab.change)

    entry = _files(response, REPO_BESSER)["besser/client.py"]
    assert GITHUB_TOKEN not in entry.patch
    assert '+TOKEN = "[REDACTED]"' in entry.patch and "+TIMEOUT = 5" in entry.patch
    assert entry.redactions == 1 and entry.omitted is None
    assert (entry.status, entry.additions, entry.deletions, entry.binary) == ("added", 2, 0, False)


def test_a_secret_env_file_is_redacted_by_name_and_a_template_is_not(lab):
    lab.write(REPO_MODELING_AGENT, ".env", "DB_PASSWORD=hunter2-correct-horse\nDEBUG=1\n")
    lab.write(REPO_MODELING_AGENT, ".env.example", "API_KEY=your-key-here\n")

    files = _files(diff_response(lab.workspace, lab.change, repo=REPO_MODELING_AGENT), REPO_MODELING_AGENT)

    assert "hunter2" not in files[".env"].patch and "+DB_PASSWORD=[REDACTED]" in files[".env"].patch
    assert "+DEBUG=1" in files[".env"].patch and files[".env"].redactions == 1
    assert "+API_KEY=your-key-here" in files[".env.example"].patch and files[".env.example"].redactions == 0


def test_repositories_come_in_merge_order_with_their_branch_and_base(lab):
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")

    response = diff_response(lab.workspace, lab.change)

    assert response.changeId == lab.change.id
    assert [(r.repo, r.branch, r.base) for r in response.repos] == [
        (REPO_FRONTEND, BRANCH, "develop"), (REPO_BESSER, BRANCH, "development"),
        (REPO_MODELING_AGENT, BRANCH, "develop")]
    assert [f.path for f in _files(response, REPO_BESSER).values()] == ["besser/note.py"]


def test_diff_omits_a_file_over_the_cap_and_serves_it_alone(lab):
    big = "".join(f"line {n:06d} of a generated lock file\n" for n in range(MAX_FILE_PATCH_BYTES // 30))
    lab.write(REPO_BESSER, "package-lock.json", big)
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")

    files = _files(diff_response(lab.workspace, lab.change), REPO_BESSER)

    assert files["package-lock.json"].patch is None and files["package-lock.json"].omitted == "too_large"
    assert files["package-lock.json"].additions == big.count("\n")
    assert files["besser/note.py"].patch is not None

    alone = diff_response(lab.workspace, lab.change, repo=REPO_BESSER, path="package-lock.json")

    assert [r.repo for r in alone.repos] == [REPO_BESSER]
    [entry] = alone.repos[0].files
    assert entry.omitted is None and entry.patch.count("\n+line ") == big.count("\n")


def test_files_past_the_response_cap_are_omitted_as_response_full(lab, monkeypatch):
    lab.write(REPO_BESSER, "besser/a.py", "A = 1\n" * 50)
    lab.write(REPO_BESSER, "besser/b.py", "B = 1\n" * 50)
    # Room for one of the two patches (about 490 bytes each).
    monkeypatch.setattr(diff_view, "MAX_DIFF_RESPONSE_BYTES", 700)

    files = _files(diff_response(lab.workspace, lab.change), REPO_BESSER)

    assert files["besser/a.py"].patch is not None
    assert files["besser/b.py"].patch is None and files["besser/b.py"].omitted == "response_full"


def test_a_binary_file_has_no_patch(lab):
    (lab.path(REPO_BESSER) / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00binary")

    entry = _files(diff_response(lab.workspace, lab.change), REPO_BESSER)["logo.png"]

    assert (entry.binary, entry.patch, entry.omitted, entry.additions) == (True, None, "binary", None)


def test_a_draft_has_no_diff(lab):
    draft = lab.workspace.create_draft(owner=LOGIN)

    assert diff_response(lab.workspace, draft).repos == []


def test_diff_of_a_closed_change_is_refused_with_a_reason(lab):
    lab.workspace.close(lab.change, outcome="closed")

    with pytest.raises(LabWorkspaceError, match="its branches were deleted when it closed; the diff is on the pull"):
        diff_response(lab.workspace, lab.change)


def test_a_file_is_named_by_its_repository_and_path():
    with pytest.raises(ValueError):
        diff_response(None, None, path="README.md")
