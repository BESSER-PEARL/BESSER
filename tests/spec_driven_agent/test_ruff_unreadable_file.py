"""A file ruff cannot read (E902) is a lint that did not run, not a lint warning.

Run r9 reported ``ruff: ...:1:1: E902 The system cannot find the path specified``
as one ordinary warning while no Python file had been linted. ruff exits 0 on
E902 under ``--exit-zero``, so the line looked like a finding. An undefined name
in the unread file is never checked, so the run must say the check was SKIPPED.
"""
from __future__ import annotations

import shutil

import pytest

from besser.spec_driven_agent.validation.issues import _DID_NOT_RUN_NOTE_RE, _classify_issue
from besser.spec_driven_agent.validation.toolchain import _collect_ruff_issues


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_an_unreadable_python_file_is_reported_as_not_linted(tmp_path):
    # Not UTF-8: ruff cannot read it, so the undefined `y` inside is never seen.
    (tmp_path / "bad.py").write_bytes(b'x = "\xff\xfe"\nprint(y)\n')
    (tmp_path / "ok.py").write_text("print(z)\n", encoding="utf-8")

    issues, _ = _collect_ruff_issues(str(tmp_path), [], frozenset(), False)

    assert not any("E902" in issue and issue.startswith("ruff:") for issue in issues), issues
    skipped = [issue for issue in issues if _DID_NOT_RUN_NOTE_RE.match(issue)]
    assert len(skipped) == 1 and "bad.py" in skipped[0], issues
    assert _classify_issue(skipped[0]).severity == "warning"
    # The readable file is still linted.
    assert any("F821" in issue and "ok.py" in issue for issue in issues), issues
