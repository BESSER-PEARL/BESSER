"""Phase 3 runs ruff with a fixed defect-only rule set that generated code cannot switch off.

``--isolated`` already stops an app's ruff.toml from ignoring F821; a
``# noqa: F811`` in the generated file did the same thing line by line, so a
redefinition that silently replaces a route handler passed as clean. The
selected rules are pinned here because three tests guarding "cosmetic codes
stay style" were deleted when the selection replaced ruff's defaults.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.toolchain import _collect_ruff_issues

_SELECTED = "E9,F5,F6,F7,F82,F811,F901,PLE0101,PLE0604,PLE1205,PLE1206"
_needs_ruff = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")


def _ruff_args(tmp_path, monkeypatch) -> list[str]:
    seen: list[list[str]] = []

    def fake_run(args, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ruff")
    monkeypatch.setattr(subprocess, "run", fake_run)
    _collect_ruff_issues(str(tmp_path), [], frozenset(), False)
    assert len(seen) == 1
    return seen[0]


def test_the_selected_rule_set_is_pinned(tmp_path, monkeypatch):
    args = _ruff_args(tmp_path, monkeypatch)
    assert args[args.index("--select") + 1] == _SELECTED
    assert "--isolated" in args


def test_a_noqa_comment_in_generated_code_is_ignored(tmp_path, monkeypatch):
    assert "--ignore-noqa" in _ruff_args(tmp_path, monkeypatch)


@_needs_ruff
def test_a_noqa_does_not_hide_a_redefined_handler(tmp_path):
    (tmp_path / "routes.py").write_text(
        "def read_booking():\n    return 1\n\n\n"
        "def read_booking():  # noqa: F811\n    return 2\n", encoding="utf-8")
    issues, _ = _collect_ruff_issues(str(tmp_path), [], frozenset(), False)
    assert any("F811" in issue for issue in issues), issues
    assert all(_classify_issue(issue).severity == "blocker" for issue in issues if "F811" in issue)


@_needs_ruff
def test_cosmetic_only_findings_are_not_reported(tmp_path):
    """An unused import (F401) and a long line (E501) are not selected: no finding, no blocker."""
    (tmp_path / "main.py").write_text(
        "import json\n\nVALUE = '" + "x" * 150 + "'\n", encoding="utf-8")
    issues, _ = _collect_ruff_issues(str(tmp_path), [], frozenset(), False)
    assert issues == []
