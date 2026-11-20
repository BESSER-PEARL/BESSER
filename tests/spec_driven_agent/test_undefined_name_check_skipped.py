"""Without pyflakes the release gate must say the undefined-name check did not run.

The python-contract sweep judges undefined names (including those behind a
star import, which ruff reports only as F405) with pyflakes. When pyflakes is
not importable the sweep returns nothing for them, which reads as a clean
pass. Restores the intent of the deleted
``test_phase3_star_import_names.py::test_a_missing_pyflakes_is_reported_as_not_checked``.
"""
from __future__ import annotations

import sys

import pytest

from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.write_diagnostics import (
    diagnose_written_content,
    undefined_name_check_skipped,
)

pytest.importorskip("pyflakes")


def test_a_missing_pyflakes_is_reported_as_not_checked(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyflakes.checker", None)
    notes = undefined_name_check_skipped()
    assert len(notes) == 1 and "did not run" in notes[0] and "pyflakes" in notes[0]
    assert _classify_issue(notes[0]).severity == "warning"


def test_nothing_is_reported_when_pyflakes_is_installed():
    assert undefined_name_check_skipped() == []


def test_the_per_write_path_stays_quiet_without_pyflakes(monkeypatch):
    """Per write the model is not shown a tooling note; only the gate reports it."""
    monkeypatch.setitem(sys.modules, "pyflakes.checker", None)
    assert diagnose_written_content("m.py", "x = never_defined\n") == []
