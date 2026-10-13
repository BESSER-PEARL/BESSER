"""The lint check reports on the app, not on files the harness adds to the
workspace (live run 7d188d29: all 20 warnings were in .besser_probe.py)."""
import shutil

import pytest

from besser.spec_driven_agent.validation.toolchain import _collect_ruff_issues

pytestmark = pytest.mark.skipif(shutil.which("ruff") is None, reason="needs the ruff binary")

_LINTY = "import os\nimport sys\n\n\ndef f():\n    return undefined_name\n"


def _snapshot(tmp_path):
    from besser.spec_driven_agent.validation.toolchain import _SNAPSHOT_DIR
    snap = tmp_path / _SNAPSHOT_DIR
    snap.mkdir()
    (snap / "old.py").write_text(_LINTY, encoding="utf-8")


def test_harness_probe_script_is_not_linted(tmp_path):
    _snapshot(tmp_path)
    (tmp_path / ".besser_probe.py").write_text(_LINTY, encoding="utf-8")
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    issues, _ = _collect_ruff_issues(str(tmp_path), [], frozenset(), False)
    assert not [i for i in issues if ".besser_probe.py" in i], issues
    # The existing snapshot exclusion must survive (a separate
    # --extend-exclude silently replaced it).
    assert not [i for i in issues if "old.py" in i], issues


def test_app_files_are_still_linted(tmp_path):
    (tmp_path / ".besser_probe.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(_LINTY, encoding="utf-8")
    issues, _ = _collect_ruff_issues(str(tmp_path), [], frozenset(), False)
    assert any("app.py" in i for i in issues), issues
