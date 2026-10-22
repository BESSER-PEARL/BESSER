"""The hourly temp-dir sweep recognises every prefix the backend creates."""

import os
import time

import pytest

from besser.utilities.web_modeling_editor.backend.constants import constants
from besser.utilities.web_modeling_editor.backend.services import cleanup

_PREFIX_NAMES = [
    "TEMP_DIR_PREFIX",
    "AGENT_TEMP_DIR_PREFIX",
    "CSV_TEMP_DIR_PREFIX",
    "LLM_TEMP_DIR_PREFIX",
    "USER_PROFILE_TEMP_DIR_PREFIX",
]


@pytest.mark.parametrize("name", _PREFIX_NAMES)
def test_every_temp_prefix_constant_is_swept(name):
    prefix = getattr(constants, name)
    assert prefix.startswith("besser_")
    assert prefix in constants.TEMP_DIR_PREFIXES
    assert prefix in cleanup._BESSER_TEMP_PREFIXES


def test_cleanup_removes_stale_current_and_legacy_dirs_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cleanup, "LLM_RUN_WORKSPACE_ROOT", None)

    old = time.time() - 2 * 3600
    stale = [
        tmp_path / f"{constants.USER_PROFILE_TEMP_DIR_PREFIX}abc_x",
        tmp_path / f"{constants.AGENT_TEMP_DIR_PREFIX}abc_x",
        tmp_path / "user_profile_legacy_x",  # pre-rename prefix
    ]
    fresh = tmp_path / f"{constants.USER_PROFILE_TEMP_DIR_PREFIX}fresh_x"
    unrelated = tmp_path / "someone_elses_dir"
    for path in (*stale, fresh, unrelated):
        path.mkdir()
    for path in (*stale, unrelated):
        os.utime(path, (old, old))

    cleanup.cleanup_old_temp_files(max_age_hours=1)

    for path in stale:
        assert not path.exists(), path.name
    assert fresh.is_dir()
    assert unrelated.is_dir()
