"""The run-record guard of ``run_command`` stays bounded and never costs the command result.

* A command that plants a huge ``.besser_recipe.json`` made the harness read it whole into
  memory (300 MB for a 300 MB file); a record above the cap is now judged by its stat alone.
* A restore that failed (the checkpoint swapped for a directory, a read-only workspace)
  replaced the whole result with an error, so the model lost the exit code and stdout.
"""

import json
import os

import pytest

from besser.spec_driven_agent.agent import tool_executor
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.execution.shell_session import ShellResult

CHECKPOINT = ".besser_checkpoint.json"
RECIPE = ".besser_recipe.json"
SMALL_CAP = 1024


def _run(tmp_path, command_effect) -> dict:
    ex = ToolExecutor(workspace=str(tmp_path), allow_shell=True)

    def run(command, *, working_dir, timeout, adopt_state=True):
        command_effect()
        return ShellResult(returncode=3, stdout="command output", stderr="")

    ex._shell.run = run
    return json.loads(ex.execute("run_command", {"command": "pytest"}))


@pytest.fixture
def reads(monkeypatch):
    """Paths the guard opened for reading, with a small cap in force."""
    monkeypatch.setattr(tool_executor, "_GUARDED_RECORD_MAX_BYTES", SMALL_CAP, raising=False)
    opened: list[str] = []
    real = tool_executor.open_plain

    def spy(path, *args, **kwargs):
        opened.append(os.path.basename(os.fspath(path)))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(tool_executor, "open_plain", spy)
    return opened


def test_a_planted_oversized_record_is_removed_without_being_read(tmp_path, reads):
    result = _run(tmp_path, lambda: (tmp_path / RECIPE).write_bytes(b"x" * (4 * SMALL_CAP)))

    assert not (tmp_path / RECIPE).exists()
    assert result["run_records_restored"] == [RECIPE]
    assert RECIPE not in reads


def test_a_record_grown_past_the_cap_is_restored_without_reading_the_new_one(tmp_path, reads):
    (tmp_path / CHECKPOINT).write_bytes(b'{"turn": 3}')
    result = _run(tmp_path, lambda: (tmp_path / CHECKPOINT).write_bytes(b"x" * (4 * SMALL_CAP)))

    assert (tmp_path / CHECKPOINT).read_bytes() == b'{"turn": 3}'
    assert result["run_records_restored"] == [CHECKPOINT]
    assert reads.count(CHECKPOINT) == 1  # the snapshot before the command, not the planted file


def test_a_legitimate_record_over_the_cap_is_not_copied(tmp_path, reads):
    (tmp_path / CHECKPOINT).write_bytes(b"y" * (2 * SMALL_CAP))

    untouched = _run(tmp_path, lambda: None)
    assert "run_records_not_restored" not in untouched
    assert CHECKPOINT not in reads

    changed = _run(tmp_path, lambda: (tmp_path / CHECKPOINT).write_bytes(b"z" * (3 * SMALL_CAP)))
    assert changed["run_records_not_restored"] == [CHECKPOINT]
    assert "no copy" in changed["run_records_not_restored_note"]
    assert CHECKPOINT not in reads


def test_a_record_swapped_for_a_directory_keeps_the_command_result(tmp_path):
    (tmp_path / CHECKPOINT).write_bytes(b'{"turn": 3}')

    def to_directory():
        (tmp_path / CHECKPOINT).unlink()
        (tmp_path / CHECKPOINT).mkdir()
        (tmp_path / CHECKPOINT / "planted.txt").write_text("x")

    result = _run(tmp_path, to_directory)

    assert result["exit_code"] == 3
    assert result["stdout"] == "command output"
    assert result["run_records_not_restored"] == [CHECKPOINT]
    assert CHECKPOINT in result["run_records_not_restored_note"]
    assert (tmp_path / CHECKPOINT).is_dir()  # left in place: no link-safe tree removal exists


def test_a_restore_that_cannot_write_keeps_the_command_result(tmp_path, monkeypatch):
    (tmp_path / CHECKPOINT).write_bytes(b'{"turn": 3}')

    def read_only(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tool_executor, "write_atomic_plain_bytes", read_only)
    result = _run(tmp_path, lambda: (tmp_path / CHECKPOINT).write_bytes(b"{}"))

    assert result["exit_code"] == 3
    assert result["stdout"] == "command output"
    assert result["run_records_not_restored"] == [CHECKPOINT]
    assert "run_records_restored" not in result


def test_a_link_planted_where_no_record_was_is_removed(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_bytes(b"outside")
    try:
        os.symlink(outside, tmp_path / "probe-link")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")
    os.remove(tmp_path / "probe-link")

    result = _run(tmp_path, lambda: os.symlink(outside, tmp_path / CHECKPOINT))

    assert not os.path.lexists(tmp_path / CHECKPOINT)
    assert outside.read_bytes() == b"outside"
    assert result["run_records_restored"] == [CHECKPOINT]
