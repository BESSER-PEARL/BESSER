"""A shell command cannot rewrite the checkpoint or the recipe.

The file tools refuse every ``.besser_*`` run record, but the sandbox binds the whole run
directory read-write, so ``run_command`` could still rewrite ``.besser_checkpoint.json``
(replayed into the conversation on resume) or ``.besser_recipe.json``. The command here is
simulated: the real one needs the Linux namespace sandbox.
"""

import json

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.execution.shell_session import ShellResult

CHECKPOINT = ".besser_checkpoint.json"
RECIPE = ".besser_recipe.json"


def _executor(tmp_path, command_effect):
    ex = ToolExecutor(workspace=str(tmp_path), allow_shell=True)

    def run(command, *, working_dir, timeout, adopt_state=True):
        command_effect()
        return ShellResult(returncode=0, stdout="", stderr="")

    ex._shell.run = run
    return ex


@pytest.mark.parametrize("record", [CHECKPOINT, RECIPE])
def test_a_rewritten_record_is_restored_and_reported(tmp_path, record):
    original = b'{"turn": 3}'
    (tmp_path / record).write_bytes(original)
    ex = _executor(tmp_path, lambda: (tmp_path / record).write_text('{"turn": 99}'))

    result = json.loads(ex.execute("run_command", {"command": "true"}))

    assert (tmp_path / record).read_bytes() == original
    assert record in result["run_records_restored"]
    assert record in result["run_records_note"]


def test_a_deleted_record_is_restored(tmp_path):
    (tmp_path / CHECKPOINT).write_bytes(b"{}")
    ex = _executor(tmp_path, lambda: (tmp_path / CHECKPOINT).unlink())

    result = json.loads(ex.execute("run_command", {"command": "true"}))

    assert (tmp_path / CHECKPOINT).read_bytes() == b"{}"
    assert result["run_records_restored"] == [CHECKPOINT]


def test_a_planted_record_is_removed(tmp_path):
    ex = _executor(tmp_path, lambda: (tmp_path / CHECKPOINT).write_text('{"turn": 1}'))

    result = json.loads(ex.execute("run_command", {"command": "true"}))

    assert not (tmp_path / CHECKPOINT).exists()
    assert result["run_records_restored"] == [CHECKPOINT]


def test_a_command_that_leaves_the_records_alone_reports_nothing(tmp_path):
    (tmp_path / CHECKPOINT).write_bytes(b"{}")
    ex = _executor(tmp_path, lambda: (tmp_path / "app.py").write_text("x = 1\n"))

    result = json.loads(ex.execute("run_command", {"command": "true"}))

    assert "run_records_restored" not in result
    assert result["success"] is True
