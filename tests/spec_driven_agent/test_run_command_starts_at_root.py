"""Every run_command starts in the workspace root unless it names a working_dir.

The shell kept the directory of an earlier ``cd``, while models write each
command as if from the root: in the recorded runs 6 of 229 commands failed with
``cd: <path>: No such file or directory`` after a previous ``cd``, and one
``ls <path> | wc -l`` silently answered 0 for a folder that existed. The
result already named the shell's directory before each failure, and the model
still wrote a root-relative path. Exported variables still carry over.
"""

import os

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.execution import shell_session
from besser.spec_driven_agent.execution.sandbox import SandboxedCommand


@pytest.fixture
def executor(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda argv, **_: SandboxedCommand(argv, False, "unconfined-platform"))
    os.makedirs(tmp_path / "web_app" / "frontend")
    return ToolExecutor(workspace=str(tmp_path), allow_shell=True)


def _run(executor, command, **args):
    return executor.execute_typed("run_command", {"command": command, **args}).payload


def test_a_root_relative_cd_works_after_an_earlier_cd(executor):
    assert _run(executor, "cd web_app/frontend")["exit_code"] == 0

    again = _run(executor, "cd web_app/frontend")

    assert again["exit_code"] == 0, again


def test_a_command_after_a_cd_runs_in_the_root(executor, tmp_path):
    _run(executor, "cd web_app")
    where = _run(executor, "cd" if os.name == "nt" else "pwd")

    assert where["stdout"].strip() == str(tmp_path), where
    assert "cwd" not in where and "ran_in" not in where


def test_working_dir_still_runs_one_command_elsewhere(executor, tmp_path):
    where = _run(executor, "cd" if os.name == "nt" else "pwd", working_dir="web_app/frontend")

    assert where["stdout"].strip() == str(tmp_path / "web_app" / "frontend"), where


def test_exported_variables_still_carry_over(executor):
    if os.name == "nt":
        first, second = "set GREETING=hello", "echo %GREETING%"
    else:
        first, second = "export GREETING=hello", "echo $GREETING"
    _run(executor, first)

    assert _run(executor, second)["stdout"].strip() == "hello"
