"""A run_command result must not read as done when part of the line did not run.

The shell keeps the directory of an earlier ``cd``, and ``a; b`` returns the
exit code of ``b``. A cleanup run from the wrong directory came back
``exit_code: 0, success: true`` with the failure only in stderr and the
directory in the last field of the result.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.execution.shell_session import ShellResult

_RUN = "besser.spec_driven_agent.execution.shell_session.ShellSession.run"
_ENOENT = "python: can't open file '/work/frontend/seed.py': [Errno 2] No such file or directory\n"


def _executor(tmp_path) -> ToolExecutor:
    return ToolExecutor(workspace=str(tmp_path), domain_model=None)


def _run(executor, command, *, returncode=0, stderr="", **args):
    result = ShellResult(returncode=returncode, stdout="", stderr=stderr)
    with patch(_RUN, return_value=result):
        return executor._run_command({"command": command, **args})


def test_a_command_started_in_a_carried_over_directory_says_so_first(tmp_path):
    executor = _executor(tmp_path)
    os.makedirs(tmp_path / "frontend")
    executor._shell.cwd = str(tmp_path / "frontend")

    result = _run(executor, "python seed.py")

    assert next(iter(result)) == "ran_in", "the directory must be read before the outcome"
    assert result["ran_in"].startswith("frontend")
    assert "cd" in result["ran_in"]


def test_a_command_in_the_workspace_root_or_an_explicit_directory_has_no_such_field(tmp_path):
    executor = _executor(tmp_path)
    os.makedirs(tmp_path / "frontend")

    assert "ran_in" not in _run(executor, "python seed.py")
    assert "ran_in" not in _run(executor, "npm test", working_dir="frontend")


@pytest.mark.parametrize("command", [
    "python seed.py; rm -rf dist",
    "python seed.py ; echo done",
    "python seed.py\nrm -rf dist",
])
def test_a_failure_hidden_by_a_later_command_is_named(tmp_path, command):
    result = _run(_executor(tmp_path), command, stderr=_ENOENT)

    assert result["exit_code"] == 0 and result["success"] is True, "the shell's own answer is kept"
    assert "last command" in result["note"] and "&&" in result["note"]


@pytest.mark.parametrize("command, stderr", [
    ("python seed.py && rm -rf dist", _ENOENT),            # && already stops at the failure
    ("python seed.py", _ENOENT),                           # one command: the exit code is its own
    ("echo 'a; b'", _ENOENT),                              # the ; is text
    ("npm ci; npm run build", "npm warn deprecated x\n"),  # stderr reports no failure
    ("npm ci; npm run build", ""),
    ('python seed.py ""', _ENOENT),                       # an empty argument is no second command
    ("sed -i '' s/a/b/ seed.py", _ENOENT),
    ("python seed.py\n", _ENOENT),                        # a trailing new line
    ("python seed.py;", _ENOENT),
    ("\npython seed.py ;\n\n", _ENOENT),
])
def test_other_lines_get_no_note(tmp_path, command, stderr):
    assert "note" not in _run(_executor(tmp_path), command, stderr=stderr)


def test_a_failing_line_keeps_its_own_report(tmp_path):
    result = _run(_executor(tmp_path), "python seed.py; false", returncode=1, stderr=_ENOENT)

    assert result["success"] is False and "note" not in result
