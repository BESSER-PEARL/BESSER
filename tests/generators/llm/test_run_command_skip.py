"""Tests for the run_command soft-skip behavior added for #4.

When the LLM invokes a shell command whose binary isn't installed in the
smart-gen container (e.g. ``ruby -c file.rb`` on an image without Ruby),
the tool used to return a real ``exit_code != 0`` with the stderr "ruby:
command not found". The LLM would dutifully surface that to the user as
if it were a failure ("I attempted to validate the Ruby syntax with
ruby -c domain.rb, but Ruby is not installed in the execution
environment.") — which the user can't act on.

The fix in tool_executor.py treats this as a soft skip: success=True
plus a ``skipped`` flag and a human-readable ``skip_reason``, so the
LLM moves on without flagging the missing runtime.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from besser.generators.llm.tool_executor import (
    ToolExecutor,
    _looks_like_command_not_found,
)


# ----------------------------------------------------------------------
# Pattern detector
# ----------------------------------------------------------------------


class TestLooksLikeCommandNotFound:
    def test_unix_command_not_found(self):
        assert _looks_like_command_not_found(
            "/bin/sh: 1: ruby: command not found\n"
        )

    def test_unix_command_not_found_alt_shell(self):
        assert _looks_like_command_not_found("bash: rustc: command not found")

    def test_windows_not_recognized(self):
        assert _looks_like_command_not_found(
            "'go' is not recognized as an internal or external command,\n"
            "operable program or batch file."
        )

    def test_no_such_file(self):
        # Some shells produce this when exec-ing an absent binary.
        assert _looks_like_command_not_found(
            "execve: /usr/local/bin/kotlin: No such file or directory"
        )

    def test_case_insensitive(self):
        assert _looks_like_command_not_found("COMMAND NOT FOUND")
        assert _looks_like_command_not_found("Command Not Found")

    def test_normal_compiler_error_is_not_match(self):
        # A real Python syntax error must NOT be treated as a missing runtime.
        # We only match the absence-of-binary fingerprints.
        assert not _looks_like_command_not_found(
            "  File \"/tmp/x.py\", line 1\n"
            "    def foo(\n"
            "           ^\n"
            "SyntaxError: unexpected EOF while parsing"
        )

    def test_empty_or_none(self):
        assert not _looks_like_command_not_found("")
        assert not _looks_like_command_not_found(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# _run_command soft-skip
# ----------------------------------------------------------------------


def _make_executor(tmp_path) -> ToolExecutor:
    """Return a ToolExecutor anchored at a real workspace dir.

    No domain model is needed for run_command tests — we only exercise
    the shell-exec path.
    """
    return ToolExecutor(
        workspace=str(tmp_path),
        domain_model=None,
        gui_model=None,
        agent_model=None,
        agent_config=None,
    )


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a CompletedProcess stub that subprocess.run can return."""
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


class TestRunCommandSoftSkip:
    def test_command_not_found_returns_soft_skip(self, tmp_path):
        ex = _make_executor(tmp_path)
        completed = _completed(
            returncode=127,
            stdout="",
            stderr="/bin/sh: 1: ruby: command not found\n",
        )
        with patch("besser.generators.llm.tool_executor.subprocess.run", return_value=completed):
            result = ex._run_command({"command": "ruby -c file.rb", "working_dir": "."})

        # The LLM sees a clean success so it doesn't escalate this to a
        # user-facing failure.
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["skipped"] is True
        assert "not installed" in result["skip_reason"]
        # stderr is scrubbed so the "ruby: command not found" string
        # doesn't leak into the LLM's context and prompt it to mention
        # the missing runtime anyway.
        assert result["stderr"] == ""

    def test_windows_not_recognized_returns_soft_skip(self, tmp_path):
        ex = _make_executor(tmp_path)
        completed = _completed(
            returncode=1,
            stdout="",
            stderr=(
                "'cargo' is not recognized as an internal or external command,\n"
                "operable program or batch file."
            ),
        )
        with patch("besser.generators.llm.tool_executor.subprocess.run", return_value=completed):
            result = ex._run_command({"command": "cargo check"})
        assert result["success"] is True
        assert result.get("skipped") is True

    def test_real_command_failure_passes_through(self, tmp_path):
        """A genuine non-zero exit with a real error message is NOT skipped."""
        ex = _make_executor(tmp_path)
        completed = _completed(
            returncode=1,
            stdout="",
            stderr=(
                "  File \"/tmp/x.py\", line 1\n"
                "    def foo(\n"
                "           ^\n"
                "SyntaxError: unexpected EOF while parsing\n"
            ),
        )
        with patch("besser.generators.llm.tool_executor.subprocess.run", return_value=completed):
            result = ex._run_command({"command": "python -c 'def foo('"})
        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "SyntaxError" in result["stderr"]
        assert "skipped" not in result

    def test_successful_run_passes_through(self, tmp_path):
        ex = _make_executor(tmp_path)
        completed = _completed(returncode=0, stdout="hello\n", stderr="")
        with patch("besser.generators.llm.tool_executor.subprocess.run", return_value=completed):
            result = ex._run_command({"command": "echo hello"})
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["stdout"] == "hello\n"
        assert "skipped" not in result
