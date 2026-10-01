"""run_command's per-command timeout.

A fixed 120 s killed legitimate long builds (`cargo build`, a first
`npm install`). The model may now ask for more, up to MAX_COMMAND_TIMEOUT, and
no command outlives the run's own deadline.
"""
import pytest

from besser.spec_driven_agent.agent import tool_executor as te
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import EXECUTION_TOOLS
from besser.spec_driven_agent.execution.shell_session import ShellResult


class _RecordingShell:
    def __init__(self, timed_out=False):
        self.timeouts = []
        self.cwd = "."
        self.timed_out = timed_out

    def run(self, command, *, working_dir, timeout, adopt_state=True):
        self.timeouts.append(timeout)
        return ShellResult(None if self.timed_out else 0, "", "", timed_out=self.timed_out)

    def relative_cwd(self):
        return "."

    def close(self):
        pass


@pytest.fixture
def executor(tmp_path):
    ex = ToolExecutor(str(tmp_path), allow_shell=True)
    ex._shell = _RecordingShell()
    return ex


def _run(ex, **args):
    return ex._run_command({"command": "echo hi", **args})


def test_default_timeout_is_unchanged(executor):
    _run(executor)
    assert executor._shell.timeouts == [te.COMMAND_TIMEOUT]


@pytest.mark.parametrize("requested, expected", [
    (300, 300),
    ("450", 450),  # some models send numbers as strings
    (5000, te.MAX_COMMAND_TIMEOUT),
    (0, 1),
    ("soon", te.COMMAND_TIMEOUT),
])
def test_requested_timeout_is_honoured_within_bounds(executor, requested, expected):
    _run(executor, timeout=requested)
    assert executor._shell.timeouts == [expected]


def test_a_command_never_outlives_the_run(executor):
    executor.time_left = lambda: 42.7
    _run(executor, timeout=600)
    _run(executor)
    assert executor._shell.timeouts == [42, 42]


def test_timeout_error_names_the_timeout_that_applied(tmp_path):
    ex = ToolExecutor(str(tmp_path), allow_shell=True)
    ex._shell = _RecordingShell(timed_out=True)
    result = ex._run_command({"command": "sleep 999", "timeout": 300})
    assert "after 300 seconds" in result["error"]


def test_install_dependencies_gets_the_install_timeout(executor, tmp_path):
    (tmp_path / "requirements.txt").write_text("six\n")
    executor._install_dependencies({})
    assert executor._shell.timeouts == [te.INSTALL_TIMEOUT]


def test_the_schema_offers_the_timeout():
    tool = next(t for t in EXECUTION_TOOLS if t["name"] == "run_command")
    prop = tool["input_schema"]["properties"]["timeout"]
    assert prop["type"] == "integer"
    assert str(te.MAX_COMMAND_TIMEOUT) in prop["description"]
    assert "timeout" not in tool["input_schema"]["required"]
