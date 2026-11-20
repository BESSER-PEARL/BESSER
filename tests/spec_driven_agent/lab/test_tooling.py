"""The lab agent's tool executor: which tools a step offers, refusals, bounds, git hygiene."""

import json
import os
import subprocess
import sys
import time

import pytest

from besser.spec_driven_agent.agent.tools import INVALID_ARGUMENTS_KEY
from besser.spec_driven_agent.execution import sandbox, shell_session
from besser.spec_driven_agent.execution.sandbox import SandboxedCommand, SandboxMounts
from besser.spec_driven_agent.lab.contracts import ExtraTool
from besser.spec_driven_agent.lab.tooling import (
    MAX_RESULT_CHARS,
    READ_TOOLS,
    RUN_RECORD_PATTERN,
    SHELL_TOOLS,
    STATE_TOOLS,
    WRITE_TOOLS,
    LabToolExecutor,
    bound_result,
    exclude_run_records,
)
from tests.spec_driven_agent.lab.test_checks_confined import assert_confined, lab_private_folders, make_change


def _extra(name, handler, writes=True):
    schema = {"name": name, "description": name, "input_schema": {"type": "object", "properties": {}}}
    return ExtraTool(schema=schema, handler=handler, writes=writes)


@pytest.fixture
def executors():
    made = []

    def make(workspace, **kwargs):
        executor = LabToolExecutor(str(workspace), **kwargs)
        made.append(executor)
        return executor
    yield make
    for executor in made:
        executor.close()


def _names(executor):
    return {schema["name"] for schema in executor.schemas}


def test_build_step_offers_file_shell_and_state_tools(tmp_path, executors):
    executor = executors(tmp_path)
    assert _names(executor) == READ_TOOLS | WRITE_TOOLS | SHELL_TOOLS | STATE_TOOLS
    assert all({"name", "description", "input_schema"} <= set(schema) for schema in executor.schemas)


def test_without_shell_run_command_is_neither_offered_nor_run(tmp_path, executors):
    executor = executors(tmp_path, allow_shell=False)
    assert "run_command" not in _names(executor)
    text, is_error = executor.execute("run_command", {"command": "echo hi > out.txt"})
    assert is_error and "Unknown tool" in text
    assert not (tmp_path / "out.txt").exists()


def test_read_only_step_offers_reads_and_non_writing_extras(tmp_path, executors):
    plan = _extra("submit_plan", lambda args: "ok", writes=False)
    checks = _extra("run_checks", lambda args: "ok", writes=True)
    executor = executors(tmp_path, read_only=True, extra_tools=[plan, checks])
    assert _names(executor) == READ_TOOLS | STATE_TOOLS | {"submit_plan"}


def test_read_only_step_answers_writes_and_shell_with_an_error(tmp_path, executors):
    called = []
    writer = _extra("scaffold", lambda args: called.append(args) or "written", writes=True)
    executor = executors(tmp_path, read_only=True, extra_tools=[writer])
    for name, args in (("write_file", {"path": "new.py", "content": "x = 1\n"}),
                       ("delete_file", {"path": "new.py"}),
                       ("run_command", {"command": "echo hi > out.txt"}),
                       ("scaffold", {})):
        text, is_error = executor.execute(name, args)
        assert is_error and "read-only" in text, name
    assert not (tmp_path / "new.py").exists() and not (tmp_path / "out.txt").exists()
    assert called == []


def test_tools_the_step_does_not_offer_are_refused(tmp_path, executors):
    (tmp_path / "requirements.txt").write_text("requests\n")
    executor = executors(tmp_path)
    for name in ("install_dependencies", "generate_django", "validate_app", "submit_plan"):
        text, is_error = executor.execute(name, {})
        assert is_error and "Unknown tool" in text, name


def test_builtin_tools_run_in_the_workspace(tmp_path, executors):
    (tmp_path / "a.py").write_text("x = 1\n")
    executor = executors(tmp_path)
    text, is_error = executor.execute("read_file", {"path": "a.py"})
    assert not is_error and "x = 1" in json.loads(text)["content"]
    text, is_error = executor.execute("modify_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"})
    assert not is_error
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    text, is_error = executor.execute("read_file", {"path": "missing.py"})
    assert is_error


def test_the_file_tools_cannot_reach_the_labs_record_of_the_change(tmp_path, executors):
    # The state folder is <workspace root>-state, outside the change's folder; notes reach every later prompt.
    change = tmp_path / "work" / "changes" / "abcdef012345"
    (change / "besser").mkdir(parents=True)
    notes = tmp_path / "work-state" / "changes" / "abcdef012345" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("## Plan\n", encoding="utf-8")
    executor = executors(change)
    outside = "../../../work-state/changes/abcdef012345/notes.md"

    for name, arguments in (("write_file", {"path": outside, "content": "x"}),
                            ("delete_file", {"path": "besser/../" + outside}),
                            ("read_file", {"path": outside})):
        text, is_error = executor.execute(name, arguments)
        assert is_error, (name, text)
        assert "## Plan" not in text
    assert notes.read_text(encoding="utf-8") == "## Plan\n"
    # A repository folder named like phase 1's state folder is an ordinary folder.
    assert executor.execute("write_file", {"path": "besser/.lab/x.txt", "content": "ok"})[1] is False


def test_the_shell_runs_with_the_mounts_it_is_given(tmp_path, executors, monkeypatch):
    """The lab's policy reaches the agent's shell: its sandbox gets exactly these mount rules."""
    planned = []

    def plan(command, **kwargs):
        planned.append(kwargs)
        return SandboxedCommand(command, isinstance(command, str), "unconfined-platform")

    monkeypatch.setattr(shell_session, "sandboxed_command", plan)
    mounts = SandboxMounts(writable=(str(tmp_path / "besser"),), read_only=(str(tmp_path / "git"),),
                           masked=(str(tmp_path / "state"),))
    (tmp_path / "besser").mkdir()
    executor = executors(tmp_path, sandbox_mounts=mounts)

    text, is_error = executor.execute("run_command", {"command": "echo confined"})

    assert not is_error, text
    assert [(p["writable"], p["read_only"], p["masked"]) for p in planned] == [
        (mounts.writable, mounts.read_only, mounts.masked)]


@pytest.mark.parametrize("read_only", [False, True])
def test_the_file_tools_cannot_reach_a_git_directory(tmp_path, executors, read_only):
    # A filter or textconv planted in .git/config (+ info/attributes) runs when the lab's own git
    # commands (diff, add, commit) read that config, outside the sandbox.
    git_dir = tmp_path / "besser" / ".git"
    (git_dir / "info").mkdir(parents=True)
    (git_dir / "config").write_text("[core]\n", encoding="utf-8")
    module = tmp_path / "besser" / ".git" / "modules" / "frontend"
    module.mkdir(parents=True)
    (module / "config").write_text("[core]\n", encoding="utf-8")
    executor = executors(tmp_path, read_only=read_only)

    attempts = [("read_file", {"path": "besser/.git/config"})]
    if not read_only:
        attempts += [
            ("write_file", {"path": "besser/.git/config", "content": "[filter \"x\"]\n\tclean = sh -c id\n"}),
            ("write_file", {"path": "besser/./.git/info/attributes", "content": "* filter=x\n"}),
            ("modify_file", {"path": "besser/.git/modules/frontend/config", "old_text": "[core]",
                             "new_text": "[diff \"x\"]"}),
            ("delete_file", {"path": "besser/src/../.git/config"}),
            ("write_file", {"path": "besser/utilities/frontend/.git", "content": "gitdir: /elsewhere\n"}),
        ]
    for name, arguments in attempts:
        text, is_error = executor.execute(name, arguments)
        assert is_error and text.startswith(f"{name} refused: .git"), (name, text)
    assert (git_dir / "config").read_text(encoding="utf-8") == "[core]\n"
    assert (module / "config").read_text(encoding="utf-8") == "[core]\n"
    assert not (git_dir / "info" / "attributes").exists()
    assert not (tmp_path / "besser" / "utilities").exists()
    # A name that merely contains ".git" is an ordinary file.
    (tmp_path / "besser" / ".gitignore").write_text("venv/\n", encoding="utf-8")
    assert executor.execute("read_file", {"path": "besser/.gitignore"})[1] is False


def test_extra_tool_results_and_refusals(tmp_path, executors):
    received = []

    def plan(args):
        received.append(args)
        if not args.get("title"):
            raise ValueError("title is required")
        return "plan recorded"

    def broken(args):
        raise RuntimeError("disk on fire")

    executor = executors(tmp_path, extra_tools=[_extra("submit_plan", plan, writes=False), _extra("broken", broken)])
    assert executor.execute("submit_plan", {"title": "Add a generator"}) == ("plan recorded", False)
    text, is_error = executor.execute("submit_plan", {})
    assert is_error and text == "submit_plan refused: title is required"
    text, is_error = executor.execute("broken", {})
    assert is_error and "disk on fire" in text
    text, is_error = executor.execute("submit_plan", {INVALID_ARGUMENTS_KEY: "Expecting ',' delimiter"})
    assert is_error and "well-formed JSON" in text
    assert received == [{"title": "Add a generator"}, {}]


def test_an_extra_tool_cannot_shadow_a_builtin_one(tmp_path):
    with pytest.raises(ValueError, match="read_file"):
        LabToolExecutor(str(tmp_path), extra_tools=[_extra("read_file", lambda args: "")])


def test_a_long_result_is_cut_with_a_note_on_reading_the_rest(tmp_path, executors):
    executor = executors(tmp_path, extra_tools=[_extra("dump", lambda args: "y" * (MAX_RESULT_CHARS + 500))])
    text, is_error = executor.execute("dump", {})
    assert not is_error
    assert text.startswith("y" * MAX_RESULT_CHARS)
    assert "500 of 80,500 characters not shown" in text and "offset and limit" in text
    assert bound_result("short") == "short"


def test_a_whole_read_of_the_largest_readable_file_is_not_cut(tmp_path, executors):
    # Quotes and backslashes double in the JSON result; read_file still stops at 60,000 characters.
    line = 'print("a \\"quoted\\" \\\\ path")  # ' + "z" * 40 + "\n"
    (tmp_path / "big.py").write_text(line * (60_000 // len(line)))
    executor = executors(tmp_path)
    text, is_error = executor.execute("read_file", {"path": "big.py"})
    assert not is_error and "result cut" not in text
    assert json.loads(text)["content"]


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def test_command_logs_stay_out_of_git_status(tmp_path, executors):
    repo = tmp_path / "besser"
    repo.mkdir()
    _git(repo, "init", "-q")
    executor = executors(repo)
    # More output than a result holds: the executor spills the full log into .besser_command_output/.
    text, is_error = executor.execute("run_command", {"command": f'"{sys.executable}" -c "print(\'x\' * 20000)"'})
    assert not is_error, text
    spilled = json.loads(text)["full_output_path"]
    assert (repo / spilled).is_file() and spilled.startswith(".besser_")
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_exclude_is_written_once_and_only_for_a_writing_step(tmp_path, executors):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text("# keep me")  # no final newline
    executors(repo, read_only=True)
    assert exclude.read_text() == "# keep me"
    executors(repo)
    executors(repo)
    assert exclude.read_text() == f"# keep me\n{RUN_RECORD_PATTERN}\n"


def test_exclude_follows_a_gitdir_file_and_a_worktrees_common_dir(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q")
    (main / "f.txt").write_text("f\n")
    _git(main, "add", "f.txt")
    _git(main, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    _git(main, "worktree", "add", "-q", str(tmp_path / "wt"))
    exclude_run_records(str(tmp_path / "wt"))
    (tmp_path / "wt" / ".besser_command_output").mkdir()
    (tmp_path / "wt" / ".besser_command_output" / "001.log").write_text("log\n")
    assert _git(tmp_path / "wt", "status", "--porcelain", "--untracked-files=all") == ""
    assert RUN_RECORD_PATTERN in (main / ".git" / "info" / "exclude").read_text()


def test_a_workspace_that_is_not_a_checkout_is_left_alone(tmp_path):
    exclude_run_records(str(tmp_path))
    assert os.listdir(tmp_path) == []


@pytest.fixture
def confined_layout(tmp_path):
    """The layout of D1 test 7, as the checks' side builds it."""
    with lab_private_folders(make_change(tmp_path)) as layout:
        yield layout


def _probe_from_the_shell(layout, executors) -> str:
    executor = executors(layout["trees"], sandbox_mounts=layout["mounts"])
    text, is_error = executor.execute("run_command", {"command": f'cd besser && "{sys.executable}" probe.py'})
    assert not is_error, text
    return json.loads(text)["stdout"]


@pytest.mark.skipif(sandbox.sandbox_selftest_error() is not None, reason="needs a working bubblewrap (Linux)")
def test_the_shell_cannot_reach_git_dirs_lab_state_or_sessions(confined_layout, executors):
    """D1 test 7, shell side: the agent's shell under the lab's policy."""
    assert_confined(_probe_from_the_shell(confined_layout, executors), confined_layout["config"], host_ports=False)


@pytest.mark.skipif(sandbox.sandbox_selftest_error() is not None, reason="needs a working bubblewrap (Linux)")
def test_the_shell_cannot_reach_host_ports(confined_layout, executors, monkeypatch):
    """D1 test 7, shell side, with deploy's BESSER_LLM_SHELL_NETWORK=private."""
    monkeypatch.setenv(sandbox.SHELL_NETWORK_ENV, "private")
    if not sandbox.shell_network_is_private():
        pytest.skip("needs pasta and /dev/net/tun")

    assert_confined(_probe_from_the_shell(confined_layout, executors), confined_layout["config"])


def test_time_left_caps_a_shell_command(tmp_path, executors):
    executor = executors(tmp_path, time_left=lambda: 1.0)
    started = time.monotonic()
    text, is_error = executor.execute(
        "run_command", {"command": f'"{sys.executable}" -c "import time; time.sleep(20)"', "timeout": 600})
    assert is_error and "timed out after 1 seconds" in text
    assert time.monotonic() - started < 15
