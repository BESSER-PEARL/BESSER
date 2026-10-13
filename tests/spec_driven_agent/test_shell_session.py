"""The run's shell session, with bubblewrap faked.

"Faked" means ``sandboxed_command`` hands back the supervisor's own argv as a
``bwrap``-mode plan: the real supervisor, channel, per-command process groups,
registry and reaper run, just not inside namespaces. That needs POSIX (pipes
in ``selectors``, ``killpg``), so those tests skip on Windows. The real
sandbox is exercised in ``test_shell_session_sandboxed.py``.

Without namespaces a background process outlives its session, so every test
that starts one records its pid and kills it itself.
"""

import os
import signal
import sys
import threading
import time

import pytest

from besser.spec_driven_agent.agent import tool_executor as te
from besser.spec_driven_agent.agent.runbook import runbook_section
from besser.spec_driven_agent.agent.tools import get_tools_for
from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.execution import shell_session
from besser.spec_driven_agent.execution.process import FLOOD_NOTE
from besser.spec_driven_agent.execution.sandbox import (
    SANDBOX_POLICY_ENV,
    SandboxUnavailable,
    SandboxedCommand,
    run_confined,
)
from besser.spec_driven_agent.execution.shell_session import ShellSession

posix_only = pytest.mark.skipif(os.name == "nt", reason="the supervisor is POSIX-only")
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux mount plan")


@pytest.fixture
def fake_bwrap(monkeypatch):
    """The supervisor runs as if it were the sandbox's init. Returns the list
    of sandboxes started."""
    started = []

    class _Counted(shell_session._SessionProcess):
        def __init__(self, argv, cwd, **kwargs):
            started.append(argv)
            super().__init__(argv, cwd, **kwargs)

    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda argv, **_: SandboxedCommand(argv, False, "bwrap"))
    monkeypatch.setattr(shell_session, "_SessionProcess", _Counted)
    return started


@pytest.fixture
def pids():
    """Pids a test's background processes wrote; killed at the end."""
    files = []
    yield files
    for path in files:
        try:
            os.kill(int(open(path).read().strip()), signal.SIGKILL)
        except (OSError, ValueError):
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie is dead for our purposes
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def _wait_for(predicate, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _run(session, command, timeout=20, **kwargs):
    return session.run(command, working_dir=kwargs.pop("working_dir", None),
                       timeout=timeout, **kwargs)


# --------------------------------------------------------------------------- #
# Channel protocol and shell state
# --------------------------------------------------------------------------- #
@posix_only
def test_streams_exit_codes_and_state_come_back_separately(tmp_path, fake_bwrap):
    session = ShellSession(str(tmp_path))
    try:
        result = _run(session, "echo out; echo err >&2; mkdir sub; cd sub; export A=1; exit 3")
        assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")
        assert result.persistent is True
        # `exit` still records the state (the EXIT trap).
        assert session.relative_cwd() == "sub"
        assert session.env["A"] == "1"

        assert _run(session, "unset A; echo ${A:-unset}").stdout == "unset\n"
        assert "A" not in session.env
        for owned in ("_", "SHLVL", "PWD"):
            assert owned not in session.env
        assert len(fake_bwrap) == 1, "one sandbox for the whole run"
    finally:
        session.close()


@posix_only
def test_working_dir_does_not_move_the_shell(tmp_path, fake_bwrap):
    (tmp_path / "frontend").mkdir()
    session = ShellSession(str(tmp_path))
    try:
        _run(session, "mkdir backend; cd backend")
        assert _run(session, "pwd", working_dir=str(tmp_path / "frontend")).stdout.strip() \
            == str(tmp_path / "frontend")
        assert session.relative_cwd() == "backend"
    finally:
        session.close()


@posix_only
def test_a_command_that_ends_outside_the_workspace_resets_to_its_root(tmp_path, fake_bwrap):
    session = ShellSession(str(tmp_path / "ws"))
    os.makedirs(session.workspace)
    try:
        result = _run(session, "cd /")
        assert session.relative_cwd() == "."
        assert "outside the run workspace" in result.notes[0]
    finally:
        session.close()


@posix_only
def test_a_timeout_kills_the_commands_group_only(tmp_path, fake_bwrap, pids):
    session = ShellSession(str(tmp_path))
    pids += [str(tmp_path / "bg.pid"), str(tmp_path / "child.pid")]
    try:
        _run(session, "sleep 300 > /dev/null 2>&1 & echo $! > bg.pid")
        started = time.monotonic()
        hung = _run(session, "sleep 300 & echo $! > child.pid; echo partial; sleep 300", timeout=1)
        assert time.monotonic() - started < 10
        assert hung.timed_out and hung.returncode is None
        assert hung.stdout == "partial\n", "output before the kill is kept"

        assert _alive(int((tmp_path / "bg.pid").read_text()))
        assert _wait_for(lambda: not _alive(int((tmp_path / "child.pid").read_text())))
        after = _run(session, "echo ok")
        assert after.stdout == "ok\n" and after.notes == []
    finally:
        session.close()


@posix_only
def test_background_output_neither_blocks_the_process_nor_reaches_later_results(
        tmp_path, fake_bwrap, pids):
    session = ShellSession(str(tmp_path))
    pids.append(str(tmp_path / "writer.pid"))
    # Far past a pipe's 64 KiB buffer: a writer nobody drains would block.
    # (What it writes before its command returns is that command's output.)
    writer = ("python3 -c \"import sys, time; time.sleep(0.5); sys.stdout.write('x' * 3000000); "
              "sys.stdout.flush(); open('done', 'w').close()\" & echo $! > writer.pid")
    try:
        started = _run(session, writer)
        assert "x" not in started.stdout
        assert _wait_for(lambda: (tmp_path / "done").exists()), "the background writer blocked"
        assert _run(session, "echo next").stdout == "next\n"
        # A writer that never stops cannot hold a command's result hostage.
        _run(session, "yes & echo $! > writer.pid")
        began = time.monotonic()
        quick = _run(session, "echo quick")
        assert quick.stdout == "quick\n" and time.monotonic() - began < 5
        assert FLOOD_NOTE not in quick.stderr
    finally:
        session.close()


@posix_only
def test_an_output_flood_kills_the_command_and_says_so(tmp_path, fake_bwrap, monkeypatch):
    monkeypatch.setattr(shell_session, "MAX_CAPTURE_BYTES", 200_000)
    session = ShellSession(str(tmp_path))
    try:
        flooded = _run(session, "yes")
        assert flooded.returncode == -signal.SIGKILL
        assert flooded.stderr.endswith(FLOOD_NOTE)
        assert _run(session, "echo alive").stdout == "alive\n"
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Lifecycle: teardown, idle timeout, cap
# --------------------------------------------------------------------------- #
@posix_only
def test_close_ends_the_supervisor_and_later_commands_do_not_leak_a_session(
        tmp_path, fake_bwrap):
    session = ShellSession(str(tmp_path))
    _run(session, "true")
    supervisor = session._proc.proc
    assert shell_session.live_session_count() == 1

    session.close()

    assert supervisor.poll() is not None, "the supervisor must have exited and been reaped"
    result = _run(session, "echo after")
    assert result.stdout == "after\n" and result.persistent is False
    assert shell_session.live_session_count() == 0


@posix_only
def test_close_sessions_for_the_run_folder(tmp_path, fake_bwrap):
    mine, other = ShellSession(str(tmp_path / "a")), ShellSession(str(tmp_path / "b"))
    os.makedirs(mine.workspace)
    os.makedirs(other.workspace)
    try:
        _run(mine, "true")
        _run(other, "true")
        shell_session.close_sessions_for(str(tmp_path / "a"))
        assert mine._proc is None and other._proc is not None
    finally:
        other.close()


@posix_only
def test_an_idle_session_is_torn_down_and_the_next_command_is_told(
        tmp_path, fake_bwrap, monkeypatch):
    monkeypatch.setenv(shell_session.IDLE_ENV, "1")
    session = ShellSession(str(tmp_path))
    try:
        _run(session, "export KEPT=yes")
        supervisor = session._proc.proc
        assert _wait_for(lambda: session._proc is None, 8), "the reaper never fired"
        assert supervisor.poll() is not None

        result = _run(session, "echo $KEPT")
        assert result.stdout == "yes\n", "the harness keeps env across a restart"
        assert any("without a command" in note for note in result.notes)
    finally:
        session.close()


@posix_only
def test_a_busy_session_is_not_reaped(tmp_path, fake_bwrap, monkeypatch):
    monkeypatch.setenv(shell_session.IDLE_ENV, "1")
    session = ShellSession(str(tmp_path))
    try:
        assert _run(session, "sleep 3; echo done", timeout=10).stdout == "done\n"
    finally:
        session.close()


@posix_only
def test_past_the_cap_a_command_runs_in_a_one_off_sandbox(tmp_path, fake_bwrap, monkeypatch):
    monkeypatch.setenv(shell_session.MAX_ENV, "1")
    first, second = ShellSession(str(tmp_path / "a")), ShellSession(str(tmp_path / "b"))
    os.makedirs(first.workspace)
    os.makedirs(second.workspace)
    try:
        assert _run(first, "true").persistent
        capped = _run(second, "mkdir s; cd s")
        assert capped.persistent is False
        assert any("one-off sandbox" in note for note in capped.notes)
        assert second.relative_cwd() == "s", "cwd still carries over"
        assert shell_session.live_session_count() == 1

        first.close()
        assert _run(second, "true").persistent, "a freed slot is reused"
    finally:
        first.close()
        second.close()


@posix_only
def test_a_cap_of_zero_gives_every_command_its_own_sandbox(tmp_path, fake_bwrap, monkeypatch):
    monkeypatch.setenv(shell_session.MAX_ENV, "0")
    session = ShellSession(str(tmp_path))
    _run(session, "true")
    _run(session, "true")
    assert len(fake_bwrap) == 2
    assert shell_session.live_session_count() == 0


# --------------------------------------------------------------------------- #
# A session that misbehaves
# --------------------------------------------------------------------------- #
@posix_only
def test_a_command_that_kills_the_supervisor_ends_only_that_session(tmp_path, fake_bwrap):
    session = ShellSession(str(tmp_path))
    try:
        # bash's parent is the supervisor. SIGTERM is ignored...
        assert _run(session, "kill -TERM $PPID; sleep 0.2; echo survived").stdout == "survived\n"
        # ...SIGKILL cannot be.
        killed = _run(session, "kill -KILL $PPID; sleep 5")
        assert killed.returncode is None
        assert "session ended" in killed.stderr
        assert shell_session.live_session_count() == 0

        again = _run(session, "echo fresh")
        assert again.stdout == "fresh\n" and again.persistent
    finally:
        session.close()


def _fake_supervisor(script: str) -> list[str]:
    return [sys.executable, "-c", "import json, struct, sys, time\n"
            "def send(m):\n"
            "    d = json.dumps(m).encode()\n"
            "    sys.stdout.buffer.write(struct.pack('>I', len(d)) + d); sys.stdout.buffer.flush()\n"
            + script]


@pytest.mark.parametrize("reply", [
    "sys.stdout.buffer.write(struct.pack('>I', 1 << 31)); sys.stdout.buffer.flush()",
    "send(['not', 'a', 'dict'])",
    "send({'id': 999, 'exit_code': 0})",
    "sys.stdout.buffer.write(struct.pack('>I', 50) + b'{broken'.ljust(50)); sys.stdout.buffer.flush()",
])
def test_a_malformed_reply_ends_the_session_instead_of_being_trusted(
        tmp_path, monkeypatch, reply):
    argv = _fake_supervisor("send({'ready': True}); sys.stdin.buffer.read(4)\n"
                            + reply + "\ntime.sleep(30)\n")
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda *_a, **_k: SandboxedCommand(argv, False, "bwrap"))
    session = ShellSession(str(tmp_path))
    try:
        result = _run(session, "echo hi")
        assert result.returncode is None and "session ended" in result.stderr
        assert session._proc is None
    finally:
        session.close()


def test_a_supervisor_that_never_answers_is_killed(tmp_path, monkeypatch):
    argv = _fake_supervisor("send({'ready': True}); time.sleep(120)\n")
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda *_a, **_k: SandboxedCommand(argv, False, "bwrap"))
    monkeypatch.setattr(shell_session, "_RESPONSE_GRACE", 1)
    session = ShellSession(str(tmp_path))
    started = time.monotonic()
    try:
        result = _run(session, "echo hi", timeout=1)
        assert time.monotonic() - started < 15
        assert "session ended" in result.stderr
    finally:
        session.close()


def test_a_sandbox_that_cannot_start_refuses_and_runs_nothing(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    argv = [sys.executable, "-c", "import sys; sys.stderr.write("
            "'bwrap: setting up uid map: Permission denied\\n'); sys.exit(1)"]
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda *_a, **_k: SandboxedCommand(argv, False, "bwrap"))

    result = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)._run_command(
        {"command": f"touch {marker}"})

    assert result["error"] == te._SANDBOX_REFUSAL
    assert "bwrap" not in result["error"]
    assert not marker.exists()
    assert shell_session.live_session_count() == 0


def test_a_bwrap_that_cannot_be_executed_is_the_generic_refusal(tmp_path, monkeypatch):
    missing = str(tmp_path / "no-such-bwrap")
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda *_a, **_k: SandboxedCommand([missing], False, "bwrap"))

    result = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)._run_command(
        {"command": "echo X"})

    assert result["error"] == te._SANDBOX_REFUSAL
    assert missing not in str(result)
    assert shell_session.live_session_count() == 0


def test_no_bwrap_on_linux_refuses_without_starting_anything(tmp_path, monkeypatch):
    monkeypatch.setenv(SANDBOX_POLICY_ENV, "auto")
    monkeypatch.setattr(sandbox_mod, "sandbox_supported_platform", lambda: True)
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda _name: None)
    started = []
    monkeypatch.setattr(shell_session, "_SessionProcess", lambda *a, **k: started.append(a))
    monkeypatch.setattr(shell_session, "run_bounded", lambda *a, **k: started.append(a))

    result = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)._run_command(
        {"command": "echo X"})

    assert result["error"] == te._SANDBOX_REFUSAL
    assert started == []


# --------------------------------------------------------------------------- #
# Command construction: the session gets exactly run_command's confinement
# --------------------------------------------------------------------------- #
@linux_only
def test_the_session_is_the_same_sandbox_a_command_used_to_get(tmp_path, monkeypatch):
    monkeypatch.setenv(SANDBOX_POLICY_ENV, "auto")
    monkeypatch.setenv(sandbox_mod.SHELL_NETWORK_ENV, "shared")
    monkeypatch.setattr(sandbox_mod.shutil, "which",
                        lambda name: "/usr/bin/bwrap" if name == "bwrap" else "/bin/bash")
    monkeypatch.setattr(sandbox_mod, "_selftest", lambda _b: (True, ""))
    seen = []

    def _capture(argv, cwd, **_):
        seen.append(argv)
        raise SandboxUnavailable("stop here")

    monkeypatch.setattr(shell_session, "_SessionProcess", _capture)
    session = ShellSession(str(tmp_path))
    with pytest.raises(SandboxUnavailable):
        _run(session, "true")

    argv = seen[0]
    per_command = sandbox_mod.sandboxed_command("true", workspace=str(tmp_path),
                                                cwd=str(tmp_path)).argv
    split = argv.index("--")
    assert argv[:split] == per_command[:per_command.index("--")], \
        "same isolation flags, mount plan, masks and HOME as a per-command sandbox"
    assert "--unshare-net" not in argv, "a shared-network session keeps the worker's network"
    for flag in ("--unshare-pid", "--unshare-user", "--die-with-parent", "--new-session"):
        assert flag in argv
    assert argv[split + 1:split + 5] == [sys.executable, "-I", "-S", "-c"]
    assert argv[split + 5] == shell_session.SUPERVISOR_SOURCE
    assert argv[split + 6:] == [session.workspace, shell_session.BASH_PRELUDE, "/bin/bash"]


def test_validators_never_go_through_the_session(tmp_path, monkeypatch):
    """A check must not run where the model's leftover state could steer it."""
    monkeypatch.setattr(shell_session.ShellSession, "run",
                        lambda *a, **k: pytest.fail("a validator used the shell session"))
    monkeypatch.setattr(shell_session, "_SessionProcess",
                        lambda *a, **k: pytest.fail("a validator started a session"))
    ex = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    ran = []
    monkeypatch.setattr(sandbox_mod, "run_bounded",
                        lambda argv, **k: ran.append(argv) or
                        __import__("subprocess").CompletedProcess(argv, 0, "", ""))

    run_confined([sys.executable, "-c", "pass"], workspace=ex.workspace,
                 cwd=ex.workspace, timeout=5)

    assert len(ran) == 1
    if ran[0][0].endswith("bwrap"):
        assert "--unshare-net" in ran[0]


# --------------------------------------------------------------------------- #
# Without a sandbox: cwd and env still carry over, in the harness
# --------------------------------------------------------------------------- #
@pytest.fixture
def unconfined(monkeypatch):
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda argv, **_: SandboxedCommand(argv, False, "unconfined-platform"))


def test_unconfined_commands_still_share_cwd_and_env(tmp_path, unconfined):
    (tmp_path / "sub dir").mkdir()
    session = ShellSession(str(tmp_path))
    if os.name == "nt":
        first, second = 'cd "sub dir" && set GREETING=hello there', "cd & echo %GREETING%"
    else:
        first, second = 'cd "sub dir" && export GREETING="hello there"', "pwd; echo $GREETING"

    assert _run(session, first).returncode == 0
    result = _run(session, second)

    assert result.stdout.splitlines() == [str(tmp_path / "sub dir"), "hello there"]
    assert result.persistent is False
    assert session.relative_cwd() == "sub dir"


def test_unconfined_exit_codes_survive_the_state_capture(tmp_path, unconfined):
    session = ShellSession(str(tmp_path))
    fail = "exit /b 4" if os.name == "nt" else "exit 4"
    assert _run(session, fail).returncode == 4
    chained = "cmd /c exit 5" if os.name == "nt" else "(exit 5)"
    assert _run(session, chained).returncode == 5


def test_install_dependencies_leaves_the_shell_where_it_was(tmp_path, unconfined):
    (tmp_path / "sub").mkdir()
    ex = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    ex._run_command({"command": "cd sub"})
    ex._install_dependencies({"command": "cd .."})
    assert ex._shell.relative_cwd() == "sub"


# --------------------------------------------------------------------------- #
# Harness-side state limits
# --------------------------------------------------------------------------- #
def test_an_oversized_environment_is_not_kept(tmp_path):
    session = ShellSession(str(tmp_path))
    session.env = {"KEEP": "1"}
    notes = []
    session._adopt(None, {"BIG": "x" * (shell_session._MAX_ENV_BYTES + 1)},
                   follow_cwd=True, notes=notes)
    assert session.env == {"KEEP": "1"}
    assert "limit" in notes[0]


def test_a_vanished_directory_restarts_in_the_workspace_root(tmp_path, unconfined):
    session = ShellSession(str(tmp_path))
    session.cwd = str(tmp_path / "gone")
    result = _run(session, "cd" if os.name == "nt" else "pwd")
    assert result.stdout.strip() == str(tmp_path)
    assert "no longer exists" in result.notes[0]


# --------------------------------------------------------------------------- #
# Wiring: every exit of a run closes its session
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("outcome", ["returns", "raises"])
def test_a_run_entry_point_closes_the_session_however_it_ends(outcome):
    closed = []

    class _Run:
        executor = type("E", (), {"close": lambda self: closed.append(1)})()

        @te.ends_shell_session
        def run(self):
            if outcome == "raises":
                raise RuntimeError("boom")
            return "done"

    if outcome == "raises":
        with pytest.raises(RuntimeError):
            _Run().run()
    else:
        assert _Run().run() == "done"
    assert closed == [1]


def test_run_resume_and_modify_are_the_wrapped_entry_points():
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    for name in ("run", "resume", "modify"):
        assert getattr(LLMOrchestrator, name).__wrapped__ is not None, name


def test_a_phase3_sweep_first_stops_what_the_model_left_running():
    """A `next dev` left running would write .next while the build check runs."""
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

    class _Stopped(Exception):
        pass

    orch = LLMOrchestrator.__new__(LLMOrchestrator)
    orch.executor = type("E", (), {"stop_shell_processes": lambda self: (_ for _ in ()).throw(_Stopped())})()
    with pytest.raises(_Stopped):
        orch._collect_validation_issues()


def test_removing_a_run_folder_closes_its_session(tmp_path, monkeypatch):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner
    closed = []
    monkeypatch.setattr(runner, "close_sessions_for", closed.append)
    runner._remove_run_dir(str(tmp_path))
    assert closed == [str(tmp_path)]


# --------------------------------------------------------------------------- #
# What the model is told
# --------------------------------------------------------------------------- #
def test_the_tool_text_describes_one_terminal_per_run():
    tools = {t["name"]: t for t in get_tools_for(has_domain_model=True, allow_shell=True)}
    doc = tools["run_command"]["description"]
    assert "does not outlive" not in doc
    for fact in ("`cd`", "keeps running", "one-off sandbox", "restarted", "cwd"):
        assert fact in doc
    working_dir = tools["run_command"]["input_schema"]["properties"]["working_dir"]
    assert "default" not in working_dir, "a filled-in default would pin every command to the root"
    assert "does not move" in working_dir["description"]
    assert "leaves the shell's directory" in tools["install_dependencies"]["description"]


def test_the_runbook_says_the_server_stays_up(tmp_path):
    (tmp_path / "main_api.py").write_text("app = None\n")
    (tmp_path / "sql_alchemy.py").write_text("")
    text = runbook_section(str(tmp_path))
    assert "does not outlive" not in text
    assert "keeps\nrunning between commands" in text or "keeps running between commands" in text


# --------------------------------------------------------------------------- #
# Spawning from a thread that exits (bwrap's PDEATHSIG is per thread)
# --------------------------------------------------------------------------- #
def test_sessions_are_forked_from_one_long_lived_thread():
    names = []
    for _ in range(3):
        box = []
        worker = threading.Thread(
            target=lambda: box.append(shell_session._spawn(lambda: threading.current_thread())))
        worker.start()
        worker.join()
        names.append(box[0])
    assert len(set(names)) == 1
    assert names[0].name == "besser-shell-spawner" and names[0].is_alive()
