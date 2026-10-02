"""run_command is one terminal per run, inside the real bubblewrap sandbox.

Before the shell session, every command was its own bwrap PID namespace, so a
server started in one command died when that command returned, and ``cd`` /
``export`` were lost. These tests run the real sandbox (they skip where it
cannot start) and assert the terminal behaviour, then that persistence did
not weaken the confinement: other runs, the telemetry and incident folders,
the worker's environment and the session's own channel stay out of reach, and
nothing survives the session's teardown.

Where a test asserts something is unreachable it first shows the same probe
succeeding against something reachable, so a probe that never worked cannot
pass as proof.
"""

import json
import os
import shutil
import socket
import tempfile
import threading
import time
import urllib.request

import pytest

from besser.spec_driven_agent.agent import tool_executor as te
from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, install_probe
from besser.spec_driven_agent.execution import shell_session
from besser.spec_driven_agent.execution.sandbox import (
    SANDBOX_POLICY_ENV,
    SHELL_NETWORK_ENV,
    run_confined,
    sandbox_selftest_error,
)

_PROBLEM = sandbox_selftest_error()
pytestmark = pytest.mark.skipif(_PROBLEM is not None,
                                reason=f"no namespace sandbox on this host: {_PROBLEM}")

_MARKER = "BESSER_TEST_FAKE_TOKEN_SESSION_Q4"


@pytest.fixture(autouse=True)
def _sandbox_on(monkeypatch):
    monkeypatch.setenv(SANDBOX_POLICY_ENV, "auto")
    # These watch a session's servers from the worker, which a private network
    # forbids; private sessions are covered in test_shell_session_private_net.py.
    monkeypatch.setenv(SHELL_NETWORK_ENV, "shared")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(port: int) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            return response.status
    except OSError:
        return 0


def _wait_for(predicate, seconds=15.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


def _host_processes_touching(path: str) -> list[str]:
    """Host pids whose command line or cwd names ``path``: the session's
    bwrap, supervisor, and anything started inside it."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
            cwd = os.readlink(f"/proc/{entry}/cwd")
        except OSError:
            continue
        if path in cmdline or cwd == path or cwd.startswith(path + "/"):
            found.append(f"{entry}: {cmdline[:120]} (cwd {cwd})")
    return found


def _my_zombies() -> list[str]:
    zombies = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] == "Z" and int(fields[1]) == os.getpid():
            zombies.append(entry)
    return zombies


@pytest.fixture
def run(tmp_path):
    workspace = tmp_path / "runs" / "besser_spec_run_a"
    workspace.mkdir(parents=True)
    executor = te.ToolExecutor(workspace=str(workspace), allow_shell=True)
    yield executor
    executor.close()


def _sh(executor, command, **extra):
    return executor._run_command({"command": command, **extra})


# --------------------------------------------------------------------------- #
# The terminal behaviour
# --------------------------------------------------------------------------- #
def test_a_server_started_in_one_command_answers_the_next(run):
    port = _free_port()
    started = _sh(run, f"python3 -m http.server {port} --bind 127.0.0.1 > server.log 2>&1 &")
    assert started["success"], started

    answered = _sh(run, f"python3 -c \"import urllib.request as u; "
                        f"print(u.urlopen('http://127.0.0.1:{port}/', timeout=5).status)\"")
    # Retried once: the first request can beat the server's bind.
    if "200" not in answered.get("stdout", ""):
        time.sleep(1.5)
        answered = _sh(run, f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/")
    assert "200" in answered["stdout"], answered
    assert "notes" not in answered, answered


def test_cd_and_export_carry_over_to_the_next_command(run):
    assert _sh(run, "mkdir -p backend && cd backend && export APP_MODE=dev")["success"]

    result = _sh(run, "pwd; echo mode=$APP_MODE")

    assert result["stdout"].splitlines() == [os.path.join(run.workspace, "backend"), "mode=dev"]
    assert result["cwd"] == "backend"


def test_working_dir_runs_one_command_elsewhere_without_moving_the_shell(run):
    os.makedirs(os.path.join(run.workspace, "frontend"))
    _sh(run, "mkdir -p backend && cd backend")

    elsewhere = _sh(run, "pwd", working_dir="frontend")
    after = _sh(run, "pwd")

    assert elsewhere["stdout"].strip() == os.path.join(run.workspace, "frontend")
    assert after["stdout"].strip() == os.path.join(run.workspace, "backend")


def test_a_timed_out_command_is_killed_but_the_session_and_its_server_survive(run, monkeypatch):
    monkeypatch.setattr(te, "COMMAND_TIMEOUT", 2)
    port = _free_port()
    _sh(run, f"python3 -m http.server {port} --bind 127.0.0.1 > server.log 2>&1 &")
    assert _wait_for(lambda: _get(port) == 200)

    started = time.monotonic()
    hung = _sh(run, "sleep 300 & echo $! > hung_child.pid; sleep 300")
    assert time.monotonic() - started < 15
    assert "timed out" in hung["error"], hung
    assert "unaffected" in hung["error"]

    assert _get(port) == 200, "the timeout killed the background server"
    child = int(open(os.path.join(run.workspace, "hung_child.pid")).read())
    check = _sh(run, f"kill -0 {child} 2>/dev/null && echo CHILD_ALIVE || echo CHILD_DEAD")
    assert check["stdout"].strip() == "CHILD_DEAD", "what the timed-out command started must die"
    assert "notes" not in check, "the session must not have restarted"


def test_the_session_survives_the_thread_that_started_it(run, monkeypatch):
    """Tools run on a per-turn ThreadPoolExecutor, whose threads exit after the
    turn. bwrap's --die-with-parent is a PR_SET_PDEATHSIG, which fires when the
    thread that forked bwrap exits, so a session forked from a tool thread died
    at the end of its first turn."""
    def first_turn(executor):
        thread = threading.Thread(target=_sh, args=(executor, "export FROM_TURN_1=yes"))
        thread.start()
        thread.join()
        time.sleep(0.5)

    first_turn(run)
    later = _sh(run, "echo turn1=$FROM_TURN_1")
    assert "notes" not in later, later
    assert later["stdout"].strip() == "turn1=yes"

    # Negative control: fork from the tool thread itself and the session dies
    # with that thread.
    other = te.ToolExecutor(workspace=os.path.join(os.path.dirname(run.workspace), "control"),
                            allow_shell=True)
    os.makedirs(other.workspace, exist_ok=True)
    monkeypatch.setattr(shell_session, "_spawn", lambda factory: factory())
    try:
        first_turn(other)
        later = _sh(other, "echo turn1=$FROM_TURN_1")
        assert any("stopped" in note for note in later.get("notes", [])), later
    finally:
        other.close()


def test_the_probe_up_then_req_reach_one_and_the_same_server(run):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    with open(os.path.join(run.workspace, "main_api.py"), "w", encoding="utf-8") as handle:
        handle.write("from fastapi import FastAPI\napp = FastAPI()\n"
                     "@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n")
    open(os.path.join(run.workspace, "sql_alchemy.py"), "w").close()
    assert install_probe(run.workspace)
    state_file = os.path.join(run.workspace, ".besser_probe.state.json")
    try:
        up = _sh(run, f"python {PROBE_FILENAME} up")
        assert "BOOT_OK" in up["stdout"], up
        booted = json.load(open(state_file))

        req = _sh(run, f"python {PROBE_FILENAME} req GET /health")
        assert req["stdout"].startswith("200 GET /health"), req
        assert json.load(open(state_file)) == booted, "req booted a second server"

        down = _sh(run, f"python {PROBE_FILENAME} down")
        assert f"STOPPED pid={booted['pid']}" in down["stdout"], down
    finally:
        _sh(run, f"python {PROBE_FILENAME} down")


# --------------------------------------------------------------------------- #
# Teardown
# --------------------------------------------------------------------------- #
def test_closing_the_session_leaves_no_process_and_no_zombie(run):
    port = _free_port()
    _sh(run, f"python3 -m http.server {port} --bind 127.0.0.1 > server.log 2>&1 &")
    _sh(run, "setsid sleep 600 > /dev/null 2>&1 < /dev/null &")
    assert _wait_for(lambda: _get(port) == 200)
    assert _host_processes_touching(run.workspace), "the probe must see the live session"

    run.close()

    assert _wait_for(lambda: not _host_processes_touching(run.workspace), 10), \
        _host_processes_touching(run.workspace)
    assert _get(port) == 0
    assert _my_zombies() == []
    assert shell_session.live_session_count() == 0


def test_a_command_after_close_does_not_start_a_lasting_session(run):
    run.close()
    result = _sh(run, "sleep 600 > /dev/null 2>&1 &")
    assert result["success"]
    assert any("one-off sandbox" in note for note in result["notes"])
    assert shell_session.live_session_count() == 0
    assert _wait_for(lambda: not _host_processes_touching(run.workspace), 10)


def test_before_a_validation_sweep_the_processes_stop_but_the_shell_state_stays(run):
    port = _free_port()
    # `;` not `&&`: `a && b &` would background the whole list, cd included.
    _sh(run, f"mkdir -p api && cd api && export MODE=x; "
             f"python3 -m http.server {port} --bind 127.0.0.1 > s.log 2>&1 &")
    assert _wait_for(lambda: _get(port) == 200)

    run.stop_shell_processes()

    assert _get(port) == 0
    after = _sh(run, "pwd; echo $MODE")
    assert after["stdout"].split() == [os.path.join(run.workspace, "api"), "x"]
    assert any("re-checked the workspace" in note for note in after["notes"])
    assert shell_session.live_session_count() == 1, "the next command got a new session"


def test_removing_the_run_folder_ends_its_session(run):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner

    _sh(run, "sleep 600 > /dev/null 2>&1 &")
    assert shell_session.live_session_count() == 1

    runner._remove_run_dir(run.workspace)

    assert shell_session.live_session_count() == 0
    assert _wait_for(lambda: not _host_processes_touching(run.workspace), 10)


# --------------------------------------------------------------------------- #
# Confinement is unchanged
# --------------------------------------------------------------------------- #
def test_two_runs_sessions_see_neither_each_others_processes_nor_files(tmp_path):
    root = tmp_path / "runs"
    a = te.ToolExecutor(workspace=str(root / "besser_spec_a"), allow_shell=True)
    b = te.ToolExecutor(workspace=str(root / "besser_spec_b"), allow_shell=True)
    os.makedirs(a.workspace)
    os.makedirs(b.workspace)
    count = "cat /proc/[0-9]*/cmdline 2>/dev/null | tr '\\0' ' ' | grep -o '{m}' | wc -l"
    # "MARKE[R]": the grep's own command line must not match itself.
    try:
        _sh(a, "echo A_SECRET > secret.txt; "
               "python3 -c 'import time; time.sleep(600)' RUN_A_MARKER > /dev/null 2>&1 &")
        _sh(b, "python3 -c 'import time; time.sleep(600)' RUN_B_MARKER > /dev/null 2>&1 &")
        time.sleep(0.5)

        assert int(_sh(a, count.format(m="RUN_A_MARKE[R]"))["stdout"]) >= 1, "probe is broken"
        assert int(_sh(b, count.format(m="RUN_A_MARKE[R]"))["stdout"]) == 0
        assert int(_sh(a, count.format(m="RUN_B_MARKE[R]"))["stdout"]) == 0

        assert "A_SECRET" in _sh(a, "cat secret.txt")["stdout"]
        leaked = _sh(b, f"cat {a.workspace}/secret.txt; ls {root}")
        assert "A_SECRET" not in leaked.get("stdout", "")
        assert "besser_spec_a" not in leaked.get("stdout", "")
    finally:
        a.close()
        b.close()


def test_the_telemetry_and_incident_masks_hold_in_the_session(run, monkeypatch):
    # Under a top-level the sandbox binds read-only, like /app/telemetry.
    base = next((d for d in (os.path.expanduser("~"), os.getcwd())
                 if os.access(d, os.W_OK) and not d.startswith("/tmp")), None)
    if base is None:
        pytest.skip("no writable directory outside /tmp")
    scratch = tempfile.mkdtemp(prefix="besser_mask_test_", dir=base)
    try:
        for name in ("telemetry", "incidents", "visible"):
            os.makedirs(os.path.join(scratch, name))
            with open(os.path.join(scratch, name, "rows.jsonl"), "w") as handle:
                handle.write(f"{name.upper()}_ROW\n")
        monkeypatch.setenv("BESSER_TELEMETRY_DIR", os.path.join(scratch, "telemetry"))
        monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", os.path.join(scratch, "incidents"))

        seen = _sh(run, f"cat {scratch}/*/rows.jsonl 2>/dev/null")["stdout"]

        assert "VISIBLE_ROW" in seen, "the probe must read an unmasked sibling"
        assert "TELEMETRY_ROW" not in seen
        assert "INCIDENTS_ROW" not in seen
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_the_session_holds_no_worker_secret(run, monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", _MARKER)
    monkeypatch.setenv("OPENAI_API_KEY", _MARKER)
    _sh(run, "export MINE=1")

    probe = _sh(run, "env; cat /proc/[0-9]*/environ 2>/dev/null | tr '\\0' '\\n'")

    assert "MINE=1" in probe["stdout"], "the probe must see the session's own environment"
    assert _MARKER not in probe["stdout"]
    assert _MARKER not in json.dumps(run._shell.env)


def test_commands_cannot_reach_or_stop_the_supervisors_channel(run):
    find = "grep -l besser-shell /proc/[0-9]*/comm | cut -d/ -f3"
    supervisor = _sh(run, find)["stdout"].split()
    assert len(supervisor) == 1, supervisor
    pid = supervisor[0]

    # The same probes against an ordinary process of the session succeed.
    probe = ("readlink /proc/{p}/fd/1 > /dev/null 2>&1 && echo LINK_OK || echo LINK_DENIED; "
             "(: < /proc/{p}/fd/0) 2>/dev/null && echo READ_OK || echo READ_DENIED; "
             "(: > /proc/{p}/fd/1) 2>/dev/null && echo WRITE_OK || echo WRITE_DENIED")
    ordinary = _sh(run, "sleep 60 > /dev/null 2>&1 < /dev/null & p=$!; "
                        + probe.format(p="$p") + "; kill $p")
    assert ordinary["stdout"].split() == ["LINK_OK", "READ_OK", "WRITE_OK"], ordinary

    channel = _sh(run, probe.format(p=pid))
    assert channel["stdout"].split() == ["LINK_DENIED", "READ_DENIED", "WRITE_DENIED"], channel

    seize = ("import ctypes, subprocess; libc = ctypes.CDLL(None)\n"
             "child = subprocess.Popen(['sleep', '60'])\n"
             "print(libc.ptrace(0x4206, child.pid, 0, 0), libc.ptrace(0x4206, {sup}, 0, 0))\n"
             "child.kill()\n").format(sup=pid)
    with open(os.path.join(run.workspace, "seize.py"), "w") as handle:
        handle.write(seize)
    # PTRACE_SEIZE: allowed on its own child, refused on the supervisor.
    assert _sh(run, "python3 seize.py")["stdout"].split() == ["0", "-1"]

    _sh(run, f"kill -TERM {pid}; kill -INT {pid}; kill -HUP {pid}; sleep 0.3")
    after = _sh(run, "echo still-here")
    assert after["stdout"].strip() == "still-here"
    assert "notes" not in after, "a polite kill must not end the session"


def test_a_validators_sandbox_sees_nothing_of_the_session(run):
    port = _free_port()
    _sh(run, f"python3 -m http.server {port} --bind 127.0.0.1 > s.log 2>&1 &")
    assert _wait_for(lambda: _get(port) == 200)

    checked = run_confined(
        ["/bin/sh", "-c",
         "cat /proc/[0-9]*/cmdline 2>/dev/null | tr '\\0' ' ' | grep -c 'http[.]server'; "
         f"python3 -c \"import urllib.request as u; u.urlopen('http://127.0.0.1:{port}/', timeout=3)\" "
         "&& echo REACHED || echo UNREACHABLE"],
        workspace=run.workspace, cwd=run.workspace, timeout=30,
    )

    lines = checked.stdout.split()
    assert lines[0] == "0", checked.stdout
    assert lines[-1] == "UNREACHABLE", checked.stdout
