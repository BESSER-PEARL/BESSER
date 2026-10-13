"""Shell sessions on a private network, with the real bwrap and pasta.

Verified live on experimental before this: a session shared the worker's
network namespace, so a command reached the worker's own API on
127.0.0.1:9000, and a second run's server on :8000 could not bind (and the
second run talked to the first run's server instead). Skipped where bwrap,
pasta or /dev/net/tun is missing.

Each unreachability check is paired with the same probe succeeding against a
reachable target, so a probe that never works cannot pass as proof.
"""

import http.server
import os
import shutil
import socket
import threading
import time

import pytest

from besser.spec_driven_agent.agent import tool_executor as te
from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.execution import shell_session
from besser.spec_driven_agent.execution.sandbox import (
    SANDBOX_POLICY_ENV,
    SHELL_NETWORK_ENV,
    sandbox_selftest_error,
)


def _network_problem() -> str | None:
    problem = sandbox_selftest_error()
    if problem:
        return problem
    ok, detail = sandbox_mod._network_selftest(shutil.which("bwrap"))
    return None if ok else detail


_PROBLEM = _network_problem()
pytestmark = pytest.mark.skipif(_PROBLEM is not None,
                                reason=f"no private network on this host: {_PROBLEM}")


@pytest.fixture(autouse=True)
def _private(monkeypatch):
    monkeypatch.setenv(SANDBOX_POLICY_ENV, "auto")
    monkeypatch.setenv(SHELL_NETWORK_ENV, "private")


def _executor(tmp_path, name):
    workspace = tmp_path / "runs" / name
    workspace.mkdir(parents=True)
    return te.ToolExecutor(workspace=str(workspace), allow_shell=True)


@pytest.fixture
def run(tmp_path):
    executor = _executor(tmp_path, "besser_spec_run_a")
    yield executor
    executor.close()


def _sh(executor, command):
    return executor._run_command({"command": command})


def _fetch(executor, url) -> str:
    """The body (or ``ERR``) of ``url`` fetched from inside the session."""
    result = _sh(executor, f"curl -s -m 3 {url} || echo ERR")
    return result.get("stdout", "").strip()


def _wait_for_body(executor, url, expected, seconds=10.0) -> str:
    deadline = time.monotonic() + seconds
    while True:
        body = _fetch(executor, url)
        if body == expected or time.monotonic() > deadline:
            return body
        time.sleep(0.3)


def _my_pastas() -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as handle:
                stat = handle.read()
        except OSError:
            continue
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        state, ppid = stat.rsplit(")", 1)[1].split()[:2]
        if comm.startswith("pasta") and int(ppid) == os.getpid():
            found.append((int(entry), state))
    return found


@pytest.fixture
def worker_listener():
    """A server on the worker's own 127.0.0.1, like the worker's API."""
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"worker-secret-api"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def test_two_runs_both_serve_on_8000_and_each_sees_only_its_own(tmp_path):
    first, second = _executor(tmp_path, "besser_spec_a"), _executor(tmp_path, "besser_spec_b")
    try:
        for executor, name in ((first, "run-a"), (second, "run-b")):
            started = _sh(executor, f"echo {name} > index.html && "
                                    "python3 -m http.server 8000 --bind 127.0.0.1 "
                                    "> server.log 2>&1 &")
            assert started["success"], started

        assert _wait_for_body(first, "http://127.0.0.1:8000/index.html", "run-a") == "run-a"
        assert _wait_for_body(second, "http://127.0.0.1:8000/index.html", "run-b") == "run-b"
        log = _sh(second, "cat server.log")["stdout"]
        assert "Address already in use" not in log, log
    finally:
        first.close()
        second.close()


def test_a_session_cannot_reach_the_workers_loopback(run, worker_listener):
    port = worker_listener
    with socket.create_connection(("127.0.0.1", port), timeout=3):
        pass  # the listener is up, seen from the worker
    assert _sh(run, "echo own > index.html && python3 -m http.server 8001 "
                    "--bind 127.0.0.1 > s.log 2>&1 &")["success"]
    assert _wait_for_body(run, "http://127.0.0.1:8001/index.html", "own") == "own", \
        "the probe itself works inside the session"

    assert _fetch(run, f"http://127.0.0.1:{port}/") == "ERR"
    gateway = _sh(run, "awk '$2 == \"00000000\" {print $3}' /proc/net/route")["stdout"].strip()
    if gateway:
        gw = socket.inet_ntoa(bytes.fromhex(gateway)[::-1])
        assert _fetch(run, f"http://{gw}:{port}/") == "ERR", f"gateway {gw} reaches the worker"


def test_outbound_dns_and_http_work(run):
    resolved = _sh(run, "getent hosts deb.debian.org")
    assert resolved["success"] and resolved["stdout"].strip(), resolved
    status = _sh(run, "curl -s -m 15 -o /dev/null -w '%{http_code}' http://deb.debian.org/debian/")
    assert status["stdout"].strip()[:1] in {"2", "3"}, status


def test_pasta_is_gone_after_close(tmp_path):
    executor = _executor(tmp_path, "besser_spec_close")
    assert _sh(executor, "true")["success"]
    pastas = _my_pastas()
    assert len(pastas) == 1 and pastas[0][1] != "Z", pastas

    executor.close()

    assert _my_pastas() == [], "no pasta left running, and none left as a zombie"


def test_a_killed_pasta_restarts_the_session_and_says_so(run):
    assert _sh(run, "true")["success"]
    (pid, _), = _my_pastas()
    os.kill(pid, 9)
    deadline = time.monotonic() + 5
    while run._shell._proc.network_alive() and time.monotonic() < deadline:
        time.sleep(0.05)

    result = _sh(run, "getent hosts deb.debian.org")

    assert any("lost its network" in note for note in result.get("notes", [])), result
    assert result["success"] and result["stdout"].strip(), "the new session has a network"
    assert [p for p, _ in _my_pastas()] != [pid]


def test_pasta_failing_to_attach_refuses_the_command(run, monkeypatch):
    marker = os.path.join(run.workspace, "ran")
    real_which = shutil.which
    monkeypatch.setattr(sandbox_mod.shutil, "which",
                        lambda name: "/bin/false" if name == "pasta" else real_which(name))

    result = _sh(run, f"touch {marker}")

    assert result["error"] == te._SANDBOX_REFUSAL, result
    assert not os.path.exists(marker), "never run on the shared network instead"
    assert shell_session.live_session_count() == 0
    assert _my_pastas() == []
