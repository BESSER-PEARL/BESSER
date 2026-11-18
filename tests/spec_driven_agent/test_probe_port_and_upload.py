"""`.besser_probe.py` must not make the agent fight its own helper.

A restart (`down`, `up`) moved the server to a new port every time, so a
command or script written with the earlier port hit "connection refused";
and `req` could send only JSON, so an upload endpoint had to be tested with
hand-written curl against that moving port.
"""
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, PROBE_SCRIPT, install_probe


def workspace(tmp_path):
    (tmp_path / "main_api.py").write_text("", encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    assert install_probe(str(tmp_path))
    return tmp_path


def load_probe(tmp_path):
    namespace = {"__file__": str(tmp_path / PROBE_FILENAME), "__name__": "besser_probe_under_test"}
    exec(compile(PROBE_SCRIPT, PROBE_FILENAME, "exec"), namespace)
    return namespace


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fake_boot(ns, new_port):
    """Boot without a real server: the child stays alive and answers at once."""
    class Child:
        pid = 4242

        def __init__(self, argv, **_kwargs):
            pass

        def poll(self):
            return None

    ns.update(free_port=lambda: new_port, is_up=lambda port: True,
              subprocess=SimpleNamespace(Popen=Child, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT,
                                         CREATE_NEW_PROCESS_GROUP=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                                         run=subprocess.run))


def test_a_restart_keeps_the_port_the_server_had(tmp_path):
    workspace(tmp_path)
    port = unused_port()
    server = subprocess.Popen([sys.executable, "-c",
                               "# BESSER_PROBE_SERVE port=%d\nimport time; time.sleep(60)" % port])
    try:
        (tmp_path / ".besser_probe.state.json").write_text(
            json.dumps({"pid": server.pid, "port": port, "backend": "."}), encoding="utf-8")
        down = subprocess.run([sys.executable, PROBE_FILENAME, "down"], cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)
        assert "STOPPED" in down.stdout, down.stdout + down.stderr
    finally:
        if server.poll() is None:
            server.kill()
        server.wait()

    ns = load_probe(tmp_path)
    fake_boot(ns, new_port=port + 1 if port < 65535 else port - 1)
    assert ns["boot"](quiet=True) == (0, port)


def test_a_port_someone_else_now_holds_is_not_taken_again(tmp_path):
    workspace(tmp_path)
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        taken = holder.getsockname()[1]
        (tmp_path / ".besser_probe.state.json").write_text(json.dumps({"port": taken}), encoding="utf-8")
        ns = load_probe(tmp_path)
        fresh = unused_port()
        fake_boot(ns, new_port=fresh)
        assert ns["boot"](quiet=True) == (0, fresh)
