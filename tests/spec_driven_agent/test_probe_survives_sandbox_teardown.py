"""The runbook's requests must work although `up`'s server is gone.

Measured in the smartgen_worker image (bwrap, seccomp/apparmor unconfined):

    python .besser_probe.py up               -> BOOT_OK  port=43979  pid=4
    python .besser_probe.py req GET /health  -> CONNECTION_FAILED (refused)

Each run_command is its own bwrap PID namespace (--unshare-pid,
--die-with-parent), so the detached server dies when `up` returns. Here the
teardown is simulated by killing the server between the two commands.
"""
import json
import os
import signal
import subprocess
import sys

import pytest

from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, install_probe

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

MAIN_API = (
    "from fastapi import FastAPI\n"
    "app = FastAPI()\n"
    "@app.get('/health')\n"
    "def health():\n"
    "    return {'status': 'ok'}\n"
)


def probe(workspace, *argv):
    return subprocess.run(
        [sys.executable, PROBE_FILENAME, *argv], cwd=workspace,
        capture_output=True, text=True, timeout=90,
    )


def kill(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        os.kill(pid, signal.SIGKILL)


def test_req_boots_its_own_server_when_ups_is_gone(tmp_path):
    (tmp_path / "main_api.py").write_text(MAIN_API, encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    assert install_probe(str(tmp_path))
    try:
        assert "BOOT_OK" in probe(tmp_path, "up").stdout
        state = json.loads((tmp_path / ".besser_probe.state.json").read_text())
        kill(state["pid"])

        result = probe(tmp_path, "req", "GET", "/health")

        assert result.stdout.startswith("200 GET /health"), result.stdout + result.stderr
        kill(json.loads((tmp_path / ".besser_probe.state.json").read_text())["pid"])
        routes = probe(tmp_path, "routes").stdout
        assert "GET    /health" in routes, routes
    finally:
        probe(tmp_path, "down")
