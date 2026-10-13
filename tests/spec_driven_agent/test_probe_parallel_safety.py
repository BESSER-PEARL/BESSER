"""Two probe invocations must not kill or borrow each other's processes.

Production run 438889bc: the model sent two run_command calls in one turn, the
orchestrator ran them in parallel, each in its own sandbox pid namespace, and
both shared `.besser_probe.state.json`. `down` killed the pid the other one had
recorded - which, inside its own namespace, was a sibling process (`grep`
printed "Terminated") - and `ensure_up` re-read a port the other call had just
overwritten, so requests went to the wrong server.
"""
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, PROBE_SCRIPT, install_probe


def workspace(tmp_path):
    (tmp_path / "main_api.py").write_text("", encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    assert install_probe(str(tmp_path))
    return tmp_path


def write_state(tmp_path, **state):
    (tmp_path / ".besser_probe.state.json").write_text(json.dumps(state), encoding="utf-8")


def sleeper(code="import time; time.sleep(60)"):
    return subprocess.Popen([sys.executable, "-c", code])


def probe(tmp_path, *argv):
    return subprocess.run([sys.executable, PROBE_FILENAME, *argv], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)


def test_down_never_kills_a_process_that_is_not_the_probe_server(tmp_path):
    workspace(tmp_path)
    bystander = sleeper()
    try:
        write_state(tmp_path, pid=bystander.pid, port=43979, backend=".")
        result = probe(tmp_path, "down")
        time.sleep(0.8)
        assert bystander.poll() is None, "down killed an unrelated process: " + result.stdout
        assert "NOT_RUNNING" in result.stdout
    finally:
        bystander.kill()
        bystander.wait()


def test_down_still_stops_the_server_it_started(tmp_path):
    workspace(tmp_path)
    server = sleeper("# BESSER_PROBE_SERVE port=43979\nimport time; time.sleep(60)")
    try:
        write_state(tmp_path, pid=server.pid, port=43979, backend=".")
        result = probe(tmp_path, "down")
        server.wait(timeout=10)
        assert "STOPPED" in result.stdout
    finally:
        if server.poll() is None:
            server.kill()


def load_probe(tmp_path):
    namespace = {"__file__": str(tmp_path / PROBE_FILENAME), "__name__": "besser_probe_under_test"}
    exec(compile(PROBE_SCRIPT, PROBE_FILENAME, "exec"), namespace)
    return namespace


def test_interleaved_invocations_each_use_the_port_they_started(tmp_path):
    workspace(tmp_path)
    ns = load_probe(tmp_path)
    started = {}

    class Child:
        pid = 4242

        def __init__(self, argv, **_kwargs):
            started["serve"] = argv[-1]

        def poll(self):
            return None

    def is_up(port):
        # The concurrent invocation boots its own server while this one waits
        # for its boot, and overwrites the shared state file.
        write_state(tmp_path, pid=4, port=6002, backend=".")
        return port == 5001

    ns.update(free_port=lambda: 5001, is_up=is_up, cmd_down=lambda quiet=False: 0,
              subprocess=SimpleNamespace(Popen=Child, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT,
                                         CREATE_NEW_PROCESS_GROUP=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)))

    assert ns["ensure_up"]() == 5001
    assert "port=5001" in started["serve"]


def test_a_recorded_port_is_reused_only_while_our_server_holds_it(tmp_path):
    workspace(tmp_path)
    ns = load_probe(tmp_path)
    # Something answers on the recorded port, but the recorded pid is not our
    # server (here: this test process), so the port belongs to someone else.
    write_state(tmp_path, pid=os.getpid(), port=6002, backend=".")
    booted = []
    ns.update(is_up=lambda port: True, boot=lambda quiet=False: booted.append(1) or (0, 5001))

    assert ns["ensure_up"]() == 5001
    assert booted


def test_req_sends_headers_and_prints_only_their_names(tmp_path):
    workspace(tmp_path)
    ns = load_probe(tmp_path)
    seen = {}

    def request(method, path, body, port, extra_headers=None, **_kwargs):
        seen.update(extra_headers or {})
        return 200, "{}"

    ns.update(ensure_up=lambda: 5001, request=request)
    # bash strips the quotes; cmd.exe passes single-quoted parts through split.
    for argv in (["GET", "/note", "-H", "Authorization: Bearer tok-123"],
                 ["GET", "/note", "-H", "'Authorization:", "Bearer", "tok-123'"],
                 ["GET", "/note", "--header=Authorization: Bearer tok-123"]):
        seen.clear()
        assert ns["cmd_req"](argv) == 0
        assert seen == {"Authorization": "Bearer tok-123"}, argv


def test_run_command_calls_in_one_turn_run_in_order(tmp_path, monkeypatch):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker

    events, active = [], []
    lock = threading.Lock()

    def run_command(self, args):
        with lock:
            active.append(args["command"])
            events.append(("start", args["command"], len(active)))
        time.sleep(0.2)
        with lock:
            active.remove(args["command"])
            events.append(("end", args["command"]))
        return {"stdout": args["command"], "exit_code": 0}

    monkeypatch.setitem(ToolExecutor._handlers, "run_command", run_command)
    client = SimpleNamespace(model="mock-model", usage=UsageTracker("mock-model"))
    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=DomainModel(name="Probe", types={Class(name="Note")}),
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
        enable_requirements_ledger=False, enable_toolchain_validation=False)
    orchestrator.executor.allow_shell = True
    blocks = [SimpleNamespace(id=f"t{i}", name="run_command", input={"command": f"cmd{i}"}) for i in range(3)]
    blocks.insert(1, SimpleNamespace(id="r", name="list_files", input={}))

    results = orchestrator._execute_tool_blocks(blocks, turn=0)

    assert [r["tool_use_id"] for r in results] == ["t0", "r", "t1", "t2"]
    starts = [event for event in events if event[0] == "start"]
    assert [event[1] for event in starts] == ["cmd0", "cmd1", "cmd2"]
    assert all(event[2] == 1 for event in starts), f"run_command calls overlapped: {events}"


@pytest.mark.parametrize("name", ["run_command", "install_dependencies"])
def test_shell_tools_share_one_serial_group(tmp_path, name):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker

    orchestrator = LLMOrchestrator(
        llm_client=SimpleNamespace(model="m", usage=UsageTracker("m")),
        domain_model=DomainModel(name="Probe", types={Class(name="Note")}), output_dir=str(tmp_path),
        enable_checkpointing=False, enable_tracing=False, enable_requirements_ledger=False,
        enable_toolchain_validation=False)
    first = SimpleNamespace(id="a", name="run_command", input={"command": "x"})
    second = SimpleNamespace(id="b", name=name, input={})
    assert orchestrator._serial_key(first) == orchestrator._serial_key(second)
