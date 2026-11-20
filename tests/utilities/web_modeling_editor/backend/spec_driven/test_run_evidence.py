"""A run's full record is kept server-side, apart from the size-bounded recipe the user downloads.

The recipe ships inside the app archive, so it is cut to a size budget: a long run
kept 5 of its 109 tool calls there, with their inputs shortened, and the trace
that held the rest lived in the run's workspace, which is deleted hours after the
run. The backend now appends every trace record, redacted, to
``<incident log dir>/runs/<run id>.jsonl`` as it happens.
"""
import asyncio
import io
import json
import logging
import os
import time
from types import SimpleNamespace

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.tracing import EVENT_PHASE_ENTER, EVENT_TOOL_CALL, TraceWriter
from besser.spec_driven_agent.validation.issues import ValidationIssue
from besser.utilities.web_modeling_editor.backend.services import cleanup
from besser.utilities.web_modeling_editor.backend.services.spec_driven import run_evidence, run_record
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import SmartGenerationRunner
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _FakeClient,
    _FakeOrchestrator,
    _build_request,
    _collect_frames,
)

RUN = "0123456789abcdef0123456789abcdef"
# A key with no known provider prefix: only the literal match finds it.
OWN_KEY = "Zq8mV3xK9pLr2Tw7Yb4Nc6Hd1Fg5Js0A"
PROVIDER_KEY = "sk-proj-" + "A" * 40
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbm4ifQ.c2lnbmF0dXJlLXZhbHVl"
CALLS = 150


class _Client:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")


def _orchestrator(output_dir, evidence=None) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
        output_dir=str(output_dir), enable_checkpointing=False, enable_requirements_ledger=False,
        enable_toolchain_validation=False, run_id=RUN, run_evidence=evidence)


def _records(incident_dir, run_id=RUN) -> list[dict]:
    path = incident_dir / "runs" / f"{run_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tool_calls(records) -> list[dict]:
    return [r for r in records if r["event"] == EVENT_TOOL_CALL]


def test_a_cut_recipe_says_what_it_kept_and_the_server_record_keeps_every_call(tmp_path, monkeypatch):
    incidents = tmp_path / "incidents"
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(incidents))
    monkeypatch.setitem(ToolExecutor._handlers, "modify_file",
                        lambda self, args: {"success": True, "path": args["path"], "log": "ok\n" * 3000})
    workspace = tmp_path / "besser_llm_run"
    workspace.mkdir()
    orchestrator = _orchestrator(workspace, run_evidence.RunEvidence(RUN, secrets=(OWN_KEY,)))
    orchestrator._trace.write(EVENT_PHASE_ENTER, phase="phase2")
    for turn in range(CALLS):
        block = SimpleNamespace(id=f"t{turn}", name="modify_file", input={
            "path": f"backend/routes_{turn % 5}.py", "old_text": f"old {turn} " + "o" * 900,
            "new_text": f"new {turn} " + "n" * 900})
        orchestrator._execute_single_tool(block, turn=turn)
    # Findings the recipe never cuts, so cutting the inputs is not enough and calls are dropped.
    orchestrator._validation_issues = [ValidationIssue("warning", f"{n} " + "w" * 1000) for n in range(115)]

    orchestrator._save_recipe("Track orders", 30.0)

    recipe = json.loads((workspace / ".besser_recipe.json").read_text(encoding="utf-8"))
    kept = len(recipe["tool_calls"])
    assert kept == 5 and recipe["tool_calls_count"] == CALLS
    assert recipe["recipe_shortened"]["note"].endswith(
        f"kept {kept} of {CALLS} tool calls; full record kept server-side under run {RUN}")

    calls = _tool_calls(_records(incidents))
    assert [c["payload"]["turn"] for c in calls] == list(range(1, CALLS + 1))
    first = calls[0]
    assert first["phase"] == "phase2" and first["run_id"] == RUN
    assert first["payload"]["input"]["new_text"] == "new 0 " + "n" * 900, "the input is whole"
    assert first["payload"]["success"] is True and first["payload"]["seconds"] >= 0
    result = first["payload"]["result"]
    assert len(result) > 9000 and result.startswith('{"success": true')
    assert "<<cut" not in result, "a result under the cap is whole"


def test_a_long_argument_and_result_are_cut_to_their_caps(tmp_path, monkeypatch):
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(tmp_path / "incidents"))
    record = run_evidence.RunEvidence(RUN)
    content = "a" * (run_evidence.VALUE_CHARS * 2)
    output = "HEAD" + "x" * (run_evidence.RESULT_CHARS * 4) + "TAIL"

    record({"ts": 0, "run_id": RUN, "event": EVENT_TOOL_CALL,
            "payload": {"tool": "write_file", "input": {"content": content}, "result": output}})

    [call] = _records(tmp_path / "incidents")
    written, result = call["payload"]["input"]["content"], call["payload"]["result"]
    assert len(written) < run_evidence.VALUE_CHARS + 100 and f"of {len(content)} chars" in written
    assert result.startswith("HEAD") and result.endswith("TAIL")
    assert len(result) < run_evidence.RESULT_CHARS + 100


def test_the_server_record_redacts_secrets_in_tool_arguments_and_results(tmp_path, monkeypatch):
    incidents = tmp_path / "incidents"
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(incidents))
    monkeypatch.setitem(ToolExecutor._handlers, "run_command", lambda self, args: {
        "stdout": f'{{"access_token": "{JWT}"}}\nOPENAI={PROVIDER_KEY}\nkey {OWN_KEY}\n', "exit_code": 0})
    orchestrator = _orchestrator(tmp_path, run_evidence.RunEvidence(RUN, secrets=(OWN_KEY,)))
    orchestrator.executor.allow_shell = True
    command = f"curl -H 'Authorization: Bearer {JWT}' -d '{OWN_KEY}' http://localhost/x"

    orchestrator._execute_single_tool(SimpleNamespace(id="t1", name="run_command", input={"command": command}),
                                      turn=0)

    [call] = _tool_calls(_records(incidents))
    stored = json.dumps(call)
    for secret in (OWN_KEY, PROVIDER_KEY, JWT):
        assert secret not in stored
    assert "[REDACTED]" in call["payload"]["input"]["command"]
    assert "[REDACTED]" in call["payload"]["result"]


def test_without_the_backend_a_run_writes_no_server_record(tmp_path, monkeypatch):
    incidents = tmp_path / "incidents"
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(incidents))
    monkeypatch.setitem(ToolExecutor._handlers, "modify_file", lambda self, args: {"success": True})
    orchestrator = _orchestrator(tmp_path / "out")

    orchestrator._execute_single_tool(
        SimpleNamespace(id="t1", name="modify_file", input={"path": "a.py", "old_text": "a", "new_text": "b"}),
        turn=0)

    assert not incidents.exists()
    assert (tmp_path / "out" / ".besser_trace.jsonl").is_file()


class _CrashesMidRun(_FakeOrchestrator):
    """Records two tool calls through the trace the runner wired, then fails."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace = TraceWriter(self.output_dir, run_id=kwargs["run_id"], mirror=kwargs["run_evidence"])

    def run(self, instructions: str) -> str:
        self.trace.write(EVENT_PHASE_ENTER, phase="phase2")
        for turn in (1, 2):
            self.trace.write(EVENT_TOOL_CALL, uncut={"input": {"api_key": _build_request().resolved_api_key()}},
                             turn=turn, tool="read_file", success=True)
        logging.getLogger("besser.spec_driven_agent.pipeline.orchestrator").warning("Phase 2: about to fail")
        raise RuntimeError("worker crashed")


def test_a_run_that_crashes_keeps_the_calls_it_made_and_logs_its_id(tmp_path, monkeypatch):
    incidents = tmp_path / "incidents"
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(incidents))
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _CrashesMidRun)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    besser_logger = logging.getLogger("besser")
    before = list(besser_logger.handlers)
    try:
        run_record.enable_run_log()
        [handler] = [h for h in besser_logger.handlers if h not in before]
        handler.setStream(written := io.StringIO())
        runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
        asyncio.run(_collect_frames(runner))
    finally:
        besser_logger.handlers[:] = before

    calls = _tool_calls(_records(incidents, runner.run_id))
    assert [c["payload"]["turn"] for c in calls] == [1, 2]
    assert _build_request().resolved_api_key() not in json.dumps(calls)
    assert f"WARNING:     run_id={runner.run_id} Phase 2: about to fail" in written.getvalue().splitlines()


def test_tool_calls_run_in_parallel_log_with_their_run_id(tmp_path, monkeypatch):
    def noisy_read(self, args):
        logging.getLogger("besser.spec_driven_agent.agent.tool_executor").warning("read %s", args["path"])
        return {"content": ""}

    monkeypatch.setitem(ToolExecutor._handlers, "read_file", noisy_read)
    orchestrator = _orchestrator(tmp_path)
    blocks = [SimpleNamespace(id=f"t{n}", name="read_file", input={"path": f"f{n}.py"}) for n in range(3)]
    besser_logger = logging.getLogger("besser")
    before = list(besser_logger.handlers)
    try:
        run_record.enable_run_log()
        [handler] = [h for h in besser_logger.handlers if h not in before]
        handler.setStream(written := io.StringIO())
        run_record.with_run_id(RUN, orchestrator._execute_tool_blocks)(blocks, 0)
        logging.getLogger("besser.spec_driven_agent").warning("outside any run")
    finally:
        besser_logger.handlers[:] = before

    lines = written.getvalue().splitlines()
    assert sorted(lines[:3]) == [f"WARNING:     run_id={RUN} read f{n}.py" for n in range(3)]
    assert lines[3] == "WARNING:     outside any run"


def test_the_hourly_cleanup_removes_a_record_past_its_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS", "30")
    runs = tmp_path / "runs"
    runs.mkdir()
    old, recent = runs / "old.jsonl", runs / "recent.jsonl"
    for path in (old, recent):
        path.write_text("{}\n", encoding="utf-8")
    month_ago = time.time() - 31 * 86400
    os.utime(old, (month_ago, month_ago))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(cleanup.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    cleanup.cleanup_old_temp_files()

    assert not old.exists() and recent.exists()
