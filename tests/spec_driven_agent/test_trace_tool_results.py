"""A tool_call trace event records what the tool answered, not only what it was asked.

Run 438889bc could not be diagnosed from its `.besser_trace.jsonl`: every
tool_call carried the input, but the only trace of a probe's "Terminated" or a
401 was the model's reaction to it. The result is now recorded - redacted
before it is cut, and bounded so the trace stays small.
"""
import json
from types import SimpleNamespace

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbm4ifQ.c2lnbmF0dXJlLXZhbHVl"
PROVIDER_KEY = "sk-proj-" + "A" * 40


def tool_call_events(tmp_path):
    lines = (tmp_path / ".besser_trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [record["payload"] for record in map(json.loads, lines) if record["event"] == "tool_call"]


def test_tool_call_event_carries_a_redacted_truncated_result(tmp_path, monkeypatch):
    def run_command(self, args):
        return {"stdout": (f"200 POST /auth/login\n{{\"access_token\": \"{JWT}\"}}\n"
                           f"OPENAI={PROVIDER_KEY}\nTerminated\n" + "x" * 20000),
                "exit_code": 1, "error": f"command failed; key {PROVIDER_KEY}"}

    monkeypatch.setitem(ToolExecutor._handlers, "run_command", run_command)
    orchestrator = LLMOrchestrator(
        llm_client=SimpleNamespace(model="mock-model", usage=UsageTracker("mock-model")),
        domain_model=DomainModel(name="Probe", types={Class(name="Note")}), output_dir=str(tmp_path),
        enable_checkpointing=False, enable_requirements_ledger=False, enable_toolchain_validation=False)
    orchestrator.executor.allow_shell = True
    command = f"python .besser_probe.py req GET /note -H 'Authorization: Bearer {JWT}'"
    block = SimpleNamespace(id="t1", name="run_command", input={"command": command})

    orchestrator._execute_single_tool(block, turn=0)

    [event] = tool_call_events(tmp_path)
    result = event["result"]
    assert "Terminated" in result["excerpt"] and "200 POST /auth/login" in result["excerpt"]
    assert result["chars"] > 20000
    assert len(result["excerpt"]) < 700, "the excerpt must stay bounded"
    assert "truncated" in result["excerpt"]
    assert "command failed" in event["error"]
    serialized = json.dumps(event)
    assert JWT not in serialized and PROVIDER_KEY not in serialized
    assert "[REDACTED]" in event["input"]["command"]
