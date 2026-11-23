"""Every paid call leaves a trace record.

Live evidence: r11-p81 traced $1.32 of cost_update while the run cost $1.64.
Routing, gap analysis, requirements extraction and every requirements-judge
call left nothing, and so did the closing end_turn reply of Phase 2 (its
cost_update was written only after a tool_use turn). r9-p21's analyst could not
tell whether the citation re-judge had run at all.
"""

import json

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _judge_outcome
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

REQUEST = ("## Assistant's notes (may be inaccurate)\n\nUse auth.\n\n"
           "## The user's original request, verbatim\n\n"
           "This is the authority. Where the summary above is shorter or differs, this text wins.\n\n"
           "Show the wastewater plants on a map.")


class _Usage:
    input_tokens = 1000
    output_tokens = 200
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _EndsTurn:
    model = "mock"
    max_tokens = None

    def __init__(self):
        self.usage = UsageTracker("mock")

    def chat(self, system, messages, tools, **_):
        self.usage.record(_Usage())
        return {"stop_reason": "end_turn", "content": [type("T", (), {"type": "text", "text": "done"})()]}


def _events(tmp_path, name):
    lines = (tmp_path / ".besser_trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if json.loads(line)["event"] == name]


def _orch(tmp_path, client):
    return LLMOrchestrator(llm_client=client, domain_model=shelf_model(author=True), output_dir=str(tmp_path),
                           enable_checkpointing=False, enable_import_smoke_check=False,
                           enable_requirements_ledger=False, use_streaming=False, max_turns=1)


def test_a_block_with_a_call_writes_one_llm_call_event(tmp_path):
    client = _EndsTurn()
    orch = _orch(tmp_path, client)
    with orch._traced_llm_call("requirements_judge") as call:
        client.chat("s", [], [])
        call.update(_judge_outcome([{"id": 1}, {"id": 2}], [{"status": "implemented"}, {"status": "missing"}]))
    with orch._traced_llm_call("gap_analysis"):
        pass  # no call: nothing to record

    (event,) = _events(tmp_path, "llm_call")
    payload = event["payload"]
    assert payload["purpose"] == "requirements_judge" and payload["outcome"] == "ok"
    assert payload["api_calls"] == 1 and payload["input_tokens"] == 1000 and payload["output_tokens"] == 200
    assert payload["cost_usd"] >= 0 and payload["verdicts"] == {"implemented": 1, "missing": 1}


def test_a_failed_judge_is_recorded_even_without_a_call(tmp_path):
    orch = _orch(tmp_path, _EndsTurn())
    with orch._traced_llm_call("requirements_rejudge") as call:
        call.update(_judge_outcome([{"id": 3}], None))
    (event,) = _events(tmp_path, "llm_call")
    assert event["payload"]["outcome"] == "failed" and event["payload"]["requirements"] == 1


class _Structured:
    """A provider that answers every forced tool call, recording its usage."""

    model = "mock"
    max_tokens = None
    planning_model = None
    _client = object()

    def __init__(self, answer):
        self.usage = UsageTracker("mock")
        self.answer = answer

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.usage.record(_Usage())
        return {"content": [{"type": "tool_use", "input": dict(self.answer)}]}


def test_a_modify_runs_routing_and_model_sync_calls_are_recorded(tmp_path):
    # Without a record, their tokens were read as part of the first loop turn.
    routing = _Structured({"target_kind": "backend", "target_description": "REST API",
                           "has_user_interface": False, "changes_target": False, "is_failure_report": False,
                           "reported_entity": "", "unsupported_stack": "", "generator": "NONE"})
    orch = _orch(tmp_path, routing)
    orch._instructions = "add a search box"
    orch._modify_mode = True
    orch._route_modify_request()
    orch.client = _Structured({"new_classes": []})
    orch._derive_and_apply_model_deltas("add a search box")

    calls = [event["payload"] for event in _events(tmp_path, "llm_call")]
    assert [(call["purpose"], call["api_calls"], call["input_tokens"]) for call in calls] == [
        ("routing", 1, 1000), ("model_deltas", 1, 1000)]


def test_a_modify_routing_call_with_no_usable_answer_is_recorded_as_failed(tmp_path):
    orch = _orch(tmp_path, _Structured({}))
    orch._instructions = "add a search box"
    orch._modify_mode = True
    orch._route_modify_request()

    (event,) = _events(tmp_path, "llm_call")
    assert event["payload"]["purpose"] == "routing" and event["payload"]["outcome"] == "failed"


def test_phase2_records_the_cost_of_the_closing_reply(tmp_path, monkeypatch):
    generate_backend(tmp_path, shelf_model(author=True))
    monkeypatch.setattr(orchestrator_module, "analyze_gaps_via_llm", lambda **_: None)
    orch = _orch(tmp_path, _EndsTurn())
    orch._generator_used = GENERATOR
    orch._instructions = "Build the shelf API"
    orch._run_phase2(orch._instructions)

    updates = _events(tmp_path, "cost_update")
    assert updates and updates[0]["payload"]["turn"] == 1 and updates[0]["payload"]["input_tokens"] >= 1000


def test_the_run_start_event_shows_the_users_own_request(tmp_path, monkeypatch):
    orch = _orch(tmp_path, _EndsTurn())

    class _Stop(Exception):
        pass

    def stop(*_):
        raise _Stop

    monkeypatch.setattr(orch, "_collect_model_contract_issues", stop)
    try:
        orch.run(REQUEST)
    except _Stop:
        pass
    (event,) = _events(tmp_path, "run_start")
    assert event["payload"]["instructions"] == "Show the wastewater plants on a map."
