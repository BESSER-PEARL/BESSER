"""End-to-end tests for the crash-resume flow.

Drives an orchestrator against a mock LLM until a checkpoint is written,
simulates a crash, then runs ``resume()`` on a fresh orchestrator and
verifies it picks up from the saved turn without re-executing completed
tool calls.
"""

from __future__ import annotations

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.checkpoint import (
    CHECKPOINT_FILENAME,
    compute_fingerprint,
    load_checkpoint,
)
from besser.spec_driven_agent.llm_client import UsageTracker
from besser.spec_driven_agent.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.tracing import TRACE_FILENAME


@pytest.fixture
def simple_model():
    StringType = PrimitiveDataType("str")
    user = Class(name="User")
    user.attributes = {Property(name="name", type=StringType)}
    return DomainModel(name="Blog", types={user})


class _MockBlock:
    def __init__(self, block_type: str, **kwargs) -> None:
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _ScriptedClient:
    """Mock client with a pre-scripted sequence of responses.

    Each ``chat`` call consumes one entry. Tests can introspect ``calls``
    to check how many turns happened on each leg of the run.
    """

    model = "mock"

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[dict] = []
        self.usage = UsageTracker("mock")

    def chat(self, system, messages, tools):
        self.calls.append({"messages": list(messages)})
        if not self.script:
            return {"stop_reason": "end_turn", "content": [
                _MockBlock("text", text="Done"),
            ]}
        return self.script.pop(0)


def _tool_use(tool_id: str, tool_name: str = "list_files", **input_):
    return {
        "stop_reason": "tool_use",
        "content": [
            _MockBlock("tool_use", id=tool_id, name=tool_name, input=input_),
        ],
    }


def _end_turn():
    return {
        "stop_reason": "end_turn",
        "content": [_MockBlock("text", text="All done")],
    }


def test_run_writes_checkpoint_per_turn(simple_model, tmp_path):
    """After each Phase 2 turn with tool use, a checkpoint must exist
    on disk so a crash can be recovered.
    """
    client = _ScriptedClient([
        _tool_use("c1"),        # turn 1 → uses tool, save ckpt
        _tool_use("c2"),        # turn 2 → uses tool, save ckpt
        _end_turn(),            # turn 3 → finish, ckpt deleted on success
    ])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=5,
        use_streaming=False,
    )
    orch.run("Build a blog")

    # Successful run deletes its own checkpoint so a later re-run
    # against the same output dir starts clean.
    assert not os.path.isfile(tmp_path / CHECKPOINT_FILENAME)

    # But the trace file must survive (it's the audit log).
    assert os.path.isfile(tmp_path / TRACE_FILENAME)


def test_checkpoint_survives_a_crash(simple_model, tmp_path):
    """Simulate a crash by raising mid-run. The checkpoint file from
    the last completed turn must remain on disk.
    """
    class _CrashingClient(_ScriptedClient):
        def chat(self, system, messages, tools):
            result = super().chat(system, messages, tools)
            if len(self.calls) >= 3:
                raise RuntimeError("simulated crash on 3rd call")
            return result

    client = _CrashingClient([
        _tool_use("c1"),        # turn 1 → ckpt saved
        _tool_use("c2"),        # turn 2 → ckpt saved
        _tool_use("c3"),        # turn 3 → crash BEFORE finishing turn
    ])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    # The orchestrator catches the client exception and breaks — so
    # .run() returns normally, but the saved checkpoint reflects the
    # last fully-completed turn.
    orch.run("Build a blog")

    # Checkpoint must have been written. Either turn 2 (if ckpt was
    # saved after turn 2 completed) or turn 1 (if crash interrupted
    # turn-2 save). Either way, the file exists.
    ckpt = load_checkpoint(str(tmp_path))
    assert ckpt is not None
    assert ckpt.turn >= 1
    assert ckpt.instructions == "Build a blog"
    assert ckpt.primary_kind == "class"
    assert len(ckpt.messages) > 0


def test_resume_refuses_when_fingerprint_differs(simple_model, tmp_path):
    """Resuming against a different project / instructions is refused."""
    # Set up a checkpoint with the original model
    client = _ScriptedClient([_tool_use("c1"), _tool_use("c2")])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=2,
        use_streaming=False,
    )
    orch.run("Build a blog")
    # Manually re-save a checkpoint (run() deletes it on clean exit)
    # by running again with a crashing client.
    class _Crasher(_ScriptedClient):
        def chat(self, *a, **kw):
            r = super().chat(*a, **kw)
            if len(self.calls) >= 2:
                raise RuntimeError("crash")
            return r

    crasher = _Crasher([_tool_use("c1"), _tool_use("c2")])
    orch2 = LLMOrchestrator(
        llm_client=crasher,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    orch2.run("Build a blog")

    assert os.path.isfile(tmp_path / CHECKPOINT_FILENAME)

    # Now try to resume with different instructions → fingerprint mismatch
    resume_client = _ScriptedClient([_end_turn()])
    resumer = LLMOrchestrator(
        llm_client=resume_client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        use_streaming=False,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        resumer.resume("Completely different request")


def test_resume_picks_up_from_saved_turn(simple_model, tmp_path):
    """Happy path: crash after turn 2, resume, finish at turn 3.

    The resume orchestrator must not re-execute tool calls from turns
    1-2 — the LLM call count on the resume leg should start from turn 3.
    """
    # --- Leg 1: crash after saving a checkpoint --------------------------
    class _Crasher(_ScriptedClient):
        def chat(self, *a, **kw):
            r = super().chat(*a, **kw)
            if len(self.calls) >= 3:
                raise RuntimeError("crash")
            return r

    leg1 = _Crasher([
        _tool_use("c1"),   # turn 1
        _tool_use("c2"),   # turn 2 → ckpt saved
        _tool_use("c3"),   # turn 3 → crash partway through
    ])
    orch1 = LLMOrchestrator(
        llm_client=leg1,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    orch1.run("Build a blog")

    ckpt = load_checkpoint(str(tmp_path))
    assert ckpt is not None
    saved_turn = ckpt.turn
    assert saved_turn >= 1

    # --- Leg 2: resume and finish ---------------------------------------
    leg2 = _ScriptedClient([_end_turn()])
    orch2 = LLMOrchestrator(
        llm_client=leg2,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=10,
        use_streaming=False,
    )
    orch2.resume("Build a blog")

    # Exactly ONE LLM call was made on the resume leg — it ended with
    # end_turn so the loop exited immediately.
    assert len(leg2.calls) == 1

    # Checkpoint is gone after clean completion.
    assert not os.path.isfile(tmp_path / CHECKPOINT_FILENAME)


def test_unclean_resume_keeps_checkpoint_for_a_second_resume(simple_model, tmp_path):
    """A resume can itself hit a cap; that must not destroy recovery state."""
    first = _ScriptedClient([_tool_use("c1")])
    LLMOrchestrator(
        llm_client=first,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=1,
        use_streaming=False,
    ).run("Build a blog")
    assert load_checkpoint(str(tmp_path)) is not None

    second = _ScriptedClient([_tool_use("c2")])
    second_orchestrator = LLMOrchestrator(
        llm_client=second,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=2,
        use_streaming=False,
    )
    second_orchestrator.resume("Build a blog")

    after_second = load_checkpoint(str(tmp_path))
    assert second_orchestrator._phase2_exited_cleanly is False
    assert after_second is not None
    assert after_second.turn >= 2

    third = _ScriptedClient([_end_turn()])
    LLMOrchestrator(
        llm_client=third,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=3,
        use_streaming=False,
    ).resume("Build a blog")
    # The editing-turn cap must still permit final validation; it must not
    # force an extra resume when the last allowed turn finishes the work.
    assert len(third.calls) == 1
    assert load_checkpoint(str(tmp_path)) is None


@pytest.mark.parametrize("external_edit", [False, True])
def test_repair_resume_retains_corrected_state_but_rechecks_results(simple_model, tmp_path, monkeypatch, external_edit):
    source = tmp_path / "app.py"
    source.write_text("value = 1\n", encoding="utf-8")
    corrected_requests = [{"method": "GET", "path": "/items/", "expected_status": 200}]
    client = _ScriptedClient([{"stop_reason": "tool_use", "content": [
        _MockBlock("tool_use", id="edit", name="modify_file", input={
            "path": "app.py", "old_text": "value = 1\n", "new_text": "value = 2\n"}),
    ]}, {"stop_reason": "tool_use", "content": [
        _MockBlock("tool_use", id="task", name="task_list", input={"action": "done", "id": 1}),
        _MockBlock("tool_use", id="scenario", name="test_api", input={
            "scenario_id": "items", "requests": corrected_requests,
            "correction_reason": "Listing existing items returns 200, not creation status 201."}),
    ]}])
    client.usage.seed_cost(0.25)
    first = LLMOrchestrator(
        llm_client=client, domain_model=simple_model, output_dir=str(tmp_path),
        max_turns=120, auto_fix_issues=True, use_streaming=False,
        should_continue=lambda: not first._api_scenarios.get("named:items", {}).get("correction_history"),
    )
    first._instructions = "Build a blog"
    first._project_fingerprint = compute_fingerprint(
        instructions=first._instructions, primary_kind=first.primary_kind, domain_model=simple_model)
    first.total_turns = 32
    first._phase2_exited_cleanly = True
    first._phase2_stop_reason = "completed"
    tasks = [{"text": "Correct value", "verify": lambda: "value = 2" in source.read_text()}]
    first.executor.set_tasks(tasks)
    first._api_scenarios = {"named:items": {
        "scenario_id": "items", "scenario": {"backend": None, "requests": [
            {"method": "GET", "path": "/items/", "expected_status": 201}]},
        "report": {"status": "failed"}, "revision": None,
    }}
    blocker = ValidationIssue("blocker", "api scenario: items has an incorrect expectation")
    monkeypatch.setattr(first, "_collect_validation_issues", lambda: [blocker])
    monkeypatch.setattr("besser.spec_driven_agent.api_probe.probe_api_scenario",
                        lambda *args, **kwargs: {"status": "passed", "boot": "passed"})
    first._run_phase3_validation()
    first._finish_checkpoint()

    saved = load_checkpoint(str(tmp_path))
    assert saved.phase == "phase3" and saved.turn == saved.total_turns == 34
    assert saved.messages == [] and saved.estimated_cost_usd == pytest.approx(0.25)
    assert saved.tasks[0]["done"] is True
    assert saved.api_scenarios[0]["scenario"]["requests"] == corrected_requests
    assert saved.api_scenarios[0]["correction_history"][0]["previous_status"] == "failed"
    assert saved.repair_progress["attempts_run"] == 1
    assert saved.source_revision == first._workspace_revision()
    assert source.read_text() == "value = 2\n"

    if external_edit:
        source.write_text("value = 2\n# changed after interruption\n", encoding="utf-8")
    fresh_client = _ScriptedClient([])
    resumed = LLMOrchestrator(
        llm_client=fresh_client, domain_model=simple_model, output_dir=str(tmp_path),
        max_turns=120, auto_fix_issues=False, use_streaming=False,
    )
    monkeypatch.setattr(resumed, "_deterministic_gap_tasks", lambda: tasks)
    monkeypatch.setattr(resumed, "_run_phase2", lambda *a, **kw: pytest.fail("replayed Phase 2"))

    def fresh_validation():
        assert resumed._resume_messages is None
        assert resumed.executor.task_snapshot()[0]["done"] is True
        record = resumed._api_scenarios["named:items"]
        assert record["scenario"]["requests"] == corrected_requests
        assert record["report"]["status"] == "unverified" and record["revision"] is None
        assert resumed._repair_progress == ({} if external_edit else saved.repair_progress)
        return [blocker]

    monkeypatch.setattr(resumed, "_collect_validation_issues", fresh_validation)
    resumed.resume("Build a blog")
    assert fresh_client.calls == [] and resumed.total_turns == 34
    assert fresh_client.usage.estimated_cost == pytest.approx(0.25)
    assert load_checkpoint(str(tmp_path)).phase == "phase3", "unresolved repair must remain resumable"
    resumed._validation_issues = []
    resumed._finish_checkpoint()
    assert load_checkpoint(str(tmp_path)) is None


def test_resume_fails_when_no_checkpoint(simple_model, tmp_path):
    """Calling ``resume()`` with no saved checkpoint is an explicit error."""
    client = _ScriptedClient([_end_turn()])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        use_streaming=False,
    )
    with pytest.raises(FileNotFoundError):
        orch.resume("Anything")


def test_trace_file_is_populated_during_run(simple_model, tmp_path):
    """The structured trace must capture phase boundaries + per-turn
    events so post-run tooling can replay what happened."""
    client = _ScriptedClient([_tool_use("c1"), _end_turn()])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=simple_model,
        output_dir=str(tmp_path),
        max_turns=3,
        use_streaming=False,
    )
    orch.run("Build a blog")

    trace_path = tmp_path / TRACE_FILENAME
    assert trace_path.is_file()

    events: list[dict] = []
    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    event_names = [e["event"] for e in events]
    # Canonical run shape: start → phase enters/exits → end
    assert event_names[0] == "run_start"
    assert event_names[-1] == "run_end"
    assert "phase_enter" in event_names
    assert "turn_start" in event_names
    # The tool call we scripted must appear
    tool_events = [e for e in events if e["event"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["payload"]["tool"] == "list_files"
