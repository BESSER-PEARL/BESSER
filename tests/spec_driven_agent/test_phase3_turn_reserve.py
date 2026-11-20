"""Phase 2 must leave Phase 3 a share of the run's tool-call budget.

Both phases count against one budget of executed tool calls (``max_turns``
times ``_CALLS_PER_TURN``). A model that never ends its turn used to spend
every turn in Phase 2, so validation found its blockers and had no turn left
to repair them. A Phase 2 cut at its share hands off to validation, and Phase 3
decides how the run ends.
"""
from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.state.checkpoint import compute_fingerprint

BLOCKER = ValidationIssue("blocker", "python contract: app/tasks.py line 1: boom")
TASKS = ("Add a due date filter to the Task list",)


@pytest.fixture
def task_model():
    task = Class(name="Task")
    task.attributes = {Property(name="title", type=PrimitiveDataType("str"))}
    return DomainModel(name="Tasks", types={task})


class _Block:
    def __init__(self, block_type: str, **kwargs) -> None:
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _NeverDoneClient:
    """Writes a new file every turn and never ends its turn."""

    model = "mock"

    def __init__(self):
        self.usage = UsageTracker("mock")
        self.calls = 0

    def chat(self, system, messages, tools, **_kwargs):
        self.calls += 1
        return {"stop_reason": "tool_use", "content": [_Block(
            "tool_use", id=f"w{self.calls}", name="write_file",
            input={"path": f"app/note_{self.calls}.py", "content": f"value = {self.calls}\n"},
        )]}


def _orchestrator(model, tmp_path, monkeypatch, client, tasks=TASKS, **kwargs) -> LLMOrchestrator:
    """``tasks`` is the Phase 2 checklist; none means gap analysis failed."""
    orch = LLMOrchestrator(
        llm_client=client, domain_model=model, output_dir=str(tmp_path),
        use_streaming=False, **kwargs,
    )
    monkeypatch.setattr(orchestrator_module, "analyze_gaps_via_llm", lambda **kw: list(tasks) or None)
    monkeypatch.setattr(orch, "_deterministic_gap_tasks", lambda: [])
    monkeypatch.setattr(orch, "_collect_validation_issues", lambda: [BLOCKER])
    monkeypatch.setattr(orch, "_apply_runtime_gate", lambda: None)
    return orch


def _record_phase3_entry(orch, monkeypatch) -> dict:
    seen: dict = {}
    original = orch._run_phase3_validation

    def wrapped():
        seen["turns"], seen["calls"] = orch.total_turns, orch.client.calls
        seen["stop_reason"] = orch._phase2_stop_reason
        original()

    monkeypatch.setattr(orch, "_run_phase3_validation", wrapped)
    return seen


class _FinishesAtClient(_NeverDoneClient):
    """Writes a new file every turn until ``finish``, then ends its turn."""

    def __init__(self, finish):
        super().__init__()
        self.finish = finish

    def chat(self, system, messages, tools, **kwargs):
        if self.calls + 1 >= self.finish:
            self.calls += 1
            return {"stop_reason": "end_turn", "content": [_Block("text", text="Done")]}
        return super().chat(system, messages, tools, **kwargs)


def test_a_model_that_never_finishes_still_gets_phase3_repair_calls(task_model, tmp_path, monkeypatch):
    client = _NeverDoneClient()
    orch = _orchestrator(task_model, tmp_path, monkeypatch, client, max_turns=30, auto_fix_issues=True)
    at_phase3 = _record_phase3_entry(orch, monkeypatch)

    orch.run("Build a task tracker")

    assert at_phase3["calls"] == 90
    # Handed off, not cut: the unrepaired blocker keeps the run incomplete.
    assert orch._phase2_stop_reason == "validation_required"
    assert "30 of 120 calls reserved for validation and repair" in orch._phase2_stop_detail
    assert not orch._phase2_exited_cleanly
    # One fix attempt with the calls left; it wrote files but retired nothing, so it is rolled back.
    assert client.calls - at_phase3["calls"] == 30, "Phase 3 must get its reserved calls"
    assert orch._tool_calls_executed() == 120, "the run's budget is never exceeded"
    assert orch._phase3_exit_reason == "attempt 1 reverted: it retired none of the 1 blocker it faced"


def _closes_every_item(orch):
    """Validation that finds nothing and settles the checklist, as a passing check does."""
    def collect():
        for task in orch.executor.open_tasks():
            task["done"] = True
        return []
    return collect


@pytest.mark.parametrize(("max_turns", "finish_at"), [(30, 95), (10, 35)])
def test_a_run_that_would_finish_inside_the_reserve_completes_when_validation_is_clean(
        task_model, tmp_path, monkeypatch, max_turns, finish_at):
    orch = _orchestrator(task_model, tmp_path, monkeypatch, _FinishesAtClient(finish_at),
                         max_turns=max_turns, auto_fix_issues=True)
    monkeypatch.setattr(orch, "_collect_validation_issues", _closes_every_item(orch))
    monkeypatch.setattr(orch, "_validate_app", lambda *a, **k: {"blocker_count": 0, "issues": []})

    orch.run("Build a task tracker")

    assert orch._phase2_stop_reason == "completed" and orch._phase2_exited_cleanly


def test_a_handoff_with_items_still_open_is_not_reported_completed(task_model, tmp_path, monkeypatch):
    """Validation turns an open item with no check into a 'task unverified' warning, which
    blocks nothing and is not repaired; the run read 'completed' with the work undone."""
    orch = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(), max_turns=30, auto_fix_issues=True)
    monkeypatch.setattr(orch, "_collect_validation_issues",
                        lambda: [_classify_issue(message) for message in orch._collect_task_issues()])
    at_phase3 = _record_phase3_entry(orch, monkeypatch)

    orch.run("Build a task tracker")

    assert at_phase3["stop_reason"] == "validation_required"
    assert [i.severity for i in orch._validation_issues] == ["warning"]
    assert orch._phase3_exit_reason == "no blocker-class issues"
    assert orch._phase2_stop_reason == "validation_required" and not orch._phase2_exited_cleanly


def test_a_cut_at_the_share_with_no_checklist_stays_a_handoff(task_model, tmp_path, monkeypatch):
    """The share is not the budget: a cut there is a handoff, not a step limit. With no
    checklist to show the work was finished, a clean validation does not complete the run:
    the model was still writing when it was cut."""
    orch = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(), tasks=(),
                         max_turns=30, auto_fix_issues=True)
    monkeypatch.setattr(orch, "_collect_validation_issues", lambda: [])
    at_phase3 = _record_phase3_entry(orch, monkeypatch)

    orch.run("Build a task tracker")

    assert at_phase3["calls"] == 90 and at_phase3["stop_reason"] == "validation_required"
    assert "30 of 120 calls reserved for validation and repair" in orch._phase2_stop_detail
    assert orch._phase2_stop_reason == "validation_required" and not orch._phase2_exited_cleanly


def test_a_resumed_repair_after_a_share_cut_with_no_checklist_stays_a_handoff(task_model, tmp_path, monkeypatch):
    first = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(), tasks=(),
                          max_turns=30, auto_fix_issues=True)
    first._instructions = "Build a task tracker"
    first._project_fingerprint = compute_fingerprint(
        instructions=first._instructions, primary_kind=first.primary_kind, domain_model=task_model)
    first._hand_off_phase2_share("tool-call", "30 of 120 calls")
    first._save_phase3_checkpoint()

    resumed = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(), tasks=(),
                            max_turns=30, auto_fix_issues=True)
    monkeypatch.setattr(resumed, "_collect_validation_issues", lambda: [])
    resumed.resume("Build a task tracker")

    assert resumed._phase3_exit_reason == "validation clean"
    assert resumed._phase2_stop_reason == "validation_required" and not resumed._phase2_exited_cleanly


@pytest.mark.parametrize(("max_turns", "auto_fix", "phase2_calls", "stop_reason"), [
    (10, True, 30, "validation_required"),   # a quarter of the budget
    (2, True, 6, "validation_required"),
    (10, False, 40, "max_turns"),            # no repair loop to reserve calls for
])
def test_the_reserve_scales_with_the_cap(task_model, tmp_path, monkeypatch, max_turns, auto_fix,
                                         phase2_calls, stop_reason):
    orch = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(),
                         max_turns=max_turns, auto_fix_issues=auto_fix)
    at_phase3 = _record_phase3_entry(orch, monkeypatch)

    orch.run("Build a task tracker")

    assert at_phase3["calls"] == phase2_calls
    assert orch._tool_calls_executed() == 4 * max_turns
    assert orch._phase2_stop_reason == stop_reason


def _resume_from_phase2_checkpoint(task_model, tmp_path, monkeypatch, turn, stop_reason, tasks=TASKS, **kwargs):
    first = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(), **kwargs)
    first.executor.set_tasks(list(tasks))
    first.tool_calls_log = [{"turn": number, "tool": "read_file", "input": {}, "success": True}
                            for number in range(1, turn + 1)]
    first._instructions = "Build a task tracker"
    first._project_fingerprint = compute_fingerprint(
        instructions=first._instructions, primary_kind=first.primary_kind, domain_model=task_model)
    first.total_turns, first._phase2_stop_reason = turn, stop_reason
    first._save_checkpoint_for_turn(turn, [{"role": "user", "content": "Build a task tracker"}], first._instructions)

    client = _NeverDoneClient()
    resumed = _orchestrator(task_model, tmp_path, monkeypatch, client, **kwargs)
    at_phase3 = _record_phase3_entry(resumed, monkeypatch)
    resumed.resume("Build a task tracker")
    return resumed, client, at_phase3


def test_a_resumed_phase2_past_its_share_goes_straight_to_repair(task_model, tmp_path, monkeypatch):
    """A Phase 2 checkpoint inside the reserve resumes into Phase 3 with what is left."""
    resumed, client, at_phase3 = _resume_from_phase2_checkpoint(
        task_model, tmp_path, monkeypatch, 100, "max_turns", max_turns=30, auto_fix_issues=True)

    assert at_phase3["calls"] == 0
    assert client.calls == 20 and resumed._tool_calls_executed() == 120
    assert resumed._phase2_stop_reason == "validation_required"


def test_a_resumed_phase2_with_no_checklist_past_its_share_hands_off_too(task_model, tmp_path, monkeypatch):
    resumed, client, at_phase3 = _resume_from_phase2_checkpoint(
        task_model, tmp_path, monkeypatch, 100, "api_error", tasks=(), max_turns=30, auto_fix_issues=True)

    assert at_phase3["stop_reason"] == "validation_required" and client.calls == 20
    assert resumed._phase2_stop_reason == "validation_required"


def test_a_resumed_phase2_that_ran_out_of_turns_does_not_keep_the_checkpoints_stop_reason(
        task_model, tmp_path, monkeypatch):
    resumed, client, _ = _resume_from_phase2_checkpoint(
        task_model, tmp_path, monkeypatch, 6, "api_error", max_turns=10, auto_fix_issues=False)

    assert client.calls == 34 and resumed._tool_calls_executed() == 40
    assert resumed._phase2_stop_reason == "max_turns"
