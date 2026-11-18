"""Phase 2 must leave Phase 3 a share of the run's turn cap.

Both phases count against one ``max_turns``. A model that never ends its
turn used to spend every turn in Phase 2, so validation found its blockers
and had no turn left to repair them.
"""
from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.checkpoint import compute_fingerprint

BLOCKER = ValidationIssue("blocker", "python contract: app/tasks.py line 1: boom")


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


def _orchestrator(model, tmp_path, monkeypatch, client, **kwargs) -> LLMOrchestrator:
    orch = LLMOrchestrator(
        llm_client=client, domain_model=model, output_dir=str(tmp_path),
        use_streaming=False, **kwargs,
    )
    monkeypatch.setattr(orch, "_collect_validation_issues", lambda: [BLOCKER])
    monkeypatch.setattr(orch, "_apply_runtime_gate", lambda: None)
    return orch


def _record_phase3_entry(orch, monkeypatch) -> dict:
    seen: dict = {}
    original = orch._run_phase3_validation

    def wrapped():
        seen["turns"], seen["calls"] = orch.total_turns, orch.client.calls
        original()

    monkeypatch.setattr(orch, "_run_phase3_validation", wrapped)
    return seen


def test_a_model_that_never_finishes_still_gets_phase3_repair_turns(task_model, tmp_path, monkeypatch):
    client = _NeverDoneClient()
    orch = _orchestrator(task_model, tmp_path, monkeypatch, client, max_turns=120, auto_fix_issues=True)
    at_phase3 = _record_phase3_entry(orch, monkeypatch)

    orch.run("Build a task tracker")

    assert at_phase3["turns"] == 100
    assert orch._phase2_stop_reason == "max_turns"
    assert "reserved for validation and repair" in orch._phase2_stop_detail
    assert client.calls - at_phase3["calls"] == 20, "Phase 3 must spend its reserved turns"
    assert orch.total_turns == 120, "the run's cap is never exceeded"
    assert orch._phase3_exit_reason == "turn budget exhausted"


@pytest.mark.parametrize(("max_turns", "auto_fix", "phase2_turns"), [
    (40, True, 30),     # a quarter of a small cap
    (8, True, 6),
    (3, True, 3),       # too small to split
    (120, False, 120),  # no repair loop to reserve turns for
])
def test_the_reserve_scales_with_the_cap(task_model, tmp_path, monkeypatch, max_turns, auto_fix, phase2_turns):
    orch = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(),
                         max_turns=max_turns, auto_fix_issues=auto_fix)
    at_phase3 = _record_phase3_entry(orch, monkeypatch)

    orch.run("Build a task tracker")

    assert at_phase3["turns"] == phase2_turns
    assert orch.total_turns == max_turns
    assert orch._phase2_stop_reason == "max_turns"


def test_a_resumed_phase2_past_its_share_goes_straight_to_repair(task_model, tmp_path, monkeypatch):
    """A Phase 2 checkpoint inside the reserve resumes into Phase 3 with what is left."""
    first = _orchestrator(task_model, tmp_path, monkeypatch, _NeverDoneClient(),
                          max_turns=120, auto_fix_issues=True)
    first._instructions = "Build a task tracker"
    first._project_fingerprint = compute_fingerprint(
        instructions=first._instructions, primary_kind=first.primary_kind, domain_model=task_model)
    first.total_turns = 110
    first._save_checkpoint_for_turn(110, [{"role": "user", "content": "Build a task tracker"}], first._instructions)

    client = _NeverDoneClient()
    resumed = _orchestrator(task_model, tmp_path, monkeypatch, client, max_turns=120, auto_fix_issues=True)
    at_phase3 = _record_phase3_entry(resumed, monkeypatch)

    resumed.resume("Build a task tracker")

    assert at_phase3 == {"turns": 110, "calls": 0}
    assert client.calls == 10 and resumed.total_turns == 120
