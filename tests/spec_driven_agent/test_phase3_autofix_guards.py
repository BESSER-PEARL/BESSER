"""Phase 3 auto-fix: web-path enablement + per-turn budget guards.

The web runner used to construct the orchestrator without ``auto_fix_issues``
(default ``False``), so Phase 3 was validate-and-report only: blocker-class
issues (Python syntax errors, broken Dockerfile refs, dependency conflicts)
were logged and the broken artifact was shipped as a green success. The agent
detected its own errors and never got to fix them.

Enabling the budgeted fix loop
for web runs required one hardening step this file pins down: the inner fix
loop must honour the SAME per-turn guards as the Phase 2 loop — cooperative
cancellation, the runtime cap, the turn cap and the cost cap — because repair turns
on an expensive model is not negligible spend.
"""

from __future__ import annotations

import time

import pytest

from besser.spec_driven_agent.orchestrator import LLMOrchestrator, ValidationIssue


class _MockStateMachine:
    name = "DummySM"


class _Usage:
    """Stub usage tracker with a directly settable estimated cost."""

    def __init__(self, estimated_cost: float = 0.0) -> None:
        self.estimated_cost = estimated_cost


class _CountingClient:
    """Scripted client: returns end_turn once, counts calls, never streams."""

    model = "mock-model"

    def __init__(self, estimated_cost: float = 0.0) -> None:
        self.usage = _Usage(estimated_cost)
        self.calls = 0

    def chat(self, system, messages, tools):
        self.calls += 1
        return {"stop_reason": "end_turn", "content": []}


def _build(tmp_path, client, **kwargs) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=client,
        state_machines=[_MockStateMachine()],
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
        **kwargs,
    )


_BLOCKER = ValidationIssue(severity="blocker", message="syntax [app/main.py]: invalid syntax")


# ---------------------------------------------------------------------------
# Per-turn guards in the fix loop
# ---------------------------------------------------------------------------


def test_fix_loop_stops_on_cancellation_before_any_llm_call(tmp_path, monkeypatch) -> None:
    client = _CountingClient()
    orch = _build(tmp_path, client, should_continue=lambda: False)
    orch._invoke_phase3_fix_loop([_BLOCKER], is_first_attempt=True)
    assert client.calls == 0
    monkeypatch.setattr(orch, "_collect_validation_issues", lambda: pytest.fail("cancelled run started validation"))
    orch._run_phase3_validation()
    assert orch._phase3_interrupted and orch._validation_issues

    allowed = [True]
    repairing = _build(tmp_path, client, should_continue=lambda: allowed[0], auto_fix_issues=True)
    checks = []
    def collect_once():
        checks.append(True)
        assert len(checks) == 1, "stop during repair must not trigger another coverage judgment"
        return [_BLOCKER]
    def cancelled_repair(*args):
        allowed[0] = False
        return 0
    monkeypatch.setattr(repairing, "_collect_validation_issues", collect_once)
    monkeypatch.setattr(repairing, "_invoke_phase3_fix_loop", cancelled_repair)
    repairing._run_phase3_validation()
    assert repairing._phase3_interrupted and repairing._validation_issues == [_BLOCKER]
    assert client.calls == 0

    # A stop received during an in-flight provider call must prevent its returned
    # tool batch from being applied; no actual mutation is needed for this test.
    allowed[0] = True
    inflight = _build(tmp_path, client, should_continue=lambda: allowed[0])
    def stopped_chat(**kwargs):
        allowed[0] = False
        return {"stop_reason": "tool_use", "content": []}
    monkeypatch.setattr(client, "chat", stopped_chat)
    monkeypatch.setattr(inflight, "_execute_tool_blocks", lambda *args: pytest.fail("executed tools after stop"))
    inflight._invoke_phase3_fix_loop([_BLOCKER], is_first_attempt=True)
    assert inflight._phase3_interrupted


def test_fix_loop_stops_on_cost_cap_before_any_llm_call(tmp_path) -> None:
    client = _CountingClient(estimated_cost=5.0)
    orch = _build(tmp_path, client, max_cost_usd=1.0)
    orch._invoke_phase3_fix_loop([_BLOCKER], is_first_attempt=True)
    assert client.calls == 0


def test_fix_loop_stops_on_runtime_cap_before_any_llm_call(tmp_path) -> None:
    client = _CountingClient()
    orch = _build(tmp_path, client, max_runtime_seconds=1)
    orch._start_time = time.monotonic() - 10_000  # run started "hours" ago
    orch._invoke_phase3_fix_loop([_BLOCKER], is_first_attempt=True)
    assert client.calls == 0


def test_fix_loop_runs_when_under_all_budgets(tmp_path) -> None:
    client = _CountingClient(estimated_cost=0.0)
    orch = _build(
        tmp_path, client,
        max_cost_usd=10.0, max_runtime_seconds=3600,
        should_continue=lambda: True,
    )
    orch._start_time = time.monotonic()
    turns_before = orch.total_turns
    orch._invoke_phase3_fix_loop([_BLOCKER], is_first_attempt=True)
    # end_turn with no edit is re-prompted once (modify_file forced), then
    # the attempt ends: two calls, both accounted as fix turns.
    assert client.calls == 2
    assert orch.total_turns == turns_before + 2

    # Repair continuation is budgeted by the run's total turns too; it must
    # never get a fresh allowance after Phase 2 has spent that budget.
    orch.total_turns = orch.max_turns
    assert orch._invoke_phase3_fix_loop([_BLOCKER], is_first_attempt=False) == 0
    assert client.calls == 2


# ---------------------------------------------------------------------------
# Web-path wiring
# ---------------------------------------------------------------------------


def test_auto_fix_flag_defaults_on() -> None:
    # CI/dev environments don't set BESSER_LLM_ENABLE_AUTO_FIX, so the
    # deploy-wide default must be ON — that's the whole enablement.
    from besser.utilities.web_modeling_editor.backend.constants.constants import (
        LLM_ENABLE_AUTO_FIX,
    )

    assert LLM_ENABLE_AUTO_FIX is True


def test_runner_passes_auto_fix_flag_to_orchestrator() -> None:
    """Source-level pin: the web runner must thread the deploy flag into
    the orchestrator ctor (this is exactly the wiring that was missing —
    the default-False parameter silently disabled the fix loop for every
    web run)."""
    import inspect

    from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
        runner,
    )

    source = inspect.getsource(runner)
    assert "auto_fix_issues=LLM_ENABLE_AUTO_FIX" in source
