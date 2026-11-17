"""A test workflow the agent replaced can be retired instead of blocking the run forever.

Reproduced on a production run: the agent's first two ``test_api`` workflows for a
``decrease_stock`` method had wrong payloads. Instead of correcting them under the same
``scenario_id`` it wrote seven variants that passed. Every retained workflow is replayed
before the run ends, so the two broken first tries failed again, became blockers, and the
run card listed them as rules the user asked for. ``action='retire'`` (with the reason)
takes a replaced workflow out of the replay; the recipe keeps it, so the report says so.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation import api_probe
from besser.spec_driven_agent.validation.issues import _classify_issue


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


_FAILED = {"status": "failed", "boot": "ok", "backend": "backend", "responses": [],
           "assertion_failures": [{"request": 1, "expected": 200, "actual": 422}]}
_PASSED = {"status": "passed", "boot": "ok", "backend": "backend", "responses": []}
_DECREASE = [{"method": "POST", "path": "/booktergre/", "json": {"stock": 15}},
             {"method": "POST", "path": "/booktergre/1/methods/decrease_stock/", "json": {"quantity": 5}}]


@pytest.fixture
def orch(simple_library_book_model, tmp_path, monkeypatch):
    orchestrator = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
    )
    monkeypatch.setattr(orchestrator, "_probeable_backends", lambda: ["backend"])
    monkeypatch.setattr(orchestrator, "_workspace_revision", lambda: "rev-1")
    return orchestrator


def _run(orch, monkeypatch, scenario_id, report):
    monkeypatch.setattr(api_probe, "probe_api_scenario", lambda *_a, **_k: dict(report))
    orch._test_api({"scenario_id": scenario_id, "requests": _DECREASE})


def test_a_replaced_failing_workflow_stops_blocking_once_retired(orch, monkeypatch):
    _run(orch, monkeypatch, "test_decrease_stock", _FAILED)
    _run(orch, monkeypatch, "test_decrease_stock_simple", _PASSED)
    assert [_classify_issue(m).severity for m in orch._collect_api_scenario_issues()] == ["blocker"]

    answer = orch._test_api({"action": "retire", "scenario_id": "test_decrease_stock",
                             "correction_reason": "Wrong payload; test_decrease_stock_simple replaces it."})

    assert answer["retired"] == "test_decrease_stock"
    assert orch._collect_api_scenario_issues() == []
    listed = [s["scenario_id"] for s in orch._test_api({"action": "list"})["scenarios"]]
    assert listed == ["test_decrease_stock_simple"]
    assert orch._retired_api_scenarios == [{
        "scenario_id": "test_decrease_stock",
        "reason": "Wrong payload; test_decrease_stock_simple replaces it.",
        "last_status": "failed",
    }]


def test_the_failure_tells_the_agent_it_can_retire_a_replaced_workflow(orch, monkeypatch):
    _run(orch, monkeypatch, "test_decrease_stock", _FAILED)
    [message] = orch._collect_api_scenario_issues()
    assert "action='retire'" in message


@pytest.mark.parametrize("args, error", [
    ({"action": "retire", "scenario_id": "test_decrease_stock"}, "correction_reason"),
    ({"action": "retire", "scenario_id": "unknown", "correction_reason": "x"}, "Unknown scenario_id"),
    ({"action": "retire", "correction_reason": "x"}, "Unknown scenario_id"),
])
def test_retiring_needs_a_known_workflow_and_a_reason(orch, monkeypatch, args, error):
    _run(orch, monkeypatch, "test_decrease_stock", _FAILED)
    assert error in orch._test_api(args)["error"]
    assert len(orch._collect_api_scenario_issues()) == 1


def test_a_retired_workflow_run_again_is_retained_again(orch, monkeypatch):
    _run(orch, monkeypatch, "test_decrease_stock", _FAILED)
    orch._test_api({"action": "retire", "scenario_id": "test_decrease_stock", "correction_reason": "replaced"})
    _run(orch, monkeypatch, "test_decrease_stock", _PASSED)

    assert [s["scenario_id"] for s in orch._test_api({"action": "list"})["scenarios"]] == ["test_decrease_stock"]
    assert orch._retired_api_scenarios == []
