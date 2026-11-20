"""Phase 3 names each step it is on, so the run card never shows a finished phase's last line.

Validation, requirement judging and fix attempts can together run for twenty minutes, with
single judge calls of one to two and a half minutes. Phase 3 reported only that validation
started, the first check's counts and each round's findings, so for minutes at a time the
card kept showing the last line of the phase before, and a working run looked hung.
"""

from __future__ import annotations

from unittest.mock import patch

from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import ValidationIssue
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.validation import constructibility
from tests.spec_driven_agent.test_phase3_findings_report import _cycle, orch  # noqa: F401
from tests.spec_driven_agent.test_rejudge_digest_gaps import REQUIREMENT
from tests.spec_driven_agent.test_rejudge_digest_gaps import orch as judged_orch  # noqa: F401

STEP = "__validation_step__"
IMPORT_ERROR = "import [backend/main.py]: No module named 'routers.order'"
REQUIREMENT_MISSING = "requirement missing: R1 — Orders can be cancelled. (no cancel route)"


def _steps(orchestrator) -> list[str]:
    steps: list[str] = []
    orchestrator.on_progress = (
        lambda turn, tool, status, *rest: steps.append(status) if tool == STEP else None)
    return steps


def test_each_round_and_fix_attempt_is_named(orch):  # noqa: F811
    steps = _steps(orch)
    _cycle(orch, [
        [ValidationIssue("blocker", IMPORT_ERROR), ValidationIssue("blocker", REQUIREMENT_MISSING)],
        [ValidationIssue("blocker", IMPORT_ERROR)],
        [],
    ])
    assert steps == [
        "First check: checking the code for errors",
        "Fix attempt 1: fixing 2 blockers (1 about requirements)",
        "Re-check after fix attempt 1: checking the code for errors",
        "Fix attempt 2: fixing 1 blocker",
        "Re-check after fix attempt 2: checking the code for errors",
    ]


def test_the_end_of_validation_is_named(orch):  # noqa: F811
    """Recipe and packaging follow; the card must not keep the last check on screen."""
    steps = _steps(orch)
    with patch.object(orch, "_collect_validation_issues", return_value=[]), \
            patch.object(orch, "_apply_runtime_gate"):
        orch._run_phase3_validation()
    assert steps == ["First check: checking the code for errors", "Packaging the result"]


def test_judging_and_re_judging_are_named_before_each_call(judged_orch, monkeypatch):  # noqa: F811
    """The judge is one LLM call per round; its step is named before the call, with the count."""
    steps = _steps(judged_orch)
    current_at_call: list[str] = []
    answers = [{"status": "unverified", "evidence": "", "note": "the handler was not shown"},
               {"status": "unverified", "evidence": "", "note": "still not found"}]

    def judge(requirements, digest, client, **kwargs):
        current_at_call.append(steps[-1] if steps else "")
        return [dict(REQUIREMENT, **answers.pop(0))]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    judged_orch._collect_requirement_issues()
    assert len(current_at_call) == 2
    assert current_at_call[0].endswith(": checking 1 requirement against the code")
    assert current_at_call[1].endswith(": checking 1 requirement again with the relevant files shown")


def test_unchanged_verdicts_are_counted_apart(judged_orch, monkeypatch):  # noqa: F811
    steps = _steps(judged_orch)
    judged_orch._requirements = [dict(REQUIREMENT), {"id": 2, "text": "Orders list their items", "kind": "action"}]
    monkeypatch.setattr(judged_orch, "_requirements_still_verified",
                        lambda scope, imports: {1: {**REQUIREMENT, "status": "implemented"}})
    monkeypatch.setattr(ledger, "judge_coverage", lambda requirements, *a, **k: None)
    judged_orch._collect_requirement_issues()
    assert steps[0].endswith(": checking 1 of 2 requirements against the code (1 unchanged since the last check)")


def test_starting_the_app_is_named_only_when_the_probe_runs(orch, monkeypatch):  # noqa: F811
    steps = _steps(orch)
    monkeypatch.setattr(orchestrator_module, "_import_smoke_issues", lambda output_dir: [])
    monkeypatch.setattr(constructibility, "collect_constructibility_report",
                        lambda output_dir, model: {"issues": [], "backends": []})
    monkeypatch.setattr(orch, "_workspace_revision", lambda: "rev1")
    orch._runtime_probe_issues()
    orch._runtime_probe_issues()  # same source: the cached result, nothing started
    assert len(steps) == 1 and steps[0].endswith(": starting the app's backend and creating test records")


def test_a_failing_listener_does_not_stop_validation(orch):  # noqa: F811
    def boom(*_args):
        raise RuntimeError("listener down")

    orch.on_progress = boom
    _cycle(orch, [[ValidationIssue("blocker", IMPORT_ERROR)], []])
    assert orch._phase3_exit_reason == "all blockers fixed"
