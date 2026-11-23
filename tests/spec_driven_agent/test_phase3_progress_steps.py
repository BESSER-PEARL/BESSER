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


def _quiet_probe(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "_import_smoke_issues", lambda output_dir: [])
    monkeypatch.setattr(constructibility, "collect_constructibility_report",
                        lambda output_dir, model: {"issues": [], "backends": []})


def _sweep_without_llm_checks(orchestrator, monkeypatch):
    """Phase 3's own sweep on an empty workspace, its LLM and toolchain checks stubbed."""
    for name in ("_collect_requirement_issues", "_collect_tsc_issues", "_collect_task_issues",
                 "_collect_ruff_issues", "_authorization_issues"):
        monkeypatch.setattr(orchestrator, name, lambda: [])
    return orchestrator._collect_validation_issues()


def test_starting_the_app_is_named_only_when_the_probe_runs(orch, monkeypatch):  # noqa: F811
    steps = _steps(orch)
    _quiet_probe(monkeypatch)
    monkeypatch.setattr(orch, "_workspace_revision", lambda: "rev1")
    orch._begin_validation_round("First check")
    _sweep_without_llm_checks(orch, monkeypatch)
    _sweep_without_llm_checks(orch, monkeypatch)  # same source: the cached result, nothing started
    probe_steps = [step for step in steps if "starting the app's backend" in step]
    assert probe_steps == ["First check: starting the app's backend and creating test records"]


def test_validate_app_outside_phase3_names_no_step(orch, monkeypatch):  # noqa: F811
    """Phase 2's validate_app (a tool call, the end_turn gate) also boots the app: it is not validation."""
    steps = _steps(orch)
    _quiet_probe(monkeypatch)
    orch.executor.app_validator()
    assert steps == []


def test_a_fix_attempt_keeps_its_own_step_through_its_validate_app_calls(orch, monkeypatch):  # noqa: F811
    steps = _steps(orch)
    _quiet_probe(monkeypatch)

    validated: list[dict] = []

    def attempt(*_args):
        validated.append(orch.executor.app_validator())  # the model's validate_app call
        return 1

    revisions = iter(f"rev{n}" for n in range(100))
    with patch.object(orch, "_collect_validation_issues",
                      side_effect=[[ValidationIssue("blocker", IMPORT_ERROR)], []]),             patch.object(orch, "_create_snapshot", return_value=True),             patch.object(orch, "_workspace_revision", side_effect=lambda: next(revisions)),             patch.object(orch, "_repair_obligations_revision", return_value="obl"),             patch.object(orch, "_invoke_phase3_fix_loop", side_effect=attempt):
        orch._run_phase3_repair_cycle()
    assert len(validated) == 1
    assert steps == ["First check: checking the code for errors", "Fix attempt 1: fixing 1 blocker",
                     "Re-check after fix attempt 1: checking the code for errors"]


def test_a_fix_attempt_counts_agree_with_the_round_it_faces(orch):  # noqa: F811
    """A check that could not run is not confirmed: neither the round nor the attempt counts it."""
    steps = _steps(orch)
    not_run = "validation unverified: frontend build [frontend] did not run completely"
    _cycle(orch, [
        [ValidationIssue("blocker", IMPORT_ERROR), ValidationIssue("blocker", REQUIREMENT_MISSING),
         ValidationIssue("warning", not_run)],
        [],
    ])
    assert steps[1] == "Fix attempt 1: fixing 2 blockers (1 about requirements)"


def test_a_resumed_repair_does_not_call_its_first_sweep_the_first_check(orch):  # noqa: F811
    steps = _steps(orch)
    orch._repair_progress = {"attempts_run": 2}
    _cycle(orch, [[ValidationIssue("blocker", IMPORT_ERROR)], []])
    assert steps[:2] == ["Check after resuming: checking the code for errors", "Fix attempt 3: fixing 1 blocker"]


def test_a_three_argument_listener_that_fails_does_not_stop_validation(orch):  # noqa: F811
    def listener(turn, tool, status):  # the original public signature
        raise RuntimeError("listener down")

    orch.on_progress = listener
    _cycle(orch, [[ValidationIssue("blocker", IMPORT_ERROR)], []])
    assert orch._phase3_exit_reason == "all blockers fixed"


def test_a_failing_listener_does_not_stop_validation(orch):  # noqa: F811
    def boom(*_args):
        raise RuntimeError("listener down")

    orch.on_progress = boom
    _cycle(orch, [[ValidationIssue("blocker", IMPORT_ERROR)], []])
    assert orch._phase3_exit_reason == "all blockers fixed"
