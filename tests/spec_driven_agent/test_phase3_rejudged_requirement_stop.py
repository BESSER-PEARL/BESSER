"""No second fix attempt for a requirement the focused re-judge still could not verify.

Live evidence, r12-p81: after attempt 2 the only blocker was R1, ``requirement
unverified`` with an empty citation, and the focused re-judge on that exact source
had run. Attempt 3 spent four turns (a read, task_list, validate_app, two test_api)
and wrote nothing. The first attempt stays: implementing an absent behaviour may be
the fix.
"""

from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

R1 = {"id": 1, "kind": "rule", "text": "Flag aggressive communication", "status": "unverified",
      "evidence": "", "note": "no verdict returned"}


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    return LLMOrchestrator(llm_client=_Client(), domain_model=simple_library_book_model, output_dir=str(tmp_path),
                           enable_checkpointing=False, enable_tracing=False, auto_fix_issues=True)


def _run(orch, verdicts, *, rejudged=True, other_blockers=()):
    orch._requirement_verdicts = verdicts
    blockers = [ValidationIssue("blocker", message) for message in ledger.ledger_issues(verdicts)]
    blockers += [ValidationIssue("blocker", message) for message in other_blockers]
    orch._requirement_evidence_retries = {"judged-source"} if rejudged else set()
    revisions = (f"rev{n}" for n in itertools.count())  # every attempt changes the source
    with patch.object(orch, "_collect_validation_issues", return_value=blockers), \
            patch.object(orch, "_requirement_cache_key", return_value="judged-source"), \
            patch.object(orch, "_requirements_to_judge", return_value=[]), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: next(revisions)), \
            patch.object(orch, "_repair_obligations_revision", return_value="obl"), \
            patch.object(orch, "_invoke_phase3_fix_loop", return_value=1) as fix:
        orch._run_phase3_repair_cycle()
    return fix.call_count


def test_after_one_attempt_a_rejudged_unverified_requirement_ends_the_loop(orch):
    assert _run(orch, [R1]) == 1
    assert orch._phase3_exit_reason == "only requirements unverified after their re-judge remain"
    assert any("R1" in issue.message and issue.severity == "blocker" for issue in orch._validation_issues)


def test_without_its_rejudge_it_keeps_its_attempts(orch):
    assert _run(orch, [R1], rejudged=False) > 1


def test_a_missing_requirement_or_another_blocker_keeps_its_attempts(orch):
    assert _run(orch, [dict(R1, status="missing")]) > 1
    assert _run(orch, [R1], other_blockers=["action contract: POST /book/ returned 500"]) > 1


# Review finding: attempt 1 went to a startup crash, so the requirement never had its
# own attempt; and a cited body that is a scaffold, or a citation of code that exists
# nowhere, is fixed by implementing it.
R3 = {"id": 3, "kind": "action", "text": "Cancel a booking", "status": "unverified",
      "evidence": [{"path": "backend/routers/booking.py",
                    "line": "def cancel_booking(booking_id: int, db: Session = Depends(get_db)):"}],
      "note": "[citation 1 (backend/routers/booking.py): the cited action body is an unimplemented scaffold]"}
CRASH = "backend startup: ImportError: cannot import name 'Booking' from 'models'"


def _crash_then(orch, verdict):
    """Before attempt 1: crash + the requirement; after it: the requirement; after attempt 2: nothing."""
    orch._requirement_verdicts = [verdict]
    req = [ValidationIssue("blocker", message) for message in ledger.ledger_issues([verdict])]
    sequence = iter([[ValidationIssue("blocker", CRASH), *req], req, [], [], []])
    orch._requirement_evidence_retries = {"judged-source"}
    revisions = (f"rev{n}" for n in itertools.count())
    with patch.object(orch, "_collect_validation_issues", side_effect=lambda: next(sequence)), \
            patch.object(orch, "_requirement_cache_key", return_value="judged-source"), \
            patch.object(orch, "_requirements_to_judge", return_value=[]), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: next(revisions)), \
            patch.object(orch, "_repair_obligations_revision", return_value="obl"), \
            patch.object(orch, "_invoke_phase3_fix_loop", return_value=1) as fix:
        orch._run_phase3_repair_cycle()
    return fix.call_count


def test_an_attempt_spent_on_another_blocker_does_not_count(orch):
    assert _crash_then(orch, dict(R1)) == 2
    assert orch._phase3_exit_reason == "all blockers fixed"


def test_a_scaffold_body_or_an_absent_citation_keeps_its_attempts(orch):
    assert _run(orch, [R3]) > 1
    assert _run(orch, [{**R1, ledger.EVIDENCE_UNLOCATED: ledger.CITATION_ABSENT}]) > 1
    assert _crash_then(orch, R3) == 2


# Review 4: the stop compared whole messages, which hold the judge's note. In r12-p81 the
# edit of attempt 2 changed the judged tree, R1 was judged again with a re-worded note,
# and attempt 3 still ran; and identical messages after an attempt whose edits were all
# rejected stopped a round the loop deliberately retries.
def _r12_p81(orch, note_after_attempt_2):
    first = [dict(R1, id=4, note="risk_service.py was not provided"),
             dict(R1, id=6, note="risk_service.py was not provided")]
    states = iter([first, [dict(R1, note="Judged again with backend/risk_service.py shown: none found")],
                   *[[dict(R1, note=note_after_attempt_2)]] * 4])
    keys = iter(["k0", "k1", "k2", "k2", "k2", "k2"])
    current = {"key": "", "rev": 0}

    def collect():
        orch._requirement_verdicts = next(states)
        current["key"] = next(keys)
        orch._requirement_evidence_retries.add(current["key"])
        return [ValidationIssue("blocker", m) for m in ledger.ledger_issues(orch._requirement_verdicts)]

    edits = iter([1, 7, 0, 0, 0])

    def fix(*_a, **_k):
        written = next(edits)
        current["rev"] += bool(written)
        return written

    with patch.object(orch, "_collect_validation_issues", side_effect=collect), \
            patch.object(orch, "_requirement_cache_key", side_effect=lambda *_: current["key"]), \
            patch.object(orch, "_requirements_to_judge", return_value=[]), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: f"rev{current['rev']}"), \
            patch.object(orch, "_repair_obligations_revision", return_value="obl"), \
            patch.object(orch, "_invoke_phase3_fix_loop", side_effect=fix) as fixer:
        orch._run_phase3_repair_cycle()
    return fixer.call_count


def test_a_reworded_note_after_the_rejudge_still_stops_before_attempt_3(orch):
    assert _r12_p81(orch, "Judged again with backend/risk_service.py shown: the check is not evidenced") == 2
    assert orch._phase3_exit_reason == "only requirements unverified after their re-judge remain"


def test_an_attempt_whose_edits_were_all_rejected_is_retried(orch):
    orch._requirement_verdicts = [R1]
    blockers = [ValidationIssue("blocker", m) for m in ledger.ledger_issues([R1])]
    orch._requirement_evidence_retries = {"k"}
    outcomes, state = iter([0, 1, 0, 0]), {"rev": 0}

    def fix(*_a, **_k):
        written = next(outcomes)
        orch.tool_calls_log.append({"tool": "modify_file", "success": bool(written), "turn": 0, "input": {}})
        state["rev"] += written
        return written

    with patch.object(orch, "_collect_validation_issues", side_effect=lambda: list(blockers)), \
            patch.object(orch, "_requirement_cache_key", return_value="k"), \
            patch.object(orch, "_requirements_to_judge", return_value=[]), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: f"rev{state['rev']}"), \
            patch.object(orch, "_repair_obligations_revision", return_value="obl"), \
            patch.object(orch, "_invoke_phase3_fix_loop", side_effect=fix) as fixer:
        orch._run_phase3_repair_cycle()
    assert fixer.call_count > 1
