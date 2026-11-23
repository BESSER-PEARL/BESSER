"""A requirement the re-judge only re-words is not a retired blocker.

Live evidence, r12-p81: after attempt 2 the only blocker was R1, judged again
with a re-worded note; attempt 3 spent four turns (a read, task_list,
validate_app, two test_api) and wrote nothing. Attempts compare requirements by
id and status rank, not by message, so an attempt that leaves R1 standing
retired nothing: it is rolled back and the loop ends. (``requirement
unverified`` is not a requirement blocker the attempts rank: a judged-implemented
requirement whose citation was not located.)
"""

from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

R1 = {"id": 1, "kind": "rule", "text": "Flag aggressive communication", "status": "missing",
      "evidence": "", "note": "no verdict returned"}
CRASH = "backend startup: ImportError: cannot import name 'Booking' from 'models'"


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


def _cycle(orch, rounds):
    """The real cycle; each validation sets the verdicts of ``rounds`` in turn and returns
    their ledger blockers, after ``CRASH`` where a round lists it."""
    states = iter(rounds)
    revisions = (f"rev{n}" for n in itertools.count())  # every attempt changes the source

    def collect():
        verdicts, crash = next(states)
        orch._requirement_verdicts = verdicts
        return ([ValidationIssue("blocker", CRASH)] if crash else []) + [
            ValidationIssue("blocker", message) for message in ledger.ledger_issues(verdicts)]

    with patch.object(orch, "_collect_validation_issues", side_effect=collect), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", side_effect=lambda: next(revisions)), \
            patch.object(orch, "_repair_obligations_revision", return_value="obl"), \
            patch.object(orch, "_invoke_phase3_fix_loop", return_value=1) as fix:
        orch._run_phase3_repair_cycle()
    return fix.call_count


def test_a_reworded_note_after_the_rejudge_is_not_a_retired_requirement(orch):
    reworded = dict(R1, note="Judged again with backend/risk_service.py shown: the check is not evidenced")
    assert _cycle(orch, [([R1], False), ([reworded], False), ([R1], False)]) == 1
    assert orch._phase3_exit_reason == "attempt 1 reverted: it retired none of the 1 blocker it faced"


def test_an_attempt_spent_on_another_blocker_is_kept(orch):
    assert _cycle(orch, [([R1], True), ([R1], False), ([], False)]) == 2
    assert orch._phase3_exit_reason == "all blockers fixed"
