"""The Phase 3 stall guards must stop a dead loop without stopping a live one.

Measured over the 355 recorded runs in ``verification/spec-iterations``
(2026-09-19..20). Those runs split cleanly in two by the commit that introduced
the current guards, and the split is visible in the traces themselves: before
it, 243 fix rounds that wrote nothing were followed by another round; after it,
zero were, because the guard always ends the loop there.

What the 134 runs on the current guards actually do:

* 47 (35%) end on the zero-write stop, 46 (34%) on the plateau guard - 69% of
  all runs - holding a median of 60+ of their 120 turns and $4.76 of their $5.
* Run 053ydac9 (gpt-5.6-terra, library): the round the harness killed as
  "cannot have moved anything" had spent its turns on ``test_api`` /
  ``read_file`` / ``task_list`` and taken 10 blockers to 4. It wrote no source
  because the source was already right; what it discharged were verification
  obligations. It shipped 4 blockers with 86 turns and $4.13 unspent.

What the 221 runs on the older, looser guards say the extra round is worth:

* after ONE barren round, the next round wrote source 38% of the time and cut
  the blocker count 19% of the time (n=127); after TWO, 6% and 4% (n=116).
* after TWO consecutive non-improving rounds, the next still improved 36% of
  the time (n=125) against a 46% no-plateau baseline; at three the payoff
  halves (23%, n=60) and at four collapses (15%, n=107).

Hence: zero writes is one no-progress round, not its own stop; two barren
rounds still end the loop; three churning rounds still end the loop.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from besser.generators.llm.errors import InvalidApiKeyError
from besser.generators.llm.orchestrator import (
    _PHASE3_NO_PROGRESS_ROUNDS,
    _PHASE3_PLATEAU_ROUNDS,
    LLMOrchestrator,
    ValidationIssue,
)
from besser.generators.llm.llm_client import UsageTracker


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - overridden where needed
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
    )


def _blockers(n, prefix="action contract"):
    return [ValidationIssue("blocker", f"{prefix}: defect {i}") for i in range(n)]


class _MutableUsage:
    """Usage tracker whose cost the test can move mid-run."""

    def __init__(self, cost: float = 0.0) -> None:
        self.estimated_cost = cost


def _drive(orchestrator, rounds, entry_blockers=6, after_round=None):
    """Run the real Phase 3 cycle over a scripted sequence of rounds.

    Each round is ``(edits, wrote_source, discharged_obligations,
    blockers_after)``. ``discharged_obligations`` models a round that closed a
    checklist item or fixed a scenario: the task/scenario revision moves even
    though not one source byte did.
    """
    state = {"attempt": 0, "rev": 0, "obl": 0}

    def fix(blockers, is_first_attempt):
        edits, wrote, obligations, _ = rounds[min(state["attempt"], len(rounds) - 1)]
        if wrote:
            state["rev"] += 1
        if obligations:
            state["obl"] += 1
        state["attempt"] += 1
        if after_round is not None:
            after_round(state["attempt"])
        return edits

    def collect():
        if state["attempt"] == 0:
            return _blockers(entry_blockers)
        return rounds[min(state["attempt"] - 1, len(rounds) - 1)][3]

    with patch.object(orchestrator, "_collect_validation_issues", side_effect=collect), \
            patch.object(orchestrator, "_create_snapshot"), \
            patch.object(orchestrator, "_restore_snapshot", return_value=True), \
            patch.object(orchestrator, "_invoke_phase3_fix_loop", side_effect=fix), \
            patch.object(orchestrator, "_repair_obligations_revision",
                         side_effect=lambda: f"obl-{state['obl']}"), \
            patch.object(orchestrator, "_workspace_revision",
                         side_effect=lambda: f"rev-{state['rev']}"):
        orchestrator._run_phase3_validation()
    return state["attempt"]


# ---------------------------------------------------------------------------
# 1. Zero writes is not proof that nothing moved
# ---------------------------------------------------------------------------

def test_a_verification_round_that_writes_no_source_is_progress(orch):
    """Run 053ydac9, replayed: fix, then a verification round that discharges
    six blockers with no write, then the round that finishes the job.

    The old guard ended the run on round 2 with four blockers standing and 86
    of 120 turns unspent, because it asked "did it write" instead of "did
    anything move".
    """
    attempts = _drive(orch, [
        (5, True, False, _blockers(10)),    # wrote; 18 -> 10
        (0, False, True, _blockers(4)),     # verified only; 10 -> 4
        (2, True, False, []),               # finishes
    ], entry_blockers=18)

    assert attempts == 3


def test_revalidation_alone_can_carry_a_round(orch):
    """Nothing written, no obligation discharged, but the re-collected
    findings are strictly better - an earlier round's edit that only shows up
    once a downstream check can finally run. A strictly better tree score is
    the strongest progress evidence the harness has; it cannot be a stall."""
    attempts = _drive(orch, [
        (0, False, False, _blockers(3)),    # 6 -> 3 with no write at all
        (1, True, False, []),
    ])

    assert attempts == 2


# ---------------------------------------------------------------------------
# 2. ... and the loop still stops when nothing moves
# ---------------------------------------------------------------------------

def test_consecutive_barren_rounds_still_end_the_loop(orch):
    """The protection the zero-write stop existed to give, kept: a model that
    writes nothing, changes nothing and discharges nothing buys exactly
    ``_PHASE3_NO_PROGRESS_ROUNDS`` rounds - one more attempt, ~11 turns, and
    the cost/runtime/turn caps are untouched underneath it."""
    attempts = _drive(orch, [(0, False, False, _blockers(6))] * 8)

    assert attempts == _PHASE3_NO_PROGRESS_ROUNDS == 2
    assert orch._phase3_exit_reason == "no-progress streak"


def test_a_spent_cost_budget_still_ends_the_first_barren_round(orch):
    """Budget beats patience: the extra round is granted only while there is
    money for it. A barren round that also exhausts the cost cap ends the loop
    on the spot, streak counter or no streak counter."""
    orch.max_cost_usd = 1.0
    orch.client.usage = _MutableUsage(0.5)

    def spend(_attempt):
        orch.client.usage.estimated_cost = 1.0

    attempts = _drive(orch, [(0, False, False, _blockers(6))] * 8, after_round=spend)

    assert attempts == 1
    assert orch._phase3_exit_reason == "cost budget exhausted"


def test_rounds_that_churn_without_improving_still_end_the_loop(orch):
    """Run mbzbzhq9 ran six such attempts into the 120-turn cap. The plateau
    guard still catches it - one round later than before, which is the round
    the corpus says improves 36% of the time."""
    attempts = _drive(orch, [(1, True, False, _blockers(6))] * 8)

    assert attempts == _PHASE3_PLATEAU_ROUNDS == 3
    assert orch._phase3_exit_reason == "plateau"


def test_a_converging_repair_is_never_cut_short(orch):
    """11 -> 45 -> 40 -> 35 is a repair converging, not a stall."""
    attempts = _drive(orch, [
        (1, True, False, _blockers(45)),
        (1, True, False, _blockers(40)),
        (1, True, False, _blockers(35)),
        (1, True, False, _blockers(30)),
        (1, True, False, []),
    ])

    assert attempts == 5


# ---------------------------------------------------------------------------
# 3. The fix loop must report what it actually did
# ---------------------------------------------------------------------------

def test_the_edit_loop_guard_does_not_erase_the_attempts_writes(orch, monkeypatch):
    """``_apply_edit_loop_guards`` ending an attempt used to fall out of the
    function with no return, so the attempt reported ``None`` writes: the
    trace recorded 0, the log said "no successful edit", and the outer cycle
    scored a productive round as a barren one."""
    responses = [{"stop_reason": "tool_use", "content": []}]

    def chat(**kwargs):
        return responses[0]

    monkeypatch.setattr(orch.client, "chat", chat)
    monkeypatch.setattr(orch, "_execute_tool_blocks", lambda blocks, turn: [])
    monkeypatch.setattr(orch, "_apply_edit_loop_guards", lambda messages, where: True)
    monkeypatch.setattr(orch, "_save_phase3_checkpoint", lambda: None)
    # Two successful writes recorded by the executor during this attempt.
    orch.tool_calls_log.extend([
        {"turn": 1, "tool": "modify_file", "input": {}, "success": True},
        {"turn": 1, "tool": "write_file", "input": {}, "success": True},
    ])
    before = len(orch.tool_calls_log)

    def executed(blocks, turn):
        orch.tool_calls_log.append(
            {"turn": turn, "tool": "modify_file", "input": {}, "success": True})
        return []

    monkeypatch.setattr(orch, "_execute_tool_blocks", executed)
    assert before  # the pre-existing entries must not be counted
    edits = orch._invoke_phase3_fix_loop(_blockers(2), is_first_attempt=True)

    assert edits == 1


def test_an_interrupted_fix_loop_says_what_interrupted_it(orch, monkeypatch):
    """Ending mid-attempt is the third commonest exit (12 of the 134 runs on
    the current guards, holding a median of 87 turns), and 40 of the 43 such
    runs across the whole corpus were still making working, paid calls at the
    time. The trace said only "repair interrupted", which does not distinguish
    a dead provider from a truncated response - so it now names the cause."""
    monkeypatch.setattr(orch.client, "chat",
                        lambda **kw: {"stop_reason": "max_tokens", "content": []})
    orch._invoke_phase3_fix_loop(_blockers(2), is_first_attempt=True)

    assert orch._phase3_interrupted
    assert orch._phase3_interrupt_detail == "unexpected stop_reason 'max_tokens'"
    assert "unexpected stop_reason" in (orch._phase3_stop_requested() or "")


def test_an_invalid_api_key_in_the_fix_loop_propagates(orch, monkeypatch):
    """Phase 2 re-raises ``InvalidApiKeyError`` so the runner can report
    INVALID_KEY. The fix loop's blanket ``except Exception`` swallowed it and
    reported "repair interrupted" instead - hiding the one thing the user can
    actually fix."""
    def chat(**kwargs):
        raise InvalidApiKeyError("bad key")

    monkeypatch.setattr(orch.client, "chat", chat)

    with pytest.raises(InvalidApiKeyError):
        orch._invoke_phase3_fix_loop(_blockers(2), is_first_attempt=True)
