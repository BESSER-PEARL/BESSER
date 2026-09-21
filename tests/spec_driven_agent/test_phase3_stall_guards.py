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

from besser.spec_driven_agent.errors import InvalidApiKeyError
from besser.spec_driven_agent.pipeline.orchestrator import (
    _PHASE3_NO_PROGRESS_ROUNDS,
    _PHASE3_PLATEAU_ROUNDS,
    LLMOrchestrator,
    ValidationIssue,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import _hard_blockers


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


def _drive(orchestrator, rounds, entry_blockers=6, after_round=None,
           rejected_edits=0):
    """Run the real Phase 3 cycle over a scripted sequence of rounds.

    Each round is ``(edits, wrote_source, discharged_obligations,
    blockers_after)``. ``discharged_obligations`` models a round that closed a
    checklist item or fixed a scenario: the task/scenario revision moves even
    though not one source byte did. ``rejected_edits`` makes every round log
    that many REFUSED ``modify_file`` calls, the way a real attempt whose
    edits the executor rejected does - a zero-write round that still reached
    for the editor.
    """
    state = {"attempt": 0, "rev": 0, "obl": 0}

    def fix(blockers, is_first_attempt):
        edits, wrote, obligations, _ = rounds[min(state["attempt"], len(rounds) - 1)]
        if wrote:
            state["rev"] += 1
        if obligations:
            state["obl"] += 1
        for _ in range(rejected_edits):
            orchestrator.tool_calls_log.append(
                {"turn": state["attempt"], "tool": "modify_file",
                 "input": {}, "success": False})
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

def test_a_round_that_never_reached_for_the_editor_ends_the_loop(orch):
    """The protection the zero-write stop existed to give, kept in full.

    This round wrote nothing, changed nothing, discharged nothing and never
    called an edit tool, so the next prompt is the one this attempt just
    answered and the next round is this round again. It ends immediately - no
    second round, no streak. Across the 221 pre-guard runs the round after a
    prose-only one wrote source 0 times in 4, and granting the second round to
    every zero-write round instead of only this one would have cost ~10 turns
    per run (243 rounds x ~8 turns over 194 runs with a repair loop)."""
    attempts = _drive(orch, [(0, False, False, _blockers(6))] * 8)

    assert attempts == 1
    assert orch._phase3_exit_reason == "replay (attempt never reached for the editor)"


def test_an_attempt_whose_edits_were_all_rejected_gets_one_more_round(orch):
    """The case the replay rule must NOT catch, and the reason the zero-write
    stop was too tight.

    A third of Qwen3-30B's edit calls are refused (1818 of 5555 across the
    corpus; gpt-5.6-terra: 5%), so "wrote nothing" is most often "tried and
    was rejected". Those rejections are fed back into the next attempt's
    prompt as recent-failures, so the next round is a different request - and
    measurably so: across the pre-guard corpus the round after one wrote
    source 57% of the time and cut the blocker count 20% (n=30), against 18%
    / 11% for a round that only read (n=209) and 0% / 0% for prose (n=4).
    It costs ~1.2 turns per run, and two of them in a row still end the loop.
    """
    attempts = _drive(orch, [(0, False, False, _blockers(6))] * 8,
                      rejected_edits=2)

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


# ---------------------------------------------------------------------------
# 4. The rollback must rank the entry tree on the entry tree's own evidence
# ---------------------------------------------------------------------------

def test_the_rollback_ranks_the_best_tree_on_its_own_measurements(orch, monkeypatch):
    """A repair that breaks three entity creates and removes one hard blocker
    must be rolled back, not shipped.

    ``_phase3_tree_score`` reads ``_runtime_probe_facts``, and by the time the
    rollback runs those facts describe the FINAL tree. Recomputing the entry
    score there gave both trees the same middle two components - entities not
    created, actions not effective - so they cancelled and the comparison
    collapsed to (boot, hard count), the component the ranking exists to
    demote. Entry (0,0,0,3) then read as (0,3,0,3), the final tree's (0,3,0,2)
    scored BETTER than it, and the broken tree shipped.
    """
    state = {"attempt": 0, "rev": 0}

    def fix(blockers, is_first_attempt):
        state["attempt"] += 1
        if state["attempt"] <= 2:
            state["rev"] += 1
        if state["attempt"] >= 2:
            # The probe measured the tree attempt 2 produced: it still boots,
            # but three entities can no longer be created. These facts stay
            # current from here on, exactly as they are when the repair loop
            # exits and the rollback runs.
            orch._runtime_probe_facts = (f"rev-{state['rev']}", [{
                "boot": "ok",
                "entities": {name: {"verdict": "refused"} for name in "abc"},
                "action_calls": [],
            }])
        # Attempt 3 writes nothing and touches nothing: the replay guard ends
        # the loop there, leaving attempt 2's tree as the final one.
        return 1 if state["attempt"] <= 2 else 0

    def collect():
        if state["attempt"] == 0:
            return _blockers(3)       # entry: three hard blockers, all created
        return _blockers(3 if state["attempt"] < 2 else 2)

    restored = {}
    with patch.object(orch, "_collect_validation_issues", side_effect=collect), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot",
                         side_effect=lambda: restored.setdefault("called", True) or True), \
            patch.object(orch, "_invoke_phase3_fix_loop", side_effect=fix), \
            patch.object(orch, "_workspace_revision",
                         side_effect=lambda: f"rev-{state['rev']}"):
        orch._run_phase3_validation()

    assert restored.get("called") is True, (
        "a repair that broke three entity creates was scored as an improvement"
    )
    assert orch._phase3_rolled_back is True


# ---------------------------------------------------------------------------
# 5. A finding we label inconclusive may not rank a tree
# ---------------------------------------------------------------------------

def test_an_unverified_create_is_not_counted_as_an_observed_defect(orch):
    """Run gpt-5.6-terra-hzllh0l6: the model marks ``Person`` abstract, the
    generated router correctly answers ``422 Person is abstract; create a
    Patron or Librarian instead``, and the probe's guessed POST produced
    ``runtime unverified: create unverified: ...``. The finding's own text
    says a guessed 4xx "does not prove this endpoint is broken", yet it was
    counted BOTH as an unconfirmed entity and as a hard blocker, and the run
    spent 17 zero-write repair rounds, 108 turns and $0.96 chasing it.

    It stays a blocker - an unverified app is not a verified one - and it
    still ranks as an entity the probe could not confirm. What it may not do
    is also count as an observed defect in the hard total, which is what
    ``_hard_blockers`` feeds: the rollback comparison and the repeated-state
    key.
    """
    inconclusive = ValidationIssue(
        "blocker",
        "runtime unverified: create unverified: backend: POST /person/ - "
        "autogenerated input was refused (422: Person is abstract; create a "
        "Patron or Librarian instead). A guessed request being refused does "
        "not prove this endpoint is broken.",
    )

    assert _hard_blockers([inconclusive]) == []
    assert orch._phase3_tree_score([inconclusive]) == (0, 1, 0, 0)
    assert inconclusive.severity == "blocker"


# ---------------------------------------------------------------------------
# 6. The obligations hash only moves when an obligation actually moved
# ---------------------------------------------------------------------------

def test_a_refused_done_attempt_is_not_an_obligation_discharged(orch):
    """``task_snapshot`` carries ``attempts``, so a REFUSED
    ``task_list(done=...)`` incremented it and the repair loop read failing to
    close an item as closing one - resetting the no-progress streak."""
    orch.executor.set_tasks(["Implement the thing"])
    before = orch._repair_obligations_revision()
    for task in orch.executor._tasks:
        task["attempts"] = int(task.get("attempts") or 0) + 1

    assert orch._repair_obligations_revision() == before

    orch.executor._tasks[0]["done"] = True
    assert orch._repair_obligations_revision() != before, (
        "actually closing an item must still count"
    )


def test_reminting_a_scenario_under_a_new_id_is_not_new_evidence(orch):
    """The model may pass its own ``scenario_id``, so the same requests
    registered twice under different names minted a second entry and moved
    the hash. Only distinct scenario CONTENT is an obligation."""
    scenario = {"requests": [{"method": "GET", "path": "/health"}], "backend": "backend"}
    orch._api_scenarios["named:first"] = {"scenario": scenario, "scenario_id": "first",
                                          "revision": "r", "report": {}}
    before = orch._repair_obligations_revision()
    orch._api_scenarios["named:second"] = {"scenario": dict(scenario), "scenario_id": "second",
                                           "revision": "r", "report": {}}

    assert orch._repair_obligations_revision() == before

    orch._api_scenarios["named:third"] = {
        "scenario": {"requests": [{"method": "GET", "path": "/books"}], "backend": "backend"},
        "scenario_id": "third", "revision": "r", "report": {}}
    assert orch._repair_obligations_revision() != before, (
        "a genuinely new workflow must still count"
    )
