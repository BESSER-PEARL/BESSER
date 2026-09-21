"""Phase 3 must stop burning turns, keep its best tree, and prove the app runs.

Measured over the 23 spec-iteration runs of 2026-09-19
(``verification/spec-iterations/``), scored against the CORRECTED acceptance
probe (``verification/rescore_corrected_probe.json``; the first scoring was
itself defective and understated the generator badly):

* 7 of those 23 delivered a fully working app. None of them may be refused by
  anything here - that is the acceptance criterion these tests defend.
* Blocker count correlated **+0.21** with whether the app worked - wrong sign,
  noise magnitude - so nothing here ranks a tree by blocker count alone.
* 34 of 104 fix attempts produced ZERO writes (222 turns); 522 turns (31% of
  every turn spent) came after the blocker count stopped moving.
* 10 of 22 runs with a repair loop ended in a worse state than one they had
  already reached. 673hzu0z walked 6-10-7-2-9-11-9-2-6-6-6 and shipped 6.
* ``test_api`` was offered in every run and 20 of 23 never called it - with no
  consequence, because an app with no runtime evidence produced exactly the
  same (empty) finding list as an app whose scenarios all passed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import (
    _PHASE3_NO_PROGRESS_ROUNDS,
    _PHASE3_PLATEAU_ROUNDS,
    LLMOrchestrator,
    ValidationIssue,
)
from besser.spec_driven_agent.validation.issues import _classify_issue


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
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


def _drive(orchestrator, scripted_attempts):
    """Run the real Phase 3 loop over a scripted sequence of attempt outcomes.

    ``scripted_attempts`` is a list of ``(edits, wrote, blockers_after)``.
    Returns the number of attempts the loop actually ran.
    """
    state = {"attempt": 0, "rev": 0}

    def fix(blockers, is_first_attempt):
        edits, wrote, _ = scripted_attempts[min(state["attempt"], len(scripted_attempts) - 1)]
        if wrote:
            state["rev"] += 1
        state["attempt"] += 1
        return edits

    def collect():
        if state["attempt"] == 0:
            return _blockers(6)
        return scripted_attempts[min(state["attempt"] - 1, len(scripted_attempts) - 1)][2]

    with patch.object(orchestrator, "_collect_validation_issues", side_effect=collect), \
            patch.object(orchestrator, "_create_snapshot") as snap, \
            patch.object(orchestrator, "_restore_snapshot", return_value=True) as restore, \
            patch.object(orchestrator, "_invoke_phase3_fix_loop", side_effect=fix), \
            patch.object(orchestrator, "_workspace_revision",
                         side_effect=lambda: f"rev-{state['rev']}"):
        orchestrator._run_phase3_validation()
    return state["attempt"], snap, restore


# ---------------------------------------------------------------------------
# 1. An attempt that never reaches for the editor ends the loop
#
# ``_drive`` patches the fix loop out, so no attempt here records a tool call:
# every zero-write round below is the REPLAY case - nothing it did can reach
# the next prompt, so the next round would be this round again. Across the 221
# runs recorded before this stop existed, the round after a prose-only one
# wrote source 0 times in 4. A round whose edits were merely REJECTED is a
# different case and does buy one more round (57% of those wrote next round,
# n=30); test_phase3_stall_guards.py owns that distinction.
# ---------------------------------------------------------------------------

def test_an_attempt_that_writes_nothing_ends_the_fix_loop(orch):
    """34 of 104 attempts wrote nothing; the loop kept paying for more."""
    attempts, _snap, _restore = _drive(orch, [(0, False, _blockers(6))] * 5)

    assert attempts == 1, "a replay attempt must not buy another attempt"


def test_a_writing_attempt_still_gets_a_second_round(orch):
    """The stop keys on ``edits == 0 AND the tree is unchanged AND the attempt
    never tried to write``, not on any one of them - an attempt that spent
    early turns reading and then wrote is real work and keeps its next
    round."""
    attempts, _snap, _restore = _drive(orch, [
        (1, True, _blockers(5)),
        (1, True, _blockers(4)),
        (0, False, _blockers(4)),
    ])

    assert attempts == 3


def test_edits_without_a_tree_change_are_not_treated_as_nothing(orch):
    """``edits > 0`` with an unchanged revision (the model rewrote identical
    bytes) is not the replay case: the softer no-progress streak owns it, and
    that streak is two rounds long."""
    attempts, _snap, _restore = _drive(orch, [(2, False, _blockers(6))] * 5)

    assert attempts == _PHASE3_NO_PROGRESS_ROUNDS == 2


# ---------------------------------------------------------------------------
# 2. A plateau ends the loop; the unchanged-state guard cannot see one
# ---------------------------------------------------------------------------

def test_rounds_that_edit_without_improving_end_the_loop(orch):
    """Run mbzbzhq9: six attempts, each ending at exactly 13 blockers, the
    120-turn cap, and a dead app.

    ``seen_states`` cannot catch this - its key includes a content hash of
    every source file, so a different useless edit each round always reads as
    a brand new state and the guard never fires.
    """
    attempts, _snap, _restore = _drive(orch, [(1, True, _blockers(6))] * 8)

    assert attempts == _PHASE3_PLATEAU_ROUNDS, (
        "a tree that keeps changing without improving must not run to the cap"
    )


def test_a_repair_that_exposes_the_errors_behind_an_import_keeps_going(orch):
    """11 -> 45 -> 40 -> 35 is a repair converging, not a stall.

    The plateau guard measures against the PREVIOUS round for exactly this:
    fixing one import legitimately exposes the CRUD errors it was hiding, and
    scoring against the best-ever tree would end the run on the exposure.
    """
    attempts, _snap, _restore = _drive(orch, [
        (1, True, _blockers(45)),
        (1, True, _blockers(40)),
        (1, True, _blockers(35)),
        (1, True, _blockers(30)),
        (1, True, []),
    ])

    assert attempts == 5


def test_a_run_that_keeps_improving_is_not_cut_short(orch):
    attempts, _snap, _restore = _drive(orch, [
        (1, True, _blockers(5)),
        (1, True, _blockers(4)),
        (1, True, _blockers(3)),
        (1, True, _blockers(2)),
        (1, True, []),
    ])

    assert attempts == 5


# ---------------------------------------------------------------------------
# 3. The BEST tree is what ships, not the last one
# ---------------------------------------------------------------------------

def test_the_best_tree_is_re_snapshotted_and_restored(orch):
    """673hzu0z reached 2 blockers twice and shipped 6.

    The old code snapshotted once, before Phase 3, and compared only the final
    state against that entry state: 6 -> 2 -> 6 ended "no worse than we began",
    so the 2-blocker tree was thrown away silently.
    """
    attempts, snap, restore = _drive(orch, [
        (1, True, _blockers(2)),   # best
        (1, True, _blockers(6)),   # regressed back to the entry count
        (1, True, _blockers(6)),
    ])

    assert attempts >= 2
    assert snap.call_count >= 1, "the improved tree was never snapshotted"
    restore.assert_called_once()
    assert orch._phase3_rolled_back is True


def test_a_tree_that_boots_beats_a_tree_with_fewer_blockers(orch):
    """Ranking is runtime-first. Blocker count correlated +0.21 with whether
    the app worked, so it must never outvote 'the application starts'."""
    boots = _blockers(9)
    dead = [ValidationIssue("blocker", "mapper config: sql_alchemy.py line 107: no property 'bill'")]

    assert orch._phase3_tree_score(boots) < orch._phase3_tree_score(dead)


def test_an_entity_that_cannot_be_created_outranks_lint_volume(orch):
    creates = [ValidationIssue(
        "blocker", "create contract: web_app/backend: POST /booking/ - observed a "
                   "server/persistence failure creating Booking")]

    assert orch._phase3_tree_score(_blockers(20)) < orch._phase3_tree_score(creates)


# ---------------------------------------------------------------------------
# 4. The boot probe must actually run
# ---------------------------------------------------------------------------

def _probe_recorder(monkeypatch, orchestrator, findings=()):
    calls = {"n": 0}

    def fake_probe(output_dir, domain_model=None):
        calls["n"] += 1
        return {"issues": list(findings), "backends": []}

    import besser.spec_driven_agent.validation.constructibility as constructibility
    monkeypatch.setattr(constructibility, "collect_constructibility_report", fake_probe)
    monkeypatch.setattr(
        "besser.spec_driven_agent.pipeline.orchestrator._import_smoke_issues", lambda _d: [])
    monkeypatch.setattr(orchestrator, "_probeable_backends", lambda: ["web_app/backend"])
    return calls


def test_a_frontend_finding_no_longer_suppresses_the_boot_probe(orch, monkeypatch):
    """The mechanism behind "+0.21": the blocker list and "does it run" were
    measuring different trees.

    ``_collect_execution_issues`` returned early on ANY cheap finding, and the
    frontend schema checker fed that same list. Nine of the 23 runs carried a
    ``frontend contract:`` line and therefore shipped with no runtime verdict
    at all; re-probing 7aybctis's delivered tree afterwards found POST
    /booking/ raising TypeError on every call.
    """
    monkeypatch.setattr(
        "besser.spec_driven_agent.validation.frontend_schema.collect_frontend_schema_issues",
        lambda _d: ["frontend contract: src/pages/Booking.tsx:25: form exposes 'bill'"],
    )
    calls = _probe_recorder(monkeypatch, orch)

    issues = orch._collect_execution_issues()

    assert calls["n"] == 1, "a React form field must not stop the backend booting"
    assert any(i.startswith("frontend contract:") for i in issues)


def test_a_module_that_cannot_be_imported_still_skips_the_probe(orch, monkeypatch):
    """Only a finding that PROVES the app cannot boot may skip it."""
    monkeypatch.setattr(
        "besser.spec_driven_agent.pipeline.orchestrator._unresolvable_local_imports",
        lambda _d: ["missing module: routers/booking.py imports 'services' which does not exist"],
    )
    calls = _probe_recorder(monkeypatch, orch)

    orch._collect_execution_issues()

    assert calls["n"] == 0


def test_the_boot_probe_is_cached_per_source_revision(orch, monkeypatch):
    """It costs up to 90s per backend and Phase 3 re-validates every attempt."""
    calls = _probe_recorder(monkeypatch, orch)
    monkeypatch.setattr(orch, "_workspace_revision", lambda: "rev-1")

    orch._collect_execution_issues()
    orch._collect_execution_issues()

    assert calls["n"] == 1

    monkeypatch.setattr(orch, "_workspace_revision", lambda: "rev-2")
    orch._collect_execution_issues()

    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# 5. Observed action failures reach the fix loop
# ---------------------------------------------------------------------------

def test_an_observed_action_500_is_a_blocker():
    """``action call:`` is the action-side twin of ``create contract:``.

    It was in no prefix list in ``_classify_issue``, so it fell through to the
    default warning - and the fix loop only consumes blockers, so an endpoint
    the probe had literally watched raise a 500 was never repaired.
    """
    issue = _classify_issue(
        "action call: web_app/backend: POST /bill/{id}/methods/registerPayment/ - "
        "observed a server/persistence failure calling registerPayment"
    )

    assert issue.severity == "blocker"


def test_an_unreachable_action_state_is_promoted_like_an_unverified_create(orch, monkeypatch):
    """``action unverified:`` had no promotion at all, so it stayed a warning.

    It gets the create side's scenario-rescued treatment rather than an
    unconditional blocker: the probe legitimately cannot construct every
    state, so a passing ``test_api`` workflow must be able to retire it.
    """
    _probe_recorder(monkeypatch, orch, findings=[
        "action unverified: web_app/backend: POST /booking/1/methods/cancel/ - "
        "cancel on Booking refused every state this probe could construct",
    ])

    issues = orch._collect_execution_issues()
    promoted = [i for i in issues if i.startswith("runtime unverified: action unverified:")]

    assert promoted, "an unsettled action must reach the blocker-only fix loop"
    assert _classify_issue(promoted[0]).severity == "blocker"


def test_a_passing_scenario_retires_an_unsettled_action(orch, monkeypatch):
    _probe_recorder(monkeypatch, orch, findings=[
        "action unverified: web_app/backend: POST /booking/1/methods/cancel/ - "
        "cancel on Booking refused every state this probe could construct",
    ])
    monkeypatch.setattr(orch, "_workspace_revision", lambda: "rev-1")
    orch._api_scenarios["named:cancel"] = {
        "scenario": {"requests": [], "backend": "web_app/backend"},
        "scenario_id": "cancel", "revision": "rev-1",
        "report": {"status": "passed", "boot": "ok", "backend": "web_app/backend",
                   "responses": [{"method": "POST", "status": 200,
                                  "path": "/booking/1/methods/cancel/"}]},
    }

    issues = orch._collect_execution_issues()

    assert not [i for i in issues if "action unverified:" in i]


# ---------------------------------------------------------------------------
# 6. The runtime gate refuses to report success
# ---------------------------------------------------------------------------

def test_the_gate_refuses_completion_when_the_app_cannot_start(orch, monkeypatch):
    """Detection alone is provably not enough.

    Run mbzbzhq9 held its fatal ``mapper config:`` blocker in top-priority
    position through six attempts and shipped anyway. The gate is the part
    that makes the run report itself incomplete, naming the runtime failure.
    """
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = 2  # _RUNTIME_FAILED
    orch._validation_issues = [ValidationIssue(
        "blocker", "mapper config: sql_alchemy.py line 107: Mapper 'Booking' has no property 'bill'")]

    orch._apply_runtime_gate()

    gate = [i for i in orch._validation_issues if i.message.startswith("runtime gate:")]
    assert len(gate) == 1
    assert gate[0].severity == "blocker"
    assert "mapper config:" in gate[0].message


def test_an_exhausted_budget_with_a_red_gate_reports_incomplete(orch, monkeypatch):
    """The honest outcome when the money runs out is a named runtime failure,
    not a silent pass."""
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = 2  # last probe: the app does not come up
    # The budget is gone before the first repair turn, so Phase 3 bails at its
    # own stop gate - the path that used to end the run with no runtime word.
    monkeypatch.setattr(orch, "_phase3_stop_requested",
                        lambda **_kwargs: "cost budget exhausted")
    orch._validation_issues = [ValidationIssue(
        "blocker", "create contract: POST /booking/ - observed a server/persistence failure")]

    with patch.object(orch, "_invoke_phase3_fix_loop",
                      side_effect=AssertionError("no repair on an exhausted budget")):
        orch._run_phase3_validation()

    gate = [i for i in orch._validation_issues if i.message.startswith("runtime gate:")]
    assert gate and gate[0].severity == "blocker"
    assert "NOT complete" in gate[0].message


def test_no_runtime_evidence_at_all_is_not_a_pass(orch, monkeypatch):
    """20 of 23 runs never called ``test_api`` and nothing noticed.

    An empty scenario dict produced an empty finding list - byte-identical to
    an app whose scenarios all passed. Absence of evidence is not evidence of
    absence.
    """
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = None
    orch._validation_issues = []

    orch._apply_runtime_gate()

    gate = [i for i in orch._validation_issues if i.message.startswith("runtime gate:")]
    assert len(gate) == 1
    assert "no runtime evidence" in gate[0].message


def test_a_business_rule_refusing_a_guessed_fixture_does_not_refuse_the_run(orch, monkeypatch):
    """The gate must stay silent on an unsettled create. Measured, twice.

    dp3trml9 - the accepted artifact - answers the probe's guessed
    ReservedRoom payload with ``409 This room is already linked to the
    booking``, and 308z4wo2 passes 11/11 corrected acceptance checks with
    EVERY create route coming back ``create unverified:``. A gate that read a
    business-rule refusal as a defect would refuse working applications, so
    the unsettled route keeps only its own ``runtime unverified:`` finding.
    """
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = 1  # _RUNTIME_UNVERIFIED
    unsettled = ValidationIssue(
        "blocker",
        "runtime unverified: create unverified: web_app/backend: POST /reservedroom/ - "
        "autogenerated input was refused (409: This room is already linked to the booking)")
    orch._validation_issues = [unsettled]

    orch._apply_runtime_gate()

    assert orch._validation_issues == [unsettled]


def test_a_failing_retained_scenario_does_not_refuse_the_run(orch, monkeypatch):
    """A retained workflow is a model-authored assertion, not the spec.

    Run dynioweu passes 11/11 corrected acceptance checks and still fails its
    own ``booking_overlap_violation`` scenario; ``test_api`` itself says "a
    generated assertion can be wrong". The scenario stays a blocker, but the
    gate does not add a second, louder refusal on top of it.
    """
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = 1  # api scenario: ranks as unsettled, not failed
    failing = ValidationIssue(
        "blocker", "api scenario: booking_overlap_violation failed: expected 200, got 400")
    orch._validation_issues = [failing]

    orch._apply_runtime_gate()

    assert orch._validation_issues == [failing]


def test_the_gate_is_silent_on_a_verified_app(orch, monkeypatch):
    monkeypatch.setattr(orch, "_probeable_backends", lambda: ["web_app/backend"])
    orch._runtime_probe_verdict = 0  # _RUNTIME_OK
    orch._validation_issues = []

    orch._apply_runtime_gate()

    assert orch._validation_issues == []


def test_the_gate_is_silent_when_there_is_nothing_to_boot(orch):
    """A Qiskit or BAF run has no backend; the gate makes no claim about it."""
    orch._runtime_probe_verdict = None
    orch._validation_issues = []

    orch._apply_runtime_gate()

    assert orch._validation_issues == []


# ---------------------------------------------------------------------------
# 7. Small wirings
# ---------------------------------------------------------------------------

def test_compaction_carries_the_checklist_blockers_and_contract(orch, monkeypatch):
    """Without ``work_state`` the model resumes after a compaction unable to
    see the checklist the end_turn gate is blocking on."""
    orch.executor.set_tasks([{"text": "Implement POST /booking/{id}/methods/cancel/"}])
    orch._validation_issues = [
        ValidationIssue("blocker", "create contract: POST /booking/ - failure"),
        ValidationIssue("style", "ruff: F401 unused import"),
    ]
    seen = {}

    def fake_maybe_compact(**kwargs):
        seen.update(kwargs)
        return kwargs["messages"], False

    monkeypatch.setattr("besser.spec_driven_agent.pipeline.orchestrator.maybe_compact", fake_maybe_compact)
    orch._maybe_compact([{"role": "user", "content": "hi"}])

    state = seen["work_state"]
    assert [t["text"] for t in state["tasks"]] == [
        "Implement POST /booking/{id}/methods/cancel/"]
    assert state["blockers"] == ["create contract: POST /booking/ - failure"]
    assert any("server-owned" in rule for rule in state["contract_rules"])


def test_seven_refusals_at_one_target_end_the_phase(orch):
    """``tool_executor`` now counts refusals per TARGET through the same
    ``last_repeat`` channel, so the stop threshold governs redrafts too."""
    assert orch._TARGET_REFUSAL_STOP_AT == orch._REPEAT_STOP_AT == 7
    orch.executor.last_repeat = ("routers/booking.py", 7)

    assert orch._escalate_repeat_rejection([]) is True
    assert orch._phase2_stop_reason == "stuck_edit_loop"


def test_six_refusals_at_one_target_do_not_end_the_phase(orch):
    orch.executor.last_repeat = ("routers/booking.py", 6)
    messages: list[dict] = []

    assert orch._escalate_repeat_rejection(messages) is False
    assert messages, "the model still gets a strategy hint"
