"""Phase 3 must stop burning turns, keep its best tree, and prove the app runs.

Measured over 23 recorded runs, scored against the CORRECTED acceptance probe
(the first scoring was itself defective and understated the generator badly):

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


# The route as the action probe prints it: the OpenAPI path template.
_UNSETTLED_CANCEL = (
    "action unverified: web_app/backend: POST /booking/{booking_id}/methods/cancel/ - "
    "cancel on Booking refused every state this probe could construct")


def test_an_unreachable_action_state_is_promoted_like_an_unverified_create(orch, monkeypatch):
    """``action unverified:`` had no promotion at all, so it stayed a warning.

    It gets the create side's scenario-rescued treatment rather than an
    unconditional blocker: the probe legitimately cannot construct every
    state, so a passing ``test_api`` workflow must be able to retire it.
    """
    _probe_recorder(monkeypatch, orch, findings=[_UNSETTLED_CANCEL])

    issues = orch._collect_execution_issues()
    promoted = [i for i in issues if i.startswith("runtime unverified: action unverified:")]

    assert promoted, "an unsettled action must reach the blocker-only fix loop"
    assert _classify_issue(promoted[0]).severity == "blocker"


def _cancel_scenario(orch, monkeypatch, response):
    """Record a passing current scenario whose only response is ``response``."""
    monkeypatch.setattr(orch, "_workspace_revision", lambda: "rev-1")
    orch._api_scenarios["named:cancel"] = {
        "scenario": {"requests": [], "backend": "web_app/backend"},
        "scenario_id": "cancel", "revision": "rev-1",
        "report": {"status": "passed", "boot": "ok", "backend": "web_app/backend",
                   "responses": [{"method": "POST", "status": 200, **response}]},
    }


def test_a_passing_scenario_retires_an_unsettled_action(orch, monkeypatch):
    """The finding names the route template; the scenario called a concrete path."""
    _probe_recorder(monkeypatch, orch, findings=[_UNSETTLED_CANCEL])
    _cancel_scenario(orch, monkeypatch, {"path": "/booking/1/methods/cancel/"})

    issues = orch._collect_execution_issues()

    assert not [i for i in issues if "action unverified:" in i]


@pytest.mark.parametrize("response", [
    {"path": "/booking/1/methods/confirm/"},                      # another action
    {"path": "/room/1/methods/cancel/"},                          # another entity
    {"path": "/booking/1/methods/cancel/extra/"},                 # a longer route
    {"path": "/booking/1/methods/cancel/", "json": {"success": False, "message": "refused"}},
])
def test_a_scenario_that_did_not_run_the_action_leaves_it_unsettled(orch, monkeypatch, response):
    """A 2xx elsewhere, or one whose body says the call did not act, proves nothing."""
    _probe_recorder(monkeypatch, orch, findings=[_UNSETTLED_CANCEL])
    _cancel_scenario(orch, monkeypatch, response)

    issues = orch._collect_execution_issues()

    assert [i for i in issues if i.startswith("runtime unverified: action unverified:")]


def test_a_read_back_under_a_nested_create_route_retires_its_unverified_create(orch, monkeypatch):
    _probe_recorder(monkeypatch, orch, findings=[
        "create unverified: web_app/backend: POST /booking/{booking_id}/guest/ - "
        "autogenerated input was refused (422)",
    ])
    monkeypatch.setattr(orch, "_workspace_revision", lambda: "rev-1")
    orch._api_scenarios["named:guest"] = {
        "scenario": {"requests": [], "backend": "web_app/backend"},
        "scenario_id": "guest", "revision": "rev-1",
        "report": {"status": "passed", "boot": "ok", "backend": "web_app/backend", "responses": [
            {"index": 0, "method": "POST", "status": 201, "path": "/booking/3/guest/",
             "json": {"id": 5, "name": "Ada"}},
            {"index": 1, "method": "GET", "status": 200, "path": "/booking/3/guest/5/",
             "json": {"id": 5, "name": "Ada"}}]},
    }

    issues = orch._collect_execution_issues()

    assert not [i for i in issues if "create unverified:" in i]
    assert orch._unconfirmed_creates([_classify_issue(
        "runtime unverified: create unverified: web_app/backend: POST /booking/{booking_id}/guest/ - "
        "autogenerated input was refused (422)")]) == set()
