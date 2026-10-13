"""A run must say what it verified, what it could not, and what ships unenforced.

The motivating run scored 11/11, passed its whole booking workflow, and will
double-sell a room: two OCL constraints failed conversion and never reached the
code. The product knew — it listed both as open tasks and reported 21 blockers.
One number hid which of the three states each finding was in.

"we checked and it works", "we could not check" and "we checked and it is
missing" must stay three lists.
"""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.validation.issues import (
    _check_did_not_run,
    required_check_unverified,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
    build_verification_report,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _FakeClient,
    _StubOrchestrator,
    _build_request,
    _collect_frames,
    _parse,
)


# The two rules the hotel run shipped without.
_NO_DOUBLE_BOOKING = {
    "id": "ocl-conversion-aaaa", "category": "ocl", "context": "Booking",
    "name": "noOverlappingBookings", "kind": "invariant",
    "expression": "self.room.bookings->forAll(b | b <> self implies not self.overlaps(b))",
    "reason": "Invalid OCL syntax: Property 'overlaps' not found on Booking",
    "source": {"diagram_title": "Hotel", "element_id": "e1"},
}
_GUEST_CAPACITY = {
    "id": "ocl-conversion-bbbb", "category": "ocl", "context": "Booking",
    "name": "guestsWithinCapacity", "kind": "invariant",
    "expression": "self.guests <= self.room.capacity",
    "reason": "Invalid OCL syntax: Property 'capacity' not found on Room",
    "source": {"diagram_title": "Hotel", "element_id": "e2"},
}


def _recovery_requirement(issue: dict, rid: int, **verdict) -> dict:
    """A judge verdict on the synthetic 'recover this rejected rule' requirement."""
    label = issue["name"]
    return {
        "id": rid, "kind": "rule",
        "text": f"Recover model constraint '{label}' rejected during conversion. "
                f"Contract: invariant on Booking. Original OCL: {issue['expression']}.",
        "status": "unverified", "evidence": "", "note": "", **verdict,
    }


def _hotel_recipe(**overrides) -> dict:
    recipe = {
        "model_conversion_issues": [_NO_DOUBLE_BOOKING, _GUEST_CAPACITY],
        "requirements": [
            {"id": 1, "kind": "rule", "text": "A guest can book a room for a date range.",
             "status": "implemented", "evidence": "backend/routers/booking.py:    booking = Booking(**payload)",
             "note": ""},
            {"id": 2, "kind": "rule", "text": "A booking can be cancelled before check-in.",
             "status": "missing", "evidence": "", "note": "no cancel route exists"},
            {"id": 3, "kind": "rule", "text": "Nightly rates vary by season.",
             "status": "unverified", "evidence": "", "note": "no verdict returned"},
            _recovery_requirement(_NO_DOUBLE_BOOKING, 4),
            _recovery_requirement(_GUEST_CAPACITY, 5),
        ],
        "api_scenarios": [{
            "scenario_id": "booking-workflow",
            "scenario": {"backend": "backend", "requests": [
                {"method": "POST", "path": "/rooms/"},
                {"method": "POST", "path": "/bookings/"},
                {"method": "GET", "path": "/bookings/"},
            ]},
            "report": {"status": "passed", "boot": "ok"},
        }],
        "validation_issues": [
            {"severity": "warning", "message": _check_did_not_run("ruff", "timed out after 30s")},
            {"severity": "blocker", "message": "requirement: R2 is not implemented"},
        ],
    }
    recipe.update(overrides)
    return recipe


def _ids(items) -> list[str]:
    return [item.id for item in items]


def test_failed_ocl_conversion_is_reported_as_shipped_unenforced():
    report = build_verification_report(_hotel_recipe())

    assert "noOverlappingBookings" in _ids(report.shippedUnenforced)
    assert "guestsWithinCapacity" in _ids(report.shippedUnenforced)
    double = next(i for i in report.shippedUnenforced if i.id == "noOverlappingBookings")
    assert "never reached the generated code" in double.why
    assert "forAll" in double.what, "the rule the app does not enforce must be readable"

    # It must NOT be able to read as verified or as merely unchecked.
    assert "noOverlappingBookings" not in _ids(report.verified)
    assert "noOverlappingBookings" not in _ids(report.notVerified)


def test_verified_says_what_was_actually_run():
    report = build_verification_report(_hotel_recipe())

    workflow = next(i for i in report.verified if i.kind == "api_workflow")
    assert workflow.id == "booking-workflow"
    assert "POST /bookings/" in workflow.how
    assert "3 request(s) run against the app" in workflow.how

    implemented = next(i for i in report.verified if i.id == "R1")
    assert "re-checked against the delivered source" in implemented.how


def test_could_not_check_never_reads_as_checked():
    report = build_verification_report(_hotel_recipe())

    assert "R3" in _ids(report.notVerified)
    assert "R3" not in _ids(report.verified) and "R3" not in _ids(report.shippedUnenforced)

    skipped = [i for i in report.notVerified if i.kind == "check"]
    assert [i.what for i in skipped] == ["ruff"]
    assert "timed out after 30s" in skipped[0].why


def test_checked_and_missing_is_separated_from_could_not_check():
    report = build_verification_report(_hotel_recipe())

    assert "R2" in _ids(report.shippedUnenforced)
    missing = next(i for i in report.shippedUnenforced if i.id == "R2")
    assert "does not implement it" in missing.why

    identified = [i for i in
                  report.verified + report.notVerified + report.shippedUnenforced if i.id]
    assert len({i.id for i in identified}) == len(identified), \
        "an identified item may appear in exactly one list"
    assert report.counts.model_dump() == {
        "verified": 2, "notVerified": 2, "shippedUnenforced": 3,
    }


def test_a_recovered_ocl_constraint_moves_out_of_shipped_unenforced():
    """Absence of evidence is not evidence of absence — and nor is the reverse."""
    recipe = _hotel_recipe()
    recipe["requirements"][3] = _recovery_requirement(
        _NO_DOUBLE_BOOKING, 4, status="implemented",
        evidence="backend/routers/booking.py:        raise HTTPException(409, 'room already booked')",
    )
    report = build_verification_report(recipe)

    assert "noOverlappingBookings" in _ids(report.verified)
    assert "noOverlappingBookings" not in _ids(report.shippedUnenforced)
    assert "guestsWithinCapacity" in _ids(report.shippedUnenforced)


def test_a_rejected_rule_nothing_judged_still_ships_unenforced():
    recipe = _hotel_recipe(requirements=[])
    report = build_verification_report(recipe)

    assert len(report.shippedUnenforced) == 2
    assert all("nothing checked whether it was re-implemented" in i.why
               for i in report.shippedUnenforced)


def test_a_ledger_that_produced_no_verdict_at_all_is_not_silence():
    """No verdicts must not read as "there was nothing to check"."""
    recipe = _hotel_recipe(requirements=[], model_conversion_issues=[], validation_issues=[{
        "severity": "warning",
        "message": "requirement unverified: requirement extraction failed; the "
                   "original specification has not been checked.",
    }])
    report = build_verification_report(recipe)

    coverage = next(i for i in report.notVerified if i.what == "original-specification coverage")
    assert "extraction failed" in coverage.why
    assert report.verified == [i for i in report.verified if i.kind == "api_workflow"]


def test_an_unimplemented_modeled_operation_is_missing_not_merely_unchecked():
    """A run's 50 issues included 6 modeled methods left as HTTP 501 and 18
    checklist items never established. Both were invisible behind the count,
    and they are not the same state."""
    recipe = _hotel_recipe(validation_issues=[
        {"severity": "blocker", "message":
            "action contract: web_app/backend/routers/booking_methods.py line 72: "
            "POST /booking/{booking_id}/methods/cancel/ (execute_booking_cancel): HTTP 501."},
        {"severity": "blocker", "message":
            "task unverified: task 1 remains unresolved: Implement POST "
            "/bill/{bill_id}/methods/registerPayment/."},
        {"severity": "style", "message": "ruff: F401 unused import"},
    ])
    report = build_verification_report(recipe)

    missing = [i for i in report.shippedUnenforced if "execute_booking_cancel" in i.what]
    assert len(missing) == 1
    assert "does not implement it" in missing[0].why

    unknown = [i for i in report.notVerified if "task 1 remains unresolved" in i.what]
    assert len(unknown) == 1
    assert "unknown, not absent" in unknown[0].why

    # Style noise stays out of all three lists.
    everything = report.verified + report.notVerified + report.shippedUnenforced
    assert not any("ruff" in i.what for i in everything)


def test_a_run_that_never_exercised_the_app_says_so():
    report = build_verification_report(_hotel_recipe(api_scenarios=[]))

    runtime = next(i for i in report.notVerified
                   if i.what == "runtime behaviour of the delivered app")
    assert "no API workflow was run against it" in runtime.why
    # Not raised on a run with no requirement verdicts (deterministic, or a
    # non-web target) — there is nothing for it to be silent about.
    quiet = build_verification_report({"requirements": [], "api_scenarios": []})
    assert quiet.notVerified == []


def test_a_rejected_constraints_reason_survives_intact():
    """The client renders `why` in full, so the server cap is the last cut.

    Sized from the corpus: across 365 run recipes a rejected OCL constraint's
    `why` runs 263-750 characters — every one of them over the 240 this used
    to truncate at, which left the user a mid-word fragment of the reason a
    room can be double-sold.
    """
    long_rule = {
        **_NO_DOUBLE_BOOKING,
        "expression": "context Room inv noOverlappingBookings: self.bookings->forAll("
                      "b1, b2 | b1 <> b2 implies b1.departureDate <= b2.arrivalDate "
                      "or b2.departureDate <= b1.arrivalDate)",
        "reason": "Warning: Invalid OCL syntax in 'context Room inv noOverlappingBookings: "
                  "self.bookings->forAll(b1, b2 | b1 <> b2 implies b1.departureDate <= "
                  "b2.arrivalDate or b2.departureDate <= b1.arrivalDate)': Property "
                  "'bookings' not found in context 'Room' (did you mean 'self.booking'?)",
    }
    report = build_verification_report(
        {"model_conversion_issues": [long_rule], "requirements": []}
    )
    item = report.shippedUnenforced[0]

    assert len(item.why) > 240
    assert "[truncated]" not in item.why
    assert item.why.endswith("nothing checked whether it was re-implemented")
    assert "did you mean" in item.why, "the actionable half of the reason must survive"


def test_a_cut_is_on_a_word_boundary_and_says_it_was_cut():
    report = build_verification_report({"model_conversion_issues": [{
        **_NO_DOUBLE_BOOKING, "reason": "rejected because " + "verylongword " * 200,
    }], "requirements": []})
    why = report.shippedUnenforced[0].why

    assert why.endswith("…[truncated]")
    assert not why.removesuffix("…[truncated]").endswith("verylongwor"), \
        "a mid-word cut reads as a complete word that was never written"


def test_how_is_capped_tighter_than_why():
    """`how` (a request list) degrades gracefully; `why` does not."""
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
        _MAX_HOW, _MAX_WHAT, _MAX_WHY,
    )
    assert _MAX_HOW < _MAX_WHY <= _MAX_WHAT


def test_the_cap_cannot_hide_a_whole_category():
    """A run can carry 86 unverified requirement verdicts; a plain head-25 cap
    dropped every skipped-check note off the end of the same list."""
    recipe = _hotel_recipe(
        requirements=[{"id": n, "kind": "rule", "text": f"Requirement {n}.",
                       "status": "unverified", "evidence": "", "note": ""}
                      for n in range(1, 87)],
        validation_issues=[{
            "severity": "warning",
            "message": _check_did_not_run("ruff", "timed out after 30s"),
        }],
    )
    report = build_verification_report(recipe)

    assert report.counts.notVerified == 87
    assert len(report.notVerified) == 25
    assert any(i.kind == "check" for i in report.notVerified), \
        "the skipped check was pushed out by the requirement verdicts"


def test_a_dropped_recipe_is_not_an_empty_report():
    """_read_recipe drops an oversized recipe; three empty lists would lie."""
    report = build_verification_report(
        {"warning": "recipe file exceeded size limit and was dropped"}
    )
    assert report.verified == [] and report.shippedUnenforced == []
    assert "could not be read" in report.notVerified[0].why


def test_required_check_unverified_lands_in_could_not_check():
    recipe = _hotel_recipe(validation_issues=[{
        "severity": "warning",
        "message": required_check_unverified("frontend build [.]", "node is unavailable"),
    }])
    report = build_verification_report(recipe)

    check = next(i for i in report.notVerified if i.kind == "check")
    assert check.what == "frontend build [.]"
    assert "node is unavailable" in check.why


def test_the_synthetic_recovery_requirement_the_orchestrator_writes_is_matchable():
    """Pins the coupling: the builder finds a rejected rule's verdict by the
    wording ``_requirements_for_validation`` gives it. If that wording changes,
    every recovered constraint would silently keep reading as unenforced."""
    owner = SimpleNamespace(
        _requirements=[],
        domain_model=SimpleNamespace(conversion_issues=[_NO_DOUBLE_BOOKING]),
    )
    synthetic = LLMOrchestrator._requirements_for_validation(owner)
    assert len(synthetic) == 1

    verdict = {**synthetic[0], "status": "implemented", "evidence": "app.py:    pass", "note": ""}
    report = build_verification_report({
        "model_conversion_issues": [_NO_DOUBLE_BOOKING], "requirements": [verdict],
    })
    assert _ids(report.verified) == ["noOverlappingBookings"]
    assert report.shippedUnenforced == []


# ----------------------------------------------------------------------
# The report has to reach the client, not just exist
# ----------------------------------------------------------------------


class _RecipeOrchestrator(_StubOrchestrator):
    """Writes the recipe a real run leaves behind, with the hotel evidence."""

    def _finish(self, method: str) -> str:
        path = super()._finish(method)
        with open(os.path.join(path, ".besser_recipe.json"), "w", encoding="utf-8") as fh:
            json.dump(_hotel_recipe(), fh)
        return path


def _cleanup():
    async def _c():
        import shutil
        async with SMART_RUN_REGISTRY._lock:
            for entry in list(SMART_RUN_REGISTRY._entries.values()):
                shutil.rmtree(entry.temp_dir, ignore_errors=True)
            SMART_RUN_REGISTRY._entries.clear()
    asyncio.run(_c())


def test_done_event_and_recipe_both_carry_the_three_lists(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _RecipeOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        frames = asyncio.run(_collect_frames(SmartGenerationRunner(_build_request())))
        done = [_parse(f) for f in frames if _parse(f).get("event") == "done"][-1]

        event_report = done["verification"]
        assert [i["id"] for i in event_report["shippedUnenforced"]] == [
            "noOverlappingBookings", "guestsWithinCapacity", "R2",
        ]
        assert event_report["counts"]["verified"] == 2
        # The recipe the user downloads must say the same thing.
        assert done["recipe"]["verification"] == event_report
    finally:
        _cleanup()
