"""The end-of-run report says when no rules were stated, and how far an implemented verdict was checked.

Live evidence: r9-p11 ("manage a hotel. To manage bookings.") stated no rule, so the
ledger was empty and the report said nothing about it; r9-p61 showed four UI
requirements as bare "implemented" while the frontend build was skipped.
Recipes written before these fields existed render exactly as before.
"""

from besser.spec_driven_agent.run_recap import NO_RULES_STATED
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    build_verification_report,
    run_summary_markdown,
)

_IMPLEMENTED = {"id": 1, "text": "Show the stock prices of the last 7 days", "status": "implemented",
                "evidence": [{"path": "frontend/src/Prices.jsx", "quote": "return <PriceTable rows={rows} />;"}]}
_NOTE = "Found in the source only; the frontend was not built"


def _markdown(recipe: dict) -> str:
    return "\n".join(run_summary_markdown(recipe, build_verification_report(recipe)))


def test_no_rules_stated_is_reported_in_the_done_event_and_the_download():
    recipe = {"generator_used": None, "business_rules_stated": False}

    report = build_verification_report(recipe)

    assert [item.what for item in report.notVerified] == [NO_RULES_STATED]
    assert NO_RULES_STATED in _markdown(recipe)


def test_rules_stated_or_unknown_adds_nothing():
    for recipe in ({"business_rules_stated": True}, {"business_rules_stated": None}, {}):
        assert build_verification_report(recipe) == build_verification_report({})
        assert NO_RULES_STATED not in _markdown({"generator_used": None, **recipe})


def test_unverified_implemented_verdict_says_it_was_found_in_the_source_only():
    verdict = {**_IMPLEMENTED, "verification": "unverified", "verification_note": _NOTE}

    how = build_verification_report({"requirements": [verdict]}).foundInCode[0].how

    assert how.startswith("implementation cited and re-checked against the delivered source: ")
    assert how.endswith(_NOTE)


def test_a_verdict_only_found_in_the_source_is_not_counted_as_confirmed():
    """A citation is read, not run: it must not count as checked and working."""
    workflow = {"scenario_id": "order_flow", "report": {"status": "passed"},
                "scenario": {"requests": [{"method": "POST", "path": "/order/"}]}}
    recipe = {"generator_used": None, "api_scenarios": [workflow], "requirements": [
        {**_IMPLEMENTED, "verification": "unverified", "verification_note": _NOTE},
        {**_IMPLEMENTED, "id": 2, "verification": "api_verified", "verification_note": "Found in the source; API: ..."},
    ]}

    report = build_verification_report(recipe)

    assert [item.id for item in report.verified] == ["R2", "order_flow"]
    assert [item.id for item in report.foundInCode] == ["R1"]
    assert (report.counts.verified, report.counts.foundInCode) == (2, 1)
    assert report.verified[1].how.startswith("a test the agent wrote passed: 1 request(s) run against the app")
    assert ("- Verification: 2 confirmed by running the app's API, 1 found in the code but not run, "
            "0 not confirmed, 0 checked and missing in the code") in _markdown(recipe)


def test_note_survives_a_long_citation():
    evidence = [{"path": f"backend/routers/r{i}.py", "quote": "x = compute_the_score(a, b, c)"} for i in range(12)]
    verdict = {**_IMPLEMENTED, "evidence": evidence, "verification": "unverified", "verification_note": _NOTE}

    assert build_verification_report({"requirements": [verdict]}).foundInCode[0].how.endswith(_NOTE)


def test_a_verified_requirement_names_the_call_that_exercised_it():
    note = "Cited in the source; API: a passing workflow called POST /order/, the cited route"
    verified = {**_IMPLEMENTED, "verification": "verified", "verification_note": note}

    how = build_verification_report({"requirements": [verified]}).verified[0].how

    assert how.startswith("implementation cited and re-checked against the delivered source: ")
    assert how.endswith(note)


def test_old_verdicts_render_as_before():
    old = build_verification_report({"requirements": [_IMPLEMENTED]}).verified[0].how

    assert old.startswith("implementation cited and re-checked against the delivered source: ")
    assert "Cited in the source" not in old and "Found in the source" not in old
