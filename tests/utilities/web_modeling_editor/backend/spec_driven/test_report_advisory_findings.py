"""Every finding lands in one state the user can see.

A plain warning (a screen that threw or got a 5xx, a call to a route no backend
serves, a form field of the wrong type) was in none of the report's lists and
in no count of ``BESSER_GENERATION.md``: a run whose screens all failed read
"0 not confirmed, 0 checked and not met". Such findings are now listed as
"seen, not blocking", with their count. They still block nothing and withhold
nothing: they are not counted as not confirmed. What the screen check could not
look at is a "could not verify" item, apart from what it saw.
"""

from besser.spec_driven_agent.validation.issues import (
    JUDGMENT_DEFERRED,
    REQUIREMENTS_NOT_EXTRACTED,
    _classify_issue,
    is_completion_issue,
    is_not_confirmed,
)
from besser.spec_driven_agent.validation.user_wording import GENERIC_SENTENCE
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    build_verification_report,
    run_summary_markdown,
)

SEEN = [
    "endpoint coherence: src/pages/Settings.tsx line 12 calls GET /api/shop/, but no generated backend "
    "route matches it (examples: /order/). Copy the exact path and method from the backend endpoint manifest.",
    "screen warning: [frontend] route /orders (src/pages/Orders.tsx): uncaught exception TypeError: x is undefined",
    "screen warning: [frontend] route /: a 5xx from the backend: GET /api/order/ -> 500",
    "form field type: src/pages/OrderForm.tsx field quantity is a text input for an integer",
]
COVERAGE = "screen warning: [frontend] route /orders: not checked: records on the screens (GET /order/ answered 500)"


def _recipe(*messages: str) -> dict:
    return {
        "validation_issues": [{"severity": _classify_issue(m).severity, "message": m} for m in messages],
        "api_scenarios": [{"scenario_id": "order-flow", "report": {"status": "passed"},
                           "scenario": {"requests": [{"method": "POST", "path": "/order/"}]}}],
    }


def test_plain_warnings_are_listed_as_seen_not_blocking():
    for message in SEEN:
        issue = _classify_issue(message)
        assert issue.severity == "warning" and not is_completion_issue(issue) and not is_not_confirmed(issue)

    report = build_verification_report(_recipe(*SEEN))

    assert report.counts.advisory == 4
    assert report.counts.notVerified == 0
    assert [item.what for item in report.advisory] == [
        "A screen calls an API route or client function the backend does not have.",
        "A screen of the app reported a problem when it was opened in a browser.",
        "A screen of the app reported a problem when it was opened in a browser.",
        "A form field does not match the type of value it collects.",
    ]
    summary = "\n".join(run_summary_markdown(_recipe(*SEEN), report))
    assert "4 seen, not blocking" in summary
    assert "### Seen, not blocking" in summary
    assert "- `A screen of the app reported a problem when it was opened in a browser.` (2 findings)" in summary


def test_what_the_screen_check_could_not_look_at_is_not_verified_not_seen():
    report = build_verification_report(_recipe(COVERAGE, "screens: not run (no browser on this server)"))
    assert report.counts.advisory == 0
    assert report.counts.notVerified == 2


def test_blockers_and_counts_notes_stay_out_of_the_list():
    report = build_verification_report(_recipe(
        "create contract: b: POST /order/ - observed a server/persistence failure creating Order (500: x)",
        "ruff: (+3 more issues truncated)",
    ))
    assert report.counts.advisory == 0 and report.advisory == []


def test_an_unconfirmed_requirement_is_listed_once_as_not_confirmed():
    recipe = _recipe("requirement unverified: R1 — An order records its quantity — the cited evidence was "
                     "not located in the delivered source", REQUIREMENTS_NOT_EXTRACTED)
    recipe["requirements"] = [{"id": 1, "text": "An order records its quantity", "status": "unverified"}]

    report = build_verification_report(recipe)

    assert report.counts.advisory == 0 and report.advisory == []
    assert [item.id for item in report.notVerified] == ["R1"]
    assert "### Seen, not blocking" not in "\n".join(run_summary_markdown(recipe, report))


def test_a_deferred_requirements_judgment_is_not_confirmed_and_a_rollback_has_its_own_sentence():
    report = build_verification_report(_recipe(
        f"{JUDGMENT_DEFERRED} until startup/data-entry blockers are fixed",
        "validation: Phase 3 fix attempt 2 was rolled back - it added blockers - to the tree from before it. "
        "Findings seen only in the discarded tree (they may still be real): none",
    ))

    assert [item.what for item in report.notVerified] == [
        "Your request's requirements were not checked: the app first has to start and save records."]
    assert [item.what for item in report.advisory] == [
        "An automatic fix attempt was undone because it made the app worse; the run's detailed report lists "
        "what was seen only in the undone version."]
    assert GENERIC_SENTENCE not in {item.what for item in report.notVerified + report.advisory}
