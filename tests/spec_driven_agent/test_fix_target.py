"""Unit tests for the reported-failure parser used by FIX/MODIFY runs.

``fix_target`` turns a user's fix request ("POST /createWatchlist returns
400", "NameError in forecast.py") into a matchable :class:`ReportedTarget`
so the orchestrator can drive the customize loop at, and gate success on,
the exact failure the user reported.
"""

from __future__ import annotations

import logging

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.spec_driven_agent.planning.fix_target import (
    ReportedTarget,
    finding_matches_target,
    looks_like_fix_request,
    parse_reported_target,
)


def _domain(*class_names: str) -> DomainModel:
    string_type = PrimitiveDataType("str")
    types = set()
    for name in class_names:
        cls = Class(name=name)
        cls.attributes = {Property(name="title", type=string_type, is_id=True)}
        types.add(cls)
    return DomainModel(name="App", types=types)


# ----------------------------------------------------------------------
# Fix-intent detection
# ----------------------------------------------------------------------


class TestLooksLikeFixRequest:
    def test_error_vocabulary_is_a_fix(self):
        assert looks_like_fix_request("It crashes with a traceback")
        assert looks_like_fix_request("Please fix the bug")
        assert looks_like_fix_request("the create form is broken")
        assert looks_like_fix_request("this endpoint returns a 400")

    def test_plain_feature_add_is_not_a_fix(self):
        assert not looks_like_fix_request("add a dark theme")
        assert not looks_like_fix_request("make the header sticky")
        assert not looks_like_fix_request("")


# ----------------------------------------------------------------------
# Parsing: HTTP endpoint + status
# ----------------------------------------------------------------------


class TestParseHttp:
    def test_endpoint_status_and_entity(self):
        target = parse_reported_target(
            "The stock app crashes: POST /createWatchlist returns a 400 error.",
            _domain("Watchlist"),
        )
        assert target is not None
        assert target.kind == "http"
        assert target.status == "400"
        assert target.method == "POST"
        assert target.path == "/createWatchlist"
        # The class name is recovered from the handler and canonicalised.
        assert "Watchlist" in target.entities
        assert "400" in target.descriptor
        assert not target.is_soft

    def test_status_only_is_still_http(self):
        target = parse_reported_target(
            "the checkout call fails with 500", _domain("Order")
        )
        assert target is not None
        assert target.kind == "http"
        assert target.status == "500"


# ----------------------------------------------------------------------
# Parsing: Python exception + file
# ----------------------------------------------------------------------


class TestParseException:
    def test_exception_and_file(self):
        target = parse_reported_target(
            "There is a NameError in forecast.py when I open the page",
            _domain("Forecast"),
        )
        assert target is not None
        assert target.kind == "exception"
        assert target.exception == "NameError"
        assert target.source_file == "forecast.py"
        assert "NameError" in target.descriptor
        assert "forecast.py" in target.descriptor


# ----------------------------------------------------------------------
# Parsing: named entity (no endpoint / no exception)
# ----------------------------------------------------------------------


class TestParseEntity:
    def test_named_class_is_the_entity(self):
        target = parse_reported_target(
            "the Watchlist feature is broken", _domain("Watchlist", "User")
        )
        assert target is not None
        # No endpoint/exception, but the class name grounds it.
        assert target.kind in ("entity", "http")
        assert "Watchlist" in target.entities
        assert not target.is_soft


# ----------------------------------------------------------------------
# Parsing: soft fallback + non-fix
# ----------------------------------------------------------------------


class TestParseSoftAndNonFix:
    def test_non_fix_returns_none(self):
        assert parse_reported_target("add a dark theme", _domain("Book")) is None

    def test_soft_when_nothing_specific_parses(self):
        target = parse_reported_target(
            "it does not work, please fix it", _domain("Book")
        )
        assert target is not None
        assert target.is_soft
        # A soft target must match nothing — we never promote a finding we
        # cannot attribute to the reported failure.
        assert not finding_matches_target(
            "acceptance: entity Book — no frontend POST for it", target
        )

    def test_missing_domain_model_is_tolerated(self):
        target = parse_reported_target(
            "POST /createWatchlist returns 400", None
        )
        assert target is not None
        assert target.status == "400"
        # Entity recovered from the handler even without a model.
        assert any("watchlist" in e.lower() for e in target.entities)


# ----------------------------------------------------------------------
# Matching validator findings to the target
# ----------------------------------------------------------------------


class TestFindingMatchesTarget:
    def test_acceptance_finding_matches_reported_entity(self):
        target = parse_reported_target(
            "POST /createWatchlist returns 400", _domain("Watchlist")
        )
        assert finding_matches_target(
            "acceptance: entity Watchlist — no frontend POST for it "
            "(create form not wired)",
            target,
        )

    def test_unrelated_finding_does_not_match(self):
        target = parse_reported_target(
            "POST /createWatchlist returns 400", _domain("Watchlist", "User")
        )
        assert not finding_matches_target(
            "acceptance: entity User — no backend REST route", target
        )

    def test_none_target_never_matches(self):
        assert not finding_matches_target("anything", None)  # type: ignore[arg-type]

    def test_empty_entities_never_matches(self):
        soft = ReportedTarget(raw="x", kind="soft", descriptor="d", entities=())
        assert not finding_matches_target("acceptance: entity Book", soft)


# ----------------------------------------------------------------------
# Precision: feature wording is not a reported failure (#659 second-run test)
# ----------------------------------------------------------------------

# Live: a modify run with the library spec became "the runtime failure you
# reported" (via "fixed" and "errors"), a warning was promoted to a blocker,
# and the run ended "could not confirm the reported failure is fixed".
_FEATURE_WORDING = (
    "Show a validation error if the email is invalid.",
    "Regenerate the app from the updated class diagram.",
    "Every loan also records a daily rate and a processing fee, fixed at the moment "
    "the book is checked out.",
    "Build a todo app with form validation that shows clear error messages, and a "
    "REST API returning 404 for unknown tasks",
    "The API should return 409 when two bookings overlap.",
    "If the payment fails, show a retry button.",
    "Raise a ValueError if the quantity is negative.",
    "You must finish the app having no errors on it, do not ask me anything.",
    # From the review corpus.
    "fix the header to the top of the page",
    "fix the sidebar width to 240px",
    "add rate limiting: return 429 after 100 requests per minute per user",
    "log every 500 error to a file together with the request body",
    "Add a custom 404 error page with a link back to the dashboard.",
    # From the PR review probes.
    "Let the manager fix the price of each Room",
    "the manager can fix the daily rate per room",
    "Use status code 409 for duplicate bookings",
    "The rooms API returns 403 to guests who are not logged in",
    "the API returns 403 for non admins",
    "error code 400 for invalid input is fine",
    "GET /books/{id} returns 404 when the book does not exist",
    "Add an Error: banner at the top of the Booking page",
)

_REAL_FAILURES = (
    "I get a 500 error when I submit the form, please fix it",
    "the endpoint 500s",
    "fix the checkout page, it crashes",
    "Can you fix it in the code and give the updated files?",
    "INFO:main_api:Response status: 400 / POST /user/methods/createWatchlist/ HTTP/1.1 400 Bad Request",
    "npm error code ENOENT / npm error syscall open",
    # From the review corpus.
    "DB generation is giving an internal server error",
    "I open the frontend and it's just a blank white page, nothing renders at all",
    "same error as before",
    "nope, 500 again",
    # Caught by origin/development, missed by the first version of this change.
    "GET /books/1 returns 404 for an existing book",
    "POST /bookings 500",
    "still 500",
    "the create endpoint throws an exception",
    "Failed to fetch",
    "npm run dev fails",
    "getting CORS error when the frontend calls the backend",
)


@pytest.mark.parametrize("text", _FEATURE_WORDING)
def test_feature_wording_is_not_a_fix(text):
    assert not looks_like_fix_request(text)
    assert parse_reported_target(text, _domain("Book", "Room")) is None


@pytest.mark.parametrize("text", _REAL_FAILURES)
def test_reported_failure_is_detected(text):
    assert looks_like_fix_request(text)


@pytest.mark.parametrize("text, status", (
    ("Unknown ids should return 404, but POST /loans returns 500 now", "500"),
    ("Add a 404 page; also POST /loans returns 500", "500"),
    ("it crashes with a 500 error", "500"),
    ("the build fails", None),
))
def test_status_comes_from_the_reported_failure(text, status):
    assert parse_reported_target(text).status == status


def _framed(original_spec: str, follow_up: str) -> str:
    """The modeling-agent's framing: its summary, then the user's words."""
    return (
        "## Assistant's notes (may be inaccurate)\n\nShow clear error messages on "
        "invalid input and return 404 for unknown books.\n\n"
        "## The user's original request, verbatim\n\nThis is the authority.\n\n"
        f"{original_spec}\n\n## Subsequent user request, verbatim\n\n"
        "This later request takes precedence where it changes earlier requirements.\n\n"
        f"{follow_up}"
    )


class TestOnlyTheNewestUserMessageCounts:
    SPEC = "A library app. Fines are fixed per day; the API errors on a missing Book."

    def test_spec_and_summary_wording_does_not_make_a_fix(self):
        assert not looks_like_fix_request(_framed(self.SPEC, "Add a search bar to the Loan list"))

    def test_the_newest_message_is_parsed(self):
        target = parse_reported_target(
            _framed(self.SPEC, "POST /loan/ returns 500 when I save"), _domain("Book", "Loan"),
        )
        assert target is not None and target.kind == "http"
        assert target.entities == ("Loan",)  # not Book, named only in the spec
        assert target.raw == "POST /loan/ returns 500 when I save"

    def test_only_the_last_of_several_follow_ups_counts(self):
        earlier = "POST /loan/ returns 500 when I save\n\n## Subsequent user request, verbatim\n\n"
        assert not looks_like_fix_request(_framed(self.SPEC, earlier + "Add a due date to Loan"))

    def test_an_empty_newest_message_is_not_a_fix(self):
        assert not looks_like_fix_request(_framed(self.SPEC, ""))


def test_the_matching_rule_is_logged(caplog):
    with caplog.at_level(logging.DEBUG, logger="besser.spec_driven_agent.planning.fix_target"):
        assert looks_like_fix_request("the endpoint 500s")
    assert "status plural" in caplog.text
