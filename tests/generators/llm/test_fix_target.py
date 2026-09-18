"""Unit tests for the reported-failure parser used by FIX/MODIFY runs.

``fix_target`` turns a user's fix request ("POST /createWatchlist returns
400", "NameError in forecast.py") into a matchable :class:`ReportedTarget`
so the orchestrator can drive the customize loop at, and gate success on,
the exact failure the user reported.
"""

from __future__ import annotations

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.fix_target import (
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
