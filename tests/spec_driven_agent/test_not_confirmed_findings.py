"""Severity follows the kind of evidence: an observed failure blocks, "could not check" does not.

A run used to end "with blockers" because the judge said a requirement IS implemented and its
citation could not be located word for word (a quote with other quote marks, a line cut short).
Nothing had failed: the check reached no conclusion. Such findings are now *not confirmed*:
listed for the user, never a blocker, never sent to repair and never counted in ``blockerCount``.
A ``missing`` or ``partial`` verdict and an observed server failure still block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.validation import constructibility
from besser.spec_driven_agent.validation.issues import (
    ValidationIssue,
    _classify_issue,
    is_completion_issue,
    is_not_confirmed,
    required_check_unverified,
    runtime_check_unverified,
)

_MODELS = '''from sqlalchemy import Column, Integer, String
from database import Base


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    reference = Column(String(40), nullable=False, unique=True)
    status = Column(String(20), nullable=False, default="draft")
'''

_PAGE = '''export default function OrdersPage() {
  return (
    <form>
      <input type="file" accept=".csv,.txt" name="attachment" />
      <button type="submit">Save order</button>
    </form>
  );
}
'''


def _app(root: Path) -> str:
    for rel, text in (("backend/models.py", _MODELS), ("frontend/src/pages/OrdersPage.tsx", _PAGE)):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(root)


def _implemented(rid: int, text: str, path: str, line: str) -> dict:
    return {"id": rid, "text": text, "kind": "ui" if path.endswith(".tsx") else "data",
            "status": "implemented", "evidence": [{"path": path, "line": line}], "note": ""}


def _blockers(messages: list[str]) -> list[ValidationIssue]:
    return [issue for issue in map(_classify_issue, messages) if is_completion_issue(issue)]


def test_affirmative_verdicts_whose_citation_was_not_located_are_not_blockers(tmp_path):
    """Stored verdicts replayed: the judge said implemented, the cited file exists, the quote
    differs in its glyphs or stops mid-line. Before: one blocker each."""
    output = _app(tmp_path)
    judged = [
        # Other quote marks than the file's.
        _implemented(1, "The order screen accepts a CSV or text attachment.",
                     "frontend/src/pages/OrdersPage.tsx", "accept=«.csv,.txt»"),
        # The quote paraphrases the declaration.
        _implemented(2, "Every order has a unique reference.",
                     "backend/models.py", "reference = Column(String, unique=True, nullable=False)"),
        # A line the cited file does not hold, although the file exists.
        _implemented(3, "An order starts as a draft.", "backend/models.py", "status = 'draft'"),
    ]
    checked = ledger.verify_evidence(judged, output)
    assert [v["status"] for v in checked] == ["unverified"] * 3
    assert all((Path(output) / v["evidence"][0]["path"]).is_file() for v in checked)

    messages = ledger.ledger_issues(checked)
    assert len(messages) == 3 and all(m.startswith("requirement unverified:") for m in messages)
    assert _blockers(messages) == []
    assert all(is_not_confirmed(m) and _classify_issue(m).severity == "warning" for m in messages)


@pytest.mark.parametrize("status, prefix", [("missing", "requirement:"), ("partial", "requirement partial:")])
def test_a_missing_or_partial_verdict_still_blocks(status, prefix):
    message, = ledger.ledger_issues([{"id": 4, "text": "An order can be cancelled.", "kind": "behaviour",
                                      "status": status, "evidence": "", "note": ""}])
    assert message.startswith(prefix)
    assert not is_not_confirmed(message)
    assert [b.message for b in _blockers([message])] == [message]


@pytest.mark.parametrize("message", [
    f"{constructibility.PREFIX} backend: POST /orders/ - observed a server/persistence failure "
    "creating Order (500: Internal Server Error); attempted input: {}",
    f"{constructibility.ACTION_PREFIX} backend: POST /orders/{{id}}/cancel - observed a server/persistence "
    "failure calling cancel on a Order in state draft (500: boom)",
    "api scenario: order-lifecycle failed: {\"assertion_failures\": [\"GET /orders/1 status 404\"]}",
    "application startup: backend/main_api.py line 3 starts the server at import time",
    "Syntax error in backend/models.py: line 4",
])
def test_an_observed_failure_still_blocks(message):
    assert not is_not_confirmed(message)
    assert [b.message for b in _blockers([message])] == [message]


def test_a_guessed_create_the_app_refused_is_not_confirmed():
    """The create probe's own guessed POST, refused with 422 and settled by no scenario, is
    re-prefixed ``runtime unverified:`` by the orchestrator. Before: a blocker."""
    refused = (f"{constructibility.UNVERIFIED_PREFIX} backend: POST /orders/ - autogenerated input was "
               "refused (422: {\"detail\": \"reference: too short\"}); attempted input: {\"reference\": \"x\"}. "
               "These guessed inputs do not prove this endpoint is broken.")
    for message in (refused, "runtime unverified: " + refused):
        issue = _classify_issue(message)
        assert issue.severity == "warning"
        assert is_not_confirmed(issue) and not is_completion_issue(issue)


@pytest.mark.parametrize("message", [
    "requirement unverified: requirement extraction failed; the original specification has not been checked.",
    "requirement unverified: verification stopped or budget exhausted; "
    "original-specification coverage is not verified.",
    "requirement unverified: R7 — requirement extraction completeness has not been established",
    required_check_unverified("the requirements judge", "it returned no verdicts"),
    runtime_check_unverified("the constructibility probe", "the shell sandbox is unavailable"),
    "action unverified: backend: POST /orders/{id}/cancel - cancel on Order refused every state",
])
def test_a_check_that_reached_no_conclusion_is_not_confirmed(message):
    issue = _classify_issue(message)
    assert issue.severity == "warning"
    assert is_not_confirmed(issue) and not is_completion_issue(issue)


def test_a_not_confirmed_finding_built_as_a_blocker_does_not_count():
    """Phase 3 skipped for a budget stop builds its finding as a blocker directly, and a recipe
    or checkpoint written before keeps the old severity: the prefix decides."""
    skipped = ValidationIssue("blocker", "requirement unverified: Phase 3 validation did not run: "
                                         "runtime budget exhausted. The current application has not "
                                         "completed final verification.")
    stored = ValidationIssue("blocker", "runtime unverified: create unverified: backend: POST /orders/ - "
                                        "autogenerated input was refused (409: duplicate)")
    assert not is_completion_issue(skipped) and not is_completion_issue(stored)
    assert is_completion_issue(ValidationIssue("blocker", "requirement: R2 — x is not implemented"))


def test_the_report_lists_them_as_not_confirmed_and_says_how_many(tmp_path):
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
        build_verification_report,
        run_summary_markdown,
    )

    checked = ledger.verify_evidence([
        _implemented(1, "The order screen accepts a CSV or text attachment.",
                     "frontend/src/pages/OrdersPage.tsx", "accept=\u00ab.csv,.txt\u00bb"),
        {"id": 2, "text": "An order can be cancelled.", "kind": "behaviour", "status": "missing",
         "evidence": "", "note": ""},
    ], _app(tmp_path))
    recipe = {"requirements": checked, "api_scenarios": [{"scenario_id": "orders", "report": {"status": "passed"},
                                                          "scenario": {"requests": []}}],
              "validation_issues": [{"severity": "warning", "message": m} for m in ledger.ledger_issues(checked)]}
    report = build_verification_report(recipe)

    assert [item.id for item in report.notVerified] == ["R1"]
    assert [item.id for item in report.shippedUnenforced] == ["R2"]
    summary = "\n".join(run_summary_markdown(recipe, report))
    assert "1 not confirmed, 1 checked and missing in the code" in summary
    assert "### Not confirmed" in summary
