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
    assert "1 not confirmed, 1 checked and not met" in summary
    assert "### Not confirmed" in summary


# ---------------------------------------------------------------------------
# A citation the verifier refutes is evidence of a defect: ``requirement unsupported:``
# ---------------------------------------------------------------------------

_SERVICE = '''from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI()


class OrderIn(BaseModel):
    start: int
    end: int

    @field_validator("end")
    def end_after_start(cls, value):
        return value


@app.post("/orders/{order_id}/cancel")
def cancel_order(order_id: int):
    raise NotImplementedError


def export_orders():
    # Orders are written to CSV here.
    pass
'''


def _service(root: Path) -> str:
    path = root / "backend" / "service.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_SERVICE, encoding="utf-8")
    (root / "backend" / "broken.py").write_text("def total(:\n    return 1\n", encoding="utf-8")
    (root / "backend" / "writer.py").write_text("def write(rows):\n    return len(rows)\n", encoding="utf-8")
    return str(root)


def _cites(rid: int, kind: str, path: str, line: str, text: str = "Orders behave as requested.") -> dict:
    return {"id": rid, "text": text, "kind": kind, "status": "implemented",
            "evidence": [{"path": path, "line": line}], "note": ""}


@pytest.mark.parametrize("verdict, refuted", [
    # An action body that is still a stub.
    (_cites(1, "transition", "backend/service.py", "raise NotImplementedError", "A customer can cancel an order."),
     ledger.REFUTED_ROLE),
    # A validator that returns its input unchanged.
    (_cites(2, "validation", "backend/service.py", "return value", "An order's end comes after its start."),
     ledger.REFUTED_ROLE),
    # A comment.
    (_cites(3, "implementation", "backend/service.py", "# Orders are written to CSV here.",
            "Orders can be exported to CSV."), ledger.REFUTED_ROLE),
    # A file that does not parse.
    (_cites(5, "implementation", "backend/broken.py", "return 1", "Orders have a total."), ledger.REFUTED_ROLE),
    # A file the app does not have, quoting code that is nowhere in it.
    (_cites(6, "implementation", "backend/export.py", "writer.writerow(order.to_row())",
            "Orders can be exported to CSV."), ledger.REFUTED_MISSING_FILE),
])
def test_a_refuted_citation_is_a_repairable_blocker(tmp_path, verdict, refuted):
    """The judge said implemented, and the code it pointed at does not do it. Before: not confirmed."""
    from besser.spec_driven_agent.validation.issues import unresolved_defects

    checked, = ledger.verify_evidence([verdict], _service(tmp_path))
    assert checked["status"] == "unverified"
    assert checked[ledger.EVIDENCE_REFUTED] == refuted

    message, = ledger.ledger_issues([checked])
    assert message.startswith("requirement unsupported:")
    issue = _classify_issue(message)
    assert issue.severity == "blocker" and is_completion_issue(issue) and not is_not_confirmed(issue)
    # Like the judge's own ``requirement:`` verdict: repaired and counted, but not "incomplete".
    assert unresolved_defects([issue]) == []


@pytest.mark.parametrize("verdict", [
    # The quote is not in the cited file, which exists: a misquote as often as invented code.
    _cites(7, "implementation", "backend/service.py", "writer.writerow(order.to_row())"),
    # The cited file is missing, but the quoted line is in another file: the wrong file named.
    _cites(8, "implementation", "backend/export.py", "return len(rows)"),
    # A declaration where a constraint needs enforcing: calibrated as false on recorded runs.
    _cites(9, "validation", "backend/service.py", "end: int", "An order's end must be after its start."),
    # A fragment of a line, not a whole line.
    _cites(10, "transition", "backend/service.py", "def cancel_order(order_id"),
    # An import: the rule may be in what it imports (``from backend.rules import check_capacity``).
    _cites(4, "rule", "backend/service.py", "from fastapi import FastAPI", "An order holds at most 100 items."),
])
def test_a_citation_that_was_only_not_located_stays_not_confirmed(tmp_path, verdict):
    checked, = ledger.verify_evidence([verdict], _service(tmp_path))
    assert checked["status"] == "unverified" and not checked.get(ledger.EVIDENCE_REFUTED)
    message, = ledger.ledger_issues([checked])
    assert message.startswith("requirement unverified:") and is_not_confirmed(message)
    assert _blockers([message]) == []


def test_a_verdict_the_judge_itself_left_unverified_stays_not_confirmed():
    message, = ledger.ledger_issues([{"id": 11, "text": "Orders can be exported.", "kind": "implementation",
                                      "status": "unverified", "evidence": "", "note": "not shown the code"}])
    assert message.startswith("requirement unverified:") and is_not_confirmed(message)


def test_a_refuted_citation_is_reported_as_checked_and_not_met(tmp_path):
    from besser.spec_driven_agent.validation.user_wording import user_sentence
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
        build_verification_report,
        run_summary_markdown,
    )

    checked = ledger.verify_evidence([
        _cites(1, "transition", "backend/service.py", "raise NotImplementedError", "A customer can cancel an order."),
        _cites(2, "implementation", "backend/service.py", "writer.writerow(order.to_row())", "Orders export."),
    ], _service(tmp_path))
    messages = ledger.ledger_issues(checked)
    recipe = {"requirements": checked, "validation_issues": [
        {"severity": _classify_issue(m).severity, "message": m} for m in messages]}
    report = build_verification_report(recipe)

    assert [(item.id, item.why) for item in report.shippedUnenforced] == [
        ("R1", "the code the check was pointed at does not do this")]
    assert [item.id for item in report.notVerified if item.kind == "requirement"] == ["R2"]
    assert user_sentence(messages[0]) == (
        "Requirement R1: the code the check was pointed at does not do this: A customer can cancel an order.")
    summary = "\n".join(run_summary_markdown(recipe, report))
    assert "### Checked and not met" in summary


def test_a_failed_workflow_is_never_listed_as_not_confirmed():
    """``api scenario: ... failed`` is a blocker; the report must not call it "could not check"."""
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

    message = "api scenario: orders failed: expected 201, got 500"
    recipe = {"api_scenarios": [{"scenario_id": "orders", "report": {"status": "failed"},
                                 "scenario": {"requests": []}}],
              "validation_issues": [{"severity": "blocker", "message": message}]}
    report = build_verification_report(recipe)
    assert is_completion_issue(_classify_issue(message))
    assert [item.id for item in report.failed] == ["orders"] and report.counts.failed == 1
    assert all(item.id != "orders" for item in report.notVerified + report.shippedUnenforced)


def test_each_listed_finding_is_in_the_bucket_its_count_puts_it_in():
    """The report and the counts read one split: not confirmed findings under "could not check",
    blockers under "checked and not met"."""
    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

    messages = ["task unverified: Order list - remains unresolved: sort orders.",
                runtime_check_unverified("the constructibility probe", "timed out"),
                "action contract: POST /orders/{id}/cancel answers HTTP 501"]
    report = build_verification_report({"validation_issues": [
        {"severity": _classify_issue(m).severity, "message": m} for m in messages]})
    assert len(report.notVerified) == sum(map(is_not_confirmed, messages)) == 2
    assert len(report.shippedUnenforced) == len(_blockers(messages)) == 1


# Correct code the first refutation rule refuted (second review). Each must stay not confirmed:
# a false refutation reads "rules not enforced" and sends repair to rewrite working code.
_MODELS_RULES = {
    "backend/models.py": (
        "from sqlalchemy import Column, Integer, String\n"
        "from sqlalchemy.orm import column_property\n"
        "from database import Base\n"
        "\n"
        "class Order(Base):\n"
        "    __tablename__ = 'orders'\n"
        "    id = Column(Integer, primary_key=True)\n"
        "    reference = Column(String(40), nullable=False, unique=True)\n"
        "    status = Column(String(20), nullable=False, default='pending')\n"
        "    quantity = Column(Integer, nullable=False)\n"
        "    unit_price = Column(Integer, nullable=False)\n"
        "    total = column_property(quantity * unit_price)\n"),
    "backend/rules.py": "def check_capacity(order):\n    if order.quantity > 100:\n        raise ValueError('x')\n",
    "backend/routers.py": "from backend.rules import check_capacity\n\ndef cancel(order_id):\n    pass\n",
}


def _rules_app(root: Path) -> str:
    for rel, text in _MODELS_RULES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return str(root)


@pytest.mark.parametrize("verdict", [
    _cites(1, "transition", "backend/models.py", "status = Column(String(20), nullable=False, default='pending')",
           "A new order is pending until paid."),
    _cites(2, "computed", "backend/models.py", "total = column_property(quantity * unit_price)",
           "An order's total is quantity times unit price."),
    _cites(3, "rule", "backend/routers.py", "from backend.rules import check_capacity",
           "An order holds at most 100 items."),
    # A wrong path prefix and a paraphrase of a line that exists.
    _cites(4, "uniqueness", "app/models/order.py", "reference = Column(String, unique=True)",
           "Order references are unique."),
])
def test_correct_code_is_never_refuted(tmp_path, verdict):
    checked, = ledger.verify_evidence([verdict], _rules_app(tmp_path))
    assert not checked.get(ledger.EVIDENCE_REFUTED), checked["note"]
    if checked["status"] != "implemented":
        message, = ledger.ledger_issues([checked])
        assert is_not_confirmed(message) and _blockers([message]) == []


def test_a_true_stub_beside_them_is_still_refuted(tmp_path):
    checked, = ledger.verify_evidence([_cites(5, "transition", "backend/routers.py", "pass",
                                              "A customer can cancel an order.")], _rules_app(tmp_path))
    assert checked[ledger.EVIDENCE_REFUTED] == ledger.REFUTED_ROLE
    message, = ledger.ledger_issues([checked])
    assert message.startswith("requirement unsupported:") and "validate_app citations" not in message
