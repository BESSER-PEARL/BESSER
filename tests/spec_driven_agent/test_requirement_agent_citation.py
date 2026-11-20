"""The fix agent can complete a requirement citation the judge cut short.

When the judge found a requirement implemented but quoted only a fragment of its
line, the requirement stayed unverified for the rest of the run: the
citation-repair round runs once per source revision and verdicts are cached on
unchanged source. A fix agent that found the working code had no way to say where
it is, spent its attempt making no edit, and the user was told the feature could
not be confirmed.

``validate_app(citations=[{requirement, path, line}])`` now takes, for a requirement
the judge found implemented, the whole line its fragment was found in (the only
line it is in, ``CITED_FRAGMENT_LINES``), checked exactly like the judge's citations.
Any other line is refused: the cited code is the judge's, completed, never a line
the agent picks, so a requirement whose code is missing cannot be cleared.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import get_all_tools_including_generators
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, ValidationIssue
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

PAGE = "frontend/src/pages/Orders.tsx"
SOURCE = """import React from "react";

export default function Orders({ rows, onRefresh }) {
  return (
    <div>
      <OrderTable id="orders-table" className="table" title="Open orders" rows={rows} />
      <button
        title="Refresh"
        onClick={() => onRefresh()}
      />
    </div>
  );
}
"""
TABLE_LINE = '<OrderTable id="orders-table" className="table" title="Open orders" rows={rows} />'
ROUTER = "backend/routers/orders.py"
ROUTER_SOURCE = '''from fastapi import APIRouter

router = APIRouter()


@router.get("/orders")
def list_orders(db):
    return db.query(Order).all()


@router.delete("/orders/{order_id}")
def delete_order(order_id: int, db):
    raise NotImplementedError
'''
KEY = "judged-source"


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


def _requirement(rid, text, status, *lines, kind="ui", path=PAGE):
    return {"id": rid, "kind": kind, "text": text, "status": status, "note": "judge note",
            "evidence": [{"path": path, "line": line} for line in lines]}


TABLE = _requirement(1, "The orders screen shows a table of open orders", "implemented", "<OrderTable id=")
EXPORT = _requirement(2, "The orders screen exports the orders as CSV", "missing")
# The judge says implemented; on disk the handler is a stub.
DELETE_FRAGMENT = _requirement(3, "Users can delete an order", "implemented", "def delete_order(",
                               kind="action", path=ROUTER)
DELETE_STUB = _requirement(4, "Users can delete an order", "implemented", "raise NotImplementedError",
                           kind="action", path=ROUTER)
DELETE_INVENTED = _requirement(5, "Users can delete an order", "implemented", "db.delete(order)",
                               kind="action", path=ROUTER)
DELETE_UNCITED = _requirement(6, "Users can delete an order", "implemented", kind="action", path=ROUTER)
JUDGED = [TABLE, EXPORT, DELETE_FRAGMENT, DELETE_STUB, DELETE_INVENTED, DELETE_UNCITED]


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    for rel, text in ((PAGE, SOURCE), (ROUTER, ROUTER_SOURCE)):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
    )
    orch._requirement_judgments[KEY] = list(JUDGED)
    orch._requirement_verdicts = ledger.verify_evidence(JUDGED, str(tmp_path))
    orch._requirement_verdicts_key = KEY
    orch._requirement_evidence_retries = {KEY}
    orch._requirements = [{"id": r["id"], "text": r["text"], "kind": r["kind"]} for r in JUDGED]
    return orch


def _patched(orch, key=KEY):
    return patch.object(orch, "_requirement_cache_key", return_value=key), \
        patch.object(orch, "_requirements_to_judge", return_value=orch._requirements)


def _cite(orch, *citations, key=KEY):
    first, second = _patched(orch, key)
    with first, second:
        return orch._cite_requirements(list(citations))


def _issues(orch):
    first, second = _patched(orch)
    with first, second:
        return orch._collect_requirement_issues()


def _status(orch, rid):
    return next(v["status"] for v in orch._requirement_verdicts if v["id"] == rid)


def test_the_completed_line_replaces_the_cut_citation_and_survives_the_next_validation(orch):
    assert any(issue.startswith("requirement unverified: R1 ") for issue in _issues(orch))

    [result] = _cite(orch, {"requirement": 1, "path": PAGE, "line": TABLE_LINE})

    assert result == {"requirement": 1, "accepted": True}
    [table] = [v for v in orch._requirement_verdicts if v["id"] == 1]
    assert table["status"] == "implemented" and table["evidence"] == [{"path": PAGE, "line": TABLE_LINE}]
    assert "The fix agent supplied the citation." in table["note"]
    # The next validation on this source reads the cached judgment: the citation stays.
    issues = _issues(orch)
    assert not any("R1 " in issue for issue in issues)
    assert any(issue.startswith("requirement: R2 ") for issue in issues)


@pytest.mark.parametrize("line", [
    "/>", "<OrderTable id=", '<OrderTable id="orders" />', 'import React from "react";',
    "<div>", "return (", "onClick={() => onRefresh()}",
])
def test_any_line_but_the_one_the_fragment_is_in_is_refused(orch, line):
    [result] = _cite(orch, {"requirement": 1, "path": PAGE, "line": line})
    assert result["accepted"] is False and result["reason"]
    assert _status(orch, 1) == "unverified"


def test_another_line_beside_the_completed_one_is_refused(orch):
    [result] = _cite(orch, {"requirement": 1, "path": PAGE, "line": TABLE_LINE},
                     {"requirement": 1, "path": PAGE, "line": "onClick={() => onRefresh()}"})
    assert not result["accepted"] and _status(orch, 1) == "unverified"


@pytest.mark.parametrize("rid", [3, 4, 5, 6])
def test_a_missing_implementation_cannot_be_cleared_by_citing_working_code(orch, rid):
    """Whether the judge cut a stub's line, cited the stub, invented code or cited nothing."""
    for line in ("return db.query(Order).all()", "def delete_order(order_id: int, db):"):
        [result] = _cite(orch, {"requirement": rid, "path": ROUTER, "line": line})
        assert result["accepted"] is False, (rid, line)
    assert _status(orch, rid) == "unverified"


def test_only_a_requirement_the_judge_found_implemented_can_be_cited(orch):
    [missing, unknown] = _cite(orch, {"requirement": 2, "path": PAGE, "line": TABLE_LINE},
                               {"requirement": 9, "path": PAGE, "line": TABLE_LINE})
    assert not missing["accepted"] and not unknown["accepted"]
    assert _status(orch, 2) == "missing"


def test_a_citation_against_source_edited_since_the_judgment_is_refused(orch):
    [result] = _cite(orch, {"requirement": 1, "path": PAGE, "line": TABLE_LINE}, key="edited-source")
    assert not result["accepted"] and "judged again" in result["reason"]
    assert _status(orch, 1) == "unverified"


def test_each_malformed_citation_gets_its_own_refusal(orch):
    results = _cite(orch, "R1: the table line", {"requirement": 1, "file": PAGE, "line": TABLE_LINE},
                    {"requirement": "one", "path": PAGE, "line": TABLE_LINE})
    assert [r["citation"] for r in results] == [0, 1, 2]
    assert all(not r["accepted"] and "requirement: <id>" in r["reason"] for r in results)


def test_before_any_judgment_the_reason_says_so(orch):
    orch._requirement_judgments.clear()
    [result] = _cite(orch, {"requirement": 1, "path": PAGE, "line": TABLE_LINE})
    assert not result["accepted"] and "not been judged yet" in result["reason"]


def test_a_phase3_attempt_that_cites_the_line_clears_the_blocker(orch):
    """The whole loop: the fix agent cites through validate_app, the next validation is clean.

    The attempt edits nothing; the accepted citation moves the repair obligations,
    so the tree is checked again."""
    orch._requirement_judgments[KEY] = [TABLE]
    orch._requirement_verdicts = ledger.verify_evidence([TABLE], orch.output_dir)
    orch._requirements = [{"id": 1, "text": TABLE["text"], "kind": "ui"}]
    orch.executor.app_validator = lambda: {"verified": False, "blocker_count": 1}

    def attempt(blockers, is_first_attempt):
        assert [b.message.split(" — ")[0] for b in blockers] == ["requirement unverified: R1"]
        result = json.loads(orch.executor.execute(
            "validate_app", {"citations": [{"requirement": 1, "path": PAGE, "line": TABLE_LINE}]}))
        assert result["citations"] == [{"requirement": 1, "accepted": True}]
        return 0

    first, second = _patched(orch)
    with first, second, \
            patch.object(orch, "_collect_validation_issues",
                         side_effect=lambda *a, **k: [ValidationIssue("blocker", m)
                                                      for m in orch._collect_requirement_issues()]), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_restore_snapshot", return_value=True), \
            patch.object(orch, "_workspace_revision", return_value="rev"), \
            patch.object(orch, "_invoke_phase3_fix_loop", side_effect=attempt) as fix:
        orch._run_phase3_repair_cycle()
    assert fix.call_count == 1
    assert not [i for i in orch._validation_issues if i.severity == "blocker"]
    assert _status(orch, 1) == "implemented"


def test_validate_app_takes_citations_before_it_validates(tmp_path):
    calls = []
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.requirement_citer = lambda citations: calls.append(("cite", citations)) or [{"requirement": 1, "accepted": True}]
    executor.app_validator = lambda: calls.append(("validate", None)) or {"verified": True, "blocker_count": 0}
    citation = {"requirement": 1, "path": PAGE, "line": TABLE_LINE}

    result = json.loads(executor.execute("validate_app", {"citations": [citation]}))

    assert calls == [("cite", [citation]), ("validate", None)]
    assert result["citations"] == [{"requirement": 1, "accepted": True}]
    schema = next(tool for tool in get_all_tools_including_generators() if tool["name"] == "validate_app")["input_schema"]
    assert set(schema["properties"]["citations"]["items"]["required"]) == {"requirement", "path", "line"}


def test_the_blocker_offers_citing_only_where_a_line_can_be_cited(orch):
    by_id = {v["id"]: v for v in orch._requirement_verdicts}
    [table] = ledger.ledger_issues([by_id[1]])
    assert f"({PAGE}:6): validate_app citations=[{{requirement: 1, path, line}}]" in table
    for rid in (5, 6):
        [issue] = ledger.ledger_issues([by_id[rid]])
        assert "validate_app" not in issue and "re-judged automatically" in issue, rid
    # A cited stub is refuted, not unlocated: a blocker that asks for the behaviour or a real citation.
    [stub] = ledger.ledger_issues([by_id[4]])
    assert stub.startswith("requirement unsupported:") and "implement the behaviour" in stub
