"""The fix agent can cite the code of a requirement whose judge citation was rejected.

When the judge found a requirement implemented but quoted a fragment of its line,
the verifier rejected the citation and the requirement stayed unverified. The
citation-repair round runs once per source revision, and verdicts are cached on
unchanged source, so a fix agent that found the working code had no way to say
where it is: its ``task_list`` evidence never reached the requirement, it spent its
attempt making no edit, and the user was told the feature could not be confirmed.

``validate_app(citations=[{requirement, path, line}])`` now takes such lines. They
are checked exactly like the judge's citations and only for a requirement the judge
itself found implemented; a verified one replaces the rejected citation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import get_all_tools_including_generators
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

PAGE = "frontend/src/pages/Orders.tsx"
SOURCE = """import React from "react";

export default function Orders({ rows, onDelete }) {
  return (
    <div>
      <OrderTable id="orders-table" className="table" title="Open orders" rows={rows} />
      <button
        title="Remove"
        onClick={() => onDelete(rows[0].id)}
      />
    </div>
  );
}
"""
TABLE_LINE = '<OrderTable id="orders-table" className="table" title="Open orders" rows={rows} />'
KEY = "judged-source"


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


def _requirement(rid, text, status, line):
    return {"id": rid, "kind": "ui", "text": text, "status": status, "note": "judge note",
            "evidence": [{"path": PAGE, "line": line}] if line else []}


TABLE = _requirement(1, "The orders screen shows a table of open orders", "implemented", "<OrderTable id=")
EXPORT = _requirement(2, "The orders screen exports the orders as CSV", "missing", "")


@pytest.fixture
def orch(simple_library_book_model, tmp_path):
    page = tmp_path / PAGE
    page.parent.mkdir(parents=True)
    page.write_text(SOURCE, encoding="utf-8")
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
    )
    orch._requirement_judgments[KEY] = [TABLE, EXPORT]
    orch._requirement_verdicts = ledger.verify_evidence([TABLE, EXPORT], str(tmp_path))
    orch._requirement_verdicts_key = KEY
    orch._requirement_evidence_retries = {KEY}
    orch._requirements = [{"id": r["id"], "text": r["text"], "kind": r["kind"]} for r in (TABLE, EXPORT)]
    return orch


def _cite(orch, *citations, key=KEY):
    with patch.object(orch, "_requirement_cache_key", return_value=key), \
            patch.object(orch, "_requirements_to_judge", return_value=orch._requirements):
        return orch._cite_requirements(list(citations))


def _issues(orch):
    with patch.object(orch, "_requirement_cache_key", return_value=KEY), \
            patch.object(orch, "_requirements_to_judge", return_value=orch._requirements):
        return orch._collect_requirement_issues()


def test_a_verified_agent_citation_replaces_the_rejected_one_and_survives_the_next_validation(orch):
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


def test_an_unverifiable_agent_citation_is_refused_with_the_reason(orch):
    for line in ("/>", "<OrderTable id=", '<OrderTable id="orders" />'):
        [result] = _cite(orch, {"requirement": 1, "path": PAGE, "line": line})
        assert result["accepted"] is False and result["reason"], line
    assert [v["status"] for v in orch._requirement_verdicts] == ["unverified", "missing"]


def test_only_a_requirement_the_judge_found_implemented_can_be_cited(orch):
    [missing, unknown] = _cite(orch, {"requirement": 2, "path": PAGE, "line": TABLE_LINE},
                               {"requirement": 9, "path": PAGE, "line": TABLE_LINE})
    assert not missing["accepted"] and not unknown["accepted"]
    assert [v["status"] for v in orch._requirement_verdicts] == ["unverified", "missing"]


def test_a_citation_against_source_edited_since_the_judgment_is_refused(orch):
    [result] = _cite(orch, {"requirement": 1, "path": PAGE, "line": TABLE_LINE}, key="edited-source")
    assert not result["accepted"] and "judged again" in result["reason"]
    assert orch._requirement_verdicts[0]["status"] == "unverified"


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


def test_the_blocker_tells_the_agent_how_to_cite():
    [verdict] = ledger.verify_evidence([TABLE], ".")
    [issue] = ledger.ledger_issues([verdict])
    assert "validate_app citations=[{requirement: 1, path, line}]" in issue
    assert "re-judged automatically" not in issue
