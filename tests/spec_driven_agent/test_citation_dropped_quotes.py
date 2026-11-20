"""A citation whose quotation marks the model left out still locates its one line.

Live evidence (2026-11-19, free tier): the judge cited JSX and Python lines with
every double quotation mark missing (``<button type=submit ...>``,
``className={ds-nav-links}``, ``raise ValueError(title must not be empty)``)
while single quotes in the same answer survived. Our decoding of the tool
arguments keeps escaped quotes (tested below), so the marks were missing in the
arguments the provider returned. Such a citation is now compared with the
quotation marks removed on both sides, and accepted only when exactly one line of
the cited file matches; it still has to pass the same role check as any line.
"""

from __future__ import annotations

import json

import pytest

from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.planning.requirements_ledger import verify_evidence
from besser.spec_driven_agent.providers.llm_client import _openai_tool_use_block

FORM = '''export function OrderForm({ saving }) {
  return (
    <form>
      <div className={"form-actions"}>
        <Link to="/orders" className="btn btn-primary">Back</Link>
        <button type="submit" className="btn btn-primary" disabled={saving}>
          Save
        </button>
      </div>
    </form>
  );
}
'''

SERVICE = '''def check(order):
    if not order.reference:
        raise ValueError("reference must not be empty")
    total = price("net", order)
    total = price(net, "order")
    return total
'''


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _check(root, path, line):
    [checked] = verify_evidence([{"id": 1, "text": "An order can be saved", "kind": "ui", "status": "implemented",
                                  "evidence": [{"path": path, "line": line}], "note": ""}], str(root))
    return checked


@pytest.fixture
def app(tmp_path):
    _write(tmp_path, "frontend/src/OrderForm.tsx", FORM)
    _write(tmp_path, "backend/orders.py", SERVICE)
    return tmp_path


@pytest.mark.parametrize("path, line", [
    ("frontend/src/OrderForm.tsx", "<button type=submit className=btn btn-primary disabled={saving}>"),
    ("frontend/src/OrderForm.tsx", "<div className={form-actions}>"),
    ("frontend/src/OrderForm.tsx", "<Link to=/orders className=btn btn-primary>Back</Link>"),
    ("backend/orders.py", "raise ValueError(reference must not be empty)"),
])
def test_a_quote_without_its_quotation_marks_locates_the_one_line(app, path, line):
    checked = _check(app, path, line)

    assert checked["status"] == "implemented", checked["note"]
    assert path in checked["cited_file_hashes"]


def test_a_quote_that_two_lines_match_without_quotation_marks_is_refused(app):
    """``price(net, order)`` is either line once their marks are gone: neither is picked."""
    checked = _check(app, "backend/orders.py", "total = price(net, order)")

    assert checked["status"] == "unverified"


def test_a_quote_without_quotation_marks_of_another_file_is_refused(app):
    checked = _check(app, "backend/orders.py", "<button type=submit className=btn btn-primary disabled={saving}>")

    assert checked["status"] == "unverified"


def test_our_decoding_of_tool_arguments_keeps_escaped_quotation_marks():
    """The provider's arguments string, through the block, the judge's tool input and its evidence parser."""
    line = '<button type="submit" className={"btn"}>'
    raw = json.dumps({"verdicts": [{"id": 1, "status": "implemented", "evidence": [
        {"path": "frontend/src/OrderForm.tsx", "line": line}]}]})
    block = _openai_tool_use_block("call_1", ledger._SUBMIT_VERDICTS_TOOL["name"], raw, {})

    payload = ledger._tool_input({"content": [block]}, ledger._SUBMIT_VERDICTS_TOOL["name"])
    evidence = ledger._parse_evidence(payload["verdicts"][0]["evidence"])

    assert evidence == [{"path": "frontend/src/OrderForm.tsx", "line": line}]


def test_a_quote_that_kept_some_quotation_marks_is_compared_with_them(tmp_path):
    """Only a quote with no mark at all lost them; ``"a, b"`` is not the two strings ``"a", "b"``."""
    _write(tmp_path, "backend/check.py", 'def check(q):\n    if q < 1:\n'
                                         '        raise HTTPException(422, "quantity", "must be positive")\n')

    checked = _check(tmp_path, "backend/check.py", 'raise HTTPException(422, "quantity, must be positive")')

    assert checked["status"] == "unverified"
