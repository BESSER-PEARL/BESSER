"""A citation whose quotation marks came back typographic still locates its code.

Live evidence (2026-11-19, free tier, three follow-up runs on one app): the judge
quoted lines of a React screen and a Python module with every ``"`` turned into
``“``/``”`` or ``»``, e.g. ``<button type=“button” onClick={handleNext}>`` for
``<button type="button" onClick={handleNext}>``. No citation could match, so 18
and 20 requirements the code implements came back ``unverified`` blockers and
the repair loop edited working screens for 20 minutes. A quotation mark's glyph
is not code: it is compared as the ASCII mark, on both the quote and the source.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.planning.requirements_ledger import verify_evidence

SCREEN = '''export function OrderList({ orders, onNext }) {
  return (
    <div>
      <button type="button" className="btn" onClick={onNext} disabled={!orders.length}>
        Next
      </button>
      {orders.length === 0 ? "No orders yet" : null}
    </div>
  );
}
'''

SERVICE = '''def make_order(item):
    return Order(
        customer=item["customer"],
        total=item["total"],
    )
'''


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _verdict(path, line):
    return {"id": 1, "text": "Orders are listed with a Next button", "kind": "ui", "status": "implemented",
            "evidence": [{"path": path, "line": line}], "note": ""}


@pytest.fixture
def app(tmp_path):
    _write(tmp_path, "frontend/src/OrderList.tsx", SCREEN)
    _write(tmp_path, "backend/orders.py", SERVICE)
    return tmp_path


@pytest.mark.parametrize("path, line", [
    ("frontend/src/OrderList.tsx",
     "<button type=“button” className=“btn” onClick={onNext} disabled={!orders.length}>"),
    ("frontend/src/OrderList.tsx",
     "<button type=»button» className=»btn» onClick={onNext} disabled={!orders.length}>"),
    ("frontend/src/OrderList.tsx", "{orders.length === 0 ? ‘No orders yet’ : null}"),
    ("backend/orders.py", "customer=item[“customer”],"),
])
def test_a_quote_with_typographic_quotation_marks_locates_the_line(app, path, line):
    [checked] = verify_evidence([_verdict(path, line)], str(app))

    assert checked["status"] == "implemented", checked["note"]
    assert path in checked["cited_file_hashes"]


def test_a_quote_of_other_code_is_still_refused(app):
    """Only the glyph of a quotation mark is folded: a different value does not match."""
    [checked] = verify_evidence(
        [_verdict("backend/orders.py", "customer=item[“buyer”],")], str(app))

    assert checked["status"] == "unverified"


def test_a_source_line_with_typographic_marks_matches_its_own_quote(tmp_path):
    _write(tmp_path, "frontend/src/Empty.tsx",
           "export const Empty = () => <p className=\"hint\">{“Nothing here”}</p>;\n")
    [checked] = verify_evidence([_verdict(
        "frontend/src/Empty.tsx",
        "export const Empty = () => <p className=\"hint\">{“Nothing here”}</p>;")], str(tmp_path))

    assert checked["status"] == "implemented", checked["note"]
