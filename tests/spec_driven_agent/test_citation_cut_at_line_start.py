"""A citation that is only a fragment of real code proves nothing, but is not invented.

A judge sometimes quotes a line only up to its first double quote
(``<OrderTable id=`` for ``<OrderTable id="orders-table" ... />``, ``title=`` for
``title="Edit"``). Such a fragment was counted as code that exists nowhere, so one
cut citation sank a verdict whose other citations were whole, verified lines, and
the requirement was reported unverified although its code works.

A fragment that is code of the cited file is now located like the part of an
abridged quote: it proves no role (a whole line must), and it does not fail the
verdict. When it is in exactly one line, the verdict records that line, the only
line the fix agent may cite in its place (``CITED_FRAGMENT_LINES``).
"""

from besser.spec_driven_agent.planning import requirements_ledger as ledger

PAGE = "frontend/src/pages/Orders.tsx"
SOURCE = """import React from "react";

export default function Orders({ rows, onEdit, onDelete }) {
  return (
    <div>
      <OrderTable id="orders-table" className="table" title="Open orders" rows={rows} />
      {rows.map((row) => (
        <span key={row.id}>
          <button
            title="Edit"
            onClick={() => onEdit(row)}
          />
          <button
            title="Remove"
            onClick={() => onDelete(row.id)}
          />
          ); // the row actions
        </span>
      ))}
    </div>
  );
}
"""
TABLE_LINE = '<OrderTable id="orders-table" className="table" title="Open orders" rows={rows} />'


def _verify(tmp_path, *lines, text="The orders screen shows a table of open orders", kind="ui"):
    path = tmp_path / PAGE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SOURCE, encoding="utf-8")
    verdict = {"id": 1, "text": text, "kind": kind,
               "status": "implemented", "evidence": [{"path": PAGE, "line": line} for line in lines], "note": ""}
    return ledger.verify_evidence([verdict], str(tmp_path))[0]


def test_a_cut_fragment_beside_a_whole_verified_line_does_not_sink_the_verdict(tmp_path):
    verdict = _verify(tmp_path, "title=", "onClick={() => onEdit(row)}",
                      text="Each row has an Edit button", kind="action")
    assert verdict["status"] == "implemented"


def test_a_fragment_alone_proves_nothing_and_points_at_its_only_line(tmp_path):
    verdict = _verify(tmp_path, "<OrderTable id=")
    assert verdict["status"] == "unverified"
    assert "no cited part is a whole source line" in verdict["note"]
    assert ledger.EVIDENCE_UNLOCATED not in verdict
    assert verdict[ledger.CITED_FRAGMENT_LINES] == [f"{PAGE}:6"]


def test_a_fragment_in_several_lines_points_at_none(tmp_path):
    verdict = _verify(tmp_path, "title=")
    assert verdict["status"] == "unverified"
    assert ledger.CITED_FRAGMENT_LINES not in verdict and ledger.EVIDENCE_UNLOCATED not in verdict


def test_an_invented_line_start_is_still_absent(tmp_path):
    verdict = _verify(tmp_path, "<CustomerTable id=")
    assert "no source line matches the part" in verdict["note"]
    assert verdict[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_ABSENT


def test_the_whole_line_verifies(tmp_path):
    assert _verify(tmp_path, TABLE_LINE)["status"] == "implemented"


def test_a_line_of_punctuation_alone_proves_nothing(tmp_path):
    """``{`` or ``/>`` is a whole line of many files; it identifies no code, nor does
    punctuation followed by a comment."""
    for lines in (("/>", "))}"), ("); // the row actions",)):
        verdict = _verify(tmp_path, *lines)
        assert verdict["status"] == "unverified", lines
        assert "identifies no code" in verdict["note"], lines
    assert _verify(tmp_path, "/>", TABLE_LINE)["status"] == "implemented"
