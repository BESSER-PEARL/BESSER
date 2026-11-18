"""A citation that is only the start of a source line names the line it starts.

A judge sometimes quotes a line only up to its first double quote
(``<OrderTable id=`` for ``<OrderTable id="orders-table" ... />``). The verifier
rightly rejects such a quote, but it said "no source line matches the part" and
labelled the cited code absent, so the citation-repair round got no hint to cite
the whole line. The cut quote is now named as the start of the line(s) it starts.
A fragment too short to identify code (``title=``) is not taken for misquoted code.
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
        </span>
      ))}
    </div>
  );
}
"""


def _verify(tmp_path, *lines):
    path = tmp_path / PAGE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SOURCE, encoding="utf-8")
    verdict = {"id": 1, "text": "The orders screen shows a table of open orders", "kind": "ui",
               "status": "implemented", "evidence": [{"path": PAGE, "line": line} for line in lines], "note": ""}
    return ledger.verify_evidence([verdict], str(tmp_path))[0]


def test_a_quote_cut_at_the_lines_first_string_is_named_as_that_lines_start(tmp_path):
    verdict = _verify(tmp_path, "<OrderTable id=")
    assert verdict["status"] == "unverified"
    assert f"the part is only the start of {PAGE} line 6; cite the whole line" in verdict["note"]
    assert verdict[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_MALFORMED


def test_a_fragment_too_short_to_identify_code_names_its_lines_but_proves_no_code_exists(tmp_path):
    verdict = _verify(tmp_path, "title=")
    assert verdict["status"] == "unverified"
    assert f"too short to identify a line (it starts {PAGE} lines 10, 14); cite the whole line" in verdict["note"]
    assert verdict[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_ABSENT


def test_an_invented_line_start_gets_no_hint(tmp_path):
    verdict = _verify(tmp_path, "<CustomerTable id=")
    assert "no source line matches the part" in verdict["note"]
    assert verdict[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_ABSENT


def test_the_whole_line_verifies(tmp_path):
    line = '<OrderTable id="orders-table" className="table" title="Open orders" rows={rows} />'
    assert _verify(tmp_path, line)["status"] == "implemented"
