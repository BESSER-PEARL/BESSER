"""A verbatim quote from the middle of a JSX line locates.

Live evidence, r12-p21 (stock tracker), R9 citation 3: the judge quoted
``<th>Target date</th><th>Method</th><th className="num">Predicted close</th>``
from the middle of WatchlistDetail.jsx line 174. The run ended blocked on it:
the quote was lexed after its whitespace was collapsed, so ``<th className="num">``
became ``<thclassName="num">``, lexed differently from the file line, and never
matched. The verdict was then labelled ``absent``, so Phase 3 spent a fix attempt
on a citation no edit could change.
"""

from besser.spec_driven_agent.planning import requirements_ledger as ledger

PAGE = "frontend/src/pages/WatchlistDetail.jsx"
SOURCE = """import React from 'react';

export default function WatchlistDetail({ fc }) {
  // per tracked company, next 2 working days
  const note = "Live market data was unavailable, so SIMULATED demo prices were stored.";
  return (
    <table>
      <thead><tr><th>Target date</th><th>Method</th><th className="num">Predicted close</th></tr></thead>
      <tbody>{fc.map((f) => (<tr key={f.id}><td>{f.targetDate}</td></tr>))}</tbody>
    </table>
  );
}
"""
R9_QUOTE = '<th>Target date</th><th>Method</th><th className="num">Predicted close</th>'


def _app(tmp_path):
    path = tmp_path / PAGE
    path.parent.mkdir(parents=True)
    path.write_text(SOURCE, encoding="utf-8")
    return str(tmp_path)


def _verify(workspace, quote):
    verdict = {"id": 9, "text": "Display the forecast for the next 2 working days", "kind": "ui",
               "status": "implemented", "evidence": [{"path": PAGE, "line": quote}], "note": ""}
    return ledger.verify_evidence([verdict], workspace)[0]


def test_a_sub_line_jsx_quote_with_an_attribute_string_locates(tmp_path):
    assert _verify(_app(tmp_path), R9_QUOTE)["status"] == "implemented"


def test_strings_comments_and_invented_quotes_still_fail(tmp_path):
    workspace = _app(tmp_path)
    for quote in ('"Live market data was unavailable, so SIMULATED demo prices were stored."',
                  "Live market data was unavailable, so SIMULATED demo prices were stored",
                  "// per tracked company, next 2 working days",
                  "per tracked company, next 2 working days",
                  '<th>Target date</th><th>Model</th><th className="num">Predicted close</th>'):
        assert _verify(workspace, quote)["status"] == "unverified", quote


def test_a_verbatim_part_that_fails_on_its_role_is_named_as_inside_its_line(tmp_path):
    """A quote found verbatim inside a line is not 'absent': the note names the line."""
    workspace = _app(tmp_path)
    files = ledger._source_files(workspace)
    full = files[PAGE]
    source = ledger._load_source(full, {}, False)
    part = "<th>Target date</th><th>Method</th>"
    note, near = ledger._where_the_quote_leaves(ledger._normalise_quote(part), source, PAGE, part, {})
    assert near and "inside frontend/src/pages/WatchlistDetail.jsx line 8" in note
    invented = "<th>Target date</th><th>Nothing</th>"
    assert not ledger._where_the_quote_leaves(ledger._normalise_quote(invented), source, PAGE, invented, {})[1]


# Review of the fix: a string's text plus its closing punctuation, or an f-string or
# template body around one interpolation, is not code; a whole string literal quoted
# as one part of an abridged citation beside a code part still counts, as before.
SERVICE = "backend/service.py"
SERVICE_SOURCE = '''from fastapi import HTTPException


def check_booking(booking):
    if booking.end < booking.start:
        raise HTTPException(
            status_code=400,
            detail="Booking end date must be after the booking start date",
        )
    message = f"The booking for {booking.guest} runs until {booking.end}, which is after the start"
    return message
'''


def _service(tmp_path):
    path = tmp_path / SERVICE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SERVICE_SOURCE, encoding="utf-8")
    return str(tmp_path)


def _check(workspace, path, evidence):
    verdict = {"id": 1, "text": "Booking end date must be after start date", "kind": "validation",
               "status": "implemented", "evidence": evidence, "note": ""}
    return ledger.verify_evidence([verdict], workspace)[0]


def test_string_text_with_closing_punctuation_is_not_code(tmp_path):
    workspace = _app(tmp_path)
    _service(tmp_path)
    for path, quote in (
            (PAGE, 'Live market data was unavailable, so SIMULATED demo prices were stored.";'),
            (SERVICE, 'Booking end date must be after the booking start date",'),
            (SERVICE, "The booking for {booking.guest} runs until {booking.end}, which is after the start")):
        assert _check(workspace, path, [{"path": path, "line": quote}])["status"] == "unverified", quote


def test_a_whole_string_literal_part_beside_code_counts(tmp_path):
    workspace = _service(tmp_path)
    abridged = f'{SERVICE}: if booking.end < booking.start: ... "Booking end date must be after the booking start date"'
    assert _check(workspace, SERVICE, abridged)["status"] == "implemented"
    alone = f'{SERVICE}: "Booking end date must be after the booking start date"'
    verdict = _check(workspace, SERVICE, alone)
    assert verdict["status"] == "unverified" and verdict.get(ledger.EVIDENCE_UNLOCATED) != ledger.CITATION_ABSENT


def test_a_short_string_literal_alone_is_misquoted_code_not_absent_code(tmp_path):
    """Review 4: "Room already booked" (under 20 characters) was labelled absent."""
    path = tmp_path / "backend/routers/rooms.py"
    path.parent.mkdir(parents=True)
    path.write_text('def book(taken):\n    if taken:\n        raise HTTPException(409, "Room already booked")\n',
                    encoding="utf-8")
    verdict = _check(str(tmp_path), "backend/routers/rooms.py",
                     [{"path": "backend/routers/rooms.py", "line": '"Room already booked"'}])
    assert verdict["status"] == "unverified"
    assert verdict.get(ledger.EVIDENCE_UNLOCATED) == ledger.CITATION_MALFORMED


def test_a_one_word_literal_alone_is_not_taken_for_misquoted_code(tmp_path):
    """Review 5: "ok" or "admin" is in almost any file; as a near match it stopped Phase 3
    from implementing behaviour that does not exist."""
    path = tmp_path / "backend/routers/health.py"
    path.parent.mkdir(parents=True)
    path.write_text('def health():\n    return {"status": "ok"}\n', encoding="utf-8")
    verdict = _check(str(tmp_path), "backend/routers/health.py",
                     [{"path": "backend/routers/health.py", "line": '"ok"'}])
    assert verdict["status"] == "unverified"
    assert verdict.get(ledger.EVIDENCE_UNLOCATED) == ledger.CITATION_ABSENT
