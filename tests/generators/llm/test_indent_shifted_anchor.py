"""An old_text that matches INSIDE a line's indentation must be refused, and a
tool call may never turn a parseable file into an unparseable one.

Live run 57160293, turn 19 (2026-09-18, Nebius Qwen3-30B-A3B-Instruct). The
scaffold's stub comment sits at indent 8; the model quoted it at indent 4
with the other five lines exact. The plain ``old_text in content`` check
matched four characters into the line, ``str.replace`` kept the file's first
four spaces, and the replacement landed as comment-at-8 plus a second
``try:`` at the enclosing level: ``expected 'except' or 'finally' block``
at line 181, computed by the executor, returned as ``diagnostics`` - and
kept on disk. Aider's line-tuple ``perfect_replace`` refuses this quote.

The fixture is the pristine scaffold file; the anchor and replacement are the
run's own bytes (the replacement reconstructed from what landed on disk).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from besser.generators.llm.tool_executor import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "run_57160293"
ROUTER = "web_app/backend/routers/booking_methods.py"

T19_OLD = (
    '    # Booking.computeAmountOwed: no body in the model - be honest: 501, never a fake "executed" success.\n'
    '        sys.stdout = sys.__stdout__\n'
    '        raise HTTPException(\n'
    '            status_code=501,\n'
    '            detail="Method \'computeAmountOwed\' of Booking is modeled but has no implementation",\n'
    '        )'
)
T19_NEW = (
    '    # Booking.computeAmountOwed: calculate total price from room agreed prices, stay duration, and extra charges\n'
    '    try:\n'
    '        # Get the booking from the database\n'
    '        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()\n'
    '        if not db_booking:\n'
    '            raise HTTPException(status_code=404, detail="Booking not found")\n'
    '\n'
    '        # Calculate the number of nights\n'
    '        nights = (db_booking.departureDate - db_booking.arrivalDate).days\n'
    '        if nights < 0:\n'
    '            raise HTTPException(status_code=400, detail="Departure date cannot be before arrival date")\n'
    '\n'
    '        # Calculate total from agreed prices of rooms\n'
    '        total = 0.0\n'
    '        for bookingroom in db_booking.bookingRooms:\n'
    '            total += bookingroom.agreedPrice * nights\n'
    '\n'
    '        # Add extra charges\n'
    '        total += db_booking.extraCharges\n'
    '\n'
    '        # Return the amount owed\n'
    '        return {"amountOwed": total}'
)
DETAIL_LINE = (
    '            detail="Method \'computeAmountOwed\' of Booking is modeled but has no implementation",'
)


@pytest.fixture
def ex(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    e = ToolExecutor(workspace=str(tmp_path))
    e.execute("read_file", {"path": ROUTER})
    return e


def _file(ex):
    return (Path(ex.workspace) / ROUTER).read_text(encoding="utf-8")


def _modify(ex, old, new):
    return json.loads(ex.execute("modify_file", {"path": ROUTER, "old_text": old, "new_text": new}))


def test_the_t19_quote_is_refused_and_the_file_is_untouched(ex):
    before = _file(ex)
    assert T19_OLD in before, "the substring match is exactly the trap"
    res = _modify(ex, T19_OLD, T19_NEW)
    assert "error" in res, res
    assert _file(ex) == before


def test_the_refusal_names_the_indentation_and_shows_the_real_line(ex):
    res = _modify(ex, T19_OLD, T19_NEW)
    assert "indent" in res["error"].lower()
    assert "did_you_mean" in res
    assert "        # Booking.computeAmountOwed: no body" in res["did_you_mean"]


def test_the_same_stub_quoted_at_the_files_indent_lands(ex):
    old = "    " + T19_OLD          # first line now at 8, like the file
    new = (
        "        # Booking.computeAmountOwed: calculate total price\n"
        '        raise HTTPException(status_code=501, detail="not implemented yet")'
    )
    res = _modify(ex, old, new)
    assert res.get("status") == "modified", res
    assert "matched_by" not in res, "an exact, line-anchored hit"
    compile(_file(ex), ROUTER, "exec")


def test_a_partial_line_quote_after_code_still_works(ex):
    res = _modify(ex, "'computeAmountOwed' of Booking is modeled but has no implementation",
                  "'computeAmountOwed' of Booking is modeled but not implemented yet")
    assert res.get("status") == "modified", res


def test_a_valid_file_never_becomes_unparseable_through_modify_file(ex):
    """The t19 replacement itself, anchored correctly: it opens a second
    ``try:`` at the enclosing level and closes nothing. Live it was written
    and its syntax error returned as a diagnostic; now it is refused."""
    before = _file(ex)
    res = _modify(ex, "    " + T19_OLD, T19_NEW)
    assert "error" in res, res
    assert "unparseable" in res["error"]
    assert "line" in res["error"]
    assert "would_write" in res
    assert res["rejection_kind"] == "syntax_error"
    assert res["syntax_line"] > 0
    assert res["would_write"].startswith("PROPOSED ONLY - NOT APPLIED")
    assert res["current_source"].startswith("CURRENT ON-DISK CONTENT")
    assert "# Booking.computeAmountOwed: no body" in res["current_source"]
    assert "calculate total price" not in res["current_source"]
    assert _file(ex) == before

    # Quoting that rejected draft must remain a miss, not a completed edit.
    retry = _modify(ex, T19_NEW, T19_NEW + "\n    finally:\n        pass")
    assert "error" in retry
    assert "not a proposed edit that was refused" in retry["error"]
    assert "will not help" not in retry["error"]
    assert _file(ex) == before


def test_a_valid_file_never_becomes_unparseable_through_write_file(ex):
    before = _file(ex)
    for _ in range(2):                       # unlock write_file on a generated file
        _modify(ex, "nomatch-1", "x")
    res = json.loads(ex.execute("write_file", {"path": ROUTER, "content": "def broken(:\n    pass\n"}))
    assert "error" in res, res
    assert res["rejection_kind"] == "syntax_error"
    assert res["would_write"].startswith("PROPOSED ONLY - NOT APPLIED")
    assert "CURRENT ON-DISK CONTENT" in res["current_source"]
    assert _file(ex) == before


def test_a_file_that_was_already_broken_is_not_blamed_on_the_edit(tmp_path):
    (tmp_path / "b.py").write_text("def broken(:\n    pass\nx = 1\n", encoding="utf-8")
    e = ToolExecutor(workspace=str(tmp_path))
    e.execute("read_file", {"path": "b.py"})
    res = json.loads(e.execute("modify_file", {"path": "b.py", "old_text": "x = 1\n", "new_text": "x = 2\n"}))
    assert res.get("status") == "modified", res
