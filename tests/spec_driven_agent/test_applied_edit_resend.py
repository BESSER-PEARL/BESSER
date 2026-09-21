"""A re-send of an edit the ladder has recorded as applied must be recognized,
even when the ladder re-indented new_text on write. A no-op is not that receipt.

Live run 4efe04ff (2026-09-18, Nebius Qwen3-30B-A3B-Instruct), turns 62-70
on routers/booking.py. The model quoted the bulk-create validation block at
indent 4; the file holds it at indent 12, so the edit landed through the
uniform-indent tier with new_text shifted by +8. It then re-sent the same
call (turn 63) and got the generic "old_text not found ... Make sure old_text
matches exactly", and re-sent new_text as a no-op (65, 68, 70) and got
"identical" without "already in the file": both already-applied checks were
byte-exact and could not see their own re-indented output. The five no-op
re-sends on booking_methods.py in the same run, where the edits had landed
exactly, were all told "this change is done" and the model moved on.

The fixture is the pristine scaffold file recovered from the run's turn-6
read_file result (22148 bytes, the size list_files reported at turn 1); the
anchor and replacement are the run's own bytes from turn 62.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from besser.spec_driven_agent.tool_executor import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "run_4efe04ff"
ROUTER = "web_app/backend/routers/booking.py"

T62_OLD = (
    '    # Basic validation for each item\n'
    '    if not item_data.employee:\n'
    '        raise ValueError("Employee ID is required")\n'
    '    if not item_data.contact:\n'
    '        raise ValueError("Person ID is required")\n'
    '\n'
    '    db_booking = Booking(\n'
    '        departureDate=item_data.departureDate,                commercialStatus=item_data.commercialStatus.value,                bookingNumber=item_data.bookingNumber,                physicalStatus=item_data.physicalStatus.value,                arrivalDate=item_data.arrivalDate,                totalPrice=item_data.totalPrice,                employee_id=item_data.employee,                contact_id=item_data.contact            )\n'
    '    database.add(db_booking)\n'
    '    database.flush()  # Get ID without committing\n'
    '    created_items.append(db_booking.id)'
)
T62_NEW = (
    '    # Basic validation for each item\n'
    '    if not item_data.employee:\n'
    '        raise ValueError("Employee ID is required")\n'
    '    if not item_data.contact:\n'
    '        raise ValueError("Person ID is required")\n'
    '    \n'
    '    # Validate guest capacity\n'
    '    if item_data.reservedRooms:\n'
    '        total_capacity = 0\n'
    '        for room in item_data.reservedRooms:\n'
    "            # This assumes we have access to the room's capacity\n"
    '            # In a real implementation, we would need to query the database\n'
    "            # For now, we'll assume the capacity is passed in the ReservedRoomCreate\n"
    '            # This is a limitation of the current design\n'
    '            # In a real app, we would validate this by querying the database\n'
    '            total_capacity += room.capacity\n'
    '            \n'
    '        if len(item_data.guest) > total_capacity:\n'
    '            raise ValueError(f"Total number of guests ({len(item_data.guest)}) exceeds the combined capacity of rooms ({total_capacity})")\n'
    '    \n'
    '    db_booking = Booking(\n'
    '        departureDate=item_data.departureDate,                commercialStatus=item_data.commercialStatus.value,                bookingNumber=item_data.bookingNumber,                physicalStatus=item_data.physicalStatus.value,                arrivalDate=item_data.arrivalDate,                totalPrice=item_data.totalPrice,                employee_id=item_data.employee,                contact_id=item_data.contact            )\n'
    '    database.add(db_booking)\n'
    '    database.flush()  # Get ID without committing\n'
    '    created_items.append(db_booking.id)'
)
# Where the block sits in the file, at the file's indent of 12.
LANDED_AT = 228


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


def test_the_t62_quote_lands_re_indented_by_the_ladder(ex):
    """The premise: the write succeeds and new_text is NOT in the file
    byte-for-byte afterwards, only shifted to the file's indent."""
    assert T62_OLD not in _file(ex), "quoted at indent 4, held at 12"
    res = _modify(ex, T62_OLD, T62_NEW)
    assert res.get("status") == "modified", res
    assert res.get("matched_by") == "flexible"
    after = _file(ex)
    assert T62_NEW not in after
    landed = after.split("\n")[LANDED_AT - 1]
    assert landed == "        " + T62_NEW.split("\n")[0]
    compile(after, ROUTER, "exec")


def test_resending_the_applied_edit_is_named_done_not_not_found(ex):
    """Turn 63: the same call again. Live it was 'old_text not found ...
    Make sure old_text matches exactly'; the anchor is gone BECAUSE the edit
    was applied, and that is what the reply must say."""
    _modify(ex, T62_OLD, T62_NEW)
    before = _file(ex)
    res = _modify(ex, T62_OLD, T62_NEW)
    assert "error" in res, res
    assert f"already in the file at line {LANDED_AT}" in res["error"], res["error"]
    assert "applied" in res["error"]
    assert "matches exactly" not in res["error"], "the generic advice was what kept it looping"
    assert "did_you_mean" in res, "the current lines stay attached for a follow-up edit"
    assert _file(ex) == before


def test_a_noop_resend_identifies_the_text_without_claiming_completion(ex):
    """Turn 65: new_text sent as both old_text and new_text. Live the reply
    had no 'already in the file', unlike the same no-op on booking_methods.py
    where the edit had landed byte-exact, and the model sent it again."""
    _modify(ex, T62_OLD, T62_NEW)
    res = _modify(ex, T62_NEW, T62_NEW)
    assert "identical" in res["error"]
    assert f"already in the file at line {LANDED_AT}" in res["error"], res["error"]
    assert "not evidence of an implemented change" in res["error"]
    assert "change is done" not in res["error"]
    assert res.get("status") != "already_applied"


def test_a_miss_on_the_pristine_file_is_not_called_applied(ex):
    """Same anchor with its first line at yet another indent, before any
    edit: new_text is nowhere in the file, so this is an ordinary miss with
    the closest real lines, never a false 'done'."""
    before = _file(ex)
    res = _modify(ex, "    " + T62_OLD, T62_NEW)
    assert "error" in res, res
    assert "applied" not in res["error"]
    assert "not found" in res["error"]
    # A quote whose anchors bracket one region answers with the located range
    # (numbered) in place of the unnumbered did_you_mean window.
    assert res["located_range"]["lines"]
    assert _file(ex) == before
