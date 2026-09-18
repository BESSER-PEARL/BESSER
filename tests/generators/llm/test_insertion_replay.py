"""Insertion replay regression from EC2 run f6770633, turns 31--105."""
import json

import pytest

from besser.generators.llm.tool_executor import ToolExecutor


# The exact search/replacement in the run retained the entire search anchor.
BOOKING_OLD = '''    database.commit()
    database.refresh(db_booking)

    guest_ids = database.query(guests.c.guest).filter(guests.c.guests == db_booking.id).all()
    bookingRooms_ids = database.query(BookingRoom.id).filter(BookingRoom.booking_id == db_booking.id).all()
    response_data = {
        "booking": db_booking,
        "guest_ids": [x[0] for x in guest_ids],
        "bookingRooms_ids": [x[0] for x in bookingRooms_ids]    }
    return response_data'''
BOOKING_NEW = (
    "    # Update the booking's total price\n"
    "    db_booking.totalPrice = db_booking.total_price\n" + BOOKING_OLD
)


def edit(executor, old, new, **kwargs):
    return json.loads(executor.execute("modify_file", {
        "path": "app.py", "old_text": old, "new_text": new, **kwargs,
    }))


@pytest.mark.parametrize("restart", [False, True])
def test_actual_run_replayed_75_times_writes_once(tmp_path, restart):
    path = tmp_path / "app.py"
    path.write_text("def booking():\n" + BOOKING_OLD + "\n", encoding="utf-8")
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    assert edit(executor, BOOKING_OLD, BOOKING_NEW)["status"] == "modified"
    expected = path.read_bytes()
    for _ in range(74):
        if restart:
            executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
        result = edit(executor, BOOKING_OLD, BOOKING_NEW)
        assert "error" in result, result
        assert path.read_bytes() == expected
    assert path.read_text().count("db_booking.totalPrice =") == 1


@pytest.mark.parametrize("new", ["y = 2\nx = 1\n", "x = 1\ny = 2\n", "y = 2\nx = 1\nz = 3\n"])
@pytest.mark.parametrize("numbered,indent,crlf", [(False, "", False), (False, "    ", False), (True, "    ", False), (False, "", True)])
def test_prefix_suffix_middle_and_normalized_replays(tmp_path, new, numbered, indent, crlf):
    old = "x = 1\n"
    content = ("def f():\n" if indent else "") + indent + old
    (tmp_path / "app.py").write_bytes(content.replace("\n", "\r\n" if crlf else "\n").encode())
    if numbered:
        old = "1| " + old
        new = "".join(f"{i}| {line}" for i, line in enumerate(new.splitlines(True), 1))
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    assert edit(executor, old, new)["status"] == "modified"
    expected = (tmp_path / "app.py").read_bytes()
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    assert "error" in edit(executor, old, new)
    assert (tmp_path / "app.py").read_bytes() == expected


def test_replace_all_skips_completed_sites_and_edits_pending_sites(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("def a():\n    y = 2\n    x = 1\n\ndef b():\n    x = 1\n")
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    result = edit(executor, "    x = 1\n", "    y = 2\n    x = 1\n", replace_all=True)
    assert result["status"] == "modified"
    assert result["replacements"] == 1
    assert path.read_text().count("y = 2") == 2
    expected = path.read_bytes()
    assert "error" in edit(executor, "    x = 1\n", "    y = 2\n    x = 1\n", replace_all=True)
    assert path.read_bytes() == expected


def test_new_text_elsewhere_does_not_block_a_real_edit(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("first = 2\nsecond = 1\n")
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    assert edit(executor, "1", "2")["status"] == "modified"
    assert path.read_text() == "first = 2\nsecond = 2\n"


def test_different_edit_and_reverted_edit_remain_allowed(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("x = 1\n")
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    assert edit(executor, "x = 1\n", "y = 2\nx = 1\n")["status"] == "modified"
    assert "error" in edit(executor, "x = 1\n", "y = 2\nx = 1\n")
    assert edit(executor, "y = 2\nx = 1\n", "x = 1\n")["status"] == "modified"
    assert edit(executor, "x = 1\n", "y = 2\nx = 1\n")["status"] == "modified"


def test_ambiguous_flexible_match_does_not_pick_the_first_function(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("def a():\n    x = 1\n\ndef b():\n    x = 1\n")
    before = path.read_bytes()
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    result = edit(executor, "x = 1\n", "x = 2\n")
    assert "ambiguous" in result.get("error", "").lower()
    assert path.read_bytes() == before


def test_crlf_quoted_edit_and_replay(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("x = 1\ny = 2\n")
    executor = ToolExecutor(str(tmp_path), per_write_diagnostics=False)
    old = "x = 1\r\ny = 2\r\n"
    new = "z = 3\r\n" + old
    assert edit(executor, old, new)["status"] == "modified"
    before = path.read_bytes()
    assert "error" in edit(executor, old, new)
    assert path.read_bytes() == before


def test_file_operations_are_serialized_across_executors_and_path_aliases(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from time import sleep

    path = tmp_path / "app.py"
    path.write_text("x = 1\n")
    executors = [ToolExecutor(str(tmp_path), per_write_diagnostics=False) for _ in range(2)]
    state = {"active": 0, "peak": 0}
    guard = Lock()

    def handler(executor, args):
        with guard:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        sleep(0.03)
        with guard:
            state["active"] -= 1
        return {"status": "modified"}

    for executor in executors:
        executor._handlers = {**executor._handlers, "modify_file": handler, "write_file": handler}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(executors[i % 2].execute_typed,
                              "modify_file" if i % 2 else "write_file",
                              {"path": "./app.py" if i % 2 else "app.py"}) for i in range(8)]
        assert all(f.result().status == "ok" for f in futures)
    assert state["peak"] == 1
