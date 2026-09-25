"""The gap analyser must not point Phase 2 at files that do not exist.

Live local repro (Nebius ``Qwen/Qwen3-30B-A3B-Instruct-2507``,
``generate_web_app`` + GUI model, 63-file scaffold). The planner emitted 16
tasks. Fourteen named ``backend/...`` paths that exist once the ``web_app/``
prefix is restored; the last two named components that exist nowhere::

    15 | In the frontend/src/components/BookingForm.tsx file, add form
         controls for arrival and departure dates, ...
    16 | In the frontend/src/components/BookingDetails.tsx file, add a
         section to display computed amount owed, ...

The real Booking screen is ``web_app/frontend/src/pages/Booking.tsx``. The
run then went::

    read_file  web_app/backend/sql_alchemy.py                      ok
    read_file  web_app/backend/routers/booking_methods.py          ok
    read_file  web_app/frontend/src/components/BookingForm.tsx     error
    read_file  web_app/frontend/src/components/BookingDetails.tsx  error
    search_in_files BookingForm / BookingDetails / Booking.tsx / Booking
    search_in_files BookingForm / BookingDetails / Booking.tsx / Booking
    search_in_files BookingForm / BookingDetails / Booking.tsx / Booking

18 turns, 19 tool calls, ZERO writes. The model resolves the missing
``web_app/`` prefix on its own (the backend reads all succeeded), so the
prefix is not what breaks a run — a path that resolves to nothing is.

These tests pin the deterministic pass that runs before Phase 2 starts:
rewrite a path whose suffix matches exactly one real file, and flag a path
that matches none so the model stops hunting for it.
"""
from __future__ import annotations

from besser.spec_driven_agent.planning.gap_analyzer import _resolve_task_paths


# The scaffold from the run above, trimmed to the paths under test.
WORKSPACE = [
    "web_app/backend/sql_alchemy.py",
    "web_app/backend/pydantic_classes.py",
    "web_app/backend/main_api.py",
    "web_app/backend/routers/booking.py",
    "web_app/backend/routers/booking_methods.py",
    "web_app/backend/routers/bill_methods.py",
    "web_app/frontend/src/App.tsx",
    "web_app/frontend/src/pages/Booking.tsx",
    "web_app/frontend/src/pages/Bill.tsx",
    "web_app/frontend/src/components/MethodButton.tsx",
]


def test_suffix_unique_path_is_rewritten_to_the_real_one():
    """``backend/sql_alchemy.py`` names exactly one real file — repair it."""
    tasks = [
        "Add the method 'registerArrival' to the Booking class in the "
        "backend/sql_alchemy.py file, implementing it to update physicalStatus."
    ]
    out = _resolve_task_paths(tasks, WORKSPACE)
    assert "web_app/backend/sql_alchemy.py" in out[0]
    assert "the backend/sql_alchemy.py file" not in out[0]


def test_exact_path_is_left_untouched():
    tasks = ["Implement the endpoint in web_app/backend/routers/bill_methods.py."]
    assert _resolve_task_paths(tasks, WORKSPACE) == tasks


def test_task_with_no_file_path_is_left_untouched():
    tasks = ["Ensure the Booking screen shows both status dimensions."]
    assert _resolve_task_paths(tasks, WORKSPACE) == tasks


def test_phantom_path_is_flagged_and_names_the_real_screen():
    """The failure that cost the run: a path matching nothing.

    The task is kept — "create this file" is legitimate work — but it must
    carry the fact that the file is absent plus the nearest real candidate,
    so Phase 2 writes instead of searching for it three times over.
    """
    tasks = [
        "In the frontend/src/components/BookingForm.tsx file, add form "
        "controls for arrival and departure dates."
    ]
    out = _resolve_task_paths(tasks, WORKSPACE)
    assert len(out) == 1
    assert "does not exist" in out[0].lower()
    assert "web_app/frontend/src/pages/Booking.tsx" in out[0]


def test_ambiguous_suffix_is_not_guessed():
    """Two real files share the suffix — rewriting would be a coin flip."""
    workspace = [
        "web_app/backend/routers/booking.py",
        "legacy/backend/routers/booking.py",
    ]
    tasks = ["Update backend/routers/booking.py to add the cancel endpoint."]
    out = _resolve_task_paths(tasks, workspace)
    assert out[0].startswith("Update backend/routers/booking.py")
    assert "web_app/backend/routers/booking.py" not in out[0]


def test_empty_workspace_is_a_no_op():
    """No inventory (non-filesystem run) — never rewrite on no evidence."""
    tasks = ["Add the method to backend/sql_alchemy.py."]
    assert _resolve_task_paths(tasks, []) == tasks


def test_live_task_list_leaves_no_unresolved_path_unflagged():
    """End-to-end over the real 16-task list from the failing run."""
    tasks = [
        "Add the method 'registerArrival' to the Booking class in the "
        "backend/sql_alchemy.py file.",
        "In the backend/routers/booking_methods.py file, implement the "
        "'booking_registerArrival' endpoint.",
        "In the backend/routers/bill_methods.py file, implement the "
        "'bill_registerPayment' endpoint.",
        "In the frontend/src/components/BookingForm.tsx file, add form controls.",
        "In the frontend/src/components/BookingDetails.tsx file, add a section.",
        "Ensure the Booking screen includes a status indicator.",
    ]
    out = _resolve_task_paths(tasks, WORKSPACE)
    assert len(out) == len(tasks)
    # Every backend path repaired to a real one.
    assert "web_app/backend/sql_alchemy.py" in out[0]
    assert "web_app/backend/routers/booking_methods.py" in out[1]
    assert "web_app/backend/routers/bill_methods.py" in out[2]
    # Both phantoms flagged, and neither silently dropped.
    for phantom in (out[3], out[4]):
        assert "does not exist" in phantom.lower()
        assert "web_app/frontend/src/pages/Booking.tsx" in phantom
    # The path-less task is untouched.
    assert out[5] == tasks[5]
