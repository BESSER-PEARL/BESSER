"""Phase 3 must exercise action/method endpoints, not just create routes.

Live evidence: run iw82zzoc shipped ``registerArrival``, ``registerDeparture``
and ``cancel`` on Booking permanently broken - each compares a loaded enum
column to ``SomeEnum.LITERAL.value`` (a raw string), which is never true for
an enum member, so the handler refuses every call in every state. The
existing create-route probe (``constructibility.py``) never calls an action
endpoint at all, and the static stub scan (``action_inventory.py``) only
catches an empty/placeholder body - not a handler with real code that is
simply wrong. Nothing in the pipeline reported this until it was found by
hand. Run fcdh0s9k has the equivalent guards written correctly (no
``.value``) and must report nothing.

The rule under test (see ``_probe_actions`` in constructibility.py): call
each action once per distinct value of a REQUIRED, name-obviously-a-status
enum field on its own entity's create schema (never chaining two actions
against the same instance, so one action's side effects can never taint
another's evidence). A single 4xx is not a finding - refusing in the wrong
state is the entire point of a guarded action. Only "refused in every state
this probe could construct" is reported, and only as ``action unverified:``
(a hedge, not a verdict); an unhandled exception is reported unconditionally
as ``action call:``. Both are new prefixes ``_classify_issue`` does not
recognise, so they land at "warning" severity like ``endpoint_coherence`` -
deliberately, given this probe cannot rule out a state reachable only
through a prior action call, or a guard on a *related* entity's field (see
the honest-stub test below, modelled on Bill.registerPayment in iw82zzoc,
whose guard reads the *linked Booking's* commercialStatus, not any field of
Bill itself).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from besser.generators.llm.constructibility import (
    ACTION_PREFIX,
    ACTION_UNVERIFIED_PREFIX,
    collect_constructibility_issues,
)
from besser.generators.llm.orchestrator import _classify_issue

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

_ROUTER_HEADER = """\
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic_classes import *
from sql_alchemy import *
from database import get_db

router = APIRouter()

"""

_HANDLER = """\
@router.post("/room/{{room_id}}/methods/activate/", response_model=None, tags=["Room Methods"])
async def execute_room_activate(
    room_id: int,
    params: dict = Body(default=None, embed=True),
    database: Session = Depends(get_db)
):
    _room = database.query(Room).filter(Room.id == room_id).first()
    if _room is None:
        raise HTTPException(status_code=404, detail="Room not found")
{body}
"""


def _scaffold(tmp_path: Path) -> str:
    """A generator-fresh backend: Room, with a required status enum, and a
    Booking that needs a Room. Mirrors test_constructibility_probe.py."""
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DomainModel, Enumeration,
        EnumerationLiteral, IntegerType, Multiplicity, Property,
    )
    from besser.generators.backend import BackendGenerator

    status = Enumeration(name="RoomStatus", literals={
        EnumerationLiteral(name="AVAILABLE"), EnumerationLiteral(name="MAINTENANCE"),
    })
    room = Class(name="Room", attributes={
        Property(name="roomNumber", type=IntegerType),
        Property(name="status", type=status),
    })
    booking = Class(name="Booking", attributes=set())
    link = BinaryAssociation(name="room_booking", ends={
        Property(name="room", type=room, multiplicity=Multiplicity(1, 1)),
        Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*")),
    })
    model = DomainModel(name="Hotel", types={room, booking, status}, associations={link})
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=model, output_dir=str(backend)).generate()
    assert (backend / "main_api.py").is_file()
    return str(tmp_path)


def _add_action(tmp_path: Path, body: str) -> None:
    """Write ``room_methods.py`` with one action whose body is ``body``, and
    wire it into main_api.py exactly like an extra generated router."""
    backend = tmp_path / "web_app" / "backend"
    router_path = backend / "routers" / "room_methods.py"
    router_path.write_text(_ROUTER_HEADER + _HANDLER.format(body=body), encoding="utf-8")
    main_api = backend / "main_api.py"
    text = main_api.read_text(encoding="utf-8")
    text = text.replace(
        "from routers import room as room_router",
        "from routers import room as room_router\nfrom routers import room_methods as room_methods_router",
        1,
    )
    text = text.replace(
        "app.include_router(room_router.router)",
        "app.include_router(room_router.router)\napp.include_router(room_methods_router.router)",
        1,
    )
    main_api.write_text(text, encoding="utf-8")


def _action_issues(workspace: str) -> list[str]:
    return [i for i in collect_constructibility_issues(workspace)
            if i.startswith((ACTION_PREFIX, ACTION_UNVERIFIED_PREFIX))]


# --------------------------------------------------------------- the live bug


def test_a_value_vs_enum_member_guard_is_reported_in_every_state(tmp_path):
    """The exact iw82zzoc defect: comparing a loaded enum column to
    ``Enum.LITERAL.value`` (a raw string) is never true, so the guard
    refuses every call regardless of the room's actual status."""
    workspace = _scaffold(tmp_path)
    _add_action(tmp_path, (
        '    if _room.status != RoomStatus.AVAILABLE.value:\n'
        '        raise HTTPException(status_code=400, detail="not available")\n'
        '    return {"success": True}\n'
    ))
    [issue] = _action_issues(workspace)
    assert issue.startswith(f"{ACTION_UNVERIFIED_PREFIX} web_app/backend: "
                            "POST /room/{room_id}/methods/activate/ -")
    assert "activate on Room refused every state" in issue
    assert "status=AVAILABLE" in issue and "status=MAINTENANCE" in issue
    assert "Not proof the action is broken" in issue
    assert "Fix site: execute_room_activate in web_app/backend/routers/room_methods.py" in issue
    # Unrecognised prefix -> the conservative default. This check cannot rule
    # out a state reachable only through a prior action call or a guard on a
    # related entity, so it must never gate the auto-fix/rollback loop itself.
    assert _classify_issue(issue).severity == "warning"


def test_a_correct_guard_that_refuses_sometimes_is_not_a_finding(tmp_path):
    """One refusal is not evidence: a state-guarded action is SUPPOSED to
    refuse in the wrong state. Comparing to the enum member itself (no
    ``.value``) is the correct form, and must be silent."""
    workspace = _scaffold(tmp_path)
    _add_action(tmp_path, (
        '    if _room.status != RoomStatus.AVAILABLE:\n'
        '        raise HTTPException(status_code=400, detail="not available")\n'
        '    return {"success": True}\n'
    ))
    assert _action_issues(workspace) == []


def test_an_unhandled_exception_is_reported_unconditionally(tmp_path):
    """A crash is decisive on its own - unlike a 4xx, no reachable state
    excuses it, so this is reported even though every observed state is a
    Python exception, not merely "every state refuses".

    It is a BLOCKER. The old ``warning`` was not a decision: ``action call:``
    appeared in no prefix list in ``_classify_issue`` and fell through to the
    default, so a 500 the probe had literally watched an action handler raise
    could never reach the blocker-only Phase 3 fix loop - while its exact
    create-side twin, ``create contract:``, always could.
    """
    workspace = _scaffold(tmp_path)
    _add_action(tmp_path, '    raise RuntimeError("boom")\n')
    [issue] = _action_issues(workspace)
    assert issue.startswith(f"{ACTION_PREFIX} web_app/backend: "
                            "POST /room/{room_id}/methods/activate/ -")
    assert "observed a server/persistence failure calling activate on a Room" in issue
    assert "RuntimeError" in issue and "boom" in issue
    assert _classify_issue(issue).severity == "blocker"


def test_an_honest_not_implemented_stub_is_left_to_the_static_scan(tmp_path):
    """The deterministic scaffold's own "no body in the model" marker
    (router_methods.py.j2: ``raise HTTPException(status_code=501, ...)``).
    action_inventory.py's static AST scan already reports this precisely as
    ``action contract: ... UNIMPLEMENTED: HTTP 501``; reporting it again here
    as a "server failure" would be redundant and mislabel an honest stub as
    a runtime defect."""
    workspace = _scaffold(tmp_path)
    _add_action(tmp_path, (
        '    raise HTTPException(\n'
        '        status_code=501,\n'
        '        detail="Method \'activate\' of Room is modeled but has no implementation",\n'
        '    )\n'
    ))
    assert _action_issues(workspace) == []


def test_no_status_like_field_means_no_evidence_and_no_finding(tmp_path):
    """Booking has no enum-typed field at all, so this probe has no lever to
    construct a second state. An action that always refuses is then
    indistinguishable from one whose precondition is only reachable through
    a *different*, prior action call (the ordering case flagged for this
    task: registerDeparture legitimately needs registerArrival first).
    Silence, not a false "broken", is the only sound outcome here."""
    workspace = _scaffold(tmp_path)
    backend = tmp_path / "web_app" / "backend"
    router_path = backend / "routers" / "booking_methods.py"
    router_path.write_text(
        _ROUTER_HEADER.replace("Room", "Booking") + (
            '@router.post("/booking/{booking_id}/methods/close/", response_model=None, '
            'tags=["Booking Methods"])\n'
            'async def execute_booking_close(\n'
            '    booking_id: int,\n'
            '    params: dict = Body(default=None, embed=True),\n'
            '    database: Session = Depends(get_db)\n'
            '):\n'
            '    _booking = database.query(Booking).filter(Booking.id == booking_id).first()\n'
            '    if _booking is None:\n'
            '        raise HTTPException(status_code=404, detail="Booking not found")\n'
            '    raise HTTPException(status_code=400, detail="never closable")\n'
        ),
        encoding="utf-8",
    )
    main_api = backend / "main_api.py"
    text = main_api.read_text(encoding="utf-8")
    text = text.replace(
        "from routers import booking as booking_router",
        "from routers import booking as booking_router\n"
        "from routers import booking_methods as booking_methods_router",
        1,
    )
    text = text.replace(
        "app.include_router(booking_router.router)",
        "app.include_router(booking_router.router)\n"
        "app.include_router(booking_methods_router.router)",
        1,
    )
    main_api.write_text(text, encoding="utf-8")
    assert _action_issues(workspace) == []


def test_a_recollection_reports_the_same_thing(tmp_path):
    """The fix loop compares blocker/warning counts across rounds."""
    workspace = _scaffold(tmp_path)
    _add_action(tmp_path, (
        '    if _room.status != RoomStatus.AVAILABLE.value:\n'
        '        raise HTTPException(status_code=400, detail="not available")\n'
        '    return {"success": True}\n'
    ))
    assert _action_issues(workspace) == _action_issues(workspace)
