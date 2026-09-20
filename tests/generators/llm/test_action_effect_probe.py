"""The harness must verify the delivered app, not ask the model to.

Measured across ~150 Qwen runs: ``test_api`` was called 0.06 times per run
(gpt-5.6: 13.1), 14-21 of ~20 checklist items were left open, and one run in
135 reported ``completed`` - on a median 66 turns of 120, so not a budget
limit. Rewriting the checklist to name ``test_api`` literally produced 0.00
calls across 24 runs. Persuasion has been tried three times. So the probe
drives the workflow itself.

Two things are new here.

**A status code is not a result.** A live gpt-5.6-terra run answered its Renew
button ``200 {"success": false, "message": "a dueDate is required"}`` and the
acceptance oracle scored it 10/10, because nothing read past the status line.
The probe now snapshots the entity - and the size of every collection - either
side of the call: an action that declares a refusal and changes nothing did
not happen.

**Refusing is often right.** Cancelling a checked-out booking SHOULD fail, and
this project has produced eleven oracle defects that failed a correct app
against one that certified a broken one. So the hard finding is entity-scoped
and needs three things at once: the instance was created one request earlier
and is still in its initial state, NO modelled action moved it (the aggregate
has no first transition at all), and the model itself says the action takes no
parameters - which only became readable today, when ``_method_entry`` stopped
omitting ``parameters`` for a zero-argument method (a886947f). Anything short
of that is ``action unverified:``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from besser.generators.llm.constructibility import (
    ACTION_PREFIX,
    ACTION_UNVERIFIED_PREFIX,
    UNVERIFIED_PREFIX,
    collect_constructibility_report,
)
from besser.generators.llm.orchestrator import _classify_issue

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _model():
    """Room.activate() - a zero-parameter modelled action - plus a Bill the
    app can create on the side, so "changed nothing" can tell a no-op from an
    action whose effect lands on another table."""
    from besser.BUML.metamodel.structural import (
        BooleanType, Class, DomainModel, IntegerType, Method, Property, StringType,
    )

    room = Class(name="Room", attributes={
        Property(name="roomNumber", type=IntegerType),
        Property(name="label", type=StringType),
    })
    room.methods = {Method(name="activate", type=BooleanType)}
    bill = Class(name="Bill", attributes={Property(name="amount", type=IntegerType)})
    return DomainModel(name="Hotel", types={room, bill})


_ROUTER = """\
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic_classes import *
from sql_alchemy import *
from database import get_db

router = APIRouter()


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


def _flag_model():
    """Room with a required boolean lifecycle flag - the Timesheet.approved
    shape (gpt-5.6-terra-1s14uohe)."""
    from besser.BUML.metamodel.structural import (
        BooleanType, Class, DomainModel, IntegerType, Method, Property,
    )

    room = Class(name="Room", attributes={
        Property(name="roomNumber", type=IntegerType),
        Property(name="activated", type=BooleanType),
    })
    room.methods = {Method(name="activate", type=BooleanType)}
    return DomainModel(name="Hotel", types={room})


def _state_model():
    """Room whose lifecycle enum lists its END state first - the
    SessionStatus shape (gpt-5.6-terra-5d9otfvo), where CANCELLED sorts
    before SCHEDULED."""
    from besser.BUML.metamodel.structural import (
        BooleanType, Class, DomainModel, Enumeration, EnumerationLiteral,
        IntegerType, Method, Property,
    )

    state = Enumeration(name="RoomState", literals={
        EnumerationLiteral(name="CANCELLED"), EnumerationLiteral(name="SCHEDULED"),
    })
    room = Class(name="Room", attributes={
        Property(name="roomNumber", type=IntegerType),
        Property(name="status", type=state),
    })
    room.methods = {Method(name="activate", type=BooleanType)}
    return DomainModel(name="Hotel", types={room, state})


def _scaffold(tmp_path: Path, body: str, model=None) -> str:
    """A generator-fresh backend whose one action handler has ``body``."""
    from besser.generators.backend import BackendGenerator

    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=model or _model(), output_dir=str(backend)).generate()
    assert (backend / "main_api.py").is_file()
    (backend / "routers" / "room_methods.py").write_text(
        _ROUTER.format(body=body), encoding="utf-8")
    return str(tmp_path)


def _probe(workspace: str, *, model=True) -> dict:
    return collect_constructibility_report(workspace, _model() if model is True else model or None)


def _action_issues(report: dict) -> list[str]:
    return [i for i in report["issues"]
            if i.startswith((ACTION_PREFIX, ACTION_UNVERIFIED_PREFIX))]


# The exact body of the live terra run: a 200 that says it failed.
_FAKE_SUCCESS = (
    '    return {"success": False, "message": "a dueDate is required"}\n'
)
_REAL_TRANSITION = (
    '    _room.label = "activated"\n'
    '    database.commit()\n'
    '    return {"status": "executed", "result": "True"}\n'
)


# --------------------------------------------------------------- the live bug


def test_a_200_that_declares_failure_and_changes_nothing_is_a_blocker(tmp_path):
    workspace = _scaffold(tmp_path, _FAKE_SUCCESS)
    [issue] = _action_issues(_probe(workspace))

    assert issue.startswith(f"{ACTION_PREFIX} web_app/backend: "
                            "POST /room/{room_id}/methods/activate/ -")
    assert "answered HTTP 200 but declared failure (success=false)" in issue
    assert "reads back identical after the call and no other record appeared" in issue
    assert "a dueDate is required" in issue
    assert "has no first transition" in issue
    # The model's own signature is what makes "it wanted an input" decisive.
    assert "declares activate() with no parameters" in issue
    assert "Do not satisfy this by deleting a business rule" in issue
    assert _classify_issue(issue).severity == "blocker"
    assert "Fix site: execute_room_activate in web_app/backend/routers/room_methods.py" in issue


def test_the_same_refusal_with_a_4xx_is_not_a_finding(tmp_path):
    """The defect is the lie, not the refusal. An app that says 400 has
    reported its refusal honestly and the create/action probes both let it
    stand - this is the path a legitimate OCL capacity rule takes."""
    workspace = _scaffold(tmp_path, (
        '    raise HTTPException(status_code=400, detail="a dueDate is required")\n'
    ))
    assert _action_issues(_probe(workspace)) == []


def test_an_action_that_moves_the_entity_is_silent_and_recorded_effective(tmp_path):
    workspace = _scaffold(tmp_path, _REAL_TRANSITION)
    report = _probe(workspace)

    assert _action_issues(report) == []
    [call] = [c for c in report["backends"][0]["action_calls"] if c["action"] == "activate"]
    assert call["verdict"] == "effective"
    assert call["changed"] is True


def test_an_effect_on_another_table_still_counts_as_having_acted(tmp_path):
    """``produceBill`` creates a Bill and leaves the Booking untouched. Reading
    only the acted-on row would score that correct action as a no-op, so the
    probe re-counts every collection too."""
    workspace = _scaffold(tmp_path, (
        '    database.add(Bill(amount=1))\n'
        '    database.commit()\n'
        '    return {"success": False, "message": "a dueDate is required"}\n'
    ))
    report = _probe(workspace)

    assert _action_issues(report) == []
    [call] = [c for c in report["backends"][0]["action_calls"] if c["action"] == "activate"]
    assert call["changed"] is True


# ------------------------------------------------- the conservative half

def test_without_the_models_parameter_list_the_finding_is_only_unverified(tmp_path):
    """``_method_entry`` omitted ``parameters`` for a zero-argument method
    until today, so an absent key has to read as "this serializer does not
    say", never as "it takes none". Without that statement the same evidence
    is a hedge, not a blocker."""
    workspace = _scaffold(tmp_path, _FAKE_SUCCESS)
    [issue] = _action_issues(_probe(workspace, model=False))

    assert issue.startswith(ACTION_UNVERIFIED_PREFIX)
    assert "declares activate() with no parameters" not in issue
    assert "may have a precondition it cannot construct" in issue


def test_a_refusing_action_is_silent_when_a_sibling_action_does_move_it(tmp_path):
    """"Cancel a checked-out booking" must fail. The probe cannot know which
    state is right, so it only speaks when NO action on the aggregate moved a
    freshly created instance - one working transition retires the whole set."""
    workspace = _scaffold(tmp_path, _FAKE_SUCCESS)
    backend = Path(workspace) / "web_app" / "backend"
    router = backend / "routers" / "room_methods.py"
    router.write_text(
        router.read_text(encoding="utf-8")
        + _ROUTER.split("router = APIRouter()", 1)[1]
        .replace("activate", "open_room")
        .format(body=_REAL_TRANSITION),
        encoding="utf-8",
    )
    assert _action_issues(_probe(workspace)) == []


def test_a_business_rule_that_refuses_the_guessed_create_is_not_an_action_finding(tmp_path):
    """Hotel's capacity and no-overlap OCL invariants now reach the generated
    app, so the probe meets backends that legitimately refuse its guessed
    payload. A create the app rejected is reported as unverified, and the
    action probe stays out of it entirely - it never runs on an entity that
    was not created."""
    workspace = _scaffold(tmp_path, _FAKE_SUCCESS)
    backend = Path(workspace) / "web_app" / "backend"
    room_router = backend / "routers" / "room.py"
    lines = room_router.read_text(encoding="utf-8").splitlines(keepends=True)
    at = next(i for i, line in enumerate(lines) if line.startswith("async def create_room("))
    lines.insert(at + 1, '    raise HTTPException(status_code=400, detail='
                         '"noOverlappingBookings: the room is already booked")\n')
    room_router.write_text("".join(lines), encoding="utf-8")

    report = _probe(workspace)
    assert _action_issues(report) == []
    [create] = [i for i in report["issues"] if "POST /room/" in i]
    assert create.startswith(UNVERIFIED_PREFIX)
    assert "noOverlappingBookings" in create
    assert report["backends"][0]["entities"]["Room"]["verdict"] == "rejected"


# ------------------------------- "initial state" has to mean it (found by replay)

def test_a_lifecycle_flag_starts_false_so_the_first_call_is_the_first_call(tmp_path):
    """Replaying this probe over the 330 recorded runs produced exactly one
    blocker on an app the oracle passed 5/5: gpt-5.6-terra-1s14uohe, whose
    ``approve`` correctly answers ``{"succeeded": false}`` for a Timesheet
    that is already approved. The fixture had set ``approved`` to True at
    create - ``_sample`` returns True for a boolean - so the probe approved
    an approved timesheet and called the right answer a defect."""
    workspace = _scaffold(tmp_path, (
        '    if _room.activated:\n'
        '        return {"succeeded": False}\n'
        '    _room.activated = True\n'
        '    database.commit()\n'
        '    return {"succeeded": True}\n'
    ), model=_flag_model())
    report = _probe(workspace, model=_flag_model())

    assert _action_issues(report) == []
    [call] = report["backends"][0]["action_calls"]
    assert call["verdict"] == "effective"


def test_every_lifecycle_state_is_tried_before_calling_a_workflow_dead(tmp_path):
    """The other two blockers the replay produced, both on the same shape:
    ``SessionStatus`` sorts CANCELLED before SCHEDULED, the probe built a
    cancelled session from the first enum literal, and then reported that
    ``cancel()`` refuses. The create schema's literal order is not the
    lifecycle's order, so every literal the entity's own create field can
    select is tried before concluding that nothing moves."""
    workspace = _scaffold(tmp_path, "    pass\n", model=_state_model())
    backend = Path(tmp_path) / "web_app" / "backend"
    # A B-UML Enumeration holds its literals in a set, so which one the create
    # schema lists first is not stable across interpreters. Guard on the one
    # the probe reaches SECOND, so the first state always refuses.
    block = backend.joinpath("sql_alchemy.py").read_text(encoding="utf-8")
    block = block.split("class RoomState(enum.Enum):", 1)[1].split("\n\n", 1)[0]
    first, second = re.findall(r"\n    (\w+) = ", block)[:2]
    backend.joinpath("routers", "room_methods.py").write_text(_ROUTER.format(body=(
        f'    if _room.status != RoomState.{second}:\n'
        '        return {"succeeded": False}\n'
        f'    _room.status = RoomState.{first}\n'
        '    database.commit()\n'
        '    return {"succeeded": True}\n'
    )), encoding="utf-8")
    report = _probe(workspace, model=_state_model())

    # The state the probe builds first is the one that refuses, so without the
    # sweep this reads as an aggregate with no first transition.
    assert report["backends"][0]["action_calls"][0]["verdict"] == "inert"
    assert _action_issues(report) == []


# ---------------------------------------------- the report is returned, not flattened

def test_the_structured_report_survives_the_call(tmp_path):
    """``collect_constructibility_issues`` flattened the probe's own
    per-entity/per-action record to ``list[str]`` and threw the record away,
    so every consumer had to re-derive runtime facts by matching prefixes back
    out of prose."""
    report = _scaffold(tmp_path, _REAL_TRANSITION)
    result = _probe(report)

    assert set(result) == {"issues", "backends"}
    [backend] = result["backends"]
    assert backend["backend"] == "web_app/backend"
    assert backend["boot"] == "ok"
    assert backend["entities"]["Room"]["verdict"] == "created"
    assert backend["entities"]["Bill"]["verdict"] == "created"
    assert [c["verdict"] for c in backend["action_calls"]] == ["effective"]


def test_the_tree_score_ranks_on_the_probe_record_not_on_issue_strings(tmp_path):
    """``_phase3_tree_score`` counted ``create unverified:``/``action
    unverified:`` lines. An action the probe confirmed effective and one it
    never reached both contributed nothing, and a finding rendered twice
    counted twice."""
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.generators.llm.orchestrator import LLMOrchestrator, ValidationIssue

    class _Usage:
        estimated_cost = 0.0

        def summary(self) -> dict:
            return {"api_calls": 0, "cost_usd": 0.0}

    class _Client:
        model = "test-model"
        usage = _Usage()
        max_tokens = 4096

        def chat(self, system=None, messages=None, tools=None, **kwargs):
            return {"stop_reason": "end_turn", "content": []}

    workspace = _scaffold(tmp_path, _FAKE_SUCCESS)
    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="Hotel", types={Class(name="Room")}),
        output_dir=workspace,
        enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    lint = [ValidationIssue("blocker", f"ruff: x.py:1:1: F821 undefined name 'x' #{n}")
            for n in range(5)]
    assert orch._phase3_tree_score(lint) == (0, 0, 0, 5)

    orch._runtime_probe_facts = (orch._workspace_revision(), [{
        "backend": "web_app/backend", "boot": "ok",
        "entities": {"Room": {"verdict": "created"}, "Bill": {"verdict": "rejected"}},
        "action_calls": [{"verdict": "effective"}, {"verdict": "inert"}],
    }])
    # One entity the app cannot create and one action that does nothing,
    # measured - not counted off the strings, which name neither.
    assert orch._phase3_tree_score(lint) == (0, 1, 1, 5)
    # Stale facts (measured on a tree that is no longer the one being ranked)
    # fall back rather than describe the wrong tree.
    orch._runtime_probe_facts = ("some-other-revision", orch._runtime_probe_facts[1])
    assert orch._phase3_tree_score(lint) == (0, 0, 0, 5)
