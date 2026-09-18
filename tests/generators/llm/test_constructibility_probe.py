"""Phase 3 must block an aggregate that no request can create.

Run 9a6063ed (2026-09-18) delivered a hotel app that passed every static
gate - imports, mappers, star-import names, 61 routes, Person/Employee/
Guest/Room all creating fine - and could not create a Booking by ANY
sequence of requests. Phase 2 had implemented the gap analyser's task
("validation in create_booking that the guests do not exceed the room
capacities across all BookedRooms") at insert time, when a Booking has no
BookedRooms yet because a BookedRoom needs a Booking id::

    with a guest    -> 400 "Total number of guests (1) exceeds room capacity (0)"
    without a guest -> 400 "At least 1 Guest(s) required"

Bill and BookedRoom, which require a Booking, were dead with it. Nothing
static sees this: the code is ordinary, the schema valid, and whether the
``raise`` is reachable is a data-flow question. So the probe runs the app
and reports an entity only when every schema-valid request is refused -
enum literals, booleans and date order are all tried first, and a 422
(our payload) never counts. The fixture is run 9a6063ed's backend, verbatim.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from besser.generators.llm.constructibility import (
    PREFIX,
    _issues_from_report,
    collect_constructibility_issues,
)
from besser.generators.llm.orchestrator import LLMOrchestrator, ValidationIssue, _classify_issue

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

FIXTURE = Path(__file__).parent / "fixtures" / "run_9a6063ed"


@pytest.fixture
def workspace(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return str(tmp_path)


# --------------------------------------------------------------- the live run


def test_the_live_run_cannot_create_a_booking(workspace):
    issues = collect_constructibility_issues(workspace)
    assert len(issues) == 1, issues
    [issue] = issues
    assert issue.startswith(f"{PREFIX} web_app/backend: POST /booking/ cannot create a Booking")
    assert "every schema-valid request is rejected (400:" in issue
    # The refused payload carried a guest, so it is the guard, not the
    # multiplicity, that refused it - and the guard is what has to move.
    assert '"guest": [' in issue
    assert "Bill, BookedRoom require a Booking id, so they cannot be created either" in issue
    assert "No Bill or BookedRoom row can exist before the Booking it requires" in issue
    assert "enforce it where those rows are created, updated or deleted" in issue
    assert _classify_issue(issue).severity == "blocker"


def test_a_recollection_reports_the_same_thing(workspace):
    """The fix loop compares blocker counts across rounds."""
    assert collect_constructibility_issues(workspace) == collect_constructibility_issues(workspace)


class _Usage:
    estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 0, "cost_usd": 0.0}


class _ScriptedClient:
    model = "test-model"
    usage = _Usage()
    max_tokens = 4096

    def chat(self, system=None, messages=None, tools=None, **kwargs):
        return {"stop_reason": "end_turn", "content": []}


def test_the_phase3_sweep_carries_the_blocker(workspace):
    """What shipped as "0 blockers" must come out of the sweep as one."""
    from besser.BUML.metamodel.structural import Class, DomainModel

    orch = LLMOrchestrator(
        llm_client=_ScriptedClient(),
        domain_model=DomainModel(name="Hotel", types={Class(name="Booking")}),
        output_dir=workspace,
        enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    issues = orch._collect_validation_issues()
    blockers = [i.message for i in issues if i.severity == "blocker"]
    assert [b for b in blockers if b.startswith(PREFIX)], blockers


# ----------------------------------------------------- a fresh generated app


def _scaffold(tmp_path) -> str:
    """A generator-fresh backend: Room (with an enum) and Booking, which
    needs a Room. Returns the workspace root."""
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DateType, DomainModel, Enumeration,
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
    booking = Class(name="Booking", attributes={Property(name="arrivalDate", type=DateType)})
    link = BinaryAssociation(name="room_booking", ends={
        Property(name="room", type=room, multiplicity=Multiplicity(1, 1)),
        Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*")),
    })
    model = DomainModel(name="Hotel", types={room, booking, status}, associations={link})
    BackendGenerator(model=model, output_dir=str(tmp_path / "web_app" / "backend")).generate()
    assert (tmp_path / "web_app" / "backend" / "main_api.py").is_file()
    return str(tmp_path)


def _first_literal(tmp_path) -> str:
    """The literal the probe sends first: the first member of the generated enum."""
    text = (tmp_path / "web_app" / "backend" / "pydantic_classes.py").read_text(encoding="utf-8")
    return re.search(r"class RoomStatus\(Enum\):\s*\n\s*(\w+)\s*=", text).group(1)


def _guard_create_room(tmp_path, condition: str) -> None:
    router = tmp_path / "web_app" / "backend" / "routers" / "room.py"
    text = router.read_text(encoding="utf-8")
    head = re.search(r"async def create_room\(.*\n", text).group(0)
    guarded = head + f'    if {condition}:\n        raise HTTPException(status_code=400, detail="refused")\n'
    router.write_text(text.replace(head, guarded, 1), encoding="utf-8")


def test_a_healthy_generated_scaffold_yields_nothing(tmp_path):
    assert collect_constructibility_issues(_scaffold(tmp_path)) == []


def test_a_rule_that_another_value_satisfies_is_not_a_finding(tmp_path):
    """One 400 means "this request was wrong". The probe keeps trying."""
    workspace = _scaffold(tmp_path)
    _guard_create_room(tmp_path, f'room_data.status.value == "{_first_literal(tmp_path)}"')
    assert collect_constructibility_issues(workspace) == []


def test_a_rule_that_no_value_satisfies_is_a_finding_that_names_the_dependents(tmp_path):
    workspace = _scaffold(tmp_path)
    _guard_create_room(tmp_path, "True")
    issues = collect_constructibility_issues(workspace)
    assert len(issues) == 1, issues
    assert issues[0].startswith(f"{PREFIX} web_app/backend: POST /room/ cannot create a Room")
    assert "Booking require a Room id, so they cannot be created either" in issues[0]


def test_a_mapper_that_fails_to_configure_is_left_to_the_smoke_check(tmp_path):
    """One defect, one blocker: ``mapper config:`` already reports it."""
    workspace = _scaffold(tmp_path)
    orm = tmp_path / "web_app" / "backend" / "sql_alchemy.py"
    orm.write_text(
        orm.read_text(encoding="utf-8") + '\nRoom.ghost = relationship("NoSuchClass")\n',
        encoding="utf-8",
    )
    assert collect_constructibility_issues(workspace) == []


def test_a_workspace_without_the_fastapi_scaffold_has_nothing_to_probe(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    assert collect_constructibility_issues(str(tmp_path)) == []


# ------------------------------------------------- report -> findings (pure)


def _report(**entities) -> dict:
    return {"boot": "ok", "entities": entities}


def _entity(verdict, status, body="", **extra) -> dict:
    return {
        "path": "/x/", "verdict": verdict,
        "attempts": [{"status": status, "outcome": verdict, "body": body, "payload": {"a": 1}}],
        **extra,
    }


def test_an_app_where_nothing_answers_is_not_checked_rather_than_seven_blockers():
    issues = _issues_from_report(
        _report(Room=_entity("crashed", 500, "db down"), Person=_entity("crashed", 500)), "b",
    )
    assert len(issues) == 1 and "did not run" in issues[0]
    assert "no create endpoint answered" in issues[0]
    assert _classify_issue(issues[0]).severity == "warning"


def test_a_crash_beside_a_working_endpoint_is_a_finding():
    issues = _issues_from_report(
        _report(Room=_entity("created", 200), Person=_entity("crashed", 500, "boom")), "b",
    )
    assert len(issues) == 1
    assert issues[0].startswith(f"{PREFIX} b: POST /x/ cannot create a Person - every schema-valid request fails with a server error (500: boom)")


def test_an_authentication_wall_is_not_checked():
    issues = _issues_from_report(_report(Room=_entity("unauthorized", 401)), "b")
    assert len(issues) == 1 and "requires authentication" in issues[0]
    assert _classify_issue(issues[0]).severity == "warning"


def test_a_payload_the_probe_could_not_build_is_no_verdict():
    assert _issues_from_report(_report(Room=_entity("invalid", 422)), "b") == []


def test_a_missing_dependency_is_reported_as_not_checked():
    issues = _issues_from_report({"boot": "missing_dependency", "error": "No module named 'httpx'"}, "b")
    assert len(issues) == 1 and "did not run" in issues[0] and "httpx" in issues[0]
    assert _classify_issue(issues[0]).severity == "warning"


def test_a_mapper_error_is_silent():
    assert _issues_from_report({"boot": "mapper_error", "error": "x"}, "b") == []


def test_create_contract_prefix_classifies_as_blocker():
    assert _classify_issue(f"{PREFIX} b: POST /x/ cannot create an X").severity == "blocker"


# ------------------------------------------------------ run 7f918e11 (NOT NULL)

FIXTURE_7F = Path(__file__).parent / "fixtures" / "run_7f918e11"


@pytest.fixture
def workspace_7f(tmp_path):
    shutil.copytree(FIXTURE_7F, tmp_path, dirs_exist_ok=True)
    return str(tmp_path)


def test_a_not_null_insert_names_the_column_and_the_fix_site(workspace_7f):
    """Run 7f918e11: Phase 2 made BookingCreate.totalAmountDue optional and
    left the column NOT NULL and the insert without it. The finding named
    the symptom exactly and not one thing the model could edit. The fixture
    is that run's backend, verbatim."""
    issues = collect_constructibility_issues(workspace_7f)
    assert len(issues) == 1, issues
    [issue] = issues
    assert issue.startswith(f"{PREFIX} web_app/backend: POST /booking/ cannot create a Booking")
    assert "Fix site: create_booking in web_app/backend/routers/booking.py line 165" in issue
    assert "also line 220" in issue  # bulk_create_booking inserts it the same way
    assert "`totalAmountDue`" in issue and "NOT NULL" in issue
    # The dependents are still named, but the create-time-rule advice from
    # run 9a6063ed is the wrong diagnosis for a constraint failure.
    assert "Bill, ReservedRoom require a Booking id" in issue
    assert "create-time rule" not in issue
    assert _classify_issue(issue).severity == "blocker"


def test_the_fix_site_gives_the_fix_prompt_an_excerpt_to_quote(workspace_7f):
    """Measured on the model that failed there: with the offending lines in
    the prompt it calls modify_file on turn one (6/6); without them it reads."""
    [issue] = collect_constructibility_issues(workspace_7f)
    orch = LLMOrchestrator.__new__(LLMOrchestrator)
    orch.output_dir = workspace_7f
    excerpts = orch._excerpts_for([ValidationIssue("blocker", issue)])
    assert len(excerpts) == 1, excerpts
    assert "  165|     db_booking = Booking(" in excerpts[0]


def test_a_site_and_a_missing_column_are_rendered_from_the_report():
    entity = _entity("rejected", 409, "NOT NULL constraint failed: x.total",
                     site={"file": "routers/x.py", "function": "create_x", "line": 7, "also": [30]})
    entity["attempts"][0]["missing_column"] = "total"
    [issue] = _issues_from_report(_report(X=entity), "b")
    assert "Fix site: create_x in b/routers/x.py line 7 (also line 30)" in issue
    assert "inserted without `total`" in issue


def test_a_rule_failure_keeps_the_create_time_rule_advice():
    dependent = {"path": "/y/", "verdict": "unresolved", "unresolved": [["x", "X"]]}
    [issue] = _issues_from_report(_report(X=_entity("rejected", 400, "refused"), Y=dependent), "b")
    assert "create-time rule" in issue
