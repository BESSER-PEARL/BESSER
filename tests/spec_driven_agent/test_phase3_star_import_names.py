"""Phase 3 must block an undefined name that hides behind ``from x import *``.

The write-time check (test_star_import_blind_spot.py) tells the model in the
same turn; this is the release gate for when it does not listen. ruff shares
pyflakes' rule: in a module with a star import every unresolved name is F405
("may be undefined, or defined from star imports"), never F821, and F405 is
not a blocker. So ``latest`` - the app delivered after run 0c537a4e's sibling
- shipped ``booking_id`` in five endpoints and ``bill_id`` in a sixth as
"0 blockers, done", and every one of those routes is a NameError.

The gate is the ``python contract:`` sweep in ``_collect_execution_issues``:
the same resolver as the write-time check, over every app file, with or
without a star import. ruff's F821 for the same name and line is dropped, so
one defect is one blocker.

Two constraints from the fix loop: it re-collects each round and rolls back
when ``len(blockers_after) > len(blockers_before)``, so the count must be
stable on identical input; and a healthy scaffold must yield nothing, so a
clean run never enters the loop because of this check.

The fixture is run 0c537a4e's backend, verbatim.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _classify_issue
from besser.spec_driven_agent.providers.llm_client import UsageTracker

pytest.importorskip("pyflakes")

FIXTURE = Path(__file__).parent / "fixtures" / "run_0c537a4e"


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")


def _orchestrator(workspace) -> LLMOrchestrator:
    cls = Class(name="Book")
    cls.attributes = {Property(name="title", type=PrimitiveDataType("str"), is_id=True)}
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="App", types={cls}),
        output_dir=str(workspace), enable_checkpointing=False, enable_tracing=False,
        enable_toolchain_validation=False, enable_import_smoke_check=False,
    )


def _undefined_names(workspace) -> list[str]:
    """The release gate's undefined-name findings."""
    return [i for i in _orchestrator(workspace)._collect_execution_issues()
            if "undefined name" in i]


@pytest.fixture
def workspace(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return str(tmp_path)


def test_the_live_run_is_blocked_not_a_clean_bill(workspace):
    """Six NameErrors from two names: one blocker per name, the repeats counted."""
    issues = _undefined_names(workspace)
    assert len(issues) == 2, issues
    assert all(_classify_issue(i).severity == "blocker" for i in issues)
    assert all("web_app/backend/routers/booking_methods.py" in i for i in issues)
    booking_id = next(i for i in issues if "'booking_id'" in i)
    assert booking_id.endswith("(+4 more uses)")
    helper = next(i for i in issues if "'computeAmountOwed'" in i)
    assert "line 193" in helper
    for module in ("sql_alchemy", "pydantic_classes", "bal_stdlib"):
        assert module in helper


def test_a_recollection_reports_the_same_thing(workspace):
    """The fix loop compares counts across rounds; identical input, identical output."""
    assert _undefined_names(workspace) == _undefined_names(workspace)


def test_a_healthy_generated_scaffold_yields_nothing(tmp_path):
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, DateType, FloatType, IntegerType, Method, Multiplicity,
    )
    from besser.generators.backend import BackendGenerator

    room = Class(name="Room", attributes={Property(name="roomNumber", type=IntegerType)})
    booking = Class(
        name="Booking",
        attributes={Property(name="arrivalDate", type=DateType)},
        methods={Method(name="computeAmountOwed", type=FloatType),
                 Method(name="cancel")},
    )
    link = BinaryAssociation(name="room_booking", ends={
        Property(name="room", type=room, multiplicity=Multiplicity(1, 1)),
        Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*")),
    })
    model = DomainModel(name="Hotel", types={room, booking}, associations={link})
    BackendGenerator(model=model, output_dir=str(tmp_path / "web_app" / "backend")).generate()
    assert (tmp_path / "web_app" / "backend" / "routers" / "booking_methods.py").is_file()
    assert _undefined_names(tmp_path) == []


def test_an_unresolvable_star_module_stays_silent(tmp_path):
    """``from fastapi import *`` cannot be read off disk: no verdict."""
    (tmp_path / "r.py").write_text("from fastapi import *\nrouter = APIRouter()\n", encoding="utf-8")
    assert _undefined_names(tmp_path) == []


def test_the_message_names_file_line_and_name(tmp_path):
    (tmp_path / "m.py").write_text("def present(): pass\n", encoding="utf-8")
    (tmp_path / "use.py").write_text("from m import *\npresent()\nabsent()\n", encoding="utf-8")
    assert _undefined_names(tmp_path) == [
        "python contract: use.py line 3: undefined name 'absent' - not defined in this file "
        "and not exported by m"
    ]


def test_snapshot_and_node_modules_are_not_walked(tmp_path):
    from besser.spec_driven_agent.pipeline.orchestrator import _SNAPSHOT_DIR
    for folder in (_SNAPSHOT_DIR, "node_modules"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "m.py").write_text("", encoding="utf-8")
        (tmp_path / folder / "use.py").write_text("from m import *\nabsent()\n", encoding="utf-8")
    assert _undefined_names(tmp_path) == []


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed on this host")
def test_one_undefined_name_is_one_blocker(tmp_path):
    """ruff's F821 and the python contract see the same NameError; the fix loop
    counts blockers, so reporting it twice doubles the defect."""
    (tmp_path / "helpers.py").write_text(
        "def total(xs):\n    return sum(xs) + offset\n", encoding="utf-8")

    blockers = [i.message for i in _orchestrator(tmp_path)._collect_validation_issues()
                if i.severity == "blocker" and "offset" in i.message]

    assert blockers == ["python contract: helpers.py line 2: undefined name 'offset'"]


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed on this host")
def test_one_syntax_error_is_one_finding(tmp_path):
    (tmp_path / "util.py").write_text("def f(:\n    pass\n", encoding="utf-8")

    findings = [i.message for i in _orchestrator(tmp_path)._collect_validation_issues()
                if "util.py" in i.message]

    assert findings == ["Syntax error in util.py line 1: invalid syntax"]


def test_one_forgotten_import_is_one_blocker_per_name(tmp_path):
    """40 endpoints using get_db and Thing were 105 blockers (80 ruff + 25 contract)."""
    (tmp_path / "router.py").write_text("".join(
        f"def r{i}(db=get_db()):\n    return db.query(Thing)\n" for i in range(40)), encoding="utf-8")

    blockers = [i.message for i in _orchestrator(tmp_path)._collect_validation_issues()
                if i.severity == "blocker"]

    assert blockers == [
        "python contract: router.py line 1: undefined name 'get_db' (+39 more uses)",
        "python contract: router.py line 2: undefined name 'Thing' (+39 more uses)",
    ]
