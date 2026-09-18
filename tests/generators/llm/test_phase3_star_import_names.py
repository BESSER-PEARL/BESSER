"""Phase 3 must block an undefined name that hides behind ``from x import *``.

The write-time check (test_star_import_blind_spot.py) tells the model in the
same turn; this is the release gate for when it does not listen. ruff shares
pyflakes' rule: in a module with a star import every unresolved name is F405
("may be undefined, or defined from star imports"), never F821, and F405 is
not a blocker. So ``latest`` - the app delivered after run 0c537a4e's sibling
- shipped ``booking_id`` in five endpoints and ``bill_id`` in a sixth as
"0 blockers, done", and every one of those routes is a NameError.

Two constraints from the fix loop: it re-collects each round and rolls back
when ``len(blockers_after) > len(blockers_before)``, so the count must be
stable on identical input; and a healthy scaffold must yield nothing, so a
clean run never enters the loop because of this check.

The fixture is run 0c537a4e's backend, verbatim.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from besser.generators.llm.orchestrator import (
    _classify_issue,
    _star_import_undefined_names,
)

pytest.importorskip("pyflakes")

FIXTURE = Path(__file__).parent / "fixtures" / "run_0c537a4e"


@pytest.fixture
def workspace(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return str(tmp_path)


def test_the_live_run_is_six_blockers_not_a_clean_bill(workspace):
    issues = _star_import_undefined_names(workspace)
    assert len(issues) == 6, issues
    assert all(_classify_issue(i).severity == "blocker" for i in issues)
    assert all("web_app/backend/routers/booking_methods.py" in i for i in issues)
    assert sum("'booking_id'" in i for i in issues) == 5
    helper = next(i for i in issues if "'computeAmountOwed'" in i)
    assert "line 193" in helper
    for module in ("sql_alchemy", "pydantic_classes", "bal_stdlib"):
        assert module in helper


def test_a_recollection_reports_the_same_thing(workspace):
    """The fix loop compares counts across rounds; identical input, identical output."""
    assert _star_import_undefined_names(workspace) == _star_import_undefined_names(workspace)


def test_a_healthy_generated_scaffold_yields_nothing(tmp_path):
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DateType, DomainModel, FloatType, IntegerType,
        Method, Multiplicity, Property,
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
    assert _star_import_undefined_names(str(tmp_path)) == []


def test_an_unresolvable_star_module_stays_silent(tmp_path):
    """``from fastapi import *`` cannot be read off disk: no verdict."""
    (tmp_path / "r.py").write_text("from fastapi import *\nrouter = APIRouter()\n", encoding="utf-8")
    assert _star_import_undefined_names(str(tmp_path)) == []


def test_a_file_without_a_star_import_is_left_to_ruff(tmp_path):
    """ruff already reports F821 there; reporting it twice would double-count blockers."""
    (tmp_path / "plain.py").write_text("x = never_defined\n", encoding="utf-8")
    assert _star_import_undefined_names(str(tmp_path)) == []


def test_the_message_names_file_line_and_name(tmp_path):
    (tmp_path / "m.py").write_text("def present(): pass\n", encoding="utf-8")
    (tmp_path / "use.py").write_text("from m import *\npresent()\nabsent()\n", encoding="utf-8")
    issues = _star_import_undefined_names(str(tmp_path))
    assert issues == [
        "undefined name: use.py line 3: 'absent' - not defined in this file "
        "and not exported by m"
    ]


def test_a_missing_pyflakes_is_reported_as_not_checked(tmp_path, monkeypatch):
    """A silent [] would read as a clean pass; say the check did not run."""
    (tmp_path / "m.py").write_text("", encoding="utf-8")
    (tmp_path / "use.py").write_text("from m import *\nabsent()\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "pyflakes.checker", None)
    issues = _star_import_undefined_names(str(tmp_path))
    assert len(issues) == 1 and "did not run" in issues[0]
    assert _classify_issue(issues[0]).severity == "warning"


def test_snapshot_and_node_modules_are_not_walked(tmp_path):
    from besser.generators.llm.orchestrator import _SNAPSHOT_DIR
    for folder in (_SNAPSHOT_DIR, "node_modules"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "m.py").write_text("", encoding="utf-8")
        (tmp_path / folder / "use.py").write_text("from m import *\nabsent()\n", encoding="utf-8")
    assert _star_import_undefined_names(str(tmp_path)) == []


def test_undefined_name_prefix_classifies_as_blocker():
    issue = _classify_issue("undefined name: web_app/backend/routers/x.py line 9: 'y' - not defined")
    assert issue.severity == "blocker"

