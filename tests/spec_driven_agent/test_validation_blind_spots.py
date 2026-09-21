"""Two ways a generated app shipped "0 blockers" while being unable to start.

Both found in a 10-app live batch on 2026-09-11, where the run's own verdict
disagreed with the app in BOTH directions: one app reported "0 blockers /
23 total, done" and could not import, while two reported INCOMPLETE and were
fine (they had merely run out of turns).

1. F811 (redefinition) was filed as a STYLE code, grouped with F401 and F841.
   Those two are cosmetic; F811 is not - it means two things share a name and
   the later one silently wins. Every one of the four F811 hits across the
   batch was a real defect: a duplicated ORM model, a duplicated Create
   schema, a duplicated endpoint function that replaced the first, and an ORM
   `User` shadowed by a Pydantic `User` that was then passed to db.query().

2. A missing MODULE is invisible to ruff. ``from sql_alchemy import *`` makes
   it report "unable to detect undefined names", which EXCUSES every name the
   module needed instead of flagging it. One app imported sql_alchemy and
   pydantic_classes with neither file present anywhere.

3. A relationship() whose string arguments resolve to nothing is invisible to
   every static gate, because SQLAlchemy configures mappers lazily, on the
   first query. Live run 52befadf (2026-09-18) booted, passed ast.parse and
   ruff, and returned 500 on every database request.
"""

import pytest

from besser.spec_driven_agent.orchestrator import (
    _classify_issue,
    _tool_call_detail,
    _unresolvable_local_imports,
)


# --------------------------------------------------- F811 is a blocker


def test_a_redefinition_is_a_blocker_not_a_style_nit():
    issue = _classify_issue(
        "ruff: backend/routers/user.py:5:50: F811 Redefinition of unused "
        "`User` from line 4: `User` redefined here"
    )
    assert issue.severity == "blocker"


@pytest.mark.parametrize("code", ["F821", "F822", "F823"])
def test_the_existing_blockers_are_unchanged(code):
    issue = _classify_issue(f"ruff: app/x.py:1:1: {code} Undefined name `X`")
    assert issue.severity == "blocker"


@pytest.mark.parametrize("code,name", [
    ("F401", "unused import"), ("F841", "unused variable"), ("E501", "long line"),
])
def test_genuinely_cosmetic_codes_stay_style(code, name):
    """Promoting these would make the fix loop chase whitespace."""
    issue = _classify_issue(f"ruff: app/x.py:1:1: {code} something about {name}")
    assert issue.severity == "style"


# ------------------------------------------- missing modules are blockers


def _write(root, rel, text=""):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_live_failure_is_caught(tmp_path):
    """backend/main_api.py imported sql_alchemy; the file was nowhere."""
    _write(tmp_path, "backend/main_api.py",
           "from sql_alchemy import *\nfrom database import get_db\n")
    _write(tmp_path, "backend/database.py", "def get_db():\n    pass\n")
    problems = _unresolvable_local_imports(str(tmp_path))
    assert any("sql_alchemy" in p for p in problems)
    assert all(_classify_issue(p).severity == "blocker" for p in problems)


def test_a_module_in_a_sibling_directory_does_not_count(tmp_path):
    """A copy of pydantic_classes.py under pydantic/ is not reachable from
    backend/. Accepting it hid half of the live breakage."""
    _write(tmp_path, "backend/main_api.py", "from pydantic_classes import *\n")
    _write(tmp_path, "pydantic/pydantic_classes.py", "X = 1\n")
    assert any("pydantic_classes" in p
               for p in _unresolvable_local_imports(str(tmp_path)))


def test_a_router_importing_from_its_service_root_is_fine(tmp_path):
    """The service runs with backend/ as cwd, so backend/ is importable from
    everything beneath it. Resolving only against the file's own directory
    condemned six imports in an app that demonstrably works."""
    _write(tmp_path, "backend/main_api.py", "from routers import team\n")
    _write(tmp_path, "backend/sql_alchemy.py", "class Team: pass\n")
    _write(tmp_path, "backend/database.py", "def get_db(): pass\n")
    _write(tmp_path, "backend/routers/__init__.py")
    _write(tmp_path, "backend/routers/team.py",
           "from sql_alchemy import *\nfrom database import get_db\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_a_namespace_package_needs_no_init(tmp_path):
    """Python 3 implicit namespace packages: `from routers import post` works
    without __init__.py, and demanding it flagged a working app."""
    _write(tmp_path, "main_api.py", "from routers import post\n")
    _write(tmp_path, "routers/post.py", "router = None\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


@pytest.mark.parametrize("line", [
    "from fastapi import FastAPI", "import sqlalchemy", "from jose import jwt",
    "from passlib.context import CryptContext", "import os, sys",
    "from typing import Optional", "from datetime import datetime",
])
def test_third_party_and_stdlib_imports_are_not_flagged(tmp_path, line):
    _write(tmp_path, "backend/main_api.py", line + "\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_a_relative_import_is_left_alone(tmp_path):
    _write(tmp_path, "pkg/__init__.py")
    _write(tmp_path, "pkg/a.py", "from .b import thing\n")
    _write(tmp_path, "pkg/b.py", "thing = 1\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_a_file_that_does_not_parse_is_left_to_the_syntax_check(tmp_path):
    _write(tmp_path, "backend/broken.py", "def oops(:\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


def test_node_modules_is_not_walked(tmp_path):
    _write(tmp_path, "node_modules/dep/setup.py", "from nonexistent import x\n")
    assert _unresolvable_local_imports(str(tmp_path)) == []


# ------------------------------------------------ the stream says what happened


def test_the_detail_names_the_file_a_write_touched():
    detail = _tool_call_detail("write_file", {"path": "backend/main_api.py",
                                              "content": "x" * 5000}, 1)
    assert "backend/main_api.py" in detail
    assert "xxxx" not in detail, "file content must never reach the event stream"


def test_the_detail_shows_which_task_ids_were_marked():
    detail = _tool_call_detail("task_list", {"action": "done", "ids": [1, 2, 3]}, 1)
    assert "action=done" in detail
    assert "1,2,3" in detail


def test_batching_is_visible_only_when_it_happened():
    """1 call per turn was the invisible default across every live run."""
    assert "batched" not in _tool_call_detail("write_file", {"path": "a.py"}, 1)
    assert "batched=4" in _tool_call_detail("write_file", {"path": "a.py"}, 4)


def test_the_detail_is_bounded_and_single_line():
    detail = _tool_call_detail(
        "run_command", {"command": "echo " + "long " * 200 + "\nsecond line"}, 1)
    assert len(detail) <= 160
    assert "\n" not in detail


def test_a_tool_with_nothing_worth_reporting_gives_an_empty_detail():
    assert _tool_call_detail("list_files", {}, 1) == ""


def test_odd_input_does_not_raise():
    for bad in (None, "a string", 42, [], {"path": None}):
        _tool_call_detail("write_file", bad, 1)


def test_a_three_argument_on_progress_callback_still_works():
    """``on_progress`` is public API of a published package: a caller may still
    pass the original (turn, tool, status) callback. Passing `detail`
    unconditionally raised TypeError for them and aborted the turn."""
    from besser.spec_driven_agent.orchestrator import LLMOrchestrator

    seen = []
    orch = object.__new__(LLMOrchestrator)       # no heavy __init__ needed
    orch.on_progress = lambda turn, tool, status: seen.append((turn, tool, status))
    orch._emit_progress(3, "write_file", "executing", "path=a.py")
    orch._emit_progress(4, "task_list", "executing", "action=done ids=1,2")
    assert seen == [(3, "write_file", "executing"), (4, "task_list", "executing")]
    # The arity is remembered, so it is not re-probed on every call.
    assert orch._progress_takes_detail is False


def test_a_four_argument_callback_receives_the_detail():
    from besser.spec_driven_agent.orchestrator import LLMOrchestrator

    seen = []
    orch = object.__new__(LLMOrchestrator)
    orch.on_progress = lambda *args: seen.append(args)
    orch._emit_progress(1, "write_file", "executing", "path=a.py")
    assert seen == [(1, "write_file", "executing", "path=a.py")]


def test_a_raising_callback_never_breaks_the_run():
    from besser.spec_driven_agent.orchestrator import LLMOrchestrator

    def explode(*_args):
        raise RuntimeError("callback bug")

    orch = object.__new__(LLMOrchestrator)
    orch.on_progress = explode
    orch._emit_progress(1, "write_file", "executing", "path=a.py")   # must not raise


# ------------------------------------------ mappers that fail on first use


# Trimmed from the live download of run 52befadf (2026-09-18): Phase 2 added
# ``relationship(..., secondary="booking_guest", ...)`` inside the class body
# of a file whose only many-to-many table is ``guests``. The module imports,
# ast.parse and ruff pass, and every database request returned 500. Removing
# just that line made 14 of 15 workflow checks pass on the real app; the same
# removal below gives the healthy fixture.
_FATAL_LINE = (
    '    guests: Mapped[List["Guest"]] = relationship("Guest", '
    'secondary="booking_guest", back_populates="booking")'
)

_LIVE52_SQL_ALCHEMY = '''import enum
import os
from typing import List, Optional, List as List_, Optional as Optional_
from sqlalchemy import (
    create_engine, Enum,
    Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, Interval,
    PickleType, String, Table, Text, Time,
    Column as Column_, ForeignKey as ForeignKey_, Table as Table_,
    Text as Text_, Boolean as Boolean_, String as String_, Date as Date_,
    Time as Time_, DateTime as DateTime_, Float as Float_, Integer as Integer_,
    Interval as Interval_, PickleType as PickleType_,
)
from sqlalchemy.orm import (
    column_property, DeclarativeBase, Mapped, Mapped as Mapped_, mapped_column,
    relationship
)


class Base(DeclarativeBase):
    pass

# Tables definition for many-to-many relationships
guests = Table_(
    "guests",
    Base.metadata,
    Column_("guests", ForeignKey_("booking.id"), primary_key=True),
    Column_("guest", ForeignKey_("guest.id"), primary_key=True),
)

# Tables definition
class Booking(Base):
    __tablename__ = "booking"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    contact_id: Mapped_[int] = mapped_column(ForeignKey_("person.id"))

    # Relationships
    contact: Mapped["Person"] = relationship("Person", back_populates="booking")
''' + _FATAL_LINE + '''

class Person(Base):
    __tablename__ = "person"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    type_spec: Mapped_[str] = mapped_column(String_(50))
    __mapper_args__ = {
        "polymorphic_identity": "person",
        "polymorphic_on": "type_spec",
    }

class Guest(Person):
    __tablename__ = "guest"
    id: Mapped_[int] = mapped_column(ForeignKey_("person.id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity": "guest",
    }


#--- Relationships of the booking table
Booking.contact: Mapped_["Person"] = relationship("Person", back_populates="booking", uselist=False, foreign_keys=[Booking.contact_id])
Booking.guest: Mapped_[List_["Guest"]] = relationship("Guest", secondary=guests, back_populates="guests")

#--- Relationships of the person table
Person.booking: Mapped_[List_["Booking"]] = relationship("Booking", back_populates="contact", foreign_keys=[Booking.contact_id])

#--- Relationships of the guest table
Guest.guests: Mapped_[List_["Booking"]] = relationship("Booking", secondary=guests, back_populates="guest")

# Database connection (override the default with the DATABASE_URL environment variable)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/Class_Diagram.db")  # SQLite connection
engine = create_engine(DATABASE_URL)
'''

_HEALTHY_SQL_ALCHEMY = _LIVE52_SQL_ALCHEMY.replace(_FATAL_LINE + "\n", "")
_FATAL_LINE_NO = _LIVE52_SQL_ALCHEMY.splitlines().index(_FATAL_LINE) + 1


def test_the_live_mapper_failure_is_a_blocker_that_names_the_line(tmp_path):
    """Nested web_app/backend/ layout, as downloaded."""
    from besser.spec_driven_agent.orchestrator import _import_smoke_issues

    _write(tmp_path, "web_app/backend/sql_alchemy.py", _LIVE52_SQL_ALCHEMY)
    issues = _import_smoke_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert issues[0].startswith(
        f"mapper config: web_app/backend/sql_alchemy.py line {_FATAL_LINE_NO}:"
    ), issues[0]
    assert "booking_guest" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


def test_the_same_file_without_that_line_is_clean(tmp_path):
    """Flat backend/ layout: the file is found wherever it is."""
    from besser.spec_driven_agent.orchestrator import _import_smoke_issues

    _write(tmp_path, "backend/sql_alchemy.py", _HEALTHY_SQL_ALCHEMY)
    assert _import_smoke_issues(str(tmp_path)) == []


def test_an_import_time_name_error_is_caught_with_its_own_line(tmp_path):
    """The star-import blind spot: ruff excuses the name, the import dies."""
    from besser.spec_driven_agent.orchestrator import _import_smoke_issues

    _write(tmp_path, "backend/sql_alchemy.py", _HEALTHY_SQL_ALCHEMY)
    _write(tmp_path, "backend/pydantic_classes.py",
           "from pydantic import BaseModel\n\n\nclass BookingCreate(BaseModel):\n"
           "    arrivalDate: dt_date\n")
    issues = _import_smoke_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert issues[0].startswith("mapper config: backend/pydantic_classes.py line 5:")
    assert "dt_date" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


def test_a_missing_interpreter_is_reported_as_not_run(tmp_path, monkeypatch):
    from besser.spec_driven_agent.orchestrator import _import_smoke_issues

    _write(tmp_path, "backend/sql_alchemy.py", _LIVE52_SQL_ALCHEMY)
    monkeypatch.setattr("sys.executable", str(tmp_path / "no-such-python"))
    issues = _import_smoke_issues(str(tmp_path))
    assert issues, "a check that could not run must say so, never return []"
    assert all("did not run" in i and "SKIPPED" in i for i in issues)
    assert all(_classify_issue(i).severity == "warning" for i in issues)


def test_a_dependency_the_harness_lacks_is_not_the_apps_fault(tmp_path):
    """The check runs in the harness interpreter, not the app's venv. A
    third-party module missing HERE proves nothing about the app."""
    from besser.spec_driven_agent.orchestrator import _import_smoke_issues

    _write(tmp_path, "backend/sql_alchemy.py",
           "import no_such_third_party_package_xyz\n" + _HEALTHY_SQL_ALCHEMY)
    issues = _import_smoke_issues(str(tmp_path))
    assert issues and all("did not run" in i for i in issues), issues
    assert all(_classify_issue(i).severity == "warning" for i in issues)


def test_the_phantom_secondary_is_named_at_write_time():
    """Same-turn feedback: the executor lints every write_file / modify_file
    through lint_file, so the model hears about it while the file is still
    in context. Advisory, not blocker - the table could be defined in another
    module; the import smoke check above is the gate that proves it."""
    from besser.spec_driven_agent.contract_checks import DataContract, lint_file

    findings = lint_file("web_app/backend/sql_alchemy.py", _LIVE52_SQL_ALCHEMY,
                         DataContract(pk_types={}))
    assert len(findings) == 1, findings
    assert findings[0].line == _FATAL_LINE_NO
    assert "booking_guest" in findings[0].message
    assert "guests" in findings[0].message, "must list the tables that do exist"
    assert not findings[0].blocker


def test_a_secondary_that_names_a_real_table_is_not_flagged():
    from besser.spec_driven_agent.contract_checks import DataContract, lint_file

    contract = DataContract(pk_types={})
    assert lint_file("backend/sql_alchemy.py", _HEALTHY_SQL_ALCHEMY, contract) == []
    quoted = _HEALTHY_SQL_ALCHEMY.replace("secondary=guests", 'secondary="guests"')
    assert lint_file("backend/sql_alchemy.py", quoted, contract) == []
