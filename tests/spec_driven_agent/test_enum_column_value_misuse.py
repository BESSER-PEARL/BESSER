"""``obj.enum_column != SomeEnum.MEMBER.value`` where ``enum_column`` is a
``Column(Enum(...))`` ORM attribute.

A recorded run: a generated hotel app's ``sql_alchemy.py`` declares

    class BookingPhysicalStatus(PyEnum):
        NOT_YET_ARRIVED = "NOT_YET_ARRIVED"
        ...

    class Booking(Base):
        physicalStatus = Column(Enum(BookingPhysicalStatus), ...)

and ``routers/booking_methods.py`` (``from sql_alchemy import *``) wrote

    if _booking_object.physicalStatus != BookingPhysicalStatus.NOT_YET_ARRIVED.value:
        raise HTTPException(status_code=400, detail="Cannot register arrival: ...")

Confirmed empirically against the running app: the stored value is the enum
*member*, ``== MEMBER`` is True, ``== MEMBER.value`` is False. So the guard
(and its two siblings on the same class) can never pass: ``registerArrival``,
``registerDeparture`` and ``cancel`` were unreachable in that run. Valid
Python, valid types - ``ast.parse``, pyflakes and ruff all accept it.
"""
from __future__ import annotations

import textwrap

import pytest

from besser.spec_driven_agent.validation.write_diagnostics import (
    diagnose_written_content, python_structural_diagnostics,
)

ROUTER_PATH = "backend/routers/booking_methods.py"
ORM_PATH = "backend/sql_alchemy.py"

SQL_ALCHEMY = textwrap.dedent("""\
    import enum
    from sqlalchemy import Column, Enum, Integer, String
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


    class Base(DeclarativeBase):
        pass


    class BookingPhysicalStatus(enum.Enum):
        NOT_YET_ARRIVED = "NOT_YET_ARRIVED"
        CHECKED_IN = "CHECKED_IN"
        CHECKED_OUT = "CHECKED_OUT"


    class Booking(Base):
        __tablename__ = "booking"
        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        bookingNumber: Mapped[str] = mapped_column(String(100))
        physicalStatus: Mapped[BookingPhysicalStatus] = mapped_column(Enum(BookingPhysicalStatus))
    """)

# A second ORM module where the enum mixes in ``str``: its members DO compare
# equal to their own ``.value`` (Grade.A == "A" is True), so the same shape
# must NOT be flagged.
SQL_ALCHEMY_STR_ENUM = textwrap.dedent("""\
    import enum
    from sqlalchemy import Enum, Integer
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


    class Base(DeclarativeBase):
        pass


    class Grade(str, enum.Enum):
        A = "A"
        B = "B"


    class Exam(Base):
        __tablename__ = "exam"
        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        grade: Mapped["Grade"] = mapped_column(Enum(Grade))
    """)


def _findings(tmp_path, orm_source, router_source, orm_path=ORM_PATH, router_path=ROUTER_PATH):
    (tmp_path / orm_path.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
    (tmp_path / orm_path).write_text(orm_source, encoding="utf-8")
    return diagnose_written_content(router_path, router_source, workspace=str(tmp_path))


def _enum_findings(findings):
    return [f for f in findings if f["code"] == "enum-column-compared-to-value"]


# -- required case: the exact live shape is reported -----------------------
def test_live_shape_star_imported_enum_column_compared_to_value_is_reported(tmp_path):
    """FAILS before the fix: python_structural_diagnostics had no notion of
    star-imported ORM columns, so this comparison produced zero findings."""
    router = textwrap.dedent("""\
        from sql_alchemy import *


        def register_arrival(_booking_object):
            if _booking_object.physicalStatus != BookingPhysicalStatus.NOT_YET_ARRIVED.value:
                raise ValueError("Cannot register arrival")
            _booking_object.physicalStatus = BookingPhysicalStatus.CHECKED_IN.value
            return _booking_object
        """)
    findings = _enum_findings(_findings(tmp_path, SQL_ALCHEMY, router))
    assert len(findings) == 1, findings
    finding = findings[0]
    assert finding["line"] == 5
    assert "physicalStatus" in finding["message"]
    assert "BookingPhysicalStatus" in finding["message"]
    assert "NOT_YET_ARRIVED" in finding["message"]
    # The assignment on the next line (also .value) is NOT the shape this
    # detector covers - see test_module docstring / final report for why.
    assert all(f["line"] != 7 for f in findings)


def test_eq_operator_is_also_reported(tmp_path):
    """FAILS before the fix, same as the ``!=`` case above."""
    router = textwrap.dedent("""\
        from sql_alchemy import *


        def is_pending(_booking_object):
            return _booking_object.physicalStatus == BookingPhysicalStatus.NOT_YET_ARRIVED.value
        """)
    findings = _enum_findings(_findings(tmp_path, SQL_ALCHEMY, router))
    assert len(findings) == 1, findings
    assert findings[0]["line"] == 5


def test_membership_in_tuple_of_values_is_reported(tmp_path):
    """FAILS before the fix: the ``in (...).value`` shape had no detector at
    all, structural or otherwise. The tuple must appear inline: this is an
    AST-only check with no data-flow analysis to resolve a variable back to
    the literal it was assigned from."""
    router = textwrap.dedent("""\
        from sql_alchemy import *


        def is_active(_booking_object):
            if _booking_object.physicalStatus not in (
                BookingPhysicalStatus.CHECKED_IN.value, BookingPhysicalStatus.CHECKED_OUT.value,
            ):
                return False
            return True
        """)
    findings = _enum_findings(_findings(tmp_path, SQL_ALCHEMY, router))
    assert len(findings) == 1, findings
    assert findings[0]["line"] == 5
    assert "CHECKED_IN" in findings[0]["message"] and "CHECKED_OUT" in findings[0]["message"]


# -- required case: the correct form is silent ------------------------------
def test_correct_form_without_value_is_not_reported(tmp_path):
    """Passes both before and after the fix - the bar the detector must not
    cross. ``!= BookingPhysicalStatus.NOT_YET_ARRIVED`` compares member to
    member, which is exactly right."""
    router = textwrap.dedent("""\
        from sql_alchemy import *


        def register_arrival(_booking_object):
            if _booking_object.physicalStatus != BookingPhysicalStatus.NOT_YET_ARRIVED:
                raise ValueError("Cannot register arrival")
            return _booking_object
        """)
    assert _enum_findings(_findings(tmp_path, SQL_ALCHEMY, router)) == []


# -- required case: a plain string column is not flagged (no false positive)
def test_string_column_compared_to_value_is_not_flagged(tmp_path):
    """FAILS before the fix in the trivial sense that it also reports
    nothing before the fix (no detector existed at all) - included to prove
    the detector, once added, does not over-fire on it. 'bookingNumber' is a
    Column(String), not a Column(Enum(...)); comparing it to a member's
    .value is ordinary, correct string comparison."""
    router = textwrap.dedent("""\
        from sql_alchemy import *


        def check_number(_booking_object, expected):
            if _booking_object.bookingNumber != BookingPhysicalStatus.NOT_YET_ARRIVED.value:
                return False
            return True
        """)
    assert _enum_findings(_findings(tmp_path, SQL_ALCHEMY, router)) == []


# -- required case: a str-mixin enum's .value comparison is correct --------
def test_str_mixin_enum_compared_to_value_is_not_flagged(tmp_path):
    """A ``class Grade(str, enum.Enum)`` member equals its own .value under
    Python's default equality, so this comparison is not a bug. Calibrates
    the detector against the one shape that looks identical to the live bug
    but is not one."""
    router = textwrap.dedent("""\
        from sql_alchemy import *


        def is_a(_exam):
            return _exam.grade == Grade.A.value
        """)
    assert _enum_findings(_findings(tmp_path, SQL_ALCHEMY_STR_ENUM, router)) == []


# -- required case: an unresolvable ORM module reports nothing -------------
def test_unresolvable_orm_module_reports_nothing(tmp_path):
    """No sql_alchemy.py on disk at all: the module cannot be resolved, so
    nothing is reported rather than guessed at. Passes both before and after
    the fix; included as the contract's explicit boundary."""
    router = textwrap.dedent("""\
        from sql_alchemy import *


        def register_arrival(_booking_object):
            if _booking_object.physicalStatus != BookingPhysicalStatus.NOT_YET_ARRIVED.value:
                raise ValueError("Cannot register arrival")
            return _booking_object
        """)
    (tmp_path / "backend" / "routers").mkdir(parents=True)
    findings = diagnose_written_content(ROUTER_PATH, router, workspace=str(tmp_path))
    assert _enum_findings(findings) == []


# -- backward compatibility: the pre-existing 2-argument call still works --
def test_python_structural_diagnostics_keeps_its_old_two_argument_call_working():
    """requirements_ledger.py calls python_structural_diagnostics(tree,
    sqlite=sqlite) with no rel_path/workspace; the new parameters must be
    optional and default to skipping the new check, not raise."""
    import ast

    tree = ast.parse(
        "class Status(Enum):\n    A = 'A'\n"
        "def f(x):\n    return x.status != Status.A.value\n"
    )
    findings = python_structural_diagnostics(tree, sqlite=False)
    assert _enum_findings(findings) == []
