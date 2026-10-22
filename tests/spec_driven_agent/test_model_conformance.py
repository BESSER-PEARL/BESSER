"""Post-run check: does the FastAPI backend follow the model change? (#659)

Live hotel case m (Person.phone removed, Room.description renamed to
amenities, Room.floor and Person.dateOfBirth added, Booking.bookingNumber
str -> int): a plain modify run applied none of it in 3/3 runs and reported
clean. The snippets below are trimmed from those runs' backend/ (``TODAY``)
and from runs that applied the change (``FOLLOWED``, 6/6 with the change
list). Calibrated on 33 recorded apps: every stale one flagged, zero findings
on the ones that follow the model.
"""
import re

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Generalization, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.planning.model_change import compare_models, model_summary
from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.model_conformance import model_conformance_issues

_HEADER = (
    "from typing import List, Optional, Optional as Optional_\n"
    "from sqlalchemy import Integer as Integer_, String as String_, Float as Float_, Date as Date_\n"
    "from sqlalchemy import ForeignKey as ForeignKey_\n"
    "from sqlalchemy.orm import DeclarativeBase, Mapped as Mapped_, mapped_column, relationship\n"
    "from datetime import date as dt_date\n\n"
    "class Base(DeclarativeBase):\n    pass\n\n"
)

TODAY_ORM = _HEADER + '''
class Room(Base):
    __tablename__ = "room"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    roomNumber: Mapped_[str] = mapped_column(String_(100), unique=True)
    maxOccupancy: Mapped_[int] = mapped_column(Integer_)
    description: Mapped_[str] = mapped_column(String_(100))
    standardPrice: Mapped_[float] = mapped_column(Float_)

class Person(Base):
    __tablename__ = "person"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    firstName: Mapped_[str] = mapped_column(String_(100))
    phone: Mapped_[str] = mapped_column(String_(100))
    email: Mapped_[str] = mapped_column(String_(100))

class Employee(Person):
    __tablename__ = "employee"
    id: Mapped_[int] = mapped_column(ForeignKey_("person.id"), primary_key=True)

class Booking(Base):
    __tablename__ = "booking"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    bookingNumber: Mapped_[str] = mapped_column(String_(100), unique=True)
    arrivalDate: Mapped_[dt_date] = mapped_column(Date_)
    handledBy_id: Mapped_[int] = mapped_column(ForeignKey_("employee.id"))

Booking.handledBy: Mapped_["Employee"] = relationship("Employee", foreign_keys=[Booking.handledBy_id])
'''

TODAY_SCHEMAS = '''
from datetime import date
from typing import List, Optional
from pydantic import BaseModel

class RoomCreate(BaseModel):
    description: str
    maxOccupancy: int
    roomNumber: str
    standardPrice: float

class PersonCreate(BaseModel):
    email: str
    firstName: str
    phone: str
    booking: Optional[List[int]] = None  # 1:N Relationship

class EmployeeCreate(PersonCreate):
    bookingsHandled: Optional[List[int]] = None

class BookingCreate(BaseModel):
    arrivalDate: date
    bookingNumber: str
    handledBy: int  # N:1 Relationship (mandatory)
'''

FOLLOWED_ORM = _HEADER + '''
class Room(Base):
    __tablename__ = "room"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    roomNumber: Mapped_[str] = mapped_column(String_(100), unique=True)
    maxOccupancy: Mapped_[int] = mapped_column(Integer_)
    amenities: Mapped_[str] = mapped_column(String_(100))
    floor: Mapped_[int] = mapped_column(Integer_)
    standardPrice: Mapped_[float] = mapped_column(Float_)

class Person(Base):
    __tablename__ = "person"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    firstName: Mapped_[str] = mapped_column(String_(100))
    dateOfBirth: Mapped_[dt_date] = mapped_column(Date_)
    email: Mapped_[str] = mapped_column(String_(100))

class Employee(Person):
    __tablename__ = "employee"
    id: Mapped_[int] = mapped_column(ForeignKey_("person.id"), primary_key=True)

class Booking(Base):
    __tablename__ = "booking"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    bookingNumber: Mapped_[int] = mapped_column(Integer_, unique=True)
    arrivalDate: Mapped_[dt_date] = mapped_column(Date_)
    handledBy_id: Mapped_[Optional_[int]] = mapped_column(ForeignKey_("employee.id"), nullable=True)
'''

FOLLOWED_SCHEMAS = '''
from datetime import date
from typing import List, Optional
from pydantic import BaseModel

class RoomCreate(BaseModel):
    amenities: str
    floor: int
    maxOccupancy: int
    roomNumber: str
    standardPrice: float

class PersonCreate(BaseModel):
    dateOfBirth: date
    email: str
    firstName: str

class EmployeeCreate(PersonCreate):
    bookingsHandled: Optional[List[int]] = None

class BookingCreate(BaseModel):
    arrivalDate: date
    bookingNumber: int
    handledBy: Optional[int] = None
'''

BASE = {
    "Booking": {"arrivalDate": "date", "bookingNumber": "str"},
    "Employee": {},
    "Person": {"email": "str", "firstName": "str", "phone": "str"},
    "Room": {"description": "str", "maxOccupancy": "int", "roomNumber": "str", "standardPrice": "float"},
}
CASE_M = {
    "Booking": {"arrivalDate": "date", "bookingNumber": "int"},
    "Employee": {},
    "Person": {"dateOfBirth": "date", "email": "str", "firstName": "str"},
    "Room": {"amenities": "str", "floor": "int", "maxOccupancy": "int", "roomNumber": "str",
             "standardPrice": "float"},
}


def _summary(classes: dict, optional=(), ids=None, parents=None) -> dict:
    ids = ids or {}
    built = {}
    for name, attrs in classes.items():
        cls = Class(name=name)
        props = set()
        for attr, kind in attrs.items():
            prop = Property(name=attr, type=PrimitiveDataType(kind),
                            is_optional=f"{name}.{attr}" in optional)
            if f"{name}.{attr}" in ids:
                prop.id = ids[f"{name}.{attr}"]
            props.add(prop)
        cls.attributes = props
        built[name] = cls
    generalizations = {Generalization(general=built[p], specific=built[c])
                       for c, p in (parents or {}).items()}
    return model_summary(DomainModel(name="Hotel", types=set(built.values()),
                                     generalizations=generalizations))


def _app(tmp_path, orm: str, schemas: str) -> str:
    backend = tmp_path / "app" / "backend"
    backend.mkdir(parents=True)
    (backend / "sql_alchemy.py").write_text(orm, encoding="utf-8")
    (backend / "pydantic_classes.py").write_text(schemas, encoding="utf-8")
    return str(tmp_path / "app")


def _rules(issues):
    return sorted(re.sub(r"^model conformance: | (in|to) the class diagram, but .*$", "", i)
                  + (" [orm]" if "sql_alchemy.py" in i else " [schema]") for i in issues)


def test_a_modify_run_that_ignored_the_change_is_flagged(tmp_path):
    issues = model_conformance_issues(
        _app(tmp_path, TODAY_ORM, TODAY_SCHEMAS), _summary(BASE), _summary(CASE_M))
    assert _rules(issues) == sorted(
        f"{finding} [{where}]" for where in ("orm", "schema") for finding in (
            "Booking.bookingNumber changed type str -> int",
            "Person.dateOfBirth was added",
            "Person.phone was removed",
            "Room.amenities was added",
            "Room.description was removed",
            "Room.floor was added",
        ))
    assert all(_classify_issue(i).severity == "blocker" for i in issues)


def test_a_run_that_followed_the_change_is_clean(tmp_path):
    assert model_conformance_issues(
        _app(tmp_path, FOLLOWED_ORM, FOLLOWED_SCHEMAS), _summary(BASE), _summary(CASE_M)) == []


def test_unchanged_attributes_are_never_checked(tmp_path):
    # The stale app is fine for a model change that touches nothing it defines.
    added_class = {**BASE, "Invoice": {"total": "float"}}
    assert model_conformance_issues(
        _app(tmp_path, TODAY_ORM, TODAY_SCHEMAS), _summary(BASE), _summary(added_class)) == []


def test_a_rename_matched_by_id_names_both_sides(tmp_path):
    old = _summary(BASE, ids={"Room.description": "a1"})
    new_classes = {**BASE, "Room": {**{k: v for k, v in BASE["Room"].items() if k != "description"},
                                    "amenities": "str"}}
    new = _summary(new_classes, ids={"Room.amenities": "a1"})
    issues = model_conformance_issues(_app(tmp_path, TODAY_ORM, TODAY_SCHEMAS), old, new)
    assert ("model conformance: Room.description was renamed to Room.amenities in the class "
            "diagram, but backend/sql_alchemy.py still defines description on Room.") in issues[0]
    assert len(issues) == 4  # description still there, amenities missing; ORM and schema


ORM_OPTIONAL = _HEADER + '''
class Person(Base):
    __tablename__ = "person"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    email: Mapped_[str] = mapped_column(String_(100))
    phone: Mapped_[Optional_[str]] = mapped_column(String_(100), nullable=True)
'''
SCHEMA_OPTIONAL = '''
from typing import Optional
from pydantic import BaseModel

class PersonCreate(BaseModel):
    email: str
    phone: Optional[str] = None
'''


def test_optionality_changes(tmp_path):
    old = _summary({"Person": {"email": "str", "phone": "str"}}, optional={"Person.phone"})
    new = _summary({"Person": {"email": "str", "phone": "str"}}, optional={"Person.email"})
    issues = model_conformance_issues(_app(tmp_path, ORM_OPTIONAL, SCHEMA_OPTIONAL), old, new)
    assert issues == [
        "model conformance: Person.email is now optional in the class diagram, but "
        "backend/sql_alchemy.py still declares it NOT NULL on Person (use Optional and nullable=True).",
        "model conformance: Person.email is now optional in the class diagram, but "
        "backend/pydantic_classes.py still requires it in PersonCreate.",
        # A NULL-able column for a required attribute is not flagged in the ORM:
        # the generator emits derived enums that way.
        "model conformance: Person.phone is now required in the class diagram, but "
        "backend/pydantic_classes.py still declares it Optional in PersonCreate.",
    ]


def test_followed_optionality_is_clean(tmp_path):
    old = _summary({"Person": {"email": "str", "phone": "str"}})
    new = _summary({"Person": {"email": "str", "phone": "str"}}, optional={"Person.phone"})
    assert model_conformance_issues(_app(tmp_path, ORM_OPTIONAL, SCHEMA_OPTIONAL), old, new) == []


@pytest.mark.parametrize("annotation", ["Optional[str] = None", "str | None = None",
                                        "Union[str, None] = None", "str = ''"])
def test_any_optional_spelling_or_a_default_satisfies_optional(tmp_path, annotation):
    old = _summary({"Person": {"phone": "str"}})
    new = _summary({"Person": {"phone": "str"}}, optional={"Person.phone"})
    schemas = ("from typing import Optional, Union\nfrom pydantic import BaseModel\n\n"
               f"class PersonCreate(BaseModel):\n    phone: {annotation}\n")
    assert model_conformance_issues(_app(tmp_path, ORM_OPTIONAL, schemas), old, new) == []


def test_inherited_and_relationship_definitions(tmp_path):
    # An attribute moved from Person to Employee is still reachable on Employee
    # through inheritance (only Person is flagged), and an association end
    # that took over a removed attribute's name is not that attribute.
    old = _summary({"Person": {"phone": "str"}, "Employee": {}, "Booking": {"handledBy": "str"}},
                   parents={"Employee": "Person"})
    new = _summary({"Person": {}, "Employee": {"phone": "str"}, "Booking": {}},
                   parents={"Employee": "Person"})
    new["associations"] = {"handledBy": {
        "ends": "Booking.handledBy[1..1] - Employee.bookingsHandled[0..*]"}}
    orm = TODAY_ORM.replace("    email: Mapped_[str] = mapped_column(String_(100))\n", "")
    issues = model_conformance_issues(_app(tmp_path, orm, TODAY_SCHEMAS), old, new)
    assert _rules(issues) == ["Person.phone was removed [orm]", "Person.phone was removed [schema]"]


def test_other_stacks_and_unparsable_files_are_skipped(tmp_path):
    old, new = _summary(BASE), _summary(CASE_M)
    (tmp_path / "manage.py").write_text("print('django')\n", encoding="utf-8")
    assert model_conformance_issues(str(tmp_path), old, new) == []
    broken = _app(tmp_path, "class Room(Base:\n", "class RoomCreate(\n")
    assert model_conformance_issues(broken, old, new) == []


def test_non_primitive_type_changes_are_not_judged(tmp_path):
    old = _summary({"Booking": {"bookingNumber": "str"}})
    new = _summary({"Booking": {"bookingNumber": "any"}})
    assert model_conformance_issues(_app(tmp_path, TODAY_ORM, TODAY_SCHEMAS), old, new) == []


# ----------------------------------------------------------------------
# Wiring: Phase 3 of a modify run with a model change
# ----------------------------------------------------------------------


def _orchestrator(workspace):
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker

    class _Client:
        model = "mock-model"
        max_tokens = 4096
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):  # pragma: no cover
            raise AssertionError("no LLM call expected")

    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="Hotel", types={Class(name="Room")}),
        output_dir=workspace, enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )


def test_phase3_reports_conformance_blockers_only_in_a_modify_run_with_a_change(tmp_path):
    workspace = _app(tmp_path, TODAY_ORM, TODAY_SCHEMAS)
    orch = _orchestrator(workspace)
    orch._model_change = compare_models(_summary(BASE), _summary(CASE_M))

    def conformance(issues):
        return [i for i in issues if i.message.startswith("model conformance:")]

    assert conformance(orch._collect_validation_issues()) == []  # not a modify run
    orch._modify_mode = True
    found = conformance(orch._collect_validation_issues())
    assert len(found) == 12 and {i.severity for i in found} == {"blocker"}
    orch._model_change = compare_models(_summary(BASE), _summary(BASE))
    assert conformance(orch._collect_validation_issues()) == []  # no change
