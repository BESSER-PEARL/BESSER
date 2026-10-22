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


def test_other_stacks_are_not_checked_and_unparsable_files_are_left_to_the_syntax_check(tmp_path):
    old, new = _summary(BASE), _summary(CASE_M)
    (tmp_path / "manage.py").write_text("print('django')\n", encoding="utf-8")
    assert model_conformance_issues(str(tmp_path), old, new) is None
    broken = _app(tmp_path, "class Room(Base:\n", "class RoomCreate(\n")
    assert model_conformance_issues(broken, old, new) == []


@pytest.mark.parametrize("column, flagged", [
    # SQLAlchemy: a bare Column() / mapped_column() is nullable.
    ("email = Column(String(100))", False),
    ("email = mapped_column(String(100))", False),
    ("email: Mapped['Optional[str]'] = mapped_column(String(100))", False),
    ("email: Mapped[Optional[str]] = mapped_column(String(100))", False),
    ("email: Mapped[str] = mapped_column(String(100), nullable=True)", False),
    ("email: Mapped[str] = mapped_column(String(100))", True),
    ("email = Column(String(100), nullable=False)", True),
])
def test_now_optional_is_flagged_only_on_a_column_that_is_really_not_null(tmp_path, column, flagged):
    orm = ("from typing import Optional\nfrom sqlalchemy import Column, Integer, String\n"
           "from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column\n"
           "class Base(DeclarativeBase):\n    pass\n"
           f"class Person(Base):\n    __tablename__ = 'person'\n"
           f"    id = Column(Integer, primary_key=True)\n    {column}\n")
    old = _summary({"Person": {"email": "str"}})
    new = _summary({"Person": {"email": "str"}}, optional={"Person.email"})
    issues = model_conformance_issues(_app(tmp_path, orm, "class Other:\n    pass\n"), old, new)
    assert bool(issues) is flagged


def test_a_removed_id_attribute_is_not_the_generators_primary_key(tmp_path):
    # The generator emits its own surrogate ``id`` primary key when the model has none.
    old = _summary({"Room": {"id": "int", "number": "str"}})
    new = _summary({"Room": {"number": "str"}})
    assert model_conformance_issues(_app(tmp_path, FOLLOWED_ORM, FOLLOWED_SCHEMAS), old, new) == []


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


# ----------------------------------------------------------------------
# Calibration on BESSER's own backend generator
# ----------------------------------------------------------------------

_TYPES = ("str", "int", "float", "bool", "date", "datetime", "time", "timedelta")


def _calibration_model(new: bool) -> DomainModel:
    """Every primitive type change, both optionality changes, an added,
    a removed and an inherited attribute, as the old or the new model."""
    attrs = {Property(f"t_{a}_{b}", PrimitiveDataType(b if new else a))
             for a in _TYPES for b in _TYPES if a != b}
    for kind in _TYPES:
        attrs.add(Property(f"o_{kind}_to_optional", PrimitiveDataType(kind), is_optional=new))
        attrs.add(Property(f"o_{kind}_to_required", PrimitiveDataType(kind), is_optional=not new))
    attrs |= {Property("kept", PrimitiveDataType("str")),
              Property("total", PrimitiveDataType("float"), is_derived=True),
              Property("fresh" if new else "gone", PrimitiveDataType("str"))}
    item = Class("Item", attributes=attrs)
    party = Class("Party", attributes={Property("name", PrimitiveDataType("str"))}
                  | (set() if new else {Property("moved", PrimitiveDataType("str"))}))
    client = Class("Client", attributes={Property("code", PrimitiveDataType("str"))}
                   | ({Property("moved", PrimitiveDataType("str"))} if new else set()))
    return DomainModel("Calib", types={item, party, client},
                       generalizations={Generalization(general=party, specific=client)})


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    from besser.generators.backend import BackendGenerator

    out = {}
    for label, new in (("old", False), ("new", True)):
        root = tmp_path_factory.mktemp(f"calibration_{label}")
        BackendGenerator(_calibration_model(new), output_dir=str(root / "backend")).generate()
        out[label] = str(root)
    return out


def test_generator_output_from_the_new_model_is_clean(generated):
    old, new = model_summary(_calibration_model(False)), model_summary(_calibration_model(True))
    assert model_conformance_issues(generated["new"], old, new) == []


def test_generator_output_from_the_old_model_gets_exactly_the_expected_findings(generated):
    old, new = model_summary(_calibration_model(False)), model_summary(_calibration_model(True))
    issues = model_conformance_issues(generated["old"], old, new)
    pairs = len(_TYPES) * (len(_TYPES) - 1)
    expected = {
        "changed type [orm]": pairs, "changed type [schema]": pairs,
        "is now optional [orm]": len(_TYPES), "is now optional [schema]": len(_TYPES),
        "is now required [schema]": len(_TYPES),
        # Item.gone and Party.moved; Client inherits moved from Party.
        "was removed [orm]": 2, "was removed [schema]": 2,
        "was added [orm]": 1, "was added [schema]": 1,  # Item.fresh
    }
    found = {}
    for issue in issues:
        rule = next(r for r in ("changed type", "is now optional", "is now required",
                                "was removed", "was added") if r in issue)
        key = f"{rule} [{'orm' if 'sql_alchemy.py' in issue else 'schema'}]"
        found[key] = found.get(key, 0) + 1
    assert found == expected


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


def test_a_modify_run_on_another_stack_reports_the_check_as_not_run(tmp_path):
    (tmp_path / "manage.py").write_text("print('django')\n", encoding="utf-8")
    orch = _orchestrator(str(tmp_path))
    orch._modify_mode = True
    orch._model_change = compare_models(_summary(BASE), _summary(CASE_M))
    [note] = orch._collect_model_conformance_issues()
    assert note.startswith("validation: the model conformance check did not run")
    assert _classify_issue(note).severity == "warning"
