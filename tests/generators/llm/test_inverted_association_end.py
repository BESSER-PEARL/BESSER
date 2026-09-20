"""An association end read off the class at the wrong side of the relation.

Live Qwen run ``Qwen-Qwen3-30B-A3B-Instruct-2507-14h282p0`` answered
``POST /booking/{id}/methods/produceBill/`` with
``AttributeError: 'Booking' object has no attribute 'bookingsHandled'``.
The scaffold had the relation right in every layer -- ``Booking.handledBy``
on the ORM class, ``Employee.bookingsHandled`` on the other side -- and the
LLM-authored method body read it the wrong way round. The serialized model
in the system prompt states the ownership outright
(``{"role": "bookingsHandled", "class": "Booking", "owner": "Employee"}``),
so the fix is enforcement, not another sentence of instruction.

Calibration is the point of this file. Every "silent" test below is a shape
that produced a finding on a probed-WORKING corpus app before the guard it
pins was added, so they are regression tests for false positives, not
hypotheticals.
"""

import os

from besser.BUML.metamodel.structural import (
    AssociationClass,
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Multiplicity,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.contract_checks import (
    build_data_contract,
    collect_inverted_end_issues,
)
from besser.generators.llm.validation.issues import _classify_issue

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")

ONE = Multiplicity(1, 1)
MANY = Multiplicity(0, 9999)


def _hotel_model() -> DomainModel:
    """Booking--Employee and Booking--Guest, as the live hotel case has them.

    ``handledBy`` is named on the Employee end, so it is a property of
    Booking; ``bookingsHandled`` is named on the Booking end, so it is a
    property of Employee.
    """
    person = Class(name="Person")
    person.attributes = {Property(name="email", type=StringType)}
    employee = Class(name="Employee")
    guest = Class(name="Guest")
    booking = Class(name="Booking")
    booking.attributes = {Property(name="bookingNumber", type=StringType)}

    handled_by = BinaryAssociation(name="handledBy", ends={
        Property(name="bookingsHandled", type=booking, multiplicity=MANY),
        Property(name="handledBy", type=employee, multiplicity=ONE),
    })
    stays = BinaryAssociation(name="stays", ends={
        Property(name="bookingsStayedOn", type=booking, multiplicity=MANY),
        Property(name="guests", type=guest, multiplicity=MANY),
    })
    # Person owns `bookings`, so Guest and Employee inherit it.
    contact = BinaryAssociation(name="contact", ends={
        Property(name="bookings", type=booking, multiplicity=MANY),
        Property(name="contact", type=person, multiplicity=ONE),
    })
    model = DomainModel(
        name="Hotel",
        types={person, employee, guest, booking},
        associations={handled_by, stays, contact},
        generalizations={Generalization(general=person, specific=employee),
                         Generalization(general=person, specific=guest)},
    )
    return model


SCAFFOLD = '''\
from sqlalchemy.orm import Mapped, relationship

class Booking(Base):
    __tablename__ = "booking"
    bookingNumber: Mapped[str]
    handledBy_id: Mapped[int]

class Person(Base):
    __tablename__ = "person"
    email: Mapped[str]

class Employee(Person):
    __tablename__ = "employee"

class Guest(Person):
    __tablename__ = "guest"

Booking.handledBy: Mapped["Employee"] = relationship(
    "Employee", back_populates="bookingsHandled")
Employee.bookingsHandled: Mapped[list["Booking"]] = relationship(
    "Booking", back_populates="handledBy")
'''


def _app(tmp_path, **files) -> str:
    """A generated backend: the scaffold plus whatever the LLM wrote."""
    backend = tmp_path / "backend"
    backend.mkdir(parents=True, exist_ok=True)
    (backend / "sql_alchemy.py").write_text(SCAFFOLD, encoding="utf-8")
    for name, source in files.items():
        (backend / f"{name}.py").write_text(source, encoding="utf-8")
    return str(tmp_path)


def _issues(tmp_path, model=None, **files) -> list:
    contract = build_data_contract(model if model is not None else _hotel_model())
    return collect_inverted_end_issues(_app(tmp_path, **files), contract)


def _blockers(issues) -> list:
    return [i for i in issues if _classify_issue(i).severity == "blocker"]


# --------------------------------------------------------------------------- #
# The live defect
# --------------------------------------------------------------------------- #

def test_reads_an_end_off_the_class_at_the_far_side(tmp_path):
    """The exact shape of Qwen run ...14h282p0's produceBill failure."""
    issues = _issues(tmp_path, booking_methods='''\
from sql_alchemy import Booking

def produce_bill(booking_id, database):
    _booking_object = database.query(Booking).filter(Booking.id == booking_id).first()
    return f"BIL{len(_booking_object.bookingsHandled) + 1:04d}"
''')
    assert len(issues) == 1, issues
    [finding] = issues
    assert "backend/booking_methods.py line 5" in finding
    # The finding has to name the class that actually carries the end, and
    # the one the receiver carries instead - a bare "wrong" is unactionable.
    assert "`bookingsHandled` is a property of `Employee`, not `Booking`" in finding
    assert "`Booking` carries `handledBy`" in finding
    assert _classify_issue(finding).severity == "blocker"


def test_constructor_keyword_for_the_other_side_is_a_blocker(tmp_path):
    """`Booking(bookingsHandled=...)` dies before any attribute is read.

    Corpus run ...2507-2_3evtiw fails here, one line before the attribute
    read the probe reported.
    """
    issues = _issues(tmp_path, seed='''\
from sql_alchemy import Booking

def seed(database):
    database.add(Booking(bookingNumber="B1", bookingsHandled=[]))
''')
    assert len(_blockers(issues)) == 1, issues
    assert "is constructed with `bookingsHandled=`" in issues[0]


def test_navigating_from_the_owner_is_silent(tmp_path):
    """The same two roles, each read off the class that owns it."""
    assert _issues(tmp_path, ok='''\
from sql_alchemy import Booking, Employee

def report(employee_id, booking_id, database):
    employee = database.query(Employee).filter(Employee.id == employee_id).one()
    booking = database.query(Booking).filter(Booking.id == booking_id).first()
    return len(employee.bookingsHandled), booking.handledBy
''') == []


# --------------------------------------------------------------------------- #
# Calibration: shapes that must stay silent
# --------------------------------------------------------------------------- #

def test_a_rename_the_whole_app_agrees_on_is_silent(tmp_path):
    """Six corpus apps rewrote the scaffold's end name; two of them work.

    The model says `bookingsHandled` belongs to Employee, but this app
    declares it on Booking and reads it from there. Nothing raises, so this
    is a naming disagreement with the model, not a defect.
    """
    tmp_path.joinpath("backend").mkdir(parents=True)
    tmp_path.joinpath("backend", "sql_alchemy.py").write_text(
        SCAFFOLD + '\nBooking.bookingsHandled: Mapped[list["Booking"]] = '
                   'relationship("Booking")\n',
        encoding="utf-8")
    tmp_path.joinpath("backend", "use.py").write_text('''\
from sql_alchemy import Booking

def count(booking_id, database):
    booking = database.query(Booking).filter(Booking.id == booking_id).first()
    return len(booking.bookingsHandled)
''', encoding="utf-8")
    contract = build_data_contract(_hotel_model())
    assert collect_inverted_end_issues(str(tmp_path), contract) == []


def test_backref_declares_the_far_side(tmp_path):
    """`backref=` puts the member on the class at the OTHER end.

    Reading it there is correct even though no `Employee.x` assignment
    exists anywhere in the tree.
    """
    tmp_path.joinpath("backend").mkdir(parents=True)
    tmp_path.joinpath("backend", "sql_alchemy.py").write_text('''\
from sqlalchemy.orm import relationship

class Booking(Base):
    handledBy = relationship("Employee", backref="bookingsHandled")

class Employee(Base):
    pass
''', encoding="utf-8")
    tmp_path.joinpath("backend", "use.py").write_text('''\
from sql_alchemy import Booking

def count(booking_id, database):
    booking = database.query(Booking).filter(Booking.id == booking_id).first()
    return booking.handledBy
''', encoding="utf-8")
    contract = build_data_contract(_hotel_model())
    assert collect_inverted_end_issues(str(tmp_path), contract) == []


def test_an_end_inherited_from_a_parent_is_silent(tmp_path):
    """Person owns `bookings`; Guest is a Person, so `guest.bookings` is fine."""
    assert _issues(tmp_path, use='''\
from sql_alchemy import Guest

def show(guest_id, database):
    guest = database.query(Guest).filter(Guest.id == guest_id).first()
    return guest.bookings
''') == []


def test_an_association_class_receiver_is_skipped(tmp_path):
    """The scaffold names a link class's navigation after the participants.

    `ReservedRoom.bookings` collides with a real role name and is not an
    inversion; an association class is never a receiver.
    """
    booking = Class(name="Booking")
    room = Class(name="Room")
    rooms = BinaryAssociation(name="rooms", ends={
        Property(name="bookings", type=booking, multiplicity=MANY),
        Property(name="rooms", type=room, multiplicity=MANY),
    })
    link = AssociationClass(name="ReservedRoom", attributes=set(), association=rooms)
    model = DomainModel(name="Hotel", types={booking, room, link},
                        associations={rooms})
    assert _issues(tmp_path, model=model, use='''\
from sql_alchemy import ReservedRoom

def show(link_id, database):
    link = database.query(ReservedRoom).filter(ReservedRoom.id == link_id).first()
    return link.bookings, link.rooms
''') == []


def test_a_role_two_associations_share_is_skipped(tmp_path):
    """`owner` named on two associations has no single owning class."""
    account = Class(name="Account")
    project = Class(name="Project")
    person = Class(name="Person")
    model = DomainModel(name="Shared", types={account, project, person},
                        associations={
                            BinaryAssociation(name="a", ends={
                                Property(name="owner", type=person, multiplicity=ONE),
                                Property(name="accounts", type=account, multiplicity=MANY),
                            }),
                            BinaryAssociation(name="b", ends={
                                Property(name="owner", type=person, multiplicity=ONE),
                                Property(name="projects", type=project, multiplicity=MANY),
                            }),
                        })
    assert _issues(tmp_path, model=model, use='''\
from sql_alchemy import Account

def show(account_id, database):
    account = database.query(Account).filter(Account.id == account_id).first()
    return account.owner
''') == []


def test_a_self_association_is_silent(tmp_path):
    """Both ends are the same class, so neither can be on the wrong one."""
    employee = Class(name="Employee")
    model = DomainModel(name="Org", types={employee}, associations={
        BinaryAssociation(name="reportsTo", ends={
            Property(name="manager", type=employee, multiplicity=ONE),
            Property(name="reports", type=employee, multiplicity=MANY),
        })})
    assert _issues(tmp_path, model=model, use='''\
from sql_alchemy import Employee

def show(employee_id, database):
    employee = database.query(Employee).filter(Employee.id == employee_id).first()
    return employee.manager, employee.reports
''') == []


def test_a_dto_receiver_is_not_a_model_class(tmp_path):
    """`BookingCreate` is the request schema, not the entity.

    Its field list is the API contract, which python_source owns; typing it
    as `Booking` produced false findings on 14 corpus apps.
    """
    assert _issues(tmp_path, pydantic_classes='''\
from pydantic import BaseModel, field_validator

class BookingCreate(BaseModel):
    bookingsHandled: list = []

    @field_validator("bookingsHandled")
    def check(self):
        return self.bookingsHandled
''') == []


# --------------------------------------------------------------------------- #
# Severity: what the fix loop is allowed to spend turns on
# --------------------------------------------------------------------------- #

def test_a_read_through_a_navigated_end_is_advisory(tmp_path):
    """`order.placedBy.warehouse` - right about the code, silent in the probe.

    Three corpus apps read an end off an expression rather than a variable
    (`db_order.placedBy.warehouse`, inventory). Two of the three PASS their
    live probe: the read is genuinely wrong, but it sits on a path nothing
    exercises. It is reported and must not spend fix turns.
    """
    issues = _issues(tmp_path, use='''\
from sql_alchemy import Booking

def show(booking_id, database):
    booking = database.query(Booking).filter(Booking.id == booking_id).first()
    return booking.handledBy.guests
''')
    assert len(issues) == 1, issues
    assert "`booking.handledBy` is a `Employee` here" in issues[0]
    assert _classify_issue(issues[0]).severity == "warning"
    assert _blockers(issues) == []


def test_a_many_valued_end_is_not_navigated_through(tmp_path):
    """`booking.guests` is a list, so `.x` on it is a different defect."""
    assert _issues(tmp_path, use='''\
from sql_alchemy import Employee

def show(employee_id, database):
    employee = database.query(Employee).filter(Employee.id == employee_id).one()
    return employee.bookingsHandled.handledBy
''') == []


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #

def test_no_model_and_no_associations_are_no_ops(tmp_path):
    app = _app(tmp_path)
    assert collect_inverted_end_issues(app, None) == []
    bare = DomainModel(name="Bare", types={Class(name="Booking")})
    assert collect_inverted_end_issues(app, build_data_contract(bare)) == []


def test_findings_come_back_in_source_order(tmp_path):
    """ast.walk order is not line order and the issue list is compared."""
    issues = _issues(tmp_path, use='''\
from sql_alchemy import Booking

def a(booking_id, database):
    booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if booking.bookingsHandled:
        pass
    if booking.bookingsHandled is None:
        pass
    return booking.bookingsHandled
''')
    lines = [int(i.split(" line ")[1].split(":")[0]) for i in issues]
    assert lines == sorted(lines) and len(lines) == 3, issues


def test_a_syntactically_broken_file_is_left_to_python_source(tmp_path):
    assert _issues(tmp_path, broken="def f(:\n") == []


def test_the_sweep_ignores_the_run_metadata_files(tmp_path):
    app = _app(tmp_path)
    os.makedirs(os.path.join(app, "node_modules"), exist_ok=True)
    with open(os.path.join(app, "node_modules", "x.py"), "w", encoding="utf-8") as fh:
        fh.write('''\
from sql_alchemy import Booking

def f(database):
    booking = database.query(Booking).first()
    return booking.bookingsHandled
''')
    assert collect_inverted_end_issues(app, build_data_contract(_hotel_model())) == []
