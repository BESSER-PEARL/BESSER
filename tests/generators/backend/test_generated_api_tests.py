"""The backend's ``tests`` option: the API tests it writes pass on the API it writes.

Each model below is generated with ``tests=True`` and pytest runs the generated ``tests/`` folder in a
subprocess, as a user would. A generated test that fails on a valid model would cost a coding agent turns
instead of saving them, so every model shape the payload builder knows is here: inheritance, enumerations
with defaults, optional attributes, every date and time type, one-to-one, many-to-many, self associations,
composition, association classes, string primary keys, required ends with lower bounds, OCL constraints.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from besser.BUML.metamodel.structural import (
    AssociationClass,
    BinaryAssociation,
    BooleanType,
    Class,
    Constraint,
    DateTimeType,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
    TimeType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.generator_interface import InvalidIdentifierError

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None,
    reason="running the generated API needs fastapi and httpx",
)

_SUMMARY = re.compile(r"(\d+) (passed|skipped|failed|error)")


def _many(minimum=0):
    return Multiplicity(minimum, "*")


def _one(minimum=1):
    return Multiplicity(minimum, 1)


def _assoc(name, *ends):
    return BinaryAssociation(name=name, ends=set(ends))


def _run_generated_tests(model: DomainModel, folder: Path, **options) -> tuple[dict[str, int], str]:
    """Generate the backend of ``model`` with its tests into ``folder`` and run them; the counts and output."""
    BackendGenerator(model, output_dir=str(folder), tests=True, **options).generate()
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests"],
        cwd=folder, capture_output=True, text=True, timeout=300, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = process.stdout + process.stderr
    counts = {kind: int(number) for number, kind in _SUMMARY.findall(output.splitlines()[-1] if output else "")}
    assert process.returncode == 0, output
    return counts, output


# ── models ──────────────────────────────────────────────────────────────────


def _inheritance_model() -> DomainModel:
    """An abstract class required through an association; its subclasses stand for it."""
    animal = Class(name="Animal", is_abstract=True, attributes={Property(name="name", type=StringType)})
    dog = Class(name="Dog", attributes={Property(name="breed", type=StringType)})
    cat = Class(name="Cat", attributes={Property(name="indoor", type=BooleanType)})
    owner = Class(name="Owner", attributes={Property(name="name", type=StringType)})
    vet = Class(name="Vet", attributes={Property(name="clinic", type=StringType)})
    return DomainModel(
        name="Pets", types={animal, dog, cat, owner, vet},
        associations={
            _assoc("owner_pets", Property(name="owner", type=owner, multiplicity=_one()),
                   Property(name="pets", type=animal, multiplicity=_many())),
            _assoc("vet_patients", Property(name="vets", type=vet, multiplicity=_many()),
                   Property(name="patients", type=animal, multiplicity=_many(1))),
        },
        generalizations={Generalization(general=animal, specific=dog), Generalization(general=animal, specific=cat)},
    )


def _attribute_types_model() -> DomainModel:
    """Every attribute type, an enumeration with a default, optional attributes."""
    status = Enumeration(name="Status", literals={EnumerationLiteral(name="TODO"), EnumerationLiteral(name="DONE")})
    task = Class(name="Task", attributes={
        Property(name="title", type=StringType),
        Property(name="status", type=status, default_value="TODO"),
        Property(name="priority", type=status, is_optional=True),
        Property(name="notes", type=StringType, is_optional=True),
        Property(name="due", type=DateType),
        Property(name="starts", type=DateTimeType),
        Property(name="reminder", type=TimeType),
        Property(name="budget", type=FloatType),
        Property(name="effort", type=IntegerType, is_optional=True),
        Property(name="done", type=BooleanType),
    })
    return DomainModel(name="Tasks", types={task, status})


def _one_to_one_model() -> DomainModel:
    """1:1 with both ends required, one optional end, and a required N:1."""
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    passport = Class(name="Passport", attributes={Property(name="number", type=StringType)})
    desk = Class(name="Desk", attributes={Property(name="floor", type=IntegerType)})
    return DomainModel(
        name="Identity", types={person, passport, desk},
        associations={
            _assoc("person_passport", Property(name="holder", type=person, multiplicity=_one()),
                   Property(name="passport", type=passport, multiplicity=_one())),
            _assoc("person_desk", Property(name="user", type=person, multiplicity=_one(0)),
                   Property(name="desk", type=desk, multiplicity=_one(0))),
        },
    )


def _many_to_many_model() -> DomainModel:
    """N:M with a lower bound on one side, and a required N:1 whose other end needs at least one."""
    student = Class(name="Student", attributes={Property(name="name", type=StringType)})
    course = Class(name="Course", attributes={Property(name="title", type=StringType)})
    school = Class(name="School", attributes={Property(name="name", type=StringType)})
    return DomainModel(
        name="Campus", types={student, course, school},
        associations={
            _assoc("enrolment", Property(name="students", type=student, multiplicity=_many()),
                   Property(name="courses", type=course, multiplicity=_many(2))),
            _assoc("school_courses", Property(name="school", type=school, multiplicity=_one()),
                   Property(name="offered", type=course, multiplicity=_many(1))),
        },
    )


def _self_association_model() -> DomainModel:
    """A required self association (no first object can exist) and an optional one.

    The backend generator itself writes a broken app for an N:M or a 1:1 self association (an ambiguous
    secondary join, an unknown ``<end>_id`` keyword): their tests are skipped (see the test at the end).
    """
    node = Class(name="Node", attributes={Property(name="label", type=StringType)})
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    return DomainModel(
        name="Graph", types={node, person},
        associations={
            _assoc("node_parent", Property(name="parent", type=node, multiplicity=_one()),
                   Property(name="children", type=node, multiplicity=_many())),
            _assoc("mentoring", Property(name="mentor", type=person, multiplicity=_one(0)),
                   Property(name="mentees", type=person, multiplicity=_many())),
        },
    )


def _composite_model() -> DomainModel:
    order = Class(name="Order", attributes={Property(name="code", type=StringType)})
    line = Class(name="OrderLine", attributes={Property(name="quantity", type=IntegerType)})
    customer = Class(name="Customer", attributes={Property(name="name", type=StringType)})
    return DomainModel(
        name="Shop", types={order, line, customer},
        associations={
            _assoc("order_lines", Property(name="order", type=order, multiplicity=_one(), is_composite=True),
                   Property(name="lines", type=line, multiplicity=_many(1))),
            _assoc("customer_orders", Property(name="customer", type=customer, multiplicity=_one()),
                   Property(name="orders", type=order, multiplicity=_many())),
        },
    )


def _ocl_model() -> DomainModel:
    """Constraints on one attribute (comparison, regex) and over two attributes."""
    player = Class(name="Player", attributes={
        Property(name="name", type=StringType),
        Property(name="age", type=IntegerType),
        Property(name="minSalary", type=FloatType),
        Property(name="maxSalary", type=FloatType),
        Property(name="code", type=StringType),
    })
    team = Class(name="Team", attributes={Property(name="name", type=StringType)})
    return DomainModel(
        name="League", types={player, team},
        associations={_assoc("team_players", Property(name="team", type=team, multiplicity=_one()),
                             Property(name="players", type=player, multiplicity=_many()))},
        constraints={
            Constraint(name="adult", context=player, expression="context Player inv: self.age >= 18", language="OCL"),
            Constraint(name="code_format", context=player, language="OCL",
                       expression="context Player inv: self.code.matches('[A-Z]{3}-[0-9]{4}')"),
            Constraint(name="salary_range", context=player, language="OCL",
                       expression="context Player inv: self.minSalary < self.maxSalary"),
        },
    )


def _association_class_model() -> DomainModel:
    student = Class(name="Student", attributes={Property(name="name", type=StringType)})
    course = Class(name="Course", attributes={Property(name="title", type=StringType)})
    enrolment = _assoc("enrolment", Property(name="students", type=student, multiplicity=_many()),
                       Property(name="courses", type=course, multiplicity=_many(1)))
    grade = AssociationClass(name="Grade", attributes={Property(name="score", type=IntegerType)},
                             association=enrolment)
    return DomainModel(name="Grades", types={student, course, grade}, associations={enrolment})


def _string_key_model() -> DomainModel:
    """Client-supplied primary keys, a string one, referenced by another class."""
    room = Class(name="Room", attributes={Property(name="number", type=StringType, is_id=True),
                                          Property(name="beds", type=IntegerType)})
    guest = Class(name="Guest", attributes={Property(name="email", type=StringType, is_external_id=True),
                                            Property(name="name", type=StringType)})
    stay = Class(name="Stay", attributes={Property(name="nights", type=IntegerType)})
    return DomainModel(
        name="Hotel", types={room, guest, stay},
        associations={
            _assoc("stay_room", Property(name="room", type=room, multiplicity=_one()),
                   Property(name="stays", type=stay, multiplicity=_many())),
            _assoc("stay_guest", Property(name="guest", type=guest, multiplicity=_one()),
                   Property(name="stays", type=stay, multiplicity=_many())),
        },
    )


def _required_cycle_model() -> DomainModel:
    """Two classes that each require the other: one of the foreign keys is deferred."""
    booking = Class(name="Booking", attributes={Property(name="ref", type=StringType)})
    guest = Class(name="Guest", attributes={Property(name="name", type=StringType)})
    return DomainModel(
        name="Bookings", types={booking, guest},
        associations={
            _assoc("booking_contact", Property(name="contact", type=guest, multiplicity=_one()),
                   Property(name="contact_for", type=booking, multiplicity=_one(0))),
            _assoc("guest_booking", Property(name="booking", type=booking, multiplicity=_one()),
                   Property(name="guest_of", type=guest, multiplicity=_one(0))),
        },
    )


def _inherited_key_model() -> DomainModel:
    """A client-supplied key inherited from a parent: the subclass's row is addressed by it."""
    party = Class(name="Party", attributes={Property(name="code", type=StringType, is_id=True),
                                            Property(name="name", type=StringType)})
    customer = Class(name="Customer", attributes={Property(name="tier", type=IntegerType)})
    return DomainModel(name="Crm", types={party, customer},
                       generalizations={Generalization(general=party, specific=customer)})


MODELS = {
    # Before: the update step changed the inherited key (code sorts before name), and the PUT answered 409.
    "inherited_key": (_inherited_key_model, {"passed": 2}),
    "inheritance": (_inheritance_model, {"passed": 4, "skipped": 1}),
    "attribute_types": (_attribute_types_model, {"passed": 1}),
    "one_to_one": (_one_to_one_model, {"passed": 3}),
    "many_to_many": (_many_to_many_model, {"passed": 3}),
    "self_association": (_self_association_model, {"passed": 1, "skipped": 1}),
    "composite": (_composite_model, {"passed": 3}),
    "ocl": (_ocl_model, {"passed": 2}),
    "association_class": (_association_class_model, {"passed": 2, "skipped": 1}),
    "string_key": (_string_key_model, {"passed": 3}),
    "required_cycle": (_required_cycle_model, {"passed": 2}),
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_generated_tests_pass(name, tmp_path):
    build, expected = MODELS[name]
    counts, output = _run_generated_tests(build(), tmp_path)
    assert counts == expected, output


@pytest.mark.parametrize("fixture", ["library_book_author_model", "employee_self_assoc_model",
                                     "simple_library_book_model", "player_team_domain_model"])
def test_generated_tests_pass_on_the_shared_fixtures(fixture, request, tmp_path):
    counts, output = _run_generated_tests(request.getfixturevalue(fixture), tmp_path)
    assert counts.get("passed", 0) >= 1 and "failed" not in counts, output


def _unsatisfiable_model() -> DomainModel:
    """A constraint no value satisfies, on a class another class requires."""
    gauge = Class(name="Gauge", attributes={Property(name="level", type=IntegerType)})
    reading = Class(name="Reading", attributes={Property(name="value", type=FloatType)})
    return DomainModel(
        name="Gauges", types={gauge, reading},
        associations={_assoc("gauge_readings", Property(name="gauge", type=gauge, multiplicity=_one()),
                             Property(name="readings", type=reading, multiplicity=_many()))},
        constraints={Constraint(name="impossible", context=gauge, language="OCL",
                                expression="context Gauge inv: self.level > 5 and self.level < 3")},
    )


def test_a_class_without_a_valid_payload_gets_a_skipped_test_that_says_why(tmp_path):
    counts, output = _run_generated_tests(_unsatisfiable_model(), tmp_path)

    assert counts == {"skipped": 2}, output
    assert "no valid Gauge payload: no value found that satisfies the OCL constraint 'impossible' of Gauge" in output
    assert "no valid Reading payload: no value found that satisfies the OCL constraint 'impossible'" in output


def test_the_tests_use_only_the_generated_http_methods(tmp_path, library_book_author_model):
    counts, output = _run_generated_tests(library_book_author_model, tmp_path, http_methods=["GET", "POST"])

    assert counts == {"passed": 3}, output
    text = (tmp_path / "tests" / "test_generated_api.py").read_text(encoding="utf-8")
    assert "client.put(" not in text and "client.delete(" not in text


def test_a_model_name_cannot_end_a_string_of_the_generated_code(tmp_path):
    """The model's name goes inside strings of the generated code. Before: an editor project titled
    Library\"\"\"<newline>code<newline>\"\"\" put that code at the top of tests/test_generated_api.py, and
    `besser apply` ran it with the tests."""
    model = _ocl_model()
    model.name = 'Library"""\n__import__("pathlib").Path("PWNED").touch()\n"""'

    with pytest.raises(InvalidIdentifierError, match="quote, a backslash or a line break"):
        BackendGenerator(model, output_dir=str(tmp_path / "out"), tests=True).generate()

    assert not list(tmp_path.rglob("test_generated_api.py"))


def test_the_tests_are_the_same_bytes_for_the_same_model(tmp_path):
    for folder in ("a", "b"):
        BackendGenerator(_ocl_model(), output_dir=str(tmp_path / folder), tests=True).generate()
        BackendGenerator(_inheritance_model(), output_dir=str(tmp_path / folder / "pets"), tests=True).generate()

    for name in ("tests/conftest.py", "tests/test_generated_api.py", "pets/tests/test_generated_api.py"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_without_the_option_nothing_changes(tmp_path, library_book_author_model):
    BackendGenerator(library_book_author_model, output_dir=str(tmp_path / "plain")).generate()
    BackendGenerator(library_book_author_model, output_dir=str(tmp_path / "tested"), tests=True).generate()

    plain = {p.relative_to(tmp_path / "plain") for p in (tmp_path / "plain").rglob("*") if p.is_file()}
    tested = {p.relative_to(tmp_path / "tested") for p in (tmp_path / "tested").rglob("*") if p.is_file()}
    assert not (tmp_path / "plain" / "tests").exists()
    assert tested - plain == {Path("tests/conftest.py"), Path("tests/test_generated_api.py")}
    for rel in plain:
        assert (tmp_path / "plain" / rel).read_bytes() == (tmp_path / "tested" / rel).read_bytes(), rel


def _generator_limit_model(ends, subclass: bool) -> DomainModel:
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    student = Class(name="Student", attributes={Property(name="school", type=StringType)})
    classes = {"Person": person, "Student": student}
    association = _assoc("link", *(Property(name=role, type=classes[cls], multiplicity=multiplicity)
                                   for role, cls, multiplicity in ends))
    return DomainModel(name="People", types={person, student} if subclass else {person},
                       associations={association},
                       generalizations={Generalization(general=person, specific=student)} if subclass else set())


# Valid models the backend generator writes an app for that does not run: an N:M self association, and a
# to-one association between a class and its parent or child, break every mapper; a 1:1 self association
# breaks POST of its class and of its subclasses.
@pytest.mark.parametrize("ends, subclass, expected, reason", [
    ([("friends", "Person", _many()), ("friend_of", "Person", _many())], False, {"skipped": 1},
     "the many-to-many self-association 'link' of Person"),
    ([("mentor", "Person", _one(0)), ("mentee", "Person", _one(0))], True, {"skipped": 2},
     "the one-to-one self-association 'link' of Person"),
    ([("tutor", "Person", _one(0)), ("pupils", "Student", _many())], True, {"skipped": 2},
     "the association 'link' between Student and Person"),
    ([("head", "Student", _one(0)), ("members", "Person", _many())], True, {"skipped": 2},
     "the association 'link' between Person and Student"),
], ids=["many_to_many_self", "one_to_one_self", "subclass_to_parent", "parent_to_subclass"])
def test_a_shape_the_backend_generator_breaks_is_skipped_naming_the_generator(ends, subclass, expected, reason,
                                                                              tmp_path):
    counts, output = _run_generated_tests(_generator_limit_model(ends, subclass), tmp_path)

    assert counts == expected, output
    assert reason in output and "A limitation of the generator, not of the model" in output


def test_deleting_an_object_linked_to_its_own_parent_class_expects_what_the_api_answers(tmp_path):
    """A many-to-many between a class and its parent: the subclass object is created with the parent end
    it requires, and no linked object needs it, so its delete succeeds (it was expected to be refused)."""
    ends = [("members", "Student", _many()), ("mentors", "Person", _many(1))]
    counts, output = _run_generated_tests(_generator_limit_model(ends, subclass=True), tmp_path)

    assert counts == {"passed": 2}, output


@pytest.mark.parametrize("abstract, expected", [
    ((False, False), {"skipped": 3}),  # two parent tables: 'multiple mapped bases'
    ((True, True), {"skipped": 3}),  # two abstract bases: their mappings stay pending
    ((True, False), {"passed": 2, "skipped": 1}),  # one of each: the generated app runs
], ids=["two_tables", "two_abstract_bases", "one_of_each"])
def test_a_class_with_two_parents_the_generator_cannot_map_is_skipped(abstract, expected, tmp_path):
    first = Class(name="Person", is_abstract=abstract[0], attributes={Property(name="name", type=StringType)})
    second = Class(name="Worker", is_abstract=abstract[1], attributes={Property(name="wage", type=FloatType)})
    student = Class(name="Student", attributes={Property(name="school", type=StringType)})
    model = DomainModel(name="People", types={first, second, student}, generalizations={
        Generalization(general=first, specific=student), Generalization(general=second, specific=student)})

    counts, output = _run_generated_tests(model, tmp_path)

    assert counts == expected, output
    if "passed" not in expected:
        assert "Student specializes Person, Worker: the backend generator maps a class onto one parent" in output
