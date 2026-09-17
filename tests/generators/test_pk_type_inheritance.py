"""A foreign key must use the PK type of the class it references.

Observed live 2026-09-17 on a generated hotel app: Person carried a string
uuid PK, Guest and Employee inherited it (joined-table inheritance, their id IS
a ForeignKey to person.id) -- but every FK pointing at Guest or Employee came
out ``Mapped_[int]``, and the matching Pydantic field came out ``guest: int``.

Consequences: the schema is internally inconsistent (SQLite creates it anyway,
so nothing complains at boot), and POST /booking/ with a real employee id
returns 422 "Input should be a valid integer". No relationship could be set
through the API at all.

Cause: the PK-type resolver looked only at a class's OWN attributes, so a
subclass -- which has no id attribute of its own -- was absent from the map and
callers fell back to the 'int' default.
"""
import re

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Generalization, Multiplicity,
    Property, StringType,
)
from besser.generators.pk_types import pk_python_types
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.structural_utils import get_pk_py_types


@pytest.fixture
def inherited_str_pk_model():
    person = Class(name="Person", attributes={
        Property(name="id", type=StringType),
        Property(name="firstName", type=StringType)})
    guest = Class(name="Guest", attributes={Property(name="guestId", type=StringType)})
    booking = Class(name="Booking", attributes={Property(name="ref", type=StringType)})
    assoc = BinaryAssociation(name="books", ends={
        Property(name="guest", type=guest, multiplicity=Multiplicity(1, 1)),
        Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*"))})
    return DomainModel(
        name="Hotel", types={person, guest, booking}, associations={assoc},
        generalizations={Generalization(general=person, specific=guest)})


def test_subclass_inherits_the_parents_pk_type(inherited_str_pk_model):
    pk = get_pk_py_types(inherited_str_pk_model)
    assert pk.get("Person") == "str"
    assert pk.get("Guest") == "str", (
        "Guest's PK IS Person's -- absent from the map means callers default it to int"
    )


def test_both_entry_points_agree(inherited_str_pk_model):
    """pk_types.py used to carry its own copy with the same blind spot."""
    assert pk_python_types(inherited_str_pk_model) == get_pk_py_types(inherited_str_pk_model)


def test_generated_foreign_key_matches_the_referenced_pk(tmp_path, inherited_str_pk_model):
    SQLAlchemyGenerator(model=inherited_str_pk_model, output_dir=str(tmp_path)).generate()
    src = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    fks = re.findall(r"(\w+_id): Mapped_\[(\w+)\] = mapped_column\(ForeignKey_\(\"(\w+)\.", src)
    assert fks, "no foreign keys generated"
    mismatched = [f for f in fks if f[2] in ("guest", "person") and f[1] != "str"]
    assert not mismatched, f"FK typed against a string PK as int: {mismatched}"


def test_integer_surrogate_still_defaults_to_int():
    """Guard against over-correcting everything into str."""
    a = Class(name="Alpha", attributes={Property(name="name", type=StringType)})
    model = DomainModel(name="Plain", types={a}, associations=set())
    assert get_pk_py_types(model).get("Alpha") is None, (
        "a class with no declared id must stay absent so callers use the int surrogate"
    )
