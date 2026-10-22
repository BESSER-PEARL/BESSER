"""``Element.id``: an optional, persistent identifier (issue #659).

It defaults to None, every class-diagram constructor accepts it, and it takes
no part in equality or hashing, so sets and dicts of elements behave as before.
"""
import copy

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, Constraint, DomainModel, Enumeration,
    EnumerationLiteral, Generalization, Method, Multiplicity, Parameter, Property, StringType,
)


def test_id_defaults_to_none():
    cls = Class(name="Book")
    assert cls.id is None
    assert Property(name="title", type=StringType).id is None
    assert Constraint(name="c", context=cls, expression="context Book inv: true", language="OCL").id is None
    assert DomainModel(name="M").id is None


def test_every_class_diagram_constructor_takes_an_id():
    book = Class(name="Book", id="c1")
    author = Class(name="Author", id="c2")
    ends = {Property(name="author", type=author, id="e1"),
            Property(name="books", type=book, multiplicity=Multiplicity(0, "*"), id="e2")}
    assoc = BinaryAssociation(name="writes", ends=ends, id="a1")
    elements = [
        book, author, assoc, *ends,
        Property(name="title", type=StringType, id="p1"),
        Method(name="renew", parameters=[Parameter(name="days", type=StringType, id="pa1")], id="m1"),
        Enumeration(name="Genre", literals={EnumerationLiteral(name="NOVEL", id="l1")}, id="en1"),
        Generalization(general=book, specific=Class(name="Novel"), id="g1"),
        AssociationClass(name="Loan", attributes=set(), association=assoc, id="ac1"),
    ]
    assert [e.id for e in elements] == ["c1", "c2", "a1", *[e.id for e in ends], "p1", "m1", "en1", "g1", "ac1"]
    assert {e.id for e in ends} == {"e1", "e2"}


def test_id_is_settable_and_copied():
    cls = Class(name="Book")
    cls.id = "c1"
    assert cls.id == "c1"
    assert copy.deepcopy(cls).id == "c1"


def test_id_does_not_change_equality_or_hashing():
    first, second = Class(name="Book", id="same"), Class(name="Book", id="same")
    assert first != second
    assert len({first, second}) == 2
    assert len({first: 1, second: 2}) == 2
