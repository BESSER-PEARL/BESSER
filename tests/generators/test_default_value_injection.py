"""``default_value`` must never reach generated source as an expression.

It arrives unvalidated from request JSON (``class_diagram_processor`` passes
``attr.get("defaultValue")`` straight to ``Property``) and the templates used to
interpolate it raw. On the SQLAlchemy path that was directly exploitable:
``SQLGenerator`` executes the generated module in a subprocess to dump DDL, so
``__import__("os").system(...)`` in a defaultValue ran as the backend user.

These tests are the guard rail. They assert the same thing for every template
that renders a default: a hostile value either raises at generation time or is
emitted as an inert literal, and legitimate defaults still work.
"""
import re

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, Property,
    PrimitiveDataType,
)
from besser.generators.default_literals import (
    InvalidDefaultValueError, enum_default, python_default,
)

PAYLOAD = '__import__("os").system("touch /tmp/pwned")'


# --------------------------------------------------------------------------- #
# The helpers themselves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("type_name", ["int", "float", "bool"])
def test_expression_payload_is_rejected_for_numeric_types(type_name):
    with pytest.raises(InvalidDefaultValueError):
        python_default(PAYLOAD, type_name)


def test_expression_payload_becomes_an_inert_string_for_str():
    out = python_default(PAYLOAD, "str")
    assert eval(out) == PAYLOAD  # a literal, evaluating to the text itself
    assert "__import__" in out and out.startswith(("'", '"'))


def test_a_quote_cannot_escape_the_literal():
    hostile = 'x" + __import__("os").getcwd() + "'
    out = python_default(hostile, "str")
    assert eval(out) == hostile


def test_unknown_types_fall_back_to_a_quoted_literal():
    # date/datetime/custom: only a literal is safe to write into source.
    assert eval(python_default("2020-01-01", "date")) == "2020-01-01"
    with pytest.raises(InvalidDefaultValueError):
        python_default("   ", "date")


def test_legitimate_values_round_trip():
    assert eval(python_default("42", "int")) == 42
    assert eval(python_default("3.5", "float")) is not None
    assert eval(python_default("true", "bool")) is True
    assert eval(python_default("Untitled", "str")) == "Untitled"


def test_enum_member_must_be_an_identifier():
    with pytest.raises(InvalidDefaultValueError):
        enum_default('OPEN) or __import__("os").getcwd()', "Status")
    with pytest.raises(InvalidDefaultValueError):
        enum_default("NOT_A_MEMBER", "Status", members=["OPEN", "CLOSED"])
    assert enum_default("OPEN", "Status", members=["OPEN"]) == "Status.OPEN"
    assert enum_default("Status.OPEN", "Status", members=["OPEN"]) == "Status.OPEN"


# --------------------------------------------------------------------------- #
# End to end, through the generators that render a default
# --------------------------------------------------------------------------- #
def _model_with_default(type_name, default):
    attr = Property(name="field", type=PrimitiveDataType(type_name))
    attr.default_value = default
    book = Class(name="Book")
    book.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True), attr}
    return DomainModel(name="M", types={book})


def _generate(generator_cls, model, out):
    generator_cls(model=model, output_dir=str(out)).generate()
    return "\n".join(
        p.read_text(encoding="utf-8") for p in out.rglob("*.py")
    )


@pytest.mark.parametrize("type_name", ["int", "float"])
def test_sqlalchemy_rejects_an_expression_default(type_name, tmp_path):
    from besser.generators.sql_alchemy import SQLAlchemyGenerator
    with pytest.raises(InvalidDefaultValueError):
        _generate(SQLAlchemyGenerator, _model_with_default(type_name, PAYLOAD), tmp_path)


@pytest.mark.parametrize("type_name", ["int", "float"])
def test_pydantic_rejects_an_expression_default(type_name, tmp_path):
    from besser.generators.pydantic_classes import PydanticGenerator
    with pytest.raises(InvalidDefaultValueError):
        _generate(PydanticGenerator, _model_with_default(type_name, PAYLOAD), tmp_path)


@pytest.mark.parametrize(
    "generator_path, generator_name",
    [
        ("besser.generators.sql_alchemy", "SQLAlchemyGenerator"),
        ("besser.generators.pydantic_classes", "PydanticGenerator"),
    ],
)
def test_a_str_default_is_emitted_as_a_literal(generator_path, generator_name, tmp_path):
    import importlib
    gen = getattr(importlib.import_module(generator_path), generator_name)
    code = _generate(gen, _model_with_default("str", PAYLOAD), tmp_path)
    # The payload text may appear -- inside a quoted literal -- but never as a
    # call the interpreter would make.
    assert not re.search(r"=\s*__import__", code)
    assert compile(code, "<generated>", "exec")


def test_generated_sqlalchemy_default_is_still_correct(tmp_path):
    from besser.generators.sql_alchemy import SQLAlchemyGenerator
    code = _generate(SQLAlchemyGenerator, _model_with_default("int", "42"), tmp_path)
    assert "default=42" in code


def test_generated_enum_default_survives(tmp_path):
    from besser.generators.sql_alchemy import SQLAlchemyGenerator
    status = Enumeration(
        name="Status",
        literals={EnumerationLiteral(name="OPEN"), EnumerationLiteral(name="CLOSED")},
    )
    attr = Property(name="state", type=status)
    attr.default_value = "OPEN"
    book = Class(name="Book")
    book.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True), attr}
    model = DomainModel(name="M", types={book, status})
    code = _generate(SQLAlchemyGenerator, model, tmp_path)
    assert "default=Status.OPEN" in code


# --------------------------------------------------------------------------- #
# Association classes render their attributes through a SECOND loop
# --------------------------------------------------------------------------- #
def _assoc_model(default):
    """Student --Enrolment--> Course, with a defaulted attribute on the
    association class itself."""
    from besser.BUML.metamodel.structural import (
        AssociationClass, BinaryAssociation, Multiplicity,
    )
    INT = PrimitiveDataType("int")
    student = Class(name="Student", attributes={Property(name="id", type=INT, is_id=True)})
    course = Class(name="Course", attributes={Property(name="id", type=INT, is_id=True)})
    assoc = BinaryAssociation(name="enrolment", ends={
        Property(name="student", type=student, multiplicity=Multiplicity(1, 1)),
        Property(name="course", type=course, multiplicity=Multiplicity(1, 1)),
    })
    grade = Property(name="grade", type=INT)
    grade.default_value = default
    enrolment = AssociationClass(name="Enrolment", attributes={grade}, association=assoc)
    return DomainModel(
        name="M", types={student, course, enrolment}, associations={assoc},
    )


def test_association_class_defaults_are_not_a_second_sink(tmp_path):
    """The regression this case exists for.

    ``sql_alchemy_template.py.j2`` renders association-class attributes in its
    own loop rather than through ``helpers.py.j2``'s macro. Hardening the macro
    left that copy interpolating raw, so an AssociationClass attribute still
    emitted ``default=__import__("os").getcwd()`` into a module SQLGenerator
    executes — the fix covered one call site and not the other.
    """
    from besser.generators.sql_alchemy import SQLAlchemyGenerator
    with pytest.raises(InvalidDefaultValueError, match="Enrolment.grade"):
        _generate(SQLAlchemyGenerator, _assoc_model(PAYLOAD), tmp_path)


def test_a_legitimate_association_class_default_still_renders(tmp_path):
    from besser.generators.sql_alchemy import SQLAlchemyGenerator
    code = _generate(SQLAlchemyGenerator, _assoc_model("5"), tmp_path)
    assert "default=5" in code
    assert not re.search(r"=\s*__import__", code)
