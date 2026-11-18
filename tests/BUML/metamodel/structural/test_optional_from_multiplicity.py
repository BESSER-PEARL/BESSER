"""An attribute whose multiplicity has a lower bound of 0 is optional.

``is_optional`` was added (#428) as the simple way to say what ``Multiplicity(0, 1)`` says. The generators read
only ``is_optional``, so ``Property(..., multiplicity=Multiplicity(0, 1))`` came out required: a ``str`` field in
the Pydantic schema and a NOT NULL column, and creating an object without it failed with 422.
"""

import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Multiplicity, Property, StringType, UNLIMITED_MAX_MULTIPLICITY,
)
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def test_lower_bound_zero_is_optional():
    assert Property(name="a", type=StringType, multiplicity=Multiplicity(0, 1)).is_optional is True
    assert Property(name="b", type=StringType,
                    multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)).is_optional is True


def test_default_multiplicity_is_not_optional():
    assert Property(name="a", type=StringType).is_optional is False
    assert Property(name="b", type=StringType, multiplicity=Multiplicity(1, 1)).is_optional is False


def test_multiplicity_set_later_counts():
    prop = Property(name="a", type=StringType)
    prop.multiplicity = Multiplicity(0, 1)
    assert prop.is_optional is True


def test_is_optional_flag_still_works():
    assert Property(name="a", type=StringType, is_optional=True).is_optional is True


def test_an_identifier_stays_required_whatever_its_multiplicity():
    # The OWL importer makes a functional key an identifier with Multiplicity(0, 1): it must keep loading,
    # and the key must stay required.
    assert Property(name="code", type=StringType, is_id=True, multiplicity=Multiplicity(0, 1)).is_optional is False
    assert Property(name="sku", type=StringType, is_external_id=True,
                    multiplicity=Multiplicity(0, 1)).is_optional is False
    prop = Property(name="code", type=StringType, multiplicity=Multiplicity(0, 1))
    prop.is_id = True
    assert prop.is_optional is False
    with pytest.raises(ValueError, match="identifier"):
        Property(name="code", type=StringType, is_id=True, is_optional=True)


def _book():
    book = Class(name="Book", attributes={
        Property(name="title", type=StringType),
        Property(name="subtitle", type=StringType, multiplicity=Multiplicity(0, 1)),
        Property(name="isbn", type=StringType, is_optional=True),
    })
    return DomainModel(name="Library", types={book})


def _line(text, name):
    return next(line.strip() for line in text.splitlines() if line.strip().startswith(name + ":"))


def test_generated_schema_and_table_make_it_optional(tmp_path):
    model = _book()
    PydanticGenerator(model, output_dir=str(tmp_path)).generate()
    SQLAlchemyGenerator(model, output_dir=str(tmp_path)).generate()
    schema = (tmp_path / "pydantic_classes.py").read_text(encoding="utf-8")
    table = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    # The same as is_optional=True gives, and title stays required.
    assert _line(schema, "subtitle") == _line(schema, "isbn").replace("isbn", "subtitle")
    assert "Optional" in _line(schema, "subtitle") and "Optional" not in _line(schema, "title")
    assert "nullable=True" in _line(table, "subtitle")
    assert "nullable=True" not in _line(table, "title")
