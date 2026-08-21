"""Tests that `is_id` / `is_external_id` flow through the SQLAlchemy generator.

Covers the `is_id` work merged via PR #486 and the `is_external_id` work
added on top (issues #230, #225, and #108's UNIQUE-attribute ask).

We import the generated module and verify the schema via
``sqlalchemy.inspect`` — cheaper and more reliable than regexing strings.
"""

import importlib.util
import os
import re
import sys

import pytest
from sqlalchemy import create_engine, inspect

from besser.BUML.metamodel.structural import (
    Class, DomainModel, IntegerType, Property, StringType,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _import_generated(model: DomainModel, tmpdir, module_name: str):
    """Render the generator, patch out the file-backed sqlite URL so nothing
    touches disk, and return the imported module."""
    output_dir = tmpdir.mkdir(module_name)
    SQLAlchemyGenerator(model=model, output_dir=str(output_dir)).generate(dbms="sqlite")
    path = os.path.join(str(output_dir), "sql_alchemy.py")

    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Drop the Base.metadata.create_all block — the test drives that itself.
    code = re.sub(
        r"# Database connection.*?Base\.metadata\.create_all\(engine, checkfirst=True\)",
        (
            "DATABASE_URL = 'sqlite:///:memory:'\n"
            "engine = create_engine(DATABASE_URL, echo=False)"
        ),
        code,
        flags=re.DOTALL,
    )
    patched = os.path.join(str(output_dir), "patched.py")
    with open(patched, "w", encoding="utf-8") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location(module_name, patched)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _shop_model_with_isbn_external_id() -> DomainModel:
    """Book class with surrogate PK + `isbn` as external id + `title` plain."""
    book = Class(name="Book")
    book.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="isbn", type=StringType, is_external_id=True),
        Property(name="title", type=StringType),
    }
    model = DomainModel(name="Shop")
    model.add_type(book)
    return model


def test_external_id_becomes_unique_column(tmpdir):
    """An `is_external_id` attribute must map to a column with a UNIQUE
    constraint while the `is_id` attribute remains the primary key."""
    model = _shop_model_with_isbn_external_id()
    module = _import_generated(model, tmpdir, "external_id_module")

    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)

    insp = inspect(engine)
    columns = {c["name"]: c for c in insp.get_columns("book")}

    # Primary key is 'id', not 'isbn'
    pk = insp.get_pk_constraint("book")
    assert pk["constrained_columns"] == ["id"]

    # 'isbn' carries a UNIQUE constraint, 'title' does not.
    unique_columns = {
        col
        for uq in insp.get_unique_constraints("book")
        for col in uq["column_names"]
    }
    assert "isbn" in unique_columns
    assert "title" not in unique_columns
    assert "id" not in unique_columns  # PK already implies uniqueness

    # Non-PK, non-external-id columns are still nullable=False by default
    # (they're not optional in the model).
    assert columns["title"]["nullable"] is False


def test_external_id_engine_round_trip(tmpdir):
    """Beyond schema introspection, the generated classes must actually work:
    inserting two rows with the same external id must raise IntegrityError."""
    model = _shop_model_with_isbn_external_id()
    module = _import_generated(model, tmpdir, "external_id_integrity_module")

    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)

    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.add(module.Book(id=1, isbn="978-1", title="First"))
        session.add(module.Book(id=2, isbn="978-2", title="Second"))
        session.commit()

    # Second session: reuse ISBN — must violate UNIQUE
    with Session(engine) as session:
        session.add(module.Book(id=3, isbn="978-1", title="Duplicate"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_non_default_is_id_attribute_becomes_primary_key(tmpdir):
    """Regression for PR #486: marking a non-`id`-named attribute as
    `is_id=True` must make it the primary key — not the auto-surrogate."""
    book = Class(name="Book")
    book.attributes = {
        Property(name="isbn", type=StringType, is_id=True),
        Property(name="title", type=StringType),
    }
    model = DomainModel(name="Shop")
    model.add_type(book)

    module = _import_generated(model, tmpdir, "is_id_as_pk_module")

    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)

    pk = inspect(engine).get_pk_constraint("book")
    assert pk["constrained_columns"] == ["isbn"]
