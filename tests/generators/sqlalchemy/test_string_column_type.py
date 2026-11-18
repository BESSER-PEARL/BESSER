"""A ``str`` attribute holds text of any length; a key column keeps a bounded type.

Imports the generated module and checks the column types, then compiles the
DDL for PostgreSQL and MySQL, which enforce a VARCHAR length.
"""

import importlib.util
import os
import sys

from sqlalchemy import String, Text
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.schema import CreateTable

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _library_model() -> DomainModel:
    author = Class(name="Author")
    author.attributes = {
        Property(name="code", type=StringType, is_id=True),
        Property(name="name", type=StringType),
    }
    book = Class(name="Book")
    book.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="isbn", type=StringType, is_external_id=True),
        Property(name="title", type=StringType),
        Property(name="summary", type=StringType, multiplicity=Multiplicity(0, 1)),
    }
    tag = Class(name="Tag")
    tag.attributes = {Property(name="label", type=StringType, is_id=True)}
    writes = BinaryAssociation(name="Writes", ends={
        Property(name="author", type=author, multiplicity=Multiplicity(1, 1)),
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
    })
    tagged = BinaryAssociation(name="Tagged", ends={
        Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*")),
        Property(name="tagged", type=book, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Library", types={author, book, tag}, associations={writes, tagged})


def _import_generated(tmpdir, module_name: str):
    output_dir = tmpdir.mkdir(module_name)
    SQLAlchemyGenerator(model=_library_model(), output_dir=str(output_dir)).generate(dbms="sqlite")
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(str(output_dir), "sql_alchemy.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_plain_string_attributes_are_unbounded_text(tmpdir):
    module = _import_generated(tmpdir, "string_text_module")
    book = module.Base.metadata.tables["book"]
    author = module.Base.metadata.tables["author"]

    for column in (book.c.title, book.c.summary, author.c.name):
        assert isinstance(column.type, Text), f"{column} is {column.type!r}"

    ddl = str(CreateTable(book).compile(dialect=postgresql.dialect()))
    assert "title TEXT NOT NULL" in ddl
    assert "VARCHAR(100)" not in ddl


def test_key_string_columns_keep_an_indexable_length(tmpdir):
    """MySQL cannot index a TEXT column without a prefix length, so primary
    keys, unique columns and the foreign keys that reference them stay VARCHAR."""
    module = _import_generated(tmpdir, "string_key_module")

    keyed = []
    for table in module.Base.metadata.sorted_tables:
        unique_columns = {
            column.name
            for constraint in table.constraints if constraint.__class__.__name__ == "UniqueConstraint"
            for column in constraint.columns
        }
        for column in table.columns:
            if column.primary_key or column.unique or column.name in unique_columns or column.foreign_keys:
                keyed.append(column)
                assert not isinstance(column.type, Text), f"{table.name}.{column.name} is {column.type!r}"
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert "VARCHAR(100)" not in ddl

    tables = module.Base.metadata.tables
    for column in (tables["author"].c.code, tables["book"].c.isbn, tables["tag"].c.label):
        assert isinstance(column.type, String) and column.type.length == 255
    assert {f"{c.table.name}.{c.name}" for c in keyed} >= {
        "author.code", "book.isbn", "book.author_id", "tag.label", "tagged.tags",
    }
