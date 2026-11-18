"""SQLGenerator maps each CREATE TABLE / CREATE TYPE statement back to the element it was generated from.

The DDL comes from running the generated SQLAlchemy module, so the statements are matched to elements by the
table or type name the SQLAlchemy generator gave them.
"""

import re

import pytest

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, IntegerType,
    Multiplicity, Property, StringType,
)
from besser.generators.sql import SQLGenerator


def _model(with_ids: bool) -> DomainModel:
    def ident(value):
        return value if with_ids else None

    status = Enumeration("Status", {EnumerationLiteral("OPEN"), EnumerationLiteral("CLOSED")}, id=ident("enum-1"))
    # "Order" is an SQL reserved word: its table name is quoted in the DDL.
    order = Class("Order", {Property("total", IntegerType, id=ident("attr-total")),
                            Property("status", status)}, id=ident("class-order"))
    reader = Class("Reader", {Property("name", StringType)}, id=ident("class-reader"))
    book = Class("Book", {Property("title", StringType)}, id=ident("class-book"))
    placed = BinaryAssociation("placed", {
        Property("reader", reader, multiplicity=Multiplicity(1, 1)),
        Property("orders", order, multiplicity=Multiplicity(0, "*")),
    }, id=ident("assoc-placed"))
    tagged = BinaryAssociation("tagged", {
        Property("taggedBooks", book, multiplicity=Multiplicity(0, "*")),
        Property("readers", reader, multiplicity=Multiplicity(0, "*")),
    }, id=ident("assoc-tagged"))
    reads = BinaryAssociation("reads", {
        Property("rbook", book, multiplicity=Multiplicity(0, "*")),
        Property("rreader", reader, multiplicity=Multiplicity(0, "*")),
    })
    loan = AssociationClass("Loan", association=reads, attributes={Property("days", IntegerType)},
                            id=ident("class-loan"))
    return DomainModel("Shop", types={status, order, reader, book, loan}, associations={placed, tagged, reads})


def _generate(model, tmp_path, name, dialect):
    out = tmp_path / name
    generator = SQLGenerator(model, output_dir=str(out), sql_dialect=dialect)
    generator.generate()
    file_name = f"tables_{dialect}.sql"
    return generator, file_name, (out / file_name).read_bytes()


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql", "mysql", "mssql", "oracle"])
def test_each_statement_is_traced_to_its_element(tmp_path, dialect):
    generator, file_name, content = _generate(_model(with_ids=True), tmp_path, "traced", dialect)
    lines = content.decode("utf-8").split("\n")
    regions = generator.traced_files[file_name].regions

    by_id = {}
    for region in regions:
        by_id.setdefault(region.element_id, []).append(region)
    expected_tables = {"class-order": "order", "class-reader": "reader", "class-book": "book",
                       "class-loan": "loan", "assoc-tagged": "tagged"}
    for element_id, table in expected_tables.items():
        (region,) = by_id[element_id]
        assert region.rule == "sql_generator.py#create_table"
        first, last = lines[region.start_line - 1], lines[region.end_line - 1]
        assert re.match(rf'CREATE TABLE ["`\[]?{table}["`\]]? \(', first), (element_id, first)
        assert last == ");"
    assert "assoc-placed" not in by_id  # a one-to-many association creates no table of its own
    if dialect == "postgresql":
        (enum_region,) = by_id["enum-1"]  # Status is used by one column, so one CREATE TYPE
        assert enum_region.rule == "sql_generator.py#create_type"
        assert lines[enum_region.start_line - 1].startswith("CREATE TYPE status AS ENUM")
    else:
        assert "enum-1" not in by_id
    # Every region is a whole statement: they do not overlap and each starts at a CREATE.
    ordered = sorted(regions, key=lambda r: r.start_line)
    assert all(a.end_line < b.start_line for a, b in zip(ordered, ordered[1:]))
    # The statements without an id (the model's "placed" tables apart) are not traced: Order, Reader, Book,
    # Loan, tagged (+ the Status type on PostgreSQL) are the ones with ids.
    assert len(regions) == len(expected_tables) + (1 if dialect == "postgresql" else 0)


def test_tracing_does_not_change_the_generated_sql(tmp_path):
    for dialect in ("sqlite", "postgresql"):
        _, name, traced = _generate(_model(with_ids=True), tmp_path, f"traced-{dialect}", dialect)
        untraced_generator, _, untraced = _generate(_model(with_ids=False), tmp_path, f"untraced-{dialect}", dialect)
        assert [re.sub(rb"\s+", b" ", s) for s in sorted(traced.split(b"\n\n"))] == \
               [re.sub(rb"\s+", b" ", s) for s in sorted(untraced.split(b"\n\n"))]
        assert untraced_generator.traced_files[name].regions == ()


def test_a_non_ascii_name_is_written_as_utf8_and_traced(tmp_path):
    # The DDL script wrote in the platform encoding, so on Windows "Bücher" came out as cp1252 and reading the
    # file back to trace it raised UnicodeDecodeError.
    books = Class("Bücher", {Property("titel", StringType)}, id="class-buecher")
    generator, name, content = _generate(DomainModel("Bibliothek", types={books}), tmp_path, "utf8", "sqlite")
    assert 'CREATE TABLE "bücher"' in content.decode("utf-8") and b"\r\n" not in content
    (region,) = generator.traced_files[name].regions
    assert region.element_id == "class-buecher"
