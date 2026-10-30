"""SQLAlchemyGenerator records which lines of sql_alchemy.py come from which model element."""

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, Multiplicity, Property, StringType,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _model(with_ids: bool) -> DomainModel:
    def ident(value):
        return value if with_ids else None

    status = Enumeration("Status", {EnumerationLiteral("OPEN"), EnumerationLiteral("CLOSED")}, id=ident("enum-1"))
    library = Class("Library", {Property("name", StringType)}, id=ident("class-1"))
    book = Class("Book", {Property("title", StringType), Property("status", status)}, id=ident("class-2"))
    has = BinaryAssociation("has", {
        Property("library", library, multiplicity=Multiplicity(1, 1)),
        Property("books", book, multiplicity=Multiplicity(0, "*")),
    }, id=ident("assoc-1"))
    shelves = BinaryAssociation("shelves", {
        Property("tagged", book, multiplicity=Multiplicity(0, "*")),
        Property("libraries", library, multiplicity=Multiplicity(0, "*")),
    }, id=ident("assoc-2"))
    return DomainModel("Library_Model", types={status, library, book}, associations={has, shelves})


def _generate(model, tmp_path, name):
    out = tmp_path / name
    generator = SQLAlchemyGenerator(model, output_dir=str(out))
    generator.generate()
    return generator, (out / "sql_alchemy.py").read_text(encoding="utf-8")


def test_trace_regions_point_at_the_generated_definitions(tmp_path):
    generator, code = _generate(_model(with_ids=True), tmp_path, "traced")
    lines = code.split("\n")
    regions = {(r.element_id, r.rule.split("#")[1]): r for r in generator.traced_files["sql_alchemy.py"].regions}

    assert lines[regions[("enum-1", "enumeration")].start_line - 1] == "class Status(enum.Enum):"
    assert lines[regions[("class-1", "class")].start_line - 1].startswith("class Library(")
    assert lines[regions[("class-2", "class")].start_line - 1].startswith("class Book(")
    assert lines[regions[("assoc-2", "association_table")].start_line - 1] == "shelves = Table_("
    fk = regions[("assoc-1", "foreign_key")]
    assert lines[fk.start_line - 1].strip().startswith("library_id:")
    # The foreign key line sits inside the class that holds it.
    book = regions[("class-2", "class")]
    assert book.start_line < fk.start_line <= book.end_line
    relationship_lines = [lines[r.start_line - 1] for r in generator.traced_files["sql_alchemy.py"].regions
                          if r.element_id == "assoc-1" and r.rule.endswith("#relationship")]
    assert any(line.startswith("Library.books") for line in relationship_lines)
    assert any(line.startswith("Book.library") for line in relationship_lines)


def test_tracing_does_not_change_the_generated_code(tmp_path):
    _, traced = _generate(_model(with_ids=True), tmp_path, "traced")
    untraced_generator, untraced = _generate(_model(with_ids=False), tmp_path, "untraced")
    assert traced == untraced
    assert untraced_generator.traced_files["sql_alchemy.py"].regions == ()
    assert "\x02" not in traced
