"""PythonGenerator records which lines of classes.py come from which model element."""

from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, DomainModel, Enumeration, EnumerationLiteral, Generalization,
    IntegerType, Method, Multiplicity, Parameter, Property, StringType,
)
from besser.generators.python_classes import PythonGenerator


def _model(with_ids: bool) -> DomainModel:
    def ident(value):
        return value if with_ids else None

    status = Enumeration("Status", {EnumerationLiteral("OPEN", id=ident("lit-open")),
                                    EnumerationLiteral("CLOSED", id=ident("lit-closed"))}, id=ident("enum-1"))
    person = Class("Person", {Property("name", StringType, id=ident("attr-pname"))}, id=ident("class-person"))
    employee = Class("Employee", {Property("salary", IntegerType, id=ident("attr-salary"))}, id=ident("class-emp"))
    library = Class("Library", {Property("title", StringType, id=ident("attr-ltitle"))}, id=ident("class-lib"))
    book = Class("Book", {Property("isbn", StringType, id=ident("attr-isbn")),
                          Property("status", status, id=ident("attr-status"))}, id=ident("class-book"))
    borrow = Method("borrow", parameters=[Parameter("who", StringType)], type=BooleanType, id=ident("m-borrow"))
    renew = Method("renew", code="def renew(self, days: int):\n    return days + 1\n", id=ident("m-renew"))
    borrow.owner = renew.owner = book
    book.methods = {borrow, renew}
    has = BinaryAssociation("has", {
        Property("library", library, multiplicity=Multiplicity(1, 1), id=ident("end-lib")),
        Property("books", book, multiplicity=Multiplicity(0, "*"), id=ident("end-books")),
    }, id=ident("assoc-has"))
    return DomainModel("Lib", types={status, person, employee, library, book}, associations={has},
                       generalizations={Generalization(general=person, specific=employee)})


def _generate(model, tmp_path, name):
    out = tmp_path / name
    generator = PythonGenerator(model, output_dir=str(out))
    generator.generate()
    return generator, (out / "classes.py").read_bytes()


def test_regions_cover_every_traced_kind_and_contain_the_element_code(tmp_path):
    generator, code = _generate(_model(with_ids=True), tmp_path, "traced")
    lines = code.decode("utf-8").split("\n")
    regions = generator.traced_files["classes.py"].regions

    kinds = {(r.element_id, r.rule.split("#")[1]) for r in regions}
    assert kinds >= {
        ("enum-1", "enumeration"), ("lit-open", "literal"), ("lit-closed", "literal"),
        ("class-person", "class"), ("class-emp", "class"), ("class-lib", "class"), ("class-book", "class"),
        ("attr-isbn", "attribute"), ("attr-status", "attribute"), ("attr-salary", "attribute"),
        ("end-lib", "association_end"), ("end-books", "association_end"),
        ("assoc-has", "association"), ("m-borrow", "method"), ("m-renew", "method"),
    }
    assert {r.element_id for r in regions} <= {k[0] for k in kinds}
    for region in regions:
        text = "\n".join(lines[region.start_line - 1:region.end_line])
        assert region.element_name in text, (region, text)
    by_id = {(r.element_id, r.rule.split("#")[1]): r for r in regions}
    assert lines[by_id[("class-book", "class")].start_line - 1] == "class Book:"
    assert lines[by_id[("enum-1", "enumeration")].start_line - 1] == "class Status(Enum):"
    assert lines[by_id[("lit-open", "literal")].start_line - 1].strip() == 'OPEN = "OPEN"'
    assert lines[by_id[("m-renew", "method")].start_line - 1].startswith("    def renew(")
    # An attribute sits inside its class, and a method's owner is its class.
    book, isbn = by_id[("class-book", "class")], by_id[("attr-isbn", "attribute")]
    assert book.start_line < isbn.start_line <= isbn.end_line <= book.end_line
    assert by_id[("m-borrow", "method")].owner.element_id == "class-book"


def test_tracing_does_not_change_the_generated_code(tmp_path):
    _, traced = _generate(_model(with_ids=True), tmp_path, "traced")
    untraced_generator, untraced = _generate(_model(with_ids=False), tmp_path, "untraced")
    assert traced == untraced
    assert untraced_generator.traced_files["classes.py"].regions == ()
    assert b"\x02" not in traced
