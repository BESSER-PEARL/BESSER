"""RDFGenerator records which lines of vocabulary.ttl come from which model element."""

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, Generalization, IntegerType,
    Multiplicity, Property, StringType,
)
from besser.generators.rdf import RDFGenerator

FILE = "vocabulary.ttl"


def _model(with_ids: bool) -> DomainModel:
    def ident(value):
        return value if with_ids else None

    status = Enumeration("Status", {EnumerationLiteral("OPEN", id=ident("lit-open")),
                                    EnumerationLiteral("CLOSED", id=ident("lit-closed"))}, id=ident("enum-1"))
    person = Class("Person", {Property("fullname", StringType, id=ident("attr-name"))}, id=ident("class-person"))
    employee = Class("Employee", {Property("salary", IntegerType, id=ident("attr-salary"))}, id=ident("class-emp"))
    library = Class("Library", {Property("title", StringType, id=ident("attr-title"))}, id=ident("class-lib"))
    book = Class("Book", {Property("state", status, id=ident("attr-state"))}, id=ident("class-book"))
    has = BinaryAssociation("has", {
        Property("library", library, multiplicity=Multiplicity(1, 1), id=ident("end-lib")),
        Property("books", book, multiplicity=Multiplicity(0, "*"), id=ident("end-books")),
    }, id=ident("assoc-has"))
    return DomainModel("Lib", types={status, person, employee, library, book}, associations={has},
                       generalizations={Generalization(general=person, specific=employee, id=ident("gen-1"))})


def _generate(model, tmp_path, name):
    out = tmp_path / name
    generator = RDFGenerator(model, output_dir=str(out))
    generator.generate()
    return generator, (out / FILE).read_bytes()


def test_regions_cover_every_traced_kind_and_contain_the_element_code(tmp_path):
    generator, code = _generate(_model(with_ids=True), tmp_path, "traced")
    lines = code.decode("utf-8").split("\n")
    regions = generator.traced_files[FILE].regions

    by_key = {(r.element_id, r.rule.split("#")[1]): r for r in regions}
    assert set(by_key) == {
        ("class-person", "class"), ("class-emp", "class"), ("class-lib", "class"), ("class-book", "class"),
        ("gen-1", "generalization"), ("attr-name", "attribute"), ("attr-salary", "attribute"),
        ("attr-title", "attribute"), ("attr-state", "attribute"), ("assoc-has", "association"),
        ("enum-1", "enumeration"), ("lit-open", "literal"), ("lit-closed", "literal"),
    }
    for region in regions:
        text = "\n".join(lines[region.start_line - 1:region.end_line])
        if region.element_type in ("Generalization", "BinaryAssociation"):
            continue  # unnamed, or its lines hold its ends' names
        assert f'"{region.element_name}"' in text or f"ex:{region.element_name} " in text, (region, text)
    assert lines[by_key[("class-person", "class")].start_line - 1] == "ex:Person rdf:type rdfs:Class ;"
    emp = lines[by_key[("gen-1", "generalization")].start_line - 1:by_key[("gen-1", "generalization")].end_line]
    assert any("rdfs:subClassOf ex:Person" in line for line in emp)
    assert lines[by_key[("attr-name", "attribute")].start_line - 1].startswith("ex:fullname rdf:type rdf:Property")
    assert lines[by_key[("assoc-has", "association")].start_line - 1].startswith("ex:books rdf:type rdf:Property")
    enum, literal = by_key[("enum-1", "enumeration")], by_key[("lit-open", "literal")]
    assert enum.start_line < literal.start_line <= literal.end_line <= enum.end_line


def test_tracing_does_not_change_the_generated_code(tmp_path):
    _, traced = _generate(_model(with_ids=True), tmp_path, "traced")
    untraced_generator, untraced = _generate(_model(with_ids=False), tmp_path, "untraced")
    assert traced == untraced
    assert untraced_generator.traced_files[FILE].regions == ()
    assert b"\x02" not in traced
