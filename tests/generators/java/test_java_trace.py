"""JavaGenerator records which lines of each .java file come from which model element."""

from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, DomainModel, Enumeration, EnumerationLiteral, IntegerType, Method,
    Multiplicity, Parameter, Property, StringType,
)
from besser.generators.java_classes import JavaGenerator


def _model(with_ids: bool) -> DomainModel:
    def ident(value):
        return value if with_ids else None

    status = Enumeration("Status", {EnumerationLiteral("OPEN", id=ident("lit-open")),
                                    EnumerationLiteral("CLOSED", id=ident("lit-closed"))}, id=ident("enum-1"))
    library = Class("Library", {Property("title", StringType, id=ident("attr-ltitle"))}, id=ident("class-lib"))
    book = Class("Book", {Property("isbn", StringType, id=ident("attr-isbn")),
                          Property("pages", IntegerType, id=ident("attr-pages")),
                          Property("status", status, id=ident("attr-status"))}, id=ident("class-book"))
    borrow = Method("borrow", parameters=[Parameter("who", StringType)], type=BooleanType, id=ident("m-borrow"))
    borrow.owner = book
    book.methods = {borrow}
    has = BinaryAssociation("has", {
        Property("library", library, multiplicity=Multiplicity(1, 1), id=ident("end-lib")),
        Property("books", book, multiplicity=Multiplicity(0, "*"), id=ident("end-books")),
    }, id=ident("assoc-has"))
    return DomainModel("Lib", types={status, library, book}, associations={has})


def _generate(model, tmp_path, name):
    out = tmp_path / name
    generator = JavaGenerator(model, output_dir=str(out), package_name="com.example")
    generator.generate()
    files = {p.name: p.read_bytes() for p in out.iterdir()}
    return generator, files


def test_each_file_names_its_element_and_regions_contain_the_element_code(tmp_path):
    generator, files = _generate(_model(with_ids=True), tmp_path, "traced")
    traces = generator.traced_files
    assert set(traces) == set(files) == {"Status.java", "Library.java", "Book.java"}

    assert [(s.element.element_id, s.rule) for s in traces["Book.java"].sources] == [
        ("class-book", "java_template.py.j2#file")]
    assert [(s.element.element_id, s.rule) for s in traces["Status.java"].sources] == [
        ("enum-1", "java_enum_template.py.j2#file")]

    seen = set()
    for path, trace in traces.items():
        lines = files[path].decode("utf-8").split("\n")
        for region in trace.regions:
            text = "\n".join(lines[region.start_line - 1:region.end_line])
            seen.add((region.element_id, region.rule.split("#")[1]))
            if region.rule.endswith("#association"):
                continue  # its lines hold its ends' names, not its own
            assert region.element_name in text, (path, region, text)
    assert seen >= {
        ("enum-1", "enum"), ("lit-open", "literal"), ("lit-closed", "literal"),
        ("class-lib", "class"), ("class-book", "class"),
        ("attr-isbn", "attribute"), ("attr-isbn", "accessors"), ("attr-status", "attribute"),
        ("end-lib", "association_end"), ("end-books", "association_end"),
        ("end-lib", "association_accessors"), ("end-books", "association_accessors"),
        ("assoc-has", "association"), ("m-borrow", "method"),
    }

    book_lines = files["Book.java"].decode("utf-8").split("\n")
    by_key = {(r.element_id, r.rule.split("#")[1]): r for r in traces["Book.java"].regions}
    assert book_lines[by_key[("class-book", "class")].start_line - 1] == "public class Book {"
    assert book_lines[by_key[("attr-isbn", "attribute")].start_line - 1].strip() == "private String isbn;"
    assert book_lines[by_key[("end-lib", "association_end")].start_line - 1].strip() == "private Library library;"
    assert book_lines[by_key[("m-borrow", "method")].start_line - 1].strip().startswith("public boolean borrow(")
    cls, attr = by_key[("class-book", "class")], by_key[("attr-isbn", "attribute")]
    assert cls.start_line < attr.start_line <= attr.end_line <= cls.end_line
    status_lines = files["Status.java"].decode("utf-8").split("\n")
    literal = next(r for r in traces["Status.java"].regions if r.element_id == "lit-open")
    assert status_lines[literal.start_line - 1].strip() == "OPEN,"


def test_tracing_does_not_change_the_generated_code(tmp_path):
    _, traced = _generate(_model(with_ids=True), tmp_path, "traced")
    untraced_generator, untraced = _generate(_model(with_ids=False), tmp_path, "untraced")
    assert traced == untraced
    assert all(b"\x02" not in content for content in traced.values())
    assert all(t.regions == () and t.sources == () for t in untraced_generator.traced_files.values())
