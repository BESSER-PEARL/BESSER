"""TestCaseGenerator records which lines of test_hypothesis.py come from which model element."""

from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, Constraint, DomainModel, Enumeration, EnumerationLiteral, IntegerType,
    Method, Multiplicity, Parameter, Property, StringType,
)
from besser.generators.testgen import TestCaseGenerator

FILE = "test_hypothesis.py"


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
    post = Constraint("BorrowPost", book, "context Book::borrow(n: Integer) post: self.pages = self.pages@pre", "OCL")
    post.id = ident("ocl-post")
    borrow.post = [post]
    has = BinaryAssociation("has", {
        Property("library", library, multiplicity=Multiplicity(1, 1), id=ident("end-lib")),
        Property("books", book, multiplicity=Multiplicity(0, "*"), id=ident("end-books")),
    }, id=ident("assoc-has"))
    return DomainModel("Lib", types={status, library, book}, associations={has})


def _generate(model, tmp_path, name):
    out = tmp_path / name
    generator = TestCaseGenerator(model, output_dir=str(out))
    generator.generate()
    return generator, (out / FILE).read_bytes()


def test_regions_cover_every_traced_kind_and_contain_the_element_code(tmp_path):
    generator, code = _generate(_model(with_ids=True), tmp_path, "traced")
    lines = code.decode("utf-8").split("\n")
    regions = generator.traced_files[FILE].regions

    kinds = {(r.element_id, r.rule.split("#")[1]) for r in regions}
    assert kinds >= {
        ("class-book", "structural_tests"), ("class-book", "strategy"), ("class-book", "property_tests"),
        ("attr-isbn", "attribute_tests"), ("attr-isbn", "strategy_param"), ("attr-isbn", "setter_tests"),
        ("enum-1", "enumeration_tests"), ("lit-open", "literal"), ("lit-closed", "literal"),
        ("end-lib", "association_end_tests"), ("end-books", "association_end_tests"),
        ("assoc-has", "association"), ("m-borrow", "method_tests"), ("ocl-post", "post_condition"),
    }
    for region in regions:
        text = "\n".join(lines[region.start_line - 1:region.end_line])
        if region.rule.endswith("#association"):
            continue  # its lines hold its ends' names, not its own
        # the generated tests spell names in lower case (test_book_ocl_borrowpost)
        assert region.element_name.lower() in text.lower(), (region, text)
    by_key = {(r.element_id, r.rule.split("#")[1]): r for r in regions}
    assert lines[by_key[("class-book", "strategy")].start_line - 1] == "Book_strategy = st.builds("
    assert lines[by_key[("attr-isbn", "attribute_tests")].start_line - 1] == "def test_book_has_isbn():"
    assert lines[by_key[("ocl-post", "post_condition")].start_line - 1] == "@given("
    assert "def test_book_ocl_borrowpost(" in "\n".join(
        lines[by_key[("ocl-post", "post_condition")].start_line - 1:by_key[("ocl-post", "post_condition")].end_line])
    book, setter = by_key[("class-book", "property_tests")], by_key[("attr-isbn", "setter_tests")]
    assert book.start_line < setter.start_line <= setter.end_line <= book.end_line


def test_tracing_does_not_change_the_generated_code(tmp_path):
    _, traced = _generate(_model(with_ids=True), tmp_path, "traced")
    untraced_generator, untraced = _generate(_model(with_ids=False), tmp_path, "untraced")
    assert traced == untraced
    assert untraced_generator.traced_files[FILE].regions == ()
    assert b"\x02" not in traced
