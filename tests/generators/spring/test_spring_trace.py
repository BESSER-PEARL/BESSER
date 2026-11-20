"""SpringBackendGenerator records which lines of its Java files come from which model element."""

import os

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, IntegerType, Method, Multiplicity,
    PrimitiveDataType, Property, StringType,
)
from besser.BUML.metamodel.traceability.traceability import model_elements
from besser.generators.spring import SpringBackendGenerator

JAVA = "src/main/java/com/example"


def _model(with_ids: bool) -> DomainModel:
    genre = Enumeration("Genre", {EnumerationLiteral("NOVEL"), EnumerationLiteral("ESSAY")})
    library = Class("Library", {Property("id", IntegerType, is_id=True), Property("name", StringType)})
    book = Class("Book", {Property("isbn", StringType, is_id=True), Property("title", StringType),
                          Property("genre", genre)},
                 methods={Method("summary", type=StringType, code='return this.title;')})
    holds = BinaryAssociation("holds", {Property("library", library, multiplicity=Multiplicity(1, 1)),
                                        Property("books", book, multiplicity=Multiplicity(0, "*"))})
    model = DomainModel("Library", types={genre, library, book}, associations={holds})
    if with_ids:  # as the editor gives them
        # The primitive types are shared by every model: they keep no id.
        for count, element in enumerate(e for e in model_elements(model) if not isinstance(e, PrimitiveDataType)):
            element.id = f"el-{count}"
    return model


def _read_all(folder) -> dict:
    files = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                files[os.path.relpath(path, folder).replace(os.sep, "/")] = f.read()
    return files


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    out = tmp_path_factory.mktemp("spring_trace")
    generator = SpringBackendGenerator(_model(with_ids=True), output_dir=str(out / "traced"))
    generator.generate()
    return generator, _read_all(out / "traced"), out


def _lines(files, path):
    return files[path].decode("utf-8").split("\n")


def _named(generator, path, label):
    return {r.element_name: r for r in generator.traced_files[path].regions if r.rule.endswith(f"#{label}")}


def test_entities_have_regions_for_attributes_methods_and_relationships(traced):
    generator, files, _ = traced
    path = f"{JAVA}/entity/Book.java"
    lines = _lines(files, path)
    sources = generator.traced_files[path].sources
    assert [(s.rule, s.element.element_name) for s in sources] == [("entity.java.j2#class", "Book")]

    attributes = _named(generator, path, "attribute")
    assert set(attributes) == {"isbn", "title", "genre"}
    assert "@Id" in lines[attributes["isbn"].start_line - 1]
    assert "String isbn;" in lines[attributes["isbn"].end_line - 1]
    assert "genre;" in lines[attributes["genre"].end_line - 1]
    assert all(r.owner.element_name == "Book" for r in attributes.values())

    method = _named(generator, path, "method")["summary"]
    assert "summary()" in lines[method.start_line - 1]
    assert method.owner.element_name == "Book"

    relationship = _named(generator, path, "relationship")["holds"]
    assert lines[relationship.start_line - 1].strip() == "@ManyToOne"
    assert "library;" in lines[relationship.end_line - 1] or "Library" in lines[relationship.end_line - 1]

    accessors = _named(generator, path, "accessors")
    assert {"title", "holds"} <= set(accessors)
    assert "getTitle()" in "\n".join(lines[accessors["title"].start_line - 1:accessors["title"].end_line])


def test_every_class_file_is_generated_from_its_class_and_enumeration_files_from_the_enumeration(traced):
    generator, files, _ = traced
    expected = [
        (f"{JAVA}/entity/Library.java", "entity.java.j2#class", "Library"),
        (f"{JAVA}/entity/Genre.java", "enum.java.j2#enumeration", "Genre"),
        (f"{JAVA}/repository/IBookRepository.java", "irepository.java.j2#class", "Book"),
        (f"{JAVA}/service/interfaces/IBookService.java", "iservice.java.j2#class", "Book"),
        (f"{JAVA}/service/impl/BookService.java", "service.java.j2#class", "Book"),
        (f"{JAVA}/controller/BookController.java", "controller.java.j2#class", "Book"),
        ("src/test/resources/http/book.http", "http.http.j2#class", "Book"),
    ]
    for path, rule, name in expected:
        assert path in files
        (source,) = generator.traced_files[path].sources
        assert (source.rule, source.element.element_name) == (rule, name)


def test_finder_methods_are_traced_to_their_attribute_in_repository_and_services(traced):
    generator, files, _ = traced
    for path in (f"{JAVA}/repository/IBookRepository.java", f"{JAVA}/service/interfaces/IBookService.java",
                 f"{JAVA}/service/impl/BookService.java"):
        finders = _named(generator, path, "finder")
        assert set(finders) == {"title", "genre"}, path  # the identifier has no finder
        text = "\n".join(_lines(files, path)[finders["title"].start_line - 1:finders["title"].end_line])
        assert "findAllByTitle" in text


def test_every_traced_path_is_a_file_of_the_project(traced):
    generator, files, _ = traced
    assert set(generator.traced_files) <= set(files)
    assert "pom.xml" in generator.traced_files and "mvnw" not in generator.traced_files  # copied: not traced


def test_tracing_does_not_change_the_generated_project(traced, tmp_path):
    _, with_ids, _ = traced
    plain_generator = SpringBackendGenerator(_model(with_ids=False), output_dir=str(tmp_path / "plain"))
    plain_generator.generate()
    assert with_ids == _read_all(tmp_path / "plain")
    assert not any(b"\x02" in content or b"\x03" in content for content in with_ids.values())
    # Elements without ids render untraced.
    assert all(not t.regions and not t.sources for t in plain_generator.traced_files.values())
