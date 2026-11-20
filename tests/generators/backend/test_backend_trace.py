"""The backend generator's trace: each file names what it was rendered from, and tracing changes no code."""

import os

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, Constraint, DomainModel, Enumeration, EnumerationLiteral, IntegerType, Method,
    Multiplicity, Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.pydantic_classes import PydanticGenerator


def _model(with_ids: bool) -> DomainModel:
    genre = Enumeration("Genre", {EnumerationLiteral("NOVEL")})
    library = Class("Library", {Property("name", StringType)},
                    methods={Method("open", type=StringType, code="def open(self):\n    return self.name\n"),
                             Method("close")})
    book = Class("Book", {Property("title", StringType), Property("pages", IntegerType), Property("genre", genre)})
    holds = BinaryAssociation("holds", {Property("library", library, multiplicity=Multiplicity(1, 1)),
                                        Property("books", book, multiplicity=Multiplicity(0, 9999))})
    pages = Constraint("PositivePages", book, "context Book inv PositivePages: self.pages > 0", "OCL")
    model = DomainModel("Library", types={genre, library, book}, associations={holds}, constraints={pages})
    if with_ids:  # as the editor gives them; without ids nothing is traceable, and the model is otherwise the same
        for type_ in (genre, library, book):
            type_.id = f"id-{type_.name}"
            for member in [*getattr(type_, "attributes", ()), *getattr(type_, "methods", ()), *getattr(type_, "literals", ())]:
                member.id = f"id-{type_.name}-{member.name}"
        holds.id = "id-holds"
        pages.id = "id-PositivePages"
    return model


def _read_all(folder) -> dict:
    files = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                files[os.path.relpath(path, folder).replace(os.sep, "/")] = f.read()
    return files


@pytest.fixture
def traced(tmp_path):
    generator = BackendGenerator(_model(with_ids=True), output_dir=str(tmp_path / "traced"))
    generator.generate()
    return generator, _read_all(tmp_path / "traced")


def test_tracing_does_not_change_the_generated_code(tmp_path, traced):
    _, with_ids = traced
    BackendGenerator(_model(with_ids=False), output_dir=str(tmp_path / "plain")).generate()
    plain = _read_all(tmp_path / "plain")
    assert set(with_ids) == set(plain)
    # Element ids differ between the two models only where the code prints none, so the files are equal.
    assert with_ids == plain
    assert not any("\x02" in code or "\x03" in code for code in with_ids.values())


def test_pydantic_tracing_does_not_change_the_generated_code(tmp_path):
    outputs = {}
    for with_ids in (True, False):
        generator = PydanticGenerator(_model(with_ids), backend=True, output_dir=str(tmp_path / str(with_ids)))
        generator.generate()
        outputs[with_ids] = (tmp_path / str(with_ids) / "pydantic_classes.py").read_text(encoding="utf-8")
    assert outputs[True] == outputs[False]
    assert generator.traced_files["pydantic_classes.py"].regions == ()  # the model without ids


def test_each_file_names_what_it_was_rendered_from(traced):
    generator, files = traced
    regions = {path: {(r.element_name, r.rule.split("#")[1]): r for r in trace.regions}
               for path, trace in generator.traced_files.items()}
    line = lambda path, region: files[path].split("\n")[region.start_line - 1]

    assert line("pydantic_classes.py", regions["pydantic_classes.py"][("Book", "class")]).startswith("class BookCreate(")
    assert line("pydantic_classes.py", regions["pydantic_classes.py"][("Genre", "enumeration")]) == "class Genre(Enum):"
    assert "library_router.router" in line("main_api.py", regions["main_api.py"][("Library", "router_include")])
    close = regions["routers/library_methods.py"][("close", "method")]
    assert "/methods/close/" in line("routers/library_methods.py", close)
    assert close.owner.element_name == "Library"
    assert [s.element.element_name for s in generator.traced_files["routers/book.py"].sources] == ["Book"]
    assert generator.traced_files["sql_alchemy.py"].regions  # the SQLAlchemy part keeps its own trace


def test_members_and_constraints_have_their_own_lines(traced):
    """Attributes, methods and OCL constraints are traced inside their class, nested under it."""
    generator, files = traced
    regions = {path: {(r.element_name, r.rule.split("#")[1]): r for r in trace.regions}
               for path, trace in generator.traced_files.items()}
    lines = lambda path, region: files[path].split("\n")[region.start_line - 1:region.end_line]

    title = regions["sql_alchemy.py"][("title", "attribute")]
    assert lines("sql_alchemy.py", title) == ["    title: Mapped_[str] = mapped_column(String_(100))"]
    assert (title.element_type, title.owner.element_name) == ("Property", "Book")
    assert lines("pydantic_classes.py", regions["pydantic_classes.py"][("title", "attribute")]) == ["    title: str"]
    method = regions["sql_alchemy.py"][("open", "method")]
    assert lines("sql_alchemy.py", method)[0] == "    def open(self):" and method.owner.element_name == "Library"
    validator = regions["pydantic_classes.py"][("PositivePages", "ocl_validator")]
    assert validator.element_type == "Constraint" and validator.owner is None
    assert lines("pydantic_classes.py", validator)[0] == "    @field_validator('pages')"
    # Each member lies within its class's block.
    book = regions["sql_alchemy.py"][("Book", "class")]
    assert book.start_line <= title.start_line <= title.end_line <= book.end_line
