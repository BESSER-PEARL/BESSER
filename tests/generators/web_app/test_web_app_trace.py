"""The web app generator's trace holds its parts' traces under the folders they are delivered in."""
import os

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.web_app import WebAppGenerator


def _models(with_ids: bool):
    title = Property("title", StringType)
    book = Class("Book", {title})
    greeting = Text(name="greeting", content="Welcome")
    screen = Screen(name="Home", description="Home", view_elements={greeting}, is_main_page=True)
    gui = GUIModel(name="LibraryApp", package="org.library", versionCode="1", versionName="1.0",
                   modules={Module(name="Main", screens={screen})}, description="Library")
    if with_ids:
        for element, ident in ((book, "class-book"), (title, "attr-title"), (screen, "screen-home"),
                               (greeting, "text-greeting")):
            element.id = ident
    return DomainModel("Library", types={book}), gui


def _read_all(folder) -> dict:
    files = {}
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                files[os.path.relpath(path, folder).replace(os.sep, "/")] = f.read()
    return files


def test_react_and_backend_traces_are_under_their_folders(tmp_path):
    generator = WebAppGenerator(*_models(with_ids=True), output_dir=str(tmp_path / "traced"))
    generator.generate()
    traced, files = generator.traced_files, _read_all(tmp_path / "traced")
    assert set(traced) <= set(files)

    # The React child's page and route, under frontend/.
    (source,) = traced["frontend/src/pages/Home.tsx"].sources
    assert (source.element.element_id, source.rule) == ("screen-home", "page_builder.py#screen")
    page = files["frontend/src/pages/Home.tsx"].decode("utf-8").split("\n")
    (greeting,) = [r for r in traced["frontend/src/pages/Home.tsx"].regions if r.element_id == "text-greeting"]
    assert "Welcome" in "\n".join(page[greeting.start_line - 1:greeting.end_line])
    routes = [r for r in traced["frontend/src/App.tsx"].regions if r.rule.endswith("#route")]
    assert [r.element_id for r in routes] == ["screen-home"]

    # The backend child's classes, under backend/.
    backend = traced["backend/sql_alchemy.py"].regions
    assert "class-book" in {r.element_id for r in backend}
    # Nothing is recorded under a path the web app did not deliver, or without its folder.
    assert not any(path in traced for path in ("src/App.tsx", "sql_alchemy.py"))


def test_tracing_does_not_change_the_generated_web_app(tmp_path):
    traced_generator = WebAppGenerator(*_models(with_ids=True), output_dir=str(tmp_path / "traced"))
    traced_generator.generate()
    plain_generator = WebAppGenerator(*_models(with_ids=False), output_dir=str(tmp_path / "plain"))
    plain_generator.generate()
    assert _read_all(tmp_path / "traced") == _read_all(tmp_path / "plain")
    assert not any(t.regions or t.sources for path, t in plain_generator.traced_files.items())
