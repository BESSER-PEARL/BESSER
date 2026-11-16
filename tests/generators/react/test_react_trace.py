"""ReactGenerator records which page, route and component lines come from which GUI element."""
import os

import pytest

from besser.BUML.metamodel.traceability.traceability import model_elements
from besser.generators.react import ReactGenerator
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)
from tests.generators.react.conftest import class_diagram, gui_json

CLASSES = {"Book": [("title", "str"), ("pages", "int")]}


def _text(component_id, content):
    return {"type": "text", "tagName": "h1", "attributes": {"id": component_id},
            "components": [{"type": "textnode", "content": content}]}


def _table():
    return {"type": "table", "attributes": {"id": "table-books", "chart-title": "Books", "data-source": "cls-Book"}}


def _button():
    return {"type": "action-button", "tagName": "button",
            "attributes": {"id": "btn-create", "button-label": "New", "data-action-type": "create",
                           "data-entity-class": "cls-Book"},
            "components": [{"type": "textnode", "content": "New"}]}


def _container():
    return {"type": "default", "tagName": "div", "attributes": {"id": "box"},
            "components": [_text("inner-title", "Inside"), _button()]}


PAGES = {"Home": [_text("home-title", "Welcome")], "Books": [_text("books-title", "Books"), _table(), _container()]}


def _models(with_ids: bool):
    class_json = class_diagram(CLASSES)
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    gui = process_gui_diagram(gui_json(PAGES), class_json, domain)
    if with_ids:
        # The editor converter gives GUI elements no id (only a component_id): set them as a caller that has them would.
        # Elements of the domain model and the shared primitive types are left alone.
        gui_elements = [e for e in model_elements(gui) if type(e).__module__.startswith("besser.BUML.metamodel.gui")]
        for count, element in enumerate(gui_elements):
            element.id = element.id or f"gui-{count}"
    return domain, gui


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
    out = tmp_path_factory.mktemp("react_trace")
    domain, gui = _models(with_ids=True)
    generator = ReactGenerator(domain, gui, output_dir=str(out / "traced"))
    generator.generate()
    return generator, gui, _read_all(out / "traced"), out


def _lines(files, path):
    return files[path].decode("utf-8").split("\n")


def test_each_page_file_is_generated_from_its_screen(traced):
    generator, gui, files, _ = traced
    for description, file_name in (("Home", "Home"), ("Books", "Books")):
        screen = next(s for m in gui.modules for s in m.screens if s.description == description)
        (source,) = generator.traced_files[f"src/pages/{file_name}.tsx"].sources
        assert (source.element.element_id, source.element.element_type, source.rule) == (
            screen.id, "Screen", "page_builder.py#screen")
        assert f"src/pages/{file_name}.tsx" in files


def test_app_routes_and_imports_are_regions_of_their_screens(traced):
    generator, gui, files, _ = traced
    lines = _lines(files, "src/App.tsx")
    regions = generator.traced_files["src/App.tsx"].regions
    by_rule = {(r.element_name, r.rule.split("#")[1]): r for r in regions}
    for screen in (s for m in gui.modules for s in m.screens):
        route = by_rule[(screen.name, "route")]
        assert route.element_id == screen.id and route.start_line == route.end_line
        assert lines[route.start_line - 1].strip().startswith(f'<Route path="{screen.route_path}"')
        imported = by_rule[(screen.name, "import")]
        assert lines[imported.start_line - 1].startswith(f"import {screen.description} from")


def test_components_are_regions_nested_in_their_container(traced):
    generator, gui, files, _ = traced
    path = "src/pages/Books.tsx"
    lines = _lines(files, path)
    regions = {r.element_name: r for r in generator.traced_files[path].regions}
    assert all(r.rule == "page_builder.py#component" for r in regions.values())

    def line(name):
        return lines[regions[name].start_line - 1]

    assert 'id="books-title"' in line("books_title")
    assert 'id="box"' in line("box")
    assert "TableBlock" in line("table_books")
    assert 'id="inner-title"' in line("inner_title")
    box, inner, button = regions["box"], regions["inner_title"], regions["btn_create"]
    assert box.start_line < inner.start_line <= inner.end_line < button.start_line <= button.end_line <= box.end_line
    assert inner.owner.element_name == "box"
    # A region holds the code of its element, not its neighbours'.
    assert 'id="books-title"' not in "\n".join(lines[box.start_line - 1:box.end_line])


def test_every_traced_path_is_a_written_file_and_static_files_are_file_level(traced):
    generator, _, files, _ = traced
    assert set(generator.traced_files) <= set(files)
    assert "src/index.tsx" in generator.traced_files
    assert not generator.traced_files["src/index.tsx"].regions and not generator.traced_files["package.json"].sources
    # Copied assets are not rendered, so they carry no trace.
    assert "src/components/InputComponents.tsx" not in generator.traced_files


def test_tracing_does_not_change_the_generated_app(traced, tmp_path):
    _, _, with_ids, _ = traced
    domain, gui = _models(with_ids=False)
    plain_generator = ReactGenerator(domain, gui, output_dir=str(tmp_path / "plain"))
    plain_generator.generate()
    assert with_ids == _read_all(tmp_path / "plain")
    texts = [content.decode("utf-8") for path, content in with_ids.items() if not path.endswith(".png")]
    assert not any("\x02" in text or "\x03" in text for text in texts)
    # GUI elements without ids render untraced.
    assert not any(t.regions or t.sources for t in plain_generator.traced_files.values())
