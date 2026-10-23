"""A GUI written by the modeling agent must keep its classes and inline SVG in code.

The agent writes CSS classes as ``attributes.class``. GrapesJS reads that on
load but saves ``classes`` only after a user edit, and the processor read
``classes`` only: an unedited agent page (a museum landing page, 25 class
names) reached the React app with no ``className`` at all and rendered as
plain text. Its inline SVG became ``<div>`` elements, so the drawing vanished.
"""
import os

from besser.BUML.metamodel.structural import DomainModel
from besser.generators.react import ReactGenerator
from besser.generators.react.page_builder import PageBuilderMixin
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
)

# Shapes copied from an agent-built page as the editor stores it before any edit.
AGENT_GUI = {
    "title": "Museum",
    "styles": [{"selectors": [], "selectorsAdd": ".museum-nav", "style": {"display": "flex"}}],
    "pages": [{
        "id": "home",
        "name": "Home",
        "frames": [{"component": {"type": "wrapper", "components": [
            {"tagName": "header", "attributes": {"id": "nav", "class": "ds-nav  museum-nav"}, "components": [
                {"tagName": "h2", "attributes": {"id": "brand", "class": "museum-brand"}, "content": "Museum"},
            ]},
            {"tagName": "svg", "type": "svg",
             "attributes": {"viewBox": "0 0 545 294", "role": "img", "aria-label": "Exhibition"},
             "components": [
                 {"tagName": "rect", "type": "svg-in", "attributes": {"width": "545", "height": "294", "fill": "#d8e6df"}},
                 {"tagName": "path", "type": "svg-in",
                  "attributes": {"d": "M66 201V90", "stroke": "#f2eee2", "stroke-width": "10", "stroke-linecap": "round"}},
                 {"tagName": "text", "type": "svg-in", "attributes": {"x": "40", "y": "40", "font-size": "18"},
                  "components": [{"type": "textnode", "content": "Fortress"}]},
             ]},
        ]}}],
    }],
}


def _elements(gui_model):
    found = {}

    def walk(element):
        found[element.component_id] = element
        for child in getattr(element, "view_elements", None) or []:
            walk(child)

    screen = next(iter(next(iter(gui_model.modules)).screens))
    for element in screen.view_elements:
        walk(element)
    return found


def _home_page(tmp_path):
    gui = process_gui_diagram(AGENT_GUI, {}, None)
    ReactGenerator(model=DomainModel(name="Museum"), gui_model=gui, output_dir=str(tmp_path)).generate()
    with open(os.path.join(str(tmp_path), "src", "pages", "Home.tsx"), encoding="utf-8") as f:
        return f.read()


def test_class_attribute_is_read_when_classes_is_absent():
    elements = _elements(process_gui_diagram(AGENT_GUI, {}, None))

    assert elements["nav"].css_classes == ["ds-nav", "museum-nav"]
    assert elements["brand"].css_classes == ["museum-brand"]


def test_generated_page_keeps_the_class_names(tmp_path):
    page = _home_page(tmp_path)

    assert '<header id="nav" className="ds-nav museum-nav"' in page
    assert 'className="museum-brand"' in page


def test_inline_svg_is_generated_as_svg(tmp_path):
    page = _home_page(tmp_path)

    assert "<svg " in page and "</svg>" in page
    assert "<rect " in page and "<path " in page
    assert '"strokeWidth": "10"' in page and '"strokeLinecap": "round"' in page
    assert '"stroke-width"' not in page
    assert '"aria-label": "Exhibition"' in page
    assert '{"Fortress"}</text>' in page


def test_svg_attribute_names_follow_react():
    attributes = PageBuilderMixin._react_svg_attributes({
        "stroke-width": "2", "xlink:href": "#a", "viewBox": "0 0 1 1",
        "aria-hidden": "true", "data-role": "icon",
    })

    assert attributes == {
        "strokeWidth": "2", "xlinkHref": "#a", "viewBox": "0 0 1 1",
        "aria-hidden": "true", "data-role": "icon",
    }
