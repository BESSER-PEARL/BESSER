"""A GUI written by the modeling agent must keep its classes and inline SVG in code.

The agent writes CSS classes as ``attributes.class``. GrapesJS reads that on
load but saves ``classes`` only after a user edit, and the processor read
``classes`` only: an unedited agent page (a museum landing page, 25 class
names) reached the React app with no ``className`` at all and rendered as
plain text. Its inline SVG became ``<div>`` elements, so the drawing vanished.
"""
from besser.generators.react.page_builder import PageBuilderMixin
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors.utils import (
    css_class_names,
)

# Shapes copied from an agent-built page as the editor stores it before any edit.
NAV = {"tagName": "header", "attributes": {"id": "nav", "class": "ds-nav  museum-nav"}, "components": [
    {"tagName": "h2", "attributes": {"id": "brand", "class": "museum-brand"}, "content": "Museum"},
]}
SVG = {"tagName": "svg", "type": "svg",
       "attributes": {"viewBox": "0 0 545 294", "role": "img", "aria-label": "Exhibition"},
       "components": [
           {"tagName": "filter", "type": "svg-in", "attributes": {"id": "shadow"},
            "components": [{"tagName": "feDropShadow", "type": "svg-in", "attributes": {"std-deviation": "2"}}]},
           {"tagName": "rect", "type": "svg-in", "attributes": {"width": "545", "height": "294", "fill": "#d8e6df"}},
           {"tagName": "path", "type": "svg-in",
            "attributes": {"d": "M66 201V90", "stroke-width": "10", "stroke-linecap": "round"}},
           {"tagName": "text", "type": "svg-in", "attributes": {"x": "40", "y": "40", "font-size": "18"},
            "components": [{"type": "textnode", "content": "Fortress"}]},
       ]}


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


def test_class_attribute_is_read_when_classes_is_absent(build_app):
    elements = _elements(build_app({}, {"Home": [NAV]}).gui_model)

    assert elements["nav"].css_classes == ["ds-nav", "museum-nav"]
    assert elements["brand"].css_classes == ["museum-brand"]
    # Kept once, as GrapesJS does: in the classes, not also as an attribute.
    assert "class" not in elements["nav"].custom_attributes


def test_saved_classes_win_over_the_class_attribute():
    component = {"classes": ["saved", {"name": "rule"}], "attributes": {"class": "stale"}}

    assert css_class_names(component) == ["saved", "rule"]
    assert css_class_names({"classes": [], "attributes": {"class": "stale"}}) == []


def test_generated_page_keeps_the_class_names(build_app, jsx):
    page = build_app({}, {"Home": [NAV]}).page("Home")

    assert 'className="ds-nav museum-nav"' in jsx(page, "header")
    assert 'className="museum-brand"' in jsx(page, "h2")


def test_inline_svg_is_generated_as_svg(build_app):
    page = build_app({}, {"Home": [SVG]}).page("Home")

    assert "<svg " in page and "</svg>" in page
    assert "<rect " in page and "<path " in page
    assert "<filter " in page and "<feDropShadow " in page
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


def test_attributes_cannot_inject_markup(build_app):
    injected = {"__html": "<img src=x onerror=alert(1)>"}
    page = build_app({}, {"Home": [
        {"tagName": "div", "attributes": {"id": "box", "dangerouslySetInnerHTML": injected}},
        {"tagName": "svg", "type": "svg", "attributes": {"id": "pic", "dangerously-set-inner-h-t-m-l": injected}},
    ]}).page("Home")

    assert "dangerouslySetInnerHTML" not in page
    assert "onerror" not in page
