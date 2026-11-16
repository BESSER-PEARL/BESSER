"""An element that holds only text keeps it.

A ``div`` is a container to the GUI converter, and a container without child
elements was generated self-closed, so the text written directly inside it (a
figure in a stat tile, a badge) was lost.
"""
from besser.BUML.metamodel.gui import Text, ViewContainer

CLASSES = {"Order": [("reference", "str")]}


def _elements(gui_model):
    found = []
    for module in gui_model.modules:
        for screen in module.screens:
            stack = list(screen.view_elements)
            while stack:
                element = stack.pop()
                found.append(element)
                stack.extend(getattr(element, "view_elements", ()) or ())
    return found


def test_text_written_directly_in_a_div_is_generated(build_app):
    tile = {"tagName": "div", "attributes": {"class": "tile"}, "components": [
        {"tagName": "div", "attributes": {"id": "figure", "class": "tile-figure"}, "content": "128"},
        {"tagName": "div", "attributes": {"id": "caption"},
         "components": [{"type": "textnode", "content": "Open orders"}]},
    ]}
    app = build_app(CLASSES, {"Home": [tile]})
    page = app.page("Home")

    assert '<div id="figure" className="tile-figure">{"128"}</div>' in page
    assert '<div id="caption">{"Open orders"}</div>' in page
    texts = {e.component_id: e for e in _elements(app.gui_model) if isinstance(e, Text)}
    assert texts["figure"].content == "128" and texts["figure"].tag_name == "div"


def test_other_text_only_block_elements_keep_their_tag(build_app):
    app = build_app(CLASSES, {"Home": [
        {"tagName": "section", "attributes": {"id": "note"}, "content": "Updated daily"},
        {"tagName": "li", "attributes": {"id": "item"}, "content": "First"},
    ]})
    page = app.page("Home")

    assert '<section id="note">{"Updated daily"}</section>' in page
    assert '<li id="item">{"First"}</li>' in page


def test_a_div_with_child_elements_or_no_text_stays_a_container(build_app):
    app = build_app(CLASSES, {"Home": [
        {"tagName": "div", "attributes": {"id": "box"}, "components": [
            {"tagName": "p", "components": [{"type": "textnode", "content": "Inside"}]}]},
        {"tagName": "div", "attributes": {"id": "spacer"}, "components": [{"type": "textnode", "content": "  "}]},
    ]})
    page = app.page("Home")

    containers = {e.component_id for e in _elements(app.gui_model) if isinstance(e, ViewContainer)}
    assert {"box", "spacer"} <= containers
    assert '<div id="box">' in page and '<div id="spacer" />' in page
