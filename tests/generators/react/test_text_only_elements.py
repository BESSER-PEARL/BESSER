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


def _text(tag, element_id, text, **component):
    node = {"attributes": {"id": element_id}, "components": [{"type": "textnode", "content": text}], **component}
    if tag:
        node["tagName"] = tag
    return node


def test_a_text_only_element_keeps_its_own_tag(build_app):
    app = build_app(CLASSES, {"Home": [
        _text("nav", "menu", "Menu"),
        _text("blockquote", "quote", "Quoted"),
        _text("pre", "code-block", "x = 1"),
        _text("address", "where", "1 Main Road"),
        {"tagName": "dl", "attributes": {"id": "terms"}, "components": [
            _text("dt", "term", "Term"), _text("dd", "definition", "Meaning")]},
    ]})
    page = app.page("Home")

    for expected in ('<nav id="menu">{"Menu"}</nav>', '<blockquote id="quote">{"Quoted"}</blockquote>',
                     '<pre id="code-block">{"x = 1"}</pre>', '<address id="where">{"1 Main Road"}</address>',
                     '<dt id="term">{"Term"}</dt>', '<dd id="definition">{"Meaning"}</dd>'):
        assert expected in page, expected


def test_a_text_only_element_without_a_tag_or_with_a_table_tag_is_a_div(build_app):
    """A GrapesJS component without ``tagName`` is a div; table cells are generated
    inside divs (rows are), where a ``<td>`` or a ``<p>`` would be invalid."""
    app = build_app(CLASSES, {"Home": [
        _text(None, "untyped", "Untyped"),
        _text(None, "cell", "In a cell", type="cell"),
        {"tagName": "table", "attributes": {"id": "grid"}, "components": [
            {"tagName": "tr", "components": [_text("td", "first", "A"), _text("th", "head", "B")]}]},
    ]})
    page = app.page("Home")

    for expected in ('<div id="untyped">{"Untyped"}</div>', '<div id="cell">{"In a cell"}</div>',
                     '<div id="first">{"A"}</div>', '<div id="head">{"B"}</div>'):
        assert expected in page, expected
    assert "<p " not in page


def test_whitespace_entities_and_placeholders_are_not_text(build_app):
    app = build_app(CLASSES, {"Home": [
        _text("div", "spacer", "&nbsp;"),
        _text("div", "spacer2", "  &#160;"),
        _text("div", "greeting", "{{ user.name }}"),
    ]})
    page = app.page("Home")

    for element_id in ("spacer", "spacer2", "greeting"):
        assert f'<div id="{element_id}" />' in page, element_id
    assert "&nbsp;" not in page and "{{" not in page
