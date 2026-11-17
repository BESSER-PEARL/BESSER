"""A link to another page of the app navigates inside the app.

A page route in a plain ``<a href>`` reloads the whole single-page app; a
link whose target is one of the GUI's own pages goes through the router.
"""
from besser.BUML.metamodel.gui import Link, Menu

CLASSES = {"Order": [("reference", "str")]}


def _link(text, **attributes):
    return {"type": "link", "attributes": attributes,
            "components": [{"type": "textnode", "content": text}]}


def _anchor(text, **attributes):
    return {"tagName": "a", "attributes": attributes,
            "components": [{"type": "textnode", "content": text}]}


def _pages(*home_components):
    return {
        "Home": [{"tagName": "div", "components": list(home_components)}],
        "Order List": [{"tagName": "p", "components": [{"type": "textnode", "content": "Orders"}]}],
    }


def _elements(gui_model, kind):
    found = []
    for module in gui_model.modules:
        for screen in module.screens:
            stack = list(screen.view_elements)
            while stack:
                element = stack.pop()
                if isinstance(element, kind):
                    found.append(element)
                stack.extend(getattr(element, "view_elements", ()) or ())
    return found


def test_a_link_to_a_page_route_uses_the_router(build_app, jsx):
    app = build_app(CLASSES, _pages(_anchor("See orders", href="/order-list")))
    page = app.page("Home")

    assert 'import { useNavigate } from "react-router-dom";' in page
    anchor = jsx(page, "a")
    assert 'href="/order-list"' in anchor
    assert 'e.preventDefault(); navigate("/order-list"); }}' in anchor


def test_navigate_to_names_the_page_whatever_the_href(build_app, jsx):
    link = _link("Orders", href="#", **{"data-navigate-to": "order-list"})
    app = build_app(CLASSES, _pages(link))

    assert _elements(app.gui_model, Link)[0].url == "/order-list"
    anchor = jsx(app.page("Home"), "a")
    assert 'href="/order-list"' in anchor and 'navigate("/order-list")' in anchor


def test_a_link_names_a_page_by_its_name_or_id(build_app):
    app = build_app(CLASSES, _pages(
        _anchor("By name", href="/Order List"),
        _link("By id", href="#", **{"data-navigate-to": "page-order list"}),
    ))

    assert {link.url for link in _elements(app.gui_model, Link)} == {"/order-list"}


def test_other_links_stay_plain_anchors(build_app):
    app = build_app(CLASSES, _pages(
        _anchor("Docs", href="https://example.org/order-list"),
        _anchor("Mail", href="mailto:orders@example.org"),
        _anchor("Jump", href="#order-list"),
        _anchor("Elsewhere", href="/pricing"),
    ))
    page = app.page("Home")

    assert "navigate(" not in page and "useNavigate" not in page
    for href in ("https://example.org/order-list", "mailto:orders@example.org", "#order-list", "/pricing"):
        assert f'href="{href}"' in page


def test_a_page_link_that_opens_a_new_tab_is_left_to_the_browser(build_app, jsx):
    app = build_app(CLASSES, _pages(_anchor("Orders", href="/order-list", target="_blank")))

    anchor = jsx(app.page("Home"), "a")
    assert 'target="_blank"' in anchor and "navigate(" not in anchor


def test_menu_items_to_pages_use_the_router(build_app):
    menu = {"tagName": "nav", "components": [
        _anchor("Home", href="/home"), _anchor("Orders", href="/Order List"),
        _anchor("Help", href="https://example.org/help"),
    ]}
    app = build_app(CLASSES, {"Home": [menu], "Order List": []})
    page = app.page("Home")

    assert {item.url for item in _elements(app.gui_model, Menu)[0].menuItems} == {
        "/home", "/order-list", "https://example.org/help",
    }
    assert 'href="/order-list" onClick={(e) => {' in page and 'e.preventDefault(); navigate("/order-list"); }}' in page
    assert '<a href="https://example.org/help">' in page


def test_a_modified_click_on_a_page_link_is_left_to_the_browser(build_app):
    """Ctrl/Cmd/Shift-click or a middle click opens the page in a new tab or window."""
    menu = {"tagName": "nav", "components": [_anchor("Orders", href="/order-list")]}
    app = build_app(CLASSES, _pages(_anchor("See orders", href="/order-list"), menu))
    page = app.page("Home")

    plain_click_only = "if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return; "
    handler = f'(e) => {{ {plain_click_only}e.preventDefault(); navigate("/order-list"); }}'
    assert page.count("onClick={" + handler + "}") == 2
    assert page.count("e.preventDefault(); navigate(") == 2
