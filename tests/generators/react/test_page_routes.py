"""A page's route is made of URL-safe characters whatever its name.

The route was the page name in lower case with spaces as hyphens, so a name
with ``&``, ``/``, ``?`` or ``#`` gave a route the router reads differently.
"""
import json
import re

from besser.generators.react.routes import page_route

CLASSES = {"Order": [("reference", "str")]}


def _anchor(text, **attributes):
    return {"tagName": "a", "attributes": attributes,
            "components": [{"type": "textnode", "content": text}]}


def _routes(app):
    return re.findall(r'<Route path="([^"*]+)" element=\{<(\w+) />\}', app.file("App.tsx"))


def test_a_route_holds_no_reserved_character():
    assert page_route("Orders & Returns") == "/orders-returns"
    assert page_route("Q&A / Help?") == "/q-a-help"
    assert page_route("50% off #1") == "/50-off-1"
    # Names that were already safe keep their route.
    assert page_route("Order List") == "/order-list"
    assert page_route("order_list") == "/order_list"
    assert page_route("Übersicht") == "/übersicht"


def test_pages_and_the_links_to_them_agree_on_the_route(build_app, jsx):
    pages = {
        "Home": [_anchor("Returns", href="/orders-&-returns", **{"data-navigate-to": "orders-&-returns"}),
                 {"type": "button", "attributes": {"id": "go", "data-action-type": "navigate",
                                                   "data-target-screen": "page-orders & returns"},
                  "components": [{"type": "textnode", "content": "Open"}]}],
        "Orders & Returns": [_anchor("Back", href="/home")],
    }
    app = build_app(CLASSES, pages)

    assert ("/orders-returns", "OrdersReturns") in _routes(app)
    home = app.page("Home")
    assert 'navigate("/orders-returns")' in jsx(home, "a")
    assert '() => navigate("/orders-returns")' in jsx(home, "button")


def test_pages_whose_names_give_one_slug_get_distinct_routes(build_app):
    app = build_app(CLASSES, {"A & B": [], "A / B": [], "???": []})

    assert sorted(route for route, _ in _routes(app)) == ["/a-b", "/a-b-2", "/page-3"]


def _screen_routes(app):
    return {screen.component_id: screen.route_path
            for module in app.gui_model.modules for screen in module.screens}


def test_a_route_set_on_the_page_is_kept(build_app, jsx):
    pages = {
        "Home": [_anchor("Shop", href="/shop")],
        "Products": {"route_path": "/shop", "components": []},
        "Orders & Returns": {"route_path": "/orders-&-returns", "components": []},
    }
    app = build_app(CLASSES, pages)

    routes = _routes(app)
    assert ("/shop", "Products") in routes
    # A stored route that is not URL-safe gives way to the one derived from the name.
    assert ("/orders-returns", "OrdersReturns") in routes
    assert 'navigate("/shop")' in jsx(app.page("Home"), "a")


def test_a_link_to_a_page_with_a_non_ascii_name_opens_it(build_app, jsx):
    app = build_app(CLASSES, {"Home": [_anchor("Orders", href="/Заказы")], "Заказы": []})

    assert f"navigate({json.dumps('/заказы')})" in jsx(app.page("Home"), "a")


def test_pages_with_one_slug_keep_their_routes_when_reordered(build_app):
    pages = {"A & B": {"id": "page-2", "components": []}, "A / B": {"id": "page-1", "components": []}}
    first = _screen_routes(build_app(CLASSES, pages))
    second = _screen_routes(build_app(CLASSES, dict(reversed(list(pages.items())))))

    assert first == second == {"page-1": "/a-b", "page-2": "/a-b-2"}
