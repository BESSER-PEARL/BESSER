"""A page's route is made of URL-safe characters whatever its name.

The route was the page name in lower case with spaces as hyphens, so a name
with ``&``, ``/``, ``?`` or ``#`` gave a route the router reads differently.
"""
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
