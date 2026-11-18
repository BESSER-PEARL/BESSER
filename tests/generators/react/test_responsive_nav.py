"""A navigation bar fits a phone screen without overriding the design.

A designed bar lays out its links in one inline flex row of fixed height, so
on a 390 px screen the last links ran off the side. The bar's fixed height is
generated as a minimum height, and on a narrow screen a low-priority rule lets
the bar and its rows wrap. A forced rule (``height: auto !important``) also
squashed bars that fit and broke designs that handle phones themselves.
"""
import re

from besser.generators.react.page_builder import PageBuilderMixin

CLASSES = {"Order": [("reference", "str")]}


def _nav(**style):
    return {"tagName": "nav", "attributes": {"id": "topnav"}, "style": {"display": "flex", **style},
            "components": [{"tagName": "a", "attributes": {"href": "/orders"},
                            "components": [{"type": "textnode", "content": "Orders"}]}]}


def test_a_fixed_nav_height_becomes_a_minimum_height(build_app, jsx):
    page = build_app(CLASSES, {"Orders": [_nav(height="64px")]}).page("Orders")

    nav = jsx(page, "nav", 'id="topnav"')
    assert '"minHeight": "64px"' in nav
    assert '"height"' not in nav


def test_only_a_fixed_height_is_turned_into_a_minimum():
    assert PageBuilderMixin._nav_style({"height": "100vh", "display": "flex"}) == {
        "display": "flex", "minHeight": "100vh"}
    # A sticky top bar grows with its rows like any bar in the flow.
    assert PageBuilderMixin._nav_style({"position": "sticky", "top": "0", "height": "64px"}) == {
        "position": "sticky", "top": "0", "minHeight": "64px"}
    for unchanged in ({"display": "flex"}, {"height": "auto"}, {"height": "48px", "minHeight": "40px"}):
        assert PageBuilderMixin._nav_style(dict(unchanged)) == unchanged


def test_a_nav_that_scrolls_is_out_of_the_flow_or_is_collapsed_keeps_its_height():
    """A sidebar that scrolls inside 100vh grew to the height of all its links,
    so the last ones could no longer be reached; a fixed or absolute bar is laid
    out against the viewport; a 0-height menu is collapsed on purpose."""
    for kept in (
        {"position": "fixed", "height": "100vh", "overflowY": "auto"},
        {"position": "sticky", "top": "0", "height": "100vh", "overflowY": "auto"},
        {"height": "320px", "overflow": "hidden"},
        {"height": "48px", "overflowX": "scroll"},
        {"position": "fixed", "bottom": "0", "height": "64px"},
        {"position": "absolute", "height": "100%"},
        {"height": "0"},
        {"height": "0px", "overflow": "visible"},
    ):
        assert PageBuilderMixin._nav_style(dict(kept)) == kept, kept
    # A visible overflow does not make a bar scroll.
    assert PageBuilderMixin._nav_style({"height": "64px", "overflow": "visible"}) == {
        "overflow": "visible", "minHeight": "64px"}


def test_the_narrow_screen_rule_yields_to_the_design(build_app):
    css = re.sub(r"\s+", " ", build_app(CLASSES, {"Orders": [_nav(height="64px")]}).file("index.css"))

    narrow = css[css.index("@layer base { @media (max-width: 640px) {"):]
    assert narrow.startswith("@layer base { @media (max-width: 640px) { nav, nav > * { flex-wrap: wrap; } } }")
    assert "!important" not in css and "height" not in narrow
