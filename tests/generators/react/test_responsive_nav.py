"""A navigation bar fits a phone screen.

A designed bar lays out its links in one inline flex row of fixed height, so
on a 390 px screen the last links ran off the side and the page scrolled
sideways. On a narrow screen the generated stylesheet lets the bar wrap.
"""
import re

CLASSES = {"Order": [("reference", "str")]}


def test_a_navigation_bar_wraps_on_a_narrow_screen(build_app):
    nav = {"tagName": "nav", "attributes": {"id": "topnav"},
           "style": {"display": "flex", "height": "64px", "justify-content": "space-between"},
           "components": [{"tagName": "a", "attributes": {"href": "/orders"},
                           "components": [{"type": "textnode", "content": "Orders"}]}]}
    app = build_app(CLASSES, {"Orders": [nav]})

    assert '<nav id="topnav"' in app.page("Orders") and '"height": "64px"' in app.page("Orders")
    assert "import './index.css';" in app.file("index.tsx")
    css = re.sub(r"\s+", " ", app.file("index.css"))
    narrow = css[css.index("@media (max-width: 640px) {"):]
    # !important: the bar's own layout is an inline style.
    assert "nav, nav * { flex-wrap: wrap !important; }" in narrow
    assert "nav { height: auto !important; }" in narrow
