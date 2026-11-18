"""A metric card shows no trend caption when it has no trend to show.

Every card with ``show_trend`` printed the fixed caption "Latest value" under
its figure, which is wrong for a count or a total; no trend is computed, so
the card takes no trend properties either.
"""

CLASSES = {"Order": [("reference", "str"), ("total", "float")]}


def test_a_card_asking_for_a_trend_prints_no_fixed_caption(build_app, jsx):
    card = {"type": "metric-card", "attributes": {"id": "orders", "metric-title": "Total orders",
                                                  "data-source": "cls-Order", "show-trend": "true"}}
    app = build_app(CLASSES, {"Dashboard": [card]})

    # The model's setting is carried, without an invented trend figure.
    block = jsx(app.page("Dashboard"), "MetricCardBlock")
    assert '"showTrend": true' in block and '"trend"' not in block
    component = app.file("components", "charts", "MetricCardComponent.tsx")
    assert "Latest value" not in component
    props = component[component.index("interface Props {"):component.index("}\n", component.index("interface Props {"))]
    for unused in ('"show-trend"', '"positive-color"', '"negative-color"', "trend?"):
        assert unused not in props
    assert "trend" not in app.file("components", "runtime", "MetricCardBlock.tsx")
