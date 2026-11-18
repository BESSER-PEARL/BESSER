"""A metric card shows no trend caption when it has no trend to show.

Every card with ``show_trend`` printed the fixed caption "Latest value" under
its figure, which is wrong for a count or a total; no trend is computed.
"""

CLASSES = {"Order": [("reference", "str"), ("total", "float")]}


def test_a_card_asking_for_a_trend_prints_no_fixed_caption(build_app, jsx):
    card = {"type": "metric-card", "attributes": {"id": "orders", "metric-title": "Total orders",
                                                  "data-source": "cls-Order", "show-trend": "true"}}
    app = build_app(CLASSES, {"Dashboard": [card]})

    assert '"showTrend": true' in jsx(app.page("Dashboard"), "MetricCardBlock")
    component = app.file("components", "charts", "MetricCardComponent.tsx")
    assert "Latest value" not in component
    rendered = component[component.index("return (\n    <div id={id}"):]
    assert "showTrend &&" not in rendered and "{displayValue}" in rendered
