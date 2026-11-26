"""A chart's legend and tooltip name a series by its label.

The editor's series name "Orders by status" becomes the identifier
``Orders_by_status``; the legend showed that identifier instead of the label.
"""
import pytest

CLASSES = {"Order": [("reference", "str"), ("status", "str")]}


def _series_chart(kind):
    series = {"name": "Orders by status", "data-source": "cls-Order", "label-field": "attr-Order-status"}
    return {"type": kind, "attributes": {"id": "orders-chart", "chart-title": "Orders", "series": [series]}}


@pytest.mark.parametrize("kind, component", [
    ("bar-chart", "BarChartComponent"),
    ("line-chart", "LineChartComponent"),
    ("radar-chart", "RadarChartComponent"),
    ("radial-bar-chart", "RadialBarChartComponent"),
])
def test_the_series_label_is_shown_and_its_name_stays_the_data_key(build_app, kind, component):
    app = build_app(CLASSES, {"Dashboard": [_series_chart(kind)]})
    page = app.page("Dashboard")
    chart = app.file("components", "charts", f"{component}.tsx")

    assert '"name": "Orders_by_status"' in page and '"label": "Orders by status"' in page
    assert "label?: string;" in chart.split("interface Props")[0]
    shown = chart[chart.index("name={s.label || s.name"):]
    assert "dataKey={s.name" in chart and "name={s.name" not in chart
    assert shown.startswith("name={s.label || s.name}") or shown.startswith("name={s.label || s.name || ")
