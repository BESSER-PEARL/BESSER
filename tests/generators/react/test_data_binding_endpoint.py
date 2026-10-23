"""Charts and metric cards read from their ``DataBinding.endpoint``.

The serializer always pointed a binding at the class's CRUD list
(``/<class>/``). A chart or metric card with an endpoint now reads from it,
flagged as a custom endpoint so the runtime fetches it as written; tables,
maps and forms keep the CRUD list, because they also write back through it.
"""
import logging
import os

import pytest

from besser.BUML.metamodel.gui import Form, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import BarChart, LineChart, Map, MapLayer, MetricCard, Series, Table
from besser.BUML.metamodel.structural import Class, DomainModel, FloatType, Property, StringType
from besser.generators.react import ReactGenerator

CLASSES = {"Sale": [("region", "str"), ("amount", "float")]}


def _metric(card_id, endpoint=None, **extra):
    attributes = {"id": card_id, "metric-title": "Revenue", "data-source": "cls-Sale",
                  "data-field": "attr-Sale-amount", "aggregation": "sum", **extra}
    if endpoint is not None:
        attributes["data-endpoint"] = endpoint
    return {"type": "metric-card", "attributes": attributes}


def _chart(chart_id, kind="bar-chart", endpoint=None):
    series = {"name": "Revenue", "data-source": "cls-Sale", "label-field": "attr-Sale-region",
              "data-field": "attr-Sale-amount"}
    if endpoint is not None:
        series["data-endpoint"] = endpoint
    return {"type": kind, "attributes": {"id": chart_id, "chart-title": "Revenue by region", "series": [series]}}


def test_the_processor_reads_the_endpoint(build_app):
    app = build_app(CLASSES, {"Dashboard": [
        _metric("revenue", "/stats/revenue?year=2025"),
        _metric("plain"),
        _chart("by-region", endpoint="/stats/revenue/by-region"),
        _chart("typed-without-slash", endpoint="stats/revenue"),
    ]})
    elements = {e.component_id: e for m in app.gui_model.modules for s in m.screens for e in s.view_elements}

    assert elements["revenue"].data_binding.endpoint == "/stats/revenue?year=2025"
    assert elements["plain"].data_binding.endpoint is None
    assert elements["by-region"].series[0].data_binding.endpoint == "/stats/revenue/by-region"
    # The generated charts add a missing leading slash; the parser does the same.
    assert elements["typed-without-slash"].series[0].data_binding.endpoint == "/stats/revenue"


def test_a_metric_card_fetches_its_endpoint(build_app, jsx):
    app = build_app(CLASSES, {"Dashboard": [
        _metric("revenue", "/stats/revenue?year=2025", filter="region=EU"),
        _metric("remote", "https://stats.example.org/v1/revenue"),
        _metric("plain"),
    ]})
    page = app.page("Dashboard")

    revenue = jsx(page, "MetricCardBlock", 'id="revenue"')
    assert '"endpoint": "/stats/revenue?year=2025"' in revenue
    assert '"custom_endpoint": true' in revenue
    assert '"filter_expression": "region=EU"' in revenue
    assert '"endpoint": "https://stats.example.org/v1/revenue"' in jsx(page, "MetricCardBlock", 'id="remote"')
    plain = jsx(page, "MetricCardBlock", 'id="plain"')
    assert '"endpoint": "/sale/"' in plain
    assert "custom_endpoint" not in plain


def test_chart_series_fetch_their_endpoint(build_app, jsx):
    app = build_app(CLASSES, {"Dashboard": [
        _chart("bars", "bar-chart", "/stats/revenue/by-region"),
        _chart("lines", "line-chart", "https://stats.example.org/revenue"),
        _chart("pie", "pie-chart"),
    ]})
    page = app.page("Dashboard")

    bars = jsx(page, "ChartBlock", 'id="bars"')
    assert '"endpoint": "/stats/revenue/by-region"' in bars
    assert '"customEndpoint": true' in bars
    assert '"endpoint": "https://stats.example.org/revenue"' in jsx(page, "ChartBlock", 'id="lines"')
    pie = jsx(page, "ChartBlock", 'id="pie"')
    assert '"endpoint": "/sale/"' in pie
    assert "customEndpoint" not in pie


def test_a_chart_level_endpoint_is_read(build_app, jsx):
    chart = {"type": "bar-chart", "attributes": {
        "id": "level", "chart-title": "Revenue", "data-source": "cls-Sale", "label-field": "attr-Sale-region",
        "data-field": "attr-Sale-amount", "data-endpoint": "/stats/revenue"}}
    page = build_app(CLASSES, {"Dashboard": [chart]}).page("Dashboard")
    block = jsx(page, "ChartBlock", 'id="level"')
    assert '"endpoint": "/stats/revenue"' in block
    assert '"custom_endpoint": true' in block


def test_the_runtime_fetches_a_custom_endpoint_as_written(build_app):
    """A filtered chart on a custom endpoint was sent to ``<endpoint>/search/`` (a
    route only the CRUD API has), so it 404'd and the chart stayed empty."""
    app = build_app(CLASSES, {"Dashboard": [_chart("bars", "bar-chart", "/stats/revenue")]})
    runtime = os.path.join(app.src, "components", "runtime")
    with open(os.path.join(runtime, "bindingData.ts"), encoding="utf-8") as handle:
        data = handle.read()
    with open(os.path.join(runtime, "ChartBlock.tsx"), encoding="utf-8") as handle:
        chart = handle.read()
    assert "if (options.custom) return baseUrl;" in data
    assert "custom: Boolean(s.customEndpoint)" in chart
    assert "custom: Boolean(dataBinding?.custom_endpoint)" in chart


def _model_with(component):
    region = Property(name="region", type=StringType)
    amount = Property(name="amount", type=FloatType)
    sale = Class(name="Sale", attributes={region, amount})
    domain = DomainModel(name="Shop", types={sale})
    binding = DataBinding(domain_concept=sale, label_field=region, data_field=amount,
                          endpoint="/stats/revenue")
    if isinstance(component, (BarChart, LineChart)):
        component.series = [Series(name="revenue", label="Revenue", data_binding=binding, styling=None)]
    elif isinstance(component, Map):
        layer = MapLayer(name="sales")
        layer.data_binding = binding
        component.layers = [layer]
    else:
        component.data_binding = binding
    screen = Screen(name="Dashboard", description="", view_elements={component}, is_main_page=True)
    gui = GUIModel(name="ShopGUI", package="shop", versionCode="1", versionName="1.0", description="",
                   modules={Module(name="Main", screens={screen})})
    return domain, gui


def _generate(tmp_path, component):
    domain, gui = _model_with(component)
    ReactGenerator(model=domain, gui_model=gui, output_dir=str(tmp_path)).generate()
    with open(tmp_path / "src" / "pages" / "Dashboard.tsx", encoding="utf-8") as handle:
        return handle.read()


def test_a_b_uml_chart_reads_its_endpoint(tmp_path):
    page = _generate(tmp_path, BarChart(name="revenue_chart", title="Revenue"))
    assert '"endpoint": "/stats/revenue"' in page
    assert '"endpoint": "/sale/"' not in page


def test_a_b_uml_metric_card_reads_its_endpoint(tmp_path):
    page = _generate(tmp_path, MetricCard(name="revenue_card", metric_title="Revenue"))
    assert '"endpoint": "/stats/revenue"' in page


@pytest.mark.parametrize("component", [
    Table(name="sales_table", title="Sales"),
    Map(name="sales_map", title="Sales"),
    Form(name="sale_form", description="", inputFields=set()),
], ids=["table", "map", "form"])
def test_a_writer_keeps_its_crud_list(tmp_path, caplog, component):
    with caplog.at_level(logging.WARNING, logger="besser.generators.react.serialization"):
        page = _generate(tmp_path, component)
    assert "/stats/revenue" not in page
    assert any("only charts and metric cards" in record.message for record in caplog.records)
