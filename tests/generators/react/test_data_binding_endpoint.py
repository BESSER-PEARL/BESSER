"""Charts and metric cards read from their ``DataBinding.endpoint``.

The serializer always pointed a binding at the class's CRUD list
(``/<class>/``). A chart or metric card with an endpoint now reads from it;
tables, maps and forms keep the CRUD list, because they also write back
through it.
"""
import logging

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import BarChart, LineChart, MetricCard, Series, Table
from besser.BUML.metamodel.structural import Class, DomainModel, FloatType, Property, StringType
from besser.generators.react import ReactGenerator

CLASSES = {"Sale": [("region", "str"), ("amount", "float")]}


def _metric(card_id, endpoint=None):
    attributes = {"id": card_id, "metric-title": "Revenue", "data-source": "cls-Sale",
                  "data-field": "attr-Sale-amount", "aggregation": "sum"}
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
    app = build_app(CLASSES, {"Dashboard": [_metric("revenue", "/stats/revenue?year=2025"), _metric("plain")]})
    page = app.page("Dashboard")

    assert '"endpoint": "/stats/revenue?year=2025"' in jsx(page, "MetricCardBlock", 'id="revenue"')
    assert '"endpoint": "/sale/"' in jsx(page, "MetricCardBlock", 'id="plain"')


def test_chart_series_fetch_their_endpoint(build_app, jsx):
    app = build_app(CLASSES, {"Dashboard": [
        _chart("bars", "bar-chart", "/stats/revenue/by-region"),
        _chart("lines", "line-chart", "https://stats.example.org/revenue"),
        _chart("pie", "pie-chart"),
    ]})
    page = app.page("Dashboard")

    assert '"endpoint": "/stats/revenue/by-region"' in jsx(page, "ChartBlock", 'id="bars"')
    assert '"endpoint": "https://stats.example.org/revenue"' in jsx(page, "ChartBlock", 'id="lines"')
    assert '"endpoint": "/sale/"' in jsx(page, "ChartBlock", 'id="pie"')


def _model_with(component):
    region = Property(name="region", type=StringType)
    amount = Property(name="amount", type=FloatType)
    sale = Class(name="Sale", attributes={region, amount})
    domain = DomainModel(name="Shop", types={sale})
    binding = DataBinding(domain_concept=sale, label_field=region, data_field=amount,
                          endpoint="/stats/revenue")
    if isinstance(component, (BarChart, LineChart)):
        component.series = [Series(name="revenue", label="Revenue", data_binding=binding, styling=None)]
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


def test_a_table_keeps_its_crud_list(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="besser.generators.react.serialization"):
        page = _generate(tmp_path, Table(name="sales_table", title="Sales"))
    assert '"endpoint": "/sale/"' in page
    assert "/stats/revenue" not in page
    assert any("only charts and metric cards" in record.message for record in caplog.records)
