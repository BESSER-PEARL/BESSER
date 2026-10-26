"""``DataBinding.endpoint`` survives the web editor and the B-UML code builder.

The editor stores it as ``data-endpoint`` on a chart, a metric card and each
chart series.
"""
import json
import logging

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import (
    BarChart, LineChart, MetricCard, PieChart, RadarChart, RadialBarChart, Series, Table,
)
from besser.BUML.metamodel.gui.style import Color, Styling
from besser.BUML.metamodel.structural import Class, DomainModel, FloatType, Property, StringType
from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _apply_component_specific_attributes,
    _serialize_gui_model,
)
from tests.utilities.web_modeling_editor.backend.services.converters.utils_gui_binding_tests import (
    class_diagram,
    elements_by_id,
    import_gui,
)

ATTRIBUTES = [("region", "str"), ("amount", "float")]
CHARTS = [
    (BarChart, "bar-chart"),
    (LineChart, "line-chart"),
    (PieChart, "pie-chart"),
    (RadarChart, "radar-chart"),
    (RadialBarChart, "radial-bar-chart"),
]
SALE_SERIES = {"data-source": "cls-Sale", "label-field": "attr-Sale-region", "data-field": "attr-Sale-amount"}


def _import(components):
    return import_gui(components, class_diagram("Sale", ATTRIBUTES))


def _imported(component_type, attrs):
    return elements_by_id(_import([{"type": component_type, "attributes": attrs}]))[attrs["id"]]


def _exported(gui_model, component_id):
    """The exported attributes of one component of ``gui_model``."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if (node.get("attributes") or {}).get("id") == component_id:
                found.append(node["attributes"])
            for child in node.get("components") or []:
                walk(child)

    for page in _serialize_gui_model(gui_model)["pages"]:
        for frame in page.get("frames", []):
            walk(frame.get("component"))
    return found[0]


def _sale():
    region = Property(name="region", type=StringType)
    amount = Property(name="amount", type=FloatType)
    return Class(name="Sale", attributes={region, amount}), region, amount


@pytest.mark.parametrize("_, component_type", CHARTS)
def test_an_editor_chart_round_trips_its_series_endpoint(_, component_type):
    series = [
        {"name": "From stats", **SALE_SERIES, "data-endpoint": "/stats/revenue?year=2025", "color": "#123456"},
        {"name": "From CRUD", **SALE_SERIES},
    ]
    gui_model = _import([{"type": component_type, "attributes": {
        "id": "chart", "chart-title": "Revenue", "series": json.dumps(series)}}])
    chart = elements_by_id(gui_model)["chart"]
    assert [s.data_binding.endpoint for s in chart.series] == ["/stats/revenue?year=2025", None]

    exported_series = json.loads(_exported(gui_model, "chart")["series"])
    assert exported_series[0]["data-endpoint"] == "/stats/revenue?year=2025"
    assert "data-endpoint" not in exported_series[1]
    assert exported_series[0]["color"] == "#123456"

    again = _imported(component_type, _exported(gui_model, "chart"))
    assert [s.data_binding.endpoint for s in again.series] == ["/stats/revenue?year=2025", None]


@pytest.mark.parametrize("_, component_type", [*CHARTS, (MetricCard, "metric-card")])
def test_a_chart_level_or_metric_card_endpoint_round_trips(_, component_type):
    attrs = {"id": "comp", "chart-title": "Revenue", "metric-title": "Revenue", "data-source": "cls-Sale",
             "label-field": "attr-Sale-region", "data-field": "attr-Sale-amount", "data-endpoint": "/stats/revenue"}
    gui_model = _import([{"type": component_type, "attributes": attrs}])
    assert elements_by_id(gui_model)["comp"].data_binding.endpoint == "/stats/revenue"
    assert _exported(gui_model, "comp")["data-endpoint"] == "/stats/revenue"


def test_an_absolute_url_on_a_metric_card_round_trips():
    attrs = {"id": "card", "metric-title": "Revenue", "data-source": "cls-Sale", "data-field": "attr-Sale-amount",
             "data-endpoint": "https://stats.example.org/v1/revenue?year=2025"}
    gui_model = _import([{"type": "metric-card", "attributes": attrs}])
    assert _exported(gui_model, "card")["data-endpoint"] == "https://stats.example.org/v1/revenue?year=2025"


def test_an_endpoint_typed_without_its_slash_gets_one_and_stays_stable():
    attrs = {"id": "card", "metric-title": "Revenue", "data-source": "cls-Sale", "data-endpoint": "stats/revenue"}
    gui_model = _import([{"type": "metric-card", "attributes": attrs}])
    exported = _exported(gui_model, "card")
    assert exported["data-endpoint"] == "/stats/revenue"
    assert _exported(_import([{"type": "metric-card", "attributes": exported}]), "card")["data-endpoint"] == \
        "/stats/revenue"


def test_an_invalid_endpoint_is_dropped_with_a_warning_and_the_import_goes_on(caplog):
    components = [
        {"type": "metric-card", "attributes": {"id": "bad", "metric-title": "Bad", "data-source": "cls-Sale",
                                               "data-endpoint": "/stats/sales 2025"}},
        {"type": "metric-card", "attributes": {"id": "good", "metric-title": "Good", "data-source": "cls-Sale",
                                               "data-endpoint": "/stats/sales"}},
    ]
    with caplog.at_level(logging.WARNING):
        elements = elements_by_id(_import(components))
    assert elements["bad"].data_binding.endpoint is None
    assert elements["good"].data_binding.endpoint == "/stats/sales"
    assert "/stats/sales 2025" in caplog.text and "'Bad'" in caplog.text


def test_a_table_takes_no_endpoint(caplog):
    attrs = {"id": "rows", "chart-title": "Rows", "data-source": "cls-Sale", "data-endpoint": "/stats/sales"}
    with caplog.at_level(logging.WARNING):
        table = _imported("table", attrs)
    assert isinstance(table, Table)
    assert table.data_binding.endpoint is None
    assert "only charts and metric cards" in caplog.text


def test_editing_the_endpoint_in_b_uml_wins_over_the_stored_one():
    attrs = {"id": "card", "metric-title": "Revenue", "data-source": "cls-Sale", "data-endpoint": "/old"}
    card = _imported("metric-card", attrs)
    card.data_binding.endpoint = "/new"
    exported = dict(card.custom_attributes)
    _apply_component_specific_attributes(card, exported)
    assert exported["data-endpoint"] == "/new"

    card.data_binding.endpoint = None
    exported = dict(card.custom_attributes)
    _apply_component_specific_attributes(card, exported)
    assert "data-endpoint" not in exported


def test_an_empty_stored_endpoint_does_not_hide_the_model_one():
    """The editor writes ``data-endpoint: ""`` on every metric card it loads."""
    sale, _, amount = _sale()
    card = MetricCard(name="revenue", metric_title="Revenue")
    card.data_binding = DataBinding(domain_concept=sale, data_field=amount, endpoint="/stats/revenue")
    attrs = {"id": "card", "data-endpoint": ""}
    _apply_component_specific_attributes(card, attrs)
    assert attrs["data-endpoint"] == "/stats/revenue"


def test_a_series_endpoint_cleared_in_b_uml_is_not_exported():
    series = [{"name": "Stats", **SALE_SERIES, "data-endpoint": "/stats/revenue"}]
    chart = _imported("bar-chart", {"id": "chart", "chart-title": "Revenue", "series": json.dumps(series)})
    chart.series[0].data_binding.endpoint = None
    attrs = dict(chart.custom_attributes)
    _apply_component_specific_attributes(chart, attrs)
    assert "data-endpoint" not in json.loads(attrs["series"])[0]


def test_series_with_the_same_name_keep_their_own_endpoint():
    series = [{"name": "Hours", **SALE_SERIES, "data-endpoint": "/a"},
              {"name": "Hours", **SALE_SERIES, "data-endpoint": "/b"}]
    gui_model = _import([{"type": "bar-chart", "attributes": {
        "id": "chart", "chart-title": "Revenue", "series": json.dumps(series)}}])
    exported = json.loads(_exported(gui_model, "chart")["series"])
    assert [s["data-endpoint"] for s in exported] == ["/a", "/b"]


@pytest.mark.parametrize("component_class, component_type", CHARTS)
def test_a_b_uml_chart_exports_its_series_endpoints(component_class, component_type):
    sale, region, amount = _sale()
    chart = component_class(name="revenue")
    chart.series = [
        Series(name="stats", label="From stats", styling=Styling(color=Color(primary_color="#0b5394")),
               data_binding=DataBinding(domain_concept=sale, label_field=region, data_field=amount,
                                        aggregation="sum", endpoint="/stats/revenue")),
        Series(name="crud", label="From CRUD", styling=None,
               data_binding=DataBinding(domain_concept=sale, label_field=region, data_field=amount)),
    ]
    attrs = {"id": "chart"}
    _apply_component_specific_attributes(chart, attrs)
    series = json.loads(attrs["series"])
    assert series[0] == {"name": "From stats", "color": "#0b5394", "data-source": "Sale",
                         "label-field": "region", "data-field": "amount", "aggregation": "sum",
                         "data-endpoint": "/stats/revenue"}
    assert "data-endpoint" not in series[1]

    again = _imported(component_type, attrs)
    assert [s.data_binding.endpoint for s in again.series] == ["/stats/revenue", None]


def test_a_b_uml_series_on_field_paths_round_trips():
    """Fields an endpoint returns need not be class attributes: they travel as paths,
    dotted ones too."""
    sale, _, _ = _sale()
    chart = LineChart(name="trend")
    chart.series = [Series(name="exports", label="Exports", styling=None, data_binding=DataBinding(
        domain_concept=sale, label_field_path="period", data_field_path="measure.value",
        endpoint="/stats/series"))]
    attrs = {"id": "chart"}
    _apply_component_specific_attributes(chart, attrs)
    entry = json.loads(attrs["series"])[0]
    assert entry["label-field-path"] == "period" and entry["data-field-path"] == "measure.value"
    assert "label-field" not in entry and "data-field" not in entry

    binding = _imported("line-chart", attrs).series[0].data_binding
    assert (binding.label_field_path, binding.data_field_path, binding.endpoint) == \
        ("period", "measure.value", "/stats/series")


@pytest.mark.parametrize("endpoint", [
    "/stats/revenue?year=2025",
    '/x")+__import__("os").getcwd()#',
    "/q\"'\\x",
    "https://h/p?x=\"+__import__('os')+\"",
])
def test_the_code_builder_writes_the_endpoint(tmp_path, endpoint):
    sale, region, amount = _sale()
    domain = DomainModel(name="Shop", types={sale})
    card = MetricCard(name="revenue", metric_title="Revenue")
    card.data_binding = DataBinding(domain_concept=sale, data_field=amount, endpoint=endpoint)
    chart = BarChart(name="by_region")
    chart.series = [Series(name="stats", label="Stats", styling=None, data_binding=DataBinding(
        domain_concept=sale, label_field=region, data_field=amount, endpoint="/stats/by-region"))]
    plain = BarChart(name="plain")
    plain.series = [Series(name="crud", label="CRUD", styling=None, data_binding=DataBinding(
        domain_concept=sale, label_field=region, data_field=amount))]
    screen = Screen(name="Dashboard", description="", view_elements={card, chart, plain}, is_main_page=True)
    gui = GUIModel(name="ShopGUI", package="shop", versionCode="1", versionName="1.0", description="",
                   modules={Module(name="Main", screens={screen})})

    path = tmp_path / "gui.py"
    gui_model_to_code(gui, str(path), domain_model=domain)
    code = path.read_text(encoding="utf-8")
    assert code.count("endpoint=") == 2

    namespace = {"domain_model": domain}
    exec(compile(code, str(path), "exec"), namespace)  # noqa: S102 - our own generated test code
    rebuilt = next(v for v in namespace.values() if isinstance(v, GUIModel))
    elements = {e.name: e for m in rebuilt.modules for s in m.screens for e in s.view_elements}
    assert elements["revenue"].data_binding.endpoint == endpoint
    assert elements["by_region"].series[0].data_binding.endpoint == "/stats/by-region"
    assert elements["plain"].series[0].data_binding.endpoint is None
