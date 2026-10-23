"""``DataBinding.endpoint`` survives the web editor and the B-UML code builder.

The editor stores it as ``data-endpoint`` on a metric card and on each chart
series. A chart built in B-UML had no ``series`` attribute in the exported
JSON at all, so its series (and their endpoints) were lost on re-import.
"""
import json

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import (
    BarChart, LineChart, MetricCard, PieChart, RadarChart, RadialBarChart, Series,
)
from besser.BUML.metamodel.structural import Class, DomainModel, FloatType, Property, StringType
from besser.BUML.metamodel.gui.style import Color, Styling
from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _apply_component_specific_attributes,
    _serialize_gui_model,
)

ATTRIBUTES = [("region", "str"), ("amount", "float")]


def _class_diagram():
    elements = {}
    attribute_ids = []
    for attr_name, attr_type in ATTRIBUTES:
        attr_id = f"attr-Sale-{attr_name}"
        attribute_ids.append(attr_id)
        elements[attr_id] = {
            "id": attr_id, "name": attr_name, "type": "ClassAttribute", "owner": "cls-Sale",
            "bounds": {"x": 0, "y": 0, "width": 200, "height": 30}, "visibility": "public",
            "attributeType": attr_type, "isOptional": False, "isDerived": False, "isId": False,
        }
    elements["cls-Sale"] = {
        "id": "cls-Sale", "name": "Sale", "type": "Class", "owner": None,
        "bounds": {"x": 0, "y": 0, "width": 220, "height": 200},
        "attributes": attribute_ids, "methods": [],
    }
    return {"version": "3.0.0", "type": "ClassDiagram", "size": {"width": 2000, "height": 1000},
            "interactive": {"elements": {}, "relationships": {}}, "elements": elements,
            "relationships": {}, "assessments": {}}


def _import(components):
    class_json = _class_diagram()
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    project_gui = {"pages": [{
        "id": "page-dashboard", "name": "Dashboard",
        "frames": [{"component": {"type": "wrapper", "components": components}}],
    }], "styles": []}
    return process_gui_diagram(project_gui, class_json, domain), class_json, domain


def _elements(gui_model):
    return {
        element.component_id: element
        for module in gui_model.modules
        for screen in module.screens
        for element in screen.view_elements
    }


def _components(project_json):
    """Every exported component, flattened."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("attributes"):
                found.append(node)
            for child in node.get("components") or []:
                walk(child)

    for page in project_json["pages"]:
        for frame in page.get("frames", []):
            walk(frame.get("component"))
    return found


def _sale():
    region = Property(name="region", type=StringType)
    amount = Property(name="amount", type=FloatType)
    return Class(name="Sale", attributes={region, amount}), region, amount


CHARTS = [
    (BarChart, "bar-chart"),
    (LineChart, "line-chart"),
    (PieChart, "pie-chart"),
    (RadarChart, "radar-chart"),
    (RadialBarChart, "radial-bar-chart"),
]


@pytest.mark.parametrize("_, component_type", CHARTS)
def test_an_editor_chart_round_trips_its_series_endpoint(_, component_type):
    series = [
        {"name": "From stats", "data-source": "cls-Sale", "label-field": "attr-Sale-region",
         "data-field": "attr-Sale-amount", "data-endpoint": "/stats/revenue?year=2025", "color": "#123456"},
        {"name": "From CRUD", "data-source": "cls-Sale", "label-field": "attr-Sale-region",
         "data-field": "attr-Sale-amount"},
    ]
    component = {"type": component_type, "attributes": {"id": "chart", "chart-title": "Revenue",
                                                         "series": json.dumps(series)}}
    gui_model, class_json, domain = _import([component])
    chart = _elements(gui_model)["chart"]
    assert [s.data_binding.endpoint for s in chart.series] == ["/stats/revenue?year=2025", None]

    exported = next(c for c in _components(_serialize_gui_model(gui_model)) if c["attributes"].get("id") == "chart")
    exported_series = json.loads(exported["attributes"]["series"])
    assert exported_series[0]["data-endpoint"] == "/stats/revenue?year=2025"
    assert "data-endpoint" not in exported_series[1]
    assert exported_series[0]["color"] == "#123456"  # the editor's own series are kept

    project_gui = {"pages": _serialize_gui_model(gui_model)["pages"], "styles": []}
    again = _elements(process_gui_diagram(project_gui, class_json, domain))["chart"]
    assert [s.data_binding.endpoint for s in again.series] == ["/stats/revenue?year=2025", None]


def test_an_editor_metric_card_round_trips_its_endpoint():
    component = {"type": "metric-card", "attributes": {
        "id": "card", "metric-title": "Revenue", "data-source": "cls-Sale",
        "data-field": "attr-Sale-amount", "aggregation": "sum", "data-endpoint": "/stats/revenue"}}
    gui_model, class_json, domain = _import([component])
    assert _elements(gui_model)["card"].data_binding.endpoint == "/stats/revenue"

    exported = next(c for c in _components(_serialize_gui_model(gui_model)) if c["attributes"].get("id") == "card")
    assert exported["attributes"]["data-endpoint"] == "/stats/revenue"


@pytest.mark.parametrize("component_class, component_type", CHARTS)
def test_a_b_uml_chart_exports_its_series(component_class, component_type):
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

    gui_model, _, _ = _import([{"type": component_type, "attributes": attrs}])
    again = _elements(gui_model)["chart"]
    assert [s.label for s in again.series] == ["From stats", "From CRUD"]
    assert [s.data_binding.endpoint for s in again.series] == ["/stats/revenue", None]
    assert again.series[0].data_binding.label_field.name == "region"
    assert again.series[0].data_binding.data_field.name == "amount"


def test_a_b_uml_metric_card_exports_its_endpoint():
    sale, _, amount = _sale()
    card = MetricCard(name="revenue", metric_title="Revenue")
    card.data_binding = DataBinding(domain_concept=sale, data_field=amount, endpoint="/stats/revenue")
    attrs = {"id": "card"}
    _apply_component_specific_attributes(card, attrs)
    assert attrs["data-endpoint"] == "/stats/revenue"
    gui_model, _, _ = _import([{"type": "metric-card", "attributes": attrs}])
    assert _elements(gui_model)["card"].data_binding.endpoint == "/stats/revenue"


def test_the_code_builder_writes_the_endpoint(tmp_path):
    sale, region, amount = _sale()
    domain = DomainModel(name="Shop", types={sale})
    card = MetricCard(name="revenue", metric_title="Revenue")
    card.data_binding = DataBinding(domain_concept=sale, data_field=amount, endpoint="/stats/revenue?year=2025")
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
    assert 'endpoint="/stats/revenue?year=2025"' in code
    assert 'endpoint="/stats/by-region"' in code
    assert code.count("endpoint=") == 2

    namespace = {"domain_model": domain}
    exec(compile(code, str(path), "exec"), namespace)  # noqa: S102 - our own generated test code
    rebuilt = next(v for v in namespace.values() if isinstance(v, GUIModel))
    elements = {e.name: e for m in rebuilt.modules for s in m.screens for e in s.view_elements}
    assert elements["revenue"].data_binding.endpoint == "/stats/revenue?year=2025"
    assert elements["by_region"].series[0].data_binding.endpoint == "/stats/by-region"
    assert elements["plain"].series[0].data_binding.endpoint is None


def test_a_b_uml_series_on_field_paths_round_trips():
    """Fields an endpoint returns need not be class attributes: they travel as paths."""
    sale, _, _ = _sale()
    chart = LineChart(name="trend")
    chart.series = [Series(name="exports", label="Exports", styling=None, data_binding=DataBinding(
        domain_concept=sale, label_field_path="period", data_field_path="value", endpoint="/stats/series"))]
    attrs = {"id": "chart"}
    _apply_component_specific_attributes(chart, attrs)
    entry = json.loads(attrs["series"])[0]
    assert entry["label-field-path"] == "period" and entry["data-field-path"] == "value"
    assert "label-field" not in entry and "data-field" not in entry

    gui_model, _, _ = _import([{"type": "line-chart", "attributes": attrs}])
    binding = _elements(gui_model)["chart"].series[0].data_binding
    assert (binding.label_field_path, binding.data_field_path, binding.endpoint) == ("period", "value", "/stats/series")
