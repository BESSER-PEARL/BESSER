"""Chart, table and metric-card bindings survive B-UML -> editor JSON -> B-UML.

``buml_to_json`` exported a metric card with ``card.title`` (unset on a card built
with ``metric_title``), so the editor showed the card's name instead of its title.
A chart's own ``data_binding`` (a chart without series) was exported as
``data-source`` / ``label-field`` / ``data-field`` but never read back, and the
exported ``data-source`` names the class where the parser only looked up element
ids. Nested field paths and filters were not exported at all, and the metric card
and table parsers ignored the exported ``filter``. The exported ``label-field`` /
``data-field`` name attributes too, and only element ids were resolved.
"""
import pytest

from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import (
    BarChart, LineChart, MetricCard, PieChart, RadarChart, RadialBarChart, Table,
)
from besser.BUML.metamodel.structural import Class, FloatType, Property, StringType
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _apply_bar_chart_attributes,
    _apply_component_specific_attributes,
    _apply_metric_card_attributes,
    _apply_table_attributes,
)

TASK_ATTRIBUTES = [("title", "str"), ("status", "str"), ("hours", "float")]


def _class_diagram():
    """Apollon class diagram with one ``Task`` class (id ``cls-Task``, attribute
    ids ``attr-Task-<name>``)."""
    elements = {}
    attribute_ids = []
    for attr_name, attr_type in TASK_ATTRIBUTES:
        attr_id = f"attr-Task-{attr_name}"
        attribute_ids.append(attr_id)
        elements[attr_id] = {
            "id": attr_id, "name": attr_name, "type": "ClassAttribute", "owner": "cls-Task",
            "bounds": {"x": 0, "y": 0, "width": 200, "height": 30}, "visibility": "public",
            "attributeType": attr_type, "isOptional": False, "isDerived": False, "isId": False,
        }
    elements["cls-Task"] = {
        "id": "cls-Task", "name": "Task", "type": "Class", "owner": None,
        "bounds": {"x": 0, "y": 0, "width": 220, "height": 200},
        "attributes": attribute_ids, "methods": [],
    }
    return {"version": "3.0.0", "type": "ClassDiagram", "size": {"width": 2000, "height": 1000},
            "interactive": {"elements": {}, "relationships": {}}, "elements": elements,
            "relationships": {}, "assessments": {}}


def _import(component_type, attrs):
    """Re-import one exported component; returns the parsed element by its id."""
    class_json = _class_diagram()
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    project_gui = {"pages": [{
        "id": "page-dashboard", "name": "Dashboard",
        "frames": [{"component": {"type": "wrapper", "components": [
            {"type": component_type, "attributes": attrs},
        ]}}],
    }], "styles": []}
    gui_model = process_gui_diagram(project_gui, class_json, domain)
    return next(
        element
        for module in gui_model.modules
        for screen in module.screens
        for element in screen.view_elements
        if element.component_id == attrs["id"]
    )


def _task():
    status = Property(name="status", type=StringType)
    hours = Property(name="hours", type=FloatType)
    return Class(name="Task", attributes={Property(name="title", type=StringType), status, hours}), status, hours


def test_a_metric_card_exports_its_title():
    task, _, hours = _task()
    card = MetricCard(name="total_card", metric_title="Total hours")
    card.data_binding = DataBinding(domain_concept=task, data_field=hours)
    attrs = {}
    _apply_metric_card_attributes(card, attrs)
    assert attrs["metric-title"] == "Total hours"


def test_a_metric_card_titled_via_title_exports_it():
    card = MetricCard(name="rev", title="Revenue")
    attrs = {"id": "rev"}
    _apply_metric_card_attributes(card, attrs)
    assert attrs["metric-title"] == "Revenue"
    assert _import("metric-card", attrs).metric_title == "Revenue"


@pytest.mark.parametrize("component_class, component_type", [
    (BarChart, "bar-chart"),
    (LineChart, "line-chart"),
    (PieChart, "pie-chart"),
    (RadarChart, "radar-chart"),
    (RadialBarChart, "radial-bar-chart"),
    (MetricCard, "metric-card"),
])
def test_a_chart_level_binding_round_trips(component_class, component_type):
    task, status, hours = _task()
    component = component_class(name="by_status")
    component.data_binding = DataBinding(domain_concept=task, label_field=status, data_field=hours,
                                         filter_expression="hours > 0")
    attrs = {"id": "by-status"}
    _apply_component_specific_attributes(component, attrs)
    assert attrs["data-source"] == "Task"
    assert attrs["label-field"] == "status"
    assert attrs["data-field"] == "hours"
    assert attrs["filter"] == "hours > 0"

    binding = _import(component_type, attrs).data_binding
    assert binding is not None
    assert binding.domain_concept.name == "Task"
    assert binding.label_field is not None and binding.label_field.name == "status"
    assert binding.data_field is not None and binding.data_field.name == "hours"
    assert binding.filter_expression == "hours > 0"


def test_a_radial_bar_series_resolves_attribute_names():
    series = {"name": "Tasks", "data-source": "Task", "features": "status", "values": "hours"}
    attrs = {"id": "radial", "chart-title": "Radial", "series": [series]}
    binding = _import("radial-bar-chart", attrs).series[0].data_binding
    assert binding.domain_concept.name == "Task"
    assert binding.label_field.name == "status"
    assert binding.data_field.name == "hours"


def test_nested_field_paths_round_trip():
    task, _, _ = _task()
    chart = BarChart(name="nested")
    chart.data_binding = DataBinding(domain_concept=task, label_field_path="owner.name",
                                     data_field_path="measure.value")
    attrs = {"id": "nested", "chart-title": "Nested"}
    _apply_bar_chart_attributes(chart, attrs)
    assert attrs["label-field-path"] == "owner.name"
    assert attrs["data-field-path"] == "measure.value"

    binding = _import("bar-chart", attrs).data_binding
    assert binding.label_field_path == "owner.name"
    assert binding.data_field_path == "measure.value"


def test_a_chart_with_series_keeps_no_chart_level_binding():
    series = {"name": "Tasks", "data-source": "cls-Task", "label-field": "attr-Task-status"}
    attrs = {"id": "plain", "chart-title": "Plain", "series": [series]}
    chart = _import("bar-chart", attrs)
    assert chart.series[0].data_binding.domain_concept.name == "Task"
    assert chart.data_binding is None


def test_a_metric_card_binding_round_trips():
    task, _, _ = _task()
    card = MetricCard(name="total_card", metric_title="Total hours")
    card.data_binding = DataBinding(domain_concept=task, label_field_path="owner.name",
                                    data_field_path="measure.value", filter_expression="hours > 0")
    attrs = {"id": "total"}
    _apply_metric_card_attributes(card, attrs)
    assert attrs["filter"] == "hours > 0"

    binding = _import("metric-card", attrs).data_binding
    assert binding.domain_concept.name == "Task"
    assert binding.filter_expression == "hours > 0"
    assert binding.label_field_path == "owner.name"
    assert binding.data_field_path == "measure.value"


def test_a_table_filter_round_trips():
    task, _, _ = _task()
    table = Table(name="tasks")
    table.data_binding = DataBinding(domain_concept=task, filter_expression="hours > 0")
    attrs = {"id": "tasks"}
    _apply_table_attributes(table, attrs)
    assert attrs["filter"] == "hours > 0"

    binding = _import("table", attrs).data_binding
    assert binding.domain_concept.name == "Task"
    assert binding.filter_expression == "hours > 0"
