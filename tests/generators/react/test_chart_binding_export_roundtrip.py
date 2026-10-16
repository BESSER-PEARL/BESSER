"""Chart and metric-card bindings survive B-UML -> editor JSON -> B-UML.

``buml_to_json`` exported a metric card with ``card.title`` (unset on a card built
with ``metric_title``), so the editor showed the card's name instead of its title.
A chart's own ``data_binding`` (a chart without series) was exported as
``data-source`` / ``label-field`` / ``data-field`` but never read back, and the
exported ``data-source`` names the class where the parser only looked up element
ids. Nested field paths and filters were not exported at all.
"""
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import BarChart, MetricCard
from besser.BUML.metamodel.structural import Class, FloatType, Property, StringType
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _apply_bar_chart_attributes,
    _apply_metric_card_attributes,
)

CLASSES = {"Task": [("title", "str"), ("status", "str"), ("hours", "float")]}


def _task():
    status = Property(name="status", type=StringType)
    hours = Property(name="hours", type=FloatType)
    return Class(name="Task", attributes={Property(name="title", type=StringType), status, hours}), status, hours


def _elements(gui_model, kind):
    return {
        element.component_id: element
        for module in gui_model.modules
        for screen in module.screens
        for element in screen.view_elements
        if isinstance(element, kind)
    }


def test_a_metric_card_exports_its_title():
    task, _, hours = _task()
    card = MetricCard(name="total_card", metric_title="Total hours")
    card.data_binding = DataBinding(domain_concept=task, data_field=hours)
    attrs = {}
    _apply_metric_card_attributes(card, attrs)
    assert attrs["metric-title"] == "Total hours"


def test_a_chart_level_binding_round_trips(build_app):
    task, status, hours = _task()
    chart = BarChart(name="by_status")
    chart.data_binding = DataBinding(domain_concept=task, label_field=status, data_field=hours,
                                     filter_expression="hours > 0")
    attrs = {"id": "by-status", "chart-title": "Hours by status"}
    _apply_bar_chart_attributes(chart, attrs)
    assert attrs["data-source"] == "Task"
    assert attrs["filter"] == "hours > 0"

    app = build_app(CLASSES, {"Dashboard": [{"type": "bar-chart", "attributes": attrs}]})
    binding = _elements(app.gui_model, BarChart)["by-status"].data_binding
    assert binding is not None
    assert binding.domain_concept.name == "Task"
    assert binding.filter_expression == "hours > 0"


def test_nested_field_paths_round_trip(build_app):
    task, _, _ = _task()
    chart = BarChart(name="nested")
    chart.data_binding = DataBinding(domain_concept=task, label_field_path="owner.name",
                                     data_field_path="measure.value")
    attrs = {"id": "nested", "chart-title": "Nested"}
    _apply_bar_chart_attributes(chart, attrs)
    assert attrs["label-field-path"] == "owner.name"
    assert attrs["data-field-path"] == "measure.value"

    app = build_app(CLASSES, {"Dashboard": [{"type": "bar-chart", "attributes": attrs}]})
    binding = _elements(app.gui_model, BarChart)["nested"].data_binding
    assert binding.label_field_path == "owner.name"
    assert binding.data_field_path == "measure.value"


def test_a_chart_with_series_keeps_no_chart_level_binding(build_app):
    series = {"name": "Tasks", "data-source": "cls-Task", "label-field": "attr-Task-status"}
    attrs = {"id": "plain", "chart-title": "Plain", "series": [series]}
    app = build_app(CLASSES, {"Dashboard": [{"type": "bar-chart", "attributes": attrs}]})
    chart = _elements(app.gui_model, BarChart)["plain"]
    assert chart.series[0].data_binding.domain_concept.name == "Task"
