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
import json

import pytest

from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import (
    BarChart, LineChart, MetricCard, PieChart, RadarChart, RadialBarChart, Series, Table,
)
from besser.BUML.metamodel.gui.style import Color, Styling
from besser.BUML.metamodel.structural import Class, FloatType, Property, StringType
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _apply_bar_chart_attributes,
    _apply_component_specific_attributes,
    _apply_metric_card_attributes,
    _apply_table_attributes,
    _serialize_gui_model,
)
from tests.utilities.web_modeling_editor.backend.services.converters.utils_gui_binding_tests import (
    class_diagram,
    elements_by_id,
    import_gui,
)

TASK_ATTRIBUTES = [("title", "str"), ("status", "str"), ("hours", "float")]


def _import(component_type, attrs):
    """Re-import one exported component; returns the parsed element by its id."""
    gui_model = import_gui([{"type": component_type, "attributes": attrs}], class_diagram("Task", TASK_ATTRIBUTES))
    return elements_by_id(gui_model)[attrs["id"]]


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


CHARTS = [
    (BarChart, "bar-chart"),
    (LineChart, "line-chart"),
    (PieChart, "pie-chart"),
    (RadarChart, "radar-chart"),
    (RadialBarChart, "radial-bar-chart"),
]


def _series(task, status, hours, name, color=None):
    styling = Styling(color=Color(primary_color=color)) if color else None
    return Series(name=name, label=name, styling=styling,
                  data_binding=DataBinding(domain_concept=task, label_field=status, data_field=hours))


def _export(gui_model, component_id):
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


@pytest.mark.parametrize("component_class, component_type", CHARTS)
def test_a_b_uml_chart_exports_its_series(component_class, component_type):
    """``buml_to_json`` wrote no ``series`` for a chart built in B-UML, so its series
    were lost on re-import; the series colour came back on line and bar charts only."""
    task, status, hours = _task()
    chart = component_class(name="by_status")
    chart.series = [_series(task, status, hours, "Open", "#0b5394"), _series(task, status, hours, "Done")]
    attrs = {"id": "chart"}
    _apply_component_specific_attributes(chart, attrs)
    assert json.loads(attrs["series"]) == [
        {"name": "Open", "color": "#0b5394", "data-source": "Task", "label-field": "status", "data-field": "hours"},
        {"name": "Done", "data-source": "Task", "label-field": "status", "data-field": "hours"},
    ]

    again = _import(component_type, attrs)
    assert [s.label for s in again.series] == ["Open", "Done"]
    assert again.series[0].data_binding.label_field.name == "status"
    assert again.series[0].styling.color.primary_color == "#0b5394"


def _editor_chart(series):
    return {"id": "chart", "chart-title": "Hours", "series": json.dumps(series)}


EDITOR_SERIES = {"data-source": "cls-Task", "label-field": "attr-Task-status", "data-field": "attr-Task-hours"}


def test_editing_the_series_in_b_uml_wins_over_the_stored_ones():
    """The model is the source of truth for the series, as for a map's layers."""
    stored = [{"name": "Open", **EDITOR_SERIES, "data": [{"name": "a", "value": 1}]},
              {"name": "Done", **EDITOR_SERIES, "filter": "status=done"}]
    chart = _import("bar-chart", _editor_chart(stored))
    task, status, hours = _task()
    del chart.series[1]
    chart.series[0].data_binding.filter_expression = "hours > 2"
    chart.series.append(_series(task, status, hours, "Added"))
    gui_model = import_gui([], class_diagram("Task", TASK_ATTRIBUTES))
    next(iter(next(iter(gui_model.modules)).screens)).view_elements.add(chart)

    exported = json.loads(_export(gui_model, "chart")["series"])
    assert [s["name"] for s in exported] == ["Open", "Added"]
    assert exported[0]["filter"] == "hours > 2"
    assert exported[0]["data"] == [{"name": "a", "value": 1}]  # editor-only keys are kept
    assert "data" not in exported[1]


def test_series_deleted_in_b_uml_do_not_come_back():
    chart = _import("line-chart", _editor_chart([{"name": "Open", **EDITOR_SERIES}]))
    chart.series = []
    attrs = {"id": "chart", "series": json.dumps([{"name": "Open", **EDITOR_SERIES}])}
    _apply_component_specific_attributes(chart, attrs)
    assert json.loads(attrs["series"]) == []


def test_series_with_the_same_name_keep_their_own_binding():
    stored = [{"name": "Hours", **EDITOR_SERIES, "filter": "status=open"},
              {"name": "Hours", **EDITOR_SERIES, "filter": "status=done"}]
    chart = _import("bar-chart", _editor_chart(stored))
    attrs = dict(chart.custom_attributes)
    _apply_component_specific_attributes(chart, attrs)
    assert [s["filter"] for s in json.loads(attrs["series"])] == ["status=open", "status=done"]


def test_a_chart_without_series_writes_none():
    task, status, hours = _task()
    chart = BarChart(name="plain")
    chart.data_binding = DataBinding(domain_concept=task, label_field=status, data_field=hours)
    attrs = {"id": "plain"}
    _apply_component_specific_attributes(chart, attrs)
    assert "series" not in attrs
