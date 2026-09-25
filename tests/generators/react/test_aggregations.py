"""Metric cards and charts aggregate the bound records.

A metric card showed the value of the *last* record, and a chart could not
count or group: the editor's ``aggregation`` attribute was never parsed (a
commented-out TODO) and the runtime blocks had no aggregation. Now
``DataBinding.aggregation`` carries it, a metric card with a field and no
aggregation sums it, one without a field counts the records, and a chart
series groups the records by its label field.
"""
from besser.BUML.metamodel.gui.binding import DataAggregation
from besser.BUML.metamodel.gui.dashboard import BarChart, MetricCard

CLASSES = {"Task": [("title", "str"), ("status", "str"), ("hours", "float")]}


def _metric(card_id, title, field=None, aggregation=None):
    attributes = {"id": card_id, "metric-title": title, "data-source": "cls-Task"}
    if field:
        attributes["data-field"] = f"attr-Task-{field}"
    if aggregation:
        attributes["aggregation"] = aggregation
    return {"type": "metric-card", "attributes": attributes}


def _chart(chart_id, aggregation=None, value_field=None, chart_level=False):
    series = {"name": "Tasks", "data-source": "cls-Task", "label-field": "attr-Task-status"}
    if value_field:
        series["data-field"] = f"attr-Task-{value_field}"
    attributes = {"id": chart_id, "chart-title": "Tasks by status", "series": [series]}
    if aggregation:
        (attributes if chart_level else series)["aggregation"] = aggregation
    return {"type": "bar-chart", "attributes": attributes}


def _elements(gui_model, kind):
    return {
        element.component_id: element
        for module in gui_model.modules
        for screen in module.screens
        for element in screen.view_elements
        if isinstance(element, kind)
    }


def test_the_processor_reads_the_aggregation(build_app):
    app = build_app(CLASSES, {"Dashboard": [
        _metric("avg-hours", "Average hours", "hours", "avg"),
        _metric("tasks", "Tasks"),
        _chart("by-status", "count"),
        _chart("hours-by-status", "sum", "hours", chart_level=True),
    ]})
    cards = _elements(app.gui_model, MetricCard)
    charts = _elements(app.gui_model, BarChart)

    assert cards["avg-hours"].data_binding.aggregation is DataAggregation.AVG
    assert cards["tasks"].data_binding.aggregation is None
    assert charts["by-status"].series[0].data_binding.aggregation is DataAggregation.COUNT
    assert charts["hours-by-status"].series[0].data_binding.aggregation is DataAggregation.SUM


def test_metric_cards_aggregate_all_records(build_app, jsx):
    app = build_app(CLASSES, {"Dashboard": [
        _metric("avg-hours", "Average hours", "hours", "avg"),
        _metric("total-hours", "Total hours", "hours"),
        _metric("tasks", "Tasks"),
    ]})
    page = app.page("Dashboard")

    assert '"aggregation": "average"' in jsx(page, "MetricCardBlock", 'id="avg-hours"')
    assert '"data_field": "hours"' in jsx(page, "MetricCardBlock", 'id="total-hours"')
    assert "aggregation" not in jsx(page, "MetricCardBlock", 'id="tasks"')

    block = app.file("components", "runtime", "MetricCardBlock.tsx")
    assert "getLastValue" not in block
    assert 'normalizeAggregation(aggregation) ?? (dataField ? "sum" : "count")' in block
    assert "setValue(metricValue(data, dataBinding?.data_field, dataBinding?.aggregation));" in block

    runtime = app.file("components", "runtime", "aggregate.ts")
    assert 'if (aggregation === "count") return rows.length;' in runtime
    assert "values.reduce((total, value) => total + value, 0) / values.length" in runtime


def test_a_chart_series_counts_or_sums_per_label(build_app, jsx):
    app = build_app(CLASSES, {"Dashboard": [_chart("by-status", "count"), _chart("hours", "sum", "hours")]})
    page = app.page("Dashboard")

    by_status = jsx(page, "ChartBlock", 'id="by-status"')
    assert '"aggregation": "count"' in by_status and '"labelField": "status"' in by_status
    assert '"aggregation": "sum"' in jsx(page, "ChartBlock", 'id="hours"')

    block = app.file("components", "runtime", "ChartBlock.tsx")
    assert "combined[key][s.name || \"Series\"] = aggregate(rows, aggregation, s.dataField);" in block
    # A bound series with no value field counts records per label.
    assert '? "count"' in block


def test_the_aggregation_survives_export_and_reimport(build_app):
    import os
    import tempfile

    from besser.BUML.metamodel.project import Project
    from besser.BUML.metamodel.structural import Metadata
    from besser.utilities.buml_code_builder.project_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters import process_class_diagram
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json,
    )

    app = build_app(CLASSES, {"Dashboard": [_metric("avg-hours", "Average hours", "hours", "avg")]})
    domain = process_class_diagram({"title": "Domain", "model": app.class_json})
    path = os.path.join(tempfile.mkdtemp(), "project.py")
    project_to_code(Project(name="p", models=[domain, app.gui_model], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        code = handle.read()
    assert 'aggregation="average"' in code

    entry = project_to_json(code)["diagrams"]["GUINoCodeDiagram"]
    page = (entry[0] if isinstance(entry, list) else entry)["model"]["pages"][0]
    card = page["frames"][0]["component"]["components"][0]
    assert card["attributes"]["aggregation"] in ("avg", "average")
