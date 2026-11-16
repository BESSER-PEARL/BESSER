"""Runtime components reload their data after a write made by another one.

A table refreshed only itself after a create, update or delete, so a metric
card or chart bound to the same class kept showing the old figures until the
page was reloaded.
"""

CLASSES = {"Order": [("reference", "str"), ("total", "float")]}
PAGES = {"Orders": [
    {"type": "table", "attributes": {"id": "orders-table", "data-source": "cls-Order", "action-buttons": "true"}},
    {"type": "metric-card", "attributes": {"data-source": "cls-Order", "aggregation": "count"}},
]}

# The dependency lists of the effects that fetch, per runtime component.
RELOADING_EFFECTS = {
    "MetricCardBlock.tsx": ["    dataVersion,\n  ]);"],
    "ChartBlock.tsx": ["}, [chartType, normalizedSeries, dataVersion]);", "    dataVersion,\n  ]);"],
    "DataListBlock.tsx": [", dataVersion]);"],
}


def _runtime(app, name):
    return app.file("components", "runtime", name)


def test_every_write_path_announces_the_change(build_app):
    app = build_app(CLASSES, PAGES)
    table = app.file("components", "table", "TableComponent.tsx")

    for request in ("await axios.post(url, processedValues);", "await axios.put(fullUrl, processedValues);",
                    "await axios.delete(fullUrl);"):
        after = table[table.index(request):]
        assert "notifyDataChanged();" in after[:after.index("} catch")], request
    form = _runtime(app, "FormBlock.tsx")
    assert form.index("await axios.post(url, payload);") < form.index("notifyDataChanged();")
    assert "notifyDataChanged();" in app.file("components", "MethodButton.tsx")


def test_every_data_bound_component_reloads_on_a_change(build_app):
    app = build_app(CLASSES, PAGES)

    store = _runtime(app, "dataVersion.ts")
    assert "export const notifyDataChanged" in store and "export const useDataVersion" in store
    for name, dependency_lists in RELOADING_EFFECTS.items():
        source = _runtime(app, name)
        assert "const dataVersion = useDataVersion();" in source, name
        for dependencies in dependency_lists:
            assert dependencies in source, (name, dependencies)
    table = app.file("components", "table", "TableComponent.tsx")
    assert "if (dataVersion !== loadedVersion.current) {" in table


def test_the_table_no_longer_keeps_its_own_refresh_registry(build_app):
    app = build_app(CLASSES, PAGES)

    for source in (app.file("contexts", "TableContext.tsx"), app.file("components", "MethodButton.tsx"),
                   app.file("components", "table", "TableComponent.tsx")):
        assert "refreshTable" not in source and "registerTableRefresh" not in source


def test_deleting_a_row_asks_for_confirmation(build_app):
    app = build_app(CLASSES, PAGES)
    table = app.file("components", "table", "TableComponent.tsx")

    remove = table[table.index('title="Remove"'):]
    remove = remove[:remove.index("</button>")]
    assert "window.confirm(" in remove
    assert remove.index("window.confirm(") < remove.index("deleteRow(row)")
