"""A generated table applies its binding's filter and takes per-screen options.

The table ignored ``DataBinding.filter_expression`` (charts and metric cards
applied it), so a filter set in the GUI model was silently dropped. A page could
not open a row, start on a filter, word the empty state or add a row button
without editing the shared TableComponent; these are now opt-in options, and a
table that sets none renders as before.
"""
import json
import re

from tests.generators.react.test_api_client import AXIOS_STUB, _runnable

CLASSES = {"Task": [("title", "str"), ("status", "str")]}
ROWS = [{"id": 1, "title": "a", "status": "OPEN"}, {"id": 2, "title": "b", "status": "DONE"}]


def _table(filter_expression=None):
    attributes = {"id": "tasks", "chart-title": "Tasks", "data-source": "cls-Task"}
    if filter_expression:
        attributes["filter"] = filter_expression
    return {"type": "table", "attributes": attributes}


def _block(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin) + len(end)]


def _harness(app, tmp_path):
    """bindingData.ts and the client, runnable by Node.js against an axios stub."""
    harness = tmp_path / "h"
    (harness / "api").mkdir(parents=True)
    (harness / "api" / "client.ts").write_text(_runnable(app.file("api", "generated", "client.ts")), encoding="utf-8")
    (harness / "bindingData.ts").write_text(
        _runnable(app.file("components", "runtime", "bindingData.ts")).replace("../../api/generated/client",
                                                                             "./api/client"), encoding="utf-8")
    axios = tmp_path / "node_modules" / "axios"
    axios.mkdir(parents=True)
    (axios / "package.json").write_text('{"name": "axios", "type": "module", "main": "index.js"}', encoding="utf-8")
    (axios / "index.js").write_text(AXIOS_STUB, encoding="utf-8")


def test_the_page_passes_the_filter_to_the_table(build_app, jsx):
    page = build_app(CLASSES, {"Tasks": [_table("status=OPEN")]}).page("Tasks")

    assert '"filter_expression": "status=OPEN"' in jsx(page, "TableBlock", 'id="tasks"')


def test_the_table_shows_only_the_rows_its_filter_keeps(build_app, tmp_path, run_ts):
    app = build_app(CLASSES, {"Tasks": [_table()]})
    _harness(app, tmp_path)
    table = app.file("components", "table", "TableComponent.tsx")
    binding_filter = _block(table, "  const bindingFilter = useMemo(", "\n  );\n")
    rows_memo = _block(table, "  const normalizedRows = useMemo(", "\n  }, [tableData, bindingFilter]);\n")

    shown = run_ts(
        'import { applyFilter, parseFilterExpression } from "./h/bindingData.ts";\n'
        "const useMemo = (compute: () => any, _deps: unknown[]) => compute();\n"
        "function rows(tableData: any[], dataBinding: any) {\n"
        + binding_filter + rows_memo
        + "  return normalizedRows;\n}\n"
        f"const data = {json.dumps(ROWS)};\n"
        "console.log(JSON.stringify([rows(data, { filter_expression: 'status=OPEN' }),"
        " rows(data, { filter: 'status=DONE' }), rows(data, {}), rows(data, undefined)]));\n",
        name="rows.mts")

    filtered, by_filter, unfiltered, unbound = shown
    assert [row["id"] for row in filtered] == [1]
    assert [row["id"] for row in by_filter] == [2]
    # A table without a filter keeps every row, as before.
    assert unfiltered == unbound == ROWS


def test_a_row_link_fills_in_the_row_values(build_app, run_ts):
    table = build_app(CLASSES, {"Tasks": [_table()]}).file("components", "table", "TableComponent.tsx")
    helpers = (_block(table, "const getNestedValue = ", "\n};\n")
               + _block(table, "const resolveRowLink = ", ")));\n"))

    links = run_ts(helpers + 'console.log(JSON.stringify([resolveRowLink("/tasks/{id}", { id: 7 }), '
                   'resolveRowLink("/t/{owner.name}/{title}", { owner: { name: "a b" }, title: "x/y" })]));\n')

    assert links == ["/tasks/7", "/t/a%20b/x%2Fy"]


def test_a_table_renders_outside_a_router(build_app, run_ts):
    """A test that mounts a page without a router: react-router's useNavigate throws there."""
    table = build_app(CLASSES, {"Tasks": [_table()]}).file("components", "table", "TableComponent.tsx")
    navigate = _block(table, "  const navigate = ", "\n")

    kinds = run_ts(
        "function table(inRouter: boolean) {\n"
        "  const useInRouterContext = () => inRouter;\n"
        "  const useNavigate = () => {\n"
        '    if (!inRouter) throw new Error("useNavigate() may be used only in the context of a <Router>");\n'
        "    return (to: string) => to;\n"
        "  };\n" + navigate + "  return typeof navigate;\n}\n"
        "console.log(JSON.stringify([table(false), table(true)]));\n")

    assert kinds == ["undefined", "function"]


def test_the_options_are_opt_in(build_app):
    app = build_app(CLASSES, {"Tasks": [_table()]})
    table = app.file("components", "table", "TableComponent.tsx")
    options = _block(table, "interface TableOptions {", "\n}\n")

    for field in ("rowLink?: string;", "initialFilters?: Record<string, string>;", "emptyText?: string;",
                  "rowActions?: Array<{ label: string; onClick: (row: any) => void }>;"):
        assert field in options
    # Unset, the table starts unfiltered, keeps its empty-state text and adds no column.
    assert "useState<Record<string, string>>(() => ({ ...(options?.initialFilters ?? {}) }))" in table
    assert 'resolvedOptions.emptyText || `No data available for ${dataBinding?.entity || "this table"}.`' in table
    assert table.count("resolvedOptions.rowActions.length > 0 &&") == 2
    assert "if (resolvedOptions.rowLink && navigate && !(event.target as HTMLElement).closest(\"button\"))" in table
    # The generated page sets none of them.
    page = app.page("Tasks")
    assert not re.search(r"rowLink|initialFilters|emptyText|rowActions", page)
