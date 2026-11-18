"""A table's column filter offers an enumeration's literals.

The filter treated an enumeration column as free text: the user had to type a
literal exactly, under a "(string)" label, although the column carries its
literals as ``options``.
"""
import json
import re

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import FieldColumn, Table
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, Property, StringType,
)
from besser.generators.react import ReactGenerator


def _generate(tmp_path):
    status = Enumeration(name="OrderStatus", literals={
        EnumerationLiteral(name=name) for name in ("OPEN", "SHIPPED", "DELIVERED")})
    status_attr = Property(name="status", type=status)
    reference = Property(name="reference", type=StringType)
    order = Class(name="Order", attributes={reference, status_attr})
    table = Table(name="OrderTable", title="Orders",
                  columns=[FieldColumn(label="Reference", field=reference), FieldColumn(label="Status", field=status_attr)],
                  data_binding=DataBinding(name="order_binding", domain_concept=order))
    screen = Screen(name="Orders", description="Orders", view_elements={table}, is_main_page=True)
    gui_model = GUIModel(name="OrdersApp", package="", versionCode="1", versionName="1.0",
                         modules={Module(name="M", screens={screen})}, description="Orders")
    generator = ReactGenerator(model=DomainModel(name="Orders", types={order, status}), gui_model=gui_model,
                               output_dir=str(tmp_path / "app"))
    generator.generate()
    return generator, tmp_path / "app"


def _filter_state(run_ts, column_filter, cases):
    """The filter's data type, default operator, operators and parsed value for
    each (column, current filter value)."""
    begin = column_filter.index("  // Determine column data type")
    block = column_filter[begin:column_filter.index("  const operators = getOperators();", begin)]
    return run_ts(
        "type FilterOperator = string;\n"
        "const useState = <T>(value: T): [T, (next: T) => void] => [value, () => {}];\n"
        "const useEffect = (..._args: unknown[]) => {};\n"
        "function state(column: any, currentValue: string) {\n" + block
        + "  return { dataType, defaultOperator, operators: getOperators().map((o) => o.value),"
        " parsed: parseFilterValue(currentValue) };\n}\n"
        f"const cases = {json.dumps(cases)};\n"
        "console.log(JSON.stringify(cases.map(([column, value]: [any, string]) => state(column, value))));\n")


def test_the_table_column_carries_the_literals(tmp_path):
    generator, _ = _generate(tmp_path)
    payload = json.loads(generator._build_generation_context()["components_json"])
    table = next(node for page in payload["pages"] for node in page["components"] if node.get("chart"))

    status = next(column for column in table["chart"]["columns"] if column["field"] == "status")
    assert (status["type"], status["options"]) == ("enum", ["OPEN", "SHIPPED", "DELIVERED"])


def test_an_enumeration_column_is_filtered_by_choosing_a_literal(tmp_path, run_ts):
    _, out = _generate(tmp_path)
    column_filter = (out / "src/components/table/ColumnFilter.tsx").read_text(encoding="utf-8")
    enum = {"field": "status", "label": "Status", "type": "enum", "options": ["OPEN", "SHIPPED"]}
    number = {"field": "total", "label": "Total", "type": "float"}

    opened_empty, reopened, as_number = _filter_state(run_ts, column_filter, [[enum, ""], [enum, "equals:OPEN"],
                                                                              [number, ""]])
    assert opened_empty["dataType"] == "enum"
    assert opened_empty["operators"] == ["equals", "empty", "not_empty"]
    assert opened_empty["parsed"] == {"operator": "equals", "value": ""}
    assert reopened["parsed"] == {"operator": "equals", "value": "OPEN"}
    # A filter opened without a value starts on the operator its select shows.
    assert as_number["parsed"]["operator"] == as_number["operators"][0] == "eq"

    # Its value is chosen from a select of the literals, without a type label.
    flat = re.sub(r"\s+", " ", column_filter)
    value_select = flat.split(') : dataType === "enum" ? (')[1].split(") : (")[0]
    assert "<select" in value_select and "column.options!.map((option) =>" in value_select
    assert '{dataType !== "enum" && (' in flat


def test_reopening_an_empty_check_shows_that_check(tmp_path, run_ts):
    """An "is empty" filter, reopened, showed "equals" with no value, and Apply
    then sent ``contains:empty``, which matched no row."""
    _, out = _generate(tmp_path)
    column_filter = (out / "src/components/table/ColumnFilter.tsx").read_text(encoding="utf-8")
    enum = {"field": "status", "label": "Status", "type": "enum", "options": ["OPEN"]}

    parsed = [state["parsed"] for state in _filter_state(
        run_ts, column_filter, [[enum, "empty"], [enum, "not_empty"], [enum, "legacy"]])]
    assert parsed == [
        {"operator": "empty", "value": ""},
        {"operator": "not_empty", "value": ""},
        {"operator": "contains", "value": "legacy"},
    ]
