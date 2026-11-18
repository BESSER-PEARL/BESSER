"""A table's column filter offers an enumeration's literals.

The filter treated an enumeration column as free text: the user had to type a
literal exactly, under a "(string)" label, although the column carries its
literals as ``options``.
"""
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
    ReactGenerator(model=DomainModel(name="Orders", types={order, status}), gui_model=gui_model,
                   output_dir=str(tmp_path)).generate()
    return tmp_path


def test_an_enumeration_column_is_filtered_by_choosing_a_literal(tmp_path):
    out = _generate(tmp_path)
    page = (out / "src/pages/Orders.tsx").read_text(encoding="utf-8")
    table = (out / "src/components/table/TableComponent.tsx").read_text(encoding="utf-8")
    column_filter = (out / "src/components/table/ColumnFilter.tsx").read_text(encoding="utf-8")

    # The column reaches the filter with its literals.
    assert '"type": "enum"' in page and '"OPEN",' in page
    assert "options: col.options," in table
    assert "<ColumnFilter\n                          column={column}" in table
    assert "options?: string[];" in column_filter.split("onFilterChange")[0]

    # An enumeration with literals is its own data type, filtered by exact match.
    assert 'if (type === "enum" && column.options && column.options.length > 0) {\n      return "enum";' \
        in column_filter
    enum_operators = column_filter.split('if (dataType === "enum") {\n      return [')[1].split("];")[0]
    assert '{ value: "equals", label: "equals" }' in enum_operators
    assert "contains" not in enum_operators
    assert 'dataType === "enum" ? "equals"' in column_filter

    # Its value is chosen from a select of the literals, without a type label.
    value_select = column_filter.split(') : dataType === "enum" ? (')[1].split(") : (")[0]
    assert "<select" in value_select and "{column.options!.map((option) => (" in value_select
    assert '{dataType !== "enum" && (' in column_filter
