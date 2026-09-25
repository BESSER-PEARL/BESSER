"""A ``readOnly`` form column is shown in the create/edit dialog, never sent.

Generated pages carry table ``formColumns``; a column marked
``"readOnly": true`` (a server-computed value, e.g. a booking's total bill)
used to render as an editable input and go out in the POST/PUT body, because
TableComponent ignored the flag.
"""
import os
import re

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import FieldColumn, Table
from besser.BUML.metamodel.structural import Class, DomainModel, FloatType, Property, StringType
from besser.generators.react import ReactGenerator


def _table_component(tmp_path):
    code = Property(name="code", type=StringType)
    bill = Property(name="bill", type=FloatType)
    booking = Class(name="Booking", attributes={code, bill})
    table = Table(
        name="bookings",
        label="Bookings",
        columns=[FieldColumn(label="Code", field=code), FieldColumn(label="Bill", field=bill)],
        data_binding=DataBinding(domain_concept=booking),
    )
    screen = Screen(name="Bookings", description="Bookings", view_elements={table}, is_main_page=True)
    gui = GUIModel(name="G", package="", versionCode="1", versionName="1", description="",
                   modules={Module(name="M", screens={screen})})
    ReactGenerator(model=DomainModel(name="Hotel", types={booking}), gui_model=gui,
                   output_dir=str(tmp_path)).generate()
    path = os.path.join(str(tmp_path), "src", "components", "table", "TableComponent.tsx")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _block(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def test_the_read_only_flag_survives_column_normalisation(tmp_path):
    source = _table_component(tmp_path)

    assert "readOnly?: boolean;" in _block(source, "interface TableColumn", "}")
    normalise = _block(source, "const normalizeOptionColumns", "const normalizedRows")
    assert "readOnly: Boolean((col as any).readOnly ?? (col as any).read_only)" in normalise


def test_a_read_only_column_is_left_out_of_the_request_body(tmp_path):
    source = _table_component(tmp_path)

    payload = _block(source, "const processedValues", "if (modalMode === 'add')")
    skip = payload.index("if (col.readOnly)")
    assert re.match(r"if \(col\.readOnly\) \{\s*(//[^\n]*\s*)?return;", payload[skip:])
    assert skip < payload.index("processedValues[")

    # ... and is not reported missing when it is required.
    validation = _block(source, "const missingFields", "if (missingFields.length > 0)")
    assert validation.index("if (col.readOnly)") < validation.index("if (col.required)")


def test_a_read_only_column_renders_display_only(tmp_path):
    source = _table_component(tmp_path)

    field_node = _block(source, "const fieldNode = (() => {", "// For lookup columns, render a select dropdown")
    assert "if (col.readOnly)" in field_node
    readonly_input = _block(field_node, "<input", "/>")
    assert "readOnly" in readonly_input
    assert "onChange" not in readonly_input
    assert "value={readOnlyText(col)" in readonly_input
