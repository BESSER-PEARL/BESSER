"""The table's Add/Edit dialog keeps empty optional values empty."""
import json
import re

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import Table
from besser.BUML.metamodel.structural import Class, DomainModel, Multiplicity, Property, StringType
from besser.generators.react import ReactGenerator

CLASSES = {"Order": [
    ("reference", "str"),
    ("note", "str", {"isOptional": True}),
    ("quantity", "int", {"isOptional": True}),
    ("shipped_on", "date", {"isOptional": True}),
]}
PAGES = {"Orders": [
    {"type": "table", "attributes": {"id": "orders", "data-source": "cls-Order", "action-buttons": "true"}},
]}


def _flat(source):
    return re.sub(r"\s+", " ", source)


def _submit_handler(table):
    start = table.index("// Process form values based on column types")
    return _flat(table[start:table.index("if (modalMode === 'add') {", start)])


def _form_columns(generator):
    payload = json.loads(generator._build_generation_context()["components_json"])
    tables = [node for page in payload["pages"] for node in page["components"] if "formColumns" in node.get("chart", {})]
    return {column["field"]: column for column in tables[0]["chart"]["formColumns"]}


def test_only_an_optional_attribute_is_nullable():
    """The backend accepts null only for an ``is_optional`` attribute; a 0..1
    attribute that is not optional is not required in the dialog but rejects null."""
    reference = Property(name="reference", type=StringType)
    note = Property(name="note", type=StringType, is_optional=True)
    code = Property(name="code", type=StringType, multiplicity=Multiplicity(0, 1))
    order = Class(name="Order", attributes={reference, note, code})
    table = Table(name="OrderTable", title="Orders", action_buttons=True,
                  data_binding=DataBinding(name="order_binding", domain_concept=order))
    gui_model = GUIModel(name="OrdersApp", package="", versionCode="1", versionName="1.0", description="Orders",
                         modules={Module(name="M", screens={Screen(name="Orders", description="Orders",
                                                                   view_elements={table}, is_main_page=True)})})
    columns = _form_columns(ReactGenerator(model=DomainModel(name="Orders", types={order}), gui_model=gui_model))

    assert (columns["reference"]["required"], columns["reference"]["nullable"]) == (True, False)
    assert (columns["note"]["required"], columns["note"]["nullable"]) == (False, True)
    assert (columns["code"]["required"], columns["code"]["nullable"]) == (False, False)


def test_an_empty_nullable_value_is_sent_as_null(build_app):
    """Saving a record whose optional note was null, without touching it, sent
    ``"note": ""``; an empty optional number was sent as 0."""
    app = build_app(CLASSES, PAGES)
    table = app.file("components", "table", "TableComponent.tsx")
    assert "nullable: Boolean((col as any).nullable)," in _flat(table)

    submit = _submit_handler(table)
    sends_null = submit.index(
        "if (col.nullable && col.type !== 'list' && col.type !== 'bool' && col.type !== 'boolean' "
        "&& (value === undefined || value === null || value === '')) { "
        "// An optional field left empty has no value: null, not \"\" or 0 processedValues[col.field] = null;")
    # Before the coercions that turned it into "" (text, dates) or 0 (numbers).
    assert sends_null < submit.index("processedValues[col.field] = isNaN(intValue) ? 0 : intValue;")
    assert sends_null < submit.index("processedValues[col.field] = value;")


def test_dialog_labels_do_not_show_type_names(build_app):
    """A field was labelled "Quantity (int)" or "Shipped on (date)"; the input
    already says what it takes."""
    table = build_app(CLASSES, PAGES).file("components", "table", "TableComponent.tsx")
    label = table[table.rindex("<label htmlFor={`modal-input-${col.field}`}"):]
    label = label[:label.index("</label>")]

    assert "col.type)" not in label and ": col.type" not in label
    assert "{col.type === 'list' && (" in label and "(comma-separated)" in label
