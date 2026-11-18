"""The table's Add/Edit dialog keeps empty optional values empty."""

CLASSES = {"Order": [
    ("reference", "str"),
    ("note", "str", {"isOptional": True}),
    ("quantity", "int", {"isOptional": True}),
    ("shipped_on", "date", {"isOptional": True}),
]}
PAGES = {"Orders": [
    {"type": "table", "attributes": {"id": "orders", "data-source": "cls-Order", "action-buttons": "true"}},
]}


def _submit_handler(table):
    start = table.index("// Process form values based on column types")
    return table[start:table.index("if (modalMode === 'add') {", start)]


def test_an_empty_optional_value_is_sent_as_null(build_app):
    """Saving a record whose optional note was null, without touching it, sent
    ``"note": ""``; an empty optional number was sent as 0."""
    app = build_app(CLASSES, PAGES)
    assert '"field": "note"' in app.page("Orders") and '"required": false' in app.page("Orders")

    submit = _submit_handler(app.file("components", "table", "TableComponent.tsx"))
    empty_optional = submit.index(
        "if (!col.required && col.type !== 'list' && col.type !== 'bool' && col.type !== 'boolean'\n"
        "                          && (value === undefined || value === null || value === '')) {")
    sends_null = submit.index("processedValues[col.field] = null;", empty_optional)
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
