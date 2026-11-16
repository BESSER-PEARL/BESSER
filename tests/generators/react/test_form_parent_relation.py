"""A bound form can set the record its new record must belong to.

A class with a mandatory to-one association (an order line must reference
its order) cannot be created from its attributes alone: the backend refuses
the request. The form gets a select of the related records.
"""
import json

from besser.BUML.metamodel.gui import Form

CLASSES = {
    "Order": [("reference", "str")],
    "OrderLine": [("quantity", "int")],
    "Product": [("label", "str")],
}
ASSOCIATIONS = [
    ("lines", ("OrderLine", "lines", "0..*"), ("Order", "order", "1")),
    ("product", ("OrderLine", "soldIn", "0..*"), ("Product", "product", "0..1")),
]


def _input(name, input_type="text", **extra):
    return {"tagName": "input", "attributes": {"type": input_type, "name": name, **extra}}


def _form(*controls, data_source="cls-OrderLine"):
    return {"tagName": "form", "attributes": {"id": "line-form", "data-source": data_source}, "components": [
        *controls,
        {"tagName": "button", "attributes": {"type": "submit"},
         "components": [{"type": "textnode", "content": "Add line"}]},
    ]}


def _fields(block):
    return json.JSONDecoder().raw_decode(block[block.index("fields={") + len("fields={"):])[0]


def _gui_form(gui_model):
    for module in gui_model.modules:
        for screen in module.screens:
            stack = list(screen.view_elements)
            while stack:
                element = stack.pop()
                if isinstance(element, Form):
                    return element
                stack.extend(getattr(element, "view_elements", ()) or ())
    return None


def test_a_missing_mandatory_parent_becomes_a_select(build_app, jsx):
    app = build_app(CLASSES, {"Lines": [_form(_input("quantity", "number"))]}, ASSOCIATIONS)
    page = app.page("Lines")

    assert 'import { LookupSelect } from "../components/runtime/LookupSelect";' in page
    select = jsx(page, "LookupSelect")
    for prop in ('id="order"', 'name="order"', 'entity="Order"', 'lookupField="reference"',
                 'targetField="id"', "required={true}"):
        assert prop in select, prop
    assert '<label htmlFor="order"' in page and '{"Order"}' in page
    # The select posts the chosen order's key as the association end.
    assert {"name": "order", "field": "order", "type": "int"} in _fields(jsx(page, "FormBlock"))
    # An optional relationship is not forced on the form.
    assert 'entity="Product"' not in page


def test_a_designed_parent_control_is_used_instead_of_a_second_one(build_app, jsx):
    form = _form(_input("quantity", "number"),
                 {"tagName": "label", "attributes": {"for": "line-order"},
                  "components": [{"type": "textnode", "content": "Belongs to"}]},
                 {"tagName": "select", "attributes": {"id": "line-order", "name": "order"}},
                 {"tagName": "select", "attributes": {"id": "line-product", "name": "product"}})
    app = build_app(CLASSES, {"Lines": [form]}, ASSOCIATIONS)
    page = app.page("Lines")

    bound = {i.name: i.data_binding.data_field.name for i in _gui_form(app.gui_model).inputFields}
    assert bound == {"quantity": "quantity", "line_order": "order", "line_product": "product"}
    assert page.count("<LookupSelect") == 2
    assert 'name="line_order"' in jsx(page, "LookupSelect", 'entity="Order"')
    assert '{"Belongs to"}</label>' in page
    assert "required" not in jsx(page, "LookupSelect", 'entity="Product"')
    fields = _fields(jsx(page, "FormBlock"))
    assert {"name": "line_order", "field": "order", "type": "int"} in fields
    assert {"name": "line_product", "field": "product", "type": "int"} in fields


def test_a_form_without_a_mandatory_parent_is_unchanged(build_app):
    form = _form(_input("reference"), data_source="cls-Order")
    app = build_app(CLASSES, {"Orders": [form]}, ASSOCIATIONS)

    assert "LookupSelect" not in app.page("Orders")


def test_the_select_and_the_table_dialog_share_one_lookup(build_app):
    table = {"type": "table", "attributes": {"id": "lines-table", "data-source": "cls-OrderLine"}}
    app = build_app(CLASSES, {"Lines": [table, _form(_input("quantity", "number"))]}, ASSOCIATIONS)

    shared = "fetchLookupOptions, lookupOptionLabel, lookupOptionValue } from"
    assert shared in app.file("components", "runtime", "LookupSelect.tsx")
    assert shared in app.file("components", "table", "TableComponent.tsx")
    assert "export const fetchLookupOptions" in app.file("components", "runtime", "lookup.ts")
