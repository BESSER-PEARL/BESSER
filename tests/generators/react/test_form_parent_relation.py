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
    assert '{"Belongs to"}<span' in page  # the order is mandatory: its label is marked
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


def test_a_form_bound_by_its_input_names_gets_the_parent_select_too(build_app, jsx):
    """No ``data-source``: the inputs name the class, and its mandatory parent is still asked for."""
    form = {"tagName": "form", "components": [
        _input("quantity", "number"),
        {"tagName": "button", "components": [{"type": "textnode", "content": "Add line"}]},
    ]}
    app = build_app(CLASSES, {"Lines": [form]}, ASSOCIATIONS)
    page = app.page("Lines")

    assert _gui_form(app.gui_model).data_binding.domain_concept.name == "OrderLine"
    assert 'endpoint="/orderline/"' in jsx(page, "FormBlock")
    assert 'entity="Order"' in jsx(page, "LookupSelect") and "required={true}" in jsx(page, "LookupSelect")


def test_a_designed_parent_control_does_not_stop_the_form_from_being_bound(build_app, jsx):
    form = {"tagName": "form", "components": [
        _input("quantity", "number"),
        {"tagName": "select", "attributes": {"name": "order"}},
        {"tagName": "button", "components": [{"type": "textnode", "content": "Add line"}]},
    ]}
    app = build_app(CLASSES, {"Lines": [form]}, ASSOCIATIONS)
    page = app.page("Lines")

    assert _gui_form(app.gui_model).data_binding.domain_concept.name == "OrderLine"
    assert page.count("<LookupSelect") == 1 and 'entity="Order"' in jsx(page, "LookupSelect")


def _unbound_form(*controls):
    return {"tagName": "form", "components": [
        *controls, {"tagName": "button", "components": [{"type": "textnode", "content": "Save"}]},
    ]}


def test_a_class_matched_by_attributes_wins_over_one_matched_through_a_parent(build_app, jsx, caplog):
    """``author`` is an attribute of Book and a parent of Review: the inputs name Book's attributes."""
    classes = {"Book": [("title", "str"), ("author", "str")], "Review": [("title", "str")],
               "Writer": [("name", "str")]}
    associations = [("wrote", ("Review", "reviews", "0..*"), ("Writer", "author", "1"))]
    app = build_app(classes, {"New": [_unbound_form(_input("title"), _input("author"))]}, associations)

    assert _gui_form(app.gui_model).data_binding.domain_concept.name == "Book"
    assert 'endpoint="/book/"' in jsx(app.page("New"), "FormBlock")
    assert "match the classes" not in caplog.text


def test_a_form_matching_two_classes_is_left_unbound_with_a_warning(build_app, caplog):
    classes = {"Book": [("title", "str")], "Album": [("title", "str")]}
    app = build_app(classes, {"New": [_unbound_form(_input("title"))]})

    assert _gui_form(app.gui_model).data_binding is None
    assert "match the classes Album, Book" in caplog.text


def test_an_input_naming_only_a_parent_does_not_make_a_create_form(build_app):
    """A search box called ``order`` is not a form creating a record of a class whose parent is ``order``
    (and that needs no attribute to be created)."""
    classes = {"Order": [("reference", "str")], "Note": []}
    associations = [("notes", ("Note", "notes", "0..*"), ("Order", "order", "0..1"))]
    app = build_app(classes, {"Search": [_unbound_form(_input("order"))]}, associations)

    assert _gui_form(app.gui_model).data_binding is None


def test_a_designed_parent_control_may_be_named_after_the_key_it_posts(build_app, jsx):
    """``order_id`` and ``orderId`` are the order's control: no second, automatic select is added."""
    for name in ("order_id", "orderId"):
        form = _form(_input("quantity", "number"), {"tagName": "select", "attributes": {"name": name}})
        app = build_app(CLASSES, {"Lines": [form]}, ASSOCIATIONS)
        page = app.page("Lines")

        assert {i.data_binding.data_field.name for i in _gui_form(app.gui_model).inputFields} == {
            "quantity", "order"}, name
        assert page.count("<LookupSelect") == 1 and "<select" not in page, name
        assert f'name="{name}"' in jsx(page, "LookupSelect", 'entity="Order"')


def test_a_designed_control_of_a_mandatory_parent_is_required(build_app, jsx):
    form = _form(_input("quantity", "number"), {"tagName": "select", "attributes": {"name": "order"}})
    app = build_app(CLASSES, {"Lines": [form]}, ASSOCIATIONS)

    assert "required={true}" in jsx(app.page("Lines"), "LookupSelect", 'entity="Order"')


def test_the_designed_selects_of_an_association_class_form_are_its_two_ends(build_app, jsx):
    classes = {"Student": [("name", "str")], "Course": [("title", "str")], "Enrolment": [("grade", "int")]}
    associations = [("enrols", ("Student", "students", "0..*"), ("Course", "courses", "0..*"))]
    form = _form(_input("grade", "number"), {"tagName": "select", "attributes": {"name": "students"}},
                 {"tagName": "select", "attributes": {"name": "courses"}}, data_source="cls-Enrolment")
    app = build_app(classes, {"Enrol": [form]}, associations, link_classes={"enrols": "Enrolment"})
    page = app.page("Enrol")

    assert page.count("<LookupSelect") == 2 and "<select" not in page
    for entity, name in (("Student", "students"), ("Course", "courses")):
        select = jsx(page, "LookupSelect", f'entity="{entity}"')
        assert f'name="{name}"' in select and "required={true}" in select
