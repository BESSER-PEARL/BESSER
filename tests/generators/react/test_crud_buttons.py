"""Create/update/delete buttons of a GUI model must act in the generated app.

The editor's action button stores ``data-action-type`` = create|update|delete,
the entity's class *id* in ``data-entity-class`` and optionally the table in
``data-instance-source``. The processor parsed them, but resolved the action's
class by *name* (so it stayed unset) and the instance source by the element's
sanitized name (so ``table-books`` never matched ``table_books``); the React
generator then rendered a plain ``<button>`` with no click handler.
"""
from besser.BUML.metamodel.gui import Button
from besser.BUML.metamodel.gui.dashboard import Table
from besser.BUML.metamodel.gui.events_actions import Create, Delete

CLASSES = {"Book": [("title", "str"), ("pages", "int")]}


def _table(table_id="table-books"):
    return {"type": "table", "attributes": {"id": table_id, "chart-title": "Books", "data-source": "cls-Book"}}


def _crud_button(action, label, table=None, confirm=False):
    attributes = {"id": f"btn-{action}", "button-label": label, "data-action-type": action,
                  "data-entity-class": "cls-Book"}
    if table:
        attributes["data-instance-source"] = table
    component = {"type": "action-button", "tagName": "button", "attributes": attributes,
                 "components": [{"type": "textnode", "content": label}]}
    if confirm:
        component["confirmation-required"] = "true"
        component["confirmation-message"] = "Delete this book?"
    return component


def _buttons(gui_model):
    found = {}

    def walk(elements):
        for element in elements:
            if isinstance(element, Button):
                found[element.component_id] = element
            walk(getattr(element, "view_elements", ()) or ())

    for module in gui_model.modules:
        for screen in module.screens:
            walk(screen.view_elements)
    return found


def test_the_processor_resolves_the_class_id_and_the_table(build_app):
    app = build_app(CLASSES, {"Books": [
        _table(), _crud_button("create", "New book"), _crud_button("delete", "Remove", table="table-books"),
    ]})
    buttons = _buttons(app.gui_model)

    create = buttons["btn-create"]
    action = next(iter(next(iter(create.events)).actions))
    assert isinstance(action, Create)
    assert action.target_class is not None and action.target_class.name == "Book"
    assert create.entity_class.name == "Book"

    delete = buttons["btn-delete"]
    assert isinstance(next(iter(next(iter(delete.events)).actions)), Delete)
    assert isinstance(delete.instance_source, Table)
    assert delete.instance_source.component_id == "table-books"


def test_crud_buttons_drive_the_bound_table(build_app, jsx):
    app = build_app(CLASSES, {"Books": [
        _table(),
        _crud_button("create", "New book"),
        _crud_button("update", "Edit", table="table-books"),
        _crud_button("delete", "Remove", table="table-books", confirm=True),
    ]})
    page = app.page("Books")

    assert 'import { CrudButton } from "../components/runtime/CrudButton";' in page
    create = jsx(page, "CrudButton", 'action="create"')
    assert 'tableId="table-books"' in create and 'label="New book"' in create
    assert "targetPath" not in create
    assert 'tableId="table-books"' in jsx(page, "CrudButton", 'action="update"')
    delete = jsx(page, "CrudButton", 'action="delete"')
    assert 'tableId="table-books"' in delete and 'confirmMessage="Delete this book?"' in delete
    assert "<button" not in page  # no inert button left

    # The table the buttons name posts/puts/deletes on the Book endpoint.
    table = jsx(page, "TableBlock")
    assert 'id="table-books"' in table and '"endpoint": "/book/"' in table


def test_the_runtime_wires_the_button_to_the_table_dialog_and_row(build_app):
    app = build_app(CLASSES, {"Books": [_table(), _crud_button("delete", "Remove", table="table-books")]})

    context = app.file("contexts", "TableContext.tsx")
    assert "runTableAction" in context and "registerTableActions" in context
    assert "const row = selectedRows[tableId];" in context

    table = app.file("components", "table", "TableComponent.tsx")
    assert "useEffect(() => registerTableActions(id, {" in table
    assert "remove: deleteRow," in table
    assert "await axios.delete(fullUrl);" in table

    button = app.file("components", "runtime", "CrudButton.tsx")
    assert "runTableAction(tableId, action)" in button
    assert "window.confirm(confirmMessage)" in button


def test_a_create_button_without_a_table_on_its_page_opens_the_table_page(build_app, jsx):
    app = build_app(CLASSES, {
        "Home": [_crud_button("create", "Add a book")],
        "Books": [_table()],
    })

    create = jsx(app.page("Home"), "CrudButton")
    assert 'tableId="table-books"' in create
    assert 'targetPath="/books"' in create
