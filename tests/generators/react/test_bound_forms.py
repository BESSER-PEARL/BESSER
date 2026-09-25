"""A form bound to a class must create a record in the generated app.

The processor kept only a form's inputs (its labels and submit button were
dropped) and never read the editor's ``data-source``; the generator rendered
``onSubmit={(e) => { e.preventDefault(); }}`` with a hard-coded "Submit" and
labels made from element ids. A bound form now posts to ``/<entity>/`` with
its own labels and submit text and shows the outcome.
"""
from besser.BUML.metamodel.gui import Form, InputField

CLASSES = {
    "Guest": [("full_name", "str"), ("email", "str"), ("age", "int"), ("vip", "bool")],
    "Room": [("number", "int"), ("floor", "int")],
}


def _input(name, input_type="text", **extra):
    attributes = {"type": input_type, "name": name, **extra}
    return {"tagName": "input", "type": "input", "attributes": attributes}


def _label(text, for_=None):
    attributes = {"for": for_} if for_ else {}
    return {"tagName": "label", "attributes": attributes, "components": [{"type": "textnode", "content": text}]}


def _editor_form(data_source="cls-Guest"):
    """The editor's gui-form block: bound class + labelled inputs + button."""
    return {
        "type": "gui-form", "tagName": "form",
        "attributes": {"id": "guest-form", "data-gui-type": "Form", "data-source": data_source, "method": "POST"},
        "components": [
            {"tagName": "div", "components": [_label("Your name", for_="gf-name"),
                                              _input("full_name", id="gf-name")]},
            {"tagName": "div", "components": [_label("Age"), _input("age", "number")]},
            _input("email", "email", placeholder="Email"),
            {"tagName": "button", "attributes": {"type": "submit"},
             "components": [{"type": "textnode", "content": "Register guest"}]},
        ],
    }


def _form(gui_model):
    for module in gui_model.modules:
        for screen in module.screens:
            stack = list(screen.view_elements)
            while stack:
                element = stack.pop()
                if isinstance(element, Form):
                    return element
                stack.extend(getattr(element, "view_elements", ()) or ())
    return None


def test_the_processor_binds_the_form_and_keeps_labels_and_submit_text(build_app):
    app = build_app(CLASSES, {"Register": [_editor_form()]})
    form = _form(app.gui_model)

    assert form.data_binding is not None and form.data_binding.domain_concept.name == "Guest"
    assert form.submit_label == "Register guest"
    inputs = {i.custom_attributes["name"]: i for i in form.inputFields if isinstance(i, InputField)}
    assert inputs["full_name"].label == "Your name"  # <label for=...>
    assert inputs["age"].label == "Age"  # label right before the input
    assert {name: i.data_binding.data_field.name for name, i in inputs.items()} == {
        "full_name": "full_name", "age": "age", "email": "email",
    }


def test_a_bound_form_posts_to_the_class_endpoint(build_app, jsx):
    app = build_app(CLASSES, {"Register": [_editor_form()]})
    page = app.page("Register")

    assert 'import { FormBlock } from "../components/runtime/FormBlock";' in page
    form = jsx(page, "FormBlock")
    assert 'endpoint="/guest/"' in form
    assert 'submitLabel="Register guest"' in form
    # Each control is read by its name (the input's id) and sets one attribute.
    assert '{"name": "gf_name", "field": "full_name", "type": "str"}' in form
    assert '{"name": "age", "field": "age", "type": "int"}' in form
    assert '{...{"name": "gf_name"}}' in page and '{...{"name": "age"}}' in page
    assert "e.preventDefault(); }}" not in page
    # Real labels: <label for>, the preceding label, else the placeholder.
    assert '{"Your name"}</label>' in page and '{"Age"}</label>' in page and '{"Email"}</label>' in page

    block = app.file("components", "runtime", "FormBlock.tsx")
    assert "await axios.post(url, payload);" in block
    assert 'role={status.ok ? "status" : "alert"}' in block  # success/error feedback


def test_a_form_whose_inputs_name_one_class_is_bound_without_data_source(build_app, jsx):
    form = {"tagName": "form", "components": [
        _input("number", "number", placeholder="Number"), _input("floor", "number", placeholder="Floor"),
        {"tagName": "button", "attributes": {"type": "button"}, "components": [{"type": "textnode", "content": "Add room"}]},
    ]}
    app = build_app(CLASSES, {"Rooms": [form]})

    block = jsx(app.page("Rooms"), "FormBlock")
    assert 'endpoint="/room/"' in block and 'submitLabel="Add room"' in block


def test_an_unbound_form_keeps_its_own_submit_text(build_app):
    form = {"tagName": "form", "components": [
        _input("message"), {"tagName": "button", "components": [{"type": "textnode", "content": "Send"}]},
    ]}
    app = build_app(CLASSES, {"Contact": [form]})
    page = app.page("Contact")

    assert "FormBlock" not in page
    assert '<button type="submit">{"Send"}</button>' in page


def test_the_binding_survives_export_and_reimport(build_app):
    """Export B-UML -> re-import -> process again gives the same bound form."""
    import os
    import tempfile

    from besser.BUML.metamodel.project import Project
    from besser.BUML.metamodel.structural import Metadata
    from besser.utilities.buml_code_builder.project_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        process_class_diagram,
        process_gui_diagram,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json,
    )

    rooms_form = {"tagName": "form", "components": [
        _input("number", "number"), _input("floor", "number"),
        {"tagName": "button", "components": [{"type": "textnode", "content": "Add room"}]},
    ]}
    app = build_app(CLASSES, {"Register": [_editor_form()], "Rooms": [rooms_form]})
    class_json = app.class_json
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    gui = process_gui_diagram(app.gui_json, class_json, domain)
    path = os.path.join(tempfile.mkdtemp(), "project.py")
    project_to_code(Project(name="p", models=[domain, gui], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        code = handle.read()
    assert ".data_binding = " in code and 'submit_label="Register guest"' in code

    entry = project_to_json(code)["diagrams"]["GUINoCodeDiagram"]
    reimported = (entry[0] if isinstance(entry, list) else entry)["model"]
    again = process_gui_diagram(reimported, class_json, domain)

    forms = {}
    for module in again.modules:
        for screen in module.screens:
            stack = list(screen.view_elements)
            while stack:
                element = stack.pop()
                if isinstance(element, Form):
                    forms[screen.description] = element
                stack.extend(getattr(element, "view_elements", ()) or ())
    register, rooms = forms["Register"], forms["Rooms"]
    assert register.data_binding.domain_concept.name == "Guest"
    assert register.submit_label == "Register guest"
    labels = {i.data_binding.data_field.name: i.label for i in register.inputFields}
    assert labels["full_name"] == "Your name" and labels["age"] == "Age"
    assert rooms.data_binding.domain_concept.name == "Room"
    assert {i.data_binding.data_field.name for i in rooms.inputFields} == {"number", "floor"}
