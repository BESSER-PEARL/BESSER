"""A form keeps its labels and submit button through export -> re-import.

``parse_form`` folds each ``<label>`` into ``InputField.label`` and the submit
button into ``Form.submit_label``; the B-UML -> JSON converter then emitted only
the inputs, so a re-imported form showed bare inputs and no button.
"""
import os
import tempfile

from besser.BUML.metamodel.gui import Form
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import Class, DomainModel, Metadata
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
)


def _page(*components):
    return {"pages": [{"id": "home", "name": "Home", "frames": [{"component": {
        "type": "wrapper", "components": list(components)}}]}]}


def _text(node):
    return "".join(c.get("content", "") for c in node.get("components") or [] if c.get("type") == "textnode")


def _roundtrip(gui_json):
    gui_model = process_gui_diagram(gui_json, {}, None)
    path = os.path.join(tempfile.mkdtemp(), "project.py")
    domain = DomainModel(name="D", types={Class(name="X")})
    project_to_code(Project(name="p", models=[domain, gui_model], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        entry = project_to_json(handle.read())["diagrams"]["GUINoCodeDiagram"]
    return (entry[0] if isinstance(entry, list) else entry)["model"]


def _form(model):
    return model["pages"][0]["frames"][0]["component"]["components"][0]


def _parsed_form(model):
    gui = process_gui_diagram(model, {}, None)
    return next(e for s in next(iter(gui.modules)).screens for e in s.view_elements if isinstance(e, Form))


SIGNUP = {"tagName": "form", "attributes": {"id": "f1", "data-gui-type": "Form"}, "components": [
    {"tagName": "label", "attributes": {"for": "in-name"}, "components": [{"type": "textnode", "content": "Your name"}]},
    {"tagName": "input", "attributes": {"id": "in-name", "name": "name", "type": "text"}},
    {"tagName": "label", "components": [{"type": "textnode", "content": "Email"}]},
    {"tagName": "input", "attributes": {"id": "in-email", "name": "email", "type": "email"}},
    {"tagName": "button", "attributes": {"type": "submit"}, "components": [{"type": "textnode", "content": "Sign up"}]},
]}


def test_labels_and_the_submit_button_come_back():
    children = _form(_roundtrip(_page(SIGNUP)))["components"]

    assert [c.get("tagName") for c in children] == ["label", "input", "label", "input", "button"]
    assert [_text(c) for c in children if c.get("tagName") == "label"] == ["Your name", "Email"]
    assert children[0]["attributes"]["for"] == children[1]["attributes"]["id"]
    assert _text(children[-1]) == "Sign up"
    assert children[-1]["attributes"]["type"] == "submit"


def test_a_second_roundtrip_is_stable():
    first = _roundtrip(_page(SIGNUP))
    form = _parsed_form(first)
    assert form.submit_label == "Sign up"
    assert sorted(i.label for i in form.inputFields) == ["Email", "Your name"]

    second = _roundtrip(first)

    tags = [c.get("tagName") for c in _form(second)["components"]]
    assert tags == ["label", "input", "label", "input", "button"]


def test_an_editor_input_that_draws_its_own_label_gets_no_second_one():
    wrapped = {"type": "gui-input-text", "tagName": "div", "attributes": {
        "id": "w1", "data-gui-type": "Text", "data-gui-component": "gui-input-text", "data-label": "Name"},
        "components": [{"tagName": "label", "components": [{"type": "textnode", "content": "Name"}]},
                       {"tagName": "input", "attributes": {"type": "text"}}]}
    form = {"type": "gui-form", "tagName": "form", "attributes": {"id": "f2", "data-gui-type": "Form"},
            "components": [wrapped]}

    children = _form(_roundtrip(_page(form)))["components"]

    # The export keeps the wrapper's data-gui-component, not its editor type.
    assert [c.get("tagName") for c in children] == ["div", "button"]
    assert children[0]["attributes"]["data-gui-component"] == "gui-input-text"
