"""Radio buttons of a form that share a ``name`` are one RadioGroup input.

Each ``<input type="radio">`` became its own RadioGroup with no options (options
were read only from ``data-options``), and a ``<label>`` around a radio dropped
the radio: the generated form showed empty radio groups that could never be set.
"""
import os
import tempfile

from besser.BUML.metamodel.gui import Form, InputField, InputFieldType
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

from .utils_gui_binding_tests import class_diagram, import_gui


def _task_classes():
    """Task(title: str, status: Status, priority: str [0..1]) with Status = Open | Closed."""
    class_json = class_diagram("Task", [("title", "str"), ("status", "Status"), ("priority", "str")])
    class_json["elements"]["attr-Task-priority"]["isOptional"] = True
    class_json["elements"].update({
        "enum-Status": {"id": "enum-Status", "name": "Status", "type": "Enumeration", "owner": None,
                        "bounds": {"x": 400, "y": 0, "width": 200, "height": 100},
                        "attributes": ["lit-open", "lit-closed"], "methods": []},
        "lit-open": {"id": "lit-open", "name": "Open", "type": "ClassAttribute", "owner": "enum-Status",
                     "bounds": {"x": 400, "y": 30, "width": 200, "height": 30}},
        "lit-closed": {"id": "lit-closed", "name": "Closed", "type": "ClassAttribute", "owner": "enum-Status",
                       "bounds": {"x": 400, "y": 60, "width": 200, "height": 30}},
    })
    return class_json


def _text(content):
    return {"type": "textnode", "content": content}


def _label(*components, **attributes):
    return {"tagName": "label", "attributes": attributes,
            "components": [_text(c) if isinstance(c, str) else c for c in components]}


def _radio(name, value=None, **attributes):
    attrs = {"type": "radio", "name": name, **({"value": value} if value is not None else {}), **attributes}
    return {"tagName": "input", "attributes": attrs}


def _form(*components, form_id="task-form"):
    return {"tagName": "form", "attributes": {"id": form_id}, "components": [
        {"tagName": "label", "attributes": {"for": "title"}, "components": [_text("Title")]},
        {"tagName": "input", "attributes": {"id": "title", "name": "title"}},
        *components,
        {"tagName": "button", "attributes": {"type": "submit"}, "components": [_text("Save")]},
    ]}


def _forms(gui_model):
    forms = []
    for module in gui_model.modules:
        for screen in module.screens:
            stack = list(screen.view_elements)
            while stack:
                element = stack.pop()
                if isinstance(element, Form):
                    forms.append(element)
                stack.extend(getattr(element, "view_elements", ()) or ())
    return sorted(forms, key=lambda f: f.component_id or "")


def _radio_groups(form):
    return sorted((i for i in form.inputFields if i.field_type == InputFieldType.RadioGroup),
                  key=lambda i: i.display_order or 0)


def _options(field):
    return [(o.value, o.label) for o in field.options or []]


# Radios wrapped by their label (the HTML the editor keeps), under a caption label
STATUS_WRAPPED = [
    _label("Status"),
    _label(_radio("status", "open"), " Open"),
    _label(_radio("status", "closed", required=""), " Closed"),
]


def test_radios_sharing_a_name_are_one_radio_group_with_their_options_in_order():
    form, = _forms(import_gui([_form(*STATUS_WRAPPED)], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("open", "Open"), ("closed", "Closed")]
    assert group.label == "Status"
    assert group.required is True  # one of its radios is required
    assert group.data_binding.data_field.name == "status"
    assert len(form.inputFields) == 2  # title + the group


def test_radios_labelled_with_for_get_their_own_labels():
    """``<label for>`` names the radio by id, or (as generated pages do) by the group's name."""
    form, = _forms(import_gui([_form(
        _label("Status"),
        {"tagName": "div", "components": [_label("Open", **{"for": "status"}), _radio("status", "Open")]},
        {"tagName": "div", "components": [_label("Closed", **{"for": "status"}), _radio("status", "Closed")]},
        _radio("priority", "low", id="p-low"), _label("Low", **{"for": "p-low"}),
        _radio("priority", "high", id="p-high"), _label("High", **{"for": "p-high"}),
    )], _task_classes()))

    status, priority = _radio_groups(form)
    assert _options(status) == [("Open", "Open"), ("Closed", "Closed")]
    assert status.label == "Status"
    assert status.required is False
    assert _options(priority) == [("low", "Low"), ("high", "High")]
    assert {i.data_binding.data_field.name for i in form.inputFields} == {"title", "status", "priority"}


def test_two_groups_in_one_form_stay_two_groups():
    form, = _forms(import_gui([_form(
        *STATUS_WRAPPED,
        _label(_radio("priority", "low"), " Low"),
        _label(_radio("priority", "high"), " High"),
    )], _task_classes()))

    status, priority = _radio_groups(form)
    assert [o.value for o in status.options] == ["open", "closed"]
    assert [o.value for o in priority.options] == ["low", "high"]
    assert priority.required is False


def test_the_same_name_in_two_forms_makes_one_group_per_form():
    forms = _forms(import_gui([
        _form(*STATUS_WRAPPED, form_id="form-a"),
        _form(_label(_radio("status", "archived"), " Archived"), form_id="form-b"),
    ], _task_classes()))

    assert [[o.value for o in _radio_groups(f)[0].options] for f in forms] == [["open", "closed"], ["archived"]]


def test_a_radio_outside_the_form_is_not_merged_into_its_group():
    gui = import_gui([_form(*STATUS_WRAPPED), _label(_radio("status", "other"), " Other")], _task_classes())
    form, = _forms(gui)

    assert [o.value for o in _radio_groups(form)[0].options] == ["open", "closed"]
    screen, = next(iter(gui.modules)).screens
    outside = [e for e in screen.view_elements if not isinstance(e, Form)]
    radio, = [i for e in outside for i in getattr(e, "view_elements", ()) if isinstance(i, InputField)]
    assert radio.label == "Other" and radio.field_type == InputFieldType.RadioGroup


def test_a_radio_group_block_with_data_options_is_unchanged():
    block = {"type": "gui-input-radio", "tagName": "div", "attributes": {
        "id": "rg", "data-gui-type": "RadioGroup", "data-field-name": "priority",
        "data-options": "Low, High", "data-label": "Priority"}}
    form, = _forms(import_gui([_form(block)], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("Low", "Low"), ("High", "High")]
    assert group.label == "Priority"


def test_valueless_radios_bound_to_an_enumeration_offer_its_literals():
    form, = _forms(import_gui([_form(
        _label(_radio("status"), " First"), _label(_radio("status"), " Second"),
    )], _task_classes()))

    group, = _radio_groups(form)
    assert [o.value for o in group.options] == ["Open", "Closed"]


def test_valueless_radios_not_bound_to_an_enumeration_submit_their_label():
    form, = _forms(import_gui([_form(
        _label(_radio("priority"), " Low"), _label(_radio("priority"), " High"),
    )], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("Low", "Low"), ("High", "High")]


def test_the_group_and_its_option_values_survive_export_and_reimport():
    class_json = _task_classes()
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    gui_json = {"pages": [{"id": "p", "name": "Tasks", "frames": [{"component": {
        "type": "wrapper", "components": [_form(*STATUS_WRAPPED)]}}]}], "styles": []}
    gui = process_gui_diagram(gui_json, class_json, domain)
    path = os.path.join(tempfile.mkdtemp(), "project.py")
    project_to_code(Project(name="p", models=[domain, gui], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        code = handle.read()
    assert 'SelectOption(value="open", label="Open")' in code

    entry = project_to_json(code)["diagrams"]["GUINoCodeDiagram"]
    reimported = process_gui_diagram((entry[0] if isinstance(entry, list) else entry)["model"], class_json, domain)

    group, = _radio_groups(_forms(reimported)[0])
    assert _options(group) == [("open", "Open"), ("closed", "Closed")]
    assert group.required is True and group.label == "Status"
    assert group.data_binding.data_field.name == "status"
