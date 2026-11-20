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
    """Task(title: str, status: Status [0..1], priority: str [0..1]) with Status = Open | Closed."""
    class_json = class_diagram("Task", [("title", "str"), ("status", "Status"), ("priority", "str")])
    class_json["elements"]["attr-Task-status"]["isOptional"] = True
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
PRIORITY_WRAPPED = [
    _label(_radio("priority", "low"), " Low, when idle"),
    _label(_radio("priority", "high"), " High"),
]


def test_radios_sharing_a_name_are_one_radio_group_with_their_options_in_order():
    form, = _forms(import_gui([_form(*STATUS_WRAPPED)], _task_classes()))

    group, = _radio_groups(form)
    # Bound to Status: the values are its literals (the backend accepts nothing else)
    assert _options(group) == [("Open", "Open"), ("Closed", "Closed")]
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
    assert [o.value for o in status.options] == ["Open", "Closed"]
    assert [o.value for o in priority.options] == ["low", "high"]
    assert priority.required is False


def test_the_same_name_in_two_forms_makes_one_group_per_form():
    forms = _forms(import_gui([
        _form(*PRIORITY_WRAPPED, form_id="form-a"),
        _form(_label(_radio("priority", "urgent"), " Urgent"), form_id="form-b"),
    ], _task_classes()))

    assert [[o.value for o in _radio_groups(f)[0].options] for f in forms] == [["low", "high"], ["urgent"]]


def test_a_radio_outside_the_form_is_not_merged_into_its_group():
    gui = import_gui([_form(*STATUS_WRAPPED), _label(_radio("status", "other"), " Other")], _task_classes())
    form, = _forms(gui)

    assert [o.value for o in _radio_groups(form)[0].options] == ["Open", "Closed"]
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


def test_radios_of_one_name_that_each_carry_data_options_list_them_once():
    radios = [_radio("priority", value, **{"data-options": "low,high", "data-label": "Priority"})
              for value in ("low", "high")]
    form, = _forms(import_gui([_form(*radios)], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("low", "low"), ("high", "high")]
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


def _export(components, class_json):
    """The GUI JSON exported back from the B-UML project code of ``components``, and that code."""
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    gui_json = {"pages": [{"id": "p", "name": "Tasks", "frames": [{"component": {
        "type": "wrapper", "components": components}}]}], "styles": []}
    gui = process_gui_diagram(gui_json, class_json, domain)
    path = os.path.join(tempfile.mkdtemp(), "project.py")
    project_to_code(Project(name="p", models=[domain, gui], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        code = handle.read()
    entry = project_to_json(code)["diagrams"]["GUINoCodeDiagram"]
    return (entry[0] if isinstance(entry, list) else entry)["model"], code, domain


def test_the_groups_and_their_option_values_survive_export_and_reimport():
    class_json = _task_classes()
    exported, code, domain = _export([_form(*STATUS_WRAPPED, *PRIORITY_WRAPPED)], class_json)
    assert 'SelectOption(value="low", label="Low, when idle")' in code

    for _ in range(2):
        status, priority = _radio_groups(_forms(process_gui_diagram(exported, class_json, domain))[0])
        assert _options(status) == [("Open", "Open"), ("Closed", "Closed")]
        assert status.required is True and status.label == "Status"
        assert status.data_binding.data_field.name == "status"
        assert _options(priority) == [("low", "Low, when idle"), ("high", "High")]
        exported = _export(exported["pages"][0]["frames"][0]["component"]["components"], class_json)[0]


def _tree(node):
    """(tag, radio name/value, text) of the exported elements, in document order."""
    attributes = node.get("attributes") or {}
    rows = []
    if node.get("tagName") in ("label", "input"):
        text = "".join(c.get("content", "") for c in node.get("components") or [] if c.get("type") == "textnode")
        rows.append((node["tagName"], attributes.get("name"), attributes.get("value"), text.strip()))
    for child in node.get("components") or []:
        rows.extend(_tree(child))
    return rows


def test_an_exported_group_is_a_caption_and_one_labelled_radio_per_option():
    """The editor canvas shows radios, not ``data-options``: the group is written as radios."""
    exported, _, _ = _export([_form(*STATUS_WRAPPED)], _task_classes())

    rows = _tree(exported["pages"][0]["frames"][0]["component"]["components"][0])
    assert rows[2:] == [
        ("label", None, None, "Status"),
        ("label", None, None, "Open"), ("input", "status", "Open", ""),
        ("label", None, None, "Closed"), ("input", "status", "Closed", ""),
    ]


def test_an_editor_radio_block_is_exported_as_the_block():
    block = {"type": "gui-input-radio", "tagName": "div", "attributes": {
        "id": "rg", "data-gui-type": "RadioGroup", "data-field-name": "priority",
        "data-options": "Low, High", "data-label": "Priority"}}
    exported, _, _ = _export([_form(block)], _task_classes())

    form = exported["pages"][0]["frames"][0]["component"]["components"][0]
    node, = [c for c in form["components"] if (c.get("attributes") or {}).get("data-gui-type") == "RadioGroup"]
    assert node["attributes"]["data-options"] == "Low, High"
    assert not [row for row in _tree(form) if row[0] == "input"][1:]  # only the title input


def test_an_editor_radio_block_without_options_bound_to_an_enumeration_offers_its_literals():
    block = {"type": "gui-input-radio", "tagName": "div", "attributes": {
        "id": "rg", "data-gui-type": "RadioGroup", "data-field-name": "status", "data-options": ""}}
    form, = _forms(import_gui([_form(block)], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("Open", "Open"), ("Closed", "Closed")]


def test_values_bound_to_an_enumeration_become_its_literals():
    """Exact, else case-insensitive, else by the label; a value matching nothing is dropped."""
    form, = _forms(import_gui([_form(
        _label(_radio("status", "OPEN"), " Open"),
        _label(_radio("status", "2"), " Closed"),
        _label(_radio("status", "later"), " Later"),
    )], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("Open", "Open"), ("Closed", "Closed")]


def test_values_matching_no_literal_are_replaced_by_the_literals():
    form, = _forms(import_gui([_form(
        _label(_radio("status", "a"), " First"), _label(_radio("status", "b"), " Second"),
    )], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("Open", "Open"), ("Closed", "Closed")]


def test_a_radio_inside_an_element_of_its_label_is_kept():
    form, = _forms(import_gui([_form(
        _label("Status"),
        _label({"tagName": "span", "components": [_radio("priority", "low"), _text(" Low")]}),
        _label(_radio("priority", "high"), " High"),
    )], _task_classes()))

    group, = _radio_groups(form)
    assert _options(group) == [("low", "Low"), ("high", "High")]
    assert group.label == "Status"


def test_options_written_as_a_json_list_are_read():
    block = {"type": "gui-input-dropdown", "tagName": "div", "attributes": {
        "data-gui-type": "Dropdown", "data-field-name": "priority",
        "data-options": '["low", "a, b"]', "data-option-labels": '["Low", "A, then B"]'}}
    form, = _forms(import_gui([_form(block)], _task_classes()))

    dropdown, = [i for i in form.inputFields if i.field_type == InputFieldType.Dropdown]
    assert _options(dropdown) == [("low", "Low"), ("a, b", "A, then B")]


def test_options_dropped_or_replaced_for_an_enumeration_are_reported(caplog):
    with caplog.at_level("WARNING"):
        _forms(import_gui([_form(
            _label(_radio("status", "open"), " Open"), _label(_radio("status", "pending"), " Pending"),
            _label(_radio("status", "closed"), " Closed"),
        )], _task_classes()))
        _forms(import_gui([_form(
            _label(_radio("status", "draft"), " Draft"), _label(_radio("status", "published"), " Published"),
        )], _task_classes()))

    dropped, replaced = [r.getMessage() for r in caplog.records if "Status" in r.getMessage()]
    assert "Field 'status'" in dropped and "(Open, Closed)" in dropped
    assert dropped.endswith("it ignores the options pending.")
    assert replaced.endswith("it offers its literals instead of the options draft, published.")
