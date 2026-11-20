"""A ``<select>`` keeps the choices of its ``<option>`` children.

Options were read only from ``data-options``: an authored select became a
Dropdown with no options and the generated page offered nothing to choose.
"""
import os
import tempfile

from besser.BUML.metamodel.gui import InputFieldType
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

from .test_gui_radio_groups import _form, _forms, _options, _task_classes
from .utils_gui_binding_tests import import_gui


def _option(text=None, **attributes):
    option = {"tagName": "option", "attributes": attributes}
    if text is not None:
        option["components"] = [{"type": "textnode", "content": text}]
    return option


def _select(name, *options, **attributes):
    return {"tagName": "select", "attributes": {"id": name, "name": name, **attributes}, "components": list(options)}


PRIORITY = _select(
    "priority",
    _option("Choose a priority", value=""),
    _option("Low", value="low"),
    {"tagName": "option", "attributes": {"value": "normal", "selected": ""}, "content": "Normal"},
    _option("Unavailable", value="urgent", disabled=""),
    {"tagName": "optgroup", "attributes": {"label": "More"}, "components": [_option("High")]},
)


def _dropdown(form):
    dropdown, = [i for i in form.inputFields if i.field_type == InputFieldType.Dropdown]
    return dropdown


def test_a_select_offers_its_options_in_order():
    """Value attribute else the text; no blank or disabled choice; optgroups are read."""
    dropdown = _dropdown(_forms(import_gui([_form(PRIORITY)], _task_classes()))[0])

    assert _options(dropdown) == [("low", "Low"), ("normal", "Normal"), ("High", "High")]
    assert dropdown.data_binding.data_field.name == "priority"


def test_data_options_win_over_the_option_children():
    select = _select("priority", _option("Low", value="low"), **{"data-options": "A, B"})
    dropdown = _dropdown(_forms(import_gui([_form(select)], _task_classes()))[0])

    assert _options(dropdown) == [("A", "A"), ("B", "B")]


def test_a_select_bound_to_an_enumeration_without_options_offers_its_literals():
    select = _select("status", _option("Choose", value=""))
    dropdown = _dropdown(_forms(import_gui([_form(select)], _task_classes()))[0])

    assert _options(dropdown) == [("Open", "Open"), ("Closed", "Closed")]


def test_the_options_survive_export_and_reimport():
    class_json = _task_classes()
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    gui_json = {"pages": [{"id": "p", "name": "Tasks", "frames": [{"component": {
        "type": "wrapper", "components": [_form(PRIORITY)]}}]}], "styles": []}
    gui = process_gui_diagram(gui_json, class_json, domain)
    path = os.path.join(tempfile.mkdtemp(), "project.py")
    project_to_code(Project(name="p", models=[domain, gui], metadata=Metadata(description="d")), path)
    with open(path, encoding="utf-8") as handle:
        code = handle.read()

    entry = project_to_json(code)["diagrams"]["GUINoCodeDiagram"]
    reimported = process_gui_diagram((entry[0] if isinstance(entry, list) else entry)["model"], class_json, domain)

    dropdown = _dropdown(_forms(reimported)[0])
    assert _options(dropdown) == [("low", "Low"), ("normal", "Normal"), ("High", "High")]
    assert dropdown.data_binding.data_field.name == "priority"


def test_a_select_bound_to_an_enumeration_offers_its_literals_for_its_values():
    select = _select("status", _option("Open", value="OPEN"), _option("Closed", value="shut"), _option("Other", value="x"))
    dropdown = _dropdown(_forms(import_gui([_form(select)], _task_classes()))[0])

    assert _options(dropdown) == [("Open", "Open"), ("Closed", "Closed")]
