"""An agent-written GUI must survive JSON -> B-UML -> JSON / code with its classes and SVG.

The agent writes classes as ``attributes.class``. Read into ``css_classes``
but also left in ``custom_attributes``, a round trip exported them twice, and
a class removed from the model came back from the stale attribute on the next
import. GrapesJS keeps a class in ``classes`` only.
"""
import os

from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
    _serialize_gui_model,
    gui_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors.utils import (
    css_class_names,
)

AGENT_GUI = {"title": "Museum", "styles": [], "pages": [{
    "id": "home", "name": "Home",
    "frames": [{"component": {"type": "wrapper", "components": [
        {"tagName": "header", "attributes": {"id": "nav", "class": "ds-nav museum-nav"}},
        {"tagName": "svg", "type": "svg", "attributes": {"id": "pic", "viewBox": "0 0 10 10"}, "components": [
            {"tagName": "text", "type": "svg-in", "attributes": {"id": "label"},
             "components": [{"type": "textnode", "content": "Fortress"}]},
        ]},
    ]}}],
}]}


def _components(editor_json):
    found = {}

    def walk(component):
        attributes = component.get("attributes") or {}
        if attributes.get("id"):
            found[attributes["id"]] = component
        for child in component.get("components") or []:
            walk(child)

    for page in editor_json["pages"]:
        walk(page["frames"][0]["component"])
    return found


def _nav(gui_model):
    screen = next(iter(next(iter(gui_model.modules)).screens))
    return next(element for element in screen.view_elements if element.component_id == "nav")


def test_classes_are_exported_once():
    nav = _components(_serialize_gui_model(process_gui_diagram(AGENT_GUI, {}, None)))["nav"]

    assert css_class_names(nav) == ["ds-nav", "museum-nav"]
    assert "class" not in nav["attributes"]


def test_a_removed_class_stays_removed_after_reimport():
    gui_model = process_gui_diagram(AGENT_GUI, {}, None)
    _nav(gui_model).css_classes = []

    reimported = process_gui_diagram(_serialize_gui_model(gui_model), {}, None)

    assert _nav(reimported).css_classes == []


def test_classes_and_svg_survive_the_code_round_trip(tmp_path):
    code_path = os.path.join(str(tmp_path), "gui.py")
    gui_model_to_code(process_gui_diagram(AGENT_GUI, {}, None), code_path)
    with open(code_path, encoding="utf-8") as f:
        components = _components(gui_buml_to_json(f.read()))

    assert css_class_names(components["nav"]) == ["ds-nav", "museum-nav"]
    assert components["pic"]["tagName"] == "svg"
    assert components["label"]["tagName"] == "text"
    assert "Fortress" in str(components["label"])
