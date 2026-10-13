"""Build a React app from editor-shaped GUI JSON (class diagram + GrapesJS pages)."""
import os

import pytest

from besser.generators.react import ReactGenerator
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)


def class_diagram(classes):
    """Apollon class diagram JSON. ``classes`` = {Name: [(attr, type), ...]};
    the class id is ``cls-<Name>`` and an attribute id ``attr-<Name>-<attr>``."""
    elements = {}
    for index, (class_name, attributes) in enumerate(classes.items()):
        class_id = f"cls-{class_name}"
        attribute_ids = []
        for attr_name, attr_type in attributes:
            attr_id = f"attr-{class_name}-{attr_name}"
            attribute_ids.append(attr_id)
            elements[attr_id] = {
                "id": attr_id, "name": attr_name, "type": "ClassAttribute", "owner": class_id,
                "bounds": {"x": 0, "y": 0, "width": 200, "height": 30}, "visibility": "public",
                "attributeType": attr_type, "isOptional": False, "isDerived": False, "isId": False,
            }
        elements[class_id] = {
            "id": class_id, "name": class_name, "type": "Class", "owner": None,
            "bounds": {"x": index * 300, "y": 0, "width": 220, "height": 200},
            "attributes": attribute_ids, "methods": [],
        }
    return {"version": "3.0.0", "type": "ClassDiagram", "size": {"width": 2000, "height": 1000},
            "interactive": {"elements": {}, "relationships": {}}, "elements": elements,
            "relationships": {}, "assessments": {}}


def gui_json(pages):
    """GrapesJS project JSON; ``pages`` = {Name: [component, ...]}."""
    return {"pages": [
        {"id": f"page-{name.lower()}", "name": name,
         "frames": [{"component": {"type": "wrapper", "components": components}}]}
        for name, components in pages.items()
    ], "styles": []}


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class GeneratedApp:
    def __init__(self, gui_model, src, class_json=None, gui_json=None):
        self.gui_model = gui_model
        self.src = src
        self.class_json = class_json
        self.gui_json = gui_json

    def page(self, name):
        return _read(os.path.join(self.src, "pages", f"{name}.tsx"))

    def file(self, *parts):
        return _read(os.path.join(self.src, *parts))


@pytest.fixture
def build_app(tmp_path):
    def build(classes, pages):
        class_json = class_diagram(classes)
        project_gui = gui_json(pages)
        domain = process_class_diagram({"title": "Domain", "model": class_json})
        gui_model = process_gui_diagram(project_gui, class_json, domain)
        ReactGenerator(model=domain, gui_model=gui_model, output_dir=str(tmp_path)).generate()
        return GeneratedApp(gui_model, os.path.join(str(tmp_path), "src"), class_json, project_gui)

    return build


def jsx_tag(source, tag, contains=None):
    """The first ``<tag ... />`` (or ``<tag ...>``) of the page source, optionally
    the first one containing ``contains``."""
    start = 0
    while True:
        begin = source.index(f"<{tag}", start)
        depth, i = 0, begin
        while True:
            char = source[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == ">" and depth == 0:
                break
            i += 1
        snippet = source[begin:i + 1]
        if contains is None or contains in snippet:
            return snippet
        start = i


@pytest.fixture
def jsx():
    return jsx_tag
