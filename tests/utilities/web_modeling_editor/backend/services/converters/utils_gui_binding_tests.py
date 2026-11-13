"""Helpers that import GUI components bound to one class through the editor's JSON converters."""
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)


def class_diagram(class_name, attributes):
    """v4 class diagram with one class (node id ``cls-<name>``) and its ``(name, type)``
    attributes (row ids ``attr-<class>-<name>``)."""
    rows = [
        {"id": f"attr-{class_name}-{attr_name}", "name": attr_name, "visibility": "public",
         "attributeType": attr_type, "isOptional": False, "isDerived": False, "isId": False}
        for attr_name, attr_type in attributes
    ]
    node = {
        "id": f"cls-{class_name}", "type": "class", "position": {"x": 0, "y": 0},
        "width": 220, "height": 200, "measured": {"width": 220, "height": 200},
        "data": {"name": class_name, "attributes": rows, "methods": []},
    }
    return {"version": "4.0.0", "type": "ClassDiagram", "title": "Domain",
            "nodes": [node], "edges": [], "assessments": {}}


def import_gui(components, class_json):
    """The GUI model of one dashboard page holding ``components`` (editor JSON)."""
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    project_gui = {"pages": [{
        "id": "page-dashboard", "name": "Dashboard",
        "frames": [{"component": {"type": "wrapper", "components": components}}],
    }], "styles": []}
    return process_gui_diagram(project_gui, class_json, domain)


def elements_by_id(gui_model):
    """The GUI model's view elements, keyed by their editor component id."""
    return {
        element.component_id: element
        for module in gui_model.modules
        for screen in module.screens
        for element in screen.view_elements
    }
