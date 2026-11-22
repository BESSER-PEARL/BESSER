"""Helpers that import GUI components bound to one class through the editor's JSON converters."""
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)


def class_diagram(class_name, attributes):
    """Apollon class diagram with one class (id ``cls-<name>``) and its ``(name, type)``
    attributes (ids ``attr-<class>-<name>``)."""
    class_id = f"cls-{class_name}"
    elements = {}
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
        "bounds": {"x": 0, "y": 0, "width": 220, "height": 200},
        "attributes": attribute_ids, "methods": [],
    }
    return {"version": "3.0.0", "type": "ClassDiagram", "size": {"width": 2000, "height": 1000},
            "interactive": {"elements": {}, "relationships": {}}, "elements": elements,
            "relationships": {}, "assessments": {}}


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
