"""Round trips through the v4 converters keep what the editor saved.

Each test records a loss or duplication found by round-tripping editor
templates and fixtures: JSON -> B-UML -> JSON, and project JSON -> project
``.py`` -> project JSON.
"""

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)


def _node(node_id, node_type, data):
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "width": 160, "height": 100, "data": data}


def _class_diagram(nodes, edges=()):
    return {"title": "Library", "model": {"version": "4.0.0", "type": "ClassDiagram",
                                         "nodes": list(nodes), "edges": list(edges)}}


def test_class_description_is_not_doubled_by_a_round_trip():
    diagram = _class_diagram([_node("book", "class", {
        "name": "Book", "description": "A printed work", "attributes": [], "methods": [],
    })])
    for _ in range(2):
        exported = class_buml_to_json(process_class_diagram(diagram))
        diagram = {"title": "Library", "model": exported}
    model = process_class_diagram(diagram)
    assert model.get_class_by_name("Book").metadata.description == "A printed work"
