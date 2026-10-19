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


def test_method_contracts_are_not_duplicated_by_a_buml_file_round_trip(tmp_path):
    from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content,
    )

    renew = {"id": "renew", "name": "renew", "visibility": "public", "returnType": "any",
             "parameters": [{"id": "p", "name": "days", "parameterType": "int"}]}
    diagram = _class_diagram([
        _node("book", "class", {"name": "Book", "attributes": [], "methods": [renew]}),
        _node("pre", "ClassOCLConstraint",
              {"expression": "context Book::renew(days: int) pre: days > 0"}),
        _node("inv", "ClassOCLConstraint",
              {"expression": "context Book inv named: true"}),
    ])
    model = process_class_diagram(diagram)
    [method] = model.get_class_by_name("Book").methods
    assert len(method.pre) == 1 and len(model.constraints) == 1

    path = tmp_path / "model.py"
    domain_model_to_code(model, str(path))
    reloaded = parse_buml_content(path.read_text(encoding="utf-8"))
    assert {c.name for c in reloaded.constraints} == {"named"}
    exported = class_buml_to_json(reloaded)
    boxes = [n for n in exported["nodes"] if n["type"] == "ClassOCLConstraint"]
    assert len(boxes) == 2
