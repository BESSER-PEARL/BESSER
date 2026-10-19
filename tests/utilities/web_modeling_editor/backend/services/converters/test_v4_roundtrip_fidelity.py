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


def _fixture_model(name):
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[5] / "fixtures" / "v4" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["model"]


def _project_round_trip(diagrams, tmp_path, description="Round trip"):
    import copy
    from besser.utilities.buml_code_builder.project_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
        json_to_buml_project,
    )
    project = {"id": "p", "type": "Project", "name": "RoundTrip", "description": description,
               "owner": "tester", "createdAt": "2026-01-01T00:00:00Z",
               "diagrams": copy.deepcopy(diagrams)}
    out = tmp_path / "project.py"
    project_to_code(json_to_buml_project(ProjectInput(**project)), str(out))
    code = out.read_text(encoding="utf-8")
    return code, project_to_json(code)


def test_several_diagrams_of_one_type_import_once_each(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent, sm, cd = (_fixture_model(n) for n in
                     ("agent_diagram_basic", "state_machine_basic", "class_diagram_basic"))
    diagrams = {
        "AgentDiagram": [{"id": "a1", "title": "AgentOne", "model": agent},
                         {"id": "a2", "title": "AgentTwo", "model": agent}],
        "StateMachineDiagram": [{"id": "s1", "title": "SmOne", "model": sm},
                                {"id": "s2", "title": "SmTwo", "model": sm}],
        "ClassDiagram": [{"id": "c1", "title": "LibA", "model": cd},
                         {"id": "c2", "title": "LibB", "model": cd}],
    }
    _, imported = _project_round_trip(diagrams, tmp_path)
    for diagram_type in diagrams:
        entries = imported["diagrams"][diagram_type]
        assert len(entries) == 2, (diagram_type, [e["title"] for e in entries])
        assert all(e["model"]["nodes"] for e in entries), diagram_type
