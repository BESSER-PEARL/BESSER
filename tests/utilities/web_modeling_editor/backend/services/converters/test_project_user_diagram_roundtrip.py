"""A UserDiagram survives a project round trip: JSON -> B-UML -> code -> JSON.

A project export writes a UserDiagram as the user reference domain model
(``user_model``) plus an object section (``user_object_model``).
``project_to_json`` has to read those back as a ``UserDiagram`` built from the
nodes ``json_to_buml``'s object processor reads (``UserModelName`` nodes whose
attribute rows carry ``attributeOperator``), and must not import the
reference model as a class diagram. A class diagram that merely has a class
named ``User`` stays a class diagram.
"""

import copy

import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
    json_to_buml_project,
)


def _node(node_id, node_type, data):
    return {"id": node_id, "type": node_type, "position": {"x": 0, "y": 0},
            "width": 200, "height": 100, "data": data}


def _user_object(node_id, name, class_name, criteria=()):
    """A UserModelName node with one equality-criterion row per (name, value)."""
    rows = [
        {"id": f"{node_id}-{attr}", "name": attr, "value": value, "attributeOperator": "=="}
        for attr, value in criteria
    ]
    return _node(node_id, "UserModelName",
                 {"name": name, "className": class_name, "attributes": rows, "methods": []})


def _object_link(edge_id, source, target, edge_type="UserModelLink"):
    return {"id": edge_id, "type": edge_type, "source": source, "target": target,
            "data": {"name": "", "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}]}}


def _user_diagram(title="Alice"):
    nodes = [
        _user_object("u-user", "user_1", "User"),
        _user_object("u-info", "personal_Information_1", "Personal_Information",
                     [("firstName", "Alice"), ("age", "30"), ("gender", "Female")]),
        _user_object("u-comp", "competence_1", "Competence"),
        _user_object("u-eng", "language_1", "Language", [("iso693_3", "eng")]),
        _user_object("u-fra", "language_2", "Language", [("iso693_3", "fra")]),
    ]
    edges = [
        _object_link("l-info", "u-user", "u-info"),
        _object_link("l-comp", "u-user", "u-comp"),
        _object_link("l-eng", "u-comp", "u-eng"),
        _object_link("l-fra", "u-comp", "u-fra"),
    ]
    return {
        "id": f"user-diagram-{title}",
        "title": title,
        "model": {"version": "4.0.0", "type": "UserDiagram", "nodes": nodes, "edges": edges,
                  "assessments": {}},
    }


def _class_diagram():
    """An ordinary class diagram that happens to have a class named ``User``."""
    def cls(node_id, name, attr_id, attr_name):
        return _node(node_id, "class", {"name": name, "methods": [], "attributes": [
            {"id": attr_id, "name": attr_name, "visibility": "public", "attributeType": "str"},
        ]})

    return {
        "id": "class-diagram",
        "title": "Library",
        "model": {
            "version": "4.0.0", "type": "ClassDiagram", "assessments": {},
            "nodes": [cls("c-user", "User", "a-uname", "username"),
                      cls("c-book", "Book", "a-title", "title")],
            "edges": [{
                "id": "r-borrows", "type": "ClassBidirectional", "source": "c-user", "target": "c-book",
                "data": {"name": "borrows", "sourceMultiplicity": "0..*", "sourceRole": "borrower",
                         "targetMultiplicity": "0..*", "targetRole": "loans", "points": []},
            }],
        },
    }


def _object_diagram():
    return {
        "id": "object-diagram",
        "title": "Loans",
        "references": {"ClassDiagram": "class-diagram"},
        "model": {
            "version": "4.0.0", "type": "ObjectDiagram", "assessments": {}, "edges": [],
            "nodes": [_node("o-bob", "objectName", {
                "name": "bob", "className": "User", "methods": [],
                "attributes": [{"id": "o-bob-name", "name": "username", "value": "bob"}],
            })],
        },
    }


def _project(diagrams):
    return {
        "id": "proj-user", "type": "Project", "schemaVersion": 3, "name": "UserProject",
        "description": "UserDiagram round trip", "owner": "tester",
        "createdAt": "2026-10-13T00:00:00Z", "currentDiagramType": "UserDiagram",
        "currentDiagramIndices": {k: 0 for k in diagrams},
        "diagrams": diagrams,
    }


def _round_trip(project_json, tmp_path):
    """JSON -> B-UML Project -> project code -> JSON (as the editor imports it)."""
    buml_project = json_to_buml_project(ProjectInput(**copy.deepcopy(project_json)))
    out = tmp_path / "project.py"
    project_to_code(buml_project, str(out))
    return buml_project, out.read_text(encoding="utf-8"), project_to_json(out.read_text(encoding="utf-8"))


def _signature(object_model: ObjectModel):
    """Objects (name, class, slots) and links (as unordered object-name pairs)."""
    objects = {
        (obj.name, obj.classifier.name,
         tuple(sorted((slot.attribute.name, str(getattr(slot.value.value, "name", slot.value.value)))
                      for slot in obj.slots)))
        for obj in object_model.objects
    }
    links = {
        frozenset(end.object.name for end in link.connections)
        for obj in object_model.objects for link in obj.links
    }
    return objects, links


def _user_object_models(project):
    return [m for m in project.models if isinstance(m, ObjectModel)
            and any(o.classifier.name == "Personal_Information" for o in m.objects)]


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_user_diagram_round_trips(tmp_path):
    project_json = _project({"UserDiagram": [_user_diagram()]})

    original, source, imported = _round_trip(project_json, tmp_path)

    assert "user_object_model" in source
    diagrams = imported["diagrams"]
    # The reference model is context, not a class diagram.
    assert diagrams["ClassDiagram"][0]["model"]["nodes"] == []
    assert imported["currentDiagramIndices"]["UserDiagram"] == 0

    [user_diagram] = diagrams["UserDiagram"]
    assert user_diagram["title"] == "Alice"
    model = user_diagram["model"]
    assert model["type"] == "UserDiagram"
    assert "referenceDiagramData" not in model
    assert {node["type"] for node in model["nodes"]} == {"UserModelName"}
    rows = [row for node in model["nodes"] for row in node["data"]["attributes"]]
    assert {(row["name"], row["value"]) for row in rows} == {
        ("firstName", "Alice"), ("age", "30"), ("gender", "Female"),
        ("iso693_3", "eng"), ("iso693_3", "fra"),
    }
    assert all(row["attributeOperator"] == "==" for row in rows)
    class_names = {node["data"]["name"]: node["data"]["className"] for node in model["nodes"]}
    assert class_names["personal_Information_1"] == "Personal_Information"
    assert {edge["type"] for edge in model["edges"]} == {"UserModelLink"}
    assert len(model["edges"]) == 4

    # JSON -> B-UML again yields the same user model.
    reimported = json_to_buml_project(ProjectInput(**imported))
    [before] = _user_object_models(original)
    [after] = _user_object_models(reimported)
    objects, links = _signature(before)
    assert len(objects) == 5 and len(links) == 4
    assert _signature(after) == _signature(before)


def test_several_user_diagrams_round_trip(tmp_path):
    project_json = _project({"UserDiagram": [_user_diagram("Alice"), _user_diagram("Bob")]})

    _, source, imported = _round_trip(project_json, tmp_path)

    assert "user_object_model_1" in source and "user_object_model_2" in source
    titles = [diagram["title"] for diagram in imported["diagrams"]["UserDiagram"]]
    assert titles == ["Alice", "Bob"]


def test_user_diagram_beside_class_diagram_with_user_class(tmp_path):
    project_json = _project({
        "ClassDiagram": [_class_diagram()],
        "ObjectDiagram": [_object_diagram()],
        "UserDiagram": [_user_diagram()],
    })

    original, source, imported = _round_trip(project_json, tmp_path)
    diagrams = imported["diagrams"]

    # The class diagram with a "User" class is exported and imported as an
    # ordinary class diagram, not as the user reference model.
    assert "domain_model" in source
    [class_diagram] = diagrams["ClassDiagram"]
    class_names = {
        node["data"]["name"] for node in class_diagram["model"]["nodes"] if node["type"] == "class"
    }
    assert class_names == {"User", "Book"}

    [object_diagram] = diagrams["ObjectDiagram"]
    assert object_diagram["model"]["type"] == "ObjectDiagram"
    assert {
        (row["name"], row["value"])
        for node in object_diagram["model"]["nodes"] if node["type"] == "objectName"
        for row in node["data"]["attributes"]
    } == {("username", "bob")}

    [user_diagram] = diagrams["UserDiagram"]
    reimported = json_to_buml_project(ProjectInput(**imported))
    [before] = _user_object_models(original)
    [after] = _user_object_models(reimported)
    assert _signature(after) == _signature(before)
    assert user_diagram["title"] == "Alice"
