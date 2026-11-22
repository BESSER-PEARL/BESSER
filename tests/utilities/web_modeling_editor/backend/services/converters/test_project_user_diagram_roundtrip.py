"""A UserDiagram survives a project round trip: JSON -> B-UML -> code -> JSON.

A project export writes a UserDiagram as the user reference domain model
(``user_model``) plus an object section (``user_object_model``).
``project_to_json`` has to read those back as a ``UserDiagram`` built from the
elements ``json_to_buml``'s object processor reads (``UserModelName``,
``UserModelAttribute`` with ``attributeOperator``), and must not import the
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


def _box(x=0, y=0):
    return {"x": x, "y": y, "width": 200, "height": 100}


def _user_object(eid, name, class_name, criteria=()):
    """A UserModelName plus one UserModelAttribute per (name, value) criterion."""
    elements = {
        eid: {
            "id": eid, "name": name, "type": "UserModelName", "owner": None,
            "bounds": _box(), "attributes": [f"{eid}-{attr}" for attr, _ in criteria],
            "methods": [], "className": class_name,
        },
    }
    for attr, value in criteria:
        elements[f"{eid}-{attr}"] = {
            "id": f"{eid}-{attr}", "name": f"{attr} = {value}",
            "type": "UserModelAttribute", "owner": eid, "attributeOperator": "==",
            "bounds": _box(),
        }
    return elements


def _object_link(rid, source, target):
    return {
        "id": rid, "name": "", "type": "ObjectLink", "owner": None, "bounds": _box(),
        "path": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
        "source": {"element": source, "direction": "Right"},
        "target": {"element": target, "direction": "Left"},
    }


def _user_diagram(title="Alice"):
    elements = {}
    elements.update(_user_object("u-user", "user_1", "User"))
    elements.update(_user_object(
        "u-info", "personal_Information_1", "Personal_Information",
        [("firstName", "Alice"), ("age", "30"), ("gender", "Female")],
    ))
    elements.update(_user_object("u-comp", "competence_1", "Competence"))
    elements.update(_user_object("u-eng", "language_1", "Language", [("iso693_3", "eng")]))
    elements.update(_user_object("u-fra", "language_2", "Language", [("iso693_3", "fra")]))
    relationships = {
        "l-info": _object_link("l-info", "u-user", "u-info"),
        "l-comp": _object_link("l-comp", "u-user", "u-comp"),
        "l-eng": _object_link("l-eng", "u-comp", "u-eng"),
        "l-fra": _object_link("l-fra", "u-comp", "u-fra"),
    }
    return {
        "id": f"user-diagram-{title}",
        "title": title,
        "model": {
            "version": "3.0.0", "type": "UserDiagram", "size": {"width": 1400, "height": 740},
            "elements": elements, "relationships": relationships,
            "interactive": {"elements": {}, "relationships": {}}, "assessments": {},
        },
    }


def _class_diagram():
    """An ordinary class diagram that happens to have a class named ``User``."""
    def cls(eid, name, attr_ids):
        return {"id": eid, "name": name, "type": "Class", "owner": None, "bounds": _box(),
                "attributes": attr_ids, "methods": []}

    def attr(eid, owner, name):
        return {"id": eid, "name": name, "type": "ClassAttribute", "owner": owner,
                "visibility": "public", "attributeType": "str", "bounds": _box()}

    return {
        "id": "class-diagram",
        "title": "Library",
        "model": {
            "version": "3.0.0", "type": "ClassDiagram", "size": {"width": 1400, "height": 740},
            "elements": {
                "c-user": cls("c-user", "User", ["a-uname"]),
                "a-uname": attr("a-uname", "c-user", "username"),
                "c-book": cls("c-book", "Book", ["a-title"]),
                "a-title": attr("a-title", "c-book", "title"),
            },
            "relationships": {
                "r-borrows": {
                    "id": "r-borrows", "name": "borrows", "type": "ClassBidirectional",
                    "owner": None, "bounds": _box(),
                    "source": {"element": "c-user", "multiplicity": "0..*", "role": "borrower"},
                    "target": {"element": "c-book", "multiplicity": "0..*", "role": "loans"},
                },
            },
            "interactive": {"elements": {}, "relationships": {}}, "assessments": {},
        },
    }


def _object_diagram():
    return {
        "id": "object-diagram",
        "title": "Loans",
        "references": {"ClassDiagram": "class-diagram"},
        "model": {
            "version": "3.0.0", "type": "ObjectDiagram", "size": {"width": 1400, "height": 740},
            "elements": {
                "o-bob": {"id": "o-bob", "name": "bob", "type": "ObjectName", "owner": None,
                          "bounds": _box(), "attributes": ["o-bob-name"], "methods": [],
                          "className": "User"},
                "o-bob-name": {"id": "o-bob-name", "name": "username = bob",
                               "type": "ObjectAttribute", "owner": "o-bob", "bounds": _box()},
            },
            "relationships": {},
            "interactive": {"elements": {}, "relationships": {}}, "assessments": {},
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
    assert diagrams["ClassDiagram"][0]["model"]["elements"] == {}
    assert imported["currentDiagramIndices"]["UserDiagram"] == 0

    [user_diagram] = diagrams["UserDiagram"]
    assert user_diagram["title"] == "Alice"
    model = user_diagram["model"]
    assert model["type"] == "UserDiagram"
    element_types = {el["type"] for el in model["elements"].values()}
    assert element_types == {"UserModelName", "UserModelAttribute"}
    criteria = {
        el["name"] for el in model["elements"].values() if el["type"] == "UserModelAttribute"
    }
    assert criteria == {
        "firstName = Alice", "age = 30", "gender = Female", "iso693_3 = eng", "iso693_3 = fra",
    }
    assert all(
        el["attributeOperator"] == "==" for el in model["elements"].values()
        if el["type"] == "UserModelAttribute"
    )
    class_names = {
        el["name"]: el["className"] for el in model["elements"].values()
        if el["type"] == "UserModelName"
    }
    assert class_names["personal_Information_1"] == "Personal_Information"
    assert len(model["relationships"]) == 4

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
        el["name"] for el in class_diagram["model"]["elements"].values() if el["type"] == "Class"
    }
    assert class_names == {"User", "Book"}

    [object_diagram] = diagrams["ObjectDiagram"]
    assert object_diagram["model"]["type"] == "ObjectDiagram"
    assert {
        el["name"] for el in object_diagram["model"]["elements"].values()
        if el["type"] == "ObjectAttribute"
    } == {"username = bob"}

    [user_diagram] = diagrams["UserDiagram"]
    reimported = json_to_buml_project(ProjectInput(**imported))
    [before] = _user_object_models(original)
    [after] = _user_object_models(reimported)
    assert _signature(after) == _signature(before)
    assert user_diagram["title"] == "Alice"
