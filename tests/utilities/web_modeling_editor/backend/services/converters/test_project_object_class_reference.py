"""Object diagrams in a project resolve their objects' classes by ``classId``.

Projects migrated from the develop editor keep only ``classId`` on each
object (develop resolved it through an embedded ``referenceDiagramData``
snapshot, which the v4 migration drops), and object names such as
``author1`` do not match any class. ``json_to_buml_project`` must resolve
``classId`` against the project's class diagram instead.
"""

from besser.BUML.metamodel.object import ObjectModel
from besser.utilities.web_modeling_editor.backend.models import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
    json_to_buml_project,
)


def _class_node(node_id, name, attr_id, attr_name):
    return {
        "id": node_id,
        "type": "class",
        "position": {"x": 0, "y": 0},
        "data": {
            "name": name,
            "methods": [],
            "attributes": [
                {"id": attr_id, "name": attr_name, "visibility": "public", "attributeType": "str"},
            ],
        },
    }


CLASS_MODEL = {
    "version": "4.0.0",
    "type": "ClassDiagram",
    "nodes": [
        _class_node("cls-author", "Author", "attr-alias", "alias"),
        _class_node("cls-book", "Book", "attr-title", "title"),
    ],
    "edges": [
        {
            "id": "assoc-1",
            "source": "cls-book",
            "target": "cls-author",
            "type": "ClassBidirectional",
            "data": {
                "name": "authors",
                "sourceRole": "books",
                "sourceMultiplicity": "*",
                "targetRole": "writers",
                "targetMultiplicity": "1..*",
            },
        }
    ],
}


def _object_node(node_id, name, class_id, attr_name, value, class_name=None):
    data = {
        "name": name,
        "classId": class_id,
        "attributes": [{"id": f"{node_id}-a", "name": attr_name, "value": value}],
    }
    if class_name is not None:
        data["className"] = class_name
    return {"id": node_id, "type": "objectName", "position": {"x": 0, "y": 0}, "data": data}


def _project(object_nodes):
    object_model = {
        "version": "4.0.0",
        "type": "ObjectDiagram",
        "nodes": object_nodes,
        "edges": [
            {
                "id": "link-1",
                "source": "o-book",
                "target": "o-author",
                "type": "ObjectLink",
                "data": {"name": "authors", "associationId": "assoc-1"},
            }
        ],
    }
    return ProjectInput(**{
        "id": "p1",
        "type": "Project",
        "name": "Library",
        "createdAt": "2025-01-01T00:00:00Z",
        "currentDiagramType": "ObjectDiagram",
        "currentDiagramIndices": {"ClassDiagram": 0, "ObjectDiagram": 0},
        "diagrams": {
            "ClassDiagram": [{"id": "cd-1", "title": "Classes", "model": CLASS_MODEL}],
            "ObjectDiagram": [
                {
                    "id": "od-1",
                    "title": "Objects",
                    "model": object_model,
                    "references": {"ClassDiagram": "cd-1"},
                }
            ],
        },
    })


def _objects(project):
    object_models = [m for m in json_to_buml_project(project).models if isinstance(m, ObjectModel)]
    assert len(object_models) == 1
    return {o.name: o for o in object_models[0].objects}


def test_objects_with_only_class_id_resolve_their_class():
    objects = _objects(_project([
        _object_node("o-author", "author1", "cls-author", "alias", "Ann"),
        _object_node("o-book", "book1", "cls-book", "title", "Dune"),
    ]))

    assert objects["author1"].classifier.name == "Author"
    assert objects["book1"].classifier.name == "Book"
    assert {s.attribute.name: s.value.value for s in objects["author1"].slots} == {"alias": "Ann"}
    assert len(objects["book1"].links) == 1


def test_class_id_wins_over_stale_class_name():
    # The class was renamed after the object was drawn: className is stale.
    objects = _objects(_project([
        _object_node("o-author", "author1", "cls-author", "alias", "Ann", class_name="Writer"),
        _object_node("o-book", "book1", "cls-book", "title", "Dune", class_name="Novel"),
    ]))

    assert objects["author1"].classifier.name == "Author"
    assert objects["book1"].classifier.name == "Book"
