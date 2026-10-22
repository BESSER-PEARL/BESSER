"""References from object diagrams to class-diagram elements keep resolving (issue #659).

An object's ``classId``, a slot's ``attributeId`` and a link's
``associationId`` point at class-diagram element ids. An object diagram whose
class was renamed failed to convert (the class was looked up by its old name),
and after a project export -> ``.py`` -> import an object link on a
composition came back without its association.
"""
import copy
import os

from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.object_diagram_converter import (
    object_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.object_diagram_processor import (
    process_object_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
    json_to_buml_project,
)

_BOUNDS = {"x": 0, "y": 0, "width": 200, "height": 100}


def _class_model():
    def cls(cid, name, attributes=(), methods=()):
        return {"id": cid, "name": name, "type": "Class", "owner": None, "bounds": _BOUNDS,
                "attributes": list(attributes), "methods": list(methods)}

    def attr(aid, owner, name):
        return {"id": aid, "name": name, "type": "ClassAttribute", "owner": owner, "bounds": _BOUNDS,
                "visibility": "public", "attributeType": "str"}

    def end(element, role, multiplicity):
        return {"element": element, "role": role, "multiplicity": multiplicity}

    return {
        "type": "ClassDiagram",
        "elements": {
            "c-library": cls("c-library", "Library", ["a-lib-name"]),
            "a-lib-name": attr("a-lib-name", "c-library", "name"),
            "c-book": cls("c-book", "Book", ["a-title"], ["m-renew"]),
            "a-title": attr("a-title", "c-book", "title"),
            "m-renew": {"id": "m-renew", "name": "+ renew(): str", "type": "ClassMethod",
                        "owner": "c-book", "bounds": _BOUNDS},
            "c-shelf": cls("c-shelf", "Shelf", ["a-code"]),
            "a-code": attr("a-code", "c-shelf", "code"),
        },
        "relationships": {
            "r-holds": {"id": "r-holds", "name": "holds", "type": "ClassBidirectional",
                        "source": end("c-library", "library", "1"), "target": end("c-book", "books", "0..*")},
            # Composition: the source is the part, the target the whole.
            "r-contains": {"id": "r-contains", "name": "contains", "type": "ClassComposition",
                           "source": end("c-shelf", "shelves", "0..*"), "target": end("c-library", "home", "1")},
        },
    }


def _object_model(class_model):
    def obj(oid, name, class_id, slots):
        return {"id": oid, "name": name, "type": "ObjectName", "owner": None, "bounds": _BOUNDS,
                "attributes": slots, "methods": [], "classId": class_id}

    def slot(sid, owner, text, attribute_id):
        return {"id": sid, "name": text, "type": "ObjectAttribute", "owner": owner, "bounds": _BOUNDS,
                "attributeId": attribute_id}

    def link(lid, name, source, target, association_id):
        return {"id": lid, "name": name, "type": "ObjectLink", "owner": None, "bounds": _BOUNDS,
                "path": [], "source": {"element": source}, "target": {"element": target},
                "associationId": association_id}

    return {
        "type": "ObjectDiagram",
        "elements": {
            "o-lib": obj("o-lib", "city", "c-library", ["s-name"]),
            "s-name": slot("s-name", "o-lib", "name = City", "a-lib-name"),
            "o-book": obj("o-book", "dune", "c-book", ["s-title"]),
            "s-title": slot("s-title", "o-book", "title = Dune", "a-title"),
            "o-shelf": obj("o-shelf", "top", "c-shelf", []),
        },
        "relationships": {
            "l-holds": link("l-holds", "books", "o-lib", "o-book", "r-holds"),
            "l-contains": link("l-contains", "home", "o-shelf", "o-lib", "r-contains"),
        },
        "referenceDiagramData": copy.deepcopy(class_model),
    }


def _gui_model():
    table = {"type": "table", "attributes": {
        "id": "t-books", "chart-title": "Books", "data-source": "c-book",
        "columns": [
            {"field": "title", "label": "Title", "columnType": "field"},
            # The editor labels an end by its role; this one is labelled by the class name.
            {"field": "Library", "label": "Library", "columnType": "lookup",
             "lookupEntity": "c-library", "lookupField": "name"},
        ]}}
    button = {"type": "action-button", "attributes": {
        "id": "b-renew", "data-action-type": "run-method", "data-button-label": "Renew",
        "data-method": "m-renew", "data-method-class": "c-book", "data-instance-source": "t-books"}}
    return {"pages": [{"id": "books", "name": "Books", "frames": [{"component": {
        "type": "wrapper", "components": [table, button]}}]}]}


def _project():
    class_model = _class_model()
    component_model = {"type": "ComponentDiagram", "relationships": {}, "elements": {
        "k-api": {"id": "k-api", "name": "Api", "type": "Component", "owner": None,
                  "bounds": _BOUNDS, "realizes": ["c-book"]}}}
    return {
        "id": "p", "type": "Project", "name": "Lib", "createdAt": "2026-01-01T00:00:00Z",
        "diagrams": {
            "ClassDiagram": [{"id": "d-class", "title": "Library", "model": class_model}],
            "ObjectDiagram": [{"id": "d-objects", "title": "Objects", "model": _object_model(class_model)}],
            "GUINoCodeDiagram": [{"id": "d-gui", "title": "GUI", "model": _gui_model()}],
            "ComponentDiagram": [{"id": "d-components", "title": "Components", "model": component_model}],
        },
    }


def _round_trip(project, tmp_path, emit_ids):
    path = os.path.join(tmp_path, "project.py")
    project_to_code(json_to_buml_project(ProjectInput.model_validate(project)), path, emit_ids=emit_ids)
    with open(path, encoding="utf-8") as handle:
        return project_to_json(handle.read())


def _model(project, diagram_type):
    return project["diagrams"][diagram_type][0]["model"]


def _object_refs(object_model):
    elements = object_model["elements"]
    names = {key: e["name"] for key, e in elements.items()}
    objects = {e["name"]: e["classId"] for e in elements.values() if e["type"] == "ObjectName"}
    slots = {e["name"].split("=")[0].strip(): e["attributeId"]
             for e in elements.values() if e["type"] == "ObjectAttribute"}
    links = {frozenset((names[r["source"]["element"]], names[r["target"]["element"]])): r["associationId"]
             for r in object_model["relationships"].values() if r["type"] == "ObjectLink"}
    return objects, slots, links


def test_object_links_on_compositions_keep_their_association(tmp_path):
    """Without ids in the .py, references still resolve to the re-imported class diagram."""
    back = _round_trip(_project(), tmp_path, emit_ids=False)

    class_model = _model(back, "ClassDiagram")
    objects, slots, links = _object_refs(_model(back, "ObjectDiagram"))
    names = {key: e.get("name") for key, e in {**class_model["elements"], **class_model["relationships"]}.items()}
    assert {name: names.get(ref) for name, ref in objects.items()} == {"city": "Library", "dune": "Book", "top": "Shelf"}
    assert {name: names.get(ref) for name, ref in slots.items()} == {"name": "name", "title": "title"}
    assert sorted(names.get(ref) for ref in links.values()) == ["contains", "holds"]


def test_objects_resolve_a_renamed_class_and_attribute_by_id():
    """The object keeps the old class name and the snapshot of the old class
    diagram; the element ids still find the renamed class and attribute."""
    renamed = _class_model()
    renamed["elements"]["c-book"]["name"] = "Novel"
    renamed["elements"]["a-title"]["name"] = "heading"
    domain = process_class_diagram({"title": "Library", "model": renamed})
    objects = _object_model(_class_model())
    objects["elements"]["o-book"]["className"] = "Book"

    object_model = process_object_diagram({"title": "Objects", "model": objects}, domain)

    dune = next(o for o in object_model.objects if o.name == "dune")
    assert dune.classifier.name == "Novel"
    assert [(slot.attribute.name, slot.value.value) for slot in dune.slots] == [("heading", "Dune")]


def test_object_model_code_links_a_composition_to_its_association(tmp_path):
    class_model = _class_model()
    domain = process_class_diagram({"title": "Library", "model": class_model})
    objects = process_object_diagram({"title": "Objects", "model": _object_model(class_model)}, domain)
    path = os.path.join(tmp_path, "objects.py")
    domain_model_to_code(domain, path, objectmodel=objects)
    with open(path, encoding="utf-8") as handle:
        exported = object_buml_to_json(handle.read(), class_model)

    associations = sorted(r["associationId"] for r in exported["relationships"].values() if r["type"] == "ObjectLink")
    assert associations == ["r-contains", "r-holds"]
