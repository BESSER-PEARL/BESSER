"""References from other diagrams to class-diagram elements keep resolving (issue #659).

Object diagrams (``classId``/``attributeId``/``associationId``), GUI diagrams
(``data-source``, ``data-method``, a table's ``lookupEntity``) and component
diagrams (``realizes``) point at class-diagram element ids. Measured on a
project export -> project ``.py`` (ids written) -> import:
class ids came back, but an object link on a composition lost its
association, a table's lookup columns came back keyed by class name, and a
lookup column whose ``field`` holds the target class name (the editor's label
for an end without a role) was dropped on conversion.
"""
import copy
import os

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import BarChart, LookupColumn, Table
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Metadata, Multiplicity, Property, StringType,
)
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
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
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


def _components(gui):
    found = {}

    def walk(component):
        attributes = component.get("attributes") or {}
        if attributes.get("id"):
            found[attributes["id"]] = attributes
        for child in component.get("components") or []:
            walk(child)

    for page in gui["pages"]:
        walk(page["frames"][0]["component"])
    return found


def _object_refs(object_model):
    elements = object_model["elements"]
    names = {key: e["name"] for key, e in elements.items()}
    objects = {e["name"]: e["classId"] for e in elements.values() if e["type"] == "ObjectName"}
    slots = {e["name"].split("=")[0].strip(): e["attributeId"]
             for e in elements.values() if e["type"] == "ObjectAttribute"}
    links = {frozenset((names[r["source"]["element"]], names[r["target"]["element"]])): r["associationId"]
             for r in object_model["relationships"].values() if r["type"] == "ObjectLink"}
    return objects, slots, links


def test_project_python_round_trip_keeps_ids_and_every_reference(tmp_path):
    back = _round_trip(_project(), tmp_path, emit_ids=True)

    class_model = _model(back, "ClassDiagram")
    assert set(_class_model()["elements"]) <= set(class_model["elements"])
    assert {"r-holds", "r-contains"} <= set(class_model["relationships"])

    assert _object_refs(_model(back, "ObjectDiagram")) == _object_refs(_object_model(_class_model()))

    components = _components(_model(back, "GUINoCodeDiagram"))
    assert components["t-books"]["data-source"] == "c-book"
    lookup = [c for c in components["t-books"]["columns"] if c["columnType"] == "lookup"]
    assert [(c["field"], c["lookupEntity"]) for c in lookup] == [("library", "c-library")]
    assert components["b-renew"]["data-method"] == "m-renew"
    assert components["b-renew"]["data-method-class"] == "c-book"

    assert _model(back, "ComponentDiagram")["elements"]["k-api"]["realizes"] == ["c-book"]
    reimported = json_to_buml_project(ProjectInput.model_validate(back))
    assert reimported._cross_diagram_errors["errors"] == []


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


def test_a_lookup_column_named_after_its_target_class_resolves_by_lookup_entity():
    class_model = _class_model()
    domain = process_class_diagram({"title": "Library", "model": class_model})

    gui = process_gui_diagram(_gui_model(), class_model, domain)

    table = next(e for m in gui.modules for s in m.screens for e in s.view_elements if isinstance(e, Table))
    lookups = [c for c in table.columns if isinstance(c, LookupColumn)]
    assert [(c.path.name, c.path.type.name, c.field.name) for c in lookups] == [("library", "Library", "name")]


def test_gui_export_names_domain_elements_by_id_when_they_have_one(tmp_path):
    name = Property(name="name", type=StringType, id="a-name")
    pages = Property(name="pages", type=IntegerType, id="a-pages")
    author = Class(name="Author", attributes={name}, id="c-author")
    book = Class(name="Book", attributes={pages}, id="c-book")
    writes = BinaryAssociation(name="writes", ends={
        Property(name="author", type=author, multiplicity=Multiplicity(1, 1)),
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
    })
    domain = DomainModel(name="Library", types={author, book}, associations={writes})
    author_end = next(end for end in writes.ends if end.name == "author")
    table = Table(name="books_table", label="Books", component_id="t-books", display_order=0,
                  columns=[LookupColumn(label="Author", path=author_end, field=name)],
                  data_binding=DataBinding(domain_concept=book))
    chart = BarChart(name="pages_chart", title="Pages", component_id="ch-pages", display_order=1,
                     data_binding=DataBinding(domain_concept=book, label_field=pages, data_field=pages))
    screen = Screen(name="Books", description="Books", view_elements={table, chart}, is_main_page=True)
    gui = GUIModel(name="ui", package="", versionCode="1", versionName="1", description="",
                   modules={Module(name="M", screens={screen})})
    path = os.path.join(tmp_path, "project.py")
    project_to_code(Project(name="p", models=[domain, gui], metadata=Metadata(description="d")), path, emit_ids=True)
    with open(path, encoding="utf-8") as handle:
        components = _components(_model(project_to_json(handle.read()), "GUINoCodeDiagram"))

    assert components["t-books"]["data-source"] == "c-book"
    assert [(c["field"], c["lookupEntity"]) for c in components["t-books"]["columns"]] == [("author", "c-author")]
    assert (components["ch-pages"]["data-source"], components["ch-pages"]["data-field"]) == ("c-book", "a-pages")


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
