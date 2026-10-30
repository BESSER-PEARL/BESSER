"""A project exported as B-UML with its trace (``/export-project-as-buml``) and imported back
(``/get-project-json-model``)."""

import copy

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, Generalization, Method, Multiplicity,
    Property, StringType,
)
from besser.utilities.web_modeling_editor.backend.services.converters import class_buml_to_json

EXPORT_URL = "/besser_api/export-project-as-buml"
IMPORT_URL = "/besser_api/get-project-json-model"


T = "2026-10-29T09:30:00.000Z"
REL_LABEL = {"ClassInheritance": "Generalization"}


@pytest.fixture
def class_diagram():
    """Every kind of element a trace can be about: classes, an attribute, a method, an
    enumeration and its literal, a named association and a generalization."""
    library = Class("Library", {Property("name", StringType)}, methods={Method("open")})
    book, publication = Class("Book"), Class("Publication")
    genre = Enumeration("Genre", {EnumerationLiteral("NOVEL")})
    holds = BinaryAssociation("holds", {Property("library", library, multiplicity=Multiplicity(1, 1)),
                                        Property("books", book, multiplicity=Multiplicity(0, 9999))})
    model = DomainModel("Library", types={library, book, publication, genre}, associations={holds},
                        generalizations={Generalization(publication, book)})
    return {"id": "diagram-1", "title": "Library", "model": class_buml_to_json(model)}


def _record_editing(trace, diagram):
    """Add what the editor records while the diagram is drawn: one artifact per element (named and
    nested as the editor names them) created in one modeling session, and a deleted class."""
    model = diagram["model"]
    artifacts = {a.get("elementId"): a for a in trace["artifacts"] if a["type"] == "ModelArtifact"}
    root = artifacts["diagram-1"]
    actor = trace["actors"][0]["id"]
    changes = []

    def artifact(element_id, name, element_type, parent):
        if element_id not in artifacts:
            artifacts[element_id] = {"id": f"a-{element_id}", "type": "ModelArtifact", "name": name, "timestamp": T,
                                     "elementId": element_id, "elementType": element_type, "parentId": parent["id"]}
            trace["artifacts"].append(artifacts[element_id])
        changes.append({"id": f"c-{element_id}", "kind": "CREATED", "artifactId": artifacts[element_id]["id"],
                        "newValue": name, "timestamp": T})
        return artifacts[element_id]

    elements = model["elements"]
    for element in elements.values():
        if element.get("owner") is None and element["type"] in ("Class", "Enumeration"):
            artifact(element["id"], element["name"], element["type"], root)
    for element in elements.values():
        owner = elements.get(element.get("owner") or "")
        if owner is not None:
            kind = ("EnumerationLiteral" if owner["type"] == "Enumeration" else
                    "Method" if element["type"] == "ClassMethod" else "Property")
            artifact(element["id"], element["name"], kind, artifacts[owner["id"]])
    for rel in model["relationships"].values():
        ends = [elements[rel[side]["element"]]["name"] for side in ("source", "target")]
        if rel["type"] == "ClassInheritance":
            artifact(rel["id"], f"{ends[0]} → {ends[1]}", "Generalization", root)
        else:
            artifact(rel["id"], rel.get("name") or f"{ends[0]} – {ends[1]}", "BinaryAssociation", root)
    artifact("gone", "Ghost", "Class", root)  # drawn, then deleted: only the record is left
    changes.append({"id": "c-gone-deleted", "kind": "DELETED", "artifactId": "a-gone", "oldValue": "Ghost",
                    "timestamp": "2026-10-29T09:31:00.000Z"})
    trace["operations"].insert(0, {
        "id": "session", "type": "Modeling", "status": "SUCCEEDED", "startedAt": T, "performedBy": [actor],
        "uses": [], "consumes": [], "produces": [], "logs": [], "changes": changes,
    })
    return trace


@pytest.fixture
def trace(client, class_diagram):
    """A trace as the editor holds it: the diagram drawn element by element, then a traced
    SQLAlchemy generation from it."""
    response = client.post("/besser_api/traceability/generate-output", json={
        **class_diagram, "generator": "sqlalchemy", "config": {"dbms": "sqlite"}, "actorName": "Ada Lovelace",
    })
    assert response.status_code == 200, response.text
    return _record_editing({**response.json()["trace"], "projectId": "project-1"}, class_diagram)


@pytest.fixture
def project(class_diagram):
    return {
        "id": "project-1", "type": "Project", "name": "Library", "owner": "Ada Lovelace",
        "createdAt": "2026-10-29T09:00:00.000Z", "currentDiagramType": "ClassDiagram",
        "currentDiagramIndices": {"ClassDiagram": 0}, "diagrams": {"ClassDiagram": [class_diagram]},
    }


def _export(client, project) -> str:
    response = client.post(EXPORT_URL, json=project)
    assert response.status_code == 200, response.text
    return response.text


def _class_names(body) -> set:
    (diagram,) = body["project"]["diagrams"]["ClassDiagram"]
    return {e["name"] for e in diagram["model"]["elements"].values() if e["type"] == "Class"}


def _without_element_ids(trace):
    """The artifacts without what the import may change: the element ids, and the name of a
    referenced method (its B-UML name ``open`` instead of the editor's label ``+ open()``)."""
    return [{k: v for k, v in a.items() if k != "elementId" and not (k == "name" and a.get("elementType") == "Method")}
            for a in trace["artifacts"]]


def _import(client, source: str):
    return client.post(IMPORT_URL, files={"buml_file": ("library.py", source.encode("utf-8"), "text/x-python")})


def test_trace_survives_export_and_import(client, project, trace):
    source = _export(client, {**project, "trace": trace})
    assert "# TRACEABILITY MODEL #" in source
    assert "trace_model" not in source.split("models=[")[1].split("]")[0]

    response = _import(client, source)
    assert response.status_code == 200, response.text
    body = response.json()
    assert _class_names(body) == {"Library", "Book", "Publication"}
    # The same trace, except that its elements now carry the ids the import gave them.
    assert _without_element_ids(body["trace"]) == _without_element_ids(trace)
    assert body["trace"]["operations"] == trace["operations"]
    assert body["trace"]["traceLinks"] == trace["traceLinks"]


def test_every_traced_element_is_found_in_the_imported_project(client, project, trace):
    """Without ids in the structural section the import gives elements new ids; the trace follows."""
    body = _import(client, _export(client, {**project, "trace": trace})).json()
    (diagram,) = body["project"]["diagrams"]["ClassDiagram"]
    imported = {**diagram["model"]["elements"], **diagram["model"]["relationships"]}
    for artifact in body["trace"]["artifacts"]:
        if artifact["type"] != "ModelArtifact" or artifact["name"] == "Ghost":
            continue
        if artifact["elementType"] == "DomainModel":
            assert artifact["elementId"] == diagram["id"]
            continue
        assert artifact["elementId"] in imported, artifact
        label = imported[artifact["elementId"]].get("name") or ""
        assert artifact["name"] in label or label in artifact["name"], artifact  # "open" in "+ open()"
    ghost = next(a for a in body["trace"]["artifacts"] if a["name"] == "Ghost")
    assert ghost["elementId"] == "gone"  # deleted: nothing to match, the record keeps its id


def test_the_trace_references_the_structural_elements(client, project, trace):
    source = _export(client, {**project, "trace": trace})
    section = source.split("# TRACEABILITY MODEL #")[1]
    for reference in ("element=domain_model,", "element=Library,", "element=Library_name,", "element=Library_m_open,",
                      "element=Genre,"):
        assert reference in section
    assert 'ModelArtifact("Ghost", element_id="gone", element_type="Class"' in section
    assert 'element_type="BinaryAssociation"' in section  # no public variable: kept as a record

    namespace = {}
    exec(compile(source, "library.py", "exec"), namespace)  # noqa: S102 -- our own export, run as a script
    library = next(a for a in namespace["trace_model"].artifacts
                   if getattr(a, "element_type", None) == "Class" and a.name == "Library")
    assert library.element is namespace["Library"]


def test_trace_section_leaves_the_models_alone(client, project, trace):
    without = _export(client, project)
    with_trace = _export(client, {**project, "trace": trace})
    models, definition = without.split("# PROJECT DEFINITION #")
    assert with_trace.startswith(models.rstrip("#\n"))
    assert with_trace.endswith(definition)


def test_project_without_trace_has_none(client, project):
    source = _export(client, project)
    assert "TRACEABILITY" not in source
    assert _import(client, source).json()["trace"] is None


def test_invalid_trace_is_rejected_on_export(client, project, trace):
    broken = copy.deepcopy(trace)
    broken["traceLinks"][0]["sourceId"] = "missing"
    response = client.post(EXPORT_URL, json={**project, "trace": broken})
    assert response.status_code == 400
    assert "unknown artifact" in response.json()["detail"]


def test_unsafe_trace_section_is_rejected_on_import(client, project, trace):
    source = _export(client, {**project, "trace": trace})
    tampered = source.replace("trace_model = TraceModel(", 'open("x", "w")\ntrace_model = TraceModel(', 1)
    response = _import(client, tampered)
    assert response.status_code == 400
    assert "traceability section" in response.json()["detail"]
