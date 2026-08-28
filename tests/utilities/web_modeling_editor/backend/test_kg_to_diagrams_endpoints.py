"""End-to-end tests for the KG → Class/Object diagram endpoints.

Posts a hand-built KnowledgeGraphDiagram payload to ``/besser_api/kg-to-class-diagram``
and ``/besser_api/kg-to-object-diagram`` and asserts the response shape the
frontend depends on.
"""

import pytest
from fastapi.testclient import TestClient

from besser.utilities.web_modeling_editor.backend.backend import app


RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
EX = "http://example.org/"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def kg_payload() -> dict:
    return {
        "title": "EndpointTest",
        "model": {
            "type": "KnowledgeGraphDiagram",
            "version": "1.0.0",
            "nodes": [
                {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
                {"id": "Employee", "nodeType": "class", "label": "Employee", "iri": EX + "Employee"},
                {"id": "Organization", "nodeType": "class", "label": "Organization", "iri": EX + "Organization"},
                {"id": "xsd_str", "nodeType": "class", "label": "string", "iri": XSD + "string"},
                {"id": "xsd_int", "nodeType": "class", "label": "integer", "iri": XSD + "integer"},
                {"id": "name", "nodeType": "property", "label": "name", "iri": EX + "name"},
                {"id": "age", "nodeType": "property", "label": "age", "iri": EX + "age"},
                {"id": "worksFor", "nodeType": "property", "label": "worksFor", "iri": EX + "worksFor"},
                {"id": "alice", "nodeType": "individual", "label": "alice", "iri": EX + "alice"},
                {"id": "acme", "nodeType": "individual", "label": "acme", "iri": EX + "acme"},
                {"id": "lit1", "nodeType": "literal", "value": "Alice"},
                {"id": "lit2", "nodeType": "literal", "value": "30", "datatype": XSD + "integer"},
                {"id": "blank1", "nodeType": "blank", "label": "_:b1"},
            ],
            "edges": [
                {"id": "e1", "source": "name", "target": "Person", "iri": RDFS + "domain"},
                {"id": "e2", "source": "name", "target": "xsd_str", "iri": RDFS + "range"},
                {"id": "e3", "source": "age", "target": "Person", "iri": RDFS + "domain"},
                {"id": "e4", "source": "age", "target": "xsd_int", "iri": RDFS + "range"},
                {"id": "e5", "source": "worksFor", "target": "Person", "iri": RDFS + "domain"},
                {"id": "e6", "source": "worksFor", "target": "Organization", "iri": RDFS + "range"},
                {"id": "e7", "source": "Employee", "target": "Person", "iri": RDFS + "subClassOf"},
                {"id": "t1", "source": "alice", "target": "Person", "iri": RDF + "type"},
                {"id": "t2", "source": "acme", "target": "Organization", "iri": RDF + "type"},
                {"id": "n1", "source": "alice", "target": "lit1", "iri": EX + "name"},
                {"id": "a1", "source": "alice", "target": "lit2", "iri": EX + "age"},
                {"id": "w1", "source": "alice", "target": "acme", "iri": EX + "worksFor"},
            ],
        },
    }


def test_class_endpoint_returns_class_diagram(client: TestClient, kg_payload: dict):
    response = client.post("/besser_api/kg-to-class-diagram", json=kg_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diagramType"] == "ClassDiagram"
    assert body["model"]["type"] == "ClassDiagram"
    elements = body["model"]["elements"]
    relationships = body["model"]["relationships"]
    types = {e["type"] for e in elements.values()}
    assert "Class" in types
    assert "ClassAttribute" in types
    rel_types = {r["type"] for r in relationships.values()}
    # A plain owl:ObjectProperty is navigable domain → range only; the reverse
    # end becomes navigable when (and only when) owl:inverseOf names it.
    assert "ClassUnidirectional" in rel_types  # worksFor association
    assert "ClassInheritance" in rel_types     # Employee → Person


def test_object_endpoint_returns_object_diagram_with_reference(client: TestClient, kg_payload: dict):
    response = client.post("/besser_api/kg-to-object-diagram", json=kg_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diagramType"] == "ObjectDiagram"
    model = body["model"]
    assert model["type"] == "ObjectDiagram"
    assert "referenceDiagramData" in model
    assert model["referenceDiagramData"].get("type") == "ClassDiagram"
    elements = model["elements"]
    rel_types = {r["type"] for r in model["relationships"].values()}
    obj_names = {e["name"] for e in elements.values() if e.get("type") == "ObjectName"}
    assert {"alice", "acme"} <= obj_names
    assert "ObjectLink" in rel_types
    # Each ObjectName element points to its class via classId.
    for elem in elements.values():
        if elem.get("type") == "ObjectName":
            assert elem.get("classId")


def test_endpoints_reject_non_kg_payload(client: TestClient):
    response = client.post(
        "/besser_api/kg-to-class-diagram",
        json={"title": "wrong", "model": {"type": "ClassDiagram"}},
    )
    assert response.status_code == 400
    assert "KnowledgeGraphDiagram" in response.json()["detail"]


def test_blank_skipped_warning_propagates(client: TestClient, kg_payload: dict):
    response = client.post("/besser_api/kg-to-object-diagram", json=kg_payload)
    body = response.json()
    codes = {w["code"] for w in (body.get("warnings") or [])}
    assert "BLANK_SKIPPED" in codes


def test_class_endpoint_emits_ocl_constraint_for_constraint_bearing_kg(client: TestClient):
    """A KG carrying a properly-linked NodeConstraint/PropertyConstraint pair
    must come back with a ClassOCLConstraint element — regression test for
    the OCL-generation pipeline, which previously had no HTTP-level coverage
    at all (every existing test called kg_to_class_diagram in-process)."""
    payload = {
        "title": "OCLEndpointTest",
        "model": {
            "type": "KnowledgeGraphDiagram",
            "version": "1.0.0",
            "nodes": [
                {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
                {"id": "name", "nodeType": "property", "label": "name", "iri": EX + "name"},
                {"id": "xsd_str", "nodeType": "class", "label": "string", "iri": XSD + "string"},
                {
                    "id": "PersonShape", "nodeType": "nodeConstraint", "label": "PersonShape",
                    "iri": EX + "PersonShape",
                    "metadata": {"constraintSpecs": [], "source": "shacl"},
                },
                {
                    "id": "pc1", "nodeType": "propertyConstraint", "label": "name_minLength",
                    "metadata": {"constraintSpecs": [{"kind": "minLength", "value": 2}], "source": "shacl"},
                },
            ],
            "edges": [
                {"id": "e1", "source": "name", "target": "Person", "iri": RDFS + "domain"},
                {"id": "e2", "source": "name", "target": "xsd_str", "iri": RDFS + "range"},
                {
                    "id": "e3", "source": "pc1", "target": "name",
                    "iri": "http://besser.local/kg#constraintTargetProperty",
                },
                {
                    "id": "e4", "source": "PersonShape", "target": "Person",
                    "iri": "http://besser.local/kg#constraintTargetClass",
                },
                {
                    "id": "e5", "source": "PersonShape", "target": "pc1",
                    "iri": "http://www.w3.org/ns/shacl#property",
                },
            ],
        },
    }
    response = client.post("/besser_api/kg-to-class-diagram", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    elements = body["model"]["elements"]
    types = {e["type"] for e in elements.values()}
    assert "ClassOCLConstraint" in types
    ocl_texts = [e["constraint"] for e in elements.values() if e["type"] == "ClassOCLConstraint"]
    assert any("self.name" in c for c in ocl_texts)


# --------------------------------------------------------------------------
# Individual-scoped object diagrams
# --------------------------------------------------------------------------


def _object_names(body: dict) -> set:
    """Object diagram elements carry the object name; associations do not."""
    elements = body["model"].get("elements", {})
    return {
        el.get("name", "").split(":")[0].strip()
        for el in elements.values()
        if el.get("type") == "ObjectName"
    }


def _with_unrelated_individual(kg_payload: dict) -> dict:
    """Add a Person who is connected to nobody, so scoping has something to drop."""
    model = {
        **kg_payload["model"],
        "nodes": kg_payload["model"]["nodes"] + [
            {"id": "zoe", "nodeType": "individual", "label": "zoe", "iri": EX + "zoe"},
        ],
        "edges": kg_payload["model"]["edges"] + [
            {"id": "tz", "source": "zoe", "target": "Person", "iri": RDF + "type"},
        ],
    }
    return {**kg_payload, "model": model}


def test_object_diagram_scoped_to_one_individual(client: TestClient, kg_payload: dict):
    payload = _with_unrelated_individual(kg_payload)
    response = client.post(
        "/besser_api/kg-to-object-diagram",
        json={**payload, "rootIndividualIds": ["acme"], "maxDepth": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # acme and alice are linked by :worksFor, so both survive at depth 1;
    # zoe is connected to nobody and must be dropped.
    names = _object_names(body)
    assert names == {"acme", "alice"}

    codes = {w["code"] for w in body.get("warnings") or []}
    assert "ABOX_SCOPED" in codes


def test_unscoped_request_keeps_every_individual(client: TestClient, kg_payload: dict):
    payload = _with_unrelated_individual(kg_payload)
    response = client.post("/besser_api/kg-to-object-diagram", json=payload)
    assert response.status_code == 200
    assert _object_names(response.json()) == {"acme", "alice", "zoe"}


def test_scoping_does_not_shrink_the_reference_class_diagram(
    client: TestClient, kg_payload: dict
):
    """The reference class diagram must not move when the ABox is scoped.

    It is what the object diagram's slots and links resolve against, so a
    scoped diagram that also lost classes would mistype its own objects.
    """
    full = client.post("/besser_api/kg-to-object-diagram", json=kg_payload)
    scoped = client.post(
        "/besser_api/kg-to-object-diagram",
        json={**kg_payload, "rootIndividualIds": ["acme"], "maxDepth": 1},
    )
    assert full.status_code == 200 and scoped.status_code == 200

    def classes(body):
        ref = body["model"].get("referenceDiagramData") or {}
        return {
            el.get("name")
            for el in (ref.get("elements") or {}).values()
            if el.get("type") == "Class"
        }

    full_classes = classes(full.json())
    assert full_classes, "reference class diagram was empty; the assertion below would be vacuous"
    assert classes(scoped.json()) == full_classes


def test_unscoped_request_is_unchanged(client: TestClient, kg_payload: dict):
    response = client.post("/besser_api/kg-to-object-diagram", json=kg_payload)
    assert response.status_code == 200
    names = _object_names(response.json())
    assert {"alice", "acme"} <= names
    codes = {w["code"] for w in response.json().get("warnings") or []}
    assert "ABOX_SCOPED" not in codes


def test_unknown_root_individual_returns_400(client: TestClient, kg_payload: dict):
    response = client.post(
        "/besser_api/kg-to-object-diagram",
        json={**kg_payload, "rootIndividualIds": ["nobody"]},
    )
    assert response.status_code == 400
    assert "nobody" in response.json()["detail"]


def test_root_that_is_not_an_individual_returns_400(client: TestClient, kg_payload: dict):
    response = client.post(
        "/besser_api/kg-to-object-diagram",
        json={**kg_payload, "rootIndividualIds": ["Person"]},
    )
    assert response.status_code == 400
    assert "not an individual" in response.json()["detail"]


def test_empty_root_list_returns_400(client: TestClient, kg_payload: dict):
    response = client.post(
        "/besser_api/kg-to-object-diagram",
        json={**kg_payload, "rootIndividualIds": []},
    )
    assert response.status_code == 400


def test_non_positive_max_depth_returns_400(client: TestClient, kg_payload: dict):
    response = client.post(
        "/besser_api/kg-to-object-diagram",
        json={**kg_payload, "rootIndividualIds": ["acme"], "maxDepth": 0},
    )
    assert response.status_code == 400
