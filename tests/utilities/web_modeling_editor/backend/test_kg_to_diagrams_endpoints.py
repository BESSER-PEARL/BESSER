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
    assert "ClassBidirectional" in rel_types  # worksFor association
    assert "ClassInheritance" in rel_types    # Employee → Person


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
