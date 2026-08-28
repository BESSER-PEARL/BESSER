"""End-to-end tests for the KG → Class diagram endpoint.

Posts a hand-built KnowledgeGraphDiagram payload to
``/besser_api/kg-to-class-diagram`` and asserts the response shape the
frontend depends on.
"""

import pytest


RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
EX = "http://example.org/"


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


def test_class_endpoint_returns_class_diagram(client, kg_payload: dict):
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


def test_endpoints_reject_non_kg_payload(client):
    response = client.post(
        "/besser_api/kg-to-class-diagram",
        json={"title": "wrong", "model": {"type": "ClassDiagram"}},
    )
    assert response.status_code == 400
    assert "KnowledgeGraphDiagram" in response.json()["detail"]


def test_class_endpoint_emits_ocl_constraint_for_constraint_bearing_kg(client):
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
