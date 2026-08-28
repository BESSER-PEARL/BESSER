"""Tests for /validate-diagram on KnowledgeGraphDiagram payloads.

The endpoint used to return an unconditional pass. It now runs the two checks
the editor's Refine KG modal runs — the static preflight and the OWL/SHACL
consistency check — and reports their findings without the modal's
recommended/skip actions, which are interactive choices with no meaning in a
validation response.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph
from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.converters import kg_to_json


CLEAN_TTL = """
@prefix : <http://ex.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Person a owl:Class ; rdfs:label "Person" .
:name   a owl:DatatypeProperty ; rdfs:label "name" ;
        rdfs:domain :Person ; rdfs:range xsd:string .
:alice  a :Person ; rdfs:label "Alice" ; :name "Alice" .
""".strip()

# `:age` has neither rdfs:domain nor rdfs:range but is asserted in the ABox —
# the PROPERTY_NO_DOMAIN / PROPERTY_NO_RANGE detectors both fire.
NO_DOMAIN_TTL = """
@prefix : <http://ex.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Person a owl:Class ; rdfs:label "Person" .
:age    a owl:DatatypeProperty ; rdfs:label "age" .
:alice  a :Person ; :age "30"^^xsd:integer .
""".strip()

# `:alice` is asserted into two disjoint classes, which owlrl + pyshacl
# surface as a consistency violation rather than a preflight finding.
DISJOINT_TTL = """
@prefix : <http://ex.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

:Person a owl:Class ; rdfs:label "Person" ; owl:disjointWith :Robot .
:Robot  a owl:Class ; rdfs:label "Robot" .
:alice  a :Person, :Robot ; rdfs:label "Alice" .
""".strip()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _payload(ttl: str, tmp_path: Path) -> dict:
    path = tmp_path / "graph.ttl"
    path.write_text(ttl, encoding="utf-8")
    kg = owl_file_to_knowledge_graph(str(path))
    envelope = kg_to_json(kg)
    model = {**envelope["model"], "type": "KnowledgeGraphDiagram"}
    return {"title": "Graph", "model": model}


def _validate(client: TestClient, ttl: str, tmp_path: Path) -> dict:
    response = client.post("/besser_api/validate-diagram", json=_payload(ttl, tmp_path))
    assert response.status_code == 200, response.text
    return response.json()


def test_clean_graph_is_valid(client, tmp_path):
    result = _validate(client, CLEAN_TTL, tmp_path)
    assert result["isValid"] is True
    assert result["errors"] == []


def test_preflight_findings_are_reported_as_errors(client, tmp_path):
    result = _validate(client, NO_DOMAIN_TTL, tmp_path)

    assert result["isValid"] is False
    codes = " ".join(result["errors"])
    assert "PROPERTY_NO_DOMAIN" in codes
    assert "PROPERTY_NO_RANGE" in codes


def test_findings_carry_the_affected_node_ids(client, tmp_path):
    result = _validate(client, NO_DOMAIN_TTL, tmp_path)
    domain_error = next(e for e in result["errors"] if "PROPERTY_NO_DOMAIN" in e)
    assert "(nodes: " in domain_error
    assert "http://ex.org/age" in domain_error


def test_no_resolution_actions_leak_into_the_response(client, tmp_path):
    # The modal offers "attach_to_thing" / "drop_property" alongside each
    # finding. A validation response must carry the finding only.
    result = _validate(client, NO_DOMAIN_TTL, tmp_path)
    blob = " ".join(result["errors"] + result["warnings"])
    for action_key in ("attach_to_thing", "drop_property", "set_range",
                       "recommendedAction", "skipAction"):
        assert action_key not in blob


def test_consistency_violations_are_reported(client, tmp_path):
    result = _validate(client, DISJOINT_TTL, tmp_path)

    assert result["isValid"] is False

    # The breach is reported by the consistency checker, not the preflight, and
    # reaches pyshacl through the OWL->SHACL shim for owl:disjointWith — so it
    # surfaces as a NotConstraintComponent naming the offending individual
    # rather than as the word "disjoint".
    errors = [e for e in result["errors"] if "NotConstraintComponent" in e]
    assert errors, result["errors"]
    assert all("http://ex.org/alice" in e for e in errors)
    assert any("must NOT be of class Robot" in e for e in errors)
    assert any("must NOT be of class Person" in e for e in errors)


def test_malformed_kg_payload_is_rejected(client):
    response = client.post(
        "/besser_api/validate-diagram",
        json={
            "title": "Broken",
            "model": {
                "type": "KnowledgeGraphDiagram",
                "nodes": [{"id": "n1", "nodeType": "class", "label": "A"}],
                "edges": [{"id": "e1", "source": "n1", "target": "missing"}],
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["isValid"] is False
    assert any("missing" in e for e in payload["errors"])
