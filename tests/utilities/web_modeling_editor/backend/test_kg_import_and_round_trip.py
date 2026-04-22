"""End-to-end tests for the Knowledge Graph backend.

Covers:
  1. rdflib → KnowledgeGraph → JSON produces the expected shape.
  2. JSON → process_kg_diagram → kg_to_json is an identity (round-trip).
  3. POST /import-owl returns a DiagramExportResponse that /validate-diagram accepts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph
from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.converters import (
    kg_to_json,
    process_kg_diagram,
)


TTL_FIXTURE = """
@prefix : <http://ex.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Person a owl:Class ; rdfs:label "Person" .
:knows  a owl:ObjectProperty ; rdfs:label "knows" .
:age    a owl:DatatypeProperty .
:alice  a :Person ; :knows :bob ; :age "30"^^xsd:integer ; rdfs:label "Alice" .
:bob    a :Person .
_:b0    a :Person .
""".strip()


@pytest.fixture
def ttl_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.ttl"
    p.write_text(TTL_FIXTURE, encoding="utf-8")
    return p


def test_owl_to_kg_classification(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))

    # At minimum: 1 class (Person), 2 individuals (alice, bob), 2 properties
    # (knows, age), 1 literal with value "30", and 1 blank node.
    kinds = {}
    for n in kg.nodes:
        kinds.setdefault(type(n).__name__, []).append(n)
    assert "KGClass" in kinds and any(n.id.endswith("Person") for n in kinds["KGClass"])
    assert "KGIndividual" in kinds and any(n.id.endswith("alice") for n in kinds["KGIndividual"])
    assert "KGProperty" in kinds and any(n.id.endswith("knows") for n in kinds["KGProperty"])
    assert "KGBlank" in kinds
    assert "KGLiteral" in kinds
    assert any(n.value == "30" for n in kinds["KGLiteral"])

    # One edge per triple in the fixture.
    assert len(kg.edges) == 11


def test_kg_json_round_trip(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    j1 = kg_to_json(kg)
    kg2 = process_kg_diagram(j1)
    j2 = kg_to_json(kg2)
    assert j1 == j2


def test_import_owl_endpoint_and_validation(ttl_path: Path):
    client = TestClient(app)
    with ttl_path.open("rb") as fh:
        resp = client.post(
            "/besser_api/import-owl",
            files={"owl_file": ("tiny.ttl", fh.read(), "text/turtle")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["diagramType"] == "KnowledgeGraphDiagram"
    model = body["model"]
    assert model["type"] == "KnowledgeGraphDiagram"
    assert isinstance(model["nodes"], list) and len(model["nodes"]) >= 6
    assert isinstance(model["edges"], list) and len(model["edges"]) == 11

    # Validation endpoint must accept the KG payload (no-op branch).
    valid_resp = client.post(
        "/besser_api/validate-diagram",
        json={"title": body["title"], "model": model},
    )
    assert valid_resp.status_code == 200
    assert valid_resp.json()["isValid"] is True


def test_import_owl_rejects_unsupported_extension(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("not an ontology", encoding="utf-8")
    client = TestClient(app)
    with p.open("rb") as fh:
        resp = client.post(
            "/besser_api/import-owl",
            files={"owl_file": ("bad.txt", fh.read(), "text/plain")},
        )
    assert resp.status_code == 415
