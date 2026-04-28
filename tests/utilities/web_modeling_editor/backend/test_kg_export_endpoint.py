"""End-to-end tests for the ``/export-kg-rdf/{fmt}`` endpoint.

Drives the full pipeline: import an OWL/TTL fixture, then export it back as
either Turtle or RDF/XML and assert the response shape (status, content type,
disposition header, body parses cleanly).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rdflib
from starlette.testclient import TestClient

from besser.utilities.web_modeling_editor.backend.backend import app


TTL_FIXTURE = """
@prefix : <http://ex.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Person a owl:Class .
:knows a owl:ObjectProperty .
:age a owl:DatatypeProperty .
:alice a :Person ; :knows :bob ; :age "30"^^xsd:integer .
:bob a :Person .
""".strip()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def ttl_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.ttl"
    p.write_text(TTL_FIXTURE, encoding="utf-8")
    return p


@pytest.fixture
def kg_diagram_payload(client: TestClient, ttl_path: Path) -> dict:
    """Import the fixture via /import-owl and reshape into a DiagramInput."""
    with ttl_path.open("rb") as fh:
        resp = client.post(
            "/besser_api/import-owl",
            files={"owl_file": ("tiny.ttl", fh.read(), "text/turtle")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return {"title": body["title"], "model": body["model"]}


def test_export_kg_ttl_returns_turtle(client: TestClient, kg_diagram_payload: dict):
    resp = client.post("/besser_api/export-kg-rdf/ttl", json=kg_diagram_payload)
    assert resp.status_code == 200, resp.text
    assert "text/turtle" in resp.headers.get("content-type", "")
    assert ".ttl" in resp.headers.get("content-disposition", "")
    g = rdflib.Graph().parse(data=resp.text, format="turtle")
    assert len(g) > 0


def test_export_kg_owl_returns_rdfxml(client: TestClient, kg_diagram_payload: dict):
    resp = client.post("/besser_api/export-kg-rdf/owl", json=kg_diagram_payload)
    assert resp.status_code == 200, resp.text
    assert "application/rdf+xml" in resp.headers.get("content-type", "")
    assert ".owl" in resp.headers.get("content-disposition", "")
    g = rdflib.Graph().parse(data=resp.text, format="xml")
    assert len(g) > 0


def test_export_kg_unsupported_format_returns_400(client: TestClient, kg_diagram_payload: dict):
    resp = client.post("/besser_api/export-kg-rdf/json", json=kg_diagram_payload)
    assert resp.status_code == 400
    assert "owl" in resp.json()["detail"] or "ttl" in resp.json()["detail"]


def test_export_kg_rejects_non_kg_payload(client: TestClient):
    class_payload = {
        "title": "NotKG",
        "model": {
            "type": "ClassDiagram",
            "version": "1.0.0",
            "elements": {},
            "relationships": {},
        },
    }
    resp = client.post("/besser_api/export-kg-rdf/ttl", json=class_payload)
    assert resp.status_code == 400
    assert "KnowledgeGraphDiagram" in resp.json()["detail"]


def test_export_kg_filename_uses_diagram_title(client: TestClient, kg_diagram_payload: dict):
    payload = {**kg_diagram_payload, "title": "My Ontology"}
    resp = client.post("/besser_api/export-kg-rdf/ttl", json=payload)
    assert resp.status_code == 200
    cd = resp.headers.get("content-disposition", "")
    assert "My_Ontology.ttl" in cd or "my_ontology.ttl" in cd.lower()


def test_round_trip_via_endpoints(client: TestClient, ttl_path: Path, tmp_path: Path):
    """Import → export → re-import should preserve node/edge counts."""
    with ttl_path.open("rb") as fh:
        first = client.post(
            "/besser_api/import-owl",
            files={"owl_file": ("tiny.ttl", fh.read(), "text/turtle")},
        )
    assert first.status_code == 200
    first_body = first.json()
    first_node_count = len(first_body["model"]["nodes"])
    first_edge_count = len(first_body["model"]["edges"])

    export_resp = client.post(
        "/besser_api/export-kg-rdf/ttl",
        json={"title": first_body["title"], "model": first_body["model"]},
    )
    assert export_resp.status_code == 200
    exported_ttl = export_resp.text

    out_path = tmp_path / "round.ttl"
    out_path.write_text(exported_ttl, encoding="utf-8")
    with out_path.open("rb") as fh:
        second = client.post(
            "/besser_api/import-owl",
            files={"owl_file": ("round.ttl", fh.read(), "text/turtle")},
        )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["diagramType"] == "KnowledgeGraphDiagram"
    assert len(second_body["model"]["nodes"]) == first_node_count
    assert len(second_body["model"]["edges"]) == first_edge_count
