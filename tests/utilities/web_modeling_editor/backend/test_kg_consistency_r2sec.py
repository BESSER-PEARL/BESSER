"""/check-kg-consistency and /validate-diagram on KGs the validator cannot check.

A SHACL property shape without ``sh:path`` makes pyshacl refuse the shapes
graph with a ``ReportableRuntimeError``, which used to surface as a generic
HTTP 500. (A shape on a property the KG never declares used to lose its
``sh:path`` on the way to pyshacl; it now keeps it and is checked normally.)
"""

from __future__ import annotations

import time
from pathlib import Path

from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph
from besser.utilities.web_modeling_editor.backend.services.converters import kg_to_json

_PREFIX = """
@prefix : <http://ex.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
:Person a owl:Class .
"""

PATHLESS_PROPERTY_SHAPE_TTL = _PREFIX + """
:alice a :Person .
:S a sh:NodeShape ; sh:targetClass :Person ; sh:property :PS .
:PS a sh:PropertyShape ; sh:minCount 2 .
"""

UNDECLARED_PROPERTY_SHAPE_TTL = _PREFIX + """
:alice a :Person ; :nickname "aaa" .
:S a sh:NodeShape ; sh:targetClass :Person ; sh:property :PS .
:PS a sh:PropertyShape ; sh:path :nickname ; sh:minCount 2 .
"""

EXPONENTIAL_PATTERN_TTL = _PREFIX + """
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:S a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:pattern "^(a+)+$" ] .
:alice a :Person ; :name "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!" .
"""


def _payload(ttl: str, tmp_path: Path) -> dict:
    path = tmp_path / "graph.ttl"
    path.write_text(ttl, encoding="utf-8")
    envelope = kg_to_json(owl_file_to_knowledge_graph(str(path)))
    return {"title": "Graph", "model": {**envelope["model"], "type": "KnowledgeGraphDiagram"}}


def test_unloadable_shapes_are_a_400_not_a_500(client, tmp_path):
    resp = client.post("/besser_api/check-kg-consistency", json=_payload(PATHLESS_PROPERTY_SHAPE_TTL, tmp_path))
    assert resp.status_code == 400, resp.text
    assert "SHACL shapes could not be loaded" in resp.json()["detail"]


def test_validate_diagram_reports_unloadable_shapes_as_an_error(client, tmp_path):
    resp = client.post("/besser_api/validate-diagram", json=_payload(PATHLESS_PROPERTY_SHAPE_TTL, tmp_path))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["isValid"] is False
    assert any("SHACL shapes could not be loaded" in e for e in body["errors"])


def test_a_shape_on_an_undeclared_property_is_checked(client, tmp_path):
    resp = client.post("/besser_api/check-kg-consistency", json=_payload(UNDECLARED_PROPERTY_SHAPE_TTL, tmp_path))
    assert resp.status_code == 200, resp.text
    issues = resp.json()["issues"]
    assert [i["code"] for i in issues] == ["MinCountConstraintComponent"]


def test_an_exponential_pattern_is_reported_promptly(client, tmp_path):
    started = time.monotonic()
    resp = client.post("/besser_api/check-kg-consistency", json=_payload(EXPONENTIAL_PATTERN_TTL, tmp_path))
    assert resp.status_code == 200, resp.text
    assert time.monotonic() - started < 15
    codes = [i["code"] for i in resp.json()["issues"]]
    assert "UnsafePatternSkipped" in codes
