"""Validating a KnowledgeGraphDiagram: ``/validate-diagram`` and ``/kg/consistency``.

``/validate-diagram`` runs the two checks the editor's Refine KG modal runs (the
static preflight and the OWL/SHACL consistency check) and reports their findings
without the modal's recommended/skip actions, which are interactive choices with
no meaning in a validation response. ``/kg/consistency`` returns the consistency
report itself. Both share ``services/kg/validation.py``: its time budget, its
concurrency cap, and how a check that cannot finish is reported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from besser.utilities.web_modeling_editor.backend.services.kg import validation as kg_validation

from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph
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


def _payload(ttl: str, tmp_path: Path) -> dict:
    path = tmp_path / "graph.ttl"
    path.write_text(ttl, encoding="utf-8")
    kg = owl_file_to_knowledge_graph(str(path))
    envelope = kg_to_json(kg)
    model = {**envelope["model"], "type": "KnowledgeGraphDiagram"}
    return {"title": "Graph", "model": model}


def _validate(client, ttl: str, tmp_path: Path) -> dict:
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


# --------------------------------------------------------------------------
# Graphs the validator cannot check, or not in time
# --------------------------------------------------------------------------

_CONSISTENCY = "/besser_api/kg/consistency"

_SHACL_PREFIX = """
@prefix : <http://ex.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
:Person a owl:Class .
"""

# A property shape without sh:path: pyshacl refuses to load the shapes graph.
PATHLESS_PROPERTY_SHAPE_TTL = _SHACL_PREFIX + """
:alice a :Person .
:S a sh:NodeShape ; sh:targetClass :Person ; sh:property :PS .
:PS a sh:PropertyShape ; sh:minCount 2 .
"""

UNDECLARED_PROPERTY_SHAPE_TTL = _SHACL_PREFIX + """
:alice a :Person ; :nickname "aaa" .
:S a sh:NodeShape ; sh:targetClass :Person ; sh:property :PS .
:PS a sh:PropertyShape ; sh:path :nickname ; sh:minCount 2 .
"""

# ``^(a+)+$`` backtracks exponentially on this value: only the timeout stops it.
RUNAWAY_PATTERN_TTL = _SHACL_PREFIX + """
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:S a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:pattern "^(a+)+$" ] .
:alice a :Person ; :name "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!" .
"""


@pytest.fixture
def one_slot(monkeypatch):
    """A consistency-check cap of one, rebuilt for the test and reset after it."""
    monkeypatch.setattr(kg_validation, "KG_CONSISTENCY_MAX_CONCURRENT", 1)
    kg_validation._reset_slots_for_tests()
    yield kg_validation._slots()
    kg_validation._reset_slots_for_tests()


def test_unloadable_shapes_are_a_400_not_a_500(client, tmp_path):
    resp = client.post(_CONSISTENCY, json=_payload(PATHLESS_PROPERTY_SHAPE_TTL, tmp_path))
    assert resp.status_code == 400, resp.text
    assert "SHACL shapes could not be loaded" in resp.json()["detail"]


def test_validate_diagram_reports_unloadable_shapes_as_an_error(client, tmp_path):
    result = _validate(client, PATHLESS_PROPERTY_SHAPE_TTL, tmp_path)
    assert result["isValid"] is False
    assert any("SHACL shapes could not be loaded" in e for e in result["errors"])


def test_a_shape_on_an_undeclared_property_is_checked(client, tmp_path):
    resp = client.post(_CONSISTENCY, json=_payload(UNDECLARED_PROPERTY_SHAPE_TTL, tmp_path))
    assert resp.status_code == 200, resp.text
    assert [i["code"] for i in resp.json()["issues"]] == ["MinCountConstraintComponent"]


def test_a_runaway_pattern_times_out_as_a_400(client, tmp_path, monkeypatch):
    monkeypatch.setattr(kg_validation, "KG_CONSISTENCY_TIMEOUT_SECONDS", 2)
    resp = client.post(_CONSISTENCY, json=_payload(RUNAWAY_PATTERN_TTL, tmp_path))
    assert resp.status_code == 400, resp.text
    assert "did not finish within 2 seconds" in resp.json()["detail"]


def test_validate_diagram_reports_a_timeout_as_a_warning(client, tmp_path, monkeypatch):
    """Running out of time says nothing about the graph: it stays valid."""
    monkeypatch.setattr(kg_validation, "KG_CONSISTENCY_TIMEOUT_SECONDS", 2)
    result = _validate(client, RUNAWAY_PATTERN_TTL, tmp_path)
    assert result["isValid"] is True, result["errors"]
    assert any("did not finish within 2 seconds" in w for w in result["warnings"])


def test_a_check_beyond_the_cap_is_a_429(client, tmp_path, one_slot):
    assert one_slot.acquire(blocking=False)
    try:
        resp = client.post(_CONSISTENCY, json=_payload(CLEAN_TTL, tmp_path))
    finally:
        one_slot.release()
    assert resp.status_code == 429, resp.text
    assert "retry" in resp.json()["detail"].lower()


def test_validate_diagram_reports_a_busy_checker_as_a_warning(client, tmp_path, one_slot):
    assert one_slot.acquire(blocking=False)
    try:
        result = _validate(client, CLEAN_TTL, tmp_path)
    finally:
        one_slot.release()
    assert result["isValid"] is True
    assert any("Too many consistency checks" in w for w in result["warnings"])


def test_each_check_gives_its_slot_back(client, tmp_path, one_slot):
    for ttl in (CLEAN_TTL, PATHLESS_PROPERTY_SHAPE_TTL, CLEAN_TTL):
        client.post(_CONSISTENCY, json=_payload(ttl, tmp_path))
    assert one_slot.acquire(blocking=False)
    one_slot.release()
