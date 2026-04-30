"""End-to-end test mirroring the manual browser smoke test.

Imports the ``tests/fixtures/complex_kg.ttl`` fixture (which intentionally
triggers most preflight issues), runs the analyze endpoint, then converts
to a Class Diagram and to an Object Diagram with a mix of accept/skip
decisions. Verifies that:

* The analyze endpoint surfaces every expected issue code.
* Conversions produce a valid BUML diagram with the user-chosen behaviour
  (accept = recommended fix applied; skip = element dropped).
* The kgSignature is enforced (mismatched signature returns 400).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph
from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.converters import kg_to_json


FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "complex_kg.ttl"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def kg_payload() -> dict:
    """Load the fixture once via the OWL importer and convert to the KG diagram JSON."""
    assert FIXTURE.exists(), f"Fixture missing: {FIXTURE}"
    kg = owl_file_to_knowledge_graph(str(FIXTURE))
    diagram_json = kg_to_json(kg)
    return {
        "title": "ComplexKG",
        "model": {**diagram_json["model"], "type": "KnowledgeGraphDiagram"},
    }


def test_analyze_surfaces_expected_issue_codes(client: TestClient, kg_payload: dict):
    resp = client.post("/besser_api/analyze-kg-for-buml-conversion", json=kg_payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    codes = {i["code"] for i in body["issues"]}
    # Every issue type the fixture is designed to trigger:
    expected = {
        "PROPERTY_NO_DOMAIN",
        "MULTIPLE_DOMAINS",
        "RESTRICTION_UNATTACHED",
        "RESTRICTION_UNSUPPORTED",
        "PROPERTY_NO_RANGE",
        "PUNNING",
        "EQUIVALENT_CLASSES",
        "INVERSE_PROPERTY",
        "BLANK_NODE_INSTANCE",
        "MULTIVALUED_LITERAL",
    }
    missing = expected - codes
    assert not missing, f"preflight missed: {missing}; saw codes={sorted(codes)}"
    # Every issue carries a recommended_action and a skip_action.
    for issue in body["issues"]:
        assert issue["recommendedAction"] is not None, issue
        assert issue["skipAction"] is not None, issue


def test_class_diagram_with_all_accepts(client: TestClient, kg_payload: dict):
    pre = client.post("/besser_api/analyze-kg-for-buml-conversion", json=kg_payload).json()
    resolutions = [{"issueId": i["id"], "decision": "accept"} for i in pre["issues"]]
    resp = client.post(
        "/besser_api/kg-to-class-diagram",
        json={**kg_payload, "kgSignature": pre["kgSignature"], "resolutions": resolutions},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["diagramType"] == "ClassDiagram"
    elements = body["model"]["elements"]
    class_names = {e["name"] for e in elements.values() if e.get("type") == "Class"}
    # Person and Pet survive; Thing exists because :likes was attached to it.
    assert "Person" in class_names
    assert "Pet" in class_names


def test_class_diagram_with_skip_drops_property(client: TestClient, kg_payload: dict):
    """When the user skips PROPERTY_NO_DOMAIN, the :likes property is dropped."""
    pre = client.post("/besser_api/analyze-kg-for-buml-conversion", json=kg_payload).json()
    resolutions = []
    for issue in pre["issues"]:
        decision = "skip" if issue["code"] == "PROPERTY_NO_DOMAIN" else "accept"
        resolutions.append({"issueId": issue["id"], "decision": decision})
    resp = client.post(
        "/besser_api/kg-to-class-diagram",
        json={**kg_payload, "kgSignature": pre["kgSignature"], "resolutions": resolutions},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    relationships = body["model"]["relationships"]
    rel_names = {r.get("name") for r in relationships.values()}
    assert "likes" not in rel_names


def test_object_diagram_pipeline(client: TestClient, kg_payload: dict):
    """Object-diagram preflight + accept-everything conversion produces objects."""
    pre = client.post(
        "/besser_api/analyze-kg-for-buml-conversion?diagramType=ObjectDiagram",
        json=kg_payload,
    ).json()
    assert pre["diagramType"] == "ObjectDiagram"
    codes = {i["code"] for i in pre["issues"]}
    # Object-only issues should appear; class-only ones should not.
    assert "BLANK_NODE_AS_OBJECT" in codes
    assert "PROPERTY_NO_DOMAIN" not in codes  # class-only
    resolutions = [{"issueId": i["id"], "decision": "accept"} for i in pre["issues"]]
    resp = client.post(
        "/besser_api/kg-to-object-diagram",
        json={**kg_payload, "kgSignature": pre["kgSignature"], "resolutions": resolutions},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["diagramType"] == "ObjectDiagram"
    obj_names = {
        e["name"] for e in body["model"]["elements"].values() if e.get("type") == "ObjectName"
    }
    # alice and bob (and rex) are individuals in the fixture.
    assert {"alice", "bob"} <= obj_names


def test_stale_signature_blocks_conversion(client: TestClient, kg_payload: dict):
    resp = client.post(
        "/besser_api/kg-to-class-diagram",
        json={**kg_payload, "kgSignature": "deadbeef00000000"},
    )
    assert resp.status_code == 400
