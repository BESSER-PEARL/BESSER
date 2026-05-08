"""End-to-end tests for the LLM-cleanup endpoints.

We mock ``requests.post`` so no real network call is made. The tests
exercise the full FastAPI request → response pipeline including
multipart-form parsing for ``/llm-clean-kg`` and JSON-body parsing for
``/apply-kg-cleanup``.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from besser.utilities.web_modeling_editor.backend.backend import app


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
EX = "http://example.org/"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _kg_diagram() -> Dict[str, Any]:
    return {
        "title": "TestKG",
        "model": {
            "type": "KnowledgeGraphDiagram",
            "version": "1.0.0",
            "nodes": [
                {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
                {"id": "Car", "nodeType": "class", "label": "Car", "iri": EX + "Car"},
                {"id": "alice", "nodeType": "individual", "label": "Alice", "iri": EX + "alice"},
                {"id": "drives", "nodeType": "property", "label": "drives", "iri": EX + "drives"},
            ],
            "edges": [
                {"id": "t1", "source": "alice", "target": "Person", "iri": RDF_TYPE},
            ],
        },
    }


def _llm_payload(suggestions):
    body = json.dumps({"suggestions": suggestions})
    return {"choices": [{"message": {"content": body}}]}


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._p = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self):
        return self._p


# --------------------------------------------------------------------------
# /llm-clean-kg
# --------------------------------------------------------------------------


def test_llm_clean_kg_returns_issue_list(client: TestClient, monkeypatch):
    suggestion = {
        "code": "LLM_DROP_CLASS",
        "description": "Car is unrelated to a people-tracking system.",
        "affected_node_ids": ["Car"],
        "affected_edge_ids": [],
        "confidence": 0.95,
        "recommended_action": {
            "key": "drop_class",
            "parameters": {"node_id": "Car"},
            "label": "Drop the Car class",
        },
    }

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResp(_llm_payload([suggestion]))

    monkeypatch.setattr(
        "besser.BUML.notations.kg_to_buml.llm_cleanup.requests.post", fake_post,
    )

    response = client.post(
        "/besser_api/llm-clean-kg",
        data={
            "diagram": json.dumps(_kg_diagram()),
            "description": "I want to track people.",
            "api_key": "sk-test",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diagramType"] == "LLMCleanup"
    assert body["issueCount"] == 1
    assert body["issues"][0]["recommendedAction"]["key"] == "drop_class"
    assert body["issues"][0]["recommendedAction"]["parameters"] == {"node_id": "Car"}
    assert body["issues"][0]["skipAction"]["key"] == "noop"
    assert isinstance(body["kgSignature"], str) and body["kgSignature"]


def test_llm_clean_kg_rejects_empty_description(client: TestClient):
    response = client.post(
        "/besser_api/llm-clean-kg",
        data={
            "diagram": json.dumps(_kg_diagram()),
            "description": "   ",
            "api_key": "sk-test",
        },
    )
    assert response.status_code == 400


def test_llm_clean_kg_rejects_non_kg_payload(client: TestClient, monkeypatch):
    payload = {
        "title": "NotAKG",
        "model": {"type": "ClassDiagram", "elements": {}, "relationships": {}},
    }
    response = client.post(
        "/besser_api/llm-clean-kg",
        data={
            "diagram": json.dumps(payload),
            "description": "Build a system.",
            "api_key": "sk-test",
        },
    )
    assert response.status_code == 400


def test_llm_clean_kg_propagates_openai_error(client: TestClient, monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResp({"error": {"message": "rate limit"}}, status_code=429)

    monkeypatch.setattr(
        "besser.BUML.notations.kg_to_buml.llm_cleanup.requests.post", fake_post,
    )
    response = client.post(
        "/besser_api/llm-clean-kg",
        data={
            "diagram": json.dumps(_kg_diagram()),
            "description": "build it",
            "api_key": "sk-test",
        },
    )
    assert response.status_code == 502


# --------------------------------------------------------------------------
# /apply-kg-cleanup
# --------------------------------------------------------------------------


def _issue_payload(node_id: str = "Car") -> Dict[str, Any]:
    return {
        "id": "issue-1",
        "code": "LLM_DROP_CLASS",
        "description": "drop irrelevant class",
        "affectedNodeIds": [node_id],
        "affectedEdgeIds": [],
        "recommendedAction": {
            "key": "drop_class",
            "parameters": {"node_id": node_id},
            "label": "Drop",
        },
        "skipAction": {"key": "noop", "parameters": {}, "label": "Keep"},
    }


def test_apply_kg_cleanup_drops_accepted_class(client: TestClient):
    diagram = _kg_diagram()
    diagram["llmIssues"] = [_issue_payload("Car")]
    diagram["resolutions"] = [{"issueId": "issue-1", "decision": "accept"}]

    response = client.post("/besser_api/apply-kg-cleanup", json=diagram)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diagramType"] == "KnowledgeGraphDiagram"
    assert body["model"]["type"] == "KnowledgeGraphDiagram"
    node_ids = {n["id"] for n in body["model"]["nodes"]}
    assert "Car" not in node_ids
    assert "Person" in node_ids
    assert isinstance(body["kgSignature"], str)


def test_apply_kg_cleanup_skips_noop(client: TestClient):
    diagram = _kg_diagram()
    diagram["llmIssues"] = [_issue_payload("Car")]
    diagram["resolutions"] = [{"issueId": "issue-1", "decision": "skip"}]

    response = client.post("/besser_api/apply-kg-cleanup", json=diagram)
    assert response.status_code == 200, response.text
    body = response.json()
    node_ids = {n["id"] for n in body["model"]["nodes"]}
    assert "Car" in node_ids


def test_apply_kg_cleanup_signature_mismatch_returns_400(client: TestClient):
    diagram = _kg_diagram()
    diagram["llmIssues"] = [_issue_payload("Car")]
    diagram["resolutions"] = [{"issueId": "issue-1", "decision": "accept"}]
    diagram["kgSignature"] = "0" * 16  # deliberately wrong

    response = client.post("/besser_api/apply-kg-cleanup", json=diagram)
    assert response.status_code == 400


def test_apply_kg_cleanup_missing_issues_400(client: TestClient):
    diagram = _kg_diagram()
    diagram["resolutions"] = [{"issueId": "issue-1", "decision": "accept"}]
    # No llmIssues field — endpoint must reject because it can't reconstruct the issue.
    response = client.post("/besser_api/apply-kg-cleanup", json=diagram)
    assert response.status_code == 400


def test_apply_kg_cleanup_no_decisions_returns_unchanged(client: TestClient):
    diagram = _kg_diagram()
    response = client.post("/besser_api/apply-kg-cleanup", json=diagram)
    assert response.status_code == 200
    body = response.json()
    node_ids = {n["id"] for n in body["model"]["nodes"]}
    assert node_ids == {"Person", "Car", "alice", "drives"}


# --------------------------------------------------------------------------
# /apply-kg-refinement
# --------------------------------------------------------------------------


def _kg_with_orphan_diagram() -> Dict[str, Any]:
    """A KG where 'Car' is class-anchored and 'ghost1'/'ghost2' are orphans."""
    return {
        "title": "RefineTestKG",
        "model": {
            "type": "KnowledgeGraphDiagram",
            "version": "1.0.0",
            "nodes": [
                {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
                {"id": "Car", "nodeType": "class", "label": "Car", "iri": EX + "Car"},
                {"id": "ghost1", "nodeType": "individual", "label": "ghost1", "iri": EX + "ghost1"},
                {"id": "ghost2", "nodeType": "individual", "label": "ghost2", "iri": EX + "ghost2"},
            ],
            "edges": [
                {"id": "ge", "source": "ghost1", "target": "ghost2", "iri": EX + "rel"},
            ],
        },
    }


def test_apply_kg_refinement_static_drops_orphans_on_accept(client: TestClient):
    diagram = _kg_with_orphan_diagram()
    pre = client.post(
        "/besser_api/analyze-kg-for-buml-conversion",
        json=diagram,
    ).json()
    orphan = next(i for i in pre["issues"] if i["code"] == "ORPHAN_NODE_NO_CLASS_LINK")

    payload = {
        **diagram,
        "source": "static",
        "kgSignature": pre["kgSignature"],
        "resolutions": [{"issueId": orphan["id"], "decision": "accept"}],
    }
    resp = client.post("/besser_api/apply-kg-refinement", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["diagramType"] == "KnowledgeGraphDiagram"
    assert body["pendingOrphanClassification"] is None
    node_ids = {n["id"] for n in body["model"]["nodes"]}
    assert "ghost1" not in node_ids
    assert "ghost2" not in node_ids
    assert "Person" in node_ids
    assert "Car" in node_ids


def test_apply_kg_refinement_static_skip_defers_to_llm(client: TestClient):
    diagram = _kg_with_orphan_diagram()
    pre = client.post(
        "/besser_api/analyze-kg-for-buml-conversion",
        json=diagram,
    ).json()
    orphan = next(i for i in pre["issues"] if i["code"] == "ORPHAN_NODE_NO_CLASS_LINK")

    payload = {
        **diagram,
        "source": "static",
        "kgSignature": pre["kgSignature"],
        "resolutions": [{"issueId": orphan["id"], "decision": "skip"}],
    }
    resp = client.post("/besser_api/apply-kg-refinement", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Orphans stay in the KG (deferred — not dropped).
    node_ids = {n["id"] for n in body["model"]["nodes"]}
    assert "ghost1" in node_ids
    assert "ghost2" in node_ids
    # Pending classification carries the orphan ids and a fresh signature.
    assert body["pendingOrphanClassification"] is not None
    assert set(body["pendingOrphanClassification"]["nodeIds"]) == {"ghost1", "ghost2"}
    assert body["pendingOrphanClassification"]["kgSignature"] == body["kgSignature"]


def test_apply_kg_refinement_llm_path_drops_class(client: TestClient):
    diagram = _kg_diagram()
    payload = {
        **diagram,
        "source": "llm",
        "llmIssues": [_issue_payload("Car")],
        "resolutions": [{"issueId": "issue-1", "decision": "accept"}],
    }
    resp = client.post("/besser_api/apply-kg-refinement", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pendingOrphanClassification"] is None
    node_ids = {n["id"] for n in body["model"]["nodes"]}
    assert "Car" not in node_ids


def test_apply_kg_refinement_invalid_source_returns_4xx(client: TestClient):
    """Pydantic's Literal validation rejects unknown ``source`` values with 422."""
    diagram = _kg_diagram()
    payload = {**diagram, "source": "bogus"}
    resp = client.post("/besser_api/apply-kg-refinement", json=payload)
    assert resp.status_code in (400, 422)


def test_apply_kg_refinement_signature_mismatch_returns_400(client: TestClient):
    diagram = _kg_with_orphan_diagram()
    payload = {
        **diagram,
        "source": "static",
        "kgSignature": "0" * 16,
        "resolutions": [],
    }
    resp = client.post("/besser_api/apply-kg-refinement", json=payload)
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# /classify-orphans-with-llm
# --------------------------------------------------------------------------


def test_classify_orphans_with_llm_returns_per_node_issues(client: TestClient, monkeypatch):
    diagram = _kg_with_orphan_diagram()
    suggestion = {
        "code": "LLM_TYPE_INDIVIDUAL",
        "description": "ghost1 fits Person.",
        "affected_node_ids": ["ghost1"],
        "affected_edge_ids": [],
        "confidence": 0.9,
        "recommended_action": {
            "key": "type_individual_as_class",
            "parameters": {"node_id": "ghost1", "class_id": "Person"},
            "label": "Type ghost1 as Person",
        },
    }

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResp(_llm_payload([suggestion]))

    monkeypatch.setattr(
        "besser.BUML.notations.kg_to_buml.llm_cleanup.requests.post", fake_post,
    )

    response = client.post(
        "/besser_api/classify-orphans-with-llm",
        data={
            "diagram": json.dumps(diagram),
            "description": "Track people.",
            "api_key": "sk-test",
            "node_ids": json.dumps(["ghost1", "ghost2"]),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diagramType"] == "OrphanClassification"
    # ghost1 → typed; ghost2 → synthetic drop_node fallback.
    assert body["issueCount"] == 2
    by_target = {i["recommendedAction"]["parameters"]["node_id"]: i for i in body["issues"]}
    assert by_target["ghost1"]["recommendedAction"]["key"] == "type_individual_as_class"
    assert by_target["ghost2"]["recommendedAction"]["key"] == "drop_node"


def test_classify_orphans_with_llm_rejects_empty_node_ids(client: TestClient):
    diagram = _kg_with_orphan_diagram()
    response = client.post(
        "/besser_api/classify-orphans-with-llm",
        data={
            "diagram": json.dumps(diagram),
            "description": "Track people.",
            "api_key": "sk-test",
            "node_ids": json.dumps([]),
        },
    )
    assert response.status_code == 400


def test_classify_orphans_with_llm_rejects_missing_api_key(client: TestClient):
    diagram = _kg_with_orphan_diagram()
    response = client.post(
        "/besser_api/classify-orphans-with-llm",
        data={
            "diagram": json.dumps(diagram),
            "description": "Track people.",
            "api_key": "   ",
            "node_ids": json.dumps(["ghost1"]),
        },
    )
    assert response.status_code == 400


def test_classify_orphans_with_llm_rejects_invalid_node_ids_json(client: TestClient):
    diagram = _kg_with_orphan_diagram()
    response = client.post(
        "/besser_api/classify-orphans-with-llm",
        data={
            "diagram": json.dumps(diagram),
            "description": "desc",
            "api_key": "sk-test",
            "node_ids": "not_json",
        },
    )
    assert response.status_code == 400
