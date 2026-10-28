"""Request validation of the ``/besser_api/kg`` endpoints.

* A malformed KG diagram is a 400 on every KG endpoint, never a generic 500
  (``process_kg_diagram`` calls ``.get`` on each node and edge, so a non-object
  entry would otherwise escape as a bare ``AttributeError``).
* The request models pin the wire contract: camelCase data fields
  (``nodeIds``, ``issueId``, typed ``resolutions`` / ``llmIssues``), and the
  ``provider`` / ``api_key`` / ``llm_model`` credentials. A body that does not
  match is a 422.
"""

from __future__ import annotations

import pytest

_API = "/besser_api"


def _malformed(nodes=None, edges=None):
    return {
        "title": "Broken",
        "model": {
            "type": "KnowledgeGraphDiagram",
            "nodes": ["x"] if nodes is None else nodes,
            "edges": [] if edges is None else edges,
        },
    }


@pytest.mark.parametrize("path", [
    "/kg/preflight",
    "/kg/to-class-diagram",
    "/kg/apply-refinement",
    "/kg/consistency",
    "/kg/export/ttl",
])
def test_a_non_object_node_is_a_400(client, path):
    response = client.post(f"{_API}{path}", json=_malformed())

    assert response.status_code == 400, response.text
    assert "must be an object" in response.json()["detail"]


def test_a_non_object_edge_is_a_400(client):
    nodes = [{"id": "Person", "nodeType": "class", "label": "Person"}]
    response = client.post(f"{_API}/kg/preflight", json=_malformed(nodes, [42]))

    assert response.status_code == 400, response.text


def test_validate_diagram_reports_a_non_object_node_as_invalid(client):
    response = client.post(f"{_API}/validate-diagram", json=_malformed())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["isValid"] is False
    assert any("must be an object" in e for e in body["errors"])


# --------------------------------------------------------------------------
# The request contract
# --------------------------------------------------------------------------

_KG = {
    "title": "G",
    "model": {
        "type": "KnowledgeGraphDiagram",
        "nodes": [{"id": "Person", "nodeType": "class", "label": "Person"}],
        "edges": [],
    },
}


def _field_errors(response) -> set:
    assert response.status_code == 422, response.text
    return {err["loc"][-1] for err in response.json()["detail"]}


def test_classify_orphans_takes_node_ids_in_camel_case(client):
    response = client.post(f"{_API}/kg/classify-orphans", json={
        "diagram": _KG, "api_key": "sk-test", "node_ids": ["Person"],
    })
    assert "nodeIds" in _field_errors(response)


def test_a_decision_must_name_its_issue_as_issue_id_in_camel_case(client):
    response = client.post(f"{_API}/kg/apply-refinement", json={
        **_KG, "resolutions": [{"issue_id": "x", "decision": "accept"}],
    })
    assert "issueId" in _field_errors(response)


def test_llm_issues_are_typed(client):
    response = client.post(f"{_API}/kg/apply-refinement", json={
        **_KG, "source": "llm",
        "llmIssues": [{"id": "i1", "recommendedAction": "drop it"}],
        "resolutions": [{"issueId": "i1", "decision": "accept"}],
    })
    assert {"code", "description", "recommendedAction"} <= _field_errors(response)


def test_the_issue_list_the_llm_endpoints_return_is_accepted_back_unchanged(client):
    """``llmIssues`` is the ``issues`` of a ``/kg/llm-clean`` response, as is."""
    issue = {
        "id": "i1", "code": "LLM_DROP_CLASS", "description": "Person is unrelated.",
        "affectedNodeIds": ["Person"], "affectedEdgeIds": [],
        "recommendedAction": {"key": "drop_class", "parameters": {"node_id": "Person"}, "label": "Drop"},
        "skipAction": {"key": "noop", "parameters": {}, "label": "Keep"},
    }
    response = client.post(f"{_API}/kg/apply-refinement", json={
        **_KG, "source": "llm", "llmIssues": [issue],
        "resolutions": [{"issueId": "i1", "decision": "accept"}],
    })
    assert response.status_code == 200, response.text
    assert response.json()["model"]["nodes"] == []


@pytest.mark.parametrize("path, extra", [
    ("/kg/llm-clean", {}),
    ("/kg/classify-orphans", {"nodeIds": ["Person"]}),
])
def test_the_llm_model_name_is_format_checked(client, path, extra):
    response = client.post(f"{_API}{path}", json={
        "diagram": _KG, "api_key": "sk-test", "llm_model": "gpt-4o; rm -rf /", **extra,
    })
    assert "llm_model" in _field_errors(response)


@pytest.mark.parametrize("old_path", [
    "/kg-config", "/import-owl", "/export-kg-rdf/ttl", "/check-kg-consistency",
    "/analyze-kg-for-buml-conversion", "/kg-to-class-diagram", "/llm-clean-kg",
    "/classify-orphans-with-llm", "/apply-kg-refinement",
])
def test_the_unreleased_kg_paths_are_gone(client, old_path):
    method = client.get if old_path == "/kg-config" else client.post
    assert method(f"{_API}{old_path}").status_code in (404, 405)
