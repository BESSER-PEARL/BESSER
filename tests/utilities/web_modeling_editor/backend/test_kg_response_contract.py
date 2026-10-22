"""Pin the wire shape of the KG endpoints that return a typed response.

The editor reads these field names directly (``useKgPreflight.ts``,
``useKgRefine.ts``, ``KgConsistencyConfirmModal.tsx``, ``KgRefineModal.tsx``),
so a renamed or dropped field is a silent frontend break. The response models
in ``models/responses.py`` declare the contract; these tests make sure the
endpoints actually emit exactly that.

Every field is camelCase, including the issues of a consistency report.
"""

from __future__ import annotations

import json

from tests.utilities.web_modeling_editor.backend.test_kg_validate_endpoint import DISJOINT_TTL
from tests.utilities.web_modeling_editor.backend.test_llm_clean_kg_endpoint import (
    _FakeResp,
    _kg_diagram,
    _kg_with_orphan_diagram,
)

_API = "/besser_api"

PREFLIGHT_KEYS = {"kgSignature", "diagramType", "issueCount", "issues"}
ISSUE_KEYS = {
    "id", "code", "description", "affectedNodeIds", "affectedEdgeIds",
    "recommendedAction", "skipAction",
}
ACTION_KEYS = {"key", "parameters", "label"}
CONSISTENCY_KEYS = {
    "issues", "issueCount", "severityCounts", "kgSignature", "inferenceUsed",
    "exportedAt", "version",
}
CONSISTENCY_ISSUE_KEYS = {
    "id", "code", "severity", "message", "affectedNodeIds", "affectedEdgeIds",
    "constraintNodeId", "specKind", "pathIri", "constraintLabel",
}
REFINEMENT_KEYS = {
    "title", "model", "diagramType", "kgSignature", "pendingOrphanClassification",
    "exportedAt", "version",
}


def _assert_preflight_shape(body: dict) -> None:
    assert set(body) == PREFLIGHT_KEYS
    for issue in body["issues"]:
        assert set(issue) == ISSUE_KEYS
        for action in (issue["recommendedAction"], issue["skipAction"]):
            assert action is None or set(action) == ACTION_KEYS


def test_analyze_response_shape(client):
    body = client.post(f"{_API}/kg/preflight", json=_kg_with_orphan_diagram()).json()

    _assert_preflight_shape(body)
    assert body["issues"], "fixture should raise at least one issue to pin the issue shape"


def test_llm_clean_and_classify_share_the_preflight_shape(client, monkeypatch):
    reply = {"choices": [{"message": {"content": json.dumps({"suggestions": [{
        "code": "LLM_DROP_CLASS", "description": "x",
        "affected_node_ids": ["Car"], "affected_edge_ids": [],
        "recommended_action": {"key": "drop_class", "parameters": {"node_id": "Car"}, "label": "Drop"},
    }]})}}]}
    monkeypatch.setattr("requests.post", lambda *a, **kw: _FakeResp(reply))
    common = {"diagram": _kg_diagram(), "description": "people", "api_key": "sk-x"}

    clean = client.post(f"{_API}/kg/llm-clean", json=common).json()
    classify = client.post(
        f"{_API}/kg/classify-orphans", json={**common, "nodeIds": ["alice"]},
    ).json()

    _assert_preflight_shape(clean)
    _assert_preflight_shape(classify)


def test_consistency_response_is_camel_case_throughout(client):
    imported = client.post(
        f"{_API}/kg/import-owl",
        files={"owl_file": ("d.ttl", DISJOINT_TTL.encode(), "text/turtle")},
    ).json()

    body = client.post(
        f"{_API}/kg/consistency", json={"title": "t", "model": imported["model"]},
    ).json()

    assert set(body) == CONSISTENCY_KEYS
    assert body["issues"], "fixture should violate disjointness"
    for issue in body["issues"]:
        assert set(issue) == CONSISTENCY_ISSUE_KEYS


def test_apply_refinement_response_shape(client):
    diagram = _kg_with_orphan_diagram()
    pre = client.post(f"{_API}/kg/preflight", json=diagram).json()
    orphan = next(i for i in pre["issues"] if i["code"] == "ORPHAN_NODE_NO_CLASS_LINK")

    body = client.post(f"{_API}/kg/apply-refinement", json={
        **diagram, "source": "static", "kgSignature": pre["kgSignature"],
        "resolutions": [{"issueId": orphan["id"], "decision": "skip"}],
    }).json()

    assert set(body) == REFINEMENT_KEYS
    assert set(body["pendingOrphanClassification"]) == {"nodeIds", "kgSignature"}


def test_typed_kg_endpoints_are_declared_in_openapi(client):
    """The point of the models: the contract is now visible to API consumers."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    for name in (
        "KGPreflightReportResponse", "KGIssueResponse", "KGActionResponse",
        "KGConsistencyReportResponse", "KGConsistencyIssueResponse",
        "KGRefinementResponse", "KGPendingOrphanClassification",
    ):
        assert name in schemas, name
