"""A malformed KG diagram is a 400 on every KG endpoint, never a generic 500.

``process_kg_diagram`` calls ``.get`` on each node and edge, so a non-object
entry (``"nodes": ["x"]``) used to escape as a bare ``AttributeError``.
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
    "/analyze-kg-for-buml-conversion",
    "/kg-to-class-diagram",
    "/apply-kg-refinement",
    "/check-kg-consistency",
    "/export-kg-rdf/ttl",
])
def test_a_non_object_node_is_a_400(client, path):
    response = client.post(f"{_API}{path}", json=_malformed())

    assert response.status_code == 400, response.text
    assert "must be an object" in response.json()["detail"]


def test_a_non_object_edge_is_a_400(client):
    nodes = [{"id": "Person", "nodeType": "class", "label": "Person"}]
    response = client.post(f"{_API}/analyze-kg-for-buml-conversion", json=_malformed(nodes, [42]))

    assert response.status_code == 400, response.text


def test_validate_diagram_reports_a_non_object_node_as_invalid(client):
    response = client.post(f"{_API}/validate-diagram", json=_malformed())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["isValid"] is False
    assert any("must be an object" in e for e in body["errors"])
