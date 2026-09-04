"""End-to-end tests for the KG preflight + conversion endpoints (v2 shape).

The v2 payload is ``[{issueId, decision: "accept" | "skip"}]``. The
backend re-runs the preflight, looks up each issue, dispatches its
recommended/skip action. v1-shaped payloads (``{issueId, choice,
parameters}``) still work for backward compat.
"""

from __future__ import annotations


RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
OWL = "http://www.w3.org/2002/07/owl#"
EX = "http://example.org/"


def _kg_payload(nodes, edges, title="PreflightTest") -> dict:
    return {
        "title": title,
        "model": {
            "type": "KnowledgeGraphDiagram",
            "version": "1.0.0",
            "nodes": nodes,
            "edges": edges,
        },
    }


# -- analyze endpoint -------------------------------------------------------


def test_preflight_returns_no_domain_issue(client):
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "likes", "nodeType": "property", "label": "likes", "iri": EX + "likes"},
        {"id": "alice", "nodeType": "individual", "label": "alice", "iri": EX + "alice"},
        {"id": "bob", "nodeType": "individual", "label": "bob", "iri": EX + "bob"},
    ]
    edges = [
        {"id": "t1", "source": "alice", "target": "Person", "iri": RDF + "type"},
        {"id": "t2", "source": "bob", "target": "Person", "iri": RDF + "type"},
        {"id": "u1", "source": "alice", "target": "bob", "iri": EX + "likes"},
    ]
    resp = client.post(
        "/besser_api/analyze-kg-for-buml-conversion",
        json=_kg_payload(nodes, edges),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["diagramType"] == "ClassDiagram"
    assert body["issueCount"] >= 1
    codes = {i["code"] for i in body["issues"]}
    assert "PROPERTY_NO_DOMAIN" in codes
    # Each issue carries a recommended_action and a skip_action.
    no_domain = next(i for i in body["issues"] if i["code"] == "PROPERTY_NO_DOMAIN")
    assert no_domain["recommendedAction"]["key"] == "attach_to_thing"
    assert no_domain["skipAction"]["key"] == "drop_property"
    assert isinstance(body["kgSignature"], str) and body["kgSignature"]


def test_preflight_clean_kg_returns_no_issues(client):
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "name", "nodeType": "property", "label": "name", "iri": EX + "name"},
        {"id": "xsd_str", "nodeType": "class", "label": "string", "iri": XSD + "string"},
    ]
    edges = [
        {"id": "e1", "source": "name", "target": "Person", "iri": RDFS + "domain"},
        {"id": "e2", "source": "name", "target": "xsd_str", "iri": RDFS + "range"},
    ]
    resp = client.post(
        "/besser_api/analyze-kg-for-buml-conversion",
        json=_kg_payload(nodes, edges),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["issueCount"] == 0


def test_preflight_rejects_non_kg_payload(client):
    resp = client.post(
        "/besser_api/analyze-kg-for-buml-conversion",
        json={"title": "wrong", "model": {"type": "ClassDiagram"}},
    )
    assert resp.status_code == 400


# -- /kg-to-class-diagram backward compat ----------------------------------


def test_class_diagram_without_resolutions_still_works(client):
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "name", "nodeType": "property", "label": "name", "iri": EX + "name"},
        {"id": "xsd_str", "nodeType": "class", "label": "string", "iri": XSD + "string"},
    ]
    edges = [
        {"id": "e1", "source": "name", "target": "Person", "iri": RDFS + "domain"},
        {"id": "e2", "source": "name", "target": "xsd_str", "iri": RDFS + "range"},
    ]
    resp = client.post("/besser_api/kg-to-class-diagram", json=_kg_payload(nodes, edges))
    assert resp.status_code == 200, resp.text
    assert resp.json()["diagramType"] == "ClassDiagram"


# -- v2 payload: accept / skip decisions -----------------------------------


def test_class_diagram_with_accept_decision(client):
    """The recommended action (attach_to_thing) attaches the property to a synthetic Thing."""
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "Pet", "nodeType": "class", "label": "Pet", "iri": EX + "Pet"},
        {"id": "likes", "nodeType": "property", "label": "likes", "iri": EX + "likes"},
        {"id": "alice", "nodeType": "individual", "label": "alice", "iri": EX + "alice"},
        {"id": "rex", "nodeType": "individual", "label": "rex", "iri": EX + "rex"},
    ]
    edges = [
        {"id": "t1", "source": "alice", "target": "Person", "iri": RDF + "type"},
        {"id": "t2", "source": "rex", "target": "Pet", "iri": RDF + "type"},
        {"id": "r1", "source": "likes", "target": "Pet", "iri": RDFS + "range"},
        {"id": "u1", "source": "alice", "target": "rex", "iri": EX + "likes"},
    ]
    payload = _kg_payload(nodes, edges)
    pre = client.post("/besser_api/analyze-kg-for-buml-conversion", json=payload).json()
    sig = pre["kgSignature"]
    no_domain = next(i for i in pre["issues"] if i["code"] == "PROPERTY_NO_DOMAIN")

    payload2 = {
        **payload,
        "kgSignature": sig,
        "resolutions": [
            {"issueId": no_domain["id"], "decision": "accept"},
        ],
    }
    resp = client.post("/besser_api/kg-to-class-diagram", json=payload2)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Thing class should be present after attach_to_thing.
    elements = body["model"]["elements"]
    class_names = {e["name"] for e in elements.values() if e.get("type") == "Class"}
    assert "Thing" in class_names


def test_class_diagram_with_skip_decision(client):
    """Skip drops the property entirely."""
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "Pet", "nodeType": "class", "label": "Pet", "iri": EX + "Pet"},
        {"id": "likes", "nodeType": "property", "label": "likes", "iri": EX + "likes"},
        {"id": "alice", "nodeType": "individual", "label": "alice", "iri": EX + "alice"},
        {"id": "rex", "nodeType": "individual", "label": "rex", "iri": EX + "rex"},
    ]
    edges = [
        {"id": "t1", "source": "alice", "target": "Person", "iri": RDF + "type"},
        {"id": "t2", "source": "rex", "target": "Pet", "iri": RDF + "type"},
        {"id": "r1", "source": "likes", "target": "Pet", "iri": RDFS + "range"},
        {"id": "u1", "source": "alice", "target": "rex", "iri": EX + "likes"},
    ]
    payload = _kg_payload(nodes, edges)
    pre = client.post("/besser_api/analyze-kg-for-buml-conversion", json=payload).json()
    sig = pre["kgSignature"]
    no_domain = next(i for i in pre["issues"] if i["code"] == "PROPERTY_NO_DOMAIN")

    payload2 = {
        **payload,
        "kgSignature": sig,
        "resolutions": [
            {"issueId": no_domain["id"], "decision": "skip"},
        ],
    }
    resp = client.post("/besser_api/kg-to-class-diagram", json=payload2)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    rel_types = {r.get("name") for r in body["model"]["relationships"].values()}
    assert "likes" not in rel_types  # property dropped


# -- Stale signature -------------------------------------------------------


def test_stale_kg_signature_returns_400(client):
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "name", "nodeType": "property", "label": "name", "iri": EX + "name"},
        {"id": "xsd_str", "nodeType": "class", "label": "string", "iri": XSD + "string"},
    ]
    edges = [
        {"id": "e1", "source": "name", "target": "Person", "iri": RDFS + "domain"},
        {"id": "e2", "source": "name", "target": "xsd_str", "iri": RDFS + "range"},
    ]
    payload = {**_kg_payload(nodes, edges), "kgSignature": "deadbeef00000000"}
    resp = client.post("/besser_api/kg-to-class-diagram", json=payload)
    assert resp.status_code == 400


# -- v1 backward compat -----------------------------------------------------


def test_v1_resolution_payload_still_works(client):
    """v1 payload {issueId, choice, parameters} routes through the legacy applier."""
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "Pet", "nodeType": "class", "label": "Pet", "iri": EX + "Pet"},
        {"id": "likes", "nodeType": "property", "label": "likes", "iri": EX + "likes"},
    ]
    edges = []
    payload = {
        **_kg_payload(nodes, edges),
        "resolutions": [
            {
                "issueId": "any",
                "choice": "assign_domain",
                "parameters": {"property_iri": EX + "likes", "class_id": "Person"},
            }
        ],
    }
    resp = client.post("/besser_api/kg-to-class-diagram", json=payload)
    assert resp.status_code == 200


def test_unknown_resolution_choice_returns_400(client):
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
    ]
    payload = {
        **_kg_payload(nodes, []),
        "resolutions": [{"issueId": "x", "choice": "totally_made_up"}],
    }
    resp = client.post("/besser_api/kg-to-class-diagram", json=payload)
    assert resp.status_code == 400


def test_unknown_decision_string_returns_400(client):
    nodes = [
        {"id": "Person", "nodeType": "class", "label": "Person", "iri": EX + "Person"},
        {"id": "likes", "nodeType": "property", "label": "likes", "iri": EX + "likes"},
        {"id": "alice", "nodeType": "individual", "label": "alice", "iri": EX + "alice"},
    ]
    edges = [
        {"id": "t1", "source": "alice", "target": "Person", "iri": RDF + "type"},
        {"id": "u1", "source": "alice", "target": "alice", "iri": EX + "likes"},
    ]
    payload = _kg_payload(nodes, edges)
    pre = client.post("/besser_api/analyze-kg-for-buml-conversion", json=payload).json()
    issue_id = pre["issues"][0]["id"]
    payload2 = {
        **payload,
        "kgSignature": pre["kgSignature"],
        "resolutions": [{"issueId": issue_id, "decision": "totally_made_up"}],
    }
    resp = client.post("/besser_api/kg-to-class-diagram", json=payload2)
    assert resp.status_code == 400
