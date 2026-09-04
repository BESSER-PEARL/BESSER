"""Tests for the LLM-driven KG cleanup analyzer.

The analyzer builds a snapshot of the KG, calls GPT-4o, and parses
suggestions back into KGIssue objects. Tests inject a fake HTTP
transport via ``_http_post=`` so no real network call is made.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGProperty,
    KnowledgeGraph,
)
from besser.BUML.notations.kg_to_buml.llm_cleanup import (
    LLM_ALLOWED_KEYS,
    MAX_LLM_SUGGESTIONS,
    MAX_ORPHAN_BATCH,
    ORPHAN_ALLOWED_KEYS,
    SAMPLE_INDIVIDUALS,
    SNAPSHOT_FULL_THRESHOLD,
    analyze_kg_with_description,
    build_kg_snapshot,
    classify_orphan_nodes_with_llm,
    parse_llm_suggestions,
)


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
EX = "http://example.org/"


def _small_kg() -> KnowledgeGraph:
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    car = KGClass(id="Car", label="Car", iri=EX + "Car")
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    bob = KGIndividual(id="bob", label="Bob", iri=EX + "bob")
    drives = KGProperty(id="drives", label="drives", iri=EX + "drives")
    blank = KGBlank(id="_:b1", metadata={"kind": "restriction"})
    lit = KGLiteral(id="lit-1", value="42", datatype="http://www.w3.org/2001/XMLSchema#integer")
    kg = KnowledgeGraph(name="TestKG", nodes={person, car, alice, bob, drives, blank, lit})
    kg.add_edge(KGEdge(id="t1", source=alice, target=person, label="type", iri=RDF_TYPE))
    kg.add_edge(KGEdge(id="t2", source=bob, target=person, label="type", iri=RDF_TYPE))
    kg.add_edge(KGEdge(id="u1", source=alice, target=bob, label="drives", iri=EX + "drives"))
    return kg


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


def _llm_completion(content: Any) -> Dict[str, Any]:
    body = json.dumps(content) if not isinstance(content, str) else content
    return {"choices": [{"message": {"content": body}}]}


# --------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------


def test_build_kg_snapshot_full_for_small_kg():
    kg = _small_kg()
    snapshot = build_kg_snapshot(kg)
    assert snapshot["mode"] == "full"
    assert {c["id"] for c in snapshot["classes"]} == {"Person", "Car"}
    assert {p["id"] for p in snapshot["properties"]} == {"drives"}
    assert {i["id"] for i in snapshot["individuals"]} == {"alice", "bob"}
    assert {e["id"] for e in snapshot["edges"]} == {"t1", "t2", "u1"}
    # rdf:type lookup populates 'types' on individuals.
    alice = next(i for i in snapshot["individuals"] if i["id"] == "alice")
    assert EX + "Person" in alice["types"]
    # Blanks and literals included verbatim.
    assert any(b["id"] == "_:b1" for b in snapshot["blanks"])
    assert any(l["id"] == "lit-1" for l in snapshot["literals"])


def test_build_kg_snapshot_summary_for_large_kg():
    kg = KnowledgeGraph(name="Big")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    kg.add_node(person)
    # Push past the threshold with individuals.
    for i in range(SNAPSHOT_FULL_THRESHOLD + 5):
        ind = KGIndividual(id=f"i{i}", label=f"i{i}", iri=EX + f"i{i}")
        kg.add_node(ind)
    snapshot = build_kg_snapshot(kg)
    assert snapshot["mode"] == "summary"
    assert "individuals" not in snapshot
    assert "individuals_sample" in snapshot
    assert len(snapshot["individuals_sample"]) == SAMPLE_INDIVIDUALS
    assert snapshot["individual_count"] == SNAPSHOT_FULL_THRESHOLD + 5
    assert "edges_by_predicate" in snapshot
    assert "edges" not in snapshot


# --------------------------------------------------------------------------
# parse_llm_suggestions
# --------------------------------------------------------------------------


def _suggestion(
    *,
    key: str = "drop_class",
    parameters: Dict[str, Any] | None = None,
    affected_node_ids: List[str] | None = None,
    code: str = "LLM_DROP_CLASS",
    confidence: float = 0.9,
) -> Dict[str, Any]:
    return {
        "code": code,
        "description": "test",
        "affected_node_ids": affected_node_ids or [],
        "affected_edge_ids": [],
        "confidence": confidence,
        "recommended_action": {
            "key": key,
            "parameters": parameters or {},
            "label": "test",
        },
    }


def test_parse_llm_suggestions_drops_unknown_node_ids():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(parameters={"node_id": "Person"}),
            _suggestion(parameters={"node_id": "DoesNotExist"}),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == 1
    assert issues[0].recommended_action.parameters["node_id"] == "Person"


def test_parse_llm_suggestions_drops_disallowed_keys():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(key="attach_to_thing", parameters={"property_iri": EX + "drives"}),
            _suggestion(parameters={"node_id": "Person"}),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == 1
    assert issues[0].recommended_action.key == "drop_class"


def test_parse_llm_suggestions_drops_duplicates():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(parameters={"node_id": "Person"}),
            _suggestion(parameters={"node_id": "Person"}),
            _suggestion(parameters={"node_id": "Car"}),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == 2
    assert {i.recommended_action.parameters["node_id"] for i in issues} == {"Person", "Car"}


def test_parse_llm_suggestions_caps_count():
    kg = KnowledgeGraph(name="Big")
    for i in range(MAX_LLM_SUGGESTIONS + 25):
        kg.add_node(KGClass(id=f"C{i}", label=f"C{i}", iri=EX + f"C{i}"))
    raw = json.dumps({
        "suggestions": [
            _suggestion(parameters={"node_id": f"C{i}"}, confidence=0.9 - i * 0.001)
            for i in range(MAX_LLM_SUGGESTIONS + 25)
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == MAX_LLM_SUGGESTIONS


def test_parse_llm_suggestions_handles_markdown_wrapped_json():
    kg = _small_kg()
    raw = "```json\n" + json.dumps({"suggestions": [_suggestion(parameters={"node_id": "Person"})]}) + "\n```"
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == 1


def test_parse_llm_suggestions_rejects_invalid_target_kind():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="reclassify_node",
                parameters={"node_id": "Person", "target_kind": "garbage"},
            ),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert issues == []


# --------------------------------------------------------------------------
# type_individual_as_class
# --------------------------------------------------------------------------


def test_parse_llm_suggestions_accepts_type_individual_as_class():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="type_individual_as_class",
                parameters={"node_id": "alice", "class_id": "Person"},
                affected_node_ids=["alice"],
                code="LLM_TYPE_INDIVIDUAL",
            ),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.recommended_action.key == "type_individual_as_class"
    assert issue.recommended_action.parameters == {"node_id": "alice", "class_id": "Person"}
    # The target class is surfaced in affected_node_ids for UI clarity.
    assert "Person" in issue.affected_node_ids
    assert "alice" in issue.affected_node_ids


def test_parse_llm_suggestions_rejects_type_when_target_is_not_a_class():
    kg = _small_kg()
    # class_id points at an individual, not a class.
    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="type_individual_as_class",
                parameters={"node_id": "alice", "class_id": "bob"},
                affected_node_ids=["alice"],
            ),
        ]
    })
    assert parse_llm_suggestions(raw, kg) == []


def test_parse_llm_suggestions_rejects_type_when_node_is_not_individual():
    kg = _small_kg()
    # node_id points at a class.
    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="type_individual_as_class",
                parameters={"node_id": "Person", "class_id": "Car"},
                affected_node_ids=["Person"],
            ),
        ]
    })
    assert parse_llm_suggestions(raw, kg) == []


def test_parse_llm_suggestions_rejects_type_with_unknown_class_id():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="type_individual_as_class",
                parameters={"node_id": "alice", "class_id": "Ghost"},
                affected_node_ids=["alice"],
            ),
        ]
    })
    assert parse_llm_suggestions(raw, kg) == []


def test_parse_llm_suggestions_rejects_drop_class_on_individual():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(parameters={"node_id": "alice"}, affected_node_ids=["alice"]),
        ]
    })
    # drop_class on an individual would crash the dispatcher; we drop at parse time.
    assert parse_llm_suggestions(raw, kg) == []


def test_parse_llm_suggestions_rejects_drop_individual_on_class():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="drop_individual",
                parameters={"node_id": "Person"},
                affected_node_ids=["Person"],
            ),
        ]
    })
    assert parse_llm_suggestions(raw, kg) == []


# --------------------------------------------------------------------------
# Ordering: classes/properties → individuals → literals → blanks
# --------------------------------------------------------------------------


def test_parse_llm_suggestions_orders_class_fixes_before_individuals():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            # Individual fix (priority 1) but very high confidence.
            _suggestion(
                key="type_individual_as_class",
                parameters={"node_id": "alice", "class_id": "Person"},
                affected_node_ids=["alice"],
                code="LLM_TYPE_INDIVIDUAL",
                confidence=0.99,
            ),
            # Class fix (priority 0) with lower confidence.
            _suggestion(
                key="drop_class",
                parameters={"node_id": "Car"},
                affected_node_ids=["Car"],
                code="LLM_DROP_CLASS",
                confidence=0.5,
            ),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == 2
    # Class fix comes first regardless of confidence.
    assert issues[0].recommended_action.key == "drop_class"
    assert issues[1].recommended_action.key == "type_individual_as_class"


def test_parse_llm_suggestions_orders_blanks_last():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    blank = KGBlank(id="_:b1", metadata={})
    lit = KGLiteral(id="lit1", value="42", datatype=None)
    kg = KnowledgeGraph(name="K", nodes={person, alice, blank, lit})

    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="drop_node",
                parameters={"node_id": "_:b1"},
                affected_node_ids=["_:b1"],
                code="LLM_DROP_NODE",
                confidence=0.95,
            ),
            _suggestion(
                key="drop_node",
                parameters={"node_id": "lit1"},
                affected_node_ids=["lit1"],
                code="LLM_DROP_NODE_LIT",
                confidence=0.95,
            ),
            _suggestion(
                key="drop_individual",
                parameters={"node_id": "alice"},
                affected_node_ids=["alice"],
                code="LLM_DROP_INDIVIDUAL",
                confidence=0.95,
            ),
            _suggestion(
                key="drop_class",
                parameters={"node_id": "Person"},
                affected_node_ids=["Person"],
                code="LLM_DROP_CLASS",
                confidence=0.95,
            ),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    keys = [i.recommended_action.key for i in issues]
    nodes = [i.recommended_action.parameters.get("node_id") for i in issues]
    # Class first, then individual, then literal (via drop_node), then blank (via drop_node).
    assert keys[0] == "drop_class" and nodes[0] == "Person"
    assert keys[1] == "drop_individual" and nodes[1] == "alice"
    assert nodes[2] == "lit1"
    assert nodes[3] == "_:b1"


def test_parse_llm_suggestions_accepts_promote_individual():
    kg = _small_kg()
    raw = json.dumps({
        "suggestions": [
            _suggestion(
                key="promote_individual_to_class",
                parameters={"node_id": "alice"},
                code="LLM_PROMOTE_INDIVIDUAL",
            ),
        ]
    })
    issues = parse_llm_suggestions(raw, kg)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.recommended_action.key == "promote_individual_to_class"
    assert issue.skip_action.key == "noop"
    assert issue.affected_node_ids == ["alice"]


# --------------------------------------------------------------------------
# analyze_kg_with_description
# --------------------------------------------------------------------------


def test_analyze_kg_with_description_mocks_openai():
    kg = _small_kg()
    suggestion = _suggestion(parameters={"node_id": "Car"})

    captured: Dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse(_llm_completion({"suggestions": [suggestion]}))

    report = analyze_kg_with_description(
        kg, "Build a system about people, not cars.", "sk-test", _http_post=fake_post,
    )

    assert report.diagram_type == "LLMCleanup"
    assert report.issue_count == 1
    assert report.kg_signature  # populated
    assert report.issues[0].recommended_action.parameters["node_id"] == "Car"

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["temperature"] == 0.0


def test_analyze_kg_with_description_raises_on_http_error():
    kg = _small_kg()

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResponse({"error": {"message": "Boom"}}, status_code=500)

    with pytest.raises(RuntimeError, match="OpenAI API call failed"):
        analyze_kg_with_description(kg, "desc", "sk-test", _http_post=fake_post)


def test_analyze_kg_with_description_validates_inputs():
    kg = _small_kg()
    with pytest.raises(ValueError):
        analyze_kg_with_description(kg, "", "sk-test", _http_post=lambda *a, **kw: None)
    with pytest.raises(ValueError):
        analyze_kg_with_description(kg, "desc", "", _http_post=lambda *a, **kw: None)


def test_analyze_kg_with_description_drops_disallowed_in_e2e():
    """Even if the LLM hallucinates a forbidden action key, the analyzer drops it."""
    kg = _small_kg()
    # Mix of an allowed and a forbidden key — only the allowed one survives.
    payload = {
        "suggestions": [
            _suggestion(key="attach_to_thing", parameters={"property_iri": EX + "drives"}),
            _suggestion(parameters={"node_id": "Person"}),
        ]
    }

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResponse(_llm_completion(payload))

    report = analyze_kg_with_description(kg, "desc", "sk-test", _http_post=fake_post)
    assert report.issue_count == 1
    assert report.issues[0].recommended_action.key == "drop_class"


def test_llm_allowed_keys_match_handler_dispatch():
    """All keys the LLM may emit must be registered in the resolutions dispatch table."""
    from besser.BUML.notations.kg_to_buml.resolutions import _HANDLERS
    missing = LLM_ALLOWED_KEYS - set(_HANDLERS)
    assert not missing, f"LLM action keys without handlers: {missing}"


def test_orphan_allowed_keys_match_handler_dispatch():
    """Orphan-classification action keys must also be registered."""
    from besser.BUML.notations.kg_to_buml.resolutions import _HANDLERS
    missing = ORPHAN_ALLOWED_KEYS - set(_HANDLERS)
    assert not missing, f"Orphan action keys without handlers: {missing}"


# --------------------------------------------------------------------------
# classify_orphan_nodes_with_llm
# --------------------------------------------------------------------------


def _kg_with_orphans_and_class() -> KnowledgeGraph:
    """Mini KG with one anchored class plus three unrelated orphans of
    different kinds (individual / individual / literal)."""
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    ghost1 = KGIndividual(id="ghost1", label="ghost1", iri=EX + "ghost1")
    ghost2 = KGIndividual(id="ghost2", label="ghost2", iri=EX + "ghost2")
    lit = KGLiteral(
        id="lit-orph",
        value="hello",
        datatype="http://www.w3.org/2001/XMLSchema#string",
    )
    return KnowledgeGraph(name="K", nodes={person, ghost1, ghost2, lit})


def test_classify_orphan_nodes_with_llm_happy_path():
    kg = _kg_with_orphans_and_class()
    captured: List[Dict[str, Any]] = []
    suggestions = [
        _suggestion(
            key="type_individual_as_class",
            parameters={"node_id": "ghost1", "class_id": "Person"},
            affected_node_ids=["ghost1"],
            code="LLM_TYPE_INDIVIDUAL",
        ),
        _suggestion(
            key="drop_node",
            parameters={"node_id": "lit-orph"},
            affected_node_ids=["lit-orph"],
            code="LLM_DROP_NODE",
        ),
    ]

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(_llm_completion({"suggestions": suggestions}))

    report = classify_orphan_nodes_with_llm(
        kg,
        ["ghost1", "ghost2", "lit-orph"],
        "A people-management system.",
        "sk-test",
        _http_post=fake_post,
    )

    assert report.diagram_type == "OrphanClassification"
    # ghost1 → typed; lit-orph → dropped; ghost2 → synthetic drop_node fallback.
    assert report.issue_count == 3
    by_target = {
        i.recommended_action.parameters["node_id"]: i for i in report.issues
    }
    assert by_target["ghost1"].recommended_action.key == "type_individual_as_class"
    assert by_target["ghost1"].recommended_action.parameters["class_id"] == "Person"
    assert by_target["lit-orph"].recommended_action.key == "drop_node"
    # ghost2 was omitted by the LLM → synthetic drop_node fallback.
    assert by_target["ghost2"].recommended_action.key == "drop_node"
    assert by_target["ghost2"].code == "LLM_DROP_NODE"

    # One HTTP call (3 nodes < MAX_ORPHAN_BATCH).
    assert len(captured) == 1


def test_classify_orphan_nodes_with_llm_chunks_large_batches():
    """If node_ids exceeds MAX_ORPHAN_BATCH, the function makes one HTTP call
    per chunk and merges the results."""
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    nodes = {person}
    node_ids: List[str] = []
    for i in range(MAX_ORPHAN_BATCH * 2 + 3):
        nid = f"orph_{i}"
        nodes.add(KGIndividual(id=nid, label=nid, iri=EX + nid))
        node_ids.append(nid)
    kg = KnowledgeGraph(name="K", nodes=nodes)

    call_count = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        call_count["n"] += 1
        # Return an empty suggestions list so every orphan gets a synthetic drop.
        return _FakeResponse(_llm_completion({"suggestions": []}))

    report = classify_orphan_nodes_with_llm(
        kg, node_ids, "Some system.", "sk-test", _http_post=fake_post,
    )

    # 3 chunks: 40 + 40 + 3.
    assert call_count["n"] == 3
    assert report.issue_count == len(node_ids)
    assert all(i.recommended_action.key == "drop_node" for i in report.issues)


def test_classify_orphan_nodes_with_llm_drops_disallowed_action():
    """Even if the LLM emits a forbidden action key (e.g. drop_class), it must
    be filtered out of the orphan-classification result."""
    kg = _kg_with_orphans_and_class()
    payload = {
        "suggestions": [
            _suggestion(
                key="drop_class",  # not in ORPHAN_ALLOWED_KEYS
                parameters={"node_id": "ghost1"},
                affected_node_ids=["ghost1"],
            ),
        ]
    }

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResponse(_llm_completion(payload))

    report = classify_orphan_nodes_with_llm(
        kg, ["ghost1"], "desc", "sk-test", _http_post=fake_post,
    )
    # The disallowed key gets dropped → ghost1 falls back to a synthetic drop_node.
    assert report.issue_count == 1
    assert report.issues[0].recommended_action.key == "drop_node"


def test_classify_orphan_nodes_with_llm_validates_inputs():
    kg = _kg_with_orphans_and_class()
    with pytest.raises(ValueError):
        classify_orphan_nodes_with_llm(kg, ["ghost1"], "", "sk-test")
    with pytest.raises(ValueError):
        classify_orphan_nodes_with_llm(kg, ["ghost1"], "desc", "")
    with pytest.raises(TypeError):
        classify_orphan_nodes_with_llm(kg, "not-a-list", "desc", "sk-test")  # type: ignore[arg-type]


def test_classify_orphan_nodes_with_llm_empty_node_ids_returns_empty():
    kg = _kg_with_orphans_and_class()
    report = classify_orphan_nodes_with_llm(
        kg, [], "desc", "sk-test", _http_post=lambda *a, **kw: None,
    )
    assert report.issue_count == 0
    assert report.diagram_type == "OrphanClassification"


def test_classify_orphan_nodes_with_llm_skips_unknown_ids():
    """Unknown node ids in the input are silently skipped."""
    kg = _kg_with_orphans_and_class()

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResponse(_llm_completion({"suggestions": []}))

    report = classify_orphan_nodes_with_llm(
        kg,
        ["ghost1", "no_such_node"],
        "desc",
        "sk-test",
        _http_post=fake_post,
    )
    # Only the existing node produces an issue (synthetic drop fallback).
    assert report.issue_count == 1
    assert report.issues[0].affected_node_ids == ["ghost1"]
