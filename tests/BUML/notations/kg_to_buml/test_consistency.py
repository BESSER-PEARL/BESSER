"""Tests for the OWL2 + SHACL consistency checker.

Each test builds a tiny ontology (via the OWL importer so the resulting
KG carries the same metadata.constraintSpecs structure the real editor
produces) and asserts the issues returned by
:func:`check_kg_consistency`.
"""

from __future__ import annotations

import tempfile
import os

import pytest

from besser.BUML.notations.kg_to_buml.consistency import check_kg_consistency
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kg(ttl: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ttl", delete=False) as f:
        f.write(ttl)
        path = f.name
    try:
        return owl_file_to_knowledge_graph(path)
    finally:
        os.unlink(path)


def _has_issue(report, *, code=None, severity=None, focus=None) -> bool:
    for i in report.issues:
        if code is not None and i.code != code:
            continue
        if severity is not None and i.severity != severity:
            continue
        if focus is not None and focus not in i.affected_node_ids:
            continue
        return True
    return False


_PREFIX = """
@prefix : <http://example.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------


def test_min_count_violation_via_shacl():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:hasFriend a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Person .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :hasFriend ; sh:minCount 2 ] .
:alice a :Person ; :hasFriend :bob .
:bob a :Person .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="MinCountConstraintComponent", focus="http://example.org/alice")


def test_min_cardinality_violation_via_owl():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:hasName a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:Person rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty :hasName ;
    owl:minCardinality "1"^^xsd:nonNegativeInteger
] .
:alice a :Person .
""")
    report = check_kg_consistency(kg)
    # OWL minCardinality round-trips as SHACL sh:minCount via vocab="both".
    assert _has_issue(report, code="MinCountConstraintComponent", focus="http://example.org/alice")


# ---------------------------------------------------------------------------
# Datatype + literal value constraints
# ---------------------------------------------------------------------------


def test_datatype_violation():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:age a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:integer .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :age ; sh:datatype xsd:integer ] .
:alice a :Person ; :age "not-a-number"^^xsd:string .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="DatatypeConstraintComponent", focus="http://example.org/alice")


def test_pattern_violation():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:pattern "^[A-Z]" ] .
:alice a :Person ; :name "lowercase" .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="PatternConstraintComponent", focus="http://example.org/alice")


def test_min_inclusive_violation():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:age a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:integer .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :age ; sh:minInclusive 0 ] .
:alice a :Person ; :age "-5"^^xsd:integer .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="MinInclusiveConstraintComponent", focus="http://example.org/alice")


def test_max_length_violation():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:maxLength 3 ] .
:alice a :Person ; :name "Alice" .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="MaxLengthConstraintComponent", focus="http://example.org/alice")


def test_in_violation():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:color a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :color ; sh:in ("red" "green" "blue") ] .
:alice a :Person ; :color "purple" .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="InConstraintComponent", focus="http://example.org/alice")


# ---------------------------------------------------------------------------
# OWL DL: disjointWith via the SHACL shim
# ---------------------------------------------------------------------------


def test_disjoint_with_violation():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:Project a owl:Class .
:Person owl:disjointWith :Project .
:alice a :Person, :Project .
""")
    report = check_kg_consistency(kg)
    # The shim adds sh:NodeShape with sh:not on each side; both fire.
    assert _has_issue(report, code="NotConstraintComponent", focus="http://example.org/alice")


def test_disjoint_with_compliant():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:Project a owl:Class .
:Person owl:disjointWith :Project .
:alice a :Person .
:p1 a :Project .
""")
    report = check_kg_consistency(kg)
    assert not _has_issue(report, code="NotConstraintComponent")


def test_complement_of_violation():
    kg = _kg(_PREFIX + """
:Adult a owl:Class .
:Child owl:complementOf :Adult .
:alice a :Adult, :Child .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="NotConstraintComponent", focus="http://example.org/alice")


# ---------------------------------------------------------------------------
# Logical operators
# ---------------------------------------------------------------------------


def test_shacl_not_violation():
    # Nested `sh:not [sh:pattern …]` constrains each VALUE of the property
    # (not a path), so the inner shape works without `sh:path`.
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:nickname a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :nickname ; sh:not [ sh:pattern "^[a-z]" ] ] .
:alice a :Person ; :nickname "lowercase" .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, code="NotConstraintComponent", focus="http://example.org/alice")


# ---------------------------------------------------------------------------
# Severity + deactivation
# ---------------------------------------------------------------------------


def test_severity_warning_propagates():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:hasFriend a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Person .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :hasFriend ; sh:minCount 2 ; sh:severity sh:Warning ] .
:alice a :Person .
""")
    report = check_kg_consistency(kg)
    assert _has_issue(report, severity="warning", focus="http://example.org/alice")
    assert report.severity_counts["warning"] >= 1


def test_deactivated_shape_suppressed():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:hasFriend a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Person .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :hasFriend ; sh:minCount 2 ; sh:deactivated true ] .
:alice a :Person .
""")
    report = check_kg_consistency(kg)
    assert not _has_issue(report, code="MinCountConstraintComponent")


# ---------------------------------------------------------------------------
# Compliant KG → empty issue list
# ---------------------------------------------------------------------------


def test_fully_compliant_kg_has_no_issues():
    kg = _kg(_PREFIX + """
:Person a owl:Class .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string ] .
:alice a :Person ; :name "Alice" .
""")
    report = check_kg_consistency(kg)
    assert report.issue_count == 0, [i.message for i in report.issues]
    assert report.severity_counts == {"info": 0, "warning": 0, "violation": 0}


# ---------------------------------------------------------------------------
# Report metadata
# ---------------------------------------------------------------------------


def test_report_carries_kg_signature_and_inference():
    kg = _kg(_PREFIX + ":alice a owl:Thing .")
    report = check_kg_consistency(kg)
    assert isinstance(report.kg_signature, str) and report.kg_signature
    assert report.inference_used == "owlrl"


# ---------------------------------------------------------------------------
# Security: SHACL-AF must never execute what the KG author wrote
# ---------------------------------------------------------------------------


# A SPARQLRule whose CONSTRUCT reaches for the cloud metadata endpoint. If the
# validator runs SHACL-AF rules, rdflib resolves the SERVICE clause with a
# plain urlopen() against no allowlist, and a KG uploaded to a shared editor
# host becomes a request forgery primitive.
_SSRF_TTL = _PREFIX + """
:Person a owl:Class, sh:NodeShape ;
    sh:targetClass :Person ;
    sh:rule [
        a sh:SPARQLRule ;
        sh:construct \"\"\"
            PREFIX : <http://example.org/>
            CONSTRUCT { ?this :leaked ?o }
            WHERE {
                SERVICE <http://169.254.169.254/latest/meta-data/iam/security-credentials/> {
                    ?s ?p ?o
                }
            }
        \"\"\" ;
    ] .
:alice a :Person .
"""


def test_sparql_rule_in_kg_never_triggers_a_network_fetch(monkeypatch):
    """The SSRF regression test: no urlopen, no matter what the KG carries."""
    calls = []

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        calls.append(args[0] if args else kwargs)
        raise AssertionError(
            "consistency checking resolved a URL chosen by the KG author: "
            f"{args[0] if args else kwargs!r}"
        )

    # rdflib's SERVICE evaluator imports urlopen into its own module namespace.
    monkeypatch.setattr(
        "rdflib.plugins.sparql.evaluate.urlopen", _explode, raising=True
    )

    report = check_kg_consistency(_kg(_SSRF_TTL))

    assert calls == []
    # The checker still returns a usable report rather than failing closed.
    assert report.kg_signature


def test_executable_shacl_is_stripped_before_validation():
    """The rule and its SPARQL body are gone from the graph that pyshacl sees."""
    from besser.BUML.notations.kg_to_buml.consistency import (
        SH,
        _strip_executable_shacl,
    )
    from besser.utilities.kg_to_owl import knowledge_graph_to_rdf

    graph = knowledge_graph_to_rdf(_kg(_SSRF_TTL), vocab="both")
    removed = _strip_executable_shacl(graph)

    assert removed > 0
    assert not list(graph.triples((None, SH.rule, None)))
    assert not list(graph.triples((None, SH.construct, None)))
    assert "169.254.169.254" not in graph.serialize(format="turtle")


def test_stripping_leaves_a_normal_besser_kg_untouched():
    """Nothing BESSER's own constraint model emits is affected by the strip."""
    from besser.BUML.notations.kg_to_buml.consistency import _strip_executable_shacl
    from besser.utilities.kg_to_owl import knowledge_graph_to_rdf

    kg = _kg(_PREFIX + """
:Person a owl:Class .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string ] .
:alice a :Person ; :name "Alice" .
""")
    graph = knowledge_graph_to_rdf(kg, vocab="both")
    before = len(graph)

    assert _strip_executable_shacl(graph) == 0
    assert len(graph) == before


# ---------------------------------------------------------------------------
# Determinism and ordering of the report
# ---------------------------------------------------------------------------


# owl:disjointWith deliberately: the shapes for it are minted by
# _augment_with_owl_shim as fresh BNodes, which is the path whose random
# labels used to end up in issue ids and in constraint_node_id. A fixture
# whose source shape resolves to a real node would not exercise it.
_VIOLATING_TTL = _PREFIX + """
:Person a owl:Class .
:Project a owl:Class .
:Person owl:disjointWith :Project .
:alice a :Person, :Project .
:bob a :Person, :Project .
:hasName a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:Person rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty :hasName ;
    owl:minCardinality "1"^^xsd:nonNegativeInteger
] .
"""


def test_issue_ids_are_stable_across_calls():
    """Ids were hashed from BNodes the validator minted for the OWL shim.

    Those labels are fresh per call, so every run returned different ids for
    the same findings in a different order -- nothing could be cached, diffed
    or referenced from a previous response.
    """
    kg = _kg(_VIOLATING_TTL)

    first = check_kg_consistency(kg)
    second = check_kg_consistency(kg)

    assert first.issue_count > 0, "fixture produced no issues to compare"
    assert [i.id for i in first.issues] == [i.id for i in second.issues]


def test_no_synthetic_blank_node_ids_leak_as_constraint_node_ids():
    """`constraint_node_id` must name a KG node or nothing at all."""
    kg = _kg(_VIOLATING_TTL)
    node_ids = {n.id for n in kg.nodes}

    report = check_kg_consistency(kg)

    for issue in report.issues:
        if issue.constraint_node_id is not None:
            assert issue.constraint_node_id in node_ids, issue.constraint_node_id


def test_issues_are_ordered_most_severe_first():
    """Sorting the severity strings gave info -> violation -> warning."""
    from besser.BUML.notations.kg_to_buml.consistency import _SEVERITY_RANK

    report = check_kg_consistency(_kg(_VIOLATING_TTL))
    ranks = [_SEVERITY_RANK[i.severity] for i in report.issues]

    assert ranks == sorted(ranks)
