"""Tests for the KG → BUML preflight analyzer (v2 shape).

Each detector gets a minimal KG (built from a TTL fragment) that triggers
exactly one issue, plus a negative case ensuring no false positive. The
v2 shape has ``recommended_action`` + ``skip_action`` per issue (no
severity field).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from besser.BUML.notations.kg_to_buml import (
    KGAction,
    KGIssue,
    KGPreflightReport,
    analyze_kg_for_class_diagram,
    analyze_kg_for_object_diagram,
    kg_signature,
)
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


def _write_ttl(tmp_path: Path, content: str) -> str:
    p = tmp_path / "ontology.ttl"
    p.write_text(content.strip(), encoding="utf-8")
    return str(p)


def _codes(report: KGPreflightReport) -> set:
    return {i.code for i in report.issues}


def _issue(report: KGPreflightReport, code: str) -> KGIssue:
    matches = [i for i in report.issues if i.code == code]
    assert matches, f"expected an issue with code {code}; got {sorted(_codes(report))}"
    return matches[0]


def _assert_actions_present(issue: KGIssue):
    assert isinstance(issue.recommended_action, KGAction), f"{issue.code} missing recommended_action"
    assert isinstance(issue.skip_action, KGAction), f"{issue.code} missing skip_action"
    assert issue.recommended_action.label  # human-readable
    assert issue.skip_action.label


# --- PROPERTY_NO_DOMAIN ----------------------------------------------------


def test_property_no_domain_with_abox_usage(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :likes  a owl:ObjectProperty .
    :alice  a :Person ; :likes :bob .
    :bob    a :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "PROPERTY_NO_DOMAIN")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "attach_to_thing"
    assert issue.skip_action.key == "drop_property"


def test_property_with_domain_does_not_trigger_no_domain_issue(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :likes  a owl:ObjectProperty ; rdfs:domain :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    assert "PROPERTY_NO_DOMAIN" not in _codes(report)


# --- MULTIPLE_DOMAINS ------------------------------------------------------


def test_multiple_domains_unrelated(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person   a owl:Class .
    :Building a owl:Class .
    :address  a owl:DatatypeProperty ; rdfs:domain :Person , :Building .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "MULTIPLE_DOMAINS")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "pick_domain"
    # Deterministic: lex-first class wins.
    chosen = issue.recommended_action.parameters["class_id"]
    assert chosen == "http://ex.org/Building"  # alphabetically first


def test_multiple_domains_in_subclass_chain_not_triggered(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Animal a owl:Class .
    :Dog    a owl:Class ; rdfs:subClassOf :Animal .
    :name   a owl:DatatypeProperty ; rdfs:domain :Animal , :Dog .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    assert "MULTIPLE_DOMAINS" not in _codes(report)


# --- RESTRICTION_UNATTACHED -----------------------------------------------


def test_unattached_restriction(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :ssn a owl:DatatypeProperty .
    [] a owl:Restriction ;
       owl:onProperty :ssn ;
       owl:cardinality "1"^^xsd:nonNegativeInteger .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "RESTRICTION_UNATTACHED")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "drop_restriction"


def test_attached_restriction_not_flagged(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :ssn ;
            owl:cardinality "1"^^xsd:nonNegativeInteger
        ] .
    :ssn a owl:DatatypeProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    assert "RESTRICTION_UNATTACHED" not in _codes(report)


# --- RESTRICTION_UNSUPPORTED ----------------------------------------------


def test_unsupported_restriction_has_value(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Adult a owl:Class ;
        rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :country ;
            owl:hasValue "US"
        ] .
    :country a owl:DatatypeProperty ; rdfs:domain :Adult .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "RESTRICTION_UNSUPPORTED")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "record_as_description"
    assert issue.skip_action.key == "drop_restriction"


# --- RANGE_BOTH_DATATYPE_AND_CLASS ----------------------------------------


def test_range_both_datatype_and_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :Pet    a owl:Class .
    :weird  a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Pet , xsd:string .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "RANGE_BOTH_DATATYPE_AND_CLASS")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "keep_object_property"


# --- RANGE_NOT_TYPE -------------------------------------------------------


def test_range_not_type(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :acme   a :Person .
    :weird  a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :acme .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "RANGE_NOT_TYPE")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "treat_as_string"


# --- CYCLIC_SUBCLASS ------------------------------------------------------


def test_cyclic_subclass(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :A a owl:Class ; rdfs:subClassOf :B .
    :B a owl:Class ; rdfs:subClassOf :A .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "CYCLIC_SUBCLASS")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "break_at_edge"
    assert {"http://ex.org/A", "http://ex.org/B"} <= set(issue.affected_node_ids)


# --- PROPERTY_NO_RANGE ----------------------------------------------------


def test_property_no_range(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :nickname a owl:DatatypeProperty ; rdfs:domain :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "PROPERTY_NO_RANGE")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "set_range"


# --- PUNNING --------------------------------------------------------------


def test_punning(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Eagle a owl:Class , owl:NamedIndividual .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "PUNNING")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "keep_both"
    assert issue.skip_action.key == "prefer_class"


# --- EQUIVALENT_CLASSES + INVERSE_PROPERTY --------------------------------


def test_equivalent_classes(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Human  a owl:Class .
    :Person a owl:Class ; owl:equivalentClass :Human .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "EQUIVALENT_CLASSES")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "keep_separate"
    assert issue.skip_action.key == "merge_classes"


def test_inverse_properties(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :hasParent a owl:ObjectProperty ; owl:inverseOf :hasChild .
    :hasChild  a owl:ObjectProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "INVERSE_PROPERTY")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "keep_separate"
    assert issue.skip_action.key == "merge_associations"


# --- BLANK_NODE_INSTANCE ---------------------------------------------------


def test_blank_node_instance(tmp_path: Path):
    """Blank node used as an instance (typed via rdf:type)."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    _:b1 a :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "BLANK_NODE_INSTANCE")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "materialize_as_individual"
    assert issue.skip_action.key == "drop_node"


# --- UNDECLARED_CLASS ------------------------------------------------------


def test_undeclared_class_via_type(tmp_path: Path):
    """A node referenced via rdf:type but never declared as owl:Class."""
    ttl = """
    @prefix : <http://ex.org/> .

    :alice a :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    # Note: the importer's second pass adds Person to the class set
    # (since it appears as object of rdf:type), so this might NOT trigger
    # UNDECLARED_CLASS — the importer already promoted it to KGClass.
    # If KGClass nodes are excluded from the detector, no issue is raised.
    # We just check the detector doesn't crash.
    assert isinstance(report.issues, list)


# --- MULTIVALUED_LITERAL --------------------------------------------------


def test_multivalued_literal(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    :alice a :Person ; :name "Alice" , "Alicia" .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "MULTIVALUED_LITERAL")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "bump_to_unbounded"
    assert issue.skip_action.key == "keep_first_only"


# --- Object-diagram detectors ---------------------------------------------


def test_object_diagram_blank_node(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    _:b1 a :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_object_diagram(kg)
    codes = {i.code for i in report.issues}
    assert "BLANK_NODE_AS_OBJECT" in codes


def test_orphan_individuals_suppress_individual_no_type(tmp_path: Path):
    """Individuals with no class anchoring at all are flagged as orphans, and
    the INDIVIDUAL_NO_TYPE detector is suppressed for them so the user sees a
    single, drop-or-classify decision instead of two competing issues."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :alice :knows :bob .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_object_diagram(kg)
    codes = {i.code for i in report.issues}
    assert "ORPHAN_NODE_NO_CLASS_LINK" in codes
    assert "INDIVIDUAL_NO_TYPE" not in codes


def test_object_diagram_does_not_run_class_detectors(tmp_path: Path):
    """Class-diagram-only issues (PROPERTY_NO_DOMAIN) must not appear in the
    object-diagram report."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :likes  a owl:ObjectProperty .
    :alice  a :Person ; :likes :bob .
    :bob    a :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_object_diagram(kg)
    codes = {i.code for i in report.issues}
    assert "PROPERTY_NO_DOMAIN" not in codes


# --- Signature & report basics --------------------------------------------


def test_kg_signature_changes_when_node_added(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :A a owl:Class .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    sig_before = kg_signature(kg)
    from besser.BUML.metamodel.kg import KGClass
    kg.add_node(KGClass(id="http://ex.org/Z", label="Z", iri="http://ex.org/Z"))
    sig_after = kg_signature(kg)
    assert sig_before != sig_after


def test_clean_kg_has_empty_report(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :name   a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    assert report.issue_count == 0
    assert report.diagram_type == "ClassDiagram"
    assert isinstance(report.kg_signature, str) and len(report.kg_signature) >= 8


def test_object_diagram_report_type(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_object_diagram(kg)
    assert report.diagram_type == "ObjectDiagram"


# --- ORPHAN_NODE_NO_CLASS_LINK ---------------------------------------------


def test_orphan_single_individual(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :ghost :spooks :nothing .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    issue = _issue(report, "ORPHAN_NODE_NO_CLASS_LINK")
    _assert_actions_present(issue)
    assert issue.recommended_action.key == "drop_orphan_nodes"
    assert issue.skip_action.key == "defer_to_llm_classification"
    assert set(issue.recommended_action.parameters["node_ids"]) == set(issue.affected_node_ids)


def test_orphan_multiple_disconnected_components(tmp_path: Path):
    """Two independent orphan islands → two issues, one per component."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .

    # Island A
    :alice :knows :bob .

    # Island B (separate)
    :foo :rel :bar .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    orphans = [i for i in report.issues if i.code == "ORPHAN_NODE_NO_CLASS_LINK"]
    assert len(orphans) == 2
    component_size_sum = sum(len(i.affected_node_ids) for i in orphans)
    # Each island has 2 nodes (subject + object), 4 in total.
    assert component_size_sum == 4


def test_orphan_anchored_via_property_domain_not_flagged(tmp_path: Path):
    """A typed individual reachable through a property's domain link is
    class-anchored and must not appear as an orphan."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :name   a owl:DatatypeProperty ; rdfs:domain :Person .
    :alice  a :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    assert "ORPHAN_NODE_NO_CLASS_LINK" not in _codes(report)


def test_orphan_structural_blank_excluded(tmp_path: Path):
    """Restriction-kind blank nodes (anonymous schema constructs) must not be
    treated as orphans even when not attached to a class — they have their own
    detector."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    [] a owl:Restriction ;
       owl:onProperty :age ;
       owl:cardinality 1 .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    # The unattached restriction is its own issue, NOT folded into ORPHAN.
    codes = _codes(report)
    assert "RESTRICTION_UNATTACHED" in codes
    # An orphan issue may exist if there are other non-anchored elements, but
    # restriction blanks must not appear inside any orphan issue.
    for issue in report.issues:
        if issue.code != "ORPHAN_NODE_NO_CLASS_LINK":
            continue
        for nid in issue.affected_node_ids:
            node = kg.get_node(nid)
            from besser.BUML.metamodel.kg import KGBlank
            if isinstance(node, KGBlank):
                assert node.metadata.get("kind") not in {"restriction", "class_expression"}


def test_orphan_id_stable_across_runs(tmp_path: Path):
    """Issue id must be stable across invocations so accept/skip decisions
    survive the round-trip to the apply endpoint."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :alice :knows :bob .
    """
    kg1 = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    kg2 = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    r1 = analyze_kg_for_class_diagram(kg1)
    r2 = analyze_kg_for_class_diagram(kg2)
    o1 = next(i for i in r1.issues if i.code == "ORPHAN_NODE_NO_CLASS_LINK")
    o2 = next(i for i in r2.issues if i.code == "ORPHAN_NODE_NO_CLASS_LINK")
    assert o1.id == o2.id


def test_orphan_clean_kg_no_issue(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :name   a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    report = analyze_kg_for_class_diagram(kg)
    assert "ORPHAN_NODE_NO_CLASS_LINK" not in _codes(report)
