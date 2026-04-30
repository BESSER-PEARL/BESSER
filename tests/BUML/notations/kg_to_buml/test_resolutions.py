"""Tests for the KG → BUML resolutions applier and its end-to-end use
through ``kg_to_class_diagram(..., resolutions=...)``."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from besser.BUML.notations.kg_to_buml import (
    KGResolution,
    ResolutionError,
    apply_resolutions,
    kg_to_class_diagram,
)
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


def _write_ttl(tmp_path: Path, content: str) -> str:
    p = tmp_path / "ontology.ttl"
    p.write_text(content.strip(), encoding="utf-8")
    return str(p)


# --- assign_domain ---------------------------------------------------------


def test_assign_domain_attaches_property_to_chosen_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :likes  a owl:ObjectProperty .
    :alice a :Person ; :likes :bob .
    :bob   a :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    res = KGResolution(
        issue_id="i1",
        choice="assign_domain",
        parameters={"property_iri": "http://ex.org/likes", "class_id": "http://ex.org/Person"},
    )
    new_kg = apply_resolutions(kg, [res])
    domain_targets = [
        e.target.id for e in new_kg.edges
        if e.iri == "http://www.w3.org/2000/01/rdf-schema#domain"
        and e.source.id == "http://ex.org/likes"
    ]
    assert "http://ex.org/Person" in domain_targets


# --- attach_to_thing -------------------------------------------------------


def test_attach_to_thing_creates_synthetic_class(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :likes  a owl:ObjectProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    res = KGResolution(
        issue_id="i1",
        choice="attach_to_thing",
        parameters={"property_iri": "http://ex.org/likes"},
    )
    new_kg = apply_resolutions(kg, [res])
    # A Thing class should now exist.
    things = [n for n in new_kg.nodes if getattr(n, "label", "") == "Thing"]
    assert things


# --- drop_property ---------------------------------------------------------


def test_drop_property_removes_node_and_incident_edges(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :likes  a owl:ObjectProperty ; rdfs:domain :Person .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    res = KGResolution(
        issue_id="i1",
        choice="drop_property",
        parameters={"property_iri": "http://ex.org/likes"},
    )
    new_kg = apply_resolutions(kg, [res])
    assert all(n.id != "http://ex.org/likes" for n in new_kg.nodes)
    assert all(
        e.source.id != "http://ex.org/likes" and e.target.id != "http://ex.org/likes"
        for e in new_kg.edges
    )


# --- pick_domain -----------------------------------------------------------


def test_pick_domain_drops_other_domain_edges(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person   a owl:Class .
    :Building a owl:Class .
    :address a owl:DatatypeProperty ; rdfs:domain :Person , :Building .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    res = KGResolution(
        issue_id="i1",
        choice="pick_domain",
        parameters={"property_iri": "http://ex.org/address", "class_id": "http://ex.org/Person"},
    )
    new_kg = apply_resolutions(kg, [res])
    domain_targets = {
        e.target.id for e in new_kg.edges
        if e.iri == "http://www.w3.org/2000/01/rdf-schema#domain"
        and e.source.id == "http://ex.org/address"
    }
    assert domain_targets == {"http://ex.org/Person"}


# --- attach_to_class (restriction) ----------------------------------------


def test_attach_to_class_links_restriction_to_owner(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    :Person a owl:Class .
    :ssn a owl:DatatypeProperty .
    [] a owl:Restriction ;
       owl:onProperty :ssn ;
       owl:cardinality "1"^^xsd:nonNegativeInteger .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    blank_id = next(n.id for n in kg.nodes if "Restriction" in (n.metadata.get("kind", "") or "") or n.metadata.get("kind") == "restriction")
    res = KGResolution(
        issue_id="i1",
        choice="attach_to_class",
        parameters={"blank_id": blank_id, "class_id": "http://ex.org/Person"},
    )
    new_kg = apply_resolutions(kg, [res])
    # Now the blank is a subClassOf-target of Person.
    has_link = any(
        e.iri == "http://www.w3.org/2000/01/rdf-schema#subClassOf"
        and e.source.id == "http://ex.org/Person"
        and e.target.id == blank_id
        for e in new_kg.edges
    )
    assert has_link


# --- end-to-end through kg_to_class_diagram ------------------------------


def test_resolutions_thread_through_class_diagram(tmp_path: Path):
    """Calling kg_to_class_diagram with resolutions=[assign_domain] must
    produce a Person class that owns the 'likes' association.

    The TTL declares full domain *and* range so ``:likes`` becomes a
    real BinaryAssociation rather than a fallback string attribute, and
    we check that the resolution makes the difference (default behaviour
    falls back to a synthetic Thing class)."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :Pet    a owl:Class .
    :likes  a owl:ObjectProperty ; rdfs:range :Pet .
    :alice a :Person ; :likes :rex .
    :rex   a :Pet .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    # Without resolutions, "likes" has no domain → attached to synthetic Thing.
    result_default = kg_to_class_diagram(kg)
    class_names = {c.name for c in result_default.domain_model.types}
    assert "Thing" in class_names

    # With an assign_domain resolution, "likes" attaches to Person and
    # becomes a BinaryAssociation between Person and Pet.
    result_resolved = kg_to_class_diagram(
        kg,
        resolutions=[KGResolution(
            issue_id="i1",
            choice="assign_domain",
            parameters={"property_iri": "http://ex.org/likes", "class_id": "http://ex.org/Person"},
        )],
    )
    assoc_names = {a.name for a in result_resolved.domain_model.associations}
    assert "likes" in assoc_names
    # Person and Pet should be the association's ends.
    likes = next(a for a in result_resolved.domain_model.associations if a.name == "likes")
    end_types = {e.type.name for e in likes.ends}
    assert end_types == {"Person", "Pet"}
    # No synthetic Thing this time.
    assert "Thing" not in {c.name for c in result_resolved.domain_model.types}


# --- input is not mutated --------------------------------------------------


def test_apply_resolutions_does_not_mutate_input_kg(tmp_path: Path):
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    :likes  a owl:ObjectProperty .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    nodes_before = {n.id for n in kg.nodes}
    edges_before = {e.id for e in kg.edges}

    res = KGResolution(
        issue_id="i1",
        choice="drop_property",
        parameters={"property_iri": "http://ex.org/likes"},
    )
    apply_resolutions(kg, [res])

    assert {n.id for n in kg.nodes} == nodes_before
    assert {e.id for e in kg.edges} == edges_before


# --- error paths -----------------------------------------------------------


def test_unknown_resolution_choice_raises():
    from besser.BUML.metamodel.kg import KnowledgeGraph
    with pytest.raises(ResolutionError):
        apply_resolutions(KnowledgeGraph(name="x"), [KGResolution(issue_id="i1", choice="bogus")])


def test_assign_domain_missing_class_id_raises():
    from besser.BUML.metamodel.kg import KnowledgeGraph
    with pytest.raises(ResolutionError):
        apply_resolutions(
            KnowledgeGraph(name="x"),
            [KGResolution(issue_id="i1", choice="assign_domain", parameters={"property_iri": "x"})],
        )


# --- empty / no-op ---------------------------------------------------------


def test_empty_resolutions_returns_same_kg_object(tmp_path: Path):
    """When resolutions is empty/None, the converter must not deep-copy."""
    ttl = """
    @prefix : <http://ex.org/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person a owl:Class .
    """
    kg = owl_file_to_knowledge_graph(_write_ttl(tmp_path, ttl))
    same = apply_resolutions(kg, [])
    assert same is kg
