"""Tests for the LLM-cleanup resolution handlers.

Each handler must mutate a deep-copied KG (input untouched), keep edges
referentially consistent, and clean up axioms whose ids are no longer
in the graph.
"""

from __future__ import annotations

import copy

import pytest

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGProperty,
    KnowledgeGraph,
)
from besser.BUML.metamodel.kg.axioms import (
    EquivalentClassesAxiom,
    InversePropertiesAxiom,
)
from besser.BUML.notations.kg_to_buml import (
    DeferredOrphanClassification,
    KGAction,
    KGIssue,
    KGResolution,
    ResolutionError,
    apply_resolutions,
    dispatch_decision,
)


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
EX = "http://example.org/"


def _kg_with_two_classes() -> KnowledgeGraph:
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    car = KGClass(id="Car", label="Car", iri=EX + "Car")
    drives = KGProperty(id="drives", label="drives", iri=EX + "drives")
    kg = KnowledgeGraph(name="K", nodes={person, car, drives})
    kg.add_edge(KGEdge(id="d1", source=drives, target=person, label="domain",
                       iri="http://www.w3.org/2000/01/rdf-schema#domain"))
    kg.add_edge(KGEdge(id="d2", source=drives, target=car, label="range",
                       iri="http://www.w3.org/2000/01/rdf-schema#range"))
    return kg


def _resolve(kg: KnowledgeGraph, key: str, parameters: dict) -> KnowledgeGraph:
    return apply_resolutions(kg, [KGResolution(issue_id="x", choice=key, parameters=parameters)])


# --------------------------------------------------------------------------
# drop_class
# --------------------------------------------------------------------------


def test_drop_class_removes_node_and_incident_edges():
    kg = _kg_with_two_classes()
    out = _resolve(kg, "drop_class", {"node_id": "Car"})
    assert "Car" not in {n.id for n in out.nodes}
    # Range edge to Car is gone; domain edge to Person stays.
    assert {e.id for e in out.edges} == {"d1"}


def test_drop_class_does_not_mutate_input():
    kg = _kg_with_two_classes()
    snapshot = (
        sorted(n.id for n in kg.nodes),
        sorted(e.id for e in kg.edges),
    )
    _resolve(kg, "drop_class", {"node_id": "Car"})
    assert (sorted(n.id for n in kg.nodes), sorted(e.id for e in kg.edges)) == snapshot


def test_drop_class_rejects_wrong_kind():
    kg = _kg_with_two_classes()
    with pytest.raises(ResolutionError):
        _resolve(kg, "drop_class", {"node_id": "drives"})  # drives is a property


def test_drop_class_strips_dangling_axioms():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    human = KGClass(id="Human", label="Human", iri=EX + "Human")
    kg = KnowledgeGraph(name="K", nodes={person, human})
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["Person", "Human"]))
    out = _resolve(kg, "drop_class", {"node_id": "Person"})
    # Equivalence has only 1 surviving class, so the axiom is dropped.
    assert out.axioms == []


def test_drop_class_silent_on_missing_node():
    """If the node isn't there (e.g. already removed by an earlier decision),
    the handler is a no-op."""
    kg = _kg_with_two_classes()
    out = _resolve(kg, "drop_class", {"node_id": "DoesNotExist"})
    assert {n.id for n in out.nodes} == {"Person", "Car", "drives"}


# --------------------------------------------------------------------------
# drop_individual
# --------------------------------------------------------------------------


def test_drop_individual_removes_node_and_edges():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    bob = KGIndividual(id="bob", label="Bob", iri=EX + "bob")
    kg = KnowledgeGraph(name="K", nodes={person, alice, bob})
    kg.add_edge(KGEdge(id="t1", source=alice, target=person, label="type", iri=RDF_TYPE))
    kg.add_edge(KGEdge(id="t2", source=bob, target=person, label="type", iri=RDF_TYPE))
    kg.add_edge(KGEdge(id="k1", source=alice, target=bob, label="knows", iri=EX + "knows"))

    out = _resolve(kg, "drop_individual", {"node_id": "alice"})
    assert "alice" not in {n.id for n in out.nodes}
    assert {e.id for e in out.edges} == {"t2"}


def test_drop_individual_rejects_class():
    kg = _kg_with_two_classes()
    with pytest.raises(ResolutionError):
        _resolve(kg, "drop_individual", {"node_id": "Person"})


# --------------------------------------------------------------------------
# promote_individual_to_class
# --------------------------------------------------------------------------


def test_promote_individual_to_class_swaps_type_and_drops_rdf_type_edge():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    bob = KGIndividual(id="bob", label="Bob", iri=EX + "bob")
    kg = KnowledgeGraph(name="K", nodes={person, alice, bob})
    kg.add_edge(KGEdge(id="t1", source=alice, target=person, label="type", iri=RDF_TYPE))
    kg.add_edge(KGEdge(id="k1", source=bob, target=alice, label="knows", iri=EX + "knows"))

    out = _resolve(kg, "promote_individual_to_class", {"node_id": "alice"})

    new_alice = next(n for n in out.nodes if n.id == "alice")
    assert isinstance(new_alice, KGClass)
    # Outgoing rdf:type from alice → Person was dropped during promotion.
    assert "t1" not in {e.id for e in out.edges}
    # The 'knows' edge is rewired (alice is now a KGClass, but edge id 'k1' survives).
    knows = next(e for e in out.edges if e.id == "k1")
    assert knows.target is new_alice


def test_promote_individual_preserves_label_and_iri():
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    kg = KnowledgeGraph(name="K", nodes={alice})
    out = _resolve(kg, "promote_individual_to_class", {"node_id": "alice"})
    new_alice = next(n for n in out.nodes if n.id == "alice")
    assert new_alice.label == "Alice"
    assert new_alice.iri == EX + "alice"


def test_promote_individual_rejects_class():
    kg = _kg_with_two_classes()
    with pytest.raises(ResolutionError):
        _resolve(kg, "promote_individual_to_class", {"node_id": "Person"})


# --------------------------------------------------------------------------
# reclassify_node
# --------------------------------------------------------------------------


def test_reclassify_node_property_to_class():
    kg = _kg_with_two_classes()  # 'drives' is currently a KGProperty.
    out = _resolve(kg, "reclassify_node", {"node_id": "drives", "target_kind": "class"})
    new_drives = next(n for n in out.nodes if n.id == "drives")
    assert isinstance(new_drives, KGClass)
    # Edge endpoints repointed.
    domain_edge = next(e for e in out.edges if e.id == "d1")
    assert domain_edge.source is new_drives


def test_reclassify_node_idempotent():
    kg = _kg_with_two_classes()
    out = _resolve(kg, "reclassify_node", {"node_id": "Person", "target_kind": "class"})
    person = next(n for n in out.nodes if n.id == "Person")
    assert isinstance(person, KGClass)


def test_reclassify_node_rejects_invalid_target_kind():
    kg = _kg_with_two_classes()
    with pytest.raises(ResolutionError):
        _resolve(kg, "reclassify_node", {"node_id": "Person", "target_kind": "garbage"})


# --------------------------------------------------------------------------
# Dispatch via dispatch_decision (LLM round-trip path)
# --------------------------------------------------------------------------


def test_dispatch_decision_handles_noop_skip_action():
    kg = _kg_with_two_classes()
    issue = KGIssue(
        id="x",
        code="LLM_DROP_CLASS",
        description="drop Car",
        affected_node_ids=["Car"],
        recommended_action=KGAction(key="drop_class", parameters={"node_id": "Car"}, label="drop"),
        skip_action=KGAction(key="noop", parameters={}, label="keep"),
    )
    out_skip = dispatch_decision(kg, issue, "skip")
    assert {n.id for n in out_skip.nodes} == {"Person", "Car", "drives"}

    out_accept = dispatch_decision(kg, issue, "accept")
    assert "Car" not in {n.id for n in out_accept.nodes}


# --------------------------------------------------------------------------
# type_individual_as_class
# --------------------------------------------------------------------------


def test_type_individual_as_class_adds_rdf_type_edge():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    kg = KnowledgeGraph(name="K", nodes={person, alice})

    out = _resolve(kg, "type_individual_as_class", {"node_id": "alice", "class_id": "Person"})

    type_edges = [
        e for e in out.edges
        if e.source.id == "alice" and e.target.id == "Person" and e.iri == RDF_TYPE
    ]
    assert len(type_edges) == 1


def test_type_individual_as_class_is_idempotent():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    kg = KnowledgeGraph(name="K", nodes={person, alice})
    kg.add_edge(KGEdge(id="t1", source=alice, target=person, label="type", iri=RDF_TYPE))

    out = _resolve(kg, "type_individual_as_class", {"node_id": "alice", "class_id": "Person"})

    type_edges = [
        e for e in out.edges
        if e.source.id == "alice" and e.target.id == "Person" and e.iri == RDF_TYPE
    ]
    assert len(type_edges) == 1


def test_type_individual_as_class_rejects_class_target_node():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    car = KGClass(id="Car", label="Car", iri=EX + "Car")
    kg = KnowledgeGraph(name="K", nodes={person, car})
    # node_id points at a class, not an individual.
    with pytest.raises(ResolutionError):
        _resolve(kg, "type_individual_as_class", {"node_id": "Person", "class_id": "Car"})


def test_type_individual_as_class_requires_existing_class():
    alice = KGIndividual(id="alice", label="Alice", iri=EX + "alice")
    kg = KnowledgeGraph(name="K", nodes={alice})
    # class_id doesn't exist in the graph.
    with pytest.raises(ResolutionError):
        _resolve(kg, "type_individual_as_class", {"node_id": "alice", "class_id": "Ghost"})


def test_type_individual_as_class_silent_on_missing_individual():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    kg = KnowledgeGraph(name="K", nodes={person})
    # individual is gone (e.g. dropped earlier in the same batch).
    out = _resolve(kg, "type_individual_as_class", {"node_id": "alice", "class_id": "Person"})
    # No edges added, no exception.
    assert out.edges == set()


def test_axiom_strip_keeps_unrelated_axiom():
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    human = KGClass(id="Human", label="Human", iri=EX + "Human")
    car = KGClass(id="Car", label="Car", iri=EX + "Car")
    kg = KnowledgeGraph(name="K", nodes={person, human, car})
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["Person", "Human"]))
    kg.add_axiom(InversePropertiesAxiom(property_a_id="x", property_b_id="y"))  # dangling

    out = _resolve(kg, "drop_class", {"node_id": "Car"})

    # Equivalence (Person, Human) survives unchanged.
    surviving_eq = [a for a in out.axioms if isinstance(a, EquivalentClassesAxiom)]
    assert len(surviving_eq) == 1
    assert sorted(surviving_eq[0].class_ids) == ["Human", "Person"]
    # Inverse-properties axiom referenced missing nodes — dropped.
    assert all(not isinstance(a, InversePropertiesAxiom) for a in out.axioms)


# --------------------------------------------------------------------------
# drop_orphan_nodes
# --------------------------------------------------------------------------


def _kg_with_orphans() -> KnowledgeGraph:
    """A KG where Person is class-anchored, but ghost1/ghost2 are orphans
    connected to each other but not to any class."""
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    ghost1 = KGIndividual(id="ghost1", label="g1", iri=EX + "ghost1")
    ghost2 = KGIndividual(id="ghost2", label="g2", iri=EX + "ghost2")
    kg = KnowledgeGraph(name="K", nodes={person, ghost1, ghost2})
    kg.add_edge(KGEdge(id="g_e", source=ghost1, target=ghost2, label="rel",
                       iri=EX + "rel"))
    return kg


def test_drop_orphan_nodes_removes_batch_and_edges():
    kg = _kg_with_orphans()
    out = _resolve(kg, "drop_orphan_nodes", {"node_ids": ["ghost1", "ghost2"]})
    surviving = {n.id for n in out.nodes}
    assert surviving == {"Person"}
    assert out.edges == set()


def test_drop_orphan_nodes_does_not_mutate_input():
    kg = _kg_with_orphans()
    snapshot = (
        sorted(n.id for n in kg.nodes),
        sorted(e.id for e in kg.edges),
    )
    _resolve(kg, "drop_orphan_nodes", {"node_ids": ["ghost1", "ghost2"]})
    assert (sorted(n.id for n in kg.nodes), sorted(e.id for e in kg.edges)) == snapshot


def test_drop_orphan_nodes_tolerates_missing_ids():
    kg = _kg_with_orphans()
    out = _resolve(
        kg,
        "drop_orphan_nodes",
        {"node_ids": ["ghost1", "ghost2", "does_not_exist"]},
    )
    # Idempotent: the unknown id is silently skipped.
    surviving = {n.id for n in out.nodes}
    assert surviving == {"Person"}


def test_drop_orphan_nodes_strips_dangling_axioms():
    kg = _kg_with_orphans()
    # Add an axiom whose ids reference orphans about to be dropped.
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["ghost1", "ghost2"]))
    # And an axiom that should survive untouched.
    person2 = KGClass(id="Human", label="Human", iri=EX + "Human")
    kg.nodes.add(person2)
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["Person", "Human"]))

    out = _resolve(kg, "drop_orphan_nodes", {"node_ids": ["ghost1", "ghost2"]})

    surviving_eq = [a for a in out.axioms if isinstance(a, EquivalentClassesAxiom)]
    # Only the (Person, Human) axiom survives.
    assert len(surviving_eq) == 1
    assert sorted(surviving_eq[0].class_ids) == ["Human", "Person"]


def test_drop_orphan_nodes_empty_list_is_noop():
    kg = _kg_with_orphans()
    out = _resolve(kg, "drop_orphan_nodes", {"node_ids": []})
    assert {n.id for n in out.nodes} == {"Person", "ghost1", "ghost2"}


def test_drop_orphan_nodes_rejects_non_list():
    kg = _kg_with_orphans()
    with pytest.raises(ResolutionError):
        _resolve(kg, "drop_orphan_nodes", {"node_ids": "not_a_list"})


# --------------------------------------------------------------------------
# defer_to_llm_classification
# --------------------------------------------------------------------------


def test_defer_to_llm_classification_raises_typed_exception():
    kg = _kg_with_orphans()
    with pytest.raises(DeferredOrphanClassification) as excinfo:
        _resolve(
            kg,
            "defer_to_llm_classification",
            {"node_ids": ["ghost1", "ghost2"]},
        )
    assert excinfo.value.node_ids == ["ghost1", "ghost2"]


def test_defer_to_llm_classification_does_not_mutate_kg():
    kg = _kg_with_orphans()
    snapshot = (
        sorted(n.id for n in kg.nodes),
        sorted(e.id for e in kg.edges),
    )
    with pytest.raises(DeferredOrphanClassification):
        _resolve(
            kg,
            "defer_to_llm_classification",
            {"node_ids": ["ghost1", "ghost2"]},
        )
    assert (sorted(n.id for n in kg.nodes), sorted(e.id for e in kg.edges)) == snapshot


def test_defer_to_llm_classification_rejects_non_list():
    kg = _kg_with_orphans()
    with pytest.raises(ResolutionError):
        _resolve(
            kg,
            "defer_to_llm_classification",
            {"node_ids": "not_a_list"},
        )
