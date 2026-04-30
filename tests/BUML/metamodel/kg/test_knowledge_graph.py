"""Tests for the Knowledge Graph metamodel."""

import pytest

from besser.BUML.metamodel.kg import (
    DisjointClassesAxiom,
    EquivalentClassesAxiom,
    HasKeyAxiom,
    ImportAxiom,
    InversePropertiesAxiom,
    KGAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGProperty,
    KnowledgeGraph,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)


def test_knowledge_graph_accepts_all_node_kinds():
    c = KGClass(id="http://ex/Person", label="Person", iri="http://ex/Person")
    i = KGIndividual(id="http://ex/alice", label="Alice", iri="http://ex/alice")
    p = KGProperty(id="http://ex/knows", label="knows", iri="http://ex/knows")
    lit = KGLiteral(id="lit:1", value="30", datatype="http://www.w3.org/2001/XMLSchema#integer")
    b = KGBlank(id="_:b0", label="b0")
    kg = KnowledgeGraph(name="graph", nodes={c, i, p, lit, b})

    assert kg.get_node("http://ex/Person") is c
    assert kg.get_node("lit:1") is lit
    assert len(kg.nodes) == 5


def test_edge_endpoints_must_be_in_graph():
    c = KGClass(id="c1", label="C1")
    i = KGIndividual(id="i1", label="I1")
    kg = KnowledgeGraph(name="graph", nodes={c, i})
    kg.add_edge(KGEdge(id="e1", source=i, target=c, label="type"))
    assert len(kg.edges) == 1

    stray = KGIndividual(id="stray", label="stray")
    with pytest.raises(ValueError):
        kg.add_edge(KGEdge(id="e2", source=i, target=stray))


def test_literal_requires_value():
    with pytest.raises(ValueError):
        KGLiteral(id="lit:2", value=None)  # type: ignore[arg-type]


def test_node_id_must_be_non_empty():
    with pytest.raises(ValueError):
        KGClass(id="", label="X")


def test_graph_name_rejects_spaces():
    # Inherited NamedElement.name validation disallows spaces.
    with pytest.raises(ValueError):
        KnowledgeGraph(name="has space")


# --- metadata field on KGNode / KGEdge -------------------------------------


def test_kgnode_metadata_defaults_to_empty_dict():
    n = KGClass(id="c1", label="C1")
    assert n.metadata == {}
    # Mutating the dict in place is the supported workflow.
    n.metadata["kind"] = "restriction"
    assert n.metadata == {"kind": "restriction"}


def test_kgnode_metadata_accepts_dict_via_constructor():
    payload = {"kind": "restriction", "on_property": "http://ex/age", "value": 1}
    n = KGBlank(id="_:b0", label="b0", metadata=payload)
    assert n.metadata == payload


def test_kgnode_metadata_setter_rejects_non_dict():
    n = KGClass(id="c1", label="C1")
    with pytest.raises(ValueError):
        n.metadata = "not-a-dict"  # type: ignore[assignment]


def test_kgnode_metadata_does_not_affect_identity():
    a = KGClass(id="c1", label="C1", metadata={"foo": 1})
    b = KGClass(id="c1", label="C1", metadata={"bar": 2})
    # Equality and hash are id-based, not metadata-based.
    assert a == b
    assert hash(a) == hash(b)


def test_kgliteral_metadata_propagates():
    lit = KGLiteral(id="lit:1", value="30", metadata={"source": "ABox"})
    assert lit.metadata == {"source": "ABox"}


def test_kgedge_metadata_defaults_and_setter():
    c = KGClass(id="c1", label="C1")
    i = KGIndividual(id="i1", label="I1")
    kg = KnowledgeGraph(name="graph", nodes={c, i})

    e = KGEdge(id="e1", source=i, target=c, label="type")
    assert e.metadata == {}

    e2 = KGEdge(id="e2", source=i, target=c, label="type", metadata={"derived_from": "e1"})
    assert e2.metadata == {"derived_from": "e1"}

    with pytest.raises(ValueError):
        e2.metadata = 42  # type: ignore[assignment]


# --- KnowledgeGraph.axioms -------------------------------------------------


def test_knowledge_graph_axioms_default_empty():
    kg = KnowledgeGraph(name="graph")
    assert kg.axioms == []


def test_add_axiom_appends_and_validates_type():
    kg = KnowledgeGraph(name="graph")
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["c1", "c2"]))
    kg.add_axiom(DisjointClassesAxiom(class_ids=["c1", "c3"]))
    assert len(kg.axioms) == 2
    assert isinstance(kg.axioms[0], EquivalentClassesAxiom)
    assert isinstance(kg.axioms[0], KGAxiom)

    with pytest.raises(ValueError):
        kg.add_axiom("not-an-axiom")  # type: ignore[arg-type]


def test_axioms_setter_rejects_non_list_and_bad_elements():
    kg = KnowledgeGraph(name="graph")
    with pytest.raises(ValueError):
        kg.axioms = {"not": "a list"}  # type: ignore[assignment]
    with pytest.raises(ValueError):
        kg.axioms = ["not an axiom"]  # type: ignore[list-item]


def test_each_axiom_subclass_round_trips():
    """Smoke-test that every axiom subclass instantiates and stores cleanly."""
    kg = KnowledgeGraph(name="graph")
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["a", "b"]))
    kg.add_axiom(DisjointClassesAxiom(class_ids=["a", "c"]))
    kg.add_axiom(SubPropertyOfAxiom(sub_property_id="hasFather", super_property_id="hasParent"))
    kg.add_axiom(InversePropertiesAxiom(property_a_id="hasParent", property_b_id="hasChild"))
    kg.add_axiom(PropertyChainAxiom(property_id="hasUncle", chain_property_ids=["hasParent", "hasBrother"]))
    kg.add_axiom(HasKeyAxiom(class_id="Person", property_ids=["ssn"]))
    kg.add_axiom(ImportAxiom(target_iri="http://xmlns.com/foaf/0.1/"))
    assert len(kg.axioms) == 7
    # Every entry is a KGAxiom.
    assert all(isinstance(a, KGAxiom) for a in kg.axioms)
