"""Tests for the Knowledge Graph metamodel."""

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
