"""Unit tests for ``besser.utilities.kg_to_owl``.

Covers the in-memory serializer in isolation: TTL/RDF-XML output, round-trip
isomorphism with the OWL importer, and edge cases around missing IRIs,
literals with datatypes, and blank nodes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rdflib
from rdflib.compare import isomorphic

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGProperty,
    KnowledgeGraph,
)
from besser.utilities.kg_to_owl import (
    DEFAULT_NAMESPACE,
    knowledge_graph_to_rdf,
    serialize_knowledge_graph,
)
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph


TTL_FIXTURE = """
@prefix : <http://ex.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Person a owl:Class .
:Employee a owl:Class ; rdfs:subClassOf :Person .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
:knows a owl:ObjectProperty ; rdfs:domain :Person ; rdfs:range :Person .
:alice a :Employee ; :name "Alice" ; :knows :bob ; :age "42"^^xsd:integer .
:bob a :Person ; :name "Bob" .
""".strip()


@pytest.fixture
def ttl_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.ttl"
    p.write_text(TTL_FIXTURE, encoding="utf-8")
    return p


def test_serialize_turtle_returns_valid_ttl(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    out = serialize_knowledge_graph(kg, fmt="turtle")
    assert isinstance(out, str)
    parsed = rdflib.Graph().parse(data=out, format="turtle")
    assert len(parsed) > 0


def test_serialize_xml_returns_valid_rdfxml(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    out = serialize_knowledge_graph(kg, fmt="xml")
    assert isinstance(out, str)
    parsed = rdflib.Graph().parse(data=out, format="xml")
    assert len(parsed) > 0


def test_round_trip_isomorphic_turtle(ttl_path: Path):
    g_orig = rdflib.Graph().parse(str(ttl_path), format="turtle")
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    out = serialize_knowledge_graph(kg, fmt="turtle")
    g_round = rdflib.Graph().parse(data=out, format="turtle")
    assert isomorphic(g_orig, g_round), (
        "Turtle round-trip is not isomorphic.\n"
        f"--- original ---\n{g_orig.serialize(format='turtle')}\n"
        f"--- round-tripped ---\n{out}"
    )


def test_round_trip_isomorphic_xml(ttl_path: Path):
    g_orig = rdflib.Graph().parse(str(ttl_path), format="turtle")
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    out = serialize_knowledge_graph(kg, fmt="xml")
    g_round = rdflib.Graph().parse(data=out, format="xml")
    assert isomorphic(g_orig, g_round)


def test_literal_with_datatype_preserved(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    out = serialize_knowledge_graph(kg, fmt="turtle")
    g_round = rdflib.Graph().parse(data=out, format="turtle")
    integer_dt = rdflib.URIRef("http://www.w3.org/2001/XMLSchema#integer")
    typed_literals = [o for _, _, o in g_round if isinstance(o, rdflib.Literal) and o.datatype == integer_dt]
    assert any(str(lit) == "42" for lit in typed_literals)


def test_blank_node_round_trip(tmp_path: Path):
    ttl_with_bnode = """
@prefix : <http://ex.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

:Person a owl:Class .
_:b0 a :Person .
:knows a owl:ObjectProperty .
:alice a :Person ; :knows _:b0 .
""".strip()
    p = tmp_path / "bnode.ttl"
    p.write_text(ttl_with_bnode, encoding="utf-8")

    g_orig = rdflib.Graph().parse(str(p), format="turtle")
    kg = owl_file_to_knowledge_graph(str(p))
    out = serialize_knowledge_graph(kg, fmt="turtle")
    g_round = rdflib.Graph().parse(data=out, format="turtle")
    assert isomorphic(g_orig, g_round)


def test_node_without_iri_synthesizes_uri():
    kg = KnowledgeGraph(name="anon")
    a = KGClass(id="a", label="Anon", iri=None)
    b = KGClass(id="b", label="Other", iri=None)
    kg.add_node(a)
    kg.add_node(b)
    kg.add_edge(KGEdge(id="e1", source=a, target=b, label="rel", iri=None))

    out = serialize_knowledge_graph(kg, fmt="turtle")
    g_round = rdflib.Graph().parse(data=out, format="turtle")
    assert len(g_round) == 1
    s, p, o = next(iter(g_round))
    assert str(s).startswith(DEFAULT_NAMESPACE)
    assert str(p).startswith(DEFAULT_NAMESPACE)
    assert str(o).startswith(DEFAULT_NAMESPACE)


def test_edge_without_iri_synthesizes_predicate():
    kg = KnowledgeGraph(name="x")
    a = KGIndividual(id="http://ex.org/a", label="a", iri="http://ex.org/a")
    b = KGIndividual(id="http://ex.org/b", label="b", iri="http://ex.org/b")
    kg.add_node(a)
    kg.add_node(b)
    kg.add_edge(KGEdge(id="e1", source=a, target=b, label="related", iri=None))

    g = knowledge_graph_to_rdf(kg)
    triples = list(g)
    assert len(triples) == 1
    _, p, _ = triples[0]
    assert "related" in str(p)


def test_empty_graph_serializes():
    kg = KnowledgeGraph(name="empty")
    out_ttl = serialize_knowledge_graph(kg, fmt="turtle")
    out_xml = serialize_knowledge_graph(kg, fmt="xml")
    assert isinstance(out_ttl, str)
    assert isinstance(out_xml, str)
    assert len(rdflib.Graph().parse(data=out_ttl, format="turtle")) == 0
    assert len(rdflib.Graph().parse(data=out_xml, format="xml")) == 0


def test_malformed_datatype_falls_back_to_plain_literal():
    kg = KnowledgeGraph(name="dt")
    a = KGIndividual(id="http://ex.org/a", label="a", iri="http://ex.org/a")
    lit = KGLiteral(id="lit:1", value="hello", datatype="not a uri")
    kg.add_node(a)
    kg.add_node(lit)
    kg.add_edge(
        KGEdge(id="e1", source=a, target=lit, label="says", iri="http://ex.org/says")
    )

    out = serialize_knowledge_graph(kg, fmt="turtle")
    g_round = rdflib.Graph().parse(data=out, format="turtle")
    literals = [o for _, _, o in g_round if isinstance(o, rdflib.Literal)]
    assert any(str(lit) == "hello" for lit in literals)


def test_property_node_round_trip(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    assert any(isinstance(n, KGProperty) for n in kg.nodes)
    out = serialize_knowledge_graph(kg, fmt="turtle")
    g_round = rdflib.Graph().parse(data=out, format="turtle")
    owl_object_property = rdflib.URIRef("http://www.w3.org/2002/07/owl#ObjectProperty")
    rdf_type = rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    knows = rdflib.URIRef("http://ex.org/knows")
    assert (knows, rdf_type, owl_object_property) in g_round


def test_blank_subject_via_blanknode_class():
    kg = KnowledgeGraph(name="blank")
    bn = KGBlank(id="_:b0", label="b0")
    cls = KGClass(id="http://ex.org/Thing", label="Thing", iri="http://ex.org/Thing")
    kg.add_node(bn)
    kg.add_node(cls)
    kg.add_edge(
        KGEdge(
            id="e1",
            source=bn,
            target=cls,
            label="type",
            iri="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        )
    )
    g = knowledge_graph_to_rdf(kg)
    triples = list(g)
    assert len(triples) == 1
    s, _, _ = triples[0]
    assert isinstance(s, rdflib.BNode)
