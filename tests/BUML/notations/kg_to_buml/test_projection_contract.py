"""The contract between the two KG -> RDF projections.

BESSER projects a ``KnowledgeGraph`` into RDF in two deliberately different ways:

=================  =========================================  ==========================
                   ``knowledge_graph_to_rdf`` (exporter)      ``kg_to_rdf`` (converter)
=================  =========================================  ==========================
Purpose            the user-facing Turtle / RDF-XML export,   input to the KG -> UML
                   and the data the consistency check         conversion
                   validates
Constraint nodes   rebuilt from their structured specs in     re-materialised onto the
                   the vocabulary the caller chooses          node's own term
``vocab``          ``owl`` / ``shacl`` / ``both``             always both
Individuals        as the KG has them                         also declared
                                                              ``owl:NamedIndividual``
=================  =========================================  ==========================

They share their leaf emitters (``kg_to_owl.emit_*``) so they cannot drift on
how any single construct is written. This module pins what is left: with the
SHACL vocabulary set aside, the two graphs must be isomorphic apart from the
converter's ``owl:NamedIndividual`` declarations.

That turns "two views of the same KG" from an unstated assumption into a
documented, drift-detecting contract. It deliberately does NOT assert that the
exporter and converter agree on SHACL: the exporter mirrors OWL-sourced
restrictions as shapes for the consistency check, which the converter must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import OWL, RDF, Graph, Namespace, URIRef
from rdflib.compare import isomorphic

from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
from besser.utilities.kg_to_owl import knowledge_graph_to_rdf
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph

SH = Namespace("http://www.w3.org/ns/shacl#")

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"      # tests/fixtures
_KG = Path(__file__).resolve().parent / "fixtures"


def _without_shacl(graph: Graph) -> Graph:
    out = Graph()
    for s, p, o in graph:
        if str(p).startswith(str(SH)) or str(o).startswith(str(SH)):
            continue
        out.add((s, p, o))
    return out


def _without_individual_declarations(graph: Graph) -> Graph:
    """Drop the converter's ``x a owl:NamedIndividual`` declarations."""
    out = Graph()
    for s, p, o in graph:
        if p == RDF.type and o == OWL.NamedIndividual:
            continue
        out.add((s, p, o))
    return out


def _views(path: Path):
    kg = owl_file_to_knowledge_graph(str(path))
    exported = _without_individual_declarations(_without_shacl(knowledge_graph_to_rdf(kg, vocab="owl")))
    converted = _without_individual_declarations(_without_shacl(kg_to_rdf(kg)[0]))
    return exported, converted


def test_the_projections_agree_on_a_graph_with_restrictions_and_individuals():
    exported, converted = _views(_FIXTURES / "complex_kg.ttl")

    assert len(exported) > 0
    assert isomorphic(exported, converted)


def test_individual_declarations_are_the_only_thing_the_contract_sets_aside():
    """Without stripping them the graphs differ; with only them stripped they match.

    So the exclusion in ``_views`` is exactly the converter's declarations and
    hides nothing else. (Compared by isomorphism: the exporter mints fresh blank
    nodes, so a raw triple set-difference would be meaningless.)
    """
    kg = owl_file_to_knowledge_graph(str(_FIXTURES / "complex_kg.ttl"))
    exported = _without_shacl(knowledge_graph_to_rdf(kg, vocab="owl"))
    converted = _without_shacl(kg_to_rdf(kg)[0])

    def declarations(graph):
        return [1 for _, p, o in graph if p == RDF.type and o == OWL.NamedIndividual]

    assert not isomorphic(exported, converted)
    assert isomorphic(
        _without_individual_declarations(exported), _without_individual_declarations(converted)
    )
    # The KG may already declare some individuals (the exporter keeps those); the
    # converter's pass declares the rest.
    assert len(declarations(converted)) > len(declarations(exported))


_BIBO_DANGLING_FILLER = pytest.mark.xfail(
    strict=True,
    reason="exporter rebuilds a blank-node restriction filler as a dangling relative IRI "
           "(kg_to_owl._emit_restriction_value); fixed together with this marker's removal",
)


@_BIBO_DANGLING_FILLER
def test_the_projections_agree_on_a_real_ontology():
    exported, converted = _views(_KG / "bibo" / "bibo.ttl")

    assert isomorphic(exported, converted)


def _dangling_fillers(graph: Graph):
    """``allValuesFrom`` fillers that are IRIs but not absolute, with no triples."""
    return [
        filler
        for _, _, filler in graph.triples((None, OWL.allValuesFrom, None))
        if isinstance(filler, URIRef)
        and "://" not in str(filler)
        and not str(filler).startswith("urn:")
    ]


@_BIBO_DANGLING_FILLER
def test_exporter_keeps_a_blank_node_restriction_filler_connected():
    """``allValuesFrom`` whose filler is a blank-node class stays a blank node.

    A restriction can name an anonymous class (here a union) as its filler. The
    exporter rebuilt the filler with ``URIRef(str(value))``, so the blank node's
    label became a relative IRI that nothing else in the graph mentions: a
    dangling reference, and a different one on every import, since the label is
    minted per import.
    """
    kg = owl_file_to_knowledge_graph(str(_KG / "bibo" / "bibo.ttl"))

    assert _dangling_fillers(knowledge_graph_to_rdf(kg, vocab="owl")) == []
    assert _dangling_fillers(kg_to_rdf(kg)[0]) == []          # the converter never had the bug
