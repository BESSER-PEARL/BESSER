"""KnowledgeGraph → OWL / RDF serialization.

Deterministic, offline serializer that converts the typed KG metamodel back
into an ``rdflib.Graph`` and emits Turtle or RDF/XML. Designed to round-trip
with :mod:`besser.utilities.owl_to_buml`: import an OWL/TTL file, edit the KG
in the web editor, then export and the resulting RDF should be isomorphic to
the original.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Union

import rdflib
from rdflib import BNode, Literal, URIRef

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGProperty,
    KnowledgeGraph,
)


__all__ = [
    "DEFAULT_NAMESPACE",
    "knowledge_graph_to_rdf",
    "serialize_knowledge_graph",
]


DEFAULT_NAMESPACE = "http://besser-pearl.org/kg#"

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Return a URI-safe slug derived from arbitrary text."""
    if not text:
        return "node"
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
    return cleaned or "node"


def _term_for_node(node: KGNode, default_ns: str) -> Union[URIRef, BNode, Literal]:
    """Map a KGNode to the appropriate rdflib term."""
    if isinstance(node, KGLiteral):
        if node.datatype:
            try:
                return Literal(node.value, datatype=URIRef(node.datatype))
            except Exception:  # pragma: no cover - rdflib accepts any string
                logger.warning(
                    "KGLiteral %r has malformed datatype %r; emitting plain literal.",
                    node.id, node.datatype,
                )
        return Literal(node.value)
    if isinstance(node, KGBlank):
        bn_id = node.id[2:] if node.id.startswith("_:") else node.id
        return BNode(bn_id) if bn_id else BNode()
    if node.iri:
        return URIRef(node.iri)
    return URIRef(default_ns + _slugify(node.label or node.id))


def _predicate_for_edge(edge: KGEdge, default_ns: str) -> URIRef:
    """Map a KGEdge to its predicate URIRef."""
    if edge.iri:
        return URIRef(edge.iri)
    return URIRef(default_ns + _slugify(edge.label or "relatedTo"))


def knowledge_graph_to_rdf(
    kg: KnowledgeGraph,
    *,
    default_namespace: str = DEFAULT_NAMESPACE,
) -> rdflib.Graph:
    """Build an ``rdflib.Graph`` from a ``KnowledgeGraph``.

    Each ``KGEdge`` becomes one triple. Nodes that are not referenced by any
    edge contribute no triples — this matches the import behavior, which only
    materialises a node when it appears in some triple.
    """
    g = rdflib.Graph()
    nodes_by_id = {n.id: n for n in kg.nodes}

    for edge in kg.edges:
        src = nodes_by_id.get(edge.source.id, edge.source)
        tgt = nodes_by_id.get(edge.target.id, edge.target)
        if src.id not in nodes_by_id or tgt.id not in nodes_by_id:
            logger.warning(
                "Skipping KGEdge %r: source/target not in graph (source=%r, target=%r).",
                edge.id, edge.source.id, edge.target.id,
            )
            continue
        if isinstance(src, KGLiteral):
            logger.warning(
                "Skipping KGEdge %r: subject is a literal, which is not valid RDF.",
                edge.id,
            )
            continue
        s = _term_for_node(src, default_namespace)
        p = _predicate_for_edge(edge, default_namespace)
        o = _term_for_node(tgt, default_namespace)
        g.add((s, p, o))

    return g


def serialize_knowledge_graph(
    kg: KnowledgeGraph,
    fmt: str = "turtle",
    *,
    default_namespace: str = DEFAULT_NAMESPACE,
) -> str:
    """Serialize a ``KnowledgeGraph`` to OWL (RDF/XML) or Turtle.

    Args:
        kg: The KnowledgeGraph instance to serialize.
        fmt: rdflib serialization format. ``"turtle"`` produces TTL,
            ``"xml"`` produces RDF/XML (the default OWL serialization).
        default_namespace: Namespace used when synthesising IRIs for nodes or
            predicates that lack one.
    """
    g = knowledge_graph_to_rdf(kg, default_namespace=default_namespace)
    serialized = g.serialize(format=fmt)
    if isinstance(serialized, bytes):  # rdflib < 6 returned bytes
        return serialized.decode("utf-8")
    return serialized
