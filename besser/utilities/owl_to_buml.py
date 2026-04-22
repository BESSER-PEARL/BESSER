"""OWL / RDF → KnowledgeGraph conversion.

Deterministic, offline parser that loads an RDF graph with rdflib and maps
each term (URI, blank node, literal) onto the five KG node types used by the
web editor's Knowledge Graph diagram.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional

import rdflib
from rdflib import RDF, RDFS, OWL, BNode, Literal, URIRef

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


__all__ = ["owl_file_to_knowledge_graph"]


_PROPERTY_TYPES = {
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    RDF.Property,
}
_CLASS_TYPES = {OWL.Class, RDFS.Class}


def _local_name(iri: str) -> str:
    """Return the local name of an IRI (after ``#`` or the last ``/``)."""
    if not iri:
        return ""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    return iri


def _literal_id(lit: Literal) -> str:
    """Stable id for a literal based on lexical form + datatype."""
    raw = f"{lit}\0{lit.datatype or ''}\0{lit.language or ''}"
    digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"lit:{digest}"


def _classify(term, classes: set, properties: set) -> str:
    """Decide the KG node kind for an rdflib term (``class`` / ``individual`` /
    ``property`` / ``literal`` / ``blank``)."""
    if isinstance(term, Literal):
        return "literal"
    if isinstance(term, BNode):
        return "blank"
    if isinstance(term, URIRef):
        if term in classes:
            return "class"
        if term in properties:
            return "property"
        return "individual"
    return "individual"


def owl_file_to_knowledge_graph(path: str) -> KnowledgeGraph:
    """Parse an OWL/RDF file and return a populated ``KnowledgeGraph``.

    Every triple becomes a ``KGEdge`` (including ``rdf:type`` triples) so the
    import is transparent — the UI can later choose to hide type edges.

    Args:
        path: Filesystem path to the ontology file. Supported serializations:
            ``.owl`` (RDF/XML), ``.rdf``, ``.xml``, ``.ttl``, ``.nt``, ``.n3``.
            The format is auto-detected via ``rdflib.util.guess_format``;
            ``.owl`` defaults to ``xml`` when guess fails.
    """
    g = rdflib.Graph()
    fmt = rdflib.util.guess_format(path)
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
        fmt = "xml" if ext in (".owl", ".rdf") else None
    g.parse(path, format=fmt) if fmt else g.parse(path)

    # First pass: collect explicit class and property IRIs.
    classes: set = set()
    properties: set = set()
    type_objects: set = set()
    for s, p, o in g.triples((None, RDF.type, None)):
        type_objects.add(o)
        if o in _CLASS_TYPES:
            classes.add(s)
        elif o in _PROPERTY_TYPES:
            properties.add(s)
    # Terms appearing as the object of rdf:type (and not already a property) are
    # treated as classes — they're being used as types.
    for o in type_objects:
        if isinstance(o, URIRef) and o not in properties and o not in _CLASS_TYPES and o not in _PROPERTY_TYPES:
            classes.add(o)

    # Second pass: build nodes.
    nodes_by_key: Dict[str, KGNode] = {}

    def _get_or_create_node(term) -> KGNode:
        kind = _classify(term, classes, properties)
        if kind == "literal":
            nid = _literal_id(term)
            if nid in nodes_by_key:
                return nodes_by_key[nid]
            datatype = str(term.datatype) if term.datatype is not None else None
            node: KGNode = KGLiteral(id=nid, value=str(term), datatype=datatype)
        elif kind == "blank":
            nid = f"_:{str(term)}"
            if nid in nodes_by_key:
                return nodes_by_key[nid]
            node = KGBlank(id=nid, label=str(term))
        else:
            iri = str(term)
            nid = iri
            if nid in nodes_by_key:
                return nodes_by_key[nid]
            label = _label_for(g, term) or _local_name(iri)
            if kind == "class":
                node = KGClass(id=nid, label=label, iri=iri)
            elif kind == "property":
                node = KGProperty(id=nid, label=label, iri=iri)
            else:
                node = KGIndividual(id=nid, label=label, iri=iri)
        nodes_by_key[nid] = node
        return node

    edges = []
    edge_counter = 0
    for s, p, o in g:
        src = _get_or_create_node(s)
        tgt = _get_or_create_node(o)
        edge_counter += 1
        predicate_iri = str(p)
        edges.append(
            KGEdge(
                id=f"edge:{edge_counter}",
                source=src,
                target=tgt,
                label=_local_name(predicate_iri),
                iri=predicate_iri,
            )
        )

    kg = KnowledgeGraph(name="knowledge_graph", nodes=set(nodes_by_key.values()))
    for e in edges:
        kg.add_edge(e)
    return kg


def _label_for(g: rdflib.Graph, term) -> Optional[str]:
    """Return the first ``rdfs:label`` for a term, if any."""
    for _, _, lbl in g.triples((term, RDFS.label, None)):
        if isinstance(lbl, Literal):
            return str(lbl)
    return None
