"""KnowledgeGraph → OWL / RDF serialization.

Deterministic, offline serializer that converts the typed KG metamodel back
into an ``rdflib.Graph`` and emits Turtle or RDF/XML. Designed to round-trip
with :mod:`~besser.BUML.notations.kg_rdf.importer`: import an OWL/TTL file, edit
the KG in the web editor, then export and the resulting RDF should be isomorphic
to the original.

Constraints are serialised using either the OWL vocabulary (anonymous
``owl:Restriction`` blank nodes hung off the target class via
``rdfs:subClassOf``) or the SHACL Core vocabulary (``sh:NodeShape`` /
``sh:PropertyShape``), or both. The ``vocab`` argument on
:func:`serialize_knowledge_graph` controls which vocabularies are emitted.

This is the *export*. The KG → UML conversion and the consistency check both
read a different projection, :func:`~besser.BUML.notations.kg_rdf.projection.kg_to_rdf`,
which keeps what this exporter drops around constraint nodes (punned
``rdfs:subClassOf``, ``sh:targetClass`` …). Both write through
:mod:`~besser.BUML.notations.kg_rdf._emit` and read constraint parameters from
``constraintSpecs`` only, and
``tests/BUML/notations/kg_rdf/test_projection_contract.py`` pins where they may
differ.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal as TypingLiteral, Optional, Set

import rdflib
from rdflib import RDF, RDFS, OWL, URIRef, XSD, BNode

from besser.BUML.metamodel.kg import (
    KGLiteral,
    KGNodeConstraint,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.BUML.metamodel.kg.constants import (
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH_PROPERTY,
)
from besser.BUML.notations.kg_rdf._emit import (
    CLASS_EXPRESSION_PREDICATES,
    DEFAULT_NAMESPACE,
    FLATTENED_LIST_PREDICATES,
    SH,
    blank_handles,
    build_constraint_indexes,
    emit_flattened_lists,
    emit_nodeconstraint_owl_axioms,
    emit_owl_class_expression,
    emit_owl_restriction,
    emit_property_chains,
    emit_shacl_constraint,
    emit_shacl_node_constraint,
    predicate_for_edge,
    spec_folded_by_term,
    term_for_node,
)


__all__ = [
    "DEFAULT_NAMESPACE",
    "VocabChoice",
    "knowledge_graph_to_rdf",
    "serialize_knowledge_graph",
]

logger = logging.getLogger(__name__)


#: Editor wiring between constraint nodes and what they constrain; the
#: constraint emitters below write the vocabulary's own form instead.
_INTERNAL_PREDICATES: Set[str] = {
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH_PROPERTY,
}


def _emit_shacl_property_shape_into(
    g: rdflib.Graph,
    pc: KGPropertyConstraint,
    property_term: Optional[Any],
    resolve_ref,
    *,
    shape_term: Any,
) -> Any:
    """Emit a ``sh:PropertyShape`` into ``shape_term`` (a pre-allocated rdflib
    term shared across the emit pass so cross-references resolve cleanly).
    """
    g.add((shape_term, RDF.type, SH.PropertyShape))
    if property_term is not None:
        g.add((shape_term, SH.path, property_term))
    for spec in pc.get_specs():
        emit_shacl_constraint(g, shape_term, spec, resolve_ref, SH.PropertyShape)
    return shape_term


def _emit_shacl_node_shape_into(
    g: rdflib.Graph,
    nc: KGNodeConstraint,
    target_classes: List[Any],
    property_shapes: List[URIRef | BNode],
    resolve_ref,
    *,
    shape_term: Any,
) -> Any:
    """Emit a ``sh:NodeShape`` into ``shape_term`` (pre-allocated)."""
    shape = shape_term
    g.add((shape, RDF.type, SH.NodeShape))
    for target in target_classes:
        g.add((shape, SH.targetClass, target))
    for ps in property_shapes:
        g.add((shape, SH.property, ps))
    for spec in nc.get_specs():
        emit_shacl_node_constraint(g, shape, spec, resolve_ref)
    return shape


VocabChoice = TypingLiteral["owl", "shacl", "both"]


def _vocab_set(vocab: VocabChoice) -> Set[str]:
    if vocab == "owl":
        return {"owl"}
    if vocab == "shacl":
        return {"shacl"}
    return {"owl", "shacl"}


def knowledge_graph_to_rdf(
    kg: KnowledgeGraph,
    *,
    default_namespace: str = DEFAULT_NAMESPACE,
    vocab: VocabChoice = "both",
) -> rdflib.Graph:
    """Build an ``rdflib.Graph`` from a ``KnowledgeGraph``.

    For each non-constraint edge, one triple is emitted. Internal predicates
    (``constraintTargetClass``, ``constraintTargetProperty``, ``sh:property``
    when used as bookkeeping between constraint nodes) are filtered out and
    replaced by vocabulary-appropriate constraint emission.

    Args:
        kg: The KnowledgeGraph to serialise.
        default_namespace: Namespace used when synthesising IRIs for nodes or
            predicates that lack one.
        vocab: Which constraint vocabularies to emit. ``"owl"`` emits
            anonymous ``owl:Restriction`` blank nodes hung from
            ``rdfs:subClassOf``; ``"shacl"`` emits ``sh:NodeShape`` /
            ``sh:PropertyShape``; ``"both"`` emits both.
    """
    g = rdflib.Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("rdf", RDF)
    g.bind("xsd", XSD)
    g.bind("sh", SH)

    nodes_by_id = {n.id: n for n in kg.nodes}
    vocabs = _vocab_set(vocab)
    terms: Dict[str, Any] = {n.id: term_for_node(n, default_namespace) for n in kg.nodes}
    folded = spec_folded_by_term(kg.nodes, terms)
    handles = blank_handles(kg.nodes, terms)
    idx = build_constraint_indexes(kg.edges, terms)

    # Emit ordinary triples (skipping internal predicates and skipping triples
    # involving constraint nodes — they're handled by the constraint emitters).
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
        if edge.iri in _INTERNAL_PREDICATES or edge.iri in FLATTENED_LIST_PREDICATES:
            continue
        if isinstance(src, (KGNodeConstraint, KGPropertyConstraint)) \
                or isinstance(tgt, (KGNodeConstraint, KGPropertyConstraint)):
            # Constraint nodes are emitted via the dedicated paths below.
            continue
        s, p, o = terms[src.id], predicate_for_edge(edge, default_namespace), terms[tgt.id]
        if str(p) in folded.get(s, ()):
            continue  # a punned shape's parameter: its spec is written below
        g.add((s, p, o))

    emit_flattened_lists(g, kg.nodes, kg.edges, terms, folded, kg.axioms)
    emit_property_chains(g, kg.axioms, terms)

    # Constraint emission. Every constraint node already has its term in
    # ``terms`` (named: its IRI; anonymous: a blank node named after its id, so
    # the consistency checker can map pyshacl's ``sh:sourceShape`` back to the
    # KG node), so a logical-operator spec like ``shaclAnd: [ {ref: "pc-other"} ]``
    # resolves whatever the emission order, and reference cycles terminate.
    def resolve_ref(node_id: str):
        node = nodes_by_id.get(node_id)
        return terms[node_id] if isinstance(node, (KGNodeConstraint, KGPropertyConstraint)) else None

    if "shacl" in vocabs:
        # Emit property shapes first (they may be referenced by node shapes
        # via sh:property and by logical operators via ref).
        for node in kg.nodes:
            if isinstance(node, KGPropertyConstraint):
                _emit_shacl_property_shape_into(
                    g, node, idx.pc_property.get(node.id), resolve_ref, shape_term=terms[node.id],
                )
        for node in kg.nodes:
            if isinstance(node, KGNodeConstraint):
                ps_terms = [terms[pc_id] for pc_id in idx.nc_pcs.get(node.id, [])]
                _emit_shacl_node_shape_into(
                    g, node, idx.nc_targets.get(node.id, []), ps_terms, resolve_ref,
                    shape_term=terms[node.id],
                )

    if "owl" in vocabs:
        owners = idx.pc_owners()
        for node in kg.nodes:
            if isinstance(node, KGPropertyConstraint):
                emit_owl_restriction(
                    g, node, idx.pc_property.get(node.id) or node.metadata.get("onPropertyIri"),
                    owners.get(node.id, []), blank_handles=handles,
                )
            elif isinstance(node, KGNodeConstraint):
                targets = idx.nc_targets.get(node.id, [])
                emit_nodeconstraint_owl_axioms(g, node, targets, handles)
                # A class expression the importer lifted stays anonymous: the
                # target class is subclassed by the expression's own node.
                if any(spec.get("kind") in CLASS_EXPRESSION_PREDICATES for spec in node.get_specs()):
                    emit_owl_class_expression(g, node, terms[node.id], handles)
                    for target in targets:
                        g.add((target, RDFS.subClassOf, terms[node.id]))

    return g


def serialize_knowledge_graph(
    kg: KnowledgeGraph,
    fmt: str = "turtle",
    *,
    default_namespace: str = DEFAULT_NAMESPACE,
    vocab: VocabChoice = "both",
) -> str:
    """Serialize a ``KnowledgeGraph`` to OWL (RDF/XML) or Turtle.

    Args:
        kg: The KnowledgeGraph instance to serialize.
        fmt: rdflib serialization format. ``"turtle"`` produces TTL,
            ``"xml"`` produces RDF/XML (the default OWL serialization).
        default_namespace: Namespace used when synthesising IRIs for nodes or
            predicates that lack one.
        vocab: Which constraint vocabularies to emit (``"owl"`` /
            ``"shacl"`` / ``"both"``). See :func:`knowledge_graph_to_rdf`.
    """
    g = knowledge_graph_to_rdf(kg, default_namespace=default_namespace, vocab=vocab)
    # rdflib >= 6 (7.0 is the pinned floor) returns ``str`` here.
    return g.serialize(format=fmt)
