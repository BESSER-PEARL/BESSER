"""KnowledgeGraph → OWL / RDF serialization.

Deterministic, offline serializer that converts the typed KG metamodel back
into an ``rdflib.Graph`` and emits Turtle or RDF/XML. Designed to round-trip
with :mod:`besser.utilities.owl_to_buml`: import an OWL/TTL file, edit the KG
in the web editor, then export and the resulting RDF should be isomorphic to
the original.

Constraints are serialised using either the OWL vocabulary (anonymous
``owl:Restriction`` blank nodes hung off the target class via
``rdfs:subClassOf``) or the SHACL Core vocabulary (``sh:NodeShape`` /
``sh:PropertyShape``), or both. The ``vocab`` argument on
:func:`serialize_knowledge_graph` controls which vocabularies are emitted.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Literal as TypingLiteral, Optional, Set, Tuple, Union

import rdflib
from rdflib import BNode, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.BUML.metamodel.kg.constants import (
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH_PROPERTY,
)


__all__ = [
    "DEFAULT_NAMESPACE",
    "knowledge_graph_to_rdf",
    "serialize_knowledge_graph",
]


DEFAULT_NAMESPACE = "http://besser-pearl.org/kg#"
SH = Namespace("http://www.w3.org/ns/shacl#")

logger = logging.getLogger(__name__)


_INTERNAL_PREDICATES: Set[str] = {
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH_PROPERTY,
}


_OWL_RESTRICTION_KINDS: Dict[str, URIRef] = {
    "minCardinality": OWL.minCardinality,
    "maxCardinality": OWL.maxCardinality,
    "exactCardinality": OWL.cardinality,
    "minQualifiedCardinality": OWL.minQualifiedCardinality,
    "maxQualifiedCardinality": OWL.maxQualifiedCardinality,
    "exactQualifiedCardinality": OWL.qualifiedCardinality,
    "someValuesFrom": OWL.someValuesFrom,
    "allValuesFrom": OWL.allValuesFrom,
    "hasValue": OWL.hasValue,
    "hasSelf": OWL.hasSelf,
}


_SHACL_PROPERTY_KINDS: Dict[str, URIRef] = {
    "minCardinality": SH.minCount,
    "maxCardinality": SH.maxCount,
    "minQualifiedCardinality": SH.qualifiedMinCount,
    "maxQualifiedCardinality": SH.qualifiedMaxCount,
    "someValuesFrom": SH["class"],
    "hasValue": SH.hasValue,
    "datatype": SH.datatype,
    "nodeKind": SH.nodeKind,
    "pattern": SH.pattern,
    "flags": SH.flags,
    "minLength": SH.minLength,
    "maxLength": SH.maxLength,
    "minInclusive": SH.minInclusive,
    "maxInclusive": SH.maxInclusive,
    "minExclusive": SH.minExclusive,
    "maxExclusive": SH.maxExclusive,
    "languageIn": SH.languageIn,
    "uniqueLang": SH.uniqueLang,
    "in": SH["in"],
    "shaclSeverity": SH.severity,
    "shaclMessage": SH.message,
    "shaclName": SH.name,
    "shaclDescription": SH.description,
    "shaclDeactivated": SH.deactivated,
    "shaclOrder": SH.order,
    "shaclGroup": SH.group,
}


_SHACL_NODE_KINDS: Dict[str, URIRef] = {
    "shaclClosed": SH.closed,
    "shaclIgnoredProperties": SH.ignoredProperties,
    "shaclDisjoint": SH.disjoint,
    "shaclSeverity": SH.severity,
    "shaclMessage": SH.message,
    "shaclName": SH.name,
    "shaclDescription": SH.description,
    "shaclDeactivated": SH.deactivated,
}


_NODECONSTRAINT_OWL_AXIOM_KINDS: Set[str] = {
    "equivalentClasses",
    "disjointWith",
    "subClassOf",
    "complementOf",
    "oneOf",
    "hasKey",
    "disjointUnionOf",
    "unionOf",
    "intersectionOf",
}


def _slugify(text: str) -> str:
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
    if isinstance(node, (KGNodeConstraint, KGPropertyConstraint)):
        if node.iri:
            return URIRef(node.iri)
        # Deterministic BNode so the consistency checker can map pyshacl's
        # `sh:sourceShape` (the emitted blank-node subject of the shape's
        # SHACL triples) back to the original KG constraint-node id. We
        # slugify because BNode ids must be valid blank-node labels.
        return BNode(_slugify(node.id))
    if node.iri:
        return URIRef(node.iri)
    return URIRef(default_ns + _slugify(node.label or node.id))


def _predicate_for_edge(edge: KGEdge, default_ns: str) -> URIRef:
    if edge.iri:
        return URIRef(edge.iri)
    return URIRef(default_ns + _slugify(edge.label or "relatedTo"))


def _spec_to_literal(value: Any) -> Optional[Literal]:
    """Convert a spec value to an rdflib Literal, picking an XSD datatype."""
    if value is None:
        return None
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.nonNegativeInteger if value >= 0 else XSD.integer)
    if isinstance(value, float):
        return Literal(value, datatype=XSD.decimal)
    if isinstance(value, dict) and "value" in value:
        lex = value.get("value")
        dt = value.get("datatype")
        lang = value.get("language")
        if lang:
            return Literal(lex, lang=lang)
        if dt:
            return Literal(lex, datatype=URIRef(dt))
        return Literal(lex)
    return Literal(str(value))


def _emit_rdf_list(g: rdflib.Graph, items: Iterable[Any]) -> BNode:
    """Emit an ``rdf:List`` for an iterable of items, returning the list head."""
    items = list(items)
    if not items:
        return RDF.nil
    head = BNode()
    cursor = head
    for i, item in enumerate(items):
        if isinstance(item, (URIRef, BNode, Literal)):
            node = item
        else:
            node = URIRef(str(item)) if isinstance(item, str) and ":" in item else _spec_to_literal(item)
        g.add((cursor, RDF.first, node))
        if i == len(items) - 1:
            g.add((cursor, RDF.rest, RDF.nil))
        else:
            next_b = BNode()
            g.add((cursor, RDF.rest, next_b))
            cursor = next_b
    return head


def _emit_owl_restriction(
    g: rdflib.Graph,
    pc: KGPropertyConstraint,
    property_iri: Optional[str],
    owner_class_iris: List[str],
    default_ns: str,
) -> None:
    """Emit one ``owl:Restriction`` blank node per OWL-compatible spec."""
    if not property_iri:
        return
    for spec in pc.get_specs():
        kind = spec.get("kind")
        pred = _OWL_RESTRICTION_KINDS.get(kind)
        if pred is None:
            continue
        restr = BNode()
        g.add((restr, RDF.type, OWL.Restriction))
        g.add((restr, OWL.onProperty, URIRef(property_iri)))
        value = spec.get("value")
        on_class = spec.get("on_class")
        if kind in ("someValuesFrom", "allValuesFrom"):
            target = URIRef(str(value)) if isinstance(value, str) else _spec_to_literal(value)
            g.add((restr, pred, target))
        elif kind == "hasValue":
            target = _spec_to_literal(value) if not isinstance(value, str) or "://" not in value \
                else URIRef(value)
            g.add((restr, pred, target))
        elif kind == "hasSelf":
            g.add((restr, pred, Literal(bool(value), datatype=XSD.boolean)))
        else:
            lit = _spec_to_literal(value)
            if lit is not None:
                g.add((restr, pred, lit))
            if on_class and kind.endswith("QualifiedCardinality"):
                g.add((restr, OWL.onClass, URIRef(on_class)))
        for owner_iri in owner_class_iris:
            g.add((URIRef(owner_iri), RDFS.subClassOf, restr))


#: SHACL logical operators → predicates. shaclNot takes a single shape
#: object; the others take an ``rdf:List`` of shapes.
_SHACL_LOGICAL_KINDS: Dict[str, URIRef] = {
    "shaclNot": SH["not"],
    "shaclAnd": SH["and"],
    "shaclOr": SH["or"],
    "shaclXone": SH.xone,
}

#: Constraint kinds whose ``value`` is a list of nested shapes — the four
#: logical operators above. Used to gate vocab-skip counting at export time.
LOGICAL_KINDS: Set[str] = set(_SHACL_LOGICAL_KINDS.keys())


def _emit_inline_shape(
    g: rdflib.Graph,
    entry: Dict[str, Any],
    resolve_ref,
    inner_shape_type: URIRef,
) -> Optional[Any]:
    """Resolve one nested-shape entry to an rdflib term.

    ``entry`` is either ``{"ref": node-id}`` (pointer to an existing
    constraint node) or ``{"specs": [...]}`` (an inline anonymous shape).
    Returns the shape's term or ``None`` if the entry is unresolvable —
    callers should skip ``None`` results when building lists.
    """
    if not isinstance(entry, dict):
        return None
    if "ref" in entry and isinstance(entry["ref"], str):
        return resolve_ref(entry["ref"])
    inline_specs = entry.get("specs") or []
    bnode = BNode()
    g.add((bnode, RDF.type, inner_shape_type))
    for spec in inline_specs:
        _emit_single_shacl_constraint(g, bnode, spec, resolve_ref, inner_shape_type)
    return bnode


def _emit_single_shacl_constraint(
    g: rdflib.Graph,
    shape_subject: Any,
    spec: Dict[str, Any],
    resolve_ref,
    inner_shape_type: URIRef,
) -> None:
    """Emit a single SHACL constraint triple onto ``shape_subject``.

    Handles every kind in ``_SHACL_PROPERTY_KINDS`` (simple predicates and
    list-valued constraints) plus the four logical operators. ``resolve_ref``
    is a callable mapping constraint-node ids to rdflib terms so refs in
    nested shapes resolve to the same blank node / IRI as the top-level
    emission. ``inner_shape_type`` is used as the ``rdf:type`` of any inline
    anonymous shape emitted under a logical operator — typically
    :data:`SH.PropertyShape` because nested shapes most often constrain a
    property; switching to :data:`SH.NodeShape` is correct when the outer
    shape is a NodeShape.
    """
    kind = spec.get("kind")
    value = spec.get("value")
    on_class = spec.get("on_class")

    # Logical operators first — these recurse via _emit_inline_shape.
    if kind in _SHACL_LOGICAL_KINDS:
        pred = _SHACL_LOGICAL_KINDS[kind]
        entries = value if isinstance(value, list) else []
        shape_terms = [_emit_inline_shape(g, e, resolve_ref, inner_shape_type) for e in entries]
        shape_terms = [t for t in shape_terms if t is not None]
        if kind == "shaclNot":
            if shape_terms:
                g.add((shape_subject, pred, shape_terms[0]))
            return
        if not shape_terms:
            return
        head = _emit_rdf_list(g, shape_terms)
        g.add((shape_subject, pred, head))
        return

    # SHACL doesn't have `sh:exactCount` / `sh:qualifiedExactCount`; the spec
    # recommends emitting BOTH min+max with the same value. Handled before the
    # `_SHACL_PROPERTY_KINDS` lookup because these kinds aren't in that map.
    if kind == "exactQualifiedCardinality" and on_class:
        qv_shape = BNode()
        g.add((qv_shape, RDF.type, SH.NodeShape))
        g.add((qv_shape, SH["class"], URIRef(on_class)))
        g.add((shape_subject, SH.qualifiedValueShape, qv_shape))
        lit = _spec_to_literal(value)
        if lit is not None:
            g.add((shape_subject, SH.qualifiedMinCount, lit))
            g.add((shape_subject, SH.qualifiedMaxCount, lit))
        return
    if kind == "exactCardinality":
        lit = _spec_to_literal(value)
        if lit is not None:
            g.add((shape_subject, SH.minCount, lit))
            g.add((shape_subject, SH.maxCount, lit))
        return

    pred = _SHACL_PROPERTY_KINDS.get(kind)
    if pred is None:
        return
    if kind == "in":
        head = _emit_rdf_list(g, value or [])
        g.add((shape_subject, pred, head))
        return
    if kind == "languageIn":
        head = _emit_rdf_list(g, value or [])
        g.add((shape_subject, pred, head))
        return
    if kind in ("minQualifiedCardinality", "maxQualifiedCardinality") and on_class:
        qv_shape = BNode()
        g.add((qv_shape, RDF.type, SH.NodeShape))
        g.add((qv_shape, SH["class"], URIRef(on_class)))
        g.add((shape_subject, SH.qualifiedValueShape, qv_shape))
        lit = _spec_to_literal(value)
        if lit is not None:
            g.add((shape_subject, pred, lit))
        return
    if kind == "someValuesFrom" and isinstance(value, str):
        g.add((shape_subject, pred, URIRef(value)))
        return
    if kind in ("datatype", "nodeKind", "shaclSeverity", "shaclGroup") and isinstance(value, str):
        g.add((shape_subject, pred, URIRef(value)))
        return
    if kind == "hasValue":
        term = URIRef(value) if isinstance(value, str) and "://" in value else _spec_to_literal(value)
        g.add((shape_subject, pred, term))
        return
    lit = _spec_to_literal(value)
    if lit is not None:
        g.add((shape_subject, pred, lit))


def _emit_shacl_property_shape_into(
    g: rdflib.Graph,
    pc: KGPropertyConstraint,
    property_iri: Optional[str],
    default_ns: str,
    resolve_ref,
    *,
    shape_term: Any,
) -> Any:
    """Emit a ``sh:PropertyShape`` into ``shape_term`` (a pre-allocated rdflib
    term shared across the emit pass so cross-references resolve cleanly).
    """
    g.add((shape_term, RDF.type, SH.PropertyShape))
    if property_iri:
        g.add((shape_term, SH.path, URIRef(property_iri)))
    for spec in pc.get_specs():
        _emit_single_shacl_constraint(g, shape_term, spec, resolve_ref, SH.PropertyShape)
    return shape_term


def _emit_shacl_node_shape_into(
    g: rdflib.Graph,
    nc: KGNodeConstraint,
    target_class_iris: List[str],
    property_shapes: List[URIRef | BNode],
    default_ns: str,
    resolve_ref,
    *,
    shape_term: Any,
) -> Any:
    """Emit a ``sh:NodeShape`` into ``shape_term`` (pre-allocated)."""
    shape = shape_term
    g.add((shape, RDF.type, SH.NodeShape))
    for target_iri in target_class_iris:
        g.add((shape, SH.targetClass, URIRef(target_iri)))
    for ps in property_shapes:
        g.add((shape, SH.property, ps))
    for spec in nc.get_specs():
        kind = spec.get("kind")
        value = spec.get("value")
        # Logical operators: delegate to the shared helper. Inner inline
        # shapes default to NodeShape here (the outer is a NodeShape) so
        # nested constraints semantically match the parent's shape kind.
        if kind in _SHACL_LOGICAL_KINDS:
            _emit_single_shacl_constraint(g, shape, spec, resolve_ref, SH.NodeShape)
            continue
        pred = _SHACL_NODE_KINDS.get(kind)
        if pred is None:
            continue
        if kind == "shaclIgnoredProperties" and isinstance(value, list):
            head = _emit_rdf_list(g, [URIRef(str(v)) for v in value])
            g.add((shape, pred, head))
            continue
        if kind in ("shaclSeverity", "shaclGroup", "shaclDisjoint") and isinstance(value, str):
            g.add((shape, pred, URIRef(value)))
            continue
        lit = _spec_to_literal(value)
        if lit is not None:
            g.add((shape, pred, lit))
    return shape


def _emit_nodeconstraint_owl_axioms(
    g: rdflib.Graph,
    nc: KGNodeConstraint,
    target_class_iris: List[str],
) -> None:
    """Translate a NodeConstraint's class-axiom specs into raw OWL triples.

    These are duplicates of what the ``KnowledgeGraph.axioms`` serialiser
    would emit, kept here so node-constraint-only edits in the editor land
    in the OWL output even when the typed axiom record has been removed.
    """
    if not target_class_iris:
        return
    anchor = URIRef(target_class_iris[0])
    for spec in nc.get_specs():
        kind = spec.get("kind")
        value = spec.get("value")
        if kind == "equivalentClasses" and isinstance(value, list):
            for other in value:
                g.add((anchor, OWL.equivalentClass, URIRef(other)))
        elif kind == "disjointWith" and isinstance(value, list):
            for other in value:
                g.add((anchor, OWL.disjointWith, URIRef(other)))
        elif kind == "subClassOf" and isinstance(value, str):
            g.add((anchor, RDFS.subClassOf, URIRef(value)))
        elif kind == "complementOf" and isinstance(value, str):
            g.add((anchor, OWL.complementOf, URIRef(value)))
        elif kind == "disjointUnionOf" and isinstance(value, list):
            head = _emit_rdf_list(g, [URIRef(v) for v in value])
            g.add((anchor, OWL.disjointUnionOf, head))
        elif kind == "unionOf" and isinstance(value, list):
            head = _emit_rdf_list(g, [URIRef(v) for v in value])
            g.add((anchor, OWL.unionOf, head))
        elif kind == "intersectionOf" and isinstance(value, list):
            head = _emit_rdf_list(g, [URIRef(v) for v in value])
            g.add((anchor, OWL.intersectionOf, head))
        elif kind == "oneOf" and isinstance(value, list):
            head = _emit_rdf_list(g, [URIRef(v) for v in value])
            g.add((anchor, OWL.oneOf, head))
        elif kind == "hasKey" and isinstance(value, list):
            head = _emit_rdf_list(g, [URIRef(v) for v in value])
            g.add((anchor, OWL.hasKey, head))


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

    # Resolve constraint relationships.
    pc_to_property_iri: Dict[str, str] = {}
    pc_to_owner_classes: Dict[str, List[str]] = {}
    nc_to_target_classes: Dict[str, List[str]] = {}
    nc_to_pcs: Dict[str, List[str]] = {}
    for e in kg.edges:
        if e.iri == CONSTRAINT_TARGET_PROPERTY and isinstance(e.source, KGPropertyConstraint) \
                and isinstance(e.target, KGProperty):
            pc_to_property_iri[e.source.id] = e.target.iri or e.target.id
        elif e.iri == CONSTRAINT_TARGET_CLASS and isinstance(e.source, KGNodeConstraint) \
                and isinstance(e.target, KGClass):
            target_iri = e.target.iri or e.target.id
            nc_to_target_classes.setdefault(e.source.id, []).append(target_iri)
        elif e.iri == SH_PROPERTY and isinstance(e.source, KGNodeConstraint) \
                and isinstance(e.target, KGPropertyConstraint):
            nc_to_pcs.setdefault(e.source.id, []).append(e.target.id)
    # Derive owner classes for each PC (via wrapping NC).
    for nc_id, pc_ids in nc_to_pcs.items():
        owner_iris = nc_to_target_classes.get(nc_id, [])
        for pc_id in pc_ids:
            pc_to_owner_classes.setdefault(pc_id, []).extend(owner_iris)

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
        if edge.iri in _INTERNAL_PREDICATES:
            continue
        if isinstance(src, (KGNodeConstraint, KGPropertyConstraint)) \
                or isinstance(tgt, (KGNodeConstraint, KGPropertyConstraint)):
            # Constraint nodes are emitted via the dedicated paths below.
            continue
        s = _term_for_node(src, default_namespace)
        p = _predicate_for_edge(edge, default_namespace)
        o = _term_for_node(tgt, default_namespace)
        g.add((s, p, o))

    # Constraint emission.
    #
    # Pre-allocate one rdflib term per top-level constraint node (URIRef for
    # named, BNode for anonymous) BEFORE emitting any triples. That way a
    # logical-operator spec like ``shaclAnd: [ {ref: "pc-other"} ]`` can
    # resolve its ref to the correct shape term regardless of emission
    # order — and refs that form a cycle still terminate (each cycle node
    # already has a stable term).
    pc_to_shape_term: Dict[str, URIRef | BNode] = {}
    nc_to_shape_term: Dict[str, URIRef | BNode] = {}
    for node in kg.nodes:
        if isinstance(node, KGPropertyConstraint):
            # Mirror `_term_for_node`: stable BNode handle from node.id so
            # downstream consumers (the consistency checker) can map
            # pyshacl's `sh:sourceShape` back to this KG node.
            pc_to_shape_term[node.id] = (
                URIRef(node.iri) if node.iri else BNode(_slugify(node.id))
            )
        elif isinstance(node, KGNodeConstraint):
            nc_to_shape_term[node.id] = (
                URIRef(node.iri) if node.iri else BNode(_slugify(node.id))
            )

    def resolve_ref(node_id: str):
        return pc_to_shape_term.get(node_id) or nc_to_shape_term.get(node_id)

    if "shacl" in vocabs:
        # Emit property shapes first (they may be referenced by node shapes
        # via sh:property and by logical operators via ref).
        for node in kg.nodes:
            if not isinstance(node, KGPropertyConstraint):
                continue
            prop_iri = pc_to_property_iri.get(node.id)
            _emit_shacl_property_shape_into(
                g, node, prop_iri, default_namespace, resolve_ref,
                shape_term=pc_to_shape_term[node.id],
            )
        # Node shapes.
        for node in kg.nodes:
            if not isinstance(node, KGNodeConstraint):
                continue
            target_iris = nc_to_target_classes.get(node.id, [])
            ps_terms = [pc_to_shape_term[pc_id] for pc_id in nc_to_pcs.get(node.id, []) if pc_id in pc_to_shape_term]
            _emit_shacl_node_shape_into(
                g, node, target_iris, ps_terms, default_namespace, resolve_ref,
                shape_term=nc_to_shape_term[node.id],
            )

    if "owl" in vocabs:
        for node in kg.nodes:
            if isinstance(node, KGPropertyConstraint):
                prop_iri = pc_to_property_iri.get(node.id) or node.metadata.get("onPropertyIri")
                owner_iris = pc_to_owner_classes.get(node.id, [])
                _emit_owl_restriction(g, node, prop_iri, owner_iris, default_namespace)
            elif isinstance(node, KGNodeConstraint):
                target_iris = nc_to_target_classes.get(node.id, [])
                _emit_nodeconstraint_owl_axioms(g, node, target_iris)

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
    serialized = g.serialize(format=fmt)
    if isinstance(serialized, bytes):  # rdflib < 6 returned bytes
        return serialized.decode("utf-8")
    return serialized
