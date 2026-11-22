"""RDF writing shared by the KG exporter and the KG projection.

:mod:`~besser.BUML.notations.kg_rdf.exporter` (the user-facing Turtle / RDF-XML
export) and :mod:`~besser.BUML.notations.kg_rdf.projection` (the graph the KG ->
B-UML conversion and the consistency check reason over) write the same
constructs, so everything that decides *how* a construct is written lives here
once: node terms, the spec-kind -> predicate tables, the leaf emitters for OWL
restrictions, class axioms, class expressions and SHACL components, the
constraint indexes, and the rebuilding of the lists the importer flattened.
``tests/BUML/notations/kg_rdf/test_projection_contract.py`` pins where the two
writers may still differ.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

import rdflib
from rdflib import BNode, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGEdge,
    KGLiteral,
    KGNode,
    KGNodeConstraint,
    KGPropertyConstraint,
    PropertyChainAxiom,
)
from besser.BUML.metamodel.kg.constants import (
    BESSER_KG_NS,
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH as SH_NS,
    SH_PATH,
    SH_PROPERTY,
    SH_TARGET_CLASS,
)
from besser.BUML.metamodel.kg.constraint_specs import (
    LOGICAL_KINDS,
    OWL_RESTRICTION_PREDICATES,
    SHACL_NODE_COMPONENTS,
    SHACL_PROPERTY_COMPONENTS,
    spec_folded_predicates,
)

__all__ = [
    "DEFAULT_NAMESPACE",
    "CLASS_EXPRESSION_PREDICATES",
    "FLATTENED_LIST_PREDICATES",
    "OWL_RESTRICTION_KINDS",
    "SHACL_LOGICAL_KINDS",
    "SHACL_NODE_KINDS",
    "SHACL_PROPERTY_KINDS",
    "BlankHandles",
    "ConstraintIndexes",
    "blank_handles",
    "build_constraint_indexes",
    "emit_flattened_lists",
    "emit_nodeconstraint_owl_axioms",
    "emit_owl_class_expression",
    "emit_owl_restriction",
    "emit_property_chains",
    "emit_rdf_list",
    "emit_shacl_constraint",
    "emit_shacl_node_constraint",
    "natural_key",
    "predicate_for_edge",
    "slugify",
    "spec_folded_by_term",
    "spec_to_literal",
    "term_for_node",
]


#: Base for the IRIs the writers mint: BESSER's one namespace.
DEFAULT_NAMESPACE = BESSER_KG_NS
SH = Namespace(SH_NS)

logger = logging.getLogger(__name__)


#: OWL list-valued predicates the importer *flattens*: it sweeps the
#: ``rdf:first``/``rdf:rest`` spine and leaves one direct subject->member edge per
#: member, so the canvas shows the operands. Re-emitting those edges as they are
#: would give the predicate several objects instead of one list head, so both
#: writers skip them in their edge loop and rebuild each list with
#: :func:`emit_flattened_lists`.
FLATTENED_LIST_PREDICATES: FrozenSet[str] = frozenset({
    str(OWL.unionOf), str(OWL.intersectionOf), str(OWL.oneOf),
    str(OWL.hasKey), str(OWL.disjointUnionOf), str(OWL.propertyChainAxiom),
    str(OWL.members),
})

#: Flattened lists whose member order carries no meaning: a key is a set of
#: properties, ``owl:members`` a set of pairwise-disjoint terms. They are
#: rebuilt in member-id order, which is canonical whatever the edge ids are.
_UNORDERED_LIST_PREDICATES: FrozenSet[str] = frozenset({str(OWL.hasKey), str(OWL.members)})


#: Spec kind -> predicate, for each vocabulary. Derived from the metamodel's
#: predicate -> kind tables (:mod:`besser.BUML.metamodel.kg.constraint_specs`),
#: which the importer reads with, so the two directions cannot drift.
OWL_RESTRICTION_KINDS: Dict[str, URIRef] = {
    kind: URIRef(predicate) for kind, predicate in OWL_RESTRICTION_PREDICATES.items()
}

# sh:class means "every value must be of this type" (forall): it pairs with
# allValuesFrom, not someValuesFrom (exists). SHACL has no faithful
# existential-quantification predicate, so someValuesFrom is owl-only. The
# qualified counts are written (with a sh:qualifiedValueShape) but not read
# into specs.
SHACL_PROPERTY_KINDS: Dict[str, URIRef] = {
    **{kind: URIRef(predicate) for predicate, kind in SHACL_PROPERTY_COMPONENTS.items()},
    "minQualifiedCardinality": SH.qualifiedMinCount,
    "maxQualifiedCardinality": SH.qualifiedMaxCount,
}

SHACL_NODE_KINDS: Dict[str, URIRef] = {
    kind: URIRef(predicate) for predicate, kind in SHACL_NODE_COMPONENTS.items()
}


def slugify(text: str) -> str:
    if not text:
        return "node"
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
    return cleaned or "node"


def term_for_node(node: KGNode, default_ns: str) -> Union[URIRef, BNode, Literal]:
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
        return BNode(slugify(node.id))
    if node.iri:
        return URIRef(node.iri)
    return URIRef(default_ns + slugify(node.label or node.id))


def predicate_for_edge(edge: KGEdge, default_ns: str) -> URIRef:
    if edge.iri:
        return URIRef(edge.iri)
    return URIRef(default_ns + slugify(edge.label or "relatedTo"))


_NUM_RE = re.compile(r"(\d+)")


def natural_key(text: str) -> Tuple:
    """Sort key that orders embedded integers numerically.

    Editor-made ids can look like ``edge:9`` / ``edge:10``; plain lexicographic
    order would put ``edge:10`` first and silently reorder a flattened RDF list.
    """
    return tuple(int(part) if part.isdigit() else part for part in _NUM_RE.split(text or ""))


#: A spec's name for an anonymous node -> that node's RDF term; see :func:`blank_handles`.
BlankHandles = Dict[str, Any]


def blank_handles(nodes: Iterable[KGNode], terms: Dict[str, Any]) -> BlankHandles:
    """Map the bare label a spec uses for an anonymous node to that node's term.

    A spec stores an anonymous class it refers to (a union filler, a restriction
    inside an intersection) by its blank-node label, the node id without
    ``_:``. That node may still be a ``KGBlank`` or may have been lifted into a
    constraint node; either way the emitters must write the node's own term, not
    a relative IRI (or a literal) that nothing else in the graph mentions.
    """
    out: BlankHandles = {}
    for node in nodes:
        term = terms.get(node.id)
        if not isinstance(term, BNode):
            continue
        if node.id.startswith("_:"):
            out[node.id[2:]] = term
        elif isinstance(node, KGBlank):
            out[node.id] = term
    return out


def spec_folded_by_term(nodes: Iterable[KGNode], terms: Dict[str, Any]) -> Dict[Any, FrozenSet[str]]:
    """Per constraint-node term, the predicates whose value lives in its specs.

    Keyed by *term*, not node: a punned shape shares its IRI with a class, and a
    parameter written on the class node (``ex:Person sh:closed true``) is the
    shape's parameter all the same. An edge with such a predicate on such a
    subject is never written as it is: the spec is.
    """
    out: Dict[Any, Set[str]] = {}
    for node in nodes:
        predicates = spec_folded_predicates(node)
        term = terms.get(node.id)
        if predicates and term is not None and not isinstance(term, Literal):
            out.setdefault(term, set()).update(predicates)
    return {term: frozenset(predicates) for term, predicates in out.items()}


class ConstraintIndexes:
    """Which classes and properties each constraint node applies to, as RDF terms."""

    def __init__(self) -> None:
        #: Node constraint id -> the classes it targets, in edge order.
        self.nc_targets: Dict[str, List[Any]] = {}
        #: Property constraint id -> the property it constrains.
        self.pc_property: Dict[str, Any] = {}
        #: Property constraint id -> the object of its raw ``owl:onProperty`` edge.
        #: The importer records a property IRI as ``onPropertyIri`` too, but an
        #: anonymous property expression (``[ owl:inverseOf :p ]``) only as this
        #: edge, to the expression's blank node.
        self.pc_on_property: Dict[str, Any] = {}
        #: Node constraint id -> the property constraints it lists (``sh:property``).
        self.nc_pcs: Dict[str, List[str]] = {}
        #: Node constraints with a target that is no usable RDF subject (a literal).
        self.nc_unresolved_targets: Set[str] = set()

    def restricted_property(self, pc: KGPropertyConstraint) -> Optional[Any]:
        """The property an OWL restriction of ``pc`` is ``owl:onProperty``, as an IRI string or term.

        The editor wiring and ``onPropertyIri`` name it when it is a property;
        the raw edge is the fallback, so an inverse property expression is not
        dropped for having no IRI.
        """
        return (
            self.pc_property.get(pc.id)
            or pc.metadata.get("onPropertyIri")
            or self.pc_on_property.get(pc.id)
        )

    def pc_owners(self) -> Dict[str, List[Any]]:
        """Property constraint id -> the classes of the node constraints listing it."""
        owners: Dict[str, List[Any]] = {}
        for nc_id, pc_ids in self.nc_pcs.items():
            for pc_id in pc_ids:
                owners.setdefault(pc_id, []).extend(self.nc_targets.get(nc_id, []))
        return owners


def build_constraint_indexes(edges: Iterable[KGEdge], terms: Dict[str, Any]) -> ConstraintIndexes:
    """Index the constraint wiring of a KG, for the exporter and the projection alike.

    Both the editor wiring (``constraintTargetClass`` / ``constraintTargetProperty``)
    and the raw SHACL edges (``sh:targetClass`` / ``sh:path``) are read: the
    importer only adds the wiring when the target is a *declared* class or
    property, and a graph drawn in the editor may only have the SHACL edges.
    When a property constraint has both, ``constraintTargetProperty`` wins: it is
    what the editor lets the user change. Raw ``owl:onProperty`` edges are kept
    apart, in ``pc_on_property``: they are the OWL restrictions' fallback only
    (:meth:`ConstraintIndexes.restricted_property`), never a SHACL path. Edges
    are read in id order, so the result does not depend on set iteration order.
    """
    idx = ConstraintIndexes()
    targeted: Set[str] = set()
    for edge in sorted(edges, key=lambda e: natural_key(e.id)):
        # ``str`` because rdflib term equality is type-sensitive: a URIRef never
        # compares equal to the plain string an editor payload carries.
        predicate = str(edge.iri or "")
        source, target = edge.source, edge.target
        term = terms.get(target.id)
        usable = term is not None and not isinstance(term, Literal)
        if predicate in (CONSTRAINT_TARGET_CLASS, SH_TARGET_CLASS) and isinstance(source, KGNodeConstraint):
            if not usable:
                idx.nc_unresolved_targets.add(source.id)
            elif term not in idx.nc_targets.setdefault(source.id, []):
                idx.nc_targets[source.id].append(term)
        elif predicate in (CONSTRAINT_TARGET_PROPERTY, SH_PATH) and isinstance(source, KGPropertyConstraint):
            if usable and (predicate == CONSTRAINT_TARGET_PROPERTY or source.id not in targeted):
                if predicate == CONSTRAINT_TARGET_PROPERTY:
                    if source.id in targeted:
                        continue
                    targeted.add(source.id)
                idx.pc_property[source.id] = term
        elif predicate == str(OWL.onProperty) and isinstance(source, KGPropertyConstraint):
            if usable:
                idx.pc_on_property.setdefault(source.id, term)
        elif predicate == SH_PROPERTY and isinstance(source, KGNodeConstraint) \
                and isinstance(target, KGPropertyConstraint):
            if target.id not in idx.nc_pcs.setdefault(source.id, []):
                idx.nc_pcs[source.id].append(target.id)
    return idx


def spec_to_literal(value: Any) -> Optional[Literal]:
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


def _as_term(value: Any, handles: Optional[BlankHandles] = None) -> Any:
    """The RDF term a spec value names: a term as is, a blank-node label as that
    blank node, a string with a ``:`` as an IRI, anything else as a literal
    (``{"value", "datatype", "language"}`` dicts keep their datatype)."""
    if isinstance(value, (URIRef, BNode, Literal)):
        return value
    if isinstance(value, str):
        if handles and value in handles:
            return handles[value]
        if ":" in value:
            return URIRef(value)
    return spec_to_literal(value)


def emit_rdf_list(g: rdflib.Graph, items: Iterable[Any], handles: Optional[BlankHandles] = None) -> BNode:
    """Emit an ``rdf:List`` for an iterable of items, returning the list head.

    Each item becomes a term as :func:`_as_term` decides, so an ``owl:oneOf`` of
    literals stays one of literals and a member that is a blank node (``handles``
    maps the KG's blank-node labels to their terms) stays connected to that node.
    """
    items = list(items)
    if not items:
        return RDF.nil
    head = BNode()
    cursor = head
    for i, item in enumerate(items):
        node = _as_term(item, handles)
        g.add((cursor, RDF.first, node))
        if i == len(items) - 1:
            g.add((cursor, RDF.rest, RDF.nil))
        else:
            next_b = BNode()
            g.add((cursor, RDF.rest, next_b))
            cursor = next_b
    return head


def _emit_restriction_value(
    g: rdflib.Graph, restriction, predicate: URIRef, spec: dict,
    blank_handles: Optional[BlankHandles] = None,
) -> None:
    """Add the value triple(s) of one OWL restriction spec to ``restriction``.

    The kind-dispatch half of restriction emission, shared by the TTL exporter
    and the KG -> UML conversion projection (``kg_to_rdf``), which used to carry
    a copy each. A spec whose value cannot be rendered as an RDF term (for
    example a ``hasValue`` with no value) emits nothing, rather than handing
    ``None`` to ``Graph.add``, which raises ``AssertionError``.

    ``blank_handles`` comes from :func:`blank_handles`. A spec stores its filler
    as a plain string, so a restriction over an anonymous class (a union, say)
    carries that blank node's bare label; rebuilding it as ``URIRef(label)``
    produced a relative IRI that nothing in the graph mentions. Such a filler is
    mapped back to the node it names.
    """
    handles = blank_handles or {}
    kind = spec.get("kind")
    value = spec.get("value")
    on_class = spec.get("on_class")
    if kind in ("someValuesFrom", "allValuesFrom"):
        if isinstance(value, str) and value in handles:
            target = handles[value]
        else:
            target = URIRef(str(value)) if isinstance(value, str) else spec_to_literal(value)
        if target is not None:
            g.add((restriction, predicate, target))
    elif kind == "hasValue":
        target = URIRef(value) if isinstance(value, str) and "://" in value else spec_to_literal(value)
        if target is not None:
            g.add((restriction, predicate, target))
    elif kind == "hasSelf":
        g.add((restriction, predicate, Literal(bool(value), datatype=XSD.boolean)))
    else:
        literal = spec_to_literal(value)
        if literal is not None:
            g.add((restriction, predicate, literal))
        if on_class and str(kind).endswith("QualifiedCardinality"):
            on_class_term = handles.get(on_class, URIRef(on_class))
            g.add((restriction, OWL.onClass, on_class_term))


def emit_owl_restriction(
    g: rdflib.Graph,
    pc: KGPropertyConstraint,
    property_iri: Optional[Any],
    owner_class_iris: List[Any],
    *,
    subject=None,
    blank_handles: Optional[BlankHandles] = None,
) -> None:
    """Emit an ``owl:Restriction`` per OWL-compatible spec of ``pc``.

    Args:
        property_iri: The restricted property, as an IRI string or term.
        owner_class_iris: The classes made ``rdfs:subClassOf`` each restriction,
            as IRI strings or terms (a blank node stays one).
        subject: The term to use as the restriction node. By default a fresh
            ``BNode`` is minted per spec, which is right for the exporter. The
            conversion projection passes the constraint node's *own* term,
            because that node already is the original restriction and reusing it
            keeps the ``rdfs:subClassOf`` link emitted elsewhere valid.
        blank_handles: The KG's anonymous nodes by label; see :func:`blank_handles`.
    """
    if not property_iri:
        return
    for spec in pc.get_specs():
        kind = spec.get("kind")
        pred = OWL_RESTRICTION_KINDS.get(kind)
        if pred is None:
            continue
        restr = subject if subject is not None else BNode()
        g.add((restr, RDF.type, OWL.Restriction))
        g.add((restr, OWL.onProperty, _as_term(property_iri)))
        _emit_restriction_value(g, restr, pred, spec, blank_handles)
        for owner in owner_class_iris:
            g.add((_as_term(owner), RDFS.subClassOf, restr))


#: SHACL logical operators → predicates. shaclNot takes a single shape
#: object; the others take an ``rdf:List`` of shapes.
SHACL_LOGICAL_KINDS: Dict[str, URIRef] = {
    "shaclNot": SH["not"],
    "shaclAnd": SH["and"],
    "shaclOr": SH["or"],
    "shaclXone": SH.xone,
}

# The metamodel owns the set of logical-operator kinds (the four above); this
# module only maps each to its SHACL predicate. Fail at import, not at export
# time, if the two ever disagree.
if LOGICAL_KINDS != set(SHACL_LOGICAL_KINDS):
    raise RuntimeError("LOGICAL_KINDS drifted from SHACL_LOGICAL_KINDS")


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
        emit_shacl_constraint(g, bnode, spec, resolve_ref, inner_shape_type)
    return bnode


def emit_shacl_constraint(
    g: rdflib.Graph,
    shape_subject: Any,
    spec: Dict[str, Any],
    resolve_ref,
    inner_shape_type: URIRef,
) -> None:
    """Emit a single SHACL constraint triple onto ``shape_subject``.

    Handles every kind in ``SHACL_PROPERTY_KINDS`` (simple predicates and
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
    if kind in SHACL_LOGICAL_KINDS:
        pred = SHACL_LOGICAL_KINDS[kind]
        entries = value if isinstance(value, list) else []
        shape_terms = [_emit_inline_shape(g, e, resolve_ref, inner_shape_type) for e in entries]
        shape_terms = [t for t in shape_terms if t is not None]
        if kind == "shaclNot":
            if shape_terms:
                g.add((shape_subject, pred, shape_terms[0]))
            return
        if not shape_terms:
            return
        head = emit_rdf_list(g, shape_terms)
        g.add((shape_subject, pred, head))
        return

    # SHACL doesn't have `sh:exactCount` / `sh:qualifiedExactCount`; the spec
    # recommends emitting BOTH min+max with the same value. Handled before the
    # `SHACL_PROPERTY_KINDS` lookup because these kinds aren't in that map.
    if kind == "exactQualifiedCardinality" and on_class:
        qv_shape = BNode()
        g.add((qv_shape, RDF.type, SH.NodeShape))
        g.add((qv_shape, SH["class"], URIRef(on_class)))
        g.add((shape_subject, SH.qualifiedValueShape, qv_shape))
        lit = spec_to_literal(value)
        if lit is not None:
            g.add((shape_subject, SH.qualifiedMinCount, lit))
            g.add((shape_subject, SH.qualifiedMaxCount, lit))
        return
    if kind == "exactCardinality":
        lit = spec_to_literal(value)
        if lit is not None:
            g.add((shape_subject, SH.minCount, lit))
            g.add((shape_subject, SH.maxCount, lit))
        return

    pred = SHACL_PROPERTY_KINDS.get(kind)
    if pred is None:
        return
    if kind == "in":
        head = emit_rdf_list(g, value or [])
        g.add((shape_subject, pred, head))
        return
    if kind == "languageIn":
        head = emit_rdf_list(g, value or [])
        g.add((shape_subject, pred, head))
        return
    if kind in ("minQualifiedCardinality", "maxQualifiedCardinality") and on_class:
        qv_shape = BNode()
        g.add((qv_shape, RDF.type, SH.NodeShape))
        g.add((qv_shape, SH["class"], URIRef(on_class)))
        g.add((shape_subject, SH.qualifiedValueShape, qv_shape))
        lit = spec_to_literal(value)
        if lit is not None:
            g.add((shape_subject, pred, lit))
        return
    if kind in (
        "allValuesFrom", "datatype", "nodeKind", "shaclSeverity", "shaclGroup",
        "equals", "lessThan", "lessThanOrEquals",
    ) and isinstance(value, str):
        g.add((shape_subject, pred, URIRef(value)))
        return
    if kind == "hasValue":
        term = URIRef(value) if isinstance(value, str) and "://" in value else spec_to_literal(value)
        g.add((shape_subject, pred, term))
        return
    lit = spec_to_literal(value)
    if lit is not None:
        g.add((shape_subject, pred, lit))


def emit_shacl_node_constraint(
    g: rdflib.Graph,
    shape: Any,
    spec: Dict[str, Any],
    resolve_ref,
) -> bool:
    """Emit one node-shape-level spec (a ``SHACL_NODE_KINDS`` kind or a logical
    operator) onto ``shape``. Returns ``False`` for any other kind, emitting nothing.
    """
    kind = spec.get("kind")
    value = spec.get("value")
    # Logical operators: delegate to the shared helper. Inner inline
    # shapes default to NodeShape here (the outer is a NodeShape) so
    # nested constraints semantically match the parent's shape kind.
    if kind in SHACL_LOGICAL_KINDS:
        emit_shacl_constraint(g, shape, spec, resolve_ref, SH.NodeShape)
        return True
    pred = SHACL_NODE_KINDS.get(kind)
    if pred is None:
        return False
    if kind == "shaclIgnoredProperties" and isinstance(value, list):
        head = emit_rdf_list(g, [URIRef(str(v)) for v in value])
        g.add((shape, pred, head))
        return True
    if kind in ("shaclSeverity", "shaclGroup", "shaclDisjoint") and isinstance(value, str):
        g.add((shape, pred, URIRef(value)))
        return True
    lit = spec_to_literal(value)
    if lit is not None:
        g.add((shape, pred, lit))
    return True


#: Class-axiom spec kinds -> the OWL predicate they are written with.
_CLASS_AXIOM_PREDICATES: Dict[str, URIRef] = {
    "equivalentClasses": OWL.equivalentClass,
    "disjointWith": OWL.disjointWith,
    "subClassOf": RDFS.subClassOf,
    "disjointUnionOf": OWL.disjointUnionOf,
    "hasKey": OWL.hasKey,
}
_LIST_AXIOM_KINDS = frozenset({"disjointUnionOf", "hasKey"})


def emit_nodeconstraint_owl_axioms(
    g: rdflib.Graph,
    nc: KGNodeConstraint,
    target_class_iris: List[Any],
    handles: Optional[BlankHandles] = None,
) -> None:
    """Write a node constraint's class-axiom specs as OWL triples on its first target.

    The spec is the copy the editor edits, so it is what gets written; the
    ``KnowledgeGraph.axioms`` records the importer also keeps are not serialised.
    ``handles`` maps the KG's blank-node labels to their terms (:func:`blank_handles`).
    """
    if not target_class_iris:
        return
    anchor = _as_term(target_class_iris[0])
    for spec in nc.get_specs():
        kind = spec.get("kind")
        value = spec.get("value")
        predicate = _CLASS_AXIOM_PREDICATES.get(kind)
        if predicate is None:
            continue
        if kind in _LIST_AXIOM_KINDS:
            if isinstance(value, list):
                g.add((anchor, predicate, emit_rdf_list(g, value, handles)))
        elif isinstance(value, list):
            for other in value:
                g.add((anchor, predicate, _as_term(other, handles)))
        elif isinstance(value, str):
            g.add((anchor, predicate, _as_term(value, handles)))


#: Spec kinds of an anonymous class expression -> their OWL predicate.
CLASS_EXPRESSION_PREDICATES: Dict[str, URIRef] = {
    "unionOf": OWL.unionOf,
    "intersectionOf": OWL.intersectionOf,
    "oneOf": OWL.oneOf,
    "complementOf": OWL.complementOf,
}


def emit_owl_class_expression(
    g: rdflib.Graph, nc: KGNodeConstraint, subject: Any, handles: Optional[BlankHandles] = None,
) -> None:
    """Write a node constraint's class-expression specs onto ``subject``.

    ``[ a owl:Class ; owl:oneOf ("red" "green") ]`` lifted into a constraint
    node keeps its operands in the spec only, so this is where the expression
    comes back from. :func:`emit_rdf_list` decides literal vs IRI per member,
    which keeps a ``oneOf`` of literals from turning into one of IRIs.
    """
    for spec in nc.get_specs():
        kind = spec.get("kind")
        value = spec.get("value")
        if kind == "complementOf" and isinstance(value, str):
            g.add((subject, OWL.complementOf, _as_term(value, handles)))
        elif kind in CLASS_EXPRESSION_PREDICATES and isinstance(value, list) and value:
            g.add((subject, RDF.type, OWL.Class))
            g.add((subject, CLASS_EXPRESSION_PREDICATES[kind], emit_rdf_list(g, value, handles)))


def _member_key(node: KGNode) -> str:
    """The string the importer recorded for ``node`` in a blank's ``metadata['members']``:
    an IRI for named terms, the lexical form for literals, the label for blank nodes."""
    if isinstance(node, KGLiteral):
        return str(node.value)
    return node.iri or node.label or node.id


def emit_flattened_lists(
    g: rdflib.Graph,
    nodes: Iterable[KGNode],
    edges: Iterable[KGEdge],
    terms: Dict[str, Any],
    folded: Dict[Any, FrozenSet[str]],
    axioms: Iterable[Any],
    on_inferred_order: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Rebuild one ``rdf:List`` per ``(subject, predicate)`` the importer flattened.

    The member *terms* come from the edges, so a literal member of a datatype
    ``owl:oneOf`` stays a literal with its datatype; the member *order* comes
    from the blank node's ``metadata['members']`` when the importer recorded
    it. Unordered lists (keys, ``owl:members``) are rebuilt in member-id order;
    any other list falls back to edge-id order and ``on_inferred_order(node_id,
    predicate)`` is told, since operand order shows in auxiliary class names.

    Skipped: lists a spec holds (``folded`` for the subject's term, see
    :func:`spec_folded_by_term`), and property chains, which come whole and in
    order (repeats included) from their :class:`PropertyChainAxiom` record
    instead (:func:`emit_property_chains`).
    """
    chained = {
        a.property_id for a in axioms if isinstance(a, PropertyChainAxiom) and a.property_id
    }
    recorded: Dict[tuple, List[str]] = {}
    for node in nodes:
        combinator, members = node.metadata.get("combinator"), node.metadata.get("members")
        if isinstance(node, KGBlank) and combinator and isinstance(members, list):
            recorded[(node.id, str(OWL) + combinator)] = [str(m) for m in members]

    groups: Dict[tuple, List[KGEdge]] = {}
    for edge in edges:
        predicate = str(edge.iri or "")
        if predicate not in FLATTENED_LIST_PREDICATES:
            continue
        subject = terms.get(edge.source.id)
        if subject is None or isinstance(subject, Literal) or predicate in folded.get(subject, ()):
            continue
        if predicate == str(OWL.propertyChainAxiom) and edge.source.id in chained:
            continue
        groups.setdefault((edge.source.id, predicate), []).append(edge)

    for (node_id, predicate), group in sorted(groups.items()):
        order = recorded.get((node_id, predicate))
        if order is not None:
            rank = {key: index for index, key in enumerate(order)}
            group.sort(key=lambda e: (rank.get(_member_key(e.target), len(rank)), natural_key(e.id)))
        elif predicate in _UNORDERED_LIST_PREDICATES:
            group.sort(key=lambda e: (natural_key(e.target.id), natural_key(e.id)))
        else:
            group.sort(key=lambda e: (natural_key(e.id), natural_key(e.target.id)))
            if on_inferred_order is not None:
                on_inferred_order(node_id, predicate)
        members = [terms[e.target.id] for e in group if terms.get(e.target.id) is not None]
        if members:
            g.add((terms[node_id], URIRef(predicate), emit_rdf_list(g, members)))


def emit_property_chains(g: rdflib.Graph, axioms: Iterable[Any], terms: Dict[str, Any]) -> None:
    """One ``owl:propertyChainAxiom`` list per :class:`PropertyChainAxiom` record.

    The flattened member edges cannot carry a chain: a KG holds one edge per
    ``(subject, member, predicate)``, so ``hasParent o hasParent`` would come back
    as ``(hasParent)``, and two chains on one property would merge. A property
    may have several chains; each record adds its own triple.
    """
    for axiom in axioms:
        if isinstance(axiom, PropertyChainAxiom) and axiom.property_id and axiom.chain_property_ids:
            subject = terms.get(axiom.property_id, URIRef(str(axiom.property_id)))
            members = [URIRef(str(m)) for m in axiom.chain_property_ids]
            g.add((subject, OWL.propertyChainAxiom, emit_rdf_list(g, members)))
