"""Project a :class:`KnowledgeGraph` into an in-memory :class:`rdflib.Graph`.

This is the entry stage of the KG → UML pipeline. The paper's transformation
rules (:mod:`besser.BUML.notations.kg_to_buml.owl2uml`) are defined over RDF
triples, while the KG metamodel stores a triple store *plus* an editor-oriented
"constraint node" view layered on top. This module reconciles the two.

Why not reuse :func:`besser.utilities.kg_to_owl.knowledge_graph_to_rdf`: that
function is the *user-facing TTL exporter*. It deliberately skips every edge
touching a constraint node and re-derives the constraint structure from specs,
which is right for a clean export but lossy as a conversion input: on a graph
that uses the SHACL punning idiom, exporting and re-reading loses
``rdfs:subClassOf``, ``rdfs:domain``, ``rdfs:range``, ``sh:targetClass`` and
``owl:disjointWith`` triples on the punned nodes. This module instead treats
the **raw edges as authoritative** and uses the constraint specs only to fill
the gaps the importer left behind. It does reuse the exporter's leaf emitters
and vocabulary tables, so the two paths cannot drift.
``test_kg_to_rdf.py`` pins the parity that matters, on both a synthetic graph
and the BIBO fixture.

Four passes:

``A`` raw edges → triples (skipping internal wiring and flattened list spines)
``B`` re-collect flattened ``rdf:List`` values in their original order
``C`` restore the structural links the constraint-node lifter consumed
``D`` back-fill triples from ``constraintSpecs`` where no raw edge exists

Passes A–C reconstruct an imported graph; pass D additionally covers constraints
authored directly in the KG editor, which never had raw edges at all.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import rdflib
from rdflib import Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from besser.BUML.metamodel.kg import (
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
    SH_PATH,
    SH_PROPERTY,
    SH_TARGET_CLASS,
)
from besser.utilities.kg_to_owl import (
    DEFAULT_NAMESPACE,
    OWL_RESTRICTION_KINDS,
    SHACL_LOGICAL_KINDS,
    SHACL_NODE_KINDS,
    SHACL_PROPERTY_KINDS,
    emit_nodeconstraint_owl_axioms,
    emit_rdf_list,
    emit_shacl_constraint,
    predicate_for_edge,
    term_for_node,
)

from besser.BUML.notations.kg_to_buml._common import KGConversionWarning, add_warning

SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
_OWL_NS = "http://www.w3.org/2002/07/owl#"

__all__ = ["kg_to_rdf", "DEFAULT_NAMESPACE"]


#: Editor-only wiring. These predicates express "this constraint node applies to
#: that class/property" and have no RDF counterpart — pass C turns them back
#: into the ``rdfs:subClassOf`` / ``sh:targetClass`` / ``sh:path`` triples the
#: source actually contained.
_INTERNAL_PREDICATES = frozenset({CONSTRAINT_TARGET_CLASS, CONSTRAINT_TARGET_PROPERTY})

#: OWL list-valued predicates the importer *flattens*: it sweeps the
#: ``rdf:first``/``rdf:rest`` spine and leaves one direct subject→member edge per
#: member, so the graph shows the operands on the canvas. Re-emitting those flat
#: would give a predicate several objects instead of one list head, so pass A
#: skips them and pass B rebuilds the spine.
_FLATTENED_LIST_PREDICATES = frozenset({
    str(OWL.unionOf), str(OWL.intersectionOf), str(OWL.oneOf),
    str(OWL.hasKey), str(OWL.disjointUnionOf), str(OWL.propertyChainAxiom),
    str(OWL.members),
})

#: SHACL list-valued predicates the importer *consumes*: the spine is deleted
#: and the operands move into ``constraintSpecs`` with no replacement edge. The
#: surviving edge points at a now-empty list head, so pass A must skip it and
#: pass B must not try to rebuild it — pass D re-emits the whole thing from the
#: specs instead.
_CONSUMED_LIST_PREDICATES = frozenset({
    str(SH["and"]), str(SH["or"]), str(SH.xone), str(SH["in"]),
    str(SH.languageIn), str(SH.ignoredProperties),
})

_LIST_VALUED_PREDICATES = _FLATTENED_LIST_PREDICATES | _CONSUMED_LIST_PREDICATES

#: Spec kinds that describe an anonymous class expression rather than a
#: constraint on an existing class (D17-D21, O06).
_CLASS_EXPRESSION_KINDS = frozenset({"unionOf", "intersectionOf", "oneOf", "complementOf"})

#: Spec kinds that are class-level OWL axioms, anchored on the target class.
_CLASS_AXIOM_KINDS = frozenset({
    "equivalentClasses", "disjointWith", "subClassOf",
    "disjointUnionOf", "hasKey",
})

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(text: str) -> Tuple:
    """Sort key that orders embedded integers numerically.

    Edge ids look like ``cedge:9`` / ``cedge:10``; plain lexicographic order
    would put ``cedge:10`` first and silently reorder a flattened RDF list.
    """
    return tuple(int(part) if part.isdigit() else part for part in _NUM_RE.split(text or ""))


def _is_owlish(node: KGNode) -> bool:
    """True unless the constraint node is explicitly SHACL-sourced.

    Mirrors the importer's tagging: constraint nodes lifted from OWL are marked
    ``"owl"`` / ``"owl-axiom"``, those lifted from SHACL shapes ``"shacl"``. A
    node with no tag at all was hand-authored in the editor, where an
    OWL-restriction payload is the only thing it could be.
    """
    return node.metadata.get("source") != "shacl"


def kg_to_rdf(
    kg: KnowledgeGraph,
    *,
    default_namespace: str = DEFAULT_NAMESPACE,
    warnings: Optional[List[KGConversionWarning]] = None,
) -> Tuple[rdflib.Graph, Optional[str]]:
    """Project ``kg`` into a single graph carrying both OWL axioms and SHACL shapes.

    Returns ``(graph, base_namespace)``. The same graph should be passed as both
    the ontology and the shapes argument of
    :func:`~besser.BUML.notations.kg_to_buml.owl2uml.build_uml_model` — the
    SHACL phase only looks at ``sh:NodeShape`` subjects, so the two vocabularies
    coexist without interfering.
    """
    warnings = warnings if warnings is not None else []
    g = rdflib.Graph()

    nodes = sorted(kg.nodes, key=lambda n: _natural_key(n.id))
    edges = sorted(kg.edges, key=lambda e: _natural_key(e.id))
    terms: Dict[str, Any] = {n.id: term_for_node(n, default_namespace) for n in nodes}

    _pass_a_raw_edges(g, edges, terms, default_namespace, warnings)
    _pass_a2_declarations(g, nodes, edges, terms, default_namespace, warnings)
    _pass_b_lists(g, nodes, edges, terms, default_namespace, warnings)
    indexes = _build_constraint_indexes(edges, terms)
    _pass_c_constraint_structure(g, nodes, terms, indexes, warnings)
    _pass_d_spec_backfill(g, nodes, terms, indexes, default_namespace, warnings)

    return g, _base_namespace(g)


# ----------------------------------------------------------------------
# Pass A — raw edges
# ----------------------------------------------------------------------


def _pass_a_raw_edges(g, edges, terms, default_ns, warnings) -> None:
    """Emit one triple per edge. The raw edges are the authoritative layer."""
    for edge in edges:
        predicate = str(predicate_for_edge(edge, default_ns))
        if predicate in _INTERNAL_PREDICATES or predicate in _LIST_VALUED_PREDICATES:
            continue
        if predicate == SH_PROPERTY and _is_owlish(edge.source):
            # The importer reuses `sh:property` to wire an OWL restriction to
            # the wrapper node that owns it. That is internal bookkeeping, not a
            # SHACL shape — pass C turns it into the `rdfs:subClassOf` the source
            # actually had.
            continue
        subject = terms.get(edge.source.id)
        if isinstance(subject, Literal):
            add_warning(
                warnings,
                "KG_LITERAL_SUBJECT",
                f"Edge {edge.id!r} has a literal subject, which is not valid RDF; skipped.",
                edge_id=edge.id,
            )
            continue
        target = terms.get(edge.target.id)
        if subject is None or target is None:
            continue
        g.add((subject, URIRef(predicate), target))


# ----------------------------------------------------------------------
# Pass A2 — declaration synthesis
# ----------------------------------------------------------------------


def _pass_a2_declarations(g, nodes, edges, terms, default_ns, warnings) -> None:
    """Declare nodes the KG types but the triples do not.

    The paper's rules are anchored on OWL declarations: a class exists because
    of ``rdf:type owl:Class``, an association because of ``owl:ObjectProperty``,
    an attribute because of ``owl:DatatypeProperty``. A KG built in the editor
    has none of those triples — it carries the same information in the *node
    types* (``KGClass``, ``KGProperty``, ``KGIndividual``) instead. Without this
    pass, a hand-drawn graph converts to a single empty class.

    Only nodes with no OWL/RDFS ``rdf:type`` of their own are declared, so an
    imported ontology is untouched and classes it merely *references* keep being
    reported as external stubs. A partially-edited graph — import a TTL, then
    add a class in the UI — gets exactly the missing declarations.
    """
    typed: Dict[str, bool] = {}
    for edge in edges:
        if str(edge.iri or "") != str(RDF.type):
            continue
        target_iri = getattr(edge.target, "iri", None) or ""
        if target_iri.startswith(_OWL_NS) or target_iri.startswith(str(RDFS)):
            typed[edge.source.id] = True

    ranges = _range_index(edges)
    for node in nodes:
        if typed.get(node.id) or not node.iri:
            continue
        term = terms.get(node.id)
        if not isinstance(term, URIRef):
            continue
        declaration = _declaration_for(node, ranges)
        if declaration is not None:
            g.add((term, RDF.type, declaration))


def _range_index(edges) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for edge in edges:
        if str(edge.iri or "") == str(RDFS.range):
            out.setdefault(edge.source.id, []).append(getattr(edge.target, "iri", None) or "")
    return out


def _declaration_for(node: KGNode, ranges) -> Optional[URIRef]:
    from besser.BUML.notations.kg_to_buml._common import is_meta_vocab, looks_like_datatype_iri

    if is_meta_vocab(node.iri):
        return None
    if isinstance(node, KGProperty):
        return _property_declaration(node, ranges.get(node.id, []))
    if isinstance(node, KGClass):
        # The importer classifies xsd:* terms as KGClass so they render on the
        # canvas; they are datatypes, not classes.
        return None if looks_like_datatype_iri(node.iri) else OWL.Class
    if isinstance(node, KGIndividual):
        return OWL.NamedIndividual
    return None


def _property_declaration(node: KGProperty, range_iris: List[str]) -> URIRef:
    """Object vs datatype property, from the recorded kind or the range.

    A property with no range at all is treated as a datatype property, so it
    lands as a (string) attribute rather than disappearing — the preflight
    ``PROPERTY_NO_RANGE`` issue is what prompts the user to give it a real one.
    """
    from besser.BUML.notations.kg_to_buml._common import looks_like_datatype_iri

    kind = node.metadata.get("kind")
    if kind == "Object":
        return OWL.ObjectProperty
    if kind == "Datatype":
        return OWL.DatatypeProperty
    if range_iris and not any(looks_like_datatype_iri(iri) for iri in range_iris):
        return OWL.ObjectProperty
    return OWL.DatatypeProperty


# ----------------------------------------------------------------------
# Pass B — deterministic list re-collection
# ----------------------------------------------------------------------


def _pass_b_lists(g, nodes, edges, terms, default_ns, warnings) -> None:
    """Rebuild the ``rdf:List`` spines the importer flattened into direct edges.

    Order matters: the auxiliary class materialised for an ``owl:unionOf``
    is *named* after its operands, so a reordered list renames the class.
    """
    recorded_order: Dict[Tuple[str, str], List[str]] = {}
    for node in nodes:
        combinator = node.metadata.get("combinator")
        members = node.metadata.get("members")
        if combinator is None and isinstance(node, KGNodeConstraint):
            # A class expression the lifter turned into a constraint node keeps
            # its operands (in order) on the spec rather than in `members`.
            for spec in node.get_specs():
                if spec.get("kind") in _CLASS_EXPRESSION_KINDS:
                    combinator = spec.get("kind")
                    value = spec.get("value")
                    members = value if isinstance(value, list) else [value]
                    break
        if not combinator or not isinstance(members, list):
            continue
        subject = terms.get(node.id)
        if subject is None or isinstance(subject, Literal):
            continue
        if combinator == "complementOf":
            # Single-valued: not a list, and it already survived pass A when the
            # importer kept the raw edge. Idempotent either way.
            if members:
                g.add((subject, OWL.complementOf, URIRef(str(members[0]))))
            continue
        recorded_order[(node.id, _OWL_NS + combinator)] = [str(m) for m in members]

    # Collect the flattened member edges, grouped by (subject, predicate).
    flattened: Dict[Tuple[str, str], List[KGEdge]] = {}
    for edge in edges:
        predicate = str(predicate_for_edge(edge, default_ns))
        if predicate not in _FLATTENED_LIST_PREDICATES:
            continue
        flattened.setdefault((edge.source.id, predicate), []).append(edge)

    for (node_id, predicate), group in sorted(flattened.items()):
        subject = terms.get(node_id)
        if subject is None or isinstance(subject, Literal):
            continue
        order = recorded_order.get((node_id, predicate))
        if order is not None:
            # The metadata supplies the *order*; the edges supply the *terms*.
            # Rebuilding terms from the recorded strings would turn a literal
            # member of an ``owl:oneOf`` into an IRI, since the importer records
            # a literal by its lexical form.
            rank = {key: index for index, key in enumerate(order)}
            group.sort(key=lambda e: (rank.get(_member_key(e.target), len(rank)),
                                      _natural_key(e.id)))
        else:
            # Editor-authored graph, or a predicate the importer does not
            # annotate. Fall back to edge-id order and say so: the operand order
            # is observable in auxiliary class names.
            group.sort(key=lambda e: (_natural_key(e.id), _natural_key(e.target.id)))
            add_warning(
                warnings,
                "LIST_ORDER_INFERRED",
                f"No recorded member order for {predicate} on node {node_id!r}; "
                f"inferred it from edge ids.",
                node_id=node_id,
            )
        members = [terms[e.target.id] for e in group if terms.get(e.target.id) is not None]
        if members:
            g.add((subject, URIRef(predicate), emit_rdf_list(g, members)))


def _member_key(node: KGNode) -> str:
    """The string the importer recorded for this node in ``metadata['members']``.

    Mirrors ``owl_to_buml._term_str(m) or str(m)``: an IRI for named terms, the
    lexical form for literals, the label for blank nodes.
    """
    if isinstance(node, KGLiteral):
        return str(node.value)
    return node.iri or node.label or node.id


# ----------------------------------------------------------------------
# Pass C — restore the structure the constraint lifter consumed
# ----------------------------------------------------------------------


class _ConstraintIndexes:
    """Which classes/properties each constraint node applies to."""

    def __init__(self) -> None:
        self.nc_targets: Dict[str, List[Any]] = {}
        self.pc_property: Dict[str, Any] = {}
        self.nc_pcs: Dict[str, List[str]] = {}
        #: Node constraints that declare a target which does not resolve to a
        #: usable RDF subject (e.g. an edge pointing at a literal).
        self.nc_unresolved_targets: Set[str] = set()


def _build_constraint_indexes(edges, terms) -> _ConstraintIndexes:
    """Index constraint wiring from both internal and raw SHACL predicates.

    Reading both means the projection works whether the KG came from the
    importer (which uses the internal predicates) or from the editor (which may
    only have the SHACL ones).
    """
    idx = _ConstraintIndexes()
    for edge in edges:
        # ``str`` because rdflib term equality is type-sensitive: a URIRef
        # never compares equal to the plain string an editor payload carries.
        predicate = str(edge.iri or "")
        source, target = edge.source, edge.target
        if predicate in (CONSTRAINT_TARGET_CLASS, SH_TARGET_CLASS) and isinstance(
            source, KGNodeConstraint
        ):
            term = terms.get(target.id)
            if term is None or isinstance(term, Literal):
                idx.nc_unresolved_targets.add(source.id)
                continue
            idx.nc_targets.setdefault(source.id, [])
            if term not in idx.nc_targets[source.id]:
                idx.nc_targets[source.id].append(term)
        elif predicate in (CONSTRAINT_TARGET_PROPERTY, SH_PATH) and isinstance(
            source, KGPropertyConstraint
        ):
            term = terms.get(target.id)
            if term is not None and not isinstance(term, Literal):
                idx.pc_property.setdefault(source.id, term)
        elif predicate == SH_PROPERTY and isinstance(target, KGPropertyConstraint):
            idx.nc_pcs.setdefault(source.id, [])
            if target.id not in idx.nc_pcs[source.id]:
                idx.nc_pcs[source.id].append(target.id)
    return idx


def _pass_c_constraint_structure(g, nodes, terms, idx, warnings) -> None:
    """Re-attach constraint nodes to the classes they constrain.

    The importer deletes the ``Class rdfs:subClassOf _:restriction`` triple when
    it lifts a restriction into a ``KGPropertyConstraint``, because the editor
    shows the link through ``constraintTargetClass`` instead. The mapper needs
    the OWL form back.

    Direction note: ``constraintTargetClass`` points constraint→class while the
    OWL triple points class→constraint, and the lifter conflates
    ``rdfs:subClassOf`` with ``owl:equivalentClass``. Always emitting
    ``rdfs:subClassOf`` is lossless here — ``Mapper._map_class_axioms`` routes
    an ``equivalentClass``-to-blank-node through the identical
    ``resolve_class(o, subsuming=...)`` call.
    """
    by_id = {n.id: n for n in nodes}
    for node in nodes:
        if not isinstance(node, KGNodeConstraint):
            continue
        targets = idx.nc_targets.get(node.id, [])
        if not targets:
            has_payload = bool(node.get_specs() or idx.nc_pcs.get(node.id))
            if has_payload or node.id in idx.nc_unresolved_targets:
                add_warning(
                    warnings,
                    "ORPHANED_CONSTRAINT",
                    f"Constraint node {node.label or node.id!r} does not target a usable "
                    f"class; it cannot be attached to the model.",
                    node_id=node.id,
                )
            continue
        kinds = {spec.get("kind") for spec in node.get_specs()}
        nc_term = terms.get(node.id)

        if kinds & _CLASS_EXPRESSION_KINDS and nc_term is not None:
            for target in targets:
                g.add((target, RDFS.subClassOf, nc_term))

        # Class-level axioms (disjointWith, hasKey, …) are emitted in pass D so
        # they go through its "already present" gate: the importer keeps them
        # *both* as specs and as flattened member edges, and re-emitting one on
        # top of the other would give the class two `owl:hasKey` lists.

        if not _is_owlish(node):
            continue
        for pc_id in idx.nc_pcs.get(node.id, []):
            pc = by_id.get(pc_id)
            pc_term = terms.get(pc_id)
            if pc is None or pc_term is None or not _is_owlish(pc):
                continue
            for target in targets:
                g.add((target, RDFS.subClassOf, pc_term))


# ----------------------------------------------------------------------
# Pass D — constraint-spec back-fill
# ----------------------------------------------------------------------


def _pass_d_spec_backfill(g, nodes, terms, idx, default_ns, warnings) -> None:
    """Emit spec-derived triples, but only where no raw edge already covers them.

    Staging into a separate graph and copying by ``(subject, predicate)``
    absence gives three things at once:

    * **Idempotence.** A raw ``sh:maxCount "1"^^xsd:integer`` edge wins over the
      spec's ``"1"^^xsd:nonNegativeInteger`` rendering, so no near-duplicate
      triple appears.
    * **Recovery of swept lists.** ``sh:or``/``sh:and``/``sh:xone``/``sh:in``
      had their list spines consumed by the importer with no replacement edge,
      so they are emitted here. This is what makes SHACL logical constraints
      survive at all.
    * **Editor support.** Constraints drawn in the UI have specs and no raw
      edges, so everything comes from this pass.
    """
    staged = rdflib.Graph()
    present = {(s, p) for s, p, _ in g}

    def resolve_ref(node_id: str):
        term = terms.get(node_id)
        if term is None:
            add_warning(
                warnings,
                "SHAPE_REF_UNRESOLVED",
                f"Nested shape references unknown constraint node {node_id!r}; dropped.",
                node_id=node_id,
            )
        return term

    for node in nodes:
        if not isinstance(node, (KGNodeConstraint, KGPropertyConstraint)):
            continue
        term = terms.get(node.id)
        if term is None or isinstance(term, Literal):
            continue

        if isinstance(node, KGPropertyConstraint):
            if _is_owlish(node):
                _stage_owl_restriction(staged, node, idx, terms)
            else:
                _stage_shacl_property_shape(staged, node, idx, terms, resolve_ref)
        elif _is_owlish(node):
            _stage_owl_class_expression(staged, node, term)
            if {spec.get("kind") for spec in node.get_specs()} & _CLASS_AXIOM_KINDS:
                targets = [str(t) for t in idx.nc_targets.get(node.id, [])]
                emit_nodeconstraint_owl_axioms(staged, node, targets)
        else:
            # Staged even with no specs of its own: a node shape whose whole job
            # is to carry `sh:property` children still needs its `rdf:type` and
            # `sh:targetClass`, which is exactly what the SHACL phase keys on.
            _stage_shacl_node_shape(staged, node, idx, resolve_ref)

    for triple in staged:
        if (triple[0], triple[1]) not in present:
            g.add(triple)


def _stage_owl_class_expression(staged, nc: KGNodeConstraint, term) -> None:
    """Re-materialise an anonymous class expression onto the constraint's own term.

    When the importer lifts ``[ a owl:Class ; owl:oneOf ("red" "green") ]`` into
    a ``KGNodeConstraint`` it keeps the operands in ``constraintSpecs`` and
    deletes the member edges outright, so — unlike the ``KGBlank`` case handled
    in pass B — there is nothing left for the raw-edge pass to find.

    The triples belong on the *expression* node, not on the class it constrains:
    pass C already emitted ``TargetClass rdfs:subClassOf <this node>``.
    ``emit_rdf_list`` decides literal-vs-IRI per member, which is what keeps an
    ``owl:oneOf`` of literals from degrading into an ``owl:oneOf`` of IRIs.
    """
    for spec in nc.get_specs():
        kind = spec.get("kind")
        value = spec.get("value")
        if kind == "complementOf" and isinstance(value, str):
            staged.add((term, OWL.complementOf, URIRef(value)))
        elif kind in ("unionOf", "intersectionOf", "oneOf") and isinstance(value, list) and value:
            staged.add((term, RDF.type, OWL.Class))
            staged.add((term, URIRef(_OWL_NS + kind), emit_rdf_list(staged, value)))


def _stage_owl_restriction(staged, pc: KGPropertyConstraint, idx, terms) -> None:
    """Re-materialise an OWL restriction onto the constraint node's own term.

    Unlike the TTL exporter, which mints a fresh blank node per restriction,
    the ``KGPropertyConstraint`` *is* the original restriction blank node — so
    reusing its term keeps the ``rdfs:subClassOf`` link pass C emitted valid.
    ``owner_class_iris`` stays empty for the same reason.
    """
    property_term = idx.pc_property.get(pc.id)
    property_iri = str(property_term) if property_term is not None else pc.metadata.get("onPropertyIri")
    if not property_iri:
        return
    restriction = terms[pc.id]
    for spec in pc.get_specs():
        predicate = OWL_RESTRICTION_KINDS.get(spec.get("kind"))
        if predicate is None:
            continue
        staged.add((restriction, RDF.type, OWL.Restriction))
        staged.add((restriction, OWL.onProperty, URIRef(property_iri)))
        _stage_restriction_value(staged, restriction, predicate, spec)


def _stage_restriction_value(staged, restriction, predicate, spec) -> None:
    from besser.utilities.kg_to_owl import spec_to_literal
    from rdflib.namespace import XSD

    kind = spec.get("kind")
    value = spec.get("value")
    on_class = spec.get("on_class")
    if kind in ("someValuesFrom", "allValuesFrom"):
        staged.add((restriction, predicate,
                    URIRef(str(value)) if isinstance(value, str) else spec_to_literal(value)))
    elif kind == "hasValue":
        target = URIRef(value) if isinstance(value, str) and "://" in value else spec_to_literal(value)
        if target is not None:
            staged.add((restriction, predicate, target))
    elif kind == "hasSelf":
        staged.add((restriction, predicate, Literal(bool(value), datatype=XSD.boolean)))
    else:
        literal = spec_to_literal(value)
        if literal is not None:
            staged.add((restriction, predicate, literal))
        if on_class and str(kind).endswith("QualifiedCardinality"):
            staged.add((restriction, OWL.onClass, URIRef(on_class)))


def _stage_shacl_property_shape(staged, pc: KGPropertyConstraint, idx, terms, resolve_ref) -> None:
    shape = terms[pc.id]
    staged.add((shape, RDF.type, SH.PropertyShape))
    property_term = idx.pc_property.get(pc.id)
    if property_term is None and pc.metadata.get("onPropertyIri"):
        property_term = URIRef(pc.metadata["onPropertyIri"])
    if property_term is not None:
        staged.add((shape, SH.path, property_term))
    for spec in pc.get_specs():
        if _is_known_shacl_kind(spec.get("kind")):
            emit_shacl_constraint(staged, shape, spec, resolve_ref, SH.PropertyShape)


def _stage_shacl_node_shape(staged, nc: KGNodeConstraint, idx, resolve_ref) -> None:
    from besser.utilities.kg_to_owl import term_for_node

    shape = term_for_node(nc, DEFAULT_NAMESPACE)
    staged.add((shape, RDF.type, SH.NodeShape))
    for target in idx.nc_targets.get(nc.id, []):
        staged.add((shape, SH.targetClass, target))
    for spec in nc.get_specs():
        if _is_known_shacl_kind(spec.get("kind")):
            emit_shacl_constraint(staged, shape, spec, resolve_ref, SH.NodeShape)


def _is_known_shacl_kind(kind: Optional[str]) -> bool:
    return bool(kind) and (
        kind in SHACL_PROPERTY_KINDS or kind in SHACL_NODE_KINDS or kind in SHACL_LOGICAL_KINDS
    )


# ----------------------------------------------------------------------
# Base namespace
# ----------------------------------------------------------------------


def _base_namespace(g: rdflib.Graph) -> Optional[str]:
    """Namespace stem of the ``owl:Ontology`` IRI, if the graph declares one.

    Used by the mapper to tell locally-defined classes from imported/foreign
    ones. ``None`` (no ontology header, which is the common case for editor-
    authored graphs) simply disables that distinction.
    """
    ontology = next(iter(sorted(g.subjects(RDF.type, OWL.Ontology), key=str)), None)
    if ontology is None:
        return None
    text = str(ontology)
    for separator in ("#", "/"):
        if separator in text:
            return text.rsplit(separator, 1)[0] + separator
    return text
