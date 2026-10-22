"""Project a :class:`KnowledgeGraph` into an in-memory :class:`rdflib.Graph`.

This is the RDF view of a KG that BESSER *reasons over*: the KG → UML
conversion reads it, and so does the consistency check (with
``mirror_shapes=True``). The paper's transformation rules
(:mod:`besser.BUML.notations.kg_to_buml.owl2uml`) are defined over RDF triples,
while the KG metamodel stores a triple store *plus* an editor-oriented
"constraint node" view layered on top. This module reconciles the two.

The user-facing TTL export, :func:`besser.utilities.kg_to_owl.knowledge_graph_to_rdf`,
is a separate function: it skips every edge touching a constraint node and
rebuilds the constraint structure from specs, which is right for a clean export
but lossy as a reasoning input. On a graph that uses the SHACL punning idiom,
exporting and re-reading loses ``rdfs:subClassOf``, ``rdfs:domain``,
``rdfs:range``, ``sh:targetClass`` and ``owl:disjointWith`` triples on the
punned nodes. This projection keeps the raw edges for that structure. Both read
constraint *parameters* from the same place, though:

**Constraint specs are the single source of truth for constraint parameters.**
The importer folds ``_:shape sh:minCount 1`` into the node's
``constraintSpecs`` and keeps no raw edge for it
(:func:`~besser.BUML.notations.owl_to_kg.owl_to_kg.spec_folded_predicates`
names those predicates), and the inspector edits only the spec. Both writers
emit those parameters from the spec and ignore any such edge drawn by hand, so
a spec edit reaches the class diagram and the consistency report alike. Both
also share the leaf emitters, the constraint indexes and the list rebuilding
(:mod:`besser.utilities.kg_to_owl`).

``mirror_shapes`` is the one difference between the two consumers. The
conversion applies its OWL and SHACL rules separately, so each constraint node
is written in the vocabulary it came from. The consistency check validates with
SHACL, so it also needs every OWL restriction mirrored as a SHACL shape (and,
for parity with the exporter's ``vocab="both"``, every SHACL shape as an OWL
restriction). ``test_projection_contract.py`` pins how the export and this
projection may differ.

Passes:

- ``A``: raw edges → triples, skipping internal wiring, flattened list members
  and spec-folded constraint parameters (pass ``D`` writes those).
- ``B``: rebuild the flattened ``rdf:List`` values, in their original order,
  and one ``owl:propertyChainAxiom`` list per axiom record.
- ``C``: restore the structural links the constraint-node lifter consumed.
- ``D``: emit the constraint specs (parameters always; other triples only where
  no raw edge already covers them).
- ``E`` (``mirror_shapes`` only): each constraint in the other vocabulary too.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import rdflib
from besser.BUML.metamodel.kg.constants import (
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    OWL as OWL_NS,
    SH as SH_NS,
    SH_PROPERTY,
)
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from besser.BUML.metamodel.kg import (
    KGClass,
    KGIndividual,
    KGNode,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.utilities.kg_to_owl import (
    CLASS_EXPRESSION_PREDICATES,
    DEFAULT_NAMESPACE,
    FLATTENED_LIST_PREDICATES,
    SHACL_LOGICAL_KINDS,
    SHACL_NODE_KINDS,
    SHACL_PROPERTY_KINDS,
    blank_handles,
    build_constraint_indexes,
    emit_flattened_lists,
    emit_nodeconstraint_owl_axioms,
    emit_owl_class_expression,
    emit_owl_restriction,
    emit_property_chains,
    emit_shacl_constraint,
    emit_shacl_node_constraint,
    natural_key,
    predicate_for_edge,
    spec_folded_by_term,
    term_for_node,
)

from besser.BUML.notations.kg_to_buml._common import (
    KGConversionWarning,
    add_warning,
    is_meta_vocab,
    looks_like_datatype_iri,
)

SH = rdflib.Namespace(SH_NS)
_OWL_NS = OWL_NS

__all__ = ["kg_to_rdf", "DEFAULT_NAMESPACE"]


#: Editor-only wiring. These predicates express "this constraint node applies to
#: that class/property" and have no RDF counterpart — pass C turns them back
#: into the ``rdfs:subClassOf`` / ``sh:targetClass`` / ``sh:path`` triples the
#: source actually contained.
_INTERNAL_PREDICATES = frozenset({CONSTRAINT_TARGET_CLASS, CONSTRAINT_TARGET_PROPERTY})

#: SHACL list-valued predicates whose list lives in the specs only (the importer
#: sweeps the spine and keeps no edge): pass A must not write a hand-drawn edge
#: with one of them as it is, and pass D writes the whole list from the specs.
_CONSUMED_LIST_PREDICATES = frozenset({
    str(SH["and"]), str(SH["or"]), str(SH.xone), str(SH["in"]),
    str(SH.languageIn), str(SH.ignoredProperties),
})

_LIST_VALUED_PREDICATES = FLATTENED_LIST_PREDICATES | _CONSUMED_LIST_PREDICATES

#: Spec kinds that are class-level OWL axioms, anchored on the target class.
_CLASS_AXIOM_KINDS = frozenset({
    "equivalentClasses", "disjointWith", "subClassOf",
    "disjointUnionOf", "hasKey",
})


def _is_owlish(node: KGNode) -> bool:
    """True unless the constraint node is explicitly SHACL-sourced.

    Mirrors the importer's tagging: constraint nodes lifted from OWL are marked
    ``"owl"`` / ``"owl-axiom"``, those lifted from SHACL shapes ``"shacl"``. A
    node with no tag at all was hand-authored in the editor, where an
    OWL-restriction payload is the only thing it could be.
    """
    return node.metadata.get("source") != "shacl"


def _is_constraint(node: KGNode) -> bool:
    return isinstance(node, (KGNodeConstraint, KGPropertyConstraint))


def kg_to_rdf(
    kg: KnowledgeGraph,
    *,
    default_namespace: str = DEFAULT_NAMESPACE,
    warnings: Optional[List[KGConversionWarning]] = None,
    mirror_shapes: bool = False,
) -> Tuple[rdflib.Graph, Optional[str]]:
    """Project ``kg`` into a single graph carrying both OWL axioms and SHACL shapes.

    Returns ``(graph, base_namespace)``. For the conversion, the same graph
    should be passed as both the ontology and the shapes argument of
    :func:`~besser.BUML.notations.kg_to_buml.owl2uml.plan_domain_model` — the
    SHACL phase only looks at ``sh:NodeShape`` subjects, so the two
    vocabularies coexist without interfering.

    Args:
        kg: The graph to project.
        default_namespace: Namespace for nodes and predicates with no IRI.
        warnings: Collects the conversion warnings raised along the way.
        mirror_shapes: Also write every constraint node in the vocabulary it
            did not come from: OWL restrictions as SHACL shapes and SHACL shapes
            as OWL restrictions. The consistency check needs this (it validates
            OWL-sourced restrictions with SHACL); the conversion must not have
            it, since it maps the two vocabularies separately.
    """
    warnings = warnings if warnings is not None else []
    g = rdflib.Graph()

    nodes = sorted(kg.nodes, key=lambda n: natural_key(n.id))
    edges = sorted(kg.edges, key=lambda e: natural_key(e.id))
    terms: Dict[str, Any] = {n.id: term_for_node(n, default_namespace) for n in nodes}
    folded = spec_folded_by_term(nodes, terms)
    handles = blank_handles(nodes, terms)

    _pass_a_raw_edges(g, edges, terms, folded, default_namespace, warnings)
    _pass_a2_declarations(g, nodes, edges, terms, default_namespace, warnings)
    _pass_b_lists(g, kg, nodes, edges, terms, folded, warnings)
    indexes = build_constraint_indexes(edges, terms)
    _pass_c_constraint_structure(g, nodes, terms, indexes, warnings)
    _pass_d_spec_emission(g, nodes, terms, indexes, folded, handles, warnings)
    if mirror_shapes:
        _pass_e_mirror(g, nodes, terms, indexes, handles)

    return g, _base_namespace(g)


# ----------------------------------------------------------------------
# Pass A — raw edges
# ----------------------------------------------------------------------


def _pass_a_raw_edges(g, edges, terms, folded, default_ns, warnings) -> None:
    """Emit one triple per edge, except the ones settled elsewhere.

    A constraint parameter (a spec-folded predicate on a constraint node's term)
    is skipped: the importer keeps none, so such an edge was drawn by hand, and
    the spec pass D writes is what the inspector shows and edits.
    """
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
        if predicate in folded.get(subject, ()):
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
        target_iri = edge.target.iri or ""
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
            out.setdefault(edge.source.id, []).append(edge.target.iri or "")
    return out


def _declaration_for(node: KGNode, ranges) -> Optional[URIRef]:

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


def _pass_b_lists(g, kg: KnowledgeGraph, nodes, edges, terms, folded, warnings) -> None:
    """Rebuild the ``rdf:List`` spines the importer flattened into direct edges,
    and the property chains from their records (see
    :func:`~besser.utilities.kg_to_owl.emit_flattened_lists`).

    Order matters: the auxiliary class materialised for an ``owl:unionOf`` is
    *named* after its operands, so a reordered list renames the class. Where no
    order was recorded the edge-id order is used, and a warning says so.
    """

    def inferred(node_id: str, predicate: str) -> None:
        add_warning(
            warnings,
            "LIST_ORDER_INFERRED",
            f"No recorded member order for {predicate} on node {node_id!r}; "
            f"inferred it from edge ids.",
            node_id=node_id,
        )

    emit_flattened_lists(g, nodes, edges, terms, folded, kg.axioms, inferred)
    emit_property_chains(g, kg.axioms, terms)


# ----------------------------------------------------------------------
# Pass C — restore the structure the constraint lifter consumed
# ----------------------------------------------------------------------


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

        if kinds & set(CLASS_EXPRESSION_PREDICATES) and nc_term is not None:
            for target in targets:
                g.add((target, RDFS.subClassOf, nc_term))

        # Class-level axioms (disjointWith, hasKey, …) are written in pass D,
        # from the specs.

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
# Pass D — constraint specs
# ----------------------------------------------------------------------


def _pass_d_spec_emission(g, nodes, terms, idx, folded, handles, warnings) -> None:
    """Emit what the constraint specs say.

    Everything is staged into a separate graph first, then copied in by
    ``(subject, predicate)``:

    * **Constraint parameters** (a spec-folded predicate on a constraint node's
      own term) always come from the spec; pass A wrote none.
    * **Everything else** is copied only where no raw edge already covers the
      pair: that gives constraints drawn in the editor (specs, no raw edges)
      their whole structure (``rdf:type``, ``sh:path``, …) without duplicating
      what an imported graph already has.

    A blank node the staging minted (a list cell, a qualified value shape) is
    copied only together with the triple that points at it, so a triple turned
    away by the gate cannot leave a dangling ``rdf:List`` behind.
    """
    staged = rdflib.Graph()

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
        if not _is_constraint(node):
            continue
        term = terms.get(node.id)
        if term is None or isinstance(term, Literal):
            continue

        if isinstance(node, KGPropertyConstraint):
            if _is_owlish(node):
                _stage_owl_restriction(staged, node, idx, terms, handles)
            else:
                _stage_shacl_property_shape(staged, node, idx, terms, resolve_ref)
        elif _is_owlish(node):
            emit_owl_class_expression(staged, node, term, handles)
            if {spec.get("kind") for spec in node.get_specs()} & _CLASS_AXIOM_KINDS:
                emit_nodeconstraint_owl_axioms(staged, node, idx.nc_targets.get(node.id, []), handles)
        else:
            # Staged even with no specs of its own: a node shape whose whole job
            # is to carry `sh:property` children still needs its `rdf:type` and
            # `sh:targetClass`, which is exactly what the SHACL phase keys on.
            _stage_shacl_node_shape(staged, node, idx, terms, resolve_ref)

    _copy_staged(g, staged, set(terms.values()), folded)


def _copy_staged(g, staged, known_terms, folded) -> None:
    """Copy ``staged`` into ``g`` under the rules of :func:`_pass_d_spec_emission`."""
    minted = {t for t in staged.all_nodes() if isinstance(t, BNode) and t not in known_terms}
    present = {(s, p) for s, p, _ in g}

    def add_with_closure(triple, seen: Set[Any]) -> None:
        g.add(triple)
        obj = triple[2]
        if obj in minted and obj not in seen:
            seen.add(obj)
            for inner in staged.triples((obj, None, None)):
                add_with_closure(inner, seen)

    grouped: Dict[Tuple[Any, Any], List[Any]] = {}
    for s, p, o in staged:
        if s not in minted:
            grouped.setdefault((s, p), []).append(o)

    for (s, p), objects in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        if str(p) in folded.get(s, ()) or (s, p) not in present:
            for o in objects:
                add_with_closure((s, p, o), set())


def _restricted_property(pc: KGPropertyConstraint, idx) -> Optional[str]:
    property_term = idx.pc_property.get(pc.id)
    return str(property_term) if property_term is not None else pc.metadata.get("onPropertyIri")


def _stage_owl_restriction(staged, pc: KGPropertyConstraint, idx, terms, handles) -> None:
    """Re-materialise an OWL restriction onto the constraint node's own term.

    Unlike the TTL exporter, which mints a fresh blank node per restriction,
    the ``KGPropertyConstraint`` *is* the original restriction blank node — so
    reusing its term keeps the ``rdfs:subClassOf`` link pass C emitted valid.
    ``owner_class_iris`` stays empty for the same reason.
    """
    emit_owl_restriction(
        staged, pc, _restricted_property(pc, idx), [], subject=terms[pc.id], blank_handles=handles,
    )


def _shape_path(pc: KGPropertyConstraint, idx) -> Optional[Any]:
    property_term = idx.pc_property.get(pc.id)
    if property_term is None and pc.metadata.get("onPropertyIri"):
        property_term = URIRef(pc.metadata["onPropertyIri"])
    return property_term


def _stage_shacl_property_shape(staged, pc: KGPropertyConstraint, idx, terms, resolve_ref) -> None:
    shape = terms[pc.id]
    staged.add((shape, RDF.type, SH.PropertyShape))
    property_term = _shape_path(pc, idx)
    if property_term is not None:
        staged.add((shape, SH.path, property_term))
    for spec in pc.get_specs():
        if _is_known_shacl_kind(spec.get("kind")):
            emit_shacl_constraint(staged, shape, spec, resolve_ref, SH.PropertyShape)


def _stage_shacl_node_shape(staged, nc: KGNodeConstraint, idx, terms, resolve_ref) -> None:
    # ``terms`` already holds this node's term, built with the caller's
    # ``default_namespace``; recomputing it here with the module default ignored
    # that argument.
    shape = terms[nc.id]
    staged.add((shape, RDF.type, SH.NodeShape))
    for target in idx.nc_targets.get(nc.id, []):
        staged.add((shape, SH.targetClass, target))
    _stage_node_shape_specs(staged, shape, nc, resolve_ref)


def _stage_node_shape_specs(staged, shape, nc: KGNodeConstraint, resolve_ref) -> None:
    """Node-level kinds (``sh:closed``, …) and logical operators through the
    node-shape emitter; value constraints a node shape may also carry
    (``sh:in``, ``sh:datatype``, …) through the property-shape one."""
    for spec in nc.get_specs():
        if emit_shacl_node_constraint(staged, shape, spec, resolve_ref):
            continue
        if _is_known_shacl_kind(spec.get("kind")):
            emit_shacl_constraint(staged, shape, spec, resolve_ref, SH.NodeShape)


def _is_known_shacl_kind(kind: Optional[str]) -> bool:
    return bool(kind) and (
        kind in SHACL_PROPERTY_KINDS or kind in SHACL_NODE_KINDS or kind in SHACL_LOGICAL_KINDS
    )


# ----------------------------------------------------------------------
# Pass E — mirror each constraint into the other vocabulary
# ----------------------------------------------------------------------


def _pass_e_mirror(g, nodes, terms, idx, handles) -> None:
    """Write each constraint node in the vocabulary it did *not* come from.

    * An OWL restriction also becomes a SHACL property shape, on the
      constraint's own term, so ``sh:sourceShape`` in a validation report still
      names the KG node. Its wrapper node constraint becomes the node shape that
      targets the restricted class and lists it under ``sh:property``.
    * A SHACL property shape also becomes an ``owl:Restriction`` on a fresh
      blank node, subclassed by every class a node shape listing it targets.

    These are additions, not replacements, so they bypass pass D's gate.
    """

    def resolve_ref(node_id: str):
        return terms.get(node_id)

    owners = idx.pc_owners()
    by_id = {n.id: n for n in nodes}
    for node in nodes:
        term = terms.get(node.id)
        if not _is_constraint(node) or term is None or isinstance(term, Literal):
            continue
        if isinstance(node, KGPropertyConstraint):
            if _is_owlish(node):
                g.add((term, RDF.type, SH.PropertyShape))
                path = _shape_path(node, idx)
                if path is not None:
                    g.add((term, SH.path, path))
                for spec in node.get_specs():
                    emit_shacl_constraint(g, term, spec, resolve_ref, SH.PropertyShape)
            else:
                emit_owl_restriction(
                    g, node, _restricted_property(node, idx), owners.get(node.id, []),
                    blank_handles=handles,
                )
        elif _is_owlish(node):
            g.add((term, RDF.type, SH.NodeShape))
            for target in idx.nc_targets.get(node.id, []):
                g.add((term, SH.targetClass, target))
            for pc_id in idx.nc_pcs.get(node.id, []):
                pc = by_id.get(pc_id)
                if pc is not None and _is_owlish(pc) and terms.get(pc_id) is not None:
                    g.add((term, SH.property, terms[pc_id]))
            _stage_node_shape_specs(g, term, node, resolve_ref)


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
