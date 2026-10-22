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
The importer keeps a raw edge for every triple (``_:shape sh:minCount 1``) *and*
folds its value into the node's ``constraintSpecs``; the inspector edits only
the spec. So a raw edge whose predicate the importer folds
(:func:`~besser.BUML.notations.owl_to_kg.owl_to_kg.spec_folded_predicates`)
never overrides its spec: the raw term is used only
when it carries the same value as the spec (it keeps the source's exact
datatype), otherwise the spec's rendering is, and a spec that was deleted takes
its parameter with it. Before, the conversion trusted the raw edge, so editing
``minCount`` from 1 to 3 changed the consistency report but not the class
diagram.

``mirror_shapes`` is the one difference between the two consumers. The
conversion applies its OWL and SHACL rules separately, so each constraint node
is written in the vocabulary it came from. The consistency check validates with
SHACL, so it also needs every OWL restriction mirrored as a SHACL shape (and,
for parity with the exporter's ``vocab="both"``, every SHACL shape as an OWL
restriction). ``test_projection_contract.py`` pins how the export and this
projection may differ.

Passes:

- ``A``: raw edges → triples, skipping internal wiring, flattened list spines
  and spec-folded constraint parameters (pass ``D`` settles those).
- ``B``: re-collect flattened ``rdf:List`` values in their original order.
- ``B2``: one ``owl:propertyChainAxiom`` / ``owl:hasKey`` list per axiom record.
- ``C``: restore the structural links the constraint-node lifter consumed.
- ``D``: emit the constraint specs (parameters always; other triples only where
  no raw edge already covers them).
- ``E`` (``mirror_shapes`` only): each constraint in the other vocabulary too.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import rdflib
from besser.BUML.metamodel.kg.constants import (
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    OWL as OWL_NS,
    SH as SH_NS,
    SH_PATH,
    SH_PROPERTY,
    SH_TARGET_CLASS,
)
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from besser.BUML.metamodel.kg import (
    HasKeyAxiom,
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
    PropertyChainAxiom,
)
from besser.BUML.notations.owl_to_kg.owl_to_kg import spec_folded_predicates
from besser.utilities.kg_to_owl import (
    DEFAULT_NAMESPACE,
    SHACL_LOGICAL_KINDS,
    SHACL_NODE_KINDS,
    SHACL_PROPERTY_KINDS,
    emit_nodeconstraint_owl_axioms,
    emit_owl_restriction,
    emit_rdf_list,
    emit_shacl_constraint,
    emit_shacl_node_constraint,
    predicate_for_edge,
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

#: Flattened lists whose member order carries no meaning: a key is a set of
#: properties, ``owl:members`` a set of pairwise-disjoint terms. Rebuilding them
#: in edge-id order is not an inference worth warning about.
_UNORDERED_LIST_PREDICATES = frozenset({str(OWL.hasKey), str(OWL.members)})

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

    nodes = sorted(kg.nodes, key=lambda n: _natural_key(n.id))
    edges = sorted(kg.edges, key=lambda e: _natural_key(e.id))
    terms: Dict[str, Any] = {n.id: term_for_node(n, default_namespace) for n in nodes}
    folded = _folded_predicates(nodes, terms)

    deferred = _pass_a_raw_edges(g, edges, terms, folded, default_namespace, warnings)
    _pass_a2_declarations(g, nodes, edges, terms, default_namespace, warnings)
    rebuilt = _rebuilt_from_records(kg, nodes)
    _pass_b_lists(g, nodes, edges, terms, rebuilt, default_namespace, warnings)
    _pass_b2_axiom_lists(g, kg, terms)
    indexes = _build_constraint_indexes(edges, terms)
    _pass_c_constraint_structure(g, nodes, terms, indexes, warnings)
    blank_handles = _blank_handles(nodes)
    _pass_d_spec_emission(
        g, kg, nodes, terms, indexes, folded, deferred, blank_handles, default_namespace, warnings,
    )
    if mirror_shapes:
        _pass_e_mirror(g, nodes, terms, indexes, blank_handles, default_namespace)

    return g, _base_namespace(g)


def _folded_predicates(nodes, terms) -> Dict[Any, FrozenSet[str]]:
    """Per constraint-node term, the predicates whose value lives in its specs.

    Keyed by *term*, not node: a punned shape shares its IRI with a class, and a
    parameter written on the class node (``ex:Person sh:closed true``) is the
    shape's parameter all the same.
    """
    out: Dict[Any, Set[str]] = {}
    for node in nodes:
        if not _is_constraint(node):
            continue
        term = terms.get(node.id)
        if term is None or isinstance(term, Literal):
            continue
        out.setdefault(term, set()).update(spec_folded_predicates(node))
    return {term: frozenset(predicates) for term, predicates in out.items()}


def _blank_handles(nodes) -> FrozenSet[str]:
    """Labels of the KG's blank nodes; see ``kg_to_owl._emit_restriction_value``."""
    return frozenset(
        n.id[2:] if n.id.startswith("_:") else n.id for n in nodes if isinstance(n, KGBlank)
    )


# ----------------------------------------------------------------------
# Pass A — raw edges
# ----------------------------------------------------------------------


def _pass_a_raw_edges(g, edges, terms, folded, default_ns, warnings):
    """Emit one triple per edge, except the ones settled elsewhere.

    Returns the spec-folded constraint parameters it held back, as
    ``{(subject, predicate): [object, ...]}``, for pass D to reconcile with the
    specs.
    """
    deferred: Dict[Tuple[Any, Any], List[Any]] = {}
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
            deferred.setdefault((subject, URIRef(predicate)), []).append(target)
            continue
        g.add((subject, URIRef(predicate), target))
    return deferred


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


def _rebuilt_from_records(kg: KnowledgeGraph, nodes) -> Set[Tuple[str, str]]:
    """``(subject id, predicate)`` pairs whose lists come from somewhere better
    than the flattened member edges, so pass B leaves them alone.

    * ``owl:propertyChainAxiom``: one list per :class:`PropertyChainAxiom`
      (pass B2), which keeps order and repeats — the edges keep neither.
    * ``owl:hasKey``: one list per :class:`HasKeyAxiom` (pass B2), or else one
      per ``hasKey`` spec of a node constraint (pass D). The flattened edges of
      two keys on one class are indistinguishable, so rebuilding from them
      merged the keys into one composite key.
    """
    covered: Set[Tuple[str, str]] = set()
    for axiom in kg.axioms:
        if isinstance(axiom, PropertyChainAxiom) and axiom.property_id:
            covered.add((axiom.property_id, str(OWL.propertyChainAxiom)))
        elif isinstance(axiom, HasKeyAxiom) and axiom.class_id:
            covered.add((axiom.class_id, str(OWL.hasKey)))
    for target_id in _classes_keyed_by_specs(kg, nodes):
        covered.add((target_id, str(OWL.hasKey)))
    return covered


def _classes_keyed_by_specs(kg: KnowledgeGraph, nodes) -> Set[str]:
    """Ids of the classes some node constraint gives a ``hasKey`` spec."""
    keyed_ncs = {
        n.id for n in nodes
        if isinstance(n, KGNodeConstraint) and any(s.get("kind") == "hasKey" for s in n.get_specs())
    }
    return {
        e.target.id for e in kg.edges
        if e.source.id in keyed_ncs and str(e.iri or "") in (CONSTRAINT_TARGET_CLASS, SH_TARGET_CLASS)
    }


def _pass_b_lists(g, nodes, edges, terms, rebuilt, default_ns, warnings) -> None:
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
            # Single-valued: not a list. Spec-folded, so pass A held back any raw
            # edge and pass D emits it; for a plain class-expression blank the
            # metadata is the only source.
            if members and not _is_constraint(node):
                g.add((subject, OWL.complementOf, URIRef(str(members[0]))))
            continue
        recorded_order[(node.id, _OWL_NS + combinator)] = [str(m) for m in members]

    # Collect the flattened member edges, grouped by (subject, predicate).
    flattened: Dict[Tuple[str, str], List[KGEdge]] = {}
    for edge in edges:
        predicate = str(predicate_for_edge(edge, default_ns))
        if predicate not in _FLATTENED_LIST_PREDICATES:
            continue
        if (edge.source.id, predicate) in rebuilt:
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
            # annotate. Fall back to edge-id order, and say so where the order
            # is observable (operand order shows in auxiliary class names).
            group.sort(key=lambda e: (_natural_key(e.id), _natural_key(e.target.id)))
            if predicate not in _UNORDERED_LIST_PREDICATES:
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


def _pass_b2_axiom_lists(g, kg: KnowledgeGraph, terms) -> None:
    """One ``owl:propertyChainAxiom`` / ``owl:hasKey`` list per axiom record.

    Pass B can only see the flattened subject→member edges, and a KG holds one
    edge per ``(subject, member, predicate)``: ``hasParent∘hasParent`` — the
    textbook chain — would come back as the one-element ``(hasParent)``, and two
    chains (or two keys) on one subject would merge into one. The records keep
    each list whole and in order, repeats included.

    A property may have several chains and a class several keys; each is its
    own triple, so every record is *added* — none replaces another.
    """
    for axiom in kg.axioms:
        if isinstance(axiom, PropertyChainAxiom):
            subject_id, predicate, members = (
                axiom.property_id, OWL.propertyChainAxiom, axiom.chain_property_ids,
            )
        elif isinstance(axiom, HasKeyAxiom):
            subject_id, predicate, members = axiom.class_id, OWL.hasKey, axiom.property_ids
        else:
            continue
        if not subject_id or not members:
            continue
        subject = terms.get(subject_id, URIRef(str(subject_id)))
        g.add((subject, predicate, emit_rdf_list(g, [URIRef(str(m)) for m in members])))


def _member_key(node: KGNode) -> str:
    """The string the importer recorded for this node in ``metadata['members']``.

    Mirrors ``owl_to_kg._term_str(m) or str(m)``: an IRI for named terms, the
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
# Pass D — constraint specs
# ----------------------------------------------------------------------


def _pass_d_spec_emission(
    g, kg, nodes, terms, idx, folded, deferred, blank_handles, default_ns, warnings,
) -> None:
    """Emit what the constraint specs say.

    Everything is staged into a separate graph first, then copied in by
    ``(subject, predicate)``:

    * **Constraint parameters** (a spec-folded predicate on a constraint node's
      own term) always come from the spec. The raw edge pass A held back is
      used instead only when it carries the same value, because it keeps the
      source's exact form (``"1"^^xsd:integer`` rather than the spec's
      ``"1"^^xsd:nonNegativeInteger``). A parameter whose spec was edited takes
      the new value; one whose spec was deleted is gone.
    * **Everything else** is copied only where no raw edge already covers the
      pair, as before: that recovers the ``sh:or``/``sh:and``/``sh:xone``/
      ``sh:in`` lists the importer consumed, and gives constraints drawn in the
      editor (specs, no raw edges) their whole structure, without duplicating
      what an imported graph already has.

    A blank node the staging minted (a list cell, a qualified value shape) is
    copied only together with the triple that points at it, so a triple turned
    away by the gate cannot leave a dangling ``rdf:List`` behind.
    """
    staged = rdflib.Graph()
    keys_from_axioms = {a.class_id for a in kg.axioms if isinstance(a, HasKeyAxiom) and a.class_id}

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
                _stage_owl_restriction(staged, node, idx, terms, blank_handles)
            else:
                _stage_shacl_property_shape(staged, node, idx, terms, resolve_ref)
        elif _is_owlish(node):
            _stage_owl_class_expression(staged, node, term)
            if {spec.get("kind") for spec in node.get_specs()} & _CLASS_AXIOM_KINDS:
                targets = [str(t) for t in idx.nc_targets.get(node.id, [])]
                emit_nodeconstraint_owl_axioms(
                    staged, _without_axiom_keys(node, targets, keys_from_axioms), targets,
                )
        else:
            # Staged even with no specs of its own: a node shape whose whole job
            # is to carry `sh:property` children still needs its `rdf:type` and
            # `sh:targetClass`, which is exactly what the SHACL phase keys on.
            _stage_shacl_node_shape(staged, node, idx, terms, resolve_ref)

    _copy_staged(g, staged, set(terms.values()), folded, deferred)


def _without_axiom_keys(nc: KGNodeConstraint, targets: List[str], keyed: Set[str]):
    """``nc``, minus its ``hasKey`` specs when a :class:`HasKeyAxiom` already keys
    its class.

    The importer records each ``owl:hasKey`` twice, as an axiom and as a node
    constraint with a ``hasKey`` spec; pass B2 has emitted the axiom's list.
    """
    if not targets or targets[0] not in keyed:
        return nc
    specs = [s for s in nc.get_specs() if s.get("kind") != "hasKey"]
    return KGNodeConstraint(id=nc.id, label=nc.label, iri=nc.iri,
                            metadata={**nc.metadata, "constraintSpecs": specs})


def _copy_staged(g, staged, known_terms, folded, deferred) -> None:
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
        if str(p) in folded.get(s, ()):
            raw = deferred.get((s, p), [])
            if raw and {_value_key(o) for o in raw} == {_value_key(o) for o in objects}:
                objects, from_spec = raw, False
            else:
                from_spec = True
            for o in objects:
                if from_spec:
                    add_with_closure((s, p, o), set())
                else:
                    g.add((s, p, o))
        elif (s, p) not in present:
            for o in objects:
                add_with_closure((s, p, o), set())


def _value_key(term) -> Tuple[str, Any]:
    """What a raw edge and a spec must agree on to count as the same value.

    Numbers by value (``"1"^^xsd:integer`` and ``1.0`` agree), other literals by
    lexical form, and named or blank references by label: a spec stores a blank
    filler as its bare label, the raw edge as the blank node itself.
    """
    if isinstance(term, Literal):
        value = term.toPython()
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, (int, float)) or type(value).__name__ == "Decimal":
            return ("number", value)
        return ("text", str(term))
    return ("ref", str(term))


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


def _restricted_property(pc: KGPropertyConstraint, idx) -> Optional[str]:
    property_term = idx.pc_property.get(pc.id)
    return str(property_term) if property_term is not None else pc.metadata.get("onPropertyIri")


def _stage_owl_restriction(staged, pc: KGPropertyConstraint, idx, terms, blank_handles) -> None:
    """Re-materialise an OWL restriction onto the constraint node's own term.

    Unlike the TTL exporter, which mints a fresh blank node per restriction,
    the ``KGPropertyConstraint`` *is* the original restriction blank node — so
    reusing its term keeps the ``rdfs:subClassOf`` link pass C emitted valid.
    ``owner_class_iris`` stays empty for the same reason.
    """
    emit_owl_restriction(
        staged, pc, _restricted_property(pc, idx), [], DEFAULT_NAMESPACE,
        subject=terms[pc.id], blank_handles=blank_handles,
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


def _pass_e_mirror(g, nodes, terms, idx, blank_handles, default_ns) -> None:
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

    owners: Dict[str, List[str]] = {}
    for nc_id, pc_ids in idx.nc_pcs.items():
        for pc_id in pc_ids:
            owners.setdefault(pc_id, []).extend(str(t) for t in idx.nc_targets.get(nc_id, []))

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
                    default_ns, blank_handles=blank_handles,
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
