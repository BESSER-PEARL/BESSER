"""KG → BUML resolutions: apply user choices to a KG before conversion.

The user reviews :class:`KGIssue` objects emitted by ``preflight.py`` and
picks a :class:`KGResolution` for each. This module rewrites the KG so
that the deterministic ``kg_to_class_diagram`` converter produces the
intended BUML model (e.g. attaches an ``rdfs:domain`` edge for a
property the user assigned, or drops a property the user no longer
wants).

Resolutions are applied to a **deep copy** of the KG so the caller's
graph is untouched. Unknown resolution keys raise ``ConversionError``
to surface protocol mismatches loudly.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGNode,
    KGProperty,
    KnowledgeGraph,
)
from besser.BUML.metamodel.kg.axioms import (
    DisjointClassesAxiom,
    DisjointUnionAxiom,
    EquivalentClassesAxiom,
    HasKeyAxiom,
    InversePropertiesAxiom,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)
from besser.BUML.notations.kg_to_buml._common import (
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUBCLASS_OF,
    RDF_TYPE,
    normalize_predicate,
)


__all__ = [
    "KGResolution",
    "apply_resolutions",
    "dispatch_decision",
    "ResolutionError",
    "DeferredOrphanClassification",
]


# We re-use the converter's ``ConversionError`` from the web backend if
# available, but importing it here would create a cross-layer dependency
# (web → notations is fine, but notations → web is not). Define a local
# alias and let callers translate.
class ResolutionError(ValueError):
    """Raised when a resolution payload is invalid or unknown."""


class DeferredOrphanClassification(Exception):
    """Raised when a resolution opts to defer orphan-node handling to the LLM.

    The web endpoint catches this exception, accumulates the carried node ids
    across resolutions, and returns them to the client as
    ``pendingOrphanClassification`` so the AI tab can call the per-node
    classifier next. The exception does NOT mutate the KG — the orphan nodes
    stay in place until the user reviews the LLM's per-node suggestions.
    """

    def __init__(self, node_ids: List[str]):
        super().__init__(
            f"Deferred to LLM classification: {len(node_ids)} node(s)"
        )
        self.node_ids: List[str] = list(node_ids)


@dataclass
class KGResolution:
    """A user choice for a single :class:`KGIssue`.

    Args:
        issue_id: The ``id`` of the :class:`KGIssue` this responds to.
        choice: The ``key`` of the chosen :class:`KGResolutionOption`.
        parameters: User-supplied values matching the option's
            ``parameters_schema``.
    """

    issue_id: str
    choice: str
    parameters: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Applier
# ----------------------------------------------------------------------


def apply_resolutions(kg: KnowledgeGraph, resolutions: List[KGResolution]) -> KnowledgeGraph:
    """Apply a list of resolutions to a *deep copy* of ``kg`` and return it.

    Resolutions are dispatched by ``choice`` key. Unknown choices raise
    :class:`ResolutionError`.
    """
    if not resolutions:
        return kg
    new_kg = copy.deepcopy(kg)
    for res in resolutions:
        handler = _HANDLERS.get(res.choice)
        if handler is None:
            raise ResolutionError(f"Unknown resolution choice {res.choice!r}.")
        handler(new_kg, res)
    return new_kg


def dispatch_decision(kg: KnowledgeGraph, issue, decision: str) -> KnowledgeGraph:
    """Apply a single ``"accept"`` / ``"skip"`` decision against ``issue``.

    Looks up the issue's ``recommended_action`` (on ``"accept"``) or
    ``skip_action`` (on ``"skip"``), constructs a synthetic
    :class:`KGResolution` with the action's pre-filled parameters, and
    dispatches via :func:`apply_resolutions`. Returns the (deep-copied)
    mutated KG.
    """
    if decision == "accept":
        action = issue.recommended_action
    elif decision == "skip":
        action = issue.skip_action
    else:
        raise ResolutionError(f"Unknown decision {decision!r}; expected 'accept' or 'skip'.")
    if action is None:
        # Some issues have no actionable resolution on one side — treat as no-op.
        return kg
    res = KGResolution(issue_id=issue.id, choice=action.key, parameters=dict(action.parameters))
    return apply_resolutions(kg, [res])


def _next_edge_id(kg: KnowledgeGraph, prefix: str = "edge") -> str:
    used = {e.id for e in kg.edges}
    while True:
        candidate = f"{prefix}:{uuid.uuid4().hex[:8]}"
        if candidate not in used:
            return candidate


def _require_class(kg: KnowledgeGraph, class_id: str) -> KGClass:
    node = kg.get_node(class_id)
    if node is None or not isinstance(node, KGClass):
        raise ResolutionError(f"Class {class_id!r} not found in KG.")
    return node


def _require_property(kg: KnowledgeGraph, prop_id: str) -> KGProperty:
    node = kg.get_node(prop_id)
    if node is None or not isinstance(node, KGProperty):
        raise ResolutionError(f"Property {prop_id!r} not found in KG.")
    return node


def _ensure_thing_class(kg: KnowledgeGraph) -> KGClass:
    """Return a synthetic 'Thing' KGClass, creating it if absent."""
    for n in kg.nodes:
        if isinstance(n, KGClass) and (n.label == "Thing" or n.id.endswith("/Thing")):
            return n
    thing = KGClass(id="urn:besser:Thing", label="Thing", iri="urn:besser:Thing")
    kg.add_node(thing)
    return thing


# ----------------------------------------------------------------------
# Handlers — keyed by KGResolutionOption.key
# ----------------------------------------------------------------------


def _h_assign_domain(kg: KnowledgeGraph, res: KGResolution) -> None:
    prop_iri = _affected_property_iri(res)
    class_id = res.parameters.get("class_id")
    if not (prop_iri and class_id):
        raise ResolutionError("'assign_domain' requires 'property_iri' (issue context) and 'class_id'.")
    prop = _find_property_by_iri(kg, prop_iri)
    cls = _require_class(kg, class_id)
    if prop is None:
        raise ResolutionError(f"Property {prop_iri!r} not found in KG.")
    kg.add_edge(KGEdge(
        id=_next_edge_id(kg, "domain"),
        source=prop,
        target=cls,
        label="domain",
        iri=RDFS_DOMAIN,
    ))


def _h_attach_to_thing(kg: KnowledgeGraph, res: KGResolution) -> None:
    prop_iri = _affected_property_iri(res)
    if not prop_iri:
        raise ResolutionError("'attach_to_thing' requires 'property_iri' on the issue context.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        raise ResolutionError(f"Property {prop_iri!r} not found in KG.")
    thing = _ensure_thing_class(kg)
    kg.add_edge(KGEdge(
        id=_next_edge_id(kg, "domain"),
        source=prop,
        target=thing,
        label="domain",
        iri=RDFS_DOMAIN,
    ))


def _h_drop_property(kg: KnowledgeGraph, res: KGResolution) -> None:
    prop_iri = _affected_property_iri(res)
    if not prop_iri:
        raise ResolutionError("'drop_property' requires 'property_iri' on the issue context.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        return
    # Remove all edges touching the property and the property node itself.
    surviving_edges = {e for e in kg.edges if e.source.id != prop.id and e.target.id != prop.id}
    surviving_nodes = {n for n in kg.nodes if n.id != prop.id}
    _replace_graph(kg, surviving_nodes, surviving_edges)


def _h_pick_domain(kg: KnowledgeGraph, res: KGResolution) -> None:
    prop_iri = _affected_property_iri(res)
    keep_class_id = res.parameters.get("class_id")
    if not (prop_iri and keep_class_id):
        raise ResolutionError("'pick_domain' requires 'property_iri' (context) and 'class_id'.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        return
    surviving_edges = set()
    for e in kg.edges:
        if e.source.id == prop.id and normalize_predicate(e.iri) == RDFS_DOMAIN and e.target.id != keep_class_id:
            continue
        surviving_edges.add(e)
    _replace_graph(kg, set(kg.nodes), surviving_edges)


def _h_split_per_domain(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Clone the property once per domain, each with a single rdfs:domain edge."""
    prop_iri = _affected_property_iri(res)
    if not prop_iri:
        raise ResolutionError("'split_per_domain' requires 'property_iri' on the issue context.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        return
    domain_edges = [
        e for e in kg.edges
        if e.source.id == prop.id and normalize_predicate(e.iri) == RDFS_DOMAIN
    ]
    if len(domain_edges) <= 1:
        return
    # Keep the first domain on the original property; clone for the rest.
    keep_edge, *other_edges = domain_edges
    new_nodes: Set[KGNode] = set(kg.nodes)
    new_edges: Set[KGEdge] = {e for e in kg.edges if e not in other_edges}
    for idx, e in enumerate(other_edges, start=2):
        clone_id = f"{prop.id}#split-{idx}"
        clone = KGProperty(
            id=clone_id,
            label=f"{prop.label}_{idx}" if prop.label else "",
            iri=prop.iri,
            metadata=dict(prop.metadata),
        )
        new_nodes.add(clone)
        new_edges.add(KGEdge(
            id=_next_edge_id(kg, "domain"),
            source=clone,
            target=e.target,
            label="domain",
            iri=RDFS_DOMAIN,
        ))
        # Mirror the range edges onto the clone.
        for re in kg.edges:
            if re.source.id == prop.id and normalize_predicate(re.iri) == RDFS_RANGE:
                new_edges.add(KGEdge(
                    id=_next_edge_id(kg, "range"),
                    source=clone,
                    target=re.target,
                    label="range",
                    iri=RDFS_RANGE,
                ))
    _replace_graph(kg, new_nodes, new_edges)


def _h_attach_to_common_ancestor(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Replace the property's multiple rdfs:domain edges with one targeting the common ancestor."""
    prop_iri = _affected_property_iri(res)
    ancestor_id = res.parameters.get("class_id")
    if not (prop_iri and ancestor_id):
        raise ResolutionError(
            "'attach_to_common_ancestor' requires 'property_iri' (context) and 'class_id'."
        )
    prop = _find_property_by_iri(kg, prop_iri)
    ancestor = _require_class(kg, ancestor_id)
    if prop is None:
        return
    new_edges = {
        e for e in kg.edges
        if not (e.source.id == prop.id and normalize_predicate(e.iri) == RDFS_DOMAIN)
    }
    new_edges.add(KGEdge(
        id=_next_edge_id(kg, "domain"),
        source=prop,
        target=ancestor,
        label="domain",
        iri=RDFS_DOMAIN,
    ))
    _replace_graph(kg, set(kg.nodes), new_edges)


def _h_attach_to_class(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Attach an unattached restriction blank to a class via rdfs:subClassOf."""
    issue_context = res.parameters
    blank_id = issue_context.get("blank_id") or _affected_blank_id(res)
    class_id = issue_context.get("class_id")
    if not (blank_id and class_id):
        raise ResolutionError("'attach_to_class' requires 'blank_id' (or first affected node) and 'class_id'.")
    blank = kg.get_node(blank_id)
    cls = _require_class(kg, class_id)
    if blank is None:
        raise ResolutionError(f"Blank {blank_id!r} not found in KG.")
    kg.add_edge(KGEdge(
        id=_next_edge_id(kg, "subClassOf"),
        source=cls,
        target=blank,
        label="subClassOf",
        iri=RDFS_SUBCLASS_OF,
    ))


def _h_drop_restriction(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Remove the unattached restriction blank node and all incident edges."""
    blank_id = res.parameters.get("blank_id") or _affected_blank_id(res)
    if not blank_id:
        raise ResolutionError("'drop_restriction' requires 'blank_id' on the issue context.")
    surviving_edges = {e for e in kg.edges if e.source.id != blank_id and e.target.id != blank_id}
    surviving_nodes = {n for n in kg.nodes if n.id != blank_id}
    _replace_graph(kg, surviving_nodes, surviving_edges)


def _h_break_at_edge(kg: KnowledgeGraph, res: KGResolution) -> None:
    edge_id = res.parameters.get("edge_id")
    if not edge_id:
        raise ResolutionError("'break_at_edge' requires 'edge_id'.")
    surviving_edges = {e for e in kg.edges if e.id != edge_id}
    if len(surviving_edges) == len(kg.edges):
        raise ResolutionError(f"Edge {edge_id!r} not found in KG.")
    _replace_graph(kg, set(kg.nodes), surviving_edges)


def _h_merge_classes(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Merge ``drop_class_id`` into ``keep_class_id``: edges are rewritten."""
    keep_id = res.parameters.get("keep_class_id") or res.parameters.get("target_class_id")
    drop_id = res.parameters.get("drop_class_id")
    # Equivalence-merge variant: only target_class_id present; merge the
    # *other* affected class into target.
    if keep_id and not drop_id:
        affected = res.parameters.get("class_ids") or []
        for c in affected:
            if c != keep_id:
                drop_id = c
                break
    if not (keep_id and drop_id):
        raise ResolutionError("'merge_classes' requires 'keep_class_id' and 'drop_class_id' (or 'target_class_id' + 'class_ids').")
    if keep_id == drop_id:
        return
    keep = _require_class(kg, keep_id)
    drop = _require_class(kg, drop_id)
    new_edges: Set[KGEdge] = set()
    for e in kg.edges:
        src = keep if e.source.id == drop.id else e.source
        tgt = keep if e.target.id == drop.id else e.target
        if src is e.source and tgt is e.target:
            new_edges.add(e)
        else:
            new_edges.add(KGEdge(
                id=e.id, source=src, target=tgt,
                label=e.label, iri=e.iri, metadata=dict(e.metadata),
            ))
    new_nodes = {n for n in kg.nodes if n.id != drop.id}
    _replace_graph(kg, new_nodes, new_edges)


def _h_merge_properties(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Merge property-name-collision peers into a single property by IRI of the first one."""
    affected_iris: List[str] = list(res.parameters.get("property_iris", []))
    if len(affected_iris) < 2:
        raise ResolutionError("'merge_properties' requires at least two 'property_iris'.")
    keep_iri, *drop_iris = affected_iris
    keep = _find_property_by_iri(kg, keep_iri)
    if keep is None:
        return
    new_edges: Set[KGEdge] = set()
    new_nodes = set(kg.nodes)
    for e in kg.edges:
        if e.iri in drop_iris:
            new_edges.add(KGEdge(id=e.id, source=e.source, target=e.target, label=e.label, iri=keep_iri, metadata=dict(e.metadata)))
        else:
            new_edges.add(e)
    for drop_iri in drop_iris:
        drop = _find_property_by_iri(kg, drop_iri)
        if drop is not None:
            new_nodes = {n for n in new_nodes if n.id != drop.id}
            new_edges = {e for e in new_edges if e.source.id != drop.id and e.target.id != drop.id}
    _replace_graph(kg, new_nodes, new_edges)


def _h_rename(kg: KnowledgeGraph, res: KGResolution) -> None:
    target_id = res.parameters.get("target_property_id")
    new_name = res.parameters.get("new_name")
    if not (target_id and new_name):
        raise ResolutionError("'rename' requires 'target_property_id' and 'new_name'.")
    prop = _require_property(kg, target_id)
    prop.label = new_name


def _h_keep_object_property(kg: KnowledgeGraph, res: KGResolution) -> None:
    """For BLOCK_RANGE_BOTH_DATATYPE_AND_CLASS: drop the datatype range edges."""
    prop_iri = _affected_property_iri(res)
    if not prop_iri:
        raise ResolutionError("'keep_object_property' requires 'property_iri' on the issue context.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        return
    from besser.BUML.notations.kg_to_buml.kg_to_class_diagram import _looks_like_datatype_iri
    new_edges = {
        e for e in kg.edges
        if not (
            e.source.id == prop.id
            and normalize_predicate(e.iri) == RDFS_RANGE
            and _looks_like_datatype_iri(getattr(e.target, "iri", None))
        )
    }
    _replace_graph(kg, set(kg.nodes), new_edges)


def _h_keep_datatype_property(kg: KnowledgeGraph, res: KGResolution) -> None:
    """For BLOCK_RANGE_BOTH_DATATYPE_AND_CLASS: drop the class-range edges, keep only datatype."""
    prop_iri = _affected_property_iri(res)
    if not prop_iri:
        raise ResolutionError("'keep_datatype_property' requires 'property_iri' on the issue context.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        return
    from besser.BUML.notations.kg_to_buml.kg_to_class_diagram import _looks_like_datatype_iri
    new_edges = {
        e for e in kg.edges
        if not (
            e.source.id == prop.id
            and normalize_predicate(e.iri) == RDFS_RANGE
            and not _looks_like_datatype_iri(getattr(e.target, "iri", None))
            and isinstance(e.target, KGClass)
        )
    }
    _replace_graph(kg, set(kg.nodes), new_edges)


def _h_set_range(kg: KnowledgeGraph, res: KGResolution) -> None:
    prop_iri = _affected_property_iri(res)
    range_iri = res.parameters.get("range_iri")
    if not (prop_iri and range_iri):
        raise ResolutionError("'set_range' requires 'property_iri' (context) and 'range_iri'.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        return
    # Find or create a node for the range IRI.
    target = kg.get_node(range_iri)
    if target is None:
        target = KGClass(id=range_iri, label="", iri=range_iri)
        kg.add_node(target)
    kg.add_edge(KGEdge(
        id=_next_edge_id(kg, "range"),
        source=prop,
        target=target,
        label="range",
        iri=RDFS_RANGE,
    ))


# Default-action choices: no-op handlers (the converter's existing behaviour
# already implements them).
def _h_noop(kg: KnowledgeGraph, res: KGResolution) -> None:
    return


# ----------------------------------------------------------------------
# v2 handlers (new detectors)
# ----------------------------------------------------------------------


def _h_materialize_as_individual(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Promote a KGBlank to a KGIndividual with a synthetic IRI."""
    blank_id = res.parameters.get("blank_id")
    if not blank_id:
        raise ResolutionError("'materialize_as_individual' requires 'blank_id'.")
    blank = kg.get_node(blank_id)
    if blank is None or not isinstance(blank, KGBlank):
        return
    # Synthesise an IRI for the new individual.
    new_iri = f"urn:besser:materialised:{blank_id.lstrip('_:')}"
    new_id = new_iri
    individual = KGIndividual(
        id=new_id,
        label=blank.label or new_id,
        iri=new_iri,
        metadata={"materialised_from_blank": blank_id},
    )
    # Rewire incident edges through the new individual.
    new_nodes = {n for n in kg.nodes if n.id != blank.id}
    new_nodes.add(individual)
    new_edges: Set[KGEdge] = set()
    for e in kg.edges:
        src = individual if e.source.id == blank.id else e.source
        tgt = individual if e.target.id == blank.id else e.target
        if src is e.source and tgt is e.target:
            new_edges.add(e)
        else:
            new_edges.add(KGEdge(
                id=e.id, source=src, target=tgt,
                label=e.label, iri=e.iri, metadata=dict(e.metadata),
            ))
    _replace_graph(kg, new_nodes, new_edges)


def _h_drop_node(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Remove a node and all incident edges."""
    node_id = res.parameters.get("node_id")
    if not node_id:
        raise ResolutionError("'drop_node' requires 'node_id'.")
    surviving_nodes = {n for n in kg.nodes if n.id != node_id}
    surviving_edges = {e for e in kg.edges if e.source.id != node_id and e.target.id != node_id}
    _replace_graph(kg, surviving_nodes, surviving_edges)


def _h_drop_references(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Remove edges referencing a node as a class (rdf:type targets, rdfs:domain
    targets, rdfs:range targets, rdfs:subClassOf endpoints) but keep the node."""
    node_id = res.parameters.get("node_id")
    if not node_id:
        raise ResolutionError("'drop_references' requires 'node_id'.")
    surviving = set()
    for e in kg.edges:
        pred = normalize_predicate(e.iri)
        if pred in (RDF_TYPE,) and e.target.id == node_id:
            continue
        if pred in (RDFS_DOMAIN, RDFS_RANGE) and e.target.id == node_id:
            continue
        if pred == RDFS_SUBCLASS_OF and (e.source.id == node_id or e.target.id == node_id):
            continue
        surviving.add(e)
    _replace_graph(kg, set(kg.nodes), surviving)


def _h_treat_as_string(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Drop the property's existing rdfs:range edges and add an xsd:string range."""
    prop_iri = res.parameters.get("property_iri")
    if not prop_iri:
        raise ResolutionError("'treat_as_string' requires 'property_iri'.")
    prop = _find_property_by_iri(kg, prop_iri)
    if prop is None:
        return
    new_edges = {
        e for e in kg.edges
        if not (e.source.id == prop.id and normalize_predicate(e.iri) == RDFS_RANGE)
    }
    # Ensure xsd:string node exists.
    xsd_string = "http://www.w3.org/2001/XMLSchema#string"
    target = kg.get_node(xsd_string)
    new_nodes = set(kg.nodes)
    if target is None:
        target = KGClass(id=xsd_string, label="string", iri=xsd_string)
        new_nodes.add(target)
    new_edges.add(KGEdge(
        id=_next_edge_id(kg, "range"),
        source=prop, target=target,
        label="range", iri=RDFS_RANGE,
    ))
    _replace_graph(kg, new_nodes, new_edges)


def _h_keep_first_only(kg: KnowledgeGraph, res: KGResolution) -> None:
    """For (individual, property) edges to literals: keep the lex-first edge,
    drop the rest."""
    prop_iri = res.parameters.get("property_iri")
    indiv_id = res.parameters.get("individual_id")
    if not prop_iri:
        raise ResolutionError("'keep_first_only' requires 'property_iri'.")
    candidates = [
        e for e in kg.edges
        if e.iri == prop_iri
        and isinstance(e.target, KGLiteral)
        and (indiv_id is None or e.source.id == indiv_id)
    ]
    if len(candidates) <= 1:
        return
    candidates.sort(key=lambda e: e.id)
    keep = candidates[0]
    surviving = {e for e in kg.edges if e is keep or e not in candidates}
    _replace_graph(kg, set(kg.nodes), surviving)


def _h_assign_thing_class(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Add an rdf:type → Thing edge for an individual."""
    indiv_id = res.parameters.get("individual_id")
    if not indiv_id:
        raise ResolutionError("'assign_thing_class' requires 'individual_id'.")
    indiv = kg.get_node(indiv_id)
    if indiv is None:
        return
    thing = _ensure_thing_class(kg)
    kg.add_edge(KGEdge(
        id=_next_edge_id(kg, "type"),
        source=indiv, target=thing,
        label="type", iri=RDF_TYPE,
    ))


def _h_pick_most_specific(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Drop all rdf:type edges from `individual_id` except the one targeting `class_id`."""
    indiv_id = res.parameters.get("individual_id")
    class_id = res.parameters.get("class_id")
    if not (indiv_id and class_id):
        raise ResolutionError("'pick_most_specific' requires 'individual_id' and 'class_id'.")
    surviving = set()
    for e in kg.edges:
        if (
            e.source.id == indiv_id
            and normalize_predicate(e.iri) == RDF_TYPE
            and e.target.id != class_id
        ):
            continue
        surviving.add(e)
    _replace_graph(kg, set(kg.nodes), surviving)


def _h_drop_slot(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Drop the edges targeting a literal node (and the literal node itself)."""
    literal_id = res.parameters.get("literal_id")
    if not literal_id:
        raise ResolutionError("'drop_slot' requires 'literal_id'.")
    surviving_edges = {e for e in kg.edges if e.source.id != literal_id and e.target.id != literal_id}
    surviving_nodes = {n for n in kg.nodes if n.id != literal_id}
    _replace_graph(kg, surviving_nodes, surviving_edges)


def _h_drop_link(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Drop a single edge by id."""
    edge_id = res.parameters.get("edge_id")
    if not edge_id:
        raise ResolutionError("'drop_link' requires 'edge_id'.")
    surviving = {e for e in kg.edges if e.id != edge_id}
    _replace_graph(kg, set(kg.nodes), surviving)


def _h_rename_with_suffix(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Rename a list of properties with deterministic name suffixes.

    Parameters: ``renames: List[{target_property_id, new_name}]``.
    """
    renames = res.parameters.get("renames") or []
    for entry in renames:
        target_id = entry.get("target_property_id")
        new_name = entry.get("new_name")
        if not (target_id and new_name):
            continue
        prop = kg.get_node(target_id)
        if prop is not None:
            prop.label = new_name


def _h_prefer_class(kg: KnowledgeGraph, res: KGResolution) -> None:
    """For OWL2 punning: drop the individual twin node."""
    indiv_id = res.parameters.get("individual_id")
    if not indiv_id:
        raise ResolutionError("'prefer_class' requires 'individual_id'.")
    surviving_nodes = {n for n in kg.nodes if n.id != indiv_id}
    surviving_edges = {e for e in kg.edges if e.source.id != indiv_id and e.target.id != indiv_id}
    _replace_graph(kg, surviving_nodes, surviving_edges)


def _h_merge_associations(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Merge two inverse properties into one: rewrite all references to property B
    to point at property A, then drop B's node."""
    a_iri = res.parameters.get("property_a_iri")
    b_iri = res.parameters.get("property_b_iri")
    if not (a_iri and b_iri):
        raise ResolutionError("'merge_associations' requires 'property_a_iri' and 'property_b_iri'.")
    a = _find_property_by_iri(kg, a_iri)
    b = _find_property_by_iri(kg, b_iri)
    if a is None or b is None:
        return
    new_edges = set()
    for e in kg.edges:
        src = a if e.source.id == b.id else e.source
        tgt = a if e.target.id == b.id else e.target
        if src is e.source and tgt is e.target:
            new_edges.add(e)
        else:
            new_edges.add(KGEdge(
                id=e.id, source=src, target=tgt,
                label=e.label, iri=e.iri, metadata=dict(e.metadata),
            ))
    new_nodes = {n for n in kg.nodes if n.id != b.id}
    _replace_graph(kg, new_nodes, new_edges)


# ----------------------------------------------------------------------
# LLM-cleanup handlers (used by the AI cleanup flow). These handlers
# operate by node id rather than property/blank IRI because the LLM
# emits suggestions keyed off the node ids it saw in the snapshot.
# ----------------------------------------------------------------------


def _h_drop_class(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Drop a KGClass node and all incident edges; clean up dangling axioms."""
    node_id = res.parameters.get("node_id")
    if not node_id:
        raise ResolutionError("'drop_class' requires 'node_id'.")
    node = kg.get_node(node_id)
    if node is None:
        return
    if not isinstance(node, KGClass):
        raise ResolutionError(f"'drop_class' expected a KGClass, got {type(node).__name__} for {node_id!r}.")
    surviving_nodes = {n for n in kg.nodes if n.id != node_id}
    surviving_edges = {e for e in kg.edges if e.source.id != node_id and e.target.id != node_id}
    _replace_graph(kg, surviving_nodes, surviving_edges)
    _strip_dangling_axioms(kg)


def _h_drop_individual(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Drop a KGIndividual node and all incident edges; clean up dangling axioms."""
    node_id = res.parameters.get("node_id")
    if not node_id:
        raise ResolutionError("'drop_individual' requires 'node_id'.")
    node = kg.get_node(node_id)
    if node is None:
        return
    if not isinstance(node, KGIndividual):
        raise ResolutionError(f"'drop_individual' expected a KGIndividual, got {type(node).__name__} for {node_id!r}.")
    surviving_nodes = {n for n in kg.nodes if n.id != node_id}
    surviving_edges = {e for e in kg.edges if e.source.id != node_id and e.target.id != node_id}
    _replace_graph(kg, surviving_nodes, surviving_edges)
    _strip_dangling_axioms(kg)


def _h_promote_individual_to_class(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Promote a KGIndividual to a KGClass with the same id, rewiring edges.

    Drops any outgoing rdf:type edges from the promoted node (a class is not
    typed by another class via rdf:type in the BUML target).
    """
    node_id = res.parameters.get("node_id")
    if not node_id:
        raise ResolutionError("'promote_individual_to_class' requires 'node_id'.")
    node = kg.get_node(node_id)
    if node is None:
        return
    if not isinstance(node, KGIndividual):
        raise ResolutionError(
            f"'promote_individual_to_class' expected a KGIndividual, got {type(node).__name__} for {node_id!r}."
        )
    new_class = KGClass(
        id=node.id,
        label=res.parameters.get("new_label") or node.label,
        iri=res.parameters.get("new_iri") or node.iri,
        metadata=dict(node.metadata),
    )
    _swap_node(kg, node, new_class, drop_outgoing_predicates={RDF_TYPE})


def _h_type_individual_as_class(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Add an ``rdf:type`` edge from a ``KGIndividual`` to a ``KGClass``.

    Used by the LLM cleanup flow when an individual is relevant to the
    described system but lacks a class link — instead of dropping it (the
    aggressive default), the LLM can recommend the right typing.

    Parameters: ``{node_id, class_id}``. Both must point at existing nodes;
    ``node_id`` must be a ``KGIndividual`` and ``class_id`` a ``KGClass``.
    Idempotent: if the rdf:type edge already exists, it is not duplicated.
    """
    node_id = res.parameters.get("node_id")
    class_id = res.parameters.get("class_id")
    if not (node_id and class_id):
        raise ResolutionError("'type_individual_as_class' requires 'node_id' and 'class_id'.")
    indiv = kg.get_node(node_id)
    if indiv is None:
        return
    if not isinstance(indiv, KGIndividual):
        raise ResolutionError(
            f"'type_individual_as_class' expected a KGIndividual, got {type(indiv).__name__} for {node_id!r}."
        )
    cls = _require_class(kg, class_id)
    # Avoid duplicate rdf:type edges between the same (individual, class) pair.
    for e in kg.edges:
        if (
            e.source.id == indiv.id
            and e.target.id == cls.id
            and normalize_predicate(e.iri) == RDF_TYPE
        ):
            return
    kg.add_edge(KGEdge(
        id=_next_edge_id(kg, "type"),
        source=indiv,
        target=cls,
        label="type",
        iri=RDF_TYPE,
    ))


def _h_drop_orphan_nodes(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Drop a batch of orphan node ids and all incident edges.

    Tolerates ids missing from the KG (idempotent) so the same accept-decision
    can be replayed safely if the user re-applies. Cleans up dangling axioms
    after the batch removal.
    """
    node_ids = res.parameters.get("node_ids") or []
    if not isinstance(node_ids, list):
        raise ResolutionError("'drop_orphan_nodes' requires 'node_ids' as a list.")
    drop = set(node_ids)
    if not drop:
        return
    surviving_nodes = {n for n in kg.nodes if n.id not in drop}
    surviving_edges = {
        e for e in kg.edges if e.source.id not in drop and e.target.id not in drop
    }
    _replace_graph(kg, surviving_nodes, surviving_edges)
    _strip_dangling_axioms(kg)


def _h_defer_to_llm_classification(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Deferred handler: signals that the user picked 'send to LLM for
    classification' for a batch of orphan nodes. Does NOT mutate the KG.

    The web endpoint catches :class:`DeferredOrphanClassification`, accumulates
    the carried node ids across resolutions, and returns them to the client as
    ``pendingOrphanClassification`` so the AI tab can call the per-node
    classifier next.
    """
    node_ids = res.parameters.get("node_ids") or []
    if not isinstance(node_ids, list):
        raise ResolutionError(
            "'defer_to_llm_classification' requires 'node_ids' as a list."
        )
    raise DeferredOrphanClassification(node_ids)


def _h_reclassify_node(kg: KnowledgeGraph, res: KGResolution) -> None:
    """Change a node's kind (class / individual / property / blank) in place.

    Preserves the node's id, label, iri, metadata, and rewires every edge
    that referenced the old node to point at the new one.
    """
    node_id = res.parameters.get("node_id")
    target_kind = (res.parameters.get("target_kind") or "").lower()
    if not (node_id and target_kind):
        raise ResolutionError("'reclassify_node' requires 'node_id' and 'target_kind'.")
    target_cls = {
        "class": KGClass,
        "individual": KGIndividual,
        "property": KGProperty,
        "blank": KGBlank,
    }.get(target_kind)
    if target_cls is None:
        raise ResolutionError(
            f"'reclassify_node' got unsupported target_kind {target_kind!r}; "
            f"expected one of class, individual, property, blank."
        )
    node = kg.get_node(node_id)
    if node is None:
        return
    if isinstance(node, target_cls):
        return
    new_node = target_cls(
        id=node.id,
        label=node.label,
        iri=getattr(node, "iri", None),
        metadata=dict(node.metadata),
    )
    _swap_node(kg, node, new_node)


# ----------------------------------------------------------------------
# Helpers for LLM-cleanup handlers
# ----------------------------------------------------------------------


def _swap_node(
    kg: KnowledgeGraph,
    old: KGNode,
    new: KGNode,
    *,
    drop_outgoing_predicates: Optional[Set[str]] = None,
) -> None:
    """Replace ``old`` with ``new`` in ``kg`` and rewire incident edges.

    ``drop_outgoing_predicates``: if provided, edges whose ``source`` is the
    swapped node and whose normalised predicate is in the set are dropped
    rather than rewired.
    """
    drop_outgoing_predicates = drop_outgoing_predicates or set()
    new_nodes = {n for n in kg.nodes if n.id != old.id}
    new_nodes.add(new)
    new_edges: Set[KGEdge] = set()
    for e in kg.edges:
        if e.source.id == old.id and normalize_predicate(e.iri) in drop_outgoing_predicates:
            continue
        src = new if e.source.id == old.id else e.source
        tgt = new if e.target.id == old.id else e.target
        if src is e.source and tgt is e.target:
            new_edges.add(e)
        else:
            new_edges.add(KGEdge(
                id=e.id, source=src, target=tgt,
                label=e.label, iri=e.iri, metadata=dict(e.metadata),
            ))
    _replace_graph(kg, new_nodes, new_edges)


def _strip_dangling_axioms(kg: KnowledgeGraph) -> None:
    """Remove (or shrink) axioms that reference node ids no longer present.

    For multi-id axioms (equivalent / disjoint / disjoint-union / property-chain
    / has-key) we filter out missing ids; if fewer than the minimum required
    ids remain, the axiom is dropped entirely. For pair-axioms (sub-property,
    inverse) the axiom is dropped if either endpoint is missing.
    """
    live_ids = {n.id for n in kg.nodes}
    surviving: List = []
    for axiom in kg.axioms:
        if isinstance(axiom, (EquivalentClassesAxiom, DisjointClassesAxiom)):
            kept = [c for c in axiom.class_ids if c in live_ids]
            if len(kept) >= 2:
                axiom.class_ids = kept
                surviving.append(axiom)
        elif isinstance(axiom, DisjointUnionAxiom):
            if axiom.union_class_id in live_ids:
                kept_parts = [c for c in axiom.part_class_ids if c in live_ids]
                if len(kept_parts) >= 1:
                    axiom.part_class_ids = kept_parts
                    surviving.append(axiom)
        elif isinstance(axiom, SubPropertyOfAxiom):
            if axiom.sub_property_id in live_ids and axiom.super_property_id in live_ids:
                surviving.append(axiom)
        elif isinstance(axiom, InversePropertiesAxiom):
            if axiom.property_a_id in live_ids and axiom.property_b_id in live_ids:
                surviving.append(axiom)
        elif isinstance(axiom, PropertyChainAxiom):
            if axiom.property_id in live_ids:
                kept = [p for p in axiom.chain_property_ids if p in live_ids]
                if len(kept) >= 1:
                    axiom.chain_property_ids = kept
                    surviving.append(axiom)
        elif isinstance(axiom, HasKeyAxiom):
            if axiom.class_id in live_ids:
                kept = [p for p in axiom.property_ids if p in live_ids]
                if len(kept) >= 1:
                    axiom.property_ids = kept
                    surviving.append(axiom)
        else:
            # Unknown / Import axioms — leave as-is.
            surviving.append(axiom)
    kg.axioms = surviving


_HANDLERS: Dict[str, Any] = {
    # v1 handlers (still used by accept/skip dispatching)
    "assign_domain": _h_assign_domain,
    "attach_to_thing": _h_attach_to_thing,
    "drop_property": _h_drop_property,
    "pick_domain": _h_pick_domain,
    "split_per_domain": _h_split_per_domain,
    "attach_to_common_ancestor": _h_attach_to_common_ancestor,
    "attach_to_class": _h_attach_to_class,
    "drop_restriction": _h_drop_restriction,
    "break_at_edge": _h_break_at_edge,
    "merge_classes": _h_merge_classes,
    "merge_properties": _h_merge_properties,
    "rename": _h_rename,
    "keep_object_property": _h_keep_object_property,
    "keep_datatype_property": _h_keep_datatype_property,
    "set_range": _h_set_range,
    # v2 handlers
    "materialize_as_individual": _h_materialize_as_individual,
    "materialize_as_object": _h_materialize_as_individual,  # alias
    "drop_node": _h_drop_node,
    "drop_references": _h_drop_references,
    "treat_as_string": _h_treat_as_string,
    "keep_first_only": _h_keep_first_only,
    "assign_thing_class": _h_assign_thing_class,
    "pick_most_specific": _h_pick_most_specific,
    "drop_slot": _h_drop_slot,
    "drop_link": _h_drop_link,
    "rename_with_suffix": _h_rename_with_suffix,
    "prefer_class": _h_prefer_class,
    "merge_associations": _h_merge_associations,
    # LLM-cleanup handlers
    "drop_class": _h_drop_class,
    "drop_individual": _h_drop_individual,
    "promote_individual_to_class": _h_promote_individual_to_class,
    "reclassify_node": _h_reclassify_node,
    "type_individual_as_class": _h_type_individual_as_class,
    # Orphan-node refinement handlers
    "drop_orphan_nodes": _h_drop_orphan_nodes,
    "defer_to_llm_classification": _h_defer_to_llm_classification,
    # No-op default actions (converter's default already does the right thing):
    "noop": _h_noop,
    "keep_separate": _h_noop,
    "keep_both": _h_noop,
    "keep_as_string": _h_noop,
    "keep_as_single": _h_noop,
    "ignore": _h_noop,
    "bump_to_unbounded": _h_noop,           # converter already bumps via ABox heuristic
    "coerce_to_string": _h_noop,            # converter already coerces
    "coerce_to_string_link": _h_noop,       # the link is preserved as raw edge
    "record_as_description": _h_noop,       # writer can read restriction metadata
    "synthesize_class": _h_noop,            # converter already synthesises
}


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _affected_property_iri(res: KGResolution) -> Optional[str]:
    """Some handlers want the property IRI from issue context; fall back to params."""
    iri = res.parameters.get("property_iri")
    return iri


def _affected_blank_id(res: KGResolution) -> Optional[str]:
    return res.parameters.get("blank_id")


def _find_property_by_iri(kg: KnowledgeGraph, iri: str) -> Optional[KGProperty]:
    if not iri:
        return None
    for n in kg.nodes:
        if isinstance(n, KGProperty) and (n.iri == iri or n.id == iri):
            return n
    return None


def _replace_graph(kg: KnowledgeGraph, nodes: Set[KGNode], edges: Set[KGEdge]) -> None:
    """Replace the KG's nodes and edges in place, preserving validation order."""
    # Setting nodes first invalidates edge endpoint validation, so clear edges then nodes.
    kg.edges = set()
    kg.nodes = nodes
    kg.edges = edges
