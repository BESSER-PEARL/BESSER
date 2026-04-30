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
]


# We re-use the converter's ``ConversionError`` from the web backend if
# available, but importing it here would create a cross-layer dependency
# (web → notations is fine, but notations → web is not). Define a local
# alias and let callers translate.
class ResolutionError(ValueError):
    """Raised when a resolution payload is invalid or unknown."""


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
    # No-op default actions (converter's default already does the right thing):
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
