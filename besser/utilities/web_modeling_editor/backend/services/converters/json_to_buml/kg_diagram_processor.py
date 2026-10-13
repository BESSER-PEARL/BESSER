"""Knowledge Graph diagram processing — JSON → BUML.

Turns the KG diagram JSON (produced by the web editor / ``/import-owl``) into
a ``KnowledgeGraph`` metamodel instance.
"""

from __future__ import annotations

from typing import Any, Dict

from besser.BUML.metamodel.kg import (
    KGAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


# Derived from the class hierarchy, not hand-listed. The hand-written table
# was missing EquivalentPropertiesAxiom, and because an unknown kind was
# silently dropped, nothing failed: bibo.ttl simply came back with 51 of its
# 61 axioms. Deriving it means a new KGAxiom subclass round-trips the day it
# is added.
_AXIOM_KIND_TO_CLASS = {cls.__name__: cls for cls in KGAxiom.__subclasses__()}


_NODE_TYPE_TO_CLASS = {
    "class": KGClass,
    "individual": KGIndividual,
    "property": KGProperty,
    "blank": KGBlank,
    "nodeconstraint": KGNodeConstraint,
    "propertyconstraint": KGPropertyConstraint,
}


def process_kg_diagram(json_data: Dict[str, Any]) -> KnowledgeGraph:
    """Convert a KG diagram JSON payload into a ``KnowledgeGraph``.

    Accepts either the diagram envelope ``{"title": ..., "model": {...}}`` or
    the raw model dict ``{"nodes": [...], "edges": [...]}``.
    """
    model = json_data.get("model") if isinstance(json_data, dict) else None
    if not model:
        model = json_data or {}

    raw_title = (json_data.get("title") if isinstance(json_data, dict) else None) or "knowledge_graph"
    title = _sanitize_name(raw_title)

    kg = KnowledgeGraph(name=title)

    for raw_node in model.get("nodes", []) or []:
        node = _build_node(raw_node)
        kg.add_node(node)

    nodes_by_id = {n.id: n for n in kg.nodes}
    for raw_edge in model.get("edges", []) or []:
        edge = _build_edge(raw_edge, nodes_by_id)
        kg.add_edge(edge)

    for raw_axiom in model.get("axioms", []) or []:
        kg.add_axiom(_build_axiom(raw_axiom))

    return kg


def _build_axiom(raw: Dict[str, Any]):
    """Rebuild one axiom from its JSON form.

    Raises ``ConversionError`` rather than returning ``None`` for anything
    malformed, matching :func:`_build_node` / :func:`_build_edge`. Returning
    ``None`` meant a bad axiom vanished from the model without a word.
    """
    kind = raw.get("kind")
    if not kind:
        raise ConversionError("KG axiom is missing 'kind'.")
    cls = _AXIOM_KIND_TO_CLASS.get(kind)
    if cls is None:
        known = ", ".join(sorted(_AXIOM_KIND_TO_CLASS))
        raise ConversionError(f"Unknown KG axiom kind '{kind}'. Expected one of: {known}.")
    payload = {k: v for k, v in raw.items() if k != "kind"}
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ConversionError(f"Malformed '{kind}' axiom: {exc}") from exc


def _build_node(raw: Dict[str, Any]):
    node_type = (raw.get("nodeType") or "").lower()
    node_id = raw.get("id")
    if not node_id:
        raise ConversionError("KG node is missing 'id'.")
    label = raw.get("label", "") or ""
    iri = raw.get("iri")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else None

    if node_type == "literal":
        value = raw.get("value")
        if value is None:
            value = label or ""
        datatype = raw.get("datatype")
        return KGLiteral(id=node_id, value=value, datatype=datatype, label=label, metadata=metadata)

    cls = _NODE_TYPE_TO_CLASS.get(node_type)
    if cls is None:
        raise ConversionError(
            f"Unknown KG nodeType '{raw.get('nodeType')!r}' (expected one of "
            "'class', 'individual', 'property', 'literal', 'blank', "
            "'nodeConstraint', 'propertyConstraint')."
        )
    return cls(id=node_id, label=label, iri=iri, metadata=metadata)


def _build_edge(raw: Dict[str, Any], nodes_by_id: Dict[str, Any]) -> KGEdge:
    edge_id = raw.get("id")
    if not edge_id:
        raise ConversionError("KG edge is missing 'id'.")
    source_id = raw.get("source")
    target_id = raw.get("target")
    if source_id not in nodes_by_id:
        raise ConversionError(f"KG edge {edge_id!r} references unknown source node {source_id!r}.")
    if target_id not in nodes_by_id:
        raise ConversionError(f"KG edge {edge_id!r} references unknown target node {target_id!r}.")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else None
    return KGEdge(
        id=edge_id,
        source=nodes_by_id[source_id],
        target=nodes_by_id[target_id],
        label=raw.get("label", "") or "",
        iri=raw.get("iri"),
        metadata=metadata,
    )


def _sanitize_name(name: str) -> str:
    if not name:
        return "knowledge_graph"
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    cleaned = cleaned.strip("_") or "knowledge_graph"
    if cleaned[0].isdigit():
        cleaned = f"kg_{cleaned}"
    return cleaned
