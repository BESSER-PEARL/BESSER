"""Knowledge Graph diagram processing — JSON → BUML.

Turns the KG diagram JSON (produced by the web editor / ``/import-owl``) into
a ``KnowledgeGraph`` metamodel instance.
"""

from __future__ import annotations

from typing import Any, Dict

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGProperty,
    KnowledgeGraph,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


_NODE_TYPE_TO_CLASS = {
    "class": KGClass,
    "individual": KGIndividual,
    "property": KGProperty,
    "blank": KGBlank,
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

    return kg


def _build_node(raw: Dict[str, Any]):
    node_type = (raw.get("nodeType") or "").lower()
    node_id = raw.get("id")
    if not node_id:
        raise ConversionError("KG node is missing 'id'.")
    label = raw.get("label", "") or ""
    iri = raw.get("iri")

    if node_type == "literal":
        value = raw.get("value")
        if value is None:
            value = label or ""
        datatype = raw.get("datatype")
        return KGLiteral(id=node_id, value=value, datatype=datatype, label=label)

    cls = _NODE_TYPE_TO_CLASS.get(node_type)
    if cls is None:
        raise ConversionError(
            f"Unknown KG nodeType '{raw.get('nodeType')!r}' (expected one of "
            "'class', 'individual', 'property', 'literal', 'blank')."
        )
    return cls(id=node_id, label=label, iri=iri)


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
    return KGEdge(
        id=edge_id,
        source=nodes_by_id[source_id],
        target=nodes_by_id[target_id],
        label=raw.get("label", "") or "",
        iri=raw.get("iri"),
    )


def _sanitize_name(name: str) -> str:
    if not name:
        return "knowledge_graph"
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    cleaned = cleaned.strip("_") or "knowledge_graph"
    if cleaned[0].isdigit():
        cleaned = f"kg_{cleaned}"
    return cleaned
