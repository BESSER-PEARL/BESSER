"""Helpers for reading the v4 ``{nodes[], edges[]}`` wire shape.

The v4 React-Flow wire shape is a flat list of nodes and edges (see
``docs/source/migrations/uml-v4-shape.md``). These helpers replace the v3
idioms ``elements[id]`` / ``relationships[id]`` and the manual
``for r in relationships.values() if r.source == ...`` loops scattered
throughout the legacy processors.

All functions take a ``model`` dict (the per-diagram payload, i.e. the
value at ``json_data['model']``) or directly a ``nodes`` / ``edges`` list
and operate on the v4 layout. They never mutate their inputs.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional


def _nodes(model: dict) -> list[dict]:
    """Return the node list from a v4 model payload (always a list)."""
    nodes = model.get("nodes")
    if isinstance(nodes, list):
        return nodes
    return []


def _edges(model: dict) -> list[dict]:
    """Return the edge list from a v4 model payload (always a list)."""
    edges = model.get("edges")
    if isinstance(edges, list):
        return edges
    return []


def find_node(model: dict, node_id: str) -> Optional[dict]:
    """Return the node with the given id, or ``None`` if not found."""
    if not node_id:
        return None
    for node in _nodes(model):
        if node.get("id") == node_id:
            return node
    return None


def iter_children(model: dict, parent_id: str) -> Iterator[dict]:
    """Yield every node whose ``parentId`` matches ``parent_id``."""
    for node in _nodes(model):
        if node.get("parentId") == parent_id:
            yield node


def iter_edges_for(model: dict, node_id: str) -> Iterator[dict]:
    """Yield every edge whose source or target equals ``node_id``."""
    for edge in _edges(model):
        if edge.get("source") == node_id or edge.get("target") == node_id:
            yield edge


def nodes_by_type(model: dict, node_type: str) -> Iterator[dict]:
    """Yield every node whose ``type`` equals ``node_type``."""
    for node in _nodes(model):
        if node.get("type") == node_type:
            yield node


def edges_by_type(model: dict, edge_type: str) -> Iterator[dict]:
    """Yield every edge whose ``type`` equals ``edge_type``."""
    for edge in _edges(model):
        if edge.get("type") == edge_type:
            yield edge


def find_edge_between(
    model: dict, source_id: str, target_id: str, edge_type: Optional[str] = None
) -> Optional[dict]:
    """Return the first edge connecting ``source_id`` -> ``target_id``.

    If ``edge_type`` is provided, only edges of that type are considered.
    """
    for edge in _edges(model):
        if edge.get("source") != source_id or edge.get("target") != target_id:
            continue
        if edge_type is not None and edge.get("type") != edge_type:
            continue
        return edge
    return None


def node_data(node: dict) -> dict:
    """Return the ``data`` dict of a node (always a dict, never ``None``)."""
    data = node.get("data")
    return data if isinstance(data, dict) else {}


def node_bounds(node: dict) -> dict:
    """Reconstruct the v3-style ``{x, y, width, height}`` bounds for a node.

    Some converters still want bounds for layout-preserving round-trips;
    this helper makes the v3 shape available without re-encoding logic
    everywhere.
    """
    pos = node.get("position") or {}
    width = node.get("width", 0)
    height = node.get("height", 0)
    measured = node.get("measured") or {}
    return {
        "x": pos.get("x", 0),
        "y": pos.get("y", 0),
        "width": width or measured.get("width", 0),
        "height": height or measured.get("height", 0),
    }
