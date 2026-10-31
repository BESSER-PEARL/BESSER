"""Helpers for reading the v4 ``{nodes[], edges[]}`` wire shape.

The v4 React-Flow wire shape is a flat list of nodes and edges (see
``docs/source/migrations/uml-v4-shape.md``). These helpers read a single
v4 node and never mutate their input.
"""

from __future__ import annotations


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


def absolute_bounds(node: dict, nodes_by_id: dict) -> dict:
    """``node_bounds`` with the position made absolute.

    A v4 ``position`` is relative to the ``parentId`` node, so the ancestors'
    positions are added (cycle-safe). Converters stash absolute bounds in the
    metamodel ``layout``, as v3 and diagram-interchange formats do.
    """
    bounds = node_bounds(node)
    seen = {node.get("id")}
    parent = nodes_by_id.get(node.get("parentId"))
    while parent is not None and parent.get("id") not in seen:
        seen.add(parent.get("id"))
        position = parent.get("position") or {}
        bounds["x"] += position.get("x", 0)
        bounds["y"] += position.get("y", 0)
        parent = nodes_by_id.get(parent.get("parentId"))
    return bounds


def edge_layout(edge: dict) -> dict:
    """Opaque layout stash for an edge: id, absolute points, handles, routing flag."""
    data = edge.get("data") if isinstance(edge.get("data"), dict) else {}
    return {
        "id": edge.get("id"),
        "points": data.get("points") or [],
        "source_direction": edge.get("sourceHandle"),
        "target_direction": edge.get("targetHandle"),
        "isManuallyLayouted": data.get("isManuallyLayouted", False),
    }


def node_layout(node: dict, nodes_by_id: dict) -> dict:
    """Opaque layout stash for a node of a Component / Deployment diagram.

    ``id``, ``parentId``, absolute ``bounds``, the stereotype header flag as
    ``displayStereotype`` (the key v8.1 ``.py`` exports already use) and colours.
    """
    data = node_data(node)
    layout = {
        "id": node.get("id"),
        "parentId": node.get("parentId"),
        "bounds": absolute_bounds(node, nodes_by_id),
    }
    header = data.get("isComponentSubsystemHeaderShown", data.get("isComponentHeaderShown"))
    if header is not None:
        layout["displayStereotype"] = bool(header)
    for style_key in ("fillColor", "strokeColor", "textColor", "highlight"):
        if style_key in data:
            layout[style_key] = data[style_key]
    return layout


def reject_unreadable_model(json_data: object, diagram_type: str) -> None:
    """Raise ``ConversionError`` for a legacy v3 or malformed diagram, which
    would otherwise convert to an empty or broken model."""
    # Lazy: the validators package imports modules that import the converters.
    from besser.utilities.web_modeling_editor.backend.services.validators.legacy_format import (
        ensure_diagram_not_legacy,
    )
    if isinstance(json_data, dict):
        title = json_data.get("title") if isinstance(json_data.get("title"), str) else None
        ensure_diagram_not_legacy(json_data.get("model"), diagram_type=diagram_type, title=title)
