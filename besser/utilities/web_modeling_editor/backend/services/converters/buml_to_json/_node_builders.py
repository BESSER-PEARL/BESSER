"""Helpers for emitting the v4 ``{nodes[], edges[]}`` wire shape.

These helpers replace the v3 ``add_element`` / ``add_relationship``
idioms used throughout the legacy converters with a tiny, consistent
node / edge builder API. Every v4 emitter in ``buml_to_json`` should
funnel through ``make_node`` and ``make_edge`` so the on-the-wire shape
stays consistent with the spec at
``docs/source/migrations/uml-v4-shape.md``.
"""

from __future__ import annotations

from typing import Any, Optional

# Default values applied when callers omit them. They mirror the spec
# defaults documented in uml-v4-shape.md.
_DEFAULT_NODE_WIDTH = 160
_DEFAULT_NODE_HEIGHT = 100


def make_node(
    node_id: str,
    type_: str,
    data: dict,
    position: dict,
    parent_id: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    measured: Optional[dict] = None,
    **extra: Any,
) -> dict:
    """Build a v4 React-Flow node.

    ``position`` is a ``{x, y}`` dict. ``width`` / ``height`` default
    to the v4 spec values when omitted. ``measured`` defaults to
    ``{width, height}`` if not given. ``extra`` is merged onto the
    top-level node dict for any one-off fields a caller needs.
    """
    if not isinstance(position, dict) or "x" not in position or "y" not in position:
        position = {"x": position.get("x", 0) if isinstance(position, dict) else 0,
                    "y": position.get("y", 0) if isinstance(position, dict) else 0}
    w = width if width is not None else _DEFAULT_NODE_WIDTH
    h = height if height is not None else _DEFAULT_NODE_HEIGHT
    node: dict = {
        "id": node_id,
        "type": type_,
        "position": {"x": position["x"], "y": position["y"]},
        "width": w,
        "height": h,
        "measured": measured if measured is not None else {"width": w, "height": h},
        "data": data or {},
    }
    if parent_id:
        node["parentId"] = parent_id
    if extra:
        # Allow callers to inject extra top-level fields without colliding
        # with the canonical ones above.
        for key, value in extra.items():
            if key in node:
                continue
            node[key] = value
    return node


def make_edge(
    edge_id: str,
    source: str,
    target: str,
    type_: str,
    data: Optional[dict] = None,
    source_handle: str = "Right",
    target_handle: str = "Left",
    **extra: Any,
) -> dict:
    """Build a v4 React-Flow edge.

    ``data`` is the per-edge payload (always normalised to include a
    ``points`` list, even if empty). ``source_handle`` / ``target_handle``
    encode v3's ``source.direction`` / ``target.direction`` strings.
    """
    edge_data: dict = dict(data or {})
    edge_data.setdefault("points", [])
    edge: dict = {
        "id": edge_id,
        "source": source,
        "target": target,
        "type": type_,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
        "data": edge_data,
    }
    if extra:
        for key, value in extra.items():
            if key in edge:
                continue
            edge[key] = value
    return edge


def compute_bounds(nodes: list[dict]) -> dict:
    """Return a ``{x, y, width, height}`` bounding box for the given nodes.

    Used to populate the optional v4 ``size`` field on a model.
    """
    if not nodes:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    xs = []
    ys = []
    rights = []
    bottoms = []
    for node in nodes:
        pos = node.get("position") or {}
        x = pos.get("x", 0)
        y = pos.get("y", 0)
        w = node.get("width", 0) or (node.get("measured") or {}).get("width", 0)
        h = node.get("height", 0) or (node.get("measured") or {}).get("height", 0)
        xs.append(x)
        ys.append(y)
        rights.append(x + w)
        bottoms.append(y + h)
    x_min = min(xs)
    y_min = min(ys)
    return {
        "x": x_min,
        "y": y_min,
        "width": max(rights) - x_min,
        "height": max(bottoms) - y_min,
    }


def empty_model(diagram_type: str, title: str = "") -> dict:
    """Return an empty v4 model envelope of the given diagram type.

    Useful for converters that need to short-circuit when there's nothing
    to emit.
    """
    return {
        "version": "4.0.0",
        "type": diagram_type,
        "title": title,
        "size": {"width": 0, "height": 0},
        "nodes": [],
        "edges": [],
        "interactive": {"elements": {}, "relationships": {}},
        "assessments": {},
    }
