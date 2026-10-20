"""Helpers for emitting the v4 ``{nodes[], edges[]}`` wire shape.

These helpers replace the v3 ``add_element`` / ``add_relationship``
idioms used throughout the legacy converters with a tiny, consistent
node / edge builder API. Every v4 emitter in ``buml_to_json`` should
funnel through ``make_node`` and ``make_edge`` so the on-the-wire shape
stays consistent with the spec at
``docs/source/migrations/uml-v4-shape.md``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

# Default values applied when callers omit them. They mirror the spec
# defaults documented in uml-v4-shape.md.
_DEFAULT_NODE_WIDTH = 160
_DEFAULT_NODE_HEIGHT = 100


# v3 ``Direction`` -> v4 ``HandleId``; mirrors the frontend's
# ``convertV3HandleToV4`` (packages/library/lib/utils/versionConverter.ts).
_V3_HANDLE_TO_V4 = {
    "Up": "top", "Right": "right", "Down": "bottom", "Left": "left",
    "Upright": "right-top", "Upleft": "left-top",
    "Downright": "right-bottom", "Downleft": "left-bottom",
    "RightTop": "top-right", "RightBottom": "bottom-right",
    "LeftTop": "top-left", "LeftBottom": "bottom-left",
}
_V3_CORNER_HANDLE = re.compile(r"^(top|bottom|left|right)(top|bottom|left|right)$", re.IGNORECASE)


def normalize_handle(handle: Optional[str]) -> Optional[str]:
    """Return the v4 handle id (``right``, ``top-left``...) for a stored handle.

    Older exports carry v3 directions (``Right``, ``Up``, ``Topleft``); React
    Flow cannot resolve those against the editor's lowercase handles and
    drops the edge.
    """
    if not handle:
        return handle
    if handle in _V3_HANDLE_TO_V4:
        return _V3_HANDLE_TO_V4[handle]
    corner = _V3_CORNER_HANDLE.match(handle)
    if corner:
        return f"{corner.group(1).lower()}-{corner.group(2).lower()}"
    return handle.lower()


def snap_up(value: float, step: int = 10) -> int:
    """Round up to the editor's 10px extension grid (``calculateMinWidth/Height``)."""
    return int(math.ceil(value / step) * step)


def grid_layout(
    nodes: Iterable[dict],
    origin: tuple = (0, 0),
    columns: int = 3,
    gap_x: int = 140,
    gap_y: int = 100,
) -> None:
    """Place ``nodes`` row by row on a grid sized from their own width/height.

    Column width is the widest node in that column and row height the
    tallest node in that row, so nodes never overlap whatever their size.
    """
    nodes = list(nodes)
    col_widths = [0] * columns
    row_heights = [0] * (len(nodes) // columns + 1)
    for i, node in enumerate(nodes):
        col_widths[i % columns] = max(col_widths[i % columns], node["width"])
        row_heights[i // columns] = max(row_heights[i // columns], node["height"])
    for i, node in enumerate(nodes):
        col, row = i % columns, i // columns
        node["position"] = {
            "x": origin[0] + sum(col_widths[:col]) + col * gap_x,
            "y": origin[1] + sum(row_heights[:row]) + row * gap_y,
        }


def points_from_layout(layout: dict) -> list:
    """Absolute v4 edge points from an edge ``layout`` stash.

    v4 stashes keep ``points``; ``.py`` files exported before the v4 migration
    carry the v3 ``path`` (relative to the edge ``bounds``) instead.
    """
    if layout.get("points"):
        return layout["points"]
    path = layout.get("path") or []
    bounds = layout.get("bounds") or {}
    return [
        {"x": p.get("x", 0) + bounds.get("x", 0), "y": p.get("y", 0) + bounds.get("y", 0)}
        for p in path
    ]


def make_positions_parent_relative(nodes: list, absolute_ids: set) -> None:
    """Turn absolute positions into ``parentId``-relative ones, in place.

    Only nodes whose position and parent position are both absolute (ids in
    ``absolute_ids``, typically the nodes placed from a stashed ``bounds``) are
    shifted; fallback positions are already relative.
    """
    absolute = {node["id"]: dict(node["position"]) for node in nodes}
    for node in nodes:
        parent_id = node.get("parentId")
        if parent_id in absolute and node["id"] in absolute_ids and parent_id in absolute_ids:
            node["position"] = {
                "x": absolute[node["id"]]["x"] - absolute[parent_id]["x"],
                "y": absolute[node["id"]]["y"] - absolute[parent_id]["y"],
            }


def layout_unplaced_nodes(nodes: list, placed_ids: set, padding: int = 20, header: int = 40) -> None:
    """Give every node not in ``placed_ids`` a non-overlapping position, in place.

    Children of a container are put in a row inside it (below a ``header``
    strip), and an unplaced container grows to enclose them; unplaced roots go
    on a grid. Used when a model built in Python carries no editor layout.
    """
    children: dict = {}
    for node in nodes:
        children.setdefault(node.get("parentId"), []).append(node)

    def fit(node: dict) -> None:
        kids = children.get(node["id"], [])
        for kid in kids:
            fit(kid)
        unplaced = [kid for kid in kids if kid["id"] not in placed_ids]
        x = padding
        for kid in unplaced:
            kid["position"] = {"x": x, "y": header}
            x += kid["width"] + padding
        if node["id"] not in placed_ids and kids:
            right = max(k["position"]["x"] + k["width"] for k in kids) + padding
            bottom = max(k["position"]["y"] + k["height"] for k in kids) + padding
            node["width"] = max(node["width"], right)
            node["height"] = max(node["height"], bottom)
            node["measured"] = {"width": node["width"], "height": node["height"]}

    roots = children.get(None, [])
    for root in roots:
        fit(root)
    placed_roots = [root for root in roots if root["id"] in placed_ids]
    origin_x = max((r["position"]["x"] + r["width"] for r in placed_roots), default=-140) + 140
    grid_layout([root for root in roots if root["id"] not in placed_ids], origin=(origin_x, 0))


def envelope_size(nodes: list) -> dict:
    """Bounding box of all top-level node bounds, with a sensible minimum."""
    max_x = 0
    max_y = 0
    for node in nodes:
        if node.get("parentId"):
            continue  # child positions are relative to their parent
        position = node.get("position") or {}
        right = (position.get("x") or 0) + (node.get("width") or 0)
        bottom = (position.get("y") or 0) + (node.get("height") or 0)
        max_x = max(max_x, right)
        max_y = max(max_y, bottom)
    return {"width": max(int(max_x) + 40, 800), "height": max(int(max_y) + 40, 600)}


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
    source_handle: str = "right",
    target_handle: str = "left",
    **extra: Any,
) -> dict:
    """Build a v4 React-Flow edge.

    ``data`` is the per-edge payload (always normalised to include a
    ``points`` list, even if empty). Handles are normalised to v4 ids, except
    on ``ClassLinkRel``, whose edge-anchored ``Center`` / ``Up`` pair is canonical.
    """
    if type_ != "ClassLinkRel":
        source_handle = normalize_handle(source_handle)
        target_handle = normalize_handle(target_handle)
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


def parents_first(nodes: list) -> list:
    """Order ``nodes`` so every parent precedes its children (React Flow requires it)."""
    by_id = {node["id"]: node for node in nodes}
    ordered: list = []
    seen: set = set()

    def visit(node: dict) -> None:
        if node["id"] in seen:
            return
        seen.add(node["id"])
        parent = by_id.get(node.get("parentId"))
        if parent is not None:
            visit(parent)
        ordered.append(node)

    for node in nodes:
        visit(node)
    return ordered


def node_from_layout(obj, node_id: str, node_type: str, data: dict, parent_id, default_size: tuple) -> dict:
    """``make_node`` from the element's stashed absolute bounds (else ``default_size``
    at the origin) and stashed colours. Positions are made parent-relative later
    (``make_positions_parent_relative``)."""
    layout = obj.layout or {}
    for style_key in ("fillColor", "strokeColor", "textColor", "highlight"):
        if style_key in layout:
            data[style_key] = layout[style_key]
    width, height = default_size
    bounds = layout.get("bounds") or {"x": 0, "y": 0, "width": width, "height": height}
    return make_node(
        node_id=node_id, type_=node_type, data=data,
        position={"x": bounds.get("x", 0), "y": bounds.get("y", 0)},
        parent_id=parent_id,
        width=bounds.get("width") or width, height=bounds.get("height") or height,
    )


def stereotyped_edge(rel, id_for, edge_type: str, stereotype: str) -> dict:
    """``make_edge`` for a Component / Deployment relationship (``name``, ``label``,
    optional ``stereotype``), routed from its layout stash."""
    layout = rel.layout or {}
    data = {
        "name": rel.name or "",
        "label": rel.name or "",
        "points": points_from_layout(layout),
        "isManuallyLayouted": bool(layout.get("isManuallyLayouted", False)),
    }
    if stereotype:
        data["stereotype"] = stereotype
    return make_edge(
        edge_id=id_for(rel), source=id_for(rel.source), target=id_for(rel.target),
        type_=edge_type, data=data,
        source_handle=layout.get("source_direction") or "right",
        target_handle=layout.get("target_direction") or "left",
    )
