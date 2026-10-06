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
