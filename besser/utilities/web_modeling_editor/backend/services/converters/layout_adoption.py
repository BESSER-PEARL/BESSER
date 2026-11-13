"""Lay a diagram imported from B-UML out like the diagram it replaces (issue #774).

A B-UML ``.py`` file carries no layout for Class, Agent and State Machine
diagrams, so importing one rebuilds the diagram on a grid with fresh ids.
``adopt_layout`` takes the diagram the editor already has and the freshly
imported one (both v4 wire shape, ``{nodes, edges}``) and returns the imported
content laid out like the existing diagram:

- a node is matched to an existing node by id when the import kept it, else by
  type and name (case-insensitive) inside the matched parent, duplicates in
  order. A matched node takes the existing id and position, and the existing
  size unless its new content needs more room;
- a node renamed in the file is matched to its old self only when, among the
  unmatched nodes of its type and parent, it is the one on each side connected
  to the same matched neighbours by the same edges (type, direction, content);
  any other rename reads as a removal plus an addition;
- member rows (class attributes and methods, agent state bodies) and agent
  components keep their existing ids, matched by name inside their match;
- an edge is matched by id, else by type and (matched) ends, preferring one
  with the same content. A matched edge takes the existing id and handles, and
  its bend points while neither end changed size;
- a new top-level node is placed on the first free slot next to a matched
  neighbour it is connected to, else right of the diagram.

Content always comes from the import. Nothing here reads B-UML.
"""

import copy
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Diagram types whose B-UML import rebuilds the layout and that adopt_layout handles.
# BPMN, Component and Deployment diagrams keep their layout through B-UML already.
LAYOUT_ADOPTION_DIAGRAM_TYPES = frozenset({"ClassDiagram", "AgentDiagram", "StateMachineDiagram"})

# Space left between a placed node and its neighbours, in canvas pixels (the grid gap of the converters).
PLACEMENT_GAP = 60
# How many rings of slots around a neighbour are tried before a new node goes right of the diagram.
MAX_PLACEMENT_RINGS = 8
# Slot directions around a neighbour, nearest first: right, below, left, above, then the corners.
_SLOT_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))

# Node data lists whose rows carry their own ids, and the row fields that identify a row.
_ROW_FIELDS = {
    "attributes": ("name",),
    "methods": ("name",),
    "bodies": ("replyType", "name"),
    "fallbackBodies": ("replyType", "name"),
}
# Edge data fields that are layout, not content.
_EDGE_LAYOUT_FIELDS = ("points", "isManuallyLayouted")


def adopt_layout(existing: Dict[str, Any], imported: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``imported`` laid out like ``existing``, and a report of what was matched.

    Both arguments are v4 diagram models (``nodes`` and ``edges`` lists); neither
    is modified. The report holds ``matched`` (nodes that kept their id and
    place), ``renamed``, ``added`` (new top-level nodes and the neighbour they
    were placed next to), ``notInImport`` (existing nodes the import no longer
    has), and the edge counts ``edgesKept``, ``edgesRerouted`` (kept, bend
    points dropped because an end changed size) and ``edgesAdded``.
    """
    result = copy.deepcopy(imported)
    old_nodes = {n["id"]: n for n in _nodes(existing)}
    new_nodes = _nodes(result)
    taken_ids = _all_ids(result)

    node_map = _match_nodes(list(old_nodes.values()), new_nodes, taken_ids)
    renamed = _match_renames(existing, result, old_nodes, new_nodes, node_map, taken_ids)

    id_map = dict(node_map)
    resized = set()
    for node in new_nodes:
        old = old_nodes.get(node_map.get(node["id"]))
        if old is None:
            continue
        _keep_place(node, old, node_map, resized)
        for field, key_fields in _ROW_FIELDS.items():
            id_map.update(_match_items(
                _rows(old, field), _rows(node, field), taken_ids,
                key=lambda row, fields=key_fields: tuple(_norm(row.get(f)) for f in fields),
            ))
    id_map.update(_match_components(existing.get("components"), result.get("components"), taken_ids))
    result = _replace_ids(result, id_map)

    edge_map, edge_report = _match_edges(existing, result, resized, taken_ids)
    result = _replace_ids(result, edge_map)

    added = _place_new_nodes(result, set(node_map.values()))

    matched_old = set(node_map.values())
    report = {
        "matched": len(node_map),
        "renamed": [
            {"id": old_id, "type": old_nodes[old_id]["type"],
             "from": _name(old_nodes[old_id]), "to": name}
            for old_id, name in renamed
        ],
        "added": added,
        "notInImport": [
            _describe(n) for n in old_nodes.values()
            if n["id"] not in matched_old and (n.get("parentId") is None or n.get("parentId") in matched_old)
        ],
        **edge_report,
    }
    return result, report


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _match_nodes(old_nodes: List[dict], new_nodes: List[dict], taken_ids: set) -> Dict[str, str]:
    """New node id -> existing node id: by id first, then by type and name inside matched parents."""
    old_by_id = {n["id"]: n for n in old_nodes}
    mapping = {n["id"]: n["id"] for n in new_nodes
               if n["id"] in old_by_id and old_by_id[n["id"]]["type"] == n["type"]}

    old_children = _children_index(old_nodes, lambda n: n.get("parentId"))
    new_children = _children_index(new_nodes, lambda n: n.get("parentId"))

    def pair(old_parent: Optional[str], new_parent: Optional[str]) -> None:
        mapping.update(_match_items(
            old_children.get(old_parent, []), new_children.get(new_parent, []), taken_ids,
            key=_node_key, already=mapping,
        ))
        for node in new_children.get(new_parent, []):
            if node["id"] in mapping and mapping[node["id"]] in old_children:
                pair(mapping[node["id"]], node["id"])

    pair(None, None)
    return mapping


def _match_renames(existing: dict, result: dict, old_nodes: Dict[str, dict], new_nodes: List[dict],
                   node_map: Dict[str, str], taken_ids: set) -> List[Tuple[str, str]]:
    """Pair a renamed node with its old self when nothing else could be meant (see module doc)."""
    matched_old = set(node_map.values())
    old_left = defaultdict(list)
    for n in old_nodes.values():
        if n["id"] not in matched_old and n["id"] not in taken_ids:
            old_left[(n.get("parentId"), n["type"])].append(n)
    new_left = defaultdict(list)
    for n in new_nodes:
        if n["id"] not in node_map:
            parent = n.get("parentId")
            new_left[(node_map.get(parent, parent), n["type"])].append(n)

    old_signatures = _neighbour_signatures(_edges(existing), lambda node_id: node_id if node_id in matched_old else None)
    new_signatures = _neighbour_signatures(_edges(result), node_map.get)
    renamed = []
    for slot, olds in old_left.items():
        old_by_signature = defaultdict(list)
        for old in olds:
            if old_signatures.get(old["id"]):
                old_by_signature[old_signatures[old["id"]]].append(old)
        new_by_signature = defaultdict(list)
        for new in new_left.get(slot, []):
            if new_signatures.get(new["id"]):
                new_by_signature[new_signatures[new["id"]]].append(new)
        for signature, news in new_by_signature.items():
            olds_alike = old_by_signature.get(signature, [])
            if len(news) == 1 and len(olds_alike) == 1:
                node_map[news[0]["id"]] = olds_alike[0]["id"]
                renamed.append((olds_alike[0]["id"], _name(news[0])))
    return renamed


def _neighbour_signatures(edges: Iterable[dict], resolve: Callable[[str], Optional[str]]) -> Dict[str, frozenset]:
    """Node id -> the (edge type, direction, neighbour, content) of its edges to resolvable neighbours."""
    signatures = defaultdict(set)
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if resolve(target) is not None:
            signatures[source].add((edge.get("type"), "out", resolve(target), _edge_content(edge)))
        if resolve(source) is not None:
            signatures[target].add((edge.get("type"), "in", resolve(source), _edge_content(edge)))
    return {node_id: frozenset(sig) for node_id, sig in signatures.items()}


def _match_components(old: Any, new: Any, taken_ids: set) -> Dict[str, str]:
    """Agent components (intents, LLMs, ...) keep their ids, matched by type and name; intent bodies by owner."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return {}
    old_items = [c for c in old.values() if isinstance(c, dict) and c.get("id")]
    new_items = [c for c in new.values() if isinstance(c, dict) and c.get("id")]
    old_by_id = {c["id"]: c for c in old_items}
    mapping = {c["id"]: c["id"] for c in new_items
               if c["id"] in old_by_id and old_by_id[c["id"]].get("type") == c.get("type")}
    mapping.update(_match_items(
        [c for c in old_items if not c.get("owner")], [c for c in new_items if not c.get("owner")],
        taken_ids, key=_node_key, already=mapping,
    ))
    old_owned = _children_index([c for c in old_items if c.get("owner")], lambda c: c["owner"])
    new_owned = _children_index([c for c in new_items if c.get("owner")], lambda c: c["owner"])
    for new_owner, items in new_owned.items():
        if new_owner in mapping:
            mapping.update(_match_items(old_owned.get(mapping[new_owner], []), items, taken_ids,
                                        key=_node_key, already=mapping))
    return mapping


def _match_items(old_items: List[dict], new_items: List[dict], taken_ids: set, key: Callable[[dict], Any],
                 already: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """New id -> old id for items with equal keys, in order.

    An old id is given once, and never when the import already uses it for something else.
    """
    already = already or {}
    used = set(already.values())
    buckets = defaultdict(list)
    for item in old_items:
        if item.get("id") and item["id"] not in used and item["id"] not in taken_ids:
            buckets[key(item)].append(item)
    mapping = {}
    for item in new_items:
        if not item.get("id") or item["id"] in already:
            continue
        candidates = [c for c in buckets.get(key(item), []) if c["id"] not in used]
        if candidates:
            mapping[item["id"]] = candidates[0]["id"]
            used.add(candidates[0]["id"])
    return mapping


def _match_edges(existing: dict, result: dict, resized: set, taken_ids: set) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Give matched edges their existing id and layout; returns (new id -> old id, edge counts)."""
    old_edges = list(_edges(existing))
    new_edges = list(_edges(result))
    old_by_id = {e["id"]: e for e in old_edges if e.get("id")}
    mapping = {e["id"]: e["id"] for e in new_edges
               if e.get("id") in old_by_id and _edge_ends(old_by_id[e["id"]]) == _edge_ends(e)}
    # Same ends and content first (two transitions between the same states differ by intent), then same ends.
    for key in (lambda e: (_edge_ends(e), _edge_content(e)), _edge_ends):
        mapping.update(_match_items(old_edges, new_edges, taken_ids, key=key, already=mapping))

    kept = rerouted = 0
    for edge in new_edges:
        old = old_by_id.get(mapping.get(edge.get("id")))
        if old is None:
            continue
        kept += 1
        for handle in ("sourceHandle", "targetHandle"):
            if old.get(handle):
                edge[handle] = old[handle]
        data = edge.setdefault("data", {})
        if {edge.get("source"), edge.get("target")} & resized:
            rerouted += 1
            data["points"] = []
            data.pop("isManuallyLayouted", None)
            continue
        old_data = old.get("data") or {}
        for field in _EDGE_LAYOUT_FIELDS:
            if field in old_data:
                data[field] = copy.deepcopy(old_data[field])
            else:
                data.pop(field, None)
        data.setdefault("points", [])
    return mapping, {"edgesKept": kept, "edgesRerouted": rerouted, "edgesAdded": len(new_edges) - kept}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _keep_place(node: dict, old: dict, node_map: Dict[str, str], resized: set) -> None:
    """Give a matched node the existing position, and the existing size unless its content needs more."""
    parent = node.get("parentId")
    if node_map.get(parent, parent) == old.get("parentId"):
        node["position"] = dict(old.get("position") or node.get("position") or {"x": 0, "y": 0})
    old_w, old_h = _size(old)
    new_w, new_h = _size(node)
    width, height = max(old_w, new_w), max(old_h, new_h)
    if (width, height) != (old_w, old_h):
        resized.add(old["id"])
    node["width"], node["height"] = width, height
    node["measured"] = {"width": width, "height": height}


def _place_new_nodes(result: dict, kept_ids: set) -> List[Dict[str, Any]]:
    """Move every new top-level node next to a neighbour; returns what was placed where."""
    nodes = {n["id"]: n for n in _nodes(result)}
    top_level = [n for n in nodes.values() if n.get("parentId") is None]
    new_top = [n for n in top_level if n["id"] not in kept_ids]
    taken = [_bounds(n) for n in top_level if n["id"] in kept_ids]
    if not taken:
        # Nothing matched: the existing diagram tells nothing about where things go.
        return [{**_describe(n), "placedNextTo": None} for n in new_top]

    def top_ancestor(node_id: Optional[str]) -> Optional[str]:
        seen = set()
        while node_id in nodes and nodes[node_id].get("parentId") and node_id not in seen:
            seen.add(node_id)
            node_id = nodes[node_id]["parentId"]
        return node_id if node_id in nodes else None

    neighbours = defaultdict(list)
    for edge in _edges(result):
        a, b = top_ancestor(edge.get("source")), top_ancestor(edge.get("target"))
        if a and b and a != b:
            neighbours[a].append(b)
            neighbours[b].append(a)

    placed: List[Dict[str, Any]] = []
    placed_ids = set()
    for node in new_top:
        candidates = neighbours.get(node["id"], [])
        anchor_id = next((n for n in candidates if n in kept_ids), None) \
            or next((n for n in candidates if n in placed_ids), None)
        width, height = _size(node)
        slot = _free_slot(width, height, _bounds(nodes[anchor_id]), taken) if anchor_id else None
        if slot is None:
            anchor_id = None
            slot = _slot_right_of(width, height, taken)
        node["position"] = {"x": slot["x"], "y": slot["y"]}
        taken.append(slot)
        placed_ids.add(node["id"])
        placed.append({**_describe(node), "placedNextTo": _name(nodes[anchor_id]) if anchor_id else None})
    return placed


def _free_slot(width: float, height: float, anchor: dict, taken: List[dict]) -> Optional[dict]:
    """The first slot around ``anchor``, nearest ring first, that overlaps nothing in ``taken``."""
    step_x = max(anchor["width"], width) + PLACEMENT_GAP
    step_y = max(anchor["height"], height) + PLACEMENT_GAP
    for ring in range(1, MAX_PLACEMENT_RINGS + 1):
        for dx, dy in _SLOT_DIRECTIONS:
            slot = {"x": anchor["x"] + dx * ring * step_x, "y": anchor["y"] + dy * ring * step_y,
                    "width": width, "height": height}
            if not any(_overlaps(slot, other) for other in taken):
                return slot
    return None


def _slot_right_of(width: float, height: float, taken: List[dict]) -> dict:
    """A slot right of everything in ``taken``, moved down until it overlaps nothing."""
    slot = {"x": max(t["x"] + t["width"] for t in taken) + PLACEMENT_GAP,
            "y": min(t["y"] for t in taken), "width": width, "height": height}
    while any(_overlaps(slot, other) for other in taken):
        slot["y"] += height + PLACEMENT_GAP
    return slot


def _overlaps(a: dict, b: dict) -> bool:
    """True when the boxes are closer than half the placement gap."""
    gap = PLACEMENT_GAP / 2
    return not (a["x"] + a["width"] + gap <= b["x"] or b["x"] + b["width"] + gap <= a["x"]
                or a["y"] + a["height"] + gap <= b["y"] or b["y"] + b["height"] + gap <= a["y"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nodes(model: dict) -> List[dict]:
    return [n for n in model.get("nodes") or [] if isinstance(n, dict) and n.get("id")]


def _edges(model: dict) -> List[dict]:
    return [e for e in model.get("edges") or [] if isinstance(e, dict) and e.get("id")]


def _rows(node: dict, field: str) -> List[dict]:
    rows = (node.get("data") or {}).get(field)
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _children_index(items: Iterable[dict], parent_of: Callable[[dict], Optional[str]]) -> Dict[Optional[str], list]:
    index = defaultdict(list)
    for item in items:
        index[parent_of(item)].append(item)
    return index


def _name(item: dict) -> str:
    data = item.get("data")
    name = data.get("name") if isinstance(data, dict) else item.get("name")
    return name if isinstance(name, str) else ""


def _norm(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value


def _node_key(item: dict) -> tuple:
    return item.get("type"), _norm(_name(item))


def _edge_ends(edge: dict) -> tuple:
    return edge.get("type"), edge.get("source"), edge.get("target")


def _edge_content(edge: dict) -> str:
    data = {k: v for k, v in (edge.get("data") or {}).items() if k not in _EDGE_LAYOUT_FIELDS}
    return json.dumps(data, sort_keys=True, default=str)


def _size(node: dict) -> Tuple[float, float]:
    measured = node.get("measured") or {}
    width = node.get("width") or measured.get("width") or 0
    height = node.get("height") or measured.get("height") or 0
    return width, height


def _bounds(node: dict) -> dict:
    position = node.get("position") or {}
    width, height = _size(node)
    return {"x": position.get("x", 0), "y": position.get("y", 0), "width": width, "height": height}


def _describe(node: dict) -> Dict[str, Any]:
    return {"id": node["id"], "type": node.get("type"), "name": _name(node)}


def _all_ids(value: Any) -> set:
    """Every ``id`` value and ``components`` key in a model."""
    ids = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get("id"), str):
                ids.add(item["id"])
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return ids


def _replace_ids(value: Any, mapping: Dict[str, str]) -> Any:
    """``value`` with every string (and dict key) found in ``mapping`` replaced: ids and references to them."""
    if not mapping:
        return value
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_ids(v, mapping) for v in value]
    if isinstance(value, dict):
        return {mapping.get(k, k): _replace_ids(v, mapping) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Import endpoint glue
# ---------------------------------------------------------------------------

def adopt_existing_layout(existing_model: str, imported: Dict[str, Any],
                          diagram_type: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Lay ``imported`` out like the editor's ``existing_model`` (JSON text); returns (model, warnings).

    ``existing_model`` is the diagram being updated: a v4 model, or a project
    diagram entry holding one under ``model``. When it cannot be used (not JSON,
    not a v4 model, another diagram type, a type without layout adoption) the
    import is returned unchanged with a ``layout-not-adopted`` warning saying why.
    Otherwise the warnings report the outcome: ``layout-adopted`` (a summary),
    ``element-renamed``, ``element-added`` and ``element-not-in-import``.
    """
    existing, reason = _read_existing_model(existing_model, diagram_type)
    if existing is None:
        return imported, [{"code": "layout-not-adopted",
                           "message": f"The diagram is imported with a new layout: {reason}"}]

    model, report = adopt_layout(existing, imported)
    warnings: List[Dict[str, Any]] = [{
        "code": "layout-adopted",
        "message": (f"Kept the place and id of {report['matched']} element(s) of the existing diagram; "
                    f"{len(report['added'])} new, {len(report['notInImport'])} not in the imported file."),
        "matched": report["matched"],
        "edgesKept": report["edgesKept"],
        "edgesRerouted": report["edgesRerouted"],
        "edgesAdded": report["edgesAdded"],
    }]
    for item in report["renamed"]:
        warnings.append({
            "code": "element-renamed",
            "message": f"{item['type']} '{item['from']}' is read as renamed to '{item['to']}' and keeps its place.",
            "elementId": item["id"], "type": item["type"], "from": item["from"], "name": item["to"],
        })
    for item in report["added"]:
        where = f"next to '{item['placedNextTo']}'" if item["placedNextTo"] else "right of the diagram"
        warnings.append({
            "code": "element-added",
            "message": f"New {item['type']} '{item['name']}' placed {where}.",
            "elementId": item["id"], "type": item["type"], "name": item["name"],
            "placedNextTo": item["placedNextTo"],
        })
    for item in report["notInImport"]:
        warnings.append({
            "code": "element-not-in-import",
            "message": f"{item['type']} '{item['name']}' of the existing diagram is not in the imported file.",
            "elementId": item["id"], "type": item["type"], "name": item["name"],
        })
    return model, warnings


def _read_existing_model(text: str, diagram_type: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """The v4 model in ``text``, or None and the reason it cannot be used."""
    if diagram_type not in LAYOUT_ADOPTION_DIAGRAM_TYPES:
        return None, f"keeping the layout of an existing {diagram_type} is not supported."
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None, "the existing diagram sent with the file is not valid JSON."
    if isinstance(value, dict) and isinstance(value.get("model"), dict):
        value = value["model"]
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list):
        return None, "the existing diagram sent with the file is not a diagram model with nodes."
    existing_type = value.get("type")
    if existing_type and existing_type != diagram_type:
        return None, f"the existing diagram is a {existing_type}, the file holds a {diagram_type}."
    return value, ""
