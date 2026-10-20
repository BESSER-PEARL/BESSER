"""Parse WME `a2a:` wire tags and annotate a BUML Agent with the parsed edges.

This module is the BESSER *consume* side of the explicit agent-to-agent (A2A)
annotations the Web Modeling Editor serializes on the Agent-diagram wire. It does
two things, both standalone (the core converter `agent_diagram_processor.py` is
NOT touched):

1. `parse_a2a_line` / `parse_a2a_out_block` — a tolerant parser for the wire
   grammar (semicolon-delimited key=value, leading `a2a:<dir>`):

       a2a:<dir>;peer=<name>;ref=<uuid|>;flow=<id>;[order=<n>;]kind=<...>

2. `annotate_agent_with_a2a` — a re-walk of the raw v4 AgentDiagram JSON that stashes
   the parsed in/out edges on `agent._a2a` (a side-map, no metamodel change). It is a
   no-op (sets no attribute) when the diagram carries no `a2a:` tag, so legacy agents
   stay byte-identical and the generator's `getattr(agent, "_a2a", None)` is falsy.

Tolerant by contract: unknown keys ignored; missing optionals → defaults; malformed
input → None (caller drops it). NEVER raises on user data.
"""
from typing import Optional

from besser.BUML.metamodel.structural import UNLIMITED_MAX_MULTIPLICITY

_KINDS = {"delegates", "supervises", "revises", "collaborates"}

# `order` of a tag that omits it: sorts after every explicit order.
_ORDER_SENTINEL = UNLIMITED_MAX_MULTIPLICITY


def parse_a2a_line(line: str) -> Optional[dict]:
    """Parse ONE `a2a:in`/`a2a:out` line into a dict, or None if it isn't an a2a tag.

    Returns: {dir, peer, ref, flow, order, kind} where
      - dir   : 'in' | 'out'
      - peer  : str (may be '' if absent — caller decides whether to drop)
      - ref   : str ('' when the peer lane was never linked → name-addressing)
      - flow  : str ('' if absent)
      - order : int (UNLIMITED_MAX_MULTIPLICITY when absent — sorts last)
      - kind  : str | None (None = plain channel, i.e. no `kind=` field)
    """
    if not isinstance(line, str):
        return None
    s = line.strip()
    if not (s.startswith("a2a:in") or s.startswith("a2a:out")):
        return None
    parts = [p for p in s.split(";") if p]
    head = parts[0]                       # "a2a:in" / "a2a:out"
    direction = head.split(":", 1)[1] if ":" in head else ""
    if direction not in ("in", "out"):
        return None
    kv = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = v.strip()
    order_raw = kv.get("order", "")
    try:
        order = int(order_raw) if order_raw != "" else _ORDER_SENTINEL
    except ValueError:
        order = _ORDER_SENTINEL
    kind = kv.get("kind")
    if kind is not None and kind not in _KINDS:
        kind = None                       # unknown kind → treat as a plain channel
    return {
        "dir": direction,
        "peer": kv.get("peer", ""),
        "ref": kv.get("ref", ""),
        "flow": kv.get("flow", ""),
        "order": order,
        "kind": kind,
    }


def format_a2a_line(edge: dict) -> Optional[str]:
    """Inverse of `parse_a2a_line`: the wire tag for a parsed edge, or None without a peer."""
    if not isinstance(edge, dict) or edge.get("dir") not in ("in", "out") or not edge.get("peer"):
        return None
    parts = [f"a2a:{edge['dir']}", f"peer={edge['peer']}", f"ref={edge.get('ref') or ''}"]
    if edge.get("flow"):
        parts.append(f"flow={edge['flow']}")
    if isinstance(edge.get("order"), int) and edge["order"] != _ORDER_SENTINEL:
        parts.append(f"order={edge['order']}")
    if edge.get("kind") in _KINDS:
        parts.append(f"kind={edge['kind']}")
    return ";".join(parts)


def parse_a2a_out_block(description: Optional[str]) -> list:
    """Parse an AgentState.description (possibly multi-line) into a list of out-edges,
    sorted by ascending `order`. Non-a2a lines (human comments) are ignored — a state
    description may legitimately mix prose and tags."""
    if not isinstance(description, str) or "a2a:out" not in description:
        return []
    edges = []
    for raw in description.splitlines():
        parsed = parse_a2a_line(raw)
        if parsed and parsed["dir"] == "out" and parsed["peer"]:
            edges.append(parsed)
    return sorted(edges, key=lambda e: e["order"])


def annotate_agent_with_a2a(agent, agent_diagram_json: dict):
    """Attach `agent._a2a = {outbound, inbound}` parsed from the v4 AgentDiagram JSON.

    Standalone: does NOT touch process_agent_diagram. Call it right after the agent
    is built. No-op (no attribute set) when the diagram carries no a2a: tag, so
    legacy agents stay byte-identical and the generator's getattr(agent,'_a2a',None) is
    falsy. Reads only strings (peer/state/intent names), so it never needs the built
    State/Transition objects — fully decoupled from the metamodel.

    Tags ride on ``AgentState`` node ``data.description`` (``a2a:out`` lines) and on
    transition edge ``data.name`` (``a2a:in``), the intent on
    ``data.predefined.intentName``.
    """
    if not isinstance(agent_diagram_json, dict):
        return agent
    model = agent_diagram_json.get("model") or agent_diagram_json
    nodes = [n for n in (model.get("nodes") or []) if isinstance(n, dict)]
    edges = [e for e in (model.get("edges") or []) if isinstance(e, dict)]
    states = {n.get("id"): n.get("data") or {} for n in nodes if n.get("type") == "AgentState"}

    def _state_name(node_id):
        return (states.get(node_id) or {}).get("name", "")

    # ── outbound: a2a:out lines on each AgentState description ──
    outbound = []
    for data in states.values():
        for edge in parse_a2a_out_block(data.get("description")):
            edge["state"] = data.get("name", "")
            outbound.append(edge)

    # ── inbound: a2a:in on a when_intent_matched transition's `name` ──
    inbound = []
    for edge in edges:
        if edge.get("type") not in ("AgentStateTransition", "AgentStateTransitionInit"):
            continue
        data = edge.get("data") or {}
        parsed = parse_a2a_line(data.get("name", ""))
        if not (parsed and parsed["dir"] == "in" and parsed["peer"]):
            continue
        predefined = data.get("predefined") or {}
        parsed["intent"] = predefined.get("intentName") or data.get("intentName") or ""
        parsed["source_state"] = _state_name(edge.get("source"))
        parsed["target_state"] = _state_name(edge.get("target"))
        inbound.append(parsed)

    if outbound or inbound:
        agent._a2a = {"outbound": outbound, "inbound": inbound}
    return agent
