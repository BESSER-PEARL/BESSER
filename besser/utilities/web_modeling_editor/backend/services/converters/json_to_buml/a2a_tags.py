"""Parse WME `a2a:` wire tags and annotate a BUML Agent with the parsed edges.

This module is the BESSER *consume* side of the explicit agent-to-agent (A2A)
annotations the Web Modeling Editor serializes on the Agent-diagram wire. It does
two things, both standalone (the core converter `agent_diagram_processor.py` is
NOT touched):

1. `parse_a2a_line` / `parse_a2a_out_block` — a tolerant parser for the wire
   grammar (semicolon-delimited key=value, leading `a2a:<dir>`):

       a2a:<dir>;peer=<name>;ref=<uuid|>;flow=<id>;[order=<n>;]kind=<...>

2. `annotate_agent_with_a2a` — a re-walk of the raw AgentDiagram JSON that stashes
   the parsed in/out edges on `agent._a2a` (a side-map, no metamodel change). It is a
   no-op (sets no attribute) when the diagram carries no `a2a:` tag, so legacy agents
   stay byte-identical and the generator's `getattr(agent, "_a2a", None)` is falsy.

Tolerant by contract: unknown keys ignored; missing optionals → defaults; malformed
input → None (caller drops it). NEVER raises on user data.
"""
from typing import Optional

_KINDS = {"delegates", "supervises", "revises", "collaborates"}

# Sentinel `order` when a tag omits it — sorts last, mirrors UNLIMITED_MAX_MULTIPLICITY.
_ORDER_SENTINEL = 9999


def parse_a2a_line(line: str) -> Optional[dict]:
    """Parse ONE `a2a:in`/`a2a:out` line into a dict, or None if it isn't an a2a tag.

    Returns: {dir, peer, ref, flow, order, kind} where
      - dir   : 'in' | 'out'
      - peer  : str (may be '' if absent — caller decides whether to drop)
      - ref   : str ('' when the peer lane was never linked → name-addressing)
      - flow  : str ('' if absent)
      - order : int (9999 sentinel when absent — sorts last)
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
    """Attach `agent._a2a = {outbound, inbound}` parsed from the AgentDiagram JSON.

    Standalone: does NOT touch process_agent_diagram. Call it right after the agent
    is built. No-op (no attribute set) when the diagram carries no a2a: tag, so
    legacy agents stay byte-identical and the generator's getattr(agent,'_a2a',None) is
    falsy. Reads only strings (peer/state/intent names), so it never needs the built
    State/Transition objects — fully decoupled from the metamodel.
    """
    if not isinstance(agent_diagram_json, dict):
        return agent
    model = agent_diagram_json.get("model") or agent_diagram_json
    elements = model.get("elements", {}) or {}
    relationships = model.get("relationships", {}) or {}

    def _state_name(elem_id):
        el = elements.get(elem_id, {})
        return el.get("name", "") if el.get("type") == "AgentState" else ""

    # ── outbound: a2a:out lines on each AgentState.description ──
    outbound = []
    for el in elements.values():
        if el.get("type") != "AgentState":
            continue
        for edge in parse_a2a_out_block(el.get("description")):
            edge["state"] = el.get("name", "")
            outbound.append(edge)

    # ── inbound: a2a:in on a when_intent_matched transition's `name` ──
    inbound = []
    for rel in relationships.values():
        if rel.get("type") not in ("AgentStateTransition", "AgentStateTransitionInit"):
            continue
        parsed = parse_a2a_line(rel.get("name", ""))
        if not (parsed and parsed["dir"] == "in" and parsed["peer"]):
            continue
        predefined = rel.get("predefined") or {}
        parsed["intent"] = predefined.get("intentName") or rel.get("intentName") or ""
        parsed["source_state"] = _state_name((rel.get("source") or {}).get("element"))
        parsed["target_state"] = _state_name((rel.get("target") or {}).get("element"))
        inbound.append(parsed)

    if outbound or inbound:
        agent._a2a = {"outbound": outbound, "inbound": inbound}
    return agent
