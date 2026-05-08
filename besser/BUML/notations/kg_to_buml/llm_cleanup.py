"""LLM-driven KG cleanup: ask GPT-4o to suggest schema-oriented edits.

The user supplies a natural-language description of the system they want
to build. We send that description plus a compact snapshot of the KG to
GPT-4o, parse its JSON response into the same :class:`KGIssue` /
:class:`KGAction` shape used by the rule-based preflight, and return a
:class:`KGPreflightReport` so the frontend can render it through the
existing accept/skip UI.

The LLM may emit any of the keys in :data:`LLM_ALLOWED_KEYS`; anything
else is dropped at validation time. Suggestions referencing unknown
node ids are also dropped, as are duplicates (same action key + sorted
parameters), and the list is capped at :data:`MAX_LLM_SUGGESTIONS`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGProperty,
    KnowledgeGraph,
)
from besser.BUML.notations.kg_to_buml._common import RDF_TYPE, normalize_predicate
from besser.BUML.notations.kg_to_buml.preflight import (
    KGAction,
    KGIssue,
    KGPreflightReport,
    kg_signature,
)
from besser.utilities.llm_utils import clean_json_response, parse_json_safely


__all__ = [
    "LLM_ALLOWED_KEYS",
    "ORPHAN_ALLOWED_KEYS",
    "SNAPSHOT_FULL_THRESHOLD",
    "SAMPLE_INDIVIDUALS",
    "MAX_LLM_SUGGESTIONS",
    "MAX_ORPHAN_BATCH",
    "ORPHAN_CONTEXT_NEIGHBORS",
    "build_kg_snapshot",
    "parse_llm_suggestions",
    "analyze_kg_with_description",
    "classify_orphan_nodes_with_llm",
]


logger = logging.getLogger(__name__)


# Action keys the LLM is allowed to emit. Anything else is dropped during
# validation. These map 1:1 to handler keys registered in resolutions.py.
LLM_ALLOWED_KEYS: Set[str] = {
    "drop_class",
    "drop_individual",
    "drop_node",
    "drop_link",
    "promote_individual_to_class",
    "reclassify_node",
    "type_individual_as_class",
}

# Priority order for surfacing suggestions in the UI. Lower = earlier.
# The ordering reflects how impactful the fix is for the eventual class
# diagram: schema-level fixes (classes / properties) come first, then
# instance-level (individuals), then literals, then blank nodes.
_NODE_KIND_PRIORITY: Dict[type, int] = {
    KGClass: 0,
    KGProperty: 0,
    KGIndividual: 1,
    KGLiteral: 2,
    KGBlank: 3,
}
_DEFAULT_NODE_PRIORITY = 4

# When a suggestion's affected_node_ids list is empty (e.g. drop_link),
# fall back to a per-action priority so it lands in a sensible bucket.
_ACTION_KEY_PRIORITY: Dict[str, int] = {
    "drop_class": 0,
    "drop_individual": 1,
    "promote_individual_to_class": 0,  # creates a class — schema-level fix
    "type_individual_as_class": 1,
    "reclassify_node": 0,  # changes node kind — schema-level
    "drop_link": 1,        # edges between individuals are instance-level
    "drop_node": 4,        # generic — depends on actual kind
}

SNAPSHOT_FULL_THRESHOLD = 150
SAMPLE_INDIVIDUALS = 30
EDGE_SAMPLE_PER_PREDICATE = 5
BLANK_METADATA_SAMPLE = 5
MAX_LLM_SUGGESTIONS = 50

# Orphan-classification: per-node LLM pass for nodes with no class anchoring.
# A small allowed-keys set keeps the LLM focused on classify-or-drop choices.
ORPHAN_ALLOWED_KEYS: Set[str] = {
    "drop_node",
    "promote_individual_to_class",
    "type_individual_as_class",
}
# Max orphan nodes per single LLM call. Larger batches go in multiple calls.
MAX_ORPHAN_BATCH = 40
# Max neighbours included in each per-orphan mini-snapshot.
ORPHAN_CONTEXT_NEIGHBORS = 5

OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


# ----------------------------------------------------------------------
# Snapshot
# ----------------------------------------------------------------------


def build_kg_snapshot(kg: KnowledgeGraph) -> Dict[str, Any]:
    """Produce a compact, JSON-serialisable snapshot of ``kg`` for the LLM.

    Below :data:`SNAPSHOT_FULL_THRESHOLD` total nodes, the full graph is
    embedded. Above it, only schema-level information is sent verbatim;
    individuals are sampled, blank nodes are summarised, and edges between
    instances are aggregated as a per-predicate histogram with a few
    sample pairs each.
    """
    classes = sorted(
        ({"id": n.id, "label": n.label, "iri": n.iri} for n in kg.nodes if isinstance(n, KGClass)),
        key=lambda d: d["id"],
    )
    properties = sorted(
        ({"id": n.id, "label": n.label, "iri": n.iri} for n in kg.nodes if isinstance(n, KGProperty)),
        key=lambda d: d["id"],
    )
    individuals_list = sorted(
        [n for n in kg.nodes if isinstance(n, KGIndividual)],
        key=lambda n: n.id,
    )
    blanks_list = sorted(
        [n for n in kg.nodes if isinstance(n, KGBlank)],
        key=lambda n: n.id,
    )
    literals_list = sorted(
        [n for n in kg.nodes if isinstance(n, KGLiteral)],
        key=lambda n: n.id,
    )

    # rdf:type lookup so each individual carries its declared types.
    types_of: Dict[str, List[str]] = {}
    for edge in kg.edges:
        if normalize_predicate(edge.iri) != RDF_TYPE:
            continue
        if isinstance(edge.source, KGIndividual):
            types_of.setdefault(edge.source.id, []).append(
                edge.target.iri or edge.target.id
            )

    axioms = [_axiom_to_dict(a) for a in kg.axioms]
    full_graph = len(kg.nodes) <= SNAPSHOT_FULL_THRESHOLD

    snapshot: Dict[str, Any] = {
        "mode": "full" if full_graph else "summary",
        "node_count": len(kg.nodes),
        "edge_count": len(kg.edges),
        "classes": classes,
        "properties": properties,
        "axioms": axioms,
    }

    if full_graph:
        snapshot["individuals"] = [
            {
                "id": n.id,
                "label": n.label,
                "iri": n.iri,
                "types": sorted(set(types_of.get(n.id, []))),
            }
            for n in individuals_list
        ]
        snapshot["blanks"] = [
            {"id": n.id, "label": n.label, "metadata": dict(n.metadata)}
            for n in blanks_list
        ]
        snapshot["literals"] = [
            {"id": n.id, "value": n.value, "datatype": n.datatype}
            for n in literals_list
        ]
        snapshot["edges"] = sorted(
            (
                {
                    "id": e.id,
                    "source": e.source.id,
                    "target": e.target.id,
                    "label": e.label,
                    "iri": e.iri,
                }
                for e in kg.edges
            ),
            key=lambda d: d["id"],
        )
    else:
        snapshot["individuals_sample"] = [
            {
                "id": n.id,
                "label": n.label,
                "iri": n.iri,
                "types": sorted(set(types_of.get(n.id, []))),
            }
            for n in individuals_list[:SAMPLE_INDIVIDUALS]
        ]
        snapshot["individual_count"] = len(individuals_list)
        snapshot["blank_count"] = len(blanks_list)
        snapshot["blank_metadata_sample"] = [
            {"id": n.id, "metadata": dict(n.metadata)}
            for n in blanks_list[:BLANK_METADATA_SAMPLE]
        ]
        snapshot["literal_count"] = len(literals_list)
        snapshot["edges_by_predicate"] = _edges_histogram(kg)

    return snapshot


def _edges_histogram(kg: KnowledgeGraph) -> List[Dict[str, Any]]:
    """Aggregate edges by predicate IRI: count + a handful of sample pairs."""
    pred_counter: Counter[str] = Counter()
    samples: Dict[str, List[Tuple[str, str]]] = {}
    for edge in sorted(kg.edges, key=lambda e: e.id):
        pred = edge.iri or edge.label or ""
        pred_counter[pred] += 1
        if len(samples.setdefault(pred, [])) < EDGE_SAMPLE_PER_PREDICATE:
            samples[pred].append((edge.source.id, edge.target.id))
    return [
        {
            "predicate": pred,
            "count": pred_counter[pred],
            "samples": [{"source": s, "target": t} for s, t in samples[pred]],
        }
        for pred in sorted(pred_counter)
    ]


def _axiom_to_dict(axiom: Any) -> Dict[str, Any]:
    """Best-effort dict view of an axiom dataclass."""
    payload: Dict[str, Any] = {"kind": type(axiom).__name__}
    for field_name in (
        "class_ids",
        "property_ids",
        "chain_property_ids",
        "part_class_ids",
        "union_class_id",
        "sub_property_id",
        "super_property_id",
        "property_a_id",
        "property_b_id",
        "property_id",
        "class_id",
        "target_iri",
        "source_ontology_iri",
    ):
        if hasattr(axiom, field_name):
            payload[field_name] = getattr(axiom, field_name)
    return payload


# ----------------------------------------------------------------------
# Suggestion parsing / validation
# ----------------------------------------------------------------------


def _llm_issue_id(action_key: str, affected_node_ids: List[str], affected_edge_ids: List[str]) -> str:
    parts = [action_key, *sorted(affected_node_ids), *sorted(affected_edge_ids)]
    raw = "LLM_CLEANUP\0" + "\0".join(parts)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:12]


def _dedup_key(action_key: str, parameters: Dict[str, Any]) -> Tuple:
    """Stable key for de-duplicating identical suggestions."""
    flat: List[Tuple[str, str]] = []
    for k in sorted(parameters):
        v = parameters[k]
        try:
            flat.append((k, json.dumps(v, sort_keys=True, default=str)))
        except (TypeError, ValueError):
            flat.append((k, repr(v)))
    return (action_key, tuple(flat))


def parse_llm_suggestions(
    raw_json: str,
    kg: KnowledgeGraph,
    *,
    allowed_keys: Optional[Set[str]] = None,
) -> List[KGIssue]:
    """Parse the LLM's raw response into validated :class:`KGIssue` objects.

    Drops suggestions that are malformed, reference unknown nodes/edges,
    use a key outside ``allowed_keys`` (default :data:`LLM_ALLOWED_KEYS`),
    or duplicate an earlier suggestion. The result is capped at
    :data:`MAX_LLM_SUGGESTIONS`.

    The ``allowed_keys`` argument lets specialised callers (e.g. the
    per-node orphan classifier) tighten the action set without duplicating
    the validator.
    """
    if not raw_json:
        return []
    keys = allowed_keys if allowed_keys is not None else LLM_ALLOWED_KEYS
    cleaned = clean_json_response(raw_json)
    parsed = parse_json_safely(cleaned)
    if not isinstance(parsed, dict):
        return []
    raw_suggestions = parsed.get("suggestions") or parsed.get("edits") or []
    if not isinstance(raw_suggestions, list):
        return []

    nodes_by_id: Dict[str, KGNode] = {n.id: n for n in kg.nodes}
    live_node_ids = set(nodes_by_id)
    live_edge_ids = {e.id for e in kg.edges}

    candidates: List[Tuple[KGIssue, int, float, str]] = []
    seen_keys: Set[Tuple] = set()

    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue
        recommended = item.get("recommended_action") or item.get("action") or {}
        if not isinstance(recommended, dict):
            continue
        action_key = recommended.get("key")
        if action_key not in keys:
            continue
        parameters = recommended.get("parameters") or {}
        if not isinstance(parameters, dict):
            continue
        affected_node_ids = list(item.get("affected_node_ids") or [])
        affected_edge_ids = list(item.get("affected_edge_ids") or [])

        # node_id parameters must reference a real node.
        node_id_param = parameters.get("node_id")
        if node_id_param and node_id_param not in live_node_ids:
            continue
        edge_id_param = parameters.get("edge_id")
        if edge_id_param and edge_id_param not in live_edge_ids:
            continue
        # affected_*_ids must reference real ids.
        if any(nid not in live_node_ids for nid in affected_node_ids):
            continue
        if any(eid not in live_edge_ids for eid in affected_edge_ids):
            continue

        # Make sure each suggestion has at least one anchor we can resolve.
        if not (node_id_param or edge_id_param or affected_node_ids or affected_edge_ids):
            continue

        # Default affected_node_ids from the action's node_id when missing.
        if not affected_node_ids and node_id_param:
            affected_node_ids = [node_id_param]
        if not affected_edge_ids and edge_id_param:
            affected_edge_ids = [edge_id_param]

        # Per-action kind / parameter validation. Anything inconsistent is
        # dropped here so a hallucinated suggestion never reaches the
        # resolution dispatcher (which would raise mid-apply).
        if action_key == "reclassify_node":
            target_kind = (parameters.get("target_kind") or "").lower()
            if target_kind not in {"class", "individual", "property", "blank"}:
                continue
            parameters = {**parameters, "target_kind": target_kind}
        elif action_key == "drop_class":
            if not isinstance(nodes_by_id.get(node_id_param), KGClass):
                continue
        elif action_key in {"drop_individual", "promote_individual_to_class"}:
            if not isinstance(nodes_by_id.get(node_id_param), KGIndividual):
                continue
        elif action_key == "type_individual_as_class":
            class_id_param = parameters.get("class_id")
            if not class_id_param or class_id_param not in live_node_ids:
                continue
            if not isinstance(nodes_by_id.get(node_id_param), KGIndividual):
                continue
            if not isinstance(nodes_by_id.get(class_id_param), KGClass):
                continue
            # Surface the target class in affected_node_ids so the user
            # can see "this individual will be typed as <Class>" in the UI
            # and the issue routes to the right priority bucket.
            if class_id_param not in affected_node_ids:
                affected_node_ids = [*affected_node_ids, class_id_param]

        dedup = _dedup_key(action_key, parameters)
        if dedup in seen_keys:
            continue
        seen_keys.add(dedup)

        code = item.get("code") or _default_code_for(action_key)
        description = item.get("description") or recommended.get("label") or code
        label = recommended.get("label") or description

        issue = KGIssue(
            id=_llm_issue_id(action_key, affected_node_ids, affected_edge_ids),
            code=code,
            description=description,
            affected_node_ids=affected_node_ids,
            affected_edge_ids=affected_edge_ids,
            recommended_action=KGAction(key=action_key, parameters=parameters, label=label),
            skip_action=KGAction(key="noop", parameters={}, label="Keep as-is"),
        )

        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        kind_priority = _issue_node_priority(affected_node_ids, nodes_by_id, action_key)
        candidates.append((issue, kind_priority, confidence, code))

    # Surface schema-level fixes first (classes / properties), then
    # individuals, then literals, then blank nodes — matches how impactful
    # the fix is for the eventual class diagram. Within a bucket: confidence
    # desc → code asc → id asc.
    candidates.sort(key=lambda quad: (quad[1], -quad[2], quad[3], quad[0].id))
    return [issue for issue, _prio, _conf, _code in candidates[:MAX_LLM_SUGGESTIONS]]


def _default_code_for(action_key: str) -> str:
    return {
        "drop_class": "LLM_DROP_CLASS",
        "drop_individual": "LLM_DROP_INDIVIDUAL",
        "drop_node": "LLM_DROP_NODE",
        "drop_link": "LLM_DROP_LINK",
        "promote_individual_to_class": "LLM_PROMOTE_INDIVIDUAL",
        "reclassify_node": "LLM_RECLASSIFY_NODE",
        "type_individual_as_class": "LLM_TYPE_INDIVIDUAL",
    }.get(action_key, "LLM_CLEANUP")


def _node_kind_priority(node: Optional[KGNode]) -> int:
    if node is None:
        return _DEFAULT_NODE_PRIORITY
    return _NODE_KIND_PRIORITY.get(type(node), _DEFAULT_NODE_PRIORITY)


def _issue_node_priority(
    affected_node_ids: List[str],
    nodes_by_id: Dict[str, KGNode],
    action_key: str,
) -> int:
    """Compute the bucket priority for a single suggestion.

    Most actions encode their priority directly (``drop_class`` is always a
    schema-level fix, ``type_individual_as_class`` is always an instance-level
    fix even though it references a target class id). For the generic
    ``drop_node`` action the priority is derived from the actual node kind,
    since the action itself is type-agnostic.
    """
    if action_key == "drop_node":
        if not affected_node_ids:
            return _DEFAULT_NODE_PRIORITY
        return min(
            _node_kind_priority(nodes_by_id.get(nid))
            for nid in affected_node_ids
        )
    return _ACTION_KEY_PRIORITY.get(action_key, _DEFAULT_NODE_PRIORITY)


# ----------------------------------------------------------------------
# Prompt + analyse
# ----------------------------------------------------------------------


def _build_prompt(snapshot: Dict[str, Any], description: str) -> Tuple[str, str]:
    """Compose the system + user messages for GPT-4o."""
    allowed = sorted(LLM_ALLOWED_KEYS)
    system = (
        "You are a senior software architect helping clean up a Knowledge Graph (KG) "
        "so it can be transformed into a UML class diagram for a specific software "
        "system. The user describes the system they want to build; you propose "
        "concrete edits to the KG that remove noise, fix misclassifications, and "
        "drop entities unrelated to the described system.\n\n"
        "You MUST return a single JSON object — no prose, no markdown fences — "
        "with this exact shape:\n"
        "{\n"
        "  \"suggestions\": [\n"
        "    {\n"
        "      \"code\": \"<short code, e.g. LLM_DROP_CLASS>\",\n"
        "      \"description\": \"<human-readable explanation, 1-2 sentences>\",\n"
        "      \"affected_node_ids\": [\"<node id from the snapshot>\", ...],\n"
        "      \"affected_edge_ids\": [\"<edge id from the snapshot>\", ...],\n"
        "      \"confidence\": 0.0-1.0,\n"
        "      \"recommended_action\": {\n"
        "        \"key\": \"<one of the allowed keys below>\",\n"
        "        \"parameters\": { ... },\n"
        "        \"label\": \"<short label for the accept button>\"\n"
        "      }\n"
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        f"Allowed action keys: {allowed}.\n\n"
        "Required parameters per key:\n"
        "- drop_class: { node_id } — the node MUST be a class in the snapshot.\n"
        "- drop_individual: { node_id } — the node MUST be an individual.\n"
        "- drop_node: { node_id } — drop any node (use for orphans, blanks, etc.).\n"
        "- drop_link: { edge_id } — remove a single edge by id.\n"
        "- promote_individual_to_class: { node_id, new_label?, new_iri? } — node MUST be an individual.\n"
        "- reclassify_node: { node_id, target_kind } where target_kind is one of "
        "'class', 'individual', 'property', 'blank'.\n"
        "- type_individual_as_class: { node_id, class_id } — adds an rdf:type edge from "
        "an individual to an existing class. ``node_id`` MUST be an individual in the "
        "snapshot; ``class_id`` MUST be a class in the snapshot. Use this whenever "
        "an individual is clearly relevant to the described system but lacks an "
        "rdf:type link — DO NOT drop it; type it instead.\n\n"
        "Decision policy:\n"
        "- For each individual that lacks classification but seems to belong to one of "
        "the listed classes, PREFER 'type_individual_as_class' over 'drop_individual'. "
        "Drop individuals only when they are unrelated to the described system.\n"
        "- When in doubt, prefer 'noop' (omit the suggestion) over an aggressive drop.\n\n"
        "Hard rules:\n"
        "1. Every id you cite MUST appear verbatim in the snapshot — never invent ids.\n"
        "2. Output ONLY valid JSON. No prose, no markdown.\n"
        "3. Do not produce duplicates (same action + same parameters).\n"
        "4. Be conservative: only suggest edits you are confident improve the KG "
        "for the described system. It is fine to return an empty 'suggestions' list.\n"
        "5. Tackle schema-level fixes first (classes, properties, reclassifications), "
        "then individuals, then literals, then blank nodes — this is how the user "
        "will review them.\n"
    )

    user = (
        f"=== Target system description ===\n{description.strip()}\n\n"
        f"=== Knowledge Graph snapshot ===\n"
        f"{json.dumps(snapshot, indent=2, default=str)}\n"
    )
    return system, user


def analyze_kg_with_description(
    kg: KnowledgeGraph,
    description: str,
    api_key: str,
    model: str = "gpt-4o",
    *,
    _http_post: Optional[Callable[..., Any]] = None,
) -> KGPreflightReport:
    """Call GPT-4o for KG cleanup suggestions and wrap them in a preflight report.

    The ``_http_post`` keyword is for tests — production callers should leave
    it ``None`` so the function looks up :func:`requests.post` at call time
    (this lets monkeypatches on ``requests.post`` take effect). Network or
    parse failures raise :class:`RuntimeError` with a contextual message;
    the caller is expected to translate that into a user-facing 4xx/5xx
    response.
    """
    if not isinstance(kg, KnowledgeGraph):
        raise TypeError("analyze_kg_with_description expects a KnowledgeGraph instance.")
    if not description or not description.strip():
        raise ValueError("A non-empty system description is required.")
    if not api_key or not api_key.strip():
        raise ValueError("A non-empty OpenAI API key is required.")

    http_post = _http_post if _http_post is not None else requests.post

    snapshot = build_kg_snapshot(kg)
    system_msg, user_msg = _build_prompt(snapshot, description)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 4000,
    }

    try:
        response = http_post(OPENAI_ENDPOINT, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    if not getattr(response, "ok", False):
        status = getattr(response, "status_code", "?")
        body = getattr(response, "text", "")
        raise RuntimeError(f"OpenAI API call failed: HTTP {status} {body}")

    try:
        response_json = response.json()
    except ValueError as exc:
        raise RuntimeError("OpenAI response was not valid JSON.") from exc

    if isinstance(response_json, dict) and "error" in response_json:
        message = (response_json.get("error") or {}).get("message") or response_json["error"]
        raise RuntimeError(f"OpenAI API error: {message}")

    raw_content = (
        (response_json.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not raw_content:
        raise RuntimeError("OpenAI returned an empty completion.")

    issues = parse_llm_suggestions(raw_content, kg)
    return KGPreflightReport(
        issues=issues,
        issue_count=len(issues),
        kg_signature=kg_signature(kg),
        diagram_type="LLMCleanup",
    )


# ----------------------------------------------------------------------
# Per-node orphan classification
# ----------------------------------------------------------------------


def _node_kind(node: KGNode) -> str:
    if isinstance(node, KGClass):
        return "class"
    if isinstance(node, KGIndividual):
        return "individual"
    if isinstance(node, KGProperty):
        return "property"
    if isinstance(node, KGLiteral):
        return "literal"
    if isinstance(node, KGBlank):
        return "blank"
    return "unknown"


def _build_orphan_node_snapshot(node: KGNode, kg: KnowledgeGraph) -> Dict[str, Any]:
    """Compact snapshot for one orphan node: id, kind, label, IRI, value, and
    up to :data:`ORPHAN_CONTEXT_NEIGHBORS` incident edges."""
    incident: List[Dict[str, Any]] = []
    for edge in kg.edges:
        if edge.source.id == node.id:
            other = edge.target
            direction = "out"
        elif edge.target.id == node.id:
            other = edge.source
            direction = "in"
        else:
            continue
        incident.append({
            "edge_id": edge.id,
            "predicate": edge.iri or edge.label,
            "direction": direction,
            "other": other.id,
            "other_label": other.label,
            "other_kind": _node_kind(other),
        })
        if len(incident) >= ORPHAN_CONTEXT_NEIGHBORS:
            break

    snap: Dict[str, Any] = {
        "id": node.id,
        "kind": _node_kind(node),
        "label": getattr(node, "label", None),
        "iri": getattr(node, "iri", None),
        "incident_edges": incident,
    }
    if isinstance(node, KGLiteral):
        snap["value"] = getattr(node, "value", None)
        snap["datatype"] = getattr(node, "datatype", None)
    return snap


def _build_orphan_classification_prompt(
    classes: List[Dict[str, Any]],
    orphans: List[Dict[str, Any]],
    description: str,
) -> Tuple[str, str]:
    """Compose the system + user messages for per-node orphan classification."""
    allowed = sorted(ORPHAN_ALLOWED_KEYS)
    class_id_list = sorted({c["id"] for c in classes})
    system = (
        "You are a senior software architect cleaning up a Knowledge Graph (KG) "
        "for a target software system. The user has already cleaned schema-level "
        "noise; what remains is a batch of *orphan* nodes — nodes with no path to "
        "any owl:Class. For each orphan node, decide whether to classify it (so "
        "it survives in the class diagram) or drop it (so it doesn't pollute the "
        "diagram).\n\n"
        "Return a single JSON object — no prose, no markdown — with this shape:\n"
        "{\n"
        "  \"suggestions\": [\n"
        "    {\n"
        "      \"code\": \"<short code, e.g. LLM_TYPE_INDIVIDUAL>\",\n"
        "      \"description\": \"<1-2 sentence rationale>\",\n"
        "      \"affected_node_ids\": [\"<orphan node id>\"],\n"
        "      \"confidence\": 0.0-1.0,\n"
        "      \"recommended_action\": {\n"
        "        \"key\": \"<one of the allowed keys below>\",\n"
        "        \"parameters\": { ... },\n"
        "        \"label\": \"<short button label>\"\n"
        "      }\n"
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        f"Allowed action keys: {allowed}.\n\n"
        "Required parameters per key:\n"
        "- type_individual_as_class: { node_id, class_id } — node MUST be an "
        "individual; class_id MUST be one of the existing classes listed below.\n"
        "- promote_individual_to_class: { node_id, new_label?, new_iri? } — node "
        "MUST be an individual; promotes it to a brand-new class with the same id.\n"
        "- drop_node: { node_id } — drops the orphan (use for unrelated noise, "
        "literals, blanks, individuals that don't fit the system).\n\n"
        f"Existing class ids you may reference: {class_id_list}\n\n"
        "Decision policy (in order of preference):\n"
        "1. If the orphan is an individual and clearly fits one of the listed "
        "classes → 'type_individual_as_class'.\n"
        "2. If the orphan is an individual whose label looks like a missing class "
        "(e.g. capitalised, type-shaped) → 'promote_individual_to_class'.\n"
        "3. Otherwise (literal, blank, or unrelated to the target system) → "
        "'drop_node'.\n\n"
        "Hard rules:\n"
        "1. Every node_id and class_id MUST appear verbatim in the snapshot — "
        "never invent ids.\n"
        "2. Each orphan must appear in AT MOST ONE suggestion.\n"
        "3. If you have no confident classification for an orphan, OMIT it — the "
        "system will treat omissions as 'drop' candidates the user can review.\n"
        "4. Output ONLY valid JSON. No prose, no markdown.\n"
    )
    user = (
        f"=== Target system description ===\n{description.strip()}\n\n"
        f"=== Existing classes ===\n{json.dumps(classes, indent=2, default=str)}\n\n"
        f"=== Orphan nodes to classify ===\n"
        f"{json.dumps(orphans, indent=2, default=str)}\n"
    )
    return system, user


def _synthesize_drop_node_issue(node: KGNode) -> KGIssue:
    """Build a synthetic 'drop_node' issue for an orphan the LLM didn't cover."""
    label = getattr(node, "label", None) or node.id
    description = (
        f"The LLM did not classify '{label}'. Drop it from the KG by default, "
        f"or skip to keep it as-is."
    )
    return KGIssue(
        id=_llm_issue_id("drop_node", [node.id], []),
        code="LLM_DROP_NODE",
        description=description,
        affected_node_ids=[node.id],
        affected_edge_ids=[],
        recommended_action=KGAction(
            key="drop_node",
            parameters={"node_id": node.id},
            label=f"Drop '{label}'",
        ),
        skip_action=KGAction(key="noop", parameters={}, label="Keep as-is"),
    )


def classify_orphan_nodes_with_llm(
    kg: KnowledgeGraph,
    node_ids: List[str],
    description: str,
    api_key: str,
    model: str = "gpt-4o",
    *,
    _http_post: Optional[Callable[..., Any]] = None,
) -> KGPreflightReport:
    """Per-node LLM classification for a batch of orphan nodes.

    For each orphan in ``node_ids`` the LLM gets a focused mini-snapshot
    (label, IRI, kind, a few incident edges, plus the full class list) and
    returns one suggestion per orphan it can confidently classify. Orphans
    the LLM omits get a synthetic ``drop_node`` issue so the user still
    sees and decides on them.

    The batch is split into chunks of :data:`MAX_ORPHAN_BATCH`, one HTTP
    call per chunk. Results are merged into a single report with
    ``diagram_type = "OrphanClassification"``.

    Network or parse failures raise :class:`RuntimeError` with a contextual
    message; the caller is expected to translate that into a 4xx/5xx.
    """
    if not isinstance(kg, KnowledgeGraph):
        raise TypeError(
            "classify_orphan_nodes_with_llm expects a KnowledgeGraph instance."
        )
    if not description or not description.strip():
        raise ValueError("A non-empty system description is required.")
    if not api_key or not api_key.strip():
        raise ValueError("A non-empty OpenAI API key is required.")
    if not isinstance(node_ids, list):
        raise TypeError("node_ids must be a list.")
    if not node_ids:
        return KGPreflightReport(
            issues=[],
            issue_count=0,
            kg_signature=kg_signature(kg),
            diagram_type="OrphanClassification",
        )

    nodes_by_id = {n.id: n for n in kg.nodes}
    orphan_nodes: List[KGNode] = []
    seen: Set[str] = set()
    for nid in node_ids:
        if nid in seen:
            continue
        seen.add(nid)
        node = nodes_by_id.get(nid)
        if node is None:
            continue
        orphan_nodes.append(node)

    if not orphan_nodes:
        return KGPreflightReport(
            issues=[],
            issue_count=0,
            kg_signature=kg_signature(kg),
            diagram_type="OrphanClassification",
        )

    classes = sorted(
        ({"id": n.id, "label": n.label, "iri": n.iri} for n in kg.nodes if isinstance(n, KGClass)),
        key=lambda d: d["id"],
    )

    http_post = _http_post if _http_post is not None else requests.post
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }

    classified: Dict[str, KGIssue] = {}

    for chunk_start in range(0, len(orphan_nodes), MAX_ORPHAN_BATCH):
        chunk = orphan_nodes[chunk_start: chunk_start + MAX_ORPHAN_BATCH]
        snaps = [_build_orphan_node_snapshot(n, kg) for n in chunk]
        system_msg, user_msg = _build_orphan_classification_prompt(
            classes=classes,
            orphans=snaps,
            description=description,
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_completion_tokens": 4000,
        }
        try:
            response = http_post(
                OPENAI_ENDPOINT, headers=headers, json=payload, timeout=120
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        if not getattr(response, "ok", False):
            status = getattr(response, "status_code", "?")
            body = getattr(response, "text", "")
            raise RuntimeError(f"OpenAI API call failed: HTTP {status} {body}")
        try:
            response_json = response.json()
        except ValueError as exc:
            raise RuntimeError("OpenAI response was not valid JSON.") from exc
        if isinstance(response_json, dict) and "error" in response_json:
            message = (
                (response_json.get("error") or {}).get("message")
                or response_json["error"]
            )
            raise RuntimeError(f"OpenAI API error: {message}")

        raw_content = (
            (response_json.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not raw_content:
            continue

        chunk_orphan_ids = {n.id for n in chunk}
        chunk_issues = parse_llm_suggestions(
            raw_content, kg, allowed_keys=ORPHAN_ALLOWED_KEYS
        )
        for issue in chunk_issues:
            target = issue.recommended_action.parameters.get("node_id") if issue.recommended_action else None
            if target is None or target not in chunk_orphan_ids:
                continue
            # Each orphan appears in at most one suggestion: keep the first.
            if target in classified:
                continue
            classified[target] = issue

    # Synthesize drop_node for any orphan the LLM did not cover.
    final_issues: List[KGIssue] = []
    for node in orphan_nodes:
        if node.id in classified:
            final_issues.append(classified[node.id])
        else:
            final_issues.append(_synthesize_drop_node_issue(node))

    return KGPreflightReport(
        issues=final_issues,
        issue_count=len(final_issues),
        kg_signature=kg_signature(kg),
        diagram_type="OrphanClassification",
    )
