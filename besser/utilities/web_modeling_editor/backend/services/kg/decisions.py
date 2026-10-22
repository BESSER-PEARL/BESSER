"""Applying the user's Refine-KG decisions to a Knowledge Graph.

Two payload shapes arrive from the editor: ``[{issueId, decision}]`` for the
static preflight, and round-tripped LLM issues with the same decisions. Each is
dispatched through ``kg_to_buml.resolutions``; a failed dispatch is a
``ConversionError``.
"""

from typing import Any, Dict, List, Optional

from besser.BUML.metamodel.kg import KnowledgeGraph

from besser.BUML.notations.kg_to_buml import (
    DeferredOrphanClassification,
    KGAction,
    KGIssue,
    ResolutionError,
    analyze_kg_for_class_diagram,
    dispatch_decision,
)
from besser.utilities.web_modeling_editor.backend.models import KGDiagramInput
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


def apply_v2_decisions(
    kg: KnowledgeGraph,
    decisions: List[Any],
    *,
    deferred_orphan_node_ids: Optional[List[str]] = None,
) -> KnowledgeGraph:
    """Apply ``[{issueId, decision}]`` choices to the static preflight's issues.

    Re-runs the preflight on ``kg``, looks up each issue by id, dispatches its
    recommended/skip action. Returns the resolved KG (deep copy mutated).

    If ``deferred_orphan_node_ids`` is a list, the deferred-LLM action
    (``defer_to_llm_classification``) does not abort the loop — its node ids
    are accumulated into the list and the orphans stay in the KG so the AI
    tab can classify them next. Without that argument the deferred handler
    raises and the request fails (preserving back-compat with the legacy
    ``/kg-to-class-diagram`` path, which doesn't surface the deferred
    branch to the user).
    """
    report = analyze_kg_for_class_diagram(kg)
    issues_by_id = {i.id: i for i in report.issues}
    resolved_kg = kg
    for entry in decisions:
        if not isinstance(entry, dict):
            continue
        issue_id = entry.get("issueId") or entry.get("issue_id")
        decision = entry.get("decision")
        if not issue_id or not decision:
            raise ConversionError("Each resolution must include 'issueId' and 'decision'.")
        issue = issues_by_id.get(issue_id)
        if issue is None:
            # Issue might have already been resolved by a prior decision, or the
            # signature check would catch a stale graph; tolerate silently.
            continue
        try:
            resolved_kg = dispatch_decision(resolved_kg, issue, decision)
        except DeferredOrphanClassification as deferred:
            if deferred_orphan_node_ids is None:
                # No deferred channel available — propagate as a conversion error.
                raise ConversionError(
                    "An orphan-classification decision was deferred to the LLM, "
                    "but this endpoint cannot route deferred decisions. Use "
                    "/apply-kg-refinement instead."
                ) from deferred
            # Deferred → KG unchanged; accumulate node ids and continue.
            for nid in deferred.node_ids:
                if nid not in deferred_orphan_node_ids:
                    deferred_orphan_node_ids.append(nid)
            continue
        except ResolutionError as exc:
            raise ConversionError(str(exc)) from exc
        # Refresh the issues map after each apply: the topology has shifted.
        report = analyze_kg_for_class_diagram(resolved_kg)
        issues_by_id = {i.id: i for i in report.issues}
    return resolved_kg


def resolve_kg(kg: KnowledgeGraph, input_data: KGDiagramInput) -> KnowledgeGraph:
    """Apply ``input_data.resolutions`` (``[{issueId, decision}]``) to ``kg``.

    Raises:
        ConversionError: an entry lacks ``issueId`` / ``decision``, or its
            decision cannot be applied (HTTP 400).
    """
    if not input_data.resolutions:
        return kg
    return apply_v2_decisions(kg, input_data.resolutions)


def issues_payload_to_objects(payload: Any) -> Dict[str, KGIssue]:
    """Reconstruct KGIssue / KGAction objects from the JSON shape returned by
    ``/llm-clean-kg``. Returns a dict keyed by issue id for fast lookup."""
    if not isinstance(payload, list):
        return {}
    out: Dict[str, KGIssue] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        issue_id = entry.get("id")
        code = entry.get("code") or "LLM_CLEANUP"
        description = entry.get("description") or ""
        affected_node_ids = list(entry.get("affectedNodeIds") or entry.get("affected_node_ids") or [])
        affected_edge_ids = list(entry.get("affectedEdgeIds") or entry.get("affected_edge_ids") or [])
        recommended = action_payload_to_object(entry.get("recommendedAction") or entry.get("recommended_action"))
        skip = action_payload_to_object(entry.get("skipAction") or entry.get("skip_action"))
        if not issue_id or recommended is None:
            continue
        out[issue_id] = KGIssue(
            id=issue_id,
            code=code,
            description=description,
            affected_node_ids=affected_node_ids,
            affected_edge_ids=affected_edge_ids,
            recommended_action=recommended,
            skip_action=skip if skip is not None else KGAction(key="noop", parameters={}, label="Keep as-is"),
        )
    return out


def action_payload_to_object(action: Any) -> Optional[KGAction]:
    """Rebuild a :class:`KGAction` from its JSON shape; ``None`` when it has no ``key``."""
    if not isinstance(action, dict):
        return None
    key = action.get("key")
    if not key:
        return None
    parameters = action.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    label = action.get("label") or ""
    return KGAction(key=key, parameters=parameters, label=label)


def apply_llm_decisions(
    kg: KnowledgeGraph,
    issues_payload: Optional[List[Dict[str, Any]]],
    decisions: Optional[List[Dict[str, Any]]],
    *,
    deferred_orphan_node_ids: Optional[List[str]] = None,
) -> KnowledgeGraph:
    """Apply round-tripped LLM-issue decisions to a deep-copied KG.

    Unlike :func:`apply_v2_decisions`, this does NOT re-run the analyzer
    between decisions: re-calling the LLM per decision would be slow,
    non-deterministic, and unnecessary because the issue list is already
    a contract between client and server.

    Symmetrical with :func:`apply_v2_decisions`: ``deferred_orphan_node_ids``
    enables routing the deferred-LLM action through ``pendingOrphanClassification``.
    """
    if not decisions:
        return kg
    issues_by_id = issues_payload_to_objects(issues_payload or [])
    if not issues_by_id:
        raise ConversionError(
            "No 'llmIssues' provided; the LLM-cleanup apply endpoint needs the "
            "issue list returned by /llm-clean-kg."
        )
    resolved = kg
    for entry in decisions:
        if not isinstance(entry, dict):
            continue
        issue_id = entry.get("issueId") or entry.get("issue_id")
        decision = entry.get("decision")
        if not issue_id or decision not in {"accept", "skip"}:
            raise ConversionError(
                "Each LLM decision must include 'issueId' and 'decision' ('accept' or 'skip')."
            )
        issue = issues_by_id.get(issue_id)
        if issue is None:
            # Issue not in the round-tripped list — tolerate silently.
            continue
        try:
            resolved = dispatch_decision(resolved, issue, decision)
        except DeferredOrphanClassification as deferred:
            if deferred_orphan_node_ids is None:
                raise ConversionError(
                    "Orphan-classification deferral is not supported on this endpoint."
                ) from deferred
            for nid in deferred.node_ids:
                if nid not in deferred_orphan_node_ids:
                    deferred_orphan_node_ids.append(nid)
            continue
        except ResolutionError as exc:
            raise ConversionError(str(exc)) from exc
    return resolved
