"""Applying the user's Refine-KG decisions to a Knowledge Graph.

A decision is a :class:`KGResolutionInput` (``issueId`` + ``"accept"`` /
``"skip"``). It refers either to an issue of the static preflight, which is
re-run here, or to one of the LLM issues the editor round-trips in
``llmIssues``. Each is dispatched through ``kg_to_buml.resolutions``; a decision
that cannot be applied is a ``ConversionError`` (HTTP 400).
"""

from typing import Dict, List, Optional

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
from besser.utilities.web_modeling_editor.backend.models.kg import KGResolutionInput
from besser.utilities.web_modeling_editor.backend.models.responses import (
    KGActionResponse,
    KGIssueResponse,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


def _dispatch(
    kg: KnowledgeGraph,
    issue: KGIssue,
    decision: str,
    deferred_orphan_node_ids: Optional[List[str]],
) -> KnowledgeGraph:
    """Apply one decision; a deferred orphan classification is collected, not applied.

    With ``deferred_orphan_node_ids`` given, the deferred-LLM action
    (``defer_to_llm_classification``) leaves the KG unchanged and adds its node
    ids to that list, so ``/kg/apply-refinement`` can hand them to the AI tab.
    Without it (``/kg/to-class-diagram``) the deferral cannot be routed and
    fails the request.
    """
    try:
        return dispatch_decision(kg, issue, decision)
    except DeferredOrphanClassification as deferred:
        if deferred_orphan_node_ids is None:
            raise ConversionError(
                "An orphan-classification decision was deferred to the LLM, but this "
                "endpoint cannot route deferred decisions. Use /kg/apply-refinement instead."
            ) from deferred
        for nid in deferred.node_ids:
            if nid not in deferred_orphan_node_ids:
                deferred_orphan_node_ids.append(nid)
        return kg
    except ResolutionError as exc:
        raise ConversionError(str(exc)) from exc


def apply_decisions(
    kg: KnowledgeGraph,
    decisions: List[KGResolutionInput],
    *,
    deferred_orphan_node_ids: Optional[List[str]] = None,
) -> KnowledgeGraph:
    """Apply ``decisions`` to the issues of the static preflight of ``kg``.

    The preflight is re-run after every applied decision, because each one
    changes the graph the next issue is looked up in. A decision whose issue
    no longer exists (an earlier decision resolved it) is skipped. See
    :func:`_dispatch` for ``deferred_orphan_node_ids``.

    Returns:
        The resolved KG (a mutated deep copy; ``kg`` itself is untouched).
    """
    issues_by_id = {i.id: i for i in analyze_kg_for_class_diagram(kg).issues}
    resolved_kg = kg
    for entry in decisions:
        issue = issues_by_id.get(entry.issueId)
        if issue is None:
            continue
        before = resolved_kg
        resolved_kg = _dispatch(resolved_kg, issue, entry.decision, deferred_orphan_node_ids)
        if resolved_kg is not before:
            # The topology has shifted: look the next issues up in the new graph.
            issues_by_id = {i.id: i for i in analyze_kg_for_class_diagram(resolved_kg).issues}
    return resolved_kg


def resolve_kg(kg: KnowledgeGraph, input_data: KGDiagramInput) -> KnowledgeGraph:
    """Apply ``input_data.resolutions`` to ``kg`` (``/kg/to-class-diagram``).

    Raises:
        ConversionError: a decision cannot be applied (HTTP 400).
    """
    if not input_data.resolutions:
        return kg
    return apply_decisions(kg, input_data.resolutions)


def _action(action: Optional[KGActionResponse]) -> Optional[KGAction]:
    if action is None or not action.key:
        return None
    return KGAction(key=action.key, parameters=dict(action.parameters), label=action.label)


def llm_issues_by_id(issues: List[KGIssueResponse]) -> Dict[str, KGIssue]:
    """Rebuild the :class:`KGIssue` objects of a round-tripped LLM issue list, by id.

    An issue without a recommended action has nothing to apply and is left out;
    one without a skip action gets a "keep as-is" no-op.
    """
    out: Dict[str, KGIssue] = {}
    for entry in issues:
        recommended = _action(entry.recommendedAction)
        if recommended is None:
            continue
        out[entry.id] = KGIssue(
            id=entry.id,
            code=entry.code or "LLM_CLEANUP",
            description=entry.description,
            affected_node_ids=list(entry.affectedNodeIds),
            affected_edge_ids=list(entry.affectedEdgeIds),
            recommended_action=recommended,
            skip_action=_action(entry.skipAction) or KGAction(key="noop", parameters={}, label="Keep as-is"),
        )
    return out


def apply_llm_decisions(
    kg: KnowledgeGraph,
    issues: Optional[List[KGIssueResponse]],
    decisions: Optional[List[KGResolutionInput]],
    *,
    deferred_orphan_node_ids: Optional[List[str]] = None,
) -> KnowledgeGraph:
    """Apply decisions to round-tripped LLM issues.

    Unlike :func:`apply_decisions`, the issues are not recomputed between
    decisions: re-calling the LLM per decision would be slow and
    non-deterministic, and the issue list is already a contract between client
    and server. A decision for an issue not in the list is skipped.

    Raises:
        ConversionError: there are decisions but no usable ``llmIssues``, or a
            decision cannot be applied (HTTP 400).
    """
    if not decisions:
        return kg
    issues_by_id = llm_issues_by_id(issues or [])
    if not issues_by_id:
        raise ConversionError(
            "No 'llmIssues' provided; applying LLM decisions needs the issue list "
            "returned by /kg/llm-clean or /kg/classify-orphans."
        )
    resolved = kg
    for entry in decisions:
        issue = issues_by_id.get(entry.issueId)
        if issue is None:
            continue
        resolved = _dispatch(resolved, issue, entry.decision, deferred_orphan_node_ids)
    return resolved
