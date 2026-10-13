"""
Knowledge Graph Router

The Knowledge Graph (KG) endpoints of the BESSER web modeling editor backend:
ontology import/export (``/import-owl``, ``/export-kg-rdf``), the KG -> class
diagram conversion with its preflight and Refine-KG flow, the OWL/SHACL
consistency check, and the LLM-assisted cleanup endpoints.

Split out of ``conversion_router`` (one router per concern). The generic
BUML import/export dispatch in that module still recognises a KG file; only the
KG-specific endpoints and their helpers live here.
"""

import asyncio
import logging
import os
import tempfile
import json

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, File, UploadFile, Body, Form
from fastapi.responses import Response

# Backend response models
from besser.utilities.web_modeling_editor.backend.models.responses import (
    DiagramExportResponse,
    KGConsistencyReportResponse,
    KGPreflightReportResponse,
    KGRefinementResponse,
)

# BESSER image-to-UML and BUML utilities
from besser.utilities.kg_to_buml import kg_to_buml
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph
from besser.utilities.kg_to_owl import serialize_knowledge_graph

# BESSER KG → B-UML (deterministic, offline)
from besser.BUML.notations.kg_to_buml import (
    DeferredOrphanClassification,
    KGAction,
    KGIssue,
    KGResolution,
    ResolutionError,
    analyze_kg_for_class_diagram,
    analyze_kg_with_description,
    apply_resolutions,
    classify_orphan_nodes_with_llm,
    dispatch_decision,
    kg_signature,
    kg_to_class_diagram as kg_to_class_diagram_buml,
)
from dataclasses import asdict as _dataclass_asdict
from besser.utilities.llm_utils import LLMUpstreamError

# Custom exception types translated to HTTP codes by @handle_endpoint_errors.
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    UpstreamServiceError,
    ValidationError,
)

# BESSER utilities

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    # JSON to BUML converters
    process_kg_diagram,
    class_buml_to_json,
    kg_to_json,
)

# Backend services - Other services

# Backend configuration

# Backend constants
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    API_VERSION,
    TEMP_DIR_PREFIX,
    DEFAULT_KG_LLM_MODEL,
    MAX_OWL_SIZE,
    MAX_ORPHAN_CLASSIFY_NODES,
    ALLOWED_KG_EXTENSIONS,
    ALLOWED_OWL_EXTENSIONS,
)

# Centralized error handling
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)

logger = logging.getLogger(__name__)

# The upload validation helpers are shared with the other conversion endpoints and
# stay in ``conversion_router``; this router only borrows them.
from besser.utilities.web_modeling_editor.backend.routers.conversion_router import (
    MAX_BUML_SIZE,
    _validate_file_content,
    _validate_upload,
    _write_file,
)

# Same OpenAPI tag as before the split, so the published API schema is unchanged.
router = APIRouter(prefix="/besser_api", tags=["conversion"])


@router.post("/get-json-model-from-kg", response_model=DiagramExportResponse)
@handle_endpoint_errors("get_json_model_from_kg")
async def get_json_model_from_kg(
    kg_file: UploadFile = File(...),
    api_key: str = Form(...)
):
    """
    Accepts an .TTL, .RDF, or .JSON knowledge graphs and an OpenAI API key, uses BESSER's kgtouml feature to transform the kg into a BUML class diagram in JSON.
    """
    # Save uploaded KG to a temp folder
    kg_content = await kg_file.read()
    _validate_upload(kg_file, max_size=MAX_BUML_SIZE, allowed_extensions=ALLOWED_KG_EXTENSIONS, content=kg_content)

    with tempfile.TemporaryDirectory() as temp_dir:
        safe_filename = os.path.basename(kg_file.filename)
        kg_path = os.path.join(temp_dir, safe_filename)
        await _write_file(kg_path, kg_content)
        # Create a folder for the kg as expected by mockup_to_buml
        kg_folder = os.path.join(temp_dir, "kgs")
        os.makedirs(kg_folder, exist_ok=True)
        os.rename(kg_path, os.path.join(kg_folder, safe_filename))
        # Use kg_to_buml to generate DomainModel from kg and API key
        # Find the kg file path
        kg_files = [f for f in os.listdir(kg_folder) if f.lower().endswith(('.ttl', '.json', '.rdf'))]
        if not kg_files:
            raise HTTPException(status_code=400, detail="No valid KG file found.")
        kg_path = os.path.join(kg_folder, kg_files[0])
        domain_model = kg_to_buml(kg_path=kg_path, openai_token=api_key)
        diagram_json = class_buml_to_json(domain_model)

        diagram_title = diagram_json.get("title", "Imported Class Diagram")
        diagram_type = "ClassDiagram"
        return {
            "title": diagram_title,
            "model": {**diagram_json, "type": diagram_type},
            "diagramType": diagram_type,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION,
        }


@router.post("/import-owl", response_model=DiagramExportResponse)
@handle_endpoint_errors("import_owl")
async def import_owl(owl_file: UploadFile = File(...)):
    """Parse an uploaded OWL/RDF ontology into a KnowledgeGraph diagram JSON.

    Accepts ``.owl``, ``.ttl``, ``.rdf``, ``.xml``, ``.nt``, ``.n3``. Returns
    the JSON shape rendered by the KG editor's Cytoscape canvas (``nodes`` +
    ``edges``) wrapped in the standard diagram-export envelope.
    """
    owl_content = await owl_file.read()
    _validate_upload(
        owl_file,
        max_size=MAX_OWL_SIZE,
        allowed_extensions=ALLOWED_OWL_EXTENSIONS,
        content=owl_content,
    )
    _validate_file_content(owl_content, owl_file.filename or "")

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
        safe_filename = os.path.basename(owl_file.filename or "ontology.owl")
        owl_path = os.path.join(temp_dir, safe_filename)
        await _write_file(owl_path, owl_content)
        # rdflib's parse is CPU-bound and can run for seconds on a large
        # ontology; on the event loop it stalls every other request.
        kg = await asyncio.to_thread(owl_file_to_knowledge_graph, owl_path)
        diagram_json = kg_to_json(kg)

        diagram_title = diagram_json.get("title", "Imported Knowledge Graph")
        diagram_type = "KnowledgeGraphDiagram"
        return {
            "title": diagram_title,
            "model": {**diagram_json["model"], "type": diagram_type},
            "diagramType": diagram_type,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION,
        }


def _kg_payload_to_kg(input_data: DiagramInput):
    """Validate that ``input_data`` carries a KG diagram and parse it.

    Raises:
        HTTPException(400): when the payload isn't a ``KnowledgeGraphDiagram``.
    """
    json_data = input_data.model_dump()
    model = json_data.get("model") or {}
    if model.get("type") != "KnowledgeGraphDiagram":
        raise HTTPException(
            status_code=400,
            detail="Expected a KnowledgeGraphDiagram payload (model.type must be 'KnowledgeGraphDiagram').",
        )
    return process_kg_diagram(json_data), json_data


def _enforce_signature(input_data: DiagramInput, kg) -> None:
    """If the request includes a ``kgSignature``, ensure it matches the current KG."""
    if not input_data.kgSignature:
        return
    actual = kg_signature(kg)
    if actual != input_data.kgSignature:
        raise ConversionError(
            "Knowledge graph changed since analysis; please re-run the preflight before "
            "applying resolutions."
        )


def _apply_v2_decisions(kg, decisions, *, deferred_orphan_node_ids=None):
    """Apply v2-style ``[{issueId, decision}]`` choices.

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


def _apply_v1_resolutions(kg, raw_list):
    """Backward-compat: v1-style ``[{issueId, choice, parameters}]`` payload."""
    parsed = []
    for r in raw_list:
        if not isinstance(r, dict):
            continue
        issue_id = r.get("issueId") or r.get("issue_id") or ""
        choice = r.get("choice") or ""
        parameters = r.get("parameters") or {}
        if not choice:
            raise ConversionError("Each resolution must include a 'choice' key.")
        parsed.append(KGResolution(issue_id=issue_id, choice=choice, parameters=parameters))
    try:
        return apply_resolutions(kg, parsed)
    except ResolutionError as exc:
        raise ConversionError(str(exc)) from exc


def _resolve_kg(kg, input_data: DiagramInput):
    """Apply ``input_data.resolutions`` to ``kg``. Auto-detects the payload shape:
    v2 entries have ``decision``; v1 entries have ``choice``."""
    if not input_data.resolutions:
        return kg
    raw_list = input_data.resolutions
    is_v2 = all(
        isinstance(r, dict) and ("decision" in r and "choice" not in r)
        for r in raw_list
    )
    if is_v2:
        return _apply_v2_decisions(kg, raw_list)
    return _apply_v1_resolutions(kg, raw_list)


def _action_to_dict(action):
    if action is None:
        return None
    return {"key": action.key, "parameters": action.parameters, "label": action.label}


def _report_to_response(report):
    return {
        "kgSignature": report.kg_signature,
        "diagramType": report.diagram_type,
        "issueCount": report.issue_count,
        "issues": [
            {
                "id": i.id,
                "code": i.code,
                "description": i.description,
                "affectedNodeIds": i.affected_node_ids,
                "affectedEdgeIds": i.affected_edge_ids,
                "recommendedAction": _action_to_dict(i.recommended_action),
                "skipAction": _action_to_dict(i.skip_action),
            }
            for i in report.issues
        ],
    }


@router.post("/check-kg-consistency", response_model=KGConsistencyReportResponse)
@handle_endpoint_errors("check_kg_consistency")
async def check_kg_consistency_endpoint(input_data: DiagramInput):
    """Run the OWL2 + SHACL consistency check on the supplied KG.

    Delegates to pyshacl (with OWL2-RL inference) plus a small OWL→SHACL
    shim for class-level axioms. Returns a list of issues with severity
    `info` / `warning` / `violation`, the offending node ids, and the
    originating constraint shape id (when resolvable). Empty issues list
    means the KG is fully conformant.
    """
    # Imported here rather than at module scope so pyshacl + owlrl are not
    # pulled in just to start the backend. (rdflib still is — it is core to
    # every KG path — but pyshacl and its owlrl dependency are not.)
    from besser.BUML.notations.kg_to_buml.consistency import check_kg_consistency

    kg, _ = _kg_payload_to_kg(input_data)
    # pyshacl + owlrl is the heaviest thing this backend runs (seconds on a
    # mid-sized ontology). Off the event loop, as the generators already do.
    report = await asyncio.to_thread(check_kg_consistency, kg)
    return {
        "issues": [_dataclass_asdict(i) for i in report.issues],
        "issueCount": report.issue_count,
        "severityCounts": report.severity_counts,
        "kgSignature": report.kg_signature,
        "inferenceUsed": report.inference_used,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "version": API_VERSION,
    }


@router.post("/analyze-kg-for-buml-conversion", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("analyze_kg_for_buml_conversion")
async def analyze_kg_for_buml_conversion_endpoint(input_data: DiagramInput):
    """Run the KG → BUML preflight and return a structured list of issues.

    Each issue carries a pre-filled ``recommendedAction`` and
    ``skipAction``; the frontend renders one row per issue with a
    checkbox + 2 buttons.
    """
    kg, _json_data = _kg_payload_to_kg(input_data)
    return _report_to_response(analyze_kg_for_class_diagram(kg))


@router.post("/kg-to-class-diagram", response_model=DiagramExportResponse)
@handle_endpoint_errors("kg_to_class_diagram")
async def kg_to_class_diagram_endpoint(input_data: DiagramInput):
    """Transform a Knowledge Graph diagram into a BUML Class Diagram (TBox).

    Optional ``resolutions`` (paired with the ``kgSignature`` from a prior
    ``/analyze-kg-for-buml-conversion`` response) carry the user's
    accept/skip choices for each preflight issue. Without resolutions the
    behaviour is unchanged.
    """
    kg, _json_data = _kg_payload_to_kg(input_data)
    _enforce_signature(input_data, kg)
    base_title = (input_data.title or kg.name or "Knowledge Graph")
    resolved_kg = _resolve_kg(kg, input_data)
    result = kg_to_class_diagram_buml(resolved_kg, model_name=base_title)
    diagram_json = class_buml_to_json(result.domain_model)
    return {
        "title": f"{base_title} (Class Diagram)",
        "model": {**diagram_json, "type": "ClassDiagram"},
        "diagramType": "ClassDiagram",
        "warnings": [w.__dict__ for w in result.warnings],
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "version": API_VERSION,
    }


# ----------------------------------------------------------------------
# LLM-driven KG cleanup
# ----------------------------------------------------------------------


def _issues_payload_to_objects(payload: Any) -> Dict[str, KGIssue]:
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
        recommended = _action_payload_to_object(entry.get("recommendedAction") or entry.get("recommended_action"))
        skip = _action_payload_to_object(entry.get("skipAction") or entry.get("skip_action"))
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


def _action_payload_to_object(action: Any) -> Optional[KGAction]:
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


def _apply_llm_decisions(
    kg,
    issues_payload: Optional[List[Dict[str, Any]]],
    decisions: Optional[List[Dict[str, Any]]],
    *,
    deferred_orphan_node_ids=None,
):
    """Apply round-tripped LLM-issue decisions to a deep-copied KG.

    Unlike :func:`_apply_v2_decisions`, this does NOT re-run the analyzer
    between decisions: re-calling the LLM per decision would be slow,
    non-deterministic, and unnecessary because the issue list is already
    a contract between client and server.

    Symmetrical with :func:`_apply_v2_decisions`: ``deferred_orphan_node_ids``
    enables routing the deferred-LLM action through ``pendingOrphanClassification``.
    """
    if not decisions:
        return kg
    issues_by_id = _issues_payload_to_objects(issues_payload or [])
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


@router.post("/llm-clean-kg", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("llm_clean_kg")
async def llm_clean_kg_endpoint(
    diagram: str = Form(...),
    description: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(DEFAULT_KG_LLM_MODEL),
):
    """Ask GPT-4o for KG cleanup suggestions targeted at a described system.

    The frontend POSTs the active KG diagram (JSON-encoded as the ``diagram``
    field), the natural-language description of the system the user wants
    to build, and an OpenAI API key. The response is a
    :class:`KGPreflightReport`-shaped JSON the frontend renders through the
    existing accept/skip suggestion UI.
    """
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="A non-empty description is required.")
    if not api_key or not api_key.strip():
        raise HTTPException(status_code=400, detail="An OpenAI API key is required.")
    try:
        diagram_payload = json.loads(diagram)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'diagram': {exc}") from exc
    try:
        diagram_input = DiagramInput.model_validate(diagram_payload)
    except Exception as exc:  # pydantic.ValidationError or similar
        raise HTTPException(status_code=400, detail=f"Invalid diagram payload: {exc}") from exc

    kg, _json_data = _kg_payload_to_kg(diagram_input)
    try:
        # Blocking HTTP with a 120 s timeout: on the event loop it freezes the
        # whole backend for the duration (generation_router does the same).
        report = await asyncio.to_thread(
            analyze_kg_with_description, kg, description, api_key, model=model
        )
    except (ValueError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc
    except LLMUpstreamError as exc:
        # Its message is already free of the provider's response body (that is
        # logged, not returned), so it is safe to hand to the client as a 502.
        raise UpstreamServiceError(str(exc)) from exc
    return _report_to_response(report)


# ----------------------------------------------------------------------
# Unified KG refinement (Refine KG modal)
# ----------------------------------------------------------------------


@router.post("/apply-kg-refinement", response_model=KGRefinementResponse)
@handle_endpoint_errors("apply_kg_refinement")
async def apply_kg_refinement_endpoint(input_data: DiagramInput):
    """Apply Refine-KG decisions and return the cleaned KG.

    Single endpoint serving both tabs of the unified Refine KG modal:

    - ``source="static"``: re-runs the static analyzer and dispatches
      each accept/skip decision. Orphan-issue ``skip`` choices
      raise :class:`DeferredOrphanClassification`; the carried node ids
      are accumulated into ``pendingOrphanClassification.nodeIds`` and
      returned to the client so the AI tab can call
      ``/classify-orphans-with-llm`` next.
    - ``source="llm"``: reconstructs LLM-issue objects from
      ``llmIssues`` and dispatches them.

    Returns the cleaned KG diagram so the frontend can replace the
    active KG. The new ``kgSignature`` is also returned for the next
    apply round-trip.
    """
    kg, _json_data = _kg_payload_to_kg(input_data)
    _enforce_signature(input_data, kg)
    base_title = input_data.title or kg.name or "Knowledge Graph"

    source = (input_data.source or "static").lower()
    if source not in {"static", "llm"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source {input_data.source!r}; expected 'static' or 'llm'.",
        )

    deferred_orphan_node_ids: List[str] = []

    if source == "static":
        resolved_kg = _apply_v2_decisions(
            kg,
            input_data.resolutions or [],
            deferred_orphan_node_ids=deferred_orphan_node_ids,
        )
    else:  # source == "llm"
        resolved_kg = _apply_llm_decisions(
            kg,
            input_data.llmIssues,
            input_data.resolutions,
            deferred_orphan_node_ids=deferred_orphan_node_ids,
        )

    cleaned_json = kg_to_json(resolved_kg)
    cleaned_model = cleaned_json.get("model", cleaned_json)
    if "type" not in cleaned_model:
        cleaned_model = {**cleaned_model, "type": "KnowledgeGraphDiagram"}

    new_signature = kg_signature(resolved_kg)
    pending_orphan_payload = None
    if deferred_orphan_node_ids:
        pending_orphan_payload = {
            "nodeIds": deferred_orphan_node_ids,
            "kgSignature": new_signature,
        }

    return {
        "title": f"{base_title} (Refined)",
        "model": cleaned_model,
        "diagramType": "KnowledgeGraphDiagram",
        "kgSignature": new_signature,
        "pendingOrphanClassification": pending_orphan_payload,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "version": API_VERSION,
    }


@router.post("/classify-orphans-with-llm", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("classify_orphans_with_llm")
async def classify_orphans_with_llm_endpoint(
    diagram: str = Form(...),
    description: str = Form(...),
    api_key: str = Form(...),
    node_ids: str = Form(...),
    model: str = Form(DEFAULT_KG_LLM_MODEL),
):
    """Per-node LLM classification for a batch of orphan node ids.

    Mirrors ``/llm-clean-kg`` but takes an explicit ``node_ids`` list and
    runs a focused per-node prompt instead of a full-graph snapshot. The
    response has the same ``KGPreflightReport`` shape so the AI tab can
    render it through the existing accept/skip UI.
    """
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="A non-empty description is required.")
    if not api_key or not api_key.strip():
        raise HTTPException(status_code=400, detail="An OpenAI API key is required.")
    try:
        diagram_payload = json.loads(diagram)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'diagram': {exc}") from exc
    try:
        diagram_input = DiagramInput.model_validate(diagram_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid diagram payload: {exc}") from exc
    try:
        parsed_node_ids = json.loads(node_ids)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'node_ids': {exc}") from exc
    if not isinstance(parsed_node_ids, list):
        raise HTTPException(status_code=400, detail="'node_ids' must be a JSON array of strings.")
    if not all(isinstance(x, str) for x in parsed_node_ids):
        raise HTTPException(status_code=400, detail="'node_ids' must contain only strings.")
    if not parsed_node_ids:
        raise HTTPException(status_code=400, detail="'node_ids' must not be empty.")
    if len(parsed_node_ids) > MAX_ORPHAN_CLASSIFY_NODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'node_ids' carries {len(parsed_node_ids)} nodes; at most "
                f"{MAX_ORPHAN_CLASSIFY_NODES} can be classified per request. "
                "Classify them in smaller batches."
            ),
        )

    kg, _json_data = _kg_payload_to_kg(diagram_input)
    _enforce_signature(diagram_input, kg)
    try:
        # One blocking OpenAI request per 40 nodes, sequentially; the cap above
        # bounds the chain and to_thread keeps it off the event loop.
        report = await asyncio.to_thread(
            classify_orphan_nodes_with_llm,
            kg,
            parsed_node_ids,
            description,
            api_key,
            model=model,
        )
    except (ValueError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc
    except LLMUpstreamError as exc:
        # Its message is already free of the provider's response body (that is
        # logged, not returned), so it is safe to hand to the client as a 502.
        raise UpstreamServiceError(str(exc)) from exc
    return _report_to_response(report)


_RDF_FORMATS = {
    "owl": ("xml", "application/rdf+xml", ".owl"),
    "ttl": ("turtle", "text/turtle", ".ttl"),
}


@router.post("/export-kg-rdf/{fmt}")
@handle_endpoint_errors("export_kg_rdf")
async def export_kg_rdf(fmt: str, input_data: DiagramInput = Body(...)):
    """Serialize a Knowledge Graph diagram as OWL (RDF/XML) or Turtle.

    The path parameter ``fmt`` selects the serialization: ``owl`` for RDF/XML,
    ``ttl`` for Turtle. The optional ``vocab`` field on the body selects which
    constraint vocabularies (OWL restrictions vs SHACL shapes vs both) to emit;
    defaults to ``both``.
    """
    if fmt not in _RDF_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Use 'owl' or 'ttl'.",
        )
    rdflib_fmt, media_type, ext = _RDF_FORMATS[fmt]

    vocab_choice = (getattr(input_data, "vocab", None) or "both").lower()
    if vocab_choice not in ("owl", "shacl", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported vocab '{vocab_choice}'. Use 'owl', 'shacl', or 'both'.",
        )

    kg, _json_data = _kg_payload_to_kg(input_data)
    serialized = serialize_knowledge_graph(kg, fmt=rdflib_fmt, vocab=vocab_choice)

    base = input_data.title or kg.name or "knowledge_graph"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in base).strip("_") or "knowledge_graph"
    filename = f"{safe}{ext}"

    return Response(
        content=serialized,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
