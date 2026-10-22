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

from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, File, UploadFile, Body, Form
from fastapi.responses import Response
from pydantic import SecretStr

# Backend response models
from besser.utilities.web_modeling_editor.backend.models.responses import (
    DiagramExportResponse,
    KGConfigResponse,
    KGConsistencyReportResponse,
    KGPreflightReportResponse,
    KGRefinementResponse,
)

# BESSER image-to-UML and BUML utilities
from besser.utilities.kg_to_buml import kg_to_buml
from besser.BUML.notations.kg_rdf import owl_file_to_knowledge_graph, serialize_knowledge_graph

# BESSER KG → B-UML (deterministic, offline)
from besser.BUML.notations.kg_to_buml import (
    analyze_kg_for_class_diagram,
    analyze_kg_with_description,
    classify_orphan_nodes_with_llm,
    kg_signature,
    kg_to_class_diagram as kg_to_class_diagram_buml,
)
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
    KGDiagramInput,
    KGLLMRequest,
    KGOrphanClassifyRequest,
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    # JSON to BUML converters
    class_buml_to_json,
    kg_to_json,
)

# Backend services - Other services
from besser.utilities.web_modeling_editor.backend.services.kg.decisions import (
    apply_llm_decisions,
    apply_v2_decisions,
    resolve_kg,
)
from besser.utilities.web_modeling_editor.backend.services.kg.llm import (
    RequestDeadline,
    build_completer,
    run_with_deadline,
)
from besser.utilities.web_modeling_editor.backend.services.kg.payloads import (
    consistency_report_to_response,
    enforce_signature,
    kg_payload_to_kg,
    report_to_response,
)

# Backend configuration

# Backend constants
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    API_VERSION,
    TEMP_DIR_PREFIX,
    MAX_OWL_SIZE,
    KG_CONSISTENCY_TIMEOUT_SECONDS,
    MAX_BUML_SIZE,
    MAX_ORPHAN_CLASSIFY_NODES,
    KG_LLM_REQUEST_TIMEOUT_SECONDS,
    ALLOWED_KG_EXTENSIONS,
    ALLOWED_OWL_EXTENSIONS,
)

# Upload validation and async file I/O, shared with conversion_router
from besser.utilities.web_modeling_editor.backend.services.utils.upload_validation import (
    validate_file_content,
    validate_upload,
    write_file_async,
)

# Centralized error handling
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)

logger = logging.getLogger(__name__)

# Same OpenAPI tag as before the split, so the published API schema is unchanged.
router = APIRouter(prefix="/besser_api", tags=["conversion"])


@router.get("/kg-config", response_model=KGConfigResponse)
@handle_endpoint_errors("get_kg_config")
async def get_kg_config() -> KGConfigResponse:
    """Server-owned limits the KG editor must respect.

    ``llmRequestTimeoutSeconds`` is the overall deadline of one KG LLM request;
    ``maxOrphanClassifyNodes`` is the most node ids one
    ``/classify-orphans-with-llm`` request may carry (longer lists are rejected
    with HTTP 400, so the editor splits them).
    """
    return KGConfigResponse(
        llmRequestTimeoutSeconds=KG_LLM_REQUEST_TIMEOUT_SECONDS,
        maxOrphanClassifyNodes=MAX_ORPHAN_CLASSIFY_NODES,
    )


@router.post("/get-json-model-from-kg", response_model=DiagramExportResponse)
@handle_endpoint_errors("get_json_model_from_kg")
async def get_json_model_from_kg(
    kg_file: UploadFile = File(...),
    api_key: SecretStr = Form(...),
):
    """
    Accepts an .TTL, .RDF, or .JSON knowledge graphs and an OpenAI API key, uses BESSER's kgtouml feature to transform the kg into a BUML class diagram in JSON.
    """
    kg_content = await kg_file.read()
    validate_upload(kg_file, max_size=MAX_BUML_SIZE, allowed_extensions=ALLOWED_KG_EXTENSIONS, content=kg_content)
    openai_token = api_key.get_secret_value()
    if not openai_token.strip():
        raise ValidationError("An API key is required.")

    deadline = RequestDeadline.start(KG_LLM_REQUEST_TIMEOUT_SECONDS)
    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
        # The extension was validated above; the name itself is never trusted.
        _, ext = os.path.splitext(kg_file.filename or "")
        kg_path = os.path.join(temp_dir, f"knowledge_graph{ext.lower()}")
        await write_file_async(kg_path, kg_content)
        try:
            # One blocking OpenAI call (up to the deadline) plus the conversion.
            domain_model = await run_with_deadline(
                deadline, kg_to_buml, kg_path, openai_token, timeout=deadline.remaining(),
            )
        except LLMUpstreamError as exc:
            # Its message never carries the provider's response body.
            raise UpstreamServiceError(str(exc)) from exc
        except RuntimeError as exc:
            # The model's answer did not make a class diagram; the details (which
            # can quote that answer) stay in the server log.
            logger.warning("KG -> B-UML conversion of the LLM answer failed: %s", exc)
            raise ConversionError(
                "The LLM's answer could not be turned into a class diagram. Please try again."
            ) from exc
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
    validate_upload(
        owl_file,
        max_size=MAX_OWL_SIZE,
        allowed_extensions=ALLOWED_OWL_EXTENSIONS,
        content=owl_content,
    )
    validate_file_content(owl_content, owl_file.filename or "")

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
        safe_filename = os.path.basename(owl_file.filename or "ontology.owl")
        owl_path = os.path.join(temp_dir, safe_filename)
        await write_file_async(owl_path, owl_content)
        # rdflib's parse is CPU-bound and can run for seconds on a large
        # ontology; on the event loop it stalls every other request.
        import_warnings: List[Dict[str, str]] = []
        kg = await asyncio.to_thread(owl_file_to_knowledge_graph, owl_path, import_warnings)
        diagram_json = kg_to_json(kg)

        diagram_title = diagram_json.get("title", "Imported Knowledge Graph")
        diagram_type = "KnowledgeGraphDiagram"
        return {
            "title": diagram_title,
            "model": {**diagram_json["model"], "type": diagram_type},
            "diagramType": diagram_type,
            "warnings": import_warnings,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "version": API_VERSION,
        }




@router.post("/check-kg-consistency", response_model=KGConsistencyReportResponse)
@handle_endpoint_errors("check_kg_consistency")
async def check_kg_consistency_endpoint(input_data: KGDiagramInput):
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
    from besser.BUML.notations.kg_to_buml.consistency import (
        ConsistencyCheckError,
        check_kg_consistency,
    )

    kg, _ = kg_payload_to_kg(input_data)
    # pyshacl + owlrl is the heaviest thing this backend runs (seconds on a
    # mid-sized ontology). It runs in a child process that is killed after the
    # timeout; waiting for it stays off the event loop.
    try:
        report = await asyncio.to_thread(
            check_kg_consistency, kg, timeout_seconds=KG_CONSISTENCY_TIMEOUT_SECONDS,
        )
    except ConsistencyCheckError as exc:
        # The KG itself is the cause (unloadable shapes, a pattern that runs away).
        raise ValidationError(str(exc)) from exc
    return {
        **consistency_report_to_response(report),
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "version": API_VERSION,
    }


@router.post("/analyze-kg-for-buml-conversion", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("analyze_kg_for_buml_conversion")
async def analyze_kg_for_buml_conversion_endpoint(input_data: KGDiagramInput):
    """Run the KG → BUML preflight and return a structured list of issues.

    Each issue carries a pre-filled ``recommendedAction`` and
    ``skipAction``; the frontend renders one row per issue with a
    checkbox + 2 buttons.
    """
    kg, _json_data = kg_payload_to_kg(input_data)
    report = await asyncio.to_thread(analyze_kg_for_class_diagram, kg)
    return report_to_response(report)


@router.post("/kg-to-class-diagram", response_model=DiagramExportResponse)
@handle_endpoint_errors("kg_to_class_diagram")
async def kg_to_class_diagram_endpoint(input_data: KGDiagramInput):
    """Transform a Knowledge Graph diagram into a BUML Class Diagram (TBox).

    Optional ``resolutions`` (paired with the ``kgSignature`` from a prior
    ``/analyze-kg-for-buml-conversion`` response) carry the user's
    accept/skip choices for each preflight issue. Without resolutions the
    behaviour is unchanged.
    """
    kg, _json_data = kg_payload_to_kg(input_data)
    enforce_signature(input_data, kg)
    base_title = (input_data.title or kg.name or "Knowledge Graph")

    def lower():
        # Re-runs the preflight per decision, then lowers: CPU-bound, so off the loop.
        resolved_kg = resolve_kg(kg, input_data)
        result = kg_to_class_diagram_buml(resolved_kg, model_name=base_title)
        return result, class_buml_to_json(result.domain_model)

    result, diagram_json = await asyncio.to_thread(lower)
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


@router.post("/llm-clean-kg", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("llm_clean_kg")
async def llm_clean_kg_endpoint(request: KGLLMRequest):
    """Ask the user's LLM for KG cleanup suggestions.

    The JSON body carries the active KG diagram, the user's own key and its
    provider (Anthropic, OpenAI, Mistral or Nebius), optionally a model, and
    optionally a natural-language description of the system the user wants to
    build (when omitted the model infers the domain from the KG). The response
    is a :class:`KGPreflightReport`-shaped JSON the frontend renders through the
    existing accept/skip suggestion UI.
    """
    kg, _json_data = kg_payload_to_kg(request.diagram)
    deadline = RequestDeadline.start(KG_LLM_REQUEST_TIMEOUT_SECONDS)
    try:
        complete = build_completer(request, deadline)
        # Blocking HTTP, so off the event loop, and bounded by the request deadline.
        report = await run_with_deadline(
            deadline, analyze_kg_with_description, kg, request.description, complete=complete,
        )
    except (ValueError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc
    except LLMUpstreamError as exc:
        # Its message is already free of the provider's response body (that is
        # logged, not returned), so it is safe to hand to the client as a 502.
        raise UpstreamServiceError(str(exc)) from exc
    return report_to_response(report)


# ----------------------------------------------------------------------
# Unified KG refinement (Refine KG modal)
# ----------------------------------------------------------------------


@router.post("/apply-kg-refinement", response_model=KGRefinementResponse)
@handle_endpoint_errors("apply_kg_refinement")
async def apply_kg_refinement_endpoint(input_data: KGDiagramInput):
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
    kg, _json_data = kg_payload_to_kg(input_data)
    enforce_signature(input_data, kg)
    base_title = input_data.title or kg.name or "Knowledge Graph"

    # ``source`` is a Literal on the model, so it is already "static" or "llm".
    source = input_data.source or "static"

    deferred_orphan_node_ids: List[str] = []

    # Both appliers are CPU-bound (the static one re-runs the preflight after
    # every decision), so they run off the event loop.
    if source == "static":
        resolved_kg = await asyncio.to_thread(
            apply_v2_decisions,
            kg,
            input_data.resolutions or [],
            deferred_orphan_node_ids=deferred_orphan_node_ids,
        )
    else:  # source == "llm"
        resolved_kg = await asyncio.to_thread(
            apply_llm_decisions,
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
async def classify_orphans_with_llm_endpoint(request: KGOrphanClassifyRequest):
    """Per-node LLM classification for a batch of orphan node ids.

    Mirrors ``/llm-clean-kg`` but takes an explicit ``node_ids`` list and
    runs a focused per-node prompt instead of a full-graph snapshot. The
    response has the same ``KGPreflightReport`` shape so the AI tab can
    render it through the existing accept/skip UI.
    """
    node_ids = request.node_ids
    if not node_ids:
        raise ValidationError("'node_ids' must not be empty.")
    if len(node_ids) > MAX_ORPHAN_CLASSIFY_NODES:
        raise ValidationError(
            f"'node_ids' carries {len(node_ids)} nodes; at most "
            f"{MAX_ORPHAN_CLASSIFY_NODES} can be classified per request. "
            "Classify them in smaller batches."
        )

    kg, _json_data = kg_payload_to_kg(request.diagram)
    enforce_signature(request.diagram, kg)
    deadline = RequestDeadline.start(KG_LLM_REQUEST_TIMEOUT_SECONDS)
    try:
        complete = build_completer(request, deadline)
        # One blocking LLM request per 40 nodes, sequentially; the cap above
        # bounds the chain, and the deadline bounds the request as a whole.
        report = await run_with_deadline(
            deadline,
            classify_orphan_nodes_with_llm,
            kg,
            node_ids,
            request.description,
            complete=complete,
        )
    except (ValueError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc
    except LLMUpstreamError as exc:
        # Its message is already free of the provider's response body (that is
        # logged, not returned), so it is safe to hand to the client as a 502.
        raise UpstreamServiceError(str(exc)) from exc
    return report_to_response(report)


_RDF_FORMATS = {
    "owl": ("xml", "application/rdf+xml", ".owl"),
    "ttl": ("turtle", "text/turtle", ".ttl"),
}


@router.post("/export-kg-rdf/{fmt}")
@handle_endpoint_errors("export_kg_rdf")
async def export_kg_rdf(fmt: str, input_data: KGDiagramInput = Body(...)):
    """Serialize a Knowledge Graph diagram as OWL (RDF/XML) or Turtle.

    The path parameter ``fmt`` selects the serialization: ``owl`` for RDF/XML,
    ``ttl`` for Turtle. The optional ``vocab`` field on the body selects which
    constraint vocabularies (OWL restrictions vs SHACL shapes vs both) to emit;
    defaults to ``both``.
    """
    if fmt not in _RDF_FORMATS:
        raise ValidationError(f"Unsupported format '{fmt}'. Use 'owl' or 'ttl'.")
    rdflib_fmt, media_type, ext = _RDF_FORMATS[fmt]

    # ``vocab`` is a Literal on the model: "owl", "shacl" or "both".
    vocab_choice = input_data.vocab or "both"

    kg, _json_data = kg_payload_to_kg(input_data)
    serialized = serialize_knowledge_graph(kg, fmt=rdflib_fmt, vocab=vocab_choice)

    base = input_data.title or kg.name or "knowledge_graph"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in base).strip("_") or "knowledge_graph"
    filename = f"{safe}{ext}"

    return Response(
        content=serialized,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
