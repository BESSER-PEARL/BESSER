"""
Knowledge Graph Router

The Knowledge Graph (KG) endpoints of the BESSER web modeling editor backend,
all under ``/besser_api/kg``: ontology import and RDF export, the OWL/SHACL
consistency check, the KG -> class diagram conversion with its preflight, and
the Refine-KG flow with its LLM-assisted cleanup.

``legacy_router`` keeps the one KG endpoint that was released before this
prefix existed, ``POST /besser_api/get-json-model-from-kg``, at its path; it is
deprecated.

The generic B-UML import/export dispatch in ``conversion_router`` still
recognises a KG file; only the KG-specific endpoints live here, and their logic
lives in ``services/kg``.
"""

import asyncio
import logging
import os
import tempfile

from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile, Body, Form
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

# BESSER KG libraries: OWL/RDF import and export, the LLM conversion
from besser.utilities.kg_llm_to_buml import LLMAnswerError, kg_to_buml
from besser.BUML.notations.kg_rdf import owl_file_to_knowledge_graph, serialize_knowledge_graph

# BESSER KG → B-UML (deterministic, offline)
from besser.BUML.notations.kg_to_buml import (
    KGPreflightReport,
    analyze_kg_for_class_diagram,
    analyze_kg_with_description,
    classify_orphan_nodes_with_llm,
    kg_signature,
    kg_to_class_diagram as kg_to_class_diagram_buml,
)
from besser.BUML.notations.kg_to_buml.consistency import ConsistencyCheckError
from besser.utilities.llm_utils import LLMUpstreamError

# Custom exception types translated to HTTP codes by @handle_endpoint_errors.
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    UpstreamError,
    ValidationError,
)

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    KGDiagramInput,
    KGLLMRequest,
    KGOrphanClassifyRequest,
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    class_buml_to_json,
    kg_to_json,
)

# Backend services - KG
from besser.utilities.web_modeling_editor.backend.services.kg.decisions import (
    apply_decisions,
    apply_llm_decisions,
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
from besser.utilities.web_modeling_editor.backend.services.kg.validation import (
    ConsistencyCheckBusy,
    run_consistency_check,
)

# Backend constants
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    API_VERSION,
    TEMP_DIR_PREFIX,
    KG_DIAGRAM_TYPE,
    MAX_OWL_SIZE,
    MAX_KG_SIZE,
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

router = APIRouter(prefix="/besser_api/kg", tags=["knowledge-graph"])
# Released before the /kg prefix existed; kept at its path, deprecated.
legacy_router = APIRouter(prefix="/besser_api", tags=["knowledge-graph"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _llm_report(call: Awaitable[KGPreflightReport]) -> Dict[str, Any]:
    """Await one LLM-backed analysis and shape its report, mapping its failures.

    A ``ValueError`` / ``TypeError`` is the request's fault (HTTP 400); an
    ``LLMUpstreamError`` carries a message that is already free of the
    provider's response body (that is logged, not returned), so it is safe to
    hand to the client as a 502.
    """
    try:
        report = await call
    except (ValueError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc
    except LLMUpstreamError as exc:
        raise UpstreamError(str(exc)) from exc
    return report_to_response(report)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@router.get("/config", response_model=KGConfigResponse)
async def get_kg_config() -> KGConfigResponse:
    """Server-owned limits the KG editor must respect.

    ``llmRequestTimeoutSeconds`` is the overall deadline of one KG LLM request;
    ``maxOrphanClassifyNodes`` is the most node ids one ``/kg/classify-orphans``
    request may carry (longer lists are rejected with HTTP 400, so the editor
    splits them).
    """
    return KGConfigResponse(
        llmRequestTimeoutSeconds=KG_LLM_REQUEST_TIMEOUT_SECONDS,
        maxOrphanClassifyNodes=MAX_ORPHAN_CLASSIFY_NODES,
    )


# ----------------------------------------------------------------------
# Ontology import / export
# ----------------------------------------------------------------------


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

    return {
        "title": diagram_json.get("title", "Imported Knowledge Graph"),
        "model": {**diagram_json["model"], "type": KG_DIAGRAM_TYPE},
        "diagramType": KG_DIAGRAM_TYPE,
        "warnings": import_warnings,
        "exportedAt": _now(),
        "version": API_VERSION,
    }


_RDF_FORMATS = {
    "owl": ("xml", "application/rdf+xml", ".owl"),
    "ttl": ("turtle", "text/turtle", ".ttl"),
}


@router.post("/export/{fmt}")
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

    kg = kg_payload_to_kg(input_data)
    # rdflib serialisation is CPU-bound on a large graph: off the event loop.
    serialized = await asyncio.to_thread(
        serialize_knowledge_graph, kg, fmt=rdflib_fmt, vocab=input_data.vocab or "both",
    )

    base = input_data.title or kg.name or "knowledge_graph"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in base).strip("_") or "knowledge_graph"
    return Response(
        content=serialized,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{safe}{ext}"'},
    )


# ----------------------------------------------------------------------
# Consistency check, preflight, conversion to a class diagram
# ----------------------------------------------------------------------


@router.post("/consistency", response_model=KGConsistencyReportResponse)
@handle_endpoint_errors("check_kg_consistency")
async def check_kg_consistency_endpoint(input_data: KGDiagramInput):
    """Run the OWL2 + SHACL consistency check on the supplied KG.

    Delegates to pyshacl (with OWL2-RL inference) plus a small OWL→SHACL
    shim for class-level axioms. Returns a list of issues with severity
    ``info`` / ``warning`` / ``violation``, the offending node ids, and the
    originating constraint shape id (when resolvable). An empty issues list
    means the KG is fully conformant. HTTP 429 when too many checks are
    already running, 400 when the KG cannot be checked (unloadable shapes,
    out of time).
    """
    kg = kg_payload_to_kg(input_data)
    try:
        report = await run_consistency_check(kg)
    except ConsistencyCheckBusy as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ConsistencyCheckError as exc:
        # The KG itself is the cause (unloadable shapes, a pattern that runs away).
        raise ValidationError(str(exc)) from exc
    return {
        **consistency_report_to_response(report),
        "exportedAt": _now(),
        "version": API_VERSION,
    }


@router.post("/preflight", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("kg_preflight")
async def kg_preflight_endpoint(input_data: KGDiagramInput):
    """Run the KG → B-UML preflight and return a structured list of issues.

    Each issue carries a pre-filled ``recommendedAction`` and
    ``skipAction``; the frontend renders one row per issue with a
    checkbox + 2 buttons.
    """
    kg = kg_payload_to_kg(input_data)
    report = await asyncio.to_thread(analyze_kg_for_class_diagram, kg)
    return report_to_response(report)


@router.post("/to-class-diagram", response_model=DiagramExportResponse)
@handle_endpoint_errors("kg_to_class_diagram")
async def kg_to_class_diagram_endpoint(input_data: KGDiagramInput):
    """Transform a Knowledge Graph diagram into a B-UML Class Diagram (TBox).

    Optional ``resolutions`` (paired with the ``kgSignature`` from a prior
    ``/kg/preflight`` response) carry the user's accept/skip choices for each
    preflight issue. Without resolutions the KG is converted as it is.
    """
    kg = kg_payload_to_kg(input_data)
    enforce_signature(input_data, kg)
    base_title = input_data.title or kg.name or "Knowledge Graph"

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
        "exportedAt": _now(),
        "version": API_VERSION,
    }


# ----------------------------------------------------------------------
# Refine KG: LLM-driven cleanup and applying the user's decisions
# ----------------------------------------------------------------------


@router.post("/llm-clean", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("kg_llm_clean")
async def kg_llm_clean_endpoint(request: KGLLMRequest):
    """Ask the user's LLM for KG cleanup suggestions.

    The JSON body carries the active KG diagram, the user's own key and its
    provider (Anthropic, OpenAI, Mistral or Nebius), optionally a model, and
    optionally a natural-language description of the system the user wants to
    build (when omitted the model infers the domain from the KG). The response
    is a :class:`KGPreflightReport`-shaped JSON the frontend renders through the
    existing accept/skip suggestion UI.
    """
    kg = kg_payload_to_kg(request.diagram)
    deadline = RequestDeadline.start(KG_LLM_REQUEST_TIMEOUT_SECONDS)
    complete = build_completer(request, deadline)
    # Blocking HTTP, so off the event loop, and bounded by the request deadline.
    return await _llm_report(run_with_deadline(
        deadline, analyze_kg_with_description, kg, request.description, complete=complete,
    ))


@router.post("/classify-orphans", response_model=KGPreflightReportResponse)
@handle_endpoint_errors("kg_classify_orphans")
async def kg_classify_orphans_endpoint(request: KGOrphanClassifyRequest):
    """Per-node LLM classification for a batch of orphan node ids.

    Mirrors ``/kg/llm-clean`` but takes an explicit ``nodeIds`` list and runs
    a focused per-node prompt instead of a full-graph snapshot. The response
    has the same ``KGPreflightReport`` shape so the AI tab can render it
    through the existing accept/skip UI.
    """
    node_ids = request.nodeIds
    if not node_ids:
        raise ValidationError("'nodeIds' must not be empty.")
    if len(node_ids) > MAX_ORPHAN_CLASSIFY_NODES:
        raise ValidationError(
            f"'nodeIds' carries {len(node_ids)} nodes; at most "
            f"{MAX_ORPHAN_CLASSIFY_NODES} can be classified per request. "
            "Classify them in smaller batches."
        )

    kg = kg_payload_to_kg(request.diagram)
    enforce_signature(request.diagram, kg)
    deadline = RequestDeadline.start(KG_LLM_REQUEST_TIMEOUT_SECONDS)
    complete = build_completer(request, deadline)
    # One blocking LLM request per 40 nodes, sequentially; the cap above
    # bounds the chain, and the deadline bounds the request as a whole.
    return await _llm_report(run_with_deadline(
        deadline, classify_orphan_nodes_with_llm, kg, node_ids, request.description, complete=complete,
    ))


@router.post("/apply-refinement", response_model=KGRefinementResponse)
@handle_endpoint_errors("kg_apply_refinement")
async def kg_apply_refinement_endpoint(input_data: KGDiagramInput):
    """Apply Refine-KG decisions and return the cleaned KG.

    Single endpoint serving both tabs of the unified Refine KG modal:

    - ``source="static"`` (the default): re-runs the preflight and applies
      each accept/skip decision. Orphan-issue ``skip`` choices are deferred to
      the LLM: their node ids are returned in
      ``pendingOrphanClassification.nodeIds`` so the AI tab can call
      ``/kg/classify-orphans`` next.
    - ``source="llm"``: applies the decisions to the round-tripped
      ``llmIssues``.

    Returns the cleaned KG diagram so the frontend can replace the active KG,
    with its new ``kgSignature`` for the next round-trip.
    """
    kg = kg_payload_to_kg(input_data)
    enforce_signature(input_data, kg)
    base_title = input_data.title or kg.name or "Knowledge Graph"

    deferred_orphan_node_ids: List[str] = []
    # Both appliers are CPU-bound (the static one re-runs the preflight after
    # every decision), so they run off the event loop.
    if input_data.source == "llm":
        resolved_kg = await asyncio.to_thread(
            apply_llm_decisions,
            kg,
            input_data.llmIssues,
            input_data.resolutions,
            deferred_orphan_node_ids=deferred_orphan_node_ids,
        )
    else:
        resolved_kg = await asyncio.to_thread(
            apply_decisions,
            kg,
            input_data.resolutions or [],
            deferred_orphan_node_ids=deferred_orphan_node_ids,
        )

    new_signature = kg_signature(resolved_kg)
    return {
        "title": f"{base_title} (Refined)",
        "model": {**kg_to_json(resolved_kg)["model"], "type": KG_DIAGRAM_TYPE},
        "diagramType": KG_DIAGRAM_TYPE,
        "kgSignature": new_signature,
        "pendingOrphanClassification": (
            {"nodeIds": deferred_orphan_node_ids, "kgSignature": new_signature}
            if deferred_orphan_node_ids else None
        ),
        "exportedAt": _now(),
        "version": API_VERSION,
    }


# ----------------------------------------------------------------------
# Deprecated: LLM conversion of an uploaded KG file
# ----------------------------------------------------------------------


@legacy_router.post("/get-json-model-from-kg", response_model=DiagramExportResponse, deprecated=True)
@handle_endpoint_errors("get_json_model_from_kg")
async def get_json_model_from_kg(
    kg_file: UploadFile = File(...),
    api_key: SecretStr = Form(...),
):
    """Ask OpenAI to turn an uploaded knowledge graph into a class diagram (deprecated).

    Accepts a ``.ttl``, ``.rdf`` or Neo4j ``.json`` export and an OpenAI API key,
    and returns the class diagram :func:`besser.utilities.kg_llm_to_buml.kg_to_buml`
    proposes. Deprecated: the editor imports an ontology with
    ``POST /kg/import-owl`` and converts it, deterministically and without a
    key, with ``POST /kg/to-class-diagram``.
    """
    kg_content = await kg_file.read()
    validate_upload(kg_file, max_size=MAX_KG_SIZE, allowed_extensions=ALLOWED_KG_EXTENSIONS, content=kg_content)
    try:
        # kg_to_buml reads the file as UTF-8; anything else is the upload's fault.
        kg_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("The knowledge graph file is not valid UTF-8 text.") from exc
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
            raise UpstreamError(str(exc)) from exc
        except LLMAnswerError as exc:
            # The model's answer did not make a class diagram; the details (which
            # can quote that answer) stay in the server log.
            logger.warning("KG -> B-UML conversion of the LLM answer failed: %s", exc)
            raise ConversionError(
                "The LLM's answer could not be turned into a class diagram. Please try again."
            ) from exc
    diagram_json = class_buml_to_json(domain_model)

    return {
        "title": diagram_json.get("title", "Imported Class Diagram"),
        "model": {**diagram_json, "type": "ClassDiagram"},
        "diagramType": "ClassDiagram",
        "exportedAt": _now(),
        "version": API_VERSION,
    }
