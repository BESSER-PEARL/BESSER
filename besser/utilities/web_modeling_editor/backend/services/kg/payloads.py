"""Request/response plumbing shared by the Knowledge Graph endpoints.

Parsing a KG diagram out of a request, checking the ``kgSignature`` the client
echoed, and shaping a preflight report into its response body. Kept out of the
router so ``routers/kg_router.py`` holds only endpoints (the same split as
``services/spec_driven``).
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from besser.BUML.metamodel.kg import KnowledgeGraph
from besser.BUML.notations.kg_to_buml import KGAction, KGPreflightReport, kg_signature
from besser.utilities.web_modeling_editor.backend.constants.constants import KG_DIAGRAM_TYPE
from besser.utilities.web_modeling_editor.backend.models import DiagramInput, KGDiagramInput
from besser.utilities.web_modeling_editor.backend.services.converters import process_kg_diagram
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConversionError,
    ValidationError,
)

if TYPE_CHECKING:
    from besser.BUML.notations.kg_to_buml.consistency import ConsistencyReport


def parse_kg_diagram(json_data: Dict[str, Any]) -> KnowledgeGraph:
    """Build the :class:`KnowledgeGraph` of a dumped KG diagram payload.

    Raises:
        ConversionError: the payload is malformed, e.g. a node or edge that is not
            an object (HTTP 400; the converter itself would fail with a bare
            ``AttributeError`` and a generic 500). Same wrapping as ``export_buml``.
    """
    try:
        return process_kg_diagram(json_data)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ConversionError(f"Malformed Knowledge Graph diagram payload: {exc}") from exc


def kg_payload_to_kg(input_data: DiagramInput) -> KnowledgeGraph:
    """Validate that ``input_data`` carries a KG diagram and parse it.

    Raises:
        ValidationError: when the payload isn't a ``KnowledgeGraphDiagram`` (HTTP 400).
        ConversionError: when the KG diagram itself is malformed (HTTP 400).
    """
    json_data = input_data.model_dump()
    model = json_data.get("model") or {}
    if model.get("type") != KG_DIAGRAM_TYPE:
        raise ValidationError(
            f"Expected a {KG_DIAGRAM_TYPE} payload (model.type must be '{KG_DIAGRAM_TYPE}')."
        )
    return parse_kg_diagram(json_data)


def enforce_signature(input_data: KGDiagramInput, kg: KnowledgeGraph) -> None:
    """If the request includes a ``kgSignature``, ensure it matches the current KG."""
    if not input_data.kgSignature:
        return
    actual = kg_signature(kg)
    if actual != input_data.kgSignature:
        raise ConversionError(
            "Knowledge graph changed since analysis; please re-run the preflight before "
            "applying resolutions."
        )


def action_to_dict(action: Optional[KGAction]) -> Optional[Dict[str, Any]]:
    """The ``KGActionResponse`` body of ``action`` (``None`` stays ``None``)."""
    if action is None:
        return None
    return {"key": action.key, "parameters": action.parameters, "label": action.label}


def report_to_response(report: KGPreflightReport) -> Dict[str, Any]:
    """Shape a ``KGPreflightReport`` into the ``KGPreflightReportResponse`` body."""
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
                "recommendedAction": action_to_dict(i.recommended_action),
                "skipAction": action_to_dict(i.skip_action),
            }
            for i in report.issues
        ],
    }


def consistency_report_to_response(report: "ConsistencyReport") -> Dict[str, Any]:
    """Shape a ``ConsistencyReport`` into the ``KGConsistencyReportResponse`` body.

    The dataclass keeps Python's snake_case; the wire format is camelCase.
    """
    return {
        "issues": [
            {
                "id": i.id,
                "code": i.code,
                "severity": i.severity,
                "message": i.message,
                "affectedNodeIds": i.affected_node_ids,
                "affectedEdgeIds": i.affected_edge_ids,
                "constraintNodeId": i.constraint_node_id,
                "specKind": i.spec_kind,
                "pathIri": i.path_iri,
                "constraintLabel": i.constraint_label,
            }
            for i in report.issues
        ],
        "issueCount": report.issue_count,
        "severityCounts": report.severity_counts,
        "kgSignature": report.kg_signature,
        "inferenceUsed": report.inference_used,
    }
