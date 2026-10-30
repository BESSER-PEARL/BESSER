"""
TraceModel -> trace file JSON (the ``trace`` of a project export).

The frontend type of this shape is ``TraceFile`` in the webapp's
``shared/types/traceability.ts``; ``json_to_buml/trace_processor.py`` reads it
back. ``parse_trace_code`` loads the Python form of a trace (written by
``buml_code_builder/trace_model_builder.py``, the trace section of a project
export). Element references are ids, timestamps are ISO 8601 UTC with a ``Z``
suffix (what the browser's ``Date.toISOString()`` writes), and optional fields
are left out when unset.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import besser.BUML.metamodel.traceability.traceability as traceability_module
from besser.BUML.metamodel.traceability import (
    AIEvidence, Artifact, Change, ChangeEvidence, CodeArtifact, Confidence, Evidence, LogRecord, ModelArtifact,
    Operation, RuleEvidence, TraceLink, TraceModel,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import TRACE_SCHEMA_VERSION
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    SAFE_DATETIME_MODULE, safe_load_buml, strip_buml_imports,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

# A trace is a flat list of constructor calls, ~40 AST nodes per element, so the
# loader's default cap (20k) would refuse a trace after a few generations. Its
# 1 MB source cap still bounds the size (about 90k nodes).
_TRACE_MAX_AST_NODES = 200_000


def utc_iso(value: datetime) -> str:
    """A naive-UTC datetime (how trace elements store time) as ISO 8601 with a ``Z`` suffix."""
    return value.isoformat(timespec="milliseconds") + "Z"


def _with_optional(target: Dict[str, Any], **fields: Optional[Any]) -> Dict[str, Any]:
    """``target`` plus every field whose value is not None."""
    target.update({key: value for key, value in fields.items() if value is not None})
    return target


def _artifact(artifact: Artifact) -> Dict[str, Any]:
    data = {"id": artifact.id, "type": type(artifact).__name__, "name": artifact.name,
            "timestamp": utc_iso(artifact.timestamp)}
    if artifact.parent is not None:
        data["parentId"] = artifact.parent.id
    if isinstance(artifact, ModelArtifact):
        data.update(elementId=artifact.element_id, elementType=artifact.element_type)
    elif isinstance(artifact, CodeArtifact):
        data.update(path=artifact.path, contentHash=artifact.content_hash)
        _with_optional(data, content=artifact.content, startLine=artifact.start_line, endLine=artifact.end_line)
    return data


def _change(change: Change) -> Dict[str, Any]:
    return _with_optional(
        {"id": change.id, "kind": change.kind.value, "artifactId": change.artifact.id,
         "timestamp": utc_iso(change.timestamp)},
        feature=change.feature, oldValue=change.old_value, newValue=change.new_value,
    )


def _log(log: LogRecord) -> Dict[str, Any]:
    return _with_optional(
        {"id": log.id, "content": log.content, "sequenceNumber": log.sequence_number,
         "logType": log.log_type.value, "timestamp": utc_iso(log.timestamp)},
        toolId=log.tool.id if log.tool else None,
    )


def _operation(operation: Operation) -> Dict[str, Any]:
    return _with_optional(
        {
            "id": operation.id,
            "operationType": operation.operation_type.value,
            "status": operation.status.value,
            "startedAt": utc_iso(operation.started_at),
            "performedBy": [a.id for a in operation.performed_by],
            "uses": [t.id for t in operation.uses],
            "consumes": [a.id for a in operation.consumes],
            "produces": [a.id for a in operation.produces],
            "changes": [_change(c) for c in operation.changes],
            "logs": [_log(log) for log in operation.logs],
        },
        endedAt=utc_iso(operation.ended_at) if operation.ended_at else None,
    )


def _evidence(evidence: Evidence) -> Dict[str, Any]:
    data = {"id": evidence.id, "type": type(evidence).__name__, "timestamp": utc_iso(evidence.timestamp)}
    if isinstance(evidence, RuleEvidence):
        data["rule"] = evidence.rule
    elif isinstance(evidence, AIEvidence):
        data.update(algorithm=evidence.algorithm, logIds=[log.id for log in evidence.logs])
    elif isinstance(evidence, ChangeEvidence):
        data["changeIds"] = [c.id for c in evidence.changes]
    return data


def _confidence(confidence: Confidence) -> Dict[str, Any]:
    return _with_optional(
        {"id": confidence.id, "value": confidence.value, "basedOn": [e.id for e in confidence.based_on],
         "timestamp": utc_iso(confidence.timestamp)},
        explanation=confidence.explanation,
    )


def _trace_link(link: TraceLink) -> Dict[str, Any]:
    return _with_optional(
        {"id": link.id, "sourceId": link.source.id, "targetId": link.target.id, "operationId": link.operation.id,
         "evidence": [_evidence(e) for e in link.evidence], "timestamp": utc_iso(link.timestamp)},
        confidence=_confidence(link.confidence) if link.confidence else None,
    )


def trace_buml_to_json(trace: TraceModel) -> Dict[str, Any]:
    """Serialize a TraceModel to the trace file JSON."""
    return {
        "schemaVersion": TRACE_SCHEMA_VERSION,
        "projectId": trace.project_id,
        "actors": [
            _with_optional({"id": a.id, "type": type(a).__name__, "name": a.name, "timestamp": utc_iso(a.timestamp)},
                           version=getattr(a, "version", None))
            for a in trace.actors
        ],
        "tools": [
            _with_optional({"id": t.id, "type": type(t).__name__, "name": t.name, "timestamp": utc_iso(t.timestamp)},
                           version=t.version, provider=getattr(t, "provider", None))
            for t in trace.tools
        ],
        "artifacts": [_artifact(a) for a in trace.artifacts],
        "operations": [_operation(op) for op in trace.operations],
        "traceLinks": [_trace_link(link) for link in trace.trace_links],
    }


def parse_trace_code(content: str) -> TraceModel:
    """Load the TraceModel that trace B-UML code (``trace_model_builder``) builds.

    ``content`` comes from a user upload, so it runs through ``safe_load_buml``:
    only the public names of ``besser.BUML.metamodel.traceability`` and
    ``datetime.datetime`` can be called.

    Raises:
        ConversionError: If the code is refused, fails, or builds no TraceModel.
    """
    allowed_names: Dict[str, Any] = {"datetime": SAFE_DATETIME_MODULE}
    for name in traceability_module.__all__:
        allowed_names[name] = getattr(traceability_module, name)
    try:
        local_vars = safe_load_buml(strip_buml_imports(content), allowed_names,
                                    max_ast_nodes=_TRACE_MAX_AST_NODES)
    except Exception as exc:
        raise ConversionError(f"Failed to load the traceability section: {exc}") from exc
    traces = [value for value in local_vars.values() if isinstance(value, TraceModel)]
    if len(traces) != 1:
        raise ConversionError(f"The traceability section must build one TraceModel, found {len(traces)}.")
    return traces[0]
