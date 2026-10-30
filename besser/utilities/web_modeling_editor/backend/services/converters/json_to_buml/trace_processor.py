"""
Trace file JSON (the ``trace`` of a project export) -> TraceModel.

The inverse of ``buml_to_json/trace_converter.py``. Unlike the diagram
processors it is strict: a trace is a record, so a missing or malformed field
is a ConversionError naming it, never a silently defaulted value.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from besser.BUML.metamodel.traceability import (
    LLM, AIEvidence, Agent, Change, ChangeEvidence, ChangeKind, CodeArtifact, CodeGeneration, CodeGenerator, Confidence,
    Conversation, Editor, LogRecord, LogType, ModelArtifact, Modeling, OperationStatus, Person, PromptArtifact,
    RuleEvidence, TraceLink, TraceModel,
)
from besser.utilities.utils import parse_iso_timestamp
from besser.utilities.web_modeling_editor.backend.constants.constants import TRACE_SCHEMA_VERSION
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

_ACTOR_TYPES = {"Person": Person, "Agent": Agent}
_OPERATION_TYPES = {"Modeling": Modeling, "CodeGeneration": CodeGeneration, "Conversation": Conversation}
_TOOL_TYPES = {"Editor": Editor, "CodeGenerator": CodeGenerator, "LLM": LLM}


def _field(data: Dict[str, Any], key: str, where: str, optional: bool = False) -> Any:
    if not isinstance(data, dict):
        raise ConversionError(f"{where} must be an object.")
    if key not in data or data[key] is None:
        if optional:
            return None
        raise ConversionError(f"{where} is missing '{key}'.")
    return data[key]


def _list(data: Dict[str, Any], key: str, where: str) -> List[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ConversionError(f"{where}.{key} must be a list.")
    return value


def _timestamp(data: Dict[str, Any], key: str, where: str, optional: bool = False) -> Optional[datetime]:
    raw = _field(data, key, where, optional)
    if raw is None:
        return None
    parsed = parse_iso_timestamp(raw)
    if parsed is None:
        raise ConversionError(f"{where}.{key} is not an ISO 8601 timestamp: {raw!r}.")
    return parsed


def _enum(enum_type, data: Dict[str, Any], key: str, where: str):
    raw = _field(data, key, where)
    try:
        return enum_type(raw)
    except ValueError:
        raise ConversionError(
            f"{where}.{key} must be one of {[e.value for e in enum_type]}, got {raw!r}.") from None


def _resolve(index: Dict[str, Any], ref_id: Any, what: str, where: str):
    if not isinstance(ref_id, str) or ref_id not in index:
        raise ConversionError(f"{where} references unknown {what} {ref_id!r}.")
    return index[ref_id]


def _build(factory: Callable, where: str, **kwargs):
    """Call a metamodel constructor, reporting its validation error as a ConversionError."""
    try:
        return factory(**kwargs)
    except (TypeError, ValueError) as error:
        raise ConversionError(f"{where}: {error}") from error


def process_trace(data: Dict[str, Any]) -> TraceModel:
    """Build a TraceModel from trace file JSON.

    Raises:
        ConversionError: If the JSON is not a valid trace of the supported schema version.
    """
    version = _field(data, "schemaVersion", "Trace")
    if version != TRACE_SCHEMA_VERSION:
        raise ConversionError(f"Unsupported trace schemaVersion {version!r}; expected {TRACE_SCHEMA_VERSION}.")

    actors, tools, artifacts, changes, logs, operations = {}, {}, {}, {}, {}, {}

    for raw in _list(data, "actors", "Trace"):
        where = f"Actor {raw.get('id') if isinstance(raw, dict) else raw!r}"
        kind = _resolve(_ACTOR_TYPES, _field(raw, "type", where), "actor type", where)
        kwargs = {"name": _field(raw, "name", where), "id": _field(raw, "id", where),
                  "timestamp": _timestamp(raw, "timestamp", where)}
        if kind is Agent:
            kwargs["version"] = _field(raw, "version", where, optional=True)
        actors[kwargs["id"]] = _build(kind, where, **kwargs)

    for raw in _list(data, "tools", "Trace"):
        where = f"Tool {raw.get('id') if isinstance(raw, dict) else raw!r}"
        kind = _resolve(_TOOL_TYPES, _field(raw, "type", where), "tool type", where)
        kwargs = {"name": _field(raw, "name", where), "version": _field(raw, "version", where, optional=True),
                  "id": _field(raw, "id", where), "timestamp": _timestamp(raw, "timestamp", where)}
        if kind is LLM:
            kwargs["provider"] = _field(raw, "provider", where)
        tools[kwargs["id"]] = _build(kind, where, **kwargs)

    raw_artifacts = _list(data, "artifacts", "Trace")
    for raw in raw_artifacts:
        where = f"Artifact {raw.get('id') if isinstance(raw, dict) else raw!r}"
        kind = _field(raw, "type", where)
        common = {"name": _field(raw, "name", where), "id": _field(raw, "id", where),
                  "timestamp": _timestamp(raw, "timestamp", where)}
        if kind == "ModelArtifact":
            artifact = _build(ModelArtifact, where, element_id=_field(raw, "elementId", where),
                              element_type=_field(raw, "elementType", where), **common)
        elif kind == "CodeArtifact":
            artifact = _build(CodeArtifact, where, path=_field(raw, "path", where),
                              content_hash=_field(raw, "contentHash", where),
                              content=_field(raw, "content", where, optional=True),
                              start_line=_field(raw, "startLine", where, optional=True),
                              end_line=_field(raw, "endLine", where, optional=True), **common)
        elif kind == "PromptArtifact":
            artifact = _build(PromptArtifact, where, content=_field(raw, "content", where),
                              content_hash=_field(raw, "contentHash", where), **common)
        else:
            raise ConversionError(f"{where} has unknown type {kind!r}.")
        if artifact.id in artifacts:
            raise ConversionError(f"Artifact id {artifact.id!r} is used twice.")
        artifacts[artifact.id] = artifact
    # Parents can come later in the list, so they are wired once every artifact exists.
    for raw in raw_artifacts:
        parent_id = raw.get("parentId")
        if parent_id is not None:
            where = f"Artifact {raw['id']!r}"
            parent = _resolve(artifacts, parent_id, "artifact", where)
            try:
                artifacts[raw["id"]].parent = parent
            except ValueError as error:
                raise ConversionError(f"{where}: {error}") from error

    trace = TraceModel(project_id=data.get("projectId"), actors=list(actors.values()),
                       tools=list(tools.values()), artifacts=list(artifacts.values()))

    for raw in _list(data, "operations", "Trace"):
        where = f"Operation {raw.get('id') if isinstance(raw, dict) else raw!r}"
        kind = _field(raw, "type", where)
        if kind not in _OPERATION_TYPES:
            raise ConversionError(f"{where} has unknown type {kind!r}; expected one of {sorted(_OPERATION_TYPES)}.")
        if kind != "Modeling" and _list(raw, "changes", where):
            raise ConversionError(f"{where} is a {kind}: only a Modeling has changes.")
        op_changes = []
        for raw_change in _list(raw, "changes", where):
            cwhere = f"Change {raw_change.get('id') if isinstance(raw_change, dict) else raw_change!r}"
            change = _build(
                Change, cwhere, kind=_enum(ChangeKind, raw_change, "kind", cwhere),
                artifact=_resolve(artifacts, _field(raw_change, "artifactId", cwhere), "artifact", cwhere),
                feature=_field(raw_change, "feature", cwhere, optional=True),
                old_value=_field(raw_change, "oldValue", cwhere, optional=True),
                new_value=_field(raw_change, "newValue", cwhere, optional=True),
                id=_field(raw_change, "id", cwhere), timestamp=_timestamp(raw_change, "timestamp", cwhere))
            changes[change.id] = change
            op_changes.append(change)
        op_logs = []
        for raw_log in _list(raw, "logs", where):
            lwhere = f"LogRecord {raw_log.get('id') if isinstance(raw_log, dict) else raw_log!r}"
            tool_id = _field(raw_log, "toolId", lwhere, optional=True)
            log = _build(
                LogRecord, lwhere, content=_field(raw_log, "content", lwhere),
                sequence_number=_field(raw_log, "sequenceNumber", lwhere),
                log_type=_enum(LogType, raw_log, "logType", lwhere),
                tool=_resolve(tools, tool_id, "tool", lwhere) if tool_id is not None else None,
                id=_field(raw_log, "id", lwhere), timestamp=_timestamp(raw_log, "timestamp", lwhere))
            logs[log.id] = log
            op_logs.append(log)
        common = {
            "performed_by": [_resolve(actors, i, "actor", where) for i in _list(raw, "performedBy", where)],
            "status": _enum(OperationStatus, raw, "status", where),
            "started_at": _timestamp(raw, "startedAt", where),
            "ended_at": _timestamp(raw, "endedAt", where, optional=True),
            "uses": [_resolve(tools, i, "tool", where) for i in _list(raw, "uses", where)],
            "consumes": [_resolve(artifacts, i, "artifact", where) for i in _list(raw, "consumes", where)],
            "produces": [_resolve(artifacts, i, "artifact", where) for i in _list(raw, "produces", where)],
            "logs": op_logs, "id": _field(raw, "id", where),
        }
        if kind == "Modeling":
            operation = _build(Modeling, where, changes=op_changes, **common)
        else:
            operation = _build(_OPERATION_TYPES[kind], where, **common)
        operations[operation.id] = operation
        trace.add_operation(operation)

    for raw in _list(data, "traceLinks", "Trace"):
        where = f"TraceLink {raw.get('id') if isinstance(raw, dict) else raw!r}"
        evidence = {}
        for raw_ev in _list(raw, "evidence", where):
            ewhere = f"Evidence {raw_ev.get('id') if isinstance(raw_ev, dict) else raw_ev!r}"
            kind = _field(raw_ev, "type", ewhere)
            common = {"id": _field(raw_ev, "id", ewhere), "timestamp": _timestamp(raw_ev, "timestamp", ewhere)}
            if kind == "RuleEvidence":
                ev = _build(RuleEvidence, ewhere, rule=_field(raw_ev, "rule", ewhere), **common)
            elif kind == "AIEvidence":
                ev = _build(AIEvidence, ewhere, algorithm=_field(raw_ev, "algorithm", ewhere),
                            logs=[_resolve(logs, i, "log record", ewhere) for i in _list(raw_ev, "logIds", ewhere)],
                            **common)
            elif kind == "ChangeEvidence":
                ev = _build(ChangeEvidence, ewhere,
                            changes=[_resolve(changes, i, "change", ewhere)
                                     for i in _list(raw_ev, "changeIds", ewhere)], **common)
            else:
                raise ConversionError(f"{ewhere} has unknown type {kind!r}.")
            evidence[ev.id] = ev
        confidence = None
        raw_conf = raw.get("confidence")
        if raw_conf is not None:
            cwhere = f"Confidence of {where}"
            confidence = _build(
                Confidence, cwhere, value=_field(raw_conf, "value", cwhere),
                based_on=[_resolve(evidence, i, "evidence", cwhere) for i in _list(raw_conf, "basedOn", cwhere)],
                explanation=_field(raw_conf, "explanation", cwhere, optional=True),
                id=_field(raw_conf, "id", cwhere), timestamp=_timestamp(raw_conf, "timestamp", cwhere))
        trace.add_trace_link(_build(
            TraceLink, where,
            source=_resolve(artifacts, _field(raw, "sourceId", where), "artifact", where),
            target=_resolve(artifacts, _field(raw, "targetId", where), "artifact", where),
            operation=_resolve(operations, _field(raw, "operationId", where), "operation", where),
            evidence=list(evidence.values()), confidence=confidence,
            id=_field(raw, "id", where), timestamp=_timestamp(raw, "timestamp", where)))

    result = trace.validate(raise_exception=False)
    if not result["success"]:
        raise ConversionError("Invalid trace: " + "; ".join(result["errors"]))
    return trace
