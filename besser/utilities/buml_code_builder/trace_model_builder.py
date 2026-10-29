"""
Trace Model Builder: Generates Python code for BESSER traceability models.

Counterpart of ``parse_trace_code`` in
``besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json
.trace_converter``, which loads the code this module writes through
``safe_load_buml``. The emitted code therefore sticks to literals and
constructor calls on the public names of ``besser.BUML.metamodel.traceability``.

Every element keeps its id and timestamp, so a trace exported as Python and
loaded back is the same trace as its JSON form. Variables are numbered
(``artifact_3``, ``change_12``) rather than named after elements: trace names
are free text, and a trace can hold many elements with the same name.
"""

from collections import Counter
from typing import Dict, List, TextIO

from besser.BUML.metamodel.traceability import (
    AIEvidence, Agent, Artifact, Change, ChangeEvidence, CodeArtifact, Confidence, Evidence, LLM, LogRecord,
    ModelArtifact, Operation, RuleEvidence, TraceElement, TraceLink, TraceModel,
)
from besser.utilities.buml_code_builder.common import _comment_safe, _escape_python_string, buml_header, section_banner

__all__ = ["trace_model_to_code", "TRACE_SECTION_KEYWORD"]

#: Section keyword of a trace in a project file; ``project_to_json`` splits on it.
TRACE_SECTION_KEYWORD = "TRACEABILITY"

_IMPORTED_NAMES = (
    "TraceModel, Person, Agent, Editor, CodeGenerator, LLM, ModelArtifact, CodeArtifact,\n"
    "    Operation, OperationType, OperationStatus, Change, ChangeKind, LogRecord, LogType,\n"
    "    RuleEvidence, AIEvidence, ChangeEvidence, Confidence, TraceLink,"
)


def trace_model_to_code(model: TraceModel, file_path: str, model_var_name: str = "trace_model"):
    """
    Generates Python code for a TraceModel.

    Args:
        model (TraceModel): The trace.
        file_path (str): The path to save the generated code.
        model_var_name (str, optional): Name of the TraceModel variable in the
            generated code. ``project_to_json`` reads a project's trace from the
            ``trace_model`` variable, so ``project_to_code`` keeps the default.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(buml_header())
        f.write(section_banner(TRACE_SECTION_KEYWORD) + "\n")
        f.write("import datetime\n")
        f.write(f"from besser.BUML.metamodel.traceability import (\n    {_IMPORTED_NAMES}\n)\n\n")
        _TraceWriter(f).write(model, model_var_name)


def _string(value: str) -> str:
    return f'"{_escape_python_string(value)}"'


def _optional(value) -> str:
    """A string or int literal, or None."""
    if value is None:
        return "None"
    return _string(value) if isinstance(value, str) else repr(value)


def _common(element: TraceElement) -> str:
    """The ``id=`` and ``timestamp=`` arguments every trace element is rebuilt with."""
    iso = element.timestamp.isoformat()
    return f"id={_string(element.id)}, timestamp=datetime.datetime.fromisoformat({_string(iso)})"


def _var_list(variables: List[str]) -> str:
    return "[" + ", ".join(variables) + "]"


class _TraceWriter:
    """Writes one trace, keeping the variable bound to each element id."""

    def __init__(self, out: TextIO):
        self.out = out
        self.vars: Dict[str, str] = {}
        self.counts: Counter = Counter()

    def _bind(self, element: TraceElement, prefix: str, constructor: str):
        self.counts[prefix] += 1
        var = f"{prefix}_{self.counts[prefix]}"
        self.vars[element.id] = var
        self.out.write(f"{var} = {constructor}\n")

    def _refs(self, elements) -> str:
        return _var_list([self.vars[e.id] for e in elements])

    def write(self, model: TraceModel, model_var_name: str):
        self.out.write("# Actors\n")
        for actor in model.actors:
            version = f", version={_optional(actor.version)}" if isinstance(actor, Agent) else ""
            self._bind(actor, "actor", f"{type(actor).__name__}({_string(actor.name)}{version}, {_common(actor)})")

        self.out.write("\n# Tools\n")
        for tool in model.tools:
            provider = f", provider={_string(tool.provider)}" if isinstance(tool, LLM) else ""
            self._bind(tool, "tool", f"{type(tool).__name__}({_string(tool.name)}{provider}, "
                                     f"version={_optional(tool.version)}, {_common(tool)})")

        self.out.write("\n# Artifacts\n")
        for artifact in _parents_first(model.artifacts):
            self._bind(artifact, "artifact", self._artifact(artifact))

        for index, operation in enumerate(model.operations, start=1):
            self._operation(index, operation)

        self.out.write("\n# Trace links\n")
        for link in model.trace_links:
            self._trace_link(link)

        self.out.write(f"\n{model_var_name} = TraceModel(\n")
        self.out.write(f"    project_id={_optional(model.project_id)},\n")
        self.out.write(f"    actors={self._refs(model.actors)},\n")
        self.out.write(f"    tools={self._refs(model.tools)},\n")
        self.out.write(f"    artifacts={self._refs(model.artifacts)},\n")
        self.out.write(f"    operations={self._refs(model.operations)},\n")
        self.out.write(f"    trace_links={self._refs(model.trace_links)},\n")
        self.out.write(")\n")

    def _artifact(self, artifact: Artifact) -> str:
        parent = f", parent={self.vars[artifact.parent.id]}" if artifact.parent is not None else ""
        if isinstance(artifact, ModelArtifact):
            return (f"ModelArtifact({_string(artifact.name)}, element_id={_string(artifact.element_id)}, "
                    f"element_type={_string(artifact.element_type)}{parent}, {_common(artifact)})")
        if isinstance(artifact, CodeArtifact):
            return (f"CodeArtifact({_string(artifact.name)}, path={_string(artifact.path)}, "
                    f"content_hash={_string(artifact.content_hash)}, content={_optional(artifact.content)}, "
                    f"start_line={_optional(artifact.start_line)}, end_line={_optional(artifact.end_line)}"
                    f"{parent}, {_common(artifact)})")
        raise TypeError(f"Unsupported artifact type {type(artifact).__name__}.")

    def _operation(self, index: int, operation: Operation):
        actors = ", ".join(_comment_safe(a.name) for a in operation.performed_by)
        self.out.write(f"\n# Operation {index}: {operation.operation_type.name} by {actors}\n")
        for change in operation.changes:
            self._bind(change, "change", self._change(change))
        for log in operation.logs:
            self._bind(log, "log", self._log(log))
        self._bind(operation, "operation", (
            f"Operation(OperationType.{operation.operation_type.name}, {self._refs(operation.performed_by)}, "
            f"status=OperationStatus.{operation.status.name}, "
            f"started_at=datetime.datetime.fromisoformat({_string(operation.started_at.isoformat())}), "
            f"ended_at={self._datetime(operation.ended_at)}, uses={self._refs(operation.uses)}, "
            f"consumes={self._refs(operation.consumes)}, produces={self._refs(operation.produces)}, "
            f"changes={self._refs(operation.changes)}, logs={self._refs(operation.logs)}, "
            f"id={_string(operation.id)})"
        ))

    @staticmethod
    def _datetime(value) -> str:
        return "None" if value is None else f"datetime.datetime.fromisoformat({_string(value.isoformat())})"

    def _change(self, change: Change) -> str:
        return (f"Change(ChangeKind.{change.kind.name}, {self.vars[change.artifact.id]}, "
                f"feature={_optional(change.feature)}, old_value={_optional(change.old_value)}, "
                f"new_value={_optional(change.new_value)}, {_common(change)})")

    def _log(self, log: LogRecord) -> str:
        tool = self.vars[log.tool.id] if log.tool is not None else "None"
        return (f"LogRecord({_string(log.content)}, {log.sequence_number}, LogType.{log.log_type.name}, "
                f"tool={tool}, {_common(log)})")

    def _evidence(self, evidence: Evidence) -> str:
        if isinstance(evidence, RuleEvidence):
            return f"RuleEvidence({_string(evidence.rule)}, {_common(evidence)})"
        if isinstance(evidence, AIEvidence):
            return f"AIEvidence({_string(evidence.algorithm)}, logs={self._refs(evidence.logs)}, {_common(evidence)})"
        if isinstance(evidence, ChangeEvidence):
            return f"ChangeEvidence({self._refs(evidence.changes)}, {_common(evidence)})"
        raise TypeError(f"Unsupported evidence type {type(evidence).__name__}.")

    def _trace_link(self, link: TraceLink):
        based_on = link.confidence.based_on if link.confidence is not None else []
        for evidence in link.evidence + based_on:
            if evidence.id not in self.vars:  # a confidence is usually based on the link's own evidence
                self._bind(evidence, "evidence", self._evidence(evidence))
        confidence = "None"
        if link.confidence is not None:
            self._bind(link.confidence, "confidence", self._confidence(link.confidence))
            confidence = self.vars[link.confidence.id]
        self._bind(link, "link", (
            f"TraceLink({self.vars[link.source.id]}, {self.vars[link.target.id]}, {self.vars[link.operation.id]}, "
            f"evidence={self._refs(link.evidence)}, confidence={confidence}, {_common(link)})"
        ))

    def _confidence(self, confidence: Confidence) -> str:
        return (f"Confidence({confidence.value!r}, based_on={self._refs(confidence.based_on)}, "
                f"explanation={_optional(confidence.explanation)}, {_common(confidence)})")


def _parents_first(artifacts: List[Artifact]) -> List[Artifact]:
    """``artifacts`` reordered so that each parent comes before its fragments."""
    ordered: List[Artifact] = []
    placed = set()

    def place(artifact: Artifact):
        if artifact.id in placed:
            return
        if artifact.parent is not None:
            place(artifact.parent)
        placed.add(artifact.id)
        ordered.append(artifact)

    for artifact in artifacts:
        place(artifact)
    return ordered
