"""
Trace Model Builder: Generates Python code for BESSER traceability models.

Counterpart of ``parse_trace_code`` in
``besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json
.trace_converter``, which loads the code this module writes through
``safe_load_buml``. The emitted code therefore sticks to literals and
constructor calls on the public names of ``besser.BUML.metamodel.traceability``.

Every trace element keeps its id and timestamp. Variables are numbered
(``artifact_3``, ``change_12``) rather than named after elements: trace names
are free text, and a trace can hold many elements with the same name.

In a project file the trace follows its models, so a model element it is about is
referenced, not repeated: ``ModelArtifact(element=Library)``. The variable is the
one the structural section binds the element to, found with the structural
builder's own public naming helpers (:func:`structural_variables`). An element
without such a variable (an association, a literal, a deleted element) stays a
record with its name, type and id; the importer matches it by name.
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

from besser.BUML.metamodel.traceability import (
    AIEvidence, Agent, Artifact, Change, ChangeEvidence, CodeArtifact, CodeGeneration, Confidence, Evidence, LLM,
    Conversation, LogRecord, ModelArtifact, Modeling, Operation, PromptArtifact, RuleEvidence, TraceElement, TraceLink,
    TraceModel,
)
from besser.BUML.metamodel.structural import AssociationClass, DomainModel
from besser.utilities.buml_code_builder.common import (
    _comment_safe, _escape_python_string, attribute_var_name, buml_header, method_var_name, safe_class_name,
    section_banner,
)

__all__ = ["trace_model_to_code", "structural_variables", "TRACE_SECTION_KEYWORD"]

#: The trace section's own variables (and names it reads): an element variable with one of
#: these names would be shadowed before the trace references it, so it is not referenced.
_TRACE_NAMES = re.compile(r"^((actor|tool|artifact|change|log|operation|evidence|confidence|link)_\d+|datetime)$")

#: Section keyword of a trace in a project file; ``project_to_json`` splits on it.
TRACE_SECTION_KEYWORD = "TRACEABILITY"

_IMPORTED_NAMES = (
    "TraceModel, Person, Agent, Editor, CodeGenerator, LLM, ModelArtifact, CodeArtifact, PromptArtifact,\n"
    "    Modeling, CodeGeneration, Conversation, OperationStatus, Change, ChangeKind, LogRecord, LogType,\n"
    "    RuleEvidence, AIEvidence, ChangeEvidence, Confidence, TraceLink,"
)


def trace_model_to_code(model: TraceModel, file_path: str, model_var_name: str = "trace_model",
                        references: Optional[Dict[str, str]] = None):
    """
    Generates Python code for a TraceModel.

    Args:
        model (TraceModel): The trace.
        file_path (str): The path to save the generated code.
        model_var_name (str, optional): Name of the TraceModel variable in the
            generated code. ``project_to_json`` reads a project's trace from the
            ``trace_model`` variable, so ``project_to_code`` keeps the default.
        references (dict, optional): The variables of the model elements the code
            can reference, from :func:`structural_variables`. Without it every
            model artifact is written as a record.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(buml_header())
        f.write(section_banner(TRACE_SECTION_KEYWORD) + "\n")
        f.write("import datetime\n")
        f.write(f"from besser.BUML.metamodel.traceability import (\n    {_IMPORTED_NAMES}\n)\n\n")
        _TraceWriter(f, references or {}).write(model, model_var_name)


def model_key(name: str) -> str:
    """The :func:`structural_variables` key of a whole model, which carries no id of its own."""
    return f"model:{name}"


def structural_variables(domain_models: Sequence[Tuple[DomainModel, str]],
                         other_models: Iterable[DomainModel] = ()) -> Dict[str, str]:
    """The variables a project file binds model elements to, for the trace to reference.

    Keyed by element id (classes, enumerations, association classes, their attributes and
    methods) and by :func:`model_key` (the models themselves). The names come from the
    structural builder's public helpers, so they are the ones its sections define. A name
    that two elements would share in the one file (two diagrams with a class ``Book``), or
    that the trace section uses itself, is left out: those elements stay records.

    Args:
        domain_models: Each written domain model with its variable (``domain_model``, ...).
        other_models: Domain models written in the same file whose names can collide
            (the user reference model), not referenced themselves.
    """
    candidates: List[Tuple[Optional[str], str]] = []  # (key or None when not referenced, variable)

    def collect(model: DomainModel, key: bool):
        for type_ in model.types:
            if not hasattr(type_, "attributes") and not hasattr(type_, "literals"):
                continue  # primitive types are not written as variables
            candidates.append((type_.id if key else None, safe_class_name(type_.name)))
            for attribute in getattr(type_, "attributes", ()):
                candidates.append((attribute.id if key else None, attribute_var_name(type_.name, attribute.name)))
            if not isinstance(type_, AssociationClass):
                for method in getattr(type_, "methods", ()):
                    candidates.append((method.id if key else None, method_var_name(type_.name, method.name)))

    for model, variable in domain_models:
        candidates.append((model_key(model.name), variable))
        collect(model, key=True)
    for model in other_models:
        collect(model, key=False)

    uses = Counter(variable for _, variable in candidates)
    keys = Counter(key for key, _ in candidates if key)
    return {key: variable for key, variable in candidates
            if key and uses[variable] == 1 and keys[key] == 1 and not _TRACE_NAMES.match(variable)}


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

    def __init__(self, out: TextIO, references: Dict[str, str]):
        self.out = out
        self.references = references
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

    def _reference(self, artifact: ModelArtifact) -> Optional[str]:
        """The variable of the element the artifact stands for, when the file defines one."""
        if artifact.element_type == "DomainModel" and artifact.parent is None:
            return self.references.get(model_key(artifact.name))
        return self.references.get(artifact.element_id)

    def _artifact(self, artifact: Artifact) -> str:
        parent = f", parent={self.vars[artifact.parent.id]}" if artifact.parent is not None else ""
        if isinstance(artifact, ModelArtifact):
            element = self._reference(artifact)
            if element:
                return f"ModelArtifact(element={element}{parent}, {_common(artifact)})"
            return (f"ModelArtifact({_string(artifact.name)}, element_id={_string(artifact.element_id)}, "
                    f"element_type={_string(artifact.element_type)}{parent}, {_common(artifact)})")
        if isinstance(artifact, CodeArtifact):
            return (f"CodeArtifact({_string(artifact.name)}, path={_string(artifact.path)}, "
                    f"content_hash={_string(artifact.content_hash)}, content={_optional(artifact.content)}, "
                    f"start_line={_optional(artifact.start_line)}, end_line={_optional(artifact.end_line)}"
                    f"{parent}, {_common(artifact)})")
        if isinstance(artifact, PromptArtifact):
            return (f"PromptArtifact({_string(artifact.name)}, content={_string(artifact.content)}, "
                    f"content_hash={_string(artifact.content_hash)}{parent}, {_common(artifact)})")
        raise TypeError(f"Unsupported artifact type {type(artifact).__name__}.")

    def _operation(self, index: int, operation: Operation):
        kind = type(operation).__name__
        actors = ", ".join(_comment_safe(a.name) for a in operation.performed_by)
        self.out.write(f"\n# Operation {index}: {kind} by {actors}\n")
        if not isinstance(operation, (Modeling, CodeGeneration, Conversation)):
            raise TypeError(f"Unsupported operation type {kind}.")
        specific = ""
        if isinstance(operation, Modeling):
            for change in operation.changes:
                self._bind(change, "change", self._change(change))
            specific = f"changes={self._refs(operation.changes)}, "
        for log in operation.logs:
            self._bind(log, "log", self._log(log))
        self._bind(operation, "operation", (
            f"{kind}({self._refs(operation.performed_by)}, {specific}"
            f"status=OperationStatus.{operation.status.name}, "
            f"started_at=datetime.datetime.fromisoformat({_string(operation.started_at.isoformat())}), "
            f"ended_at={self._datetime(operation.ended_at)}, uses={self._refs(operation.uses)}, "
            f"consumes={self._refs(operation.consumes)}, produces={self._refs(operation.produces)}, "
            f"logs={self._refs(operation.logs)}, id={_string(operation.id)})"
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
