"""
Matching a trace loaded from a project file to the project imported from the same file.

Each model artifact tree of the trace (a root per model, see the traceability metamodel) is
matched to an imported diagram of its model's type (``DomainModel`` to a ``ClassDiagram``,
``GUIModel`` to a ``GUINoCodeDiagram``, ...): the only one of that type, else the one with the
model's name. The root takes the diagram's new id.

A project's Python export has no element ids in its structural sections, so importing it
gives every class diagram element a new id. The trace refers to the elements as they were, by
reference (``ModelArtifact(element=Library)``) or by record (name, type and the old id), so each
of its class diagram artifacts is matched by qualified name: ``Library``, ``Library.name``, an
association's name, ``Color.RED``. Names are unique inside one diagram, so the match is exact.
The other diagram types keep their element ids through the export where the B-UML file carries
them (a GUI's pages and components, the ``layout`` of BPMN, deployment and component elements, a
knowledge graph's nodes and edges), so an artifact of one keeps its id when the imported diagram
still has it. Agent and state machine sections carry no ids: their states, intents, tools,
skills, workspaces and RAGs are matched by kind and name instead, when the name is the only one
of its kind in the diagram (transitions and bodies have no name to match by). A quantum gate is
known by its cell under the diagram id (``<diagram id>:gate:<column>:<wire>``), so it follows its
diagram's new id. An artifact that matches nothing (a deleted element, kept as history) keeps
its recorded id.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from besser.BUML.metamodel.traceability import ModelArtifact, TraceModel

logger = logging.getLogger(__name__)

Key = Tuple[str, ...]

_TOP_LEVEL_TYPES = {"Class", "AbstractClass", "Interface", "Enumeration", "AssociationClass"}
_METHOD_NAME = re.compile(r"^\s*[+\-#~]?\s*([^(\s]+)")


def _method_name(label: str) -> str:
    """``open`` from an editor label such as ``+ open(days: int): bool``."""
    match = _METHOD_NAME.match(label)
    return match.group(1) if match else label


def _diagram_keys(model: Dict[str, Any]) -> Dict[Key, str]:
    """Qualified-name key -> element id, for one imported class diagram (editor JSON)."""
    elements = model.get("elements", {})
    keys: Dict[Key, str] = {}
    for element in elements.values():
        owner = elements.get(element.get("owner") or "")
        kind, name = element.get("type"), element.get("name", "")
        if owner is None and kind in _TOP_LEVEL_TYPES:
            keys[("type", name)] = element["id"]
        elif kind == "ClassOCLConstraint":
            keys[("constraint", name)] = element["id"]
        elif owner is not None and kind == "ClassMethod":
            keys[("method", owner.get("name", ""), _method_name(name))] = element["id"]
        elif owner is not None and kind == "ClassAttribute":
            member = "literal" if owner.get("type") == "Enumeration" else "attribute"
            keys[(member, owner.get("name", ""), name)] = element["id"]
    for relationship in model.get("relationships", {}).values():
        ends = [elements.get(relationship.get(side, {}).get("element") or "", {}).get("name", "")
                for side in ("source", "target")]
        if relationship.get("type") == "ClassInheritance":
            keys[("generalization", *ends)] = relationship["id"]
        else:
            if relationship.get("name"):
                keys[("association", relationship["name"])] = relationship["id"]
            keys[("association-ends", *sorted(ends))] = relationship["id"]
    return keys


def _artifact_keys(artifact: ModelArtifact) -> List[Key]:
    """The keys an artifact may match, best first."""
    kind, name = artifact.element_type, artifact.name
    owner = artifact.parent.name if artifact.parent is not None else ""
    if kind in ("Class", "Enumeration", "AssociationClass"):
        return [("type", name)]
    if kind == "Property":
        return [("attribute", owner, name)]
    if kind == "Method":
        return [("method", owner, _method_name(name))]
    if kind == "EnumerationLiteral":
        return [("literal", owner, name)]
    if kind == "Constraint":
        return [("constraint", name)]
    if kind == "Generalization" and " → " in name:
        return [("generalization", *name.split(" → ", 1))]
    if kind == "BinaryAssociation":
        keys: List[Key] = [("association", name)]
        if " – " in name:  # the editor's label of an unnamed association: "Book – Library"
            keys.append(("association-ends", *sorted(name.split(" – ", 1))))
        return keys
    return []


#: The project diagram type a model of each metaclass is drawn in (``ProjectInput.diagrams`` keys).
_DIAGRAM_TYPE_OF_MODEL = {
    "DomainModel": "ClassDiagram",
    "GUIModel": "GUINoCodeDiagram",
    "Agent": "AgentDiagram",
    "StateMachine": "StateMachineDiagram",
    "ObjectModel": "ObjectDiagram",
    "NN": "NNDiagram",
    "QuantumCircuit": "QuantumCircuitDiagram",
    "BPMNModel": "BPMN",
    "DeploymentModel": "DeploymentDiagram",
    "ComponentModel": "ComponentDiagram",
    "KnowledgeGraph": "KnowledgeGraphDiagram",
}
_CLASS_DIAGRAM = "ClassDiagram"


def _ids_in(model: Any) -> Set[str]:
    """Every id an imported diagram's JSON holds: the keys of its ``elements`` and ``relationships``
    and every ``id`` value, at any depth (a GUI's pages and components carry theirs in ``id`` and
    ``attributes.id``)."""
    ids: Set[str] = set()
    pending = [model]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key in ("elements", "relationships"):
                if isinstance(value.get(key), dict):
                    ids.update(k for k in value[key] if isinstance(k, str))
            if isinstance(value.get("id"), str):
                ids.add(value["id"])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return ids


# Editor element type -> the metaclass the trace records an element of it by, for the diagrams whose
# elements are matched by name (``AgentState`` with ``stateType: reasoning`` is a ``ReasoningState``).
_NAMED_ELEMENT_TYPES = {
    "AgentState": "AgentState", "AgentReasoningState": "ReasoningState", "AgentIntent": "Intent",
    "AgentTool": "Tool", "AgentSkill": "Skill", "AgentWorkspace": "Workspace", "AgentRagElement": "RAG",
    "State": "State",
}
_MATCHED_BY_NAME = {"AgentDiagram", "StateMachineDiagram"}


def _named_keys(model: Dict[str, Any]) -> Dict[Key, str]:
    """(metaclass, name) -> element id, for the elements whose name is the only one of their kind."""
    keys: Dict[Key, str] = {}
    ambiguous: Set[Key] = set()
    for element in {**(model.get("elements") or {}), **(model.get("components") or {})}.values():
        metaclass = _NAMED_ELEMENT_TYPES.get(element.get("type"))
        if metaclass == "AgentState" and element.get("stateType") == "reasoning":
            metaclass = "ReasoningState"
        if metaclass is None or not element.get("name"):
            continue
        key = (metaclass, element["name"])
        if key in keys:
            ambiguous.add(key)
        keys[key] = element["id"]
    return {key: element_id for key, element_id in keys.items() if key not in ambiguous}


def _root(artifact: ModelArtifact) -> ModelArtifact:
    while isinstance(artifact.parent, ModelArtifact):
        artifact = artifact.parent
    return artifact


def match_trace_to_project(trace: TraceModel, project: Dict[str, Any], model_names: List[str]) -> int:
    """Give the trace's model artifacts the ids of the matching elements of ``project``.

    Args:
        trace: The trace loaded from the project file.
        project: The imported project (``project_to_json``).
        model_names: The name of the domain model of each imported class diagram, in order.

    Returns:
        How many artifacts matched nothing and keep their recorded id.
    """
    all_diagrams = project.get("diagrams", {})

    def diagram_of(root: ModelArtifact) -> Optional[Dict[str, Any]]:
        diagram_type = _DIAGRAM_TYPE_OF_MODEL.get(root.element_type)
        diagrams = all_diagrams.get(diagram_type, []) if diagram_type else []
        if len(diagrams) == 1:
            return diagrams[0]
        names = model_names if diagram_type == _CLASS_DIAGRAM else [d.get("title") for d in diagrams]
        indexes = [i for i, name in enumerate(names) if name == root.name]
        return diagrams[indexes[0]] if len(indexes) == 1 and indexes[0] < len(diagrams) else None

    keys_of: Dict[int, Dict[Key, str]] = {}
    ids_of: Dict[int, Set[str]] = {}
    # The roots' recorded ids, before they take their diagram's: a quantum gate's id starts with it.
    recorded_root_ids = {id(a): a.element_id for a in trace.artifacts if isinstance(a, ModelArtifact) and a.parent is None}
    unmatched = 0
    for artifact in trace.artifacts:
        if not isinstance(artifact, ModelArtifact):
            continue
        root = _root(artifact)
        diagram = diagram_of(root)
        diagram_type = _DIAGRAM_TYPE_OF_MODEL.get(root.element_type)
        if diagram is None:
            new_id = None
        elif artifact is root:
            new_id = diagram["id"]
        elif diagram_type == _CLASS_DIAGRAM:
            keys = keys_of.setdefault(id(diagram), _diagram_keys(diagram.get("model", {})))
            new_id = next((keys[key] for key in _artifact_keys(artifact) if key in keys), None)
        elif diagram_type == "QuantumCircuitDiagram":
            prefix = f"{recorded_root_ids.get(id(root))}:"
            new_id = (f"{diagram['id']}:{artifact.element_id[len(prefix):]}"
                      if artifact.element_id and artifact.element_id.startswith(prefix) else None)
        else:  # its element ids survive the export, or (agent, state machine) its names do
            ids = ids_of.setdefault(id(diagram), _ids_in(diagram.get("model", {})))
            new_id = artifact.element_id if artifact.element_id in ids else None
            if new_id is None and diagram_type in _MATCHED_BY_NAME:
                keys = keys_of.setdefault(id(diagram), _named_keys(diagram.get("model", {})))
                new_id = keys.get((artifact.element_type, artifact.name))
        if new_id is not None:
            artifact.element_id = new_id
        elif artifact.element_id is None:
            # A reference the import could not place: it keeps its trace artifact's own id.
            logger.warning("Trace artifact %r (%s %r) matches no imported element.",
                           artifact.id, artifact.element_type, artifact.name)
            artifact.element_id = artifact.id
            unmatched += 1
        else:
            unmatched += 1
    return unmatched
