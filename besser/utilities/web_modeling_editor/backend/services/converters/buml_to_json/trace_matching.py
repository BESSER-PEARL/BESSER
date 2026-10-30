"""
Matching a trace loaded from a project file to the project imported from the same file.

A project's Python export has no element ids in its structural sections, so importing it
gives every element a new id. The trace refers to the elements as they were, by reference
(``ModelArtifact(element=Library)``) or by record (name, type and the old id), so each of its
model artifacts is matched to the imported diagrams by qualified name: ``Library``,
``Library.name``, an association's name, ``Color.RED``. Names are unique inside one diagram,
so the match is exact; an artifact that matches nothing (a deleted element, kept as
history) keeps its recorded id.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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
    diagrams = project.get("diagrams", {}).get("ClassDiagram", [])

    def diagram_of(root: ModelArtifact) -> Optional[Dict[str, Any]]:
        if len(diagrams) == 1:
            return diagrams[0]
        indexes = [i for i, name in enumerate(model_names) if name == root.name]
        return diagrams[indexes[0]] if len(indexes) == 1 and indexes[0] < len(diagrams) else None

    keys_of: Dict[int, Dict[Key, str]] = {}
    unmatched = 0
    for artifact in trace.artifacts:
        if not isinstance(artifact, ModelArtifact):
            continue
        root = _root(artifact)
        diagram = diagram_of(root)
        if diagram is None:
            new_id = None
        elif artifact is root:
            new_id = diagram["id"]
        else:
            keys = keys_of.setdefault(id(diagram), _diagram_keys(diagram.get("model", {})))
            new_id = next((keys[key] for key in _artifact_keys(artifact) if key in keys), None)
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
