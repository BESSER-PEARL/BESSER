"""Class diagrams of BESSER Lab: a plan's ``model_spec`` to the editor's JSON, and back from a project.

A metamodel or diagram-type plan carries a class diagram of the extension (``Plan.model_spec``, shaped by
``prompts.MODEL_SPEC_SCHEMA``). It reaches the editor as a B-UML ``DomainModel`` converted by the editor's own
``class_buml_to_json``; the user's reviewed diagram (the one the editor marks as the change's) comes back
through ``process_class_diagram`` and is listed
for the build step in the same ``model_spec`` shape. In both directions a composition's target is the whole,
as in the editor's class diagrams.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    Property,
    Type,
    data_types,
)
from besser.spec_driven_agent.lab.contracts import Plan
from besser.utilities.utils import sort_by_timestamp
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.utils import sanitize_name
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.multiplicity_parser import (
    format_multiplicity_label,
    parse_multiplicity,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)

_PRIMITIVES = {primitive.name: primitive for primitive in data_types}
# What a spec can get wrong that B-UML or the converter refuses.
SPEC_ERRORS = (ConversionError, KeyError, TypeError, ValueError)
# ``DiagramInput.config`` keys by which the editor ties a class diagram to a lab change (see _change_diagram).
_CHANGE_MARKERS = ("labProposalChangeId", "labChangeId")


def model_spec_to_domain_model(spec: Mapping[str, Any], name: str) -> DomainModel:
    """Build the B-UML class diagram ``spec`` describes.

    Raises:
        ConversionError, KeyError, TypeError, ValueError: the spec names an unknown type or class, an invalid
            multiplicity, or a name B-UML refuses.
    """
    enumerations = {
        e["name"]: Enumeration(e["name"], literals={EnumerationLiteral(literal) for literal in e["literals"]})
        for e in spec.get("enumerations") or []
    }
    classes = {c["name"]: Class(c["name"], is_abstract=bool(c.get("is_abstract"))) for c in spec.get("classes") or []}

    def resolve(type_name: str) -> Type:
        found = classes.get(type_name) or enumerations.get(type_name) or _PRIMITIVES.get(type_name)
        if found is None:
            raise ValueError(f"unknown type '{type_name}'")
        return found

    for class_spec in spec.get("classes") or []:
        classes[class_spec["name"]].attributes = {
            Property(attribute["name"], resolve(attribute["type"])) for attribute in class_spec.get("attributes") or []
        }

    parents: dict[str, set[str]] = {}
    for g in spec.get("generalizations") or []:
        parents.setdefault(g["child"], set()).add(g["parent"])

    def ancestors(name: str) -> set[str]:
        found, todo = set(), list(parents.get(name, ()))
        while todo:
            parent = todo.pop()
            if parent == name:  # B-UML fails on a cycle with an AttributeError
                raise ValueError(f"the generalizations of '{name}' form a cycle")
            if parent not in found:
                found.add(parent)
                todo.extend(parents.get(parent, ()))
        return found

    lineage = {name: ancestors(name) for name in classes}

    # End names are the other class's name, suffixed when the class, an ancestor or a descendant (which
    # inherit each other's ends) already has an end of that name.
    taken_ends: dict[str, set[str]] = {}

    def end_name(owner: str, class_name: str) -> str:
        related = {owner, *lineage[owner], *(name for name, up in lineage.items() if owner in up)}
        taken = set().union(*(taken_ends.get(name, set()) for name in related))
        candidate, counter = class_name.lower(), 1
        while candidate in taken:
            candidate = f"{class_name.lower()}_{counter}"
            counter += 1
        taken_ends.setdefault(owner, set()).add(candidate)
        return candidate

    associations = set()
    for a in spec.get("associations") or []:
        source, target = classes[a["source"]], classes[a["target"]]
        # Created source first: buml_to_json draws an association in the order its ends were created.
        source_end = Property(end_name(target.name, source.name), source,
                              multiplicity=parse_multiplicity(a["source_multiplicity"]))
        target_end = Property(end_name(source.name, target.name), target,
                              multiplicity=parse_multiplicity(a["target_multiplicity"]),
                              is_composite=bool(a.get("composition")))
        associations.add(BinaryAssociation(a["name"], ends={source_end, target_end}))

    generalizations = {Generalization(general=classes[g["parent"]], specific=classes[g["child"]])
                       for g in spec.get("generalizations") or []}
    return DomainModel(sanitize_name(name), types={*classes.values(), *enumerations.values()},
                       associations=associations, generalizations=generalizations)


def proposal_from_plan(plan: Plan) -> Optional[dict[str, Any]]:
    """The editor's ``modelProposal`` for ``plan``, or None when the plan has no model.

    A model that cannot be converted is not raised: the reason is added to the plan's questions, so the user
    sees it with the plan, and None is returned.
    """
    if not plan.model_spec:
        return None
    try:
        domain_model = model_spec_to_domain_model(plan.model_spec, plan.title)
        model = {**class_buml_to_json(domain_model), "type": "ClassDiagram"}
    except SPEC_ERRORS as exc:
        logger.warning("BESSER Lab: the plan's class diagram could not be converted: %s", exc)
        plan.questions.append(f"The proposed class diagram could not be shown in the editor ({exc}); "
                              "describe the model again or answer with the corrections.")
        return None
    return {"title": plan.title, "diagramType": "ClassDiagram", "model": model}


def domain_model_to_spec(domain_model: DomainModel) -> dict[str, Any]:
    """``domain_model`` in the ``model_spec`` shape, every list sorted by name."""
    def attributes(cls: Class) -> list[dict[str, str]]:
        return [{"name": p.name, "type": p.type.name} for p in sorted(cls.attributes, key=lambda p: p.name)]

    associations = []
    for association in sorted(domain_model.associations, key=lambda a: a.name):
        ends = sort_by_timestamp(association.ends)
        if len(ends) != 2:
            continue  # n-ary associations have no place in a model_spec
        source, target = ends
        if source.is_composite and not target.is_composite:
            source, target = target, source
        associations.append({
            "name": association.name, "source": source.type.name, "target": target.type.name,
            "source_multiplicity": format_multiplicity_label(source.multiplicity),
            "target_multiplicity": format_multiplicity_label(target.multiplicity),
            "composition": bool(target.is_composite),
        })
    return {
        "classes": [{"name": c.name, "attributes": attributes(c), "is_abstract": bool(c.is_abstract)}
                    for c in sorted(domain_model.get_classes(), key=lambda c: c.name)],
        "enumerations": [{"name": e.name, "literals": sorted(literal.name for literal in e.literals)}
                         for e in sorted(domain_model.get_enumerations(), key=lambda e: e.name)],
        "associations": associations,
        "generalizations": sorted(({"child": g.specific.name, "parent": g.general.name}
                                   for g in domain_model.generalizations),
                                  key=lambda g: (g["child"], g["parent"])),
    }


def _change_diagram(project: ProjectInput, change_id: str) -> Optional[DiagramInput]:
    """The project's class diagram of change ``change_id``: the active one when it is the change's, else the last.

    The editor marks the proposal tab with ``config.labProposalChangeId`` and a kept proposal with
    ``config.labChangeId``. Any other class diagram is the user's own model, never the change's.
    """
    diagrams = [d for d in project.diagrams.get("ClassDiagram", [])
                if change_id in ((d.config or {}).get(key) for key in _CHANGE_MARKERS)]
    active = project.get_active_diagram("ClassDiagram")
    if any(diagram is active for diagram in diagrams):
        return active
    return diagrams[-1] if diagrams else None


def model_spec_from_project(project: ProjectInput, change_id: str) -> Optional[dict[str, Any]]:
    """The class diagram of change ``change_id`` in the ``model_spec`` shape; None without one or without classes.

    Raises:
        ConversionError, KeyError, TypeError, ValueError: the diagram cannot be converted.
    """
    diagram = _change_diagram(project, change_id)
    if diagram is None or not diagram.model:
        return None
    spec = domain_model_to_spec(process_class_diagram(diagram.model_dump()))
    return spec if spec["classes"] else None
