"""What a B-UML model holds, in short: the data and the lines of ``besser info``."""

from __future__ import annotations

from typing import Any, Callable

from besser.BUML.metamodel.structural import UNLIMITED_MAX_MULTIPLICITY, DomainModel
from besser.cli.loading import LoadedModel
from besser.utilities import sort_by_timestamp


def _count(*attributes: str) -> Callable[[Any], dict[str, int]]:
    """A counter of the model's collections of those names."""
    return lambda model: {attribute: len(getattr(model, attribute)) for attribute in attributes}


def _gui_counts(model: Any) -> dict[str, int]:
    return {"modules": len(model.modules), "screens": sum(len(module.screens) for module in model.modules)}


def _domain_counts(model: DomainModel) -> dict[str, int]:
    return {
        "classes": len(model.get_classes()),
        "enumerations": len(model.get_enumerations()),
        "associations": len(model.associations),
        "generalizations": len(model.generalizations),
        "constraints": len(model.constraints),
    }


# Model kind -> what is counted in a model of that kind (a feature model is a tree: nothing to count).
_COUNTS: dict[str, Callable[[Any], dict[str, int]]] = {
    "domain": _domain_counts,
    "object": _count("objects", "links"),
    "gui": _gui_counts,
    "agent": _count("states", "intents"),
    "state_machine": _count("states"),
    "quantum": _count("qregs", "cregs", "operations"),
    "nn": _count("layers", "sub_nns", "tensor_ops"),
    "bpmn": _count("processes"),
    "deployment": _count("clusters"),
    "uml_deployment": _count("nodes", "artifacts"),
    "component": _count("components", "interfaces"),
    "kg": _count("nodes", "edges"),
}
# The singular of each counted name that is not the name without its "s".
_SINGULAR = {"classes": "class", "processes": "process"}


def _name(element: Any) -> str:
    """An element's name; a type given as text (an invalid model) is that text."""
    return getattr(element, "name", None) or str(element)


def _multiplicity(end: Any) -> str:
    low, high = end.multiplicity.min, end.multiplicity.max
    high = "*" if high == UNLIMITED_MAX_MULTIPLICITY else high
    return str(low) if low == high else f"{low}..{high}"


def _domain_details(model: DomainModel) -> dict[str, Any]:
    """The classes, enumerations and associations of a domain model, in the order they were defined."""
    return {
        "classes": [
            {
                "name": cls.name,
                "is_abstract": cls.is_abstract,
                "parents": sorted(_name(parent) for parent in cls.parents()),
                "attributes": [
                    {"name": attribute.name, "type": _name(attribute.type)}
                    for attribute in sort_by_timestamp(cls.attributes)
                ],
                "methods": [method.name for method in sort_by_timestamp(cls.methods)],
            }
            for cls in sort_by_timestamp(model.get_classes())
        ],
        "enumerations": [
            {"name": enumeration.name, "literals": [literal.name for literal in sort_by_timestamp(enumeration.literals)]}
            for enumeration in sort_by_timestamp(model.get_enumerations())
        ],
        "associations": [
            {
                "name": association.name,
                "ends": [
                    {"name": end.name, "type": _name(end.type), "multiplicity": _multiplicity(end)}
                    for end in sort_by_timestamp(association.ends)
                ],
            }
            for association in sort_by_timestamp(model.associations)
        ],
    }


def describe_model(loaded: LoadedModel) -> dict[str, Any]:
    """One model as ``besser info --json`` shows it: what it is, what it holds, and a domain model's content."""
    entry = {
        "name": loaded.name,
        "kind": loaded.kind,
        "type": type(loaded.model).__name__,
        "model_name": getattr(loaded.model, "name", None),
        "counts": _COUNTS.get(loaded.kind, _count())(loaded.model),
    }
    if loaded.kind == "domain":
        entry.update(_domain_details(loaded.model))
    return entry


def summary_lines(entry: dict[str, Any]) -> list[str]:
    """The lines of the text report for one ``describe_model`` entry."""
    counts = ", ".join(
        f"{count} {_SINGULAR.get(noun, noun[:-1]) if count == 1 else noun}".replace("_", " ")
        for noun, count in entry["counts"].items() if count
    )
    title = f"{entry['name']} ({entry['type']}"
    if entry.get("model_name") and entry["model_name"] != entry["name"]:
        title += f" '{entry['model_name']}'"
    lines = [f"  {title}){': ' + counts if counts else ''}"]
    for cls in entry.get("classes", []):
        header = f"{'abstract ' if cls['is_abstract'] else ''}class {cls['name']}"
        if cls["parents"]:
            header += f"({', '.join(cls['parents'])})"
        members = [f"{attribute['name']}: {attribute['type']}" for attribute in cls["attributes"]]
        members.extend(f"{method}()" for method in cls["methods"])
        lines.append(f"    {header}{': ' + ', '.join(members) if members else ''}")
    for enumeration in entry.get("enumerations", []):
        lines.append(f"    enumeration {enumeration['name']}: {', '.join(enumeration['literals'])}")
    for association in entry.get("associations", []):
        ends = " -- ".join(f"{end['name']}: {end['type']} [{end['multiplicity']}]" for end in association["ends"])
        lines.append(f"    association {association['name']}: {ends}")
    return lines
