"""What a B-UML model holds, in short: the data and the lines of ``besser info``."""

from __future__ import annotations

from typing import Any, Callable

from besser.BUML.metamodel.structural import UNLIMITED_MAX_MULTIPLICITY, DomainModel
from besser.cli.loading import LoadedModel
from besser.utilities import sort_by_timestamp


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


# Model kind -> the collections of a model of that kind that are counted, and listed by name
# (a feature model is a tree: nothing to count).
_COLLECTIONS: dict[str, tuple[str, ...]] = {
    "object": ("objects", "links"),
    "agent": ("states", "intents"),
    "state_machine": ("states",),
    "quantum": ("qregs", "cregs", "operations"),
    "nn": ("layers", "sub_nns", "tensor_ops"),
    "bpmn": ("processes",),
    "deployment": ("clusters",),
    "uml_deployment": ("nodes", "artifacts"),
    "component": ("components", "interfaces"),
    "kg": ("nodes", "edges"),
}
# Model kind -> what is counted when it is not the collections above.
_COUNTS: dict[str, Callable[[Any], dict[str, int]]] = {"domain": _domain_counts, "gui": _gui_counts}
# A collection longer than this is counted and not listed: a knowledge graph has thousands of nodes.
_MAX_ELEMENTS_LISTED = 50


def _counts(kind: str, model: Any) -> dict[str, int]:
    if kind in _COUNTS:
        return _COUNTS[kind](model)
    return {attribute: len(getattr(model, attribute)) for attribute in _COLLECTIONS.get(kind, ())}


def _elements(kind: str, model: Any) -> dict[str, list[str]]:
    """The names of what the model holds, per collection whose elements all have one."""
    if kind == "gui":
        collections = {"modules": list(model.modules),
                       "screens": [screen for module in model.modules for screen in module.screens]}
    else:
        collections = {attribute: list(getattr(model, attribute)) for attribute in _COLLECTIONS.get(kind, ())}
    elements = {}
    for attribute, items in collections.items():
        names = [getattr(item, "name", None) for item in items]
        if names and len(names) <= _MAX_ELEMENTS_LISTED and all(isinstance(name, str) and name for name in names):
            # A set has no order of its own: the names are sorted so that two runs print the same.
            elements[attribute] = names if isinstance(getattr(model, attribute, None), list) else sorted(names)
    return elements


# The singular of each counted name that is not the name without its "s".
_SINGULAR = {"classes": "class", "processes": "process"}


def _name(element: Any) -> str:
    """An element's name; a type given as text (an invalid model) is that text."""
    return getattr(element, "name", None) or str(element)


def _multiplicity(end: Any) -> str:
    low, high = end.multiplicity.min, end.multiplicity.max
    high = "*" if high == UNLIMITED_MAX_MULTIPLICITY else high
    return str(low) if low == high else f"{low}..{high}"


def _plain(value: Any) -> Any:
    """``value`` as JSON holds it: a default that is a date or an enumeration literal becomes its text."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "name", None) or str(value)


def _endpoints(model: DomainModel) -> dict[str, list[str]]:
    """Per class, the routes the backend generator writes for it."""
    # Imported here: the backend generator's templates are only needed for a domain model.
    from besser.generators.backend.endpoints import class_endpoints

    return {cls.name: class_endpoints(cls) for cls in sort_by_timestamp(model.get_classes())}


def _domain_details(model: DomainModel) -> dict[str, Any]:
    """The classes, enumerations, associations and constraints of a domain model, in the order they were
    defined, and the routes the backend generator writes for each class."""
    return {
        "classes": [
            {
                "name": cls.name,
                "is_abstract": cls.is_abstract,
                "parents": sorted(_name(parent) for parent in cls.parents()),
                "attributes": [
                    {"name": attribute.name, "type": _name(attribute.type), "optional": attribute.is_optional,
                     "default": _plain(attribute.default_value)}
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
        "constraints": [
            {"name": constraint.name, "context": _name(constraint.context), "expression": constraint.expression}
            for constraint in sorted(model.constraints, key=lambda constraint: constraint.name)
        ],
        "backend_endpoints": _endpoints(model),
    }


def describe_model(loaded: LoadedModel, current: bool = False) -> dict[str, Any]:
    """One model as ``besser info --json`` shows it: what it is, what it holds, and a domain model's content.

    ``current``: the model is that of the diagram the editor project marks as current, among
    several of its kind; generators read it when ``--use`` names none.
    """
    entry = {
        "name": loaded.name,
        "kind": loaded.kind,
        "type": type(loaded.model).__name__,
        "model_name": getattr(loaded.model, "name", None),
        "current": current,
        "counts": _counts(loaded.kind, loaded.model),
    }
    if loaded.kind == "domain":
        entry.update(_domain_details(loaded.model))
    else:
        entry["elements"] = _elements(loaded.kind, loaded.model)
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
    title += ")"
    if entry.get("current"):
        title += " (current in the editor)"
    lines = [f"  {title}{': ' + counts if counts else ''}"]
    for collection, names in entry.get("elements", {}).items():
        lines.append(f"    {collection.replace('_', ' ')}: {', '.join(names)}")
    for cls in entry.get("classes", []):
        header = f"{'abstract ' if cls['is_abstract'] else ''}class {cls['name']}"
        if cls["parents"]:
            header += f"({', '.join(cls['parents'])})"
        members = [
            f"{attribute['name']}: {attribute['type']}{'?' if attribute.get('optional') else ''}"
            + (f" = {attribute['default']!r}" if attribute.get("default") is not None else "")
            for attribute in cls["attributes"]
        ]
        members.extend(f"{method}()" for method in cls["methods"])
        lines.append(f"    {header}{': ' + ', '.join(members) if members else ''}")
    for enumeration in entry.get("enumerations", []):
        lines.append(f"    enumeration {enumeration['name']}: {', '.join(enumeration['literals'])}")
    for association in entry.get("associations", []):
        ends = " -- ".join(f"{end['name']}: {end['type']} [{end['multiplicity']}]" for end in association["ends"])
        lines.append(f"    association {association['name']}: {ends}")
    for constraint in entry.get("constraints", []):
        lines.append(f"    constraint {constraint['name']} on {constraint['context']}: {constraint['expression']}")
    if entry.get("backend_endpoints"):
        lines.append("    backend routes per class X: POST|GET /x/, GET|PUT|DELETE /x/{x_id}/, "
                     "/x/bulk|count|paginated|search/")
        for cls, routes in entry["backend_endpoints"].items():
            methods = [route for route in routes if "/methods/" in route]
            if methods:
                lines.append(f"    {cls} methods: {', '.join(methods)}")
    return lines
