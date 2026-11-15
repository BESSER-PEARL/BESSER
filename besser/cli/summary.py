"""What a B-UML model holds, in short: the data and the lines of ``besser info``."""

from __future__ import annotations

import re
from typing import Any, Callable

from besser.BUML.metamodel.gui import Button, Chart, Link, LookupColumn, Menu, MetricCard, Table
from besser.BUML.metamodel.structural import UNLIMITED_MAX_MULTIPLICITY, DomainModel
from besser.cli.loading import LoadedModel
from besser.utilities import sort_by_timestamp
from besser.utilities.utils import display_order_key


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
    """An end's multiplicity as ``besser model add-association`` takes it: 1, 0..1, *, 1..*."""
    low, high = end.multiplicity.min, end.multiplicity.max
    if high == UNLIMITED_MAX_MULTIPLICITY:
        return "*" if low == 0 else f"{low}..*"
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

    return {cls.name: class_endpoints(cls, model) for cls in sort_by_timestamp(model.get_classes())}


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
                     "default": _plain(attribute.default_value), "derived": bool(attribute.is_derived)}
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


def _view_elements(container: Any) -> list[Any]:
    """The view elements of a screen or container, at any depth, in name order."""
    found = []
    for element in sorted(getattr(container, "view_elements", None) or (), key=lambda e: e.name):
        found.append(element)
        found.extend(_view_elements(element))
    return found


def _column(column: Any) -> str:
    """A table column as ``add-column`` takes it: an attribute, or ``end.attribute`` of a related object."""
    if getattr(column, "field", None) is None:
        return column.label
    path = getattr(column, "path", None) if isinstance(column, LookupColumn) else None
    return f"{_name(path)}.{_name(column.field)}" if path is not None else _name(column.field)


def _computed(element: Any) -> str:
    """What a card or chart shows, as ``add-card`` and ``add-chart`` take it: ``sum:Deal.amount by stage
    where stage == 'WON'``."""
    binding = getattr(element, "data_binding", None)
    if binding is None:
        return "unbound"
    aggregation = getattr(binding, "aggregation", None)
    text = f"{aggregation.name.lower()}:" if aggregation is not None else ""
    text += _name(binding.domain_concept) if binding.domain_concept is not None else "?"
    if getattr(binding, "data_field", None) is not None:
        text += f".{_name(binding.data_field)}"
    if getattr(binding, "label_field", None) is not None:
        text += f" by {_name(binding.label_field)}"
    if getattr(binding, "filter_expression", None):
        text += f" where {binding.filter_expression}"
    return text


def _gui_screens(model: Any) -> list[dict[str, Any]]:
    """Per screen of a GUI model, in the generators' order: its route, whether it is the main page, the
    tables it shows (the class each is bound to and what its columns show), its cards and charts in the
    page's order (their title and what they compute), whether it shows a menu, and the screens its
    buttons open."""
    screens = []
    for module in sorted(model.modules, key=lambda m: m.name):
        for screen in sorted(module.screens, key=lambda s: s.name):
            elements = _view_elements(screen)
            screens.append({
                "name": screen.name,
                "route": screen.route_path,
                "main": bool(screen.is_main_page),
                "tables": [
                    {"name": table.name,
                     "class": _name(getattr(table.data_binding, "domain_concept", None))
                     if getattr(table, "data_binding", None) else None,
                     "columns": [_column(column) for column in table.columns]}
                    for table in elements if isinstance(table, Table)
                ],
                "cards": [{"title": card.metric_title, "shows": _computed(card)}
                          for card in sorted(elements, key=display_order_key) if isinstance(card, MetricCard)],
                "charts": [{"title": chart.title, "kind": type(chart).__name__.removesuffix("Chart").lower(),
                            "shows": _computed(chart)}
                           for chart in sorted(elements, key=display_order_key) if isinstance(chart, Chart)],
                "menu": any(isinstance(element, Menu) for element in elements),
                "links": [{"label": link.label, "url": link.url} for link in elements if isinstance(link, Link)],
                "opens": sorted({element.targetScreen.name for element in elements
                                 if isinstance(element, Button) and element.targetScreen is not None}),
            })
    return screens


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
    if loaded.kind == "gui":
        entry["screens"] = _gui_screens(loaded.model)
    return entry


def attribute_text(attribute: dict[str, Any]) -> str:
    """An attribute as ``besser model add-attribute`` takes it: ``name:type``, ``?`` when optional, ``=default``."""
    text = f"{attribute['name']}:{attribute['type']}{'?' if attribute.get('optional') else ''}"
    text += f"={attribute['default']!r}" if attribute.get("default") is not None else ""
    return text + (" (derived)" if attribute.get("derived") else "")


def end_text(end: dict[str, Any]) -> str:
    """An association end as ``besser model add-association`` takes it: ``role:Class[multiplicity]``."""
    return f"{end['name']}:{end['type']}[{end['multiplicity']}]"


_OCL_HEADER = re.compile(r"^\s*context\s+\w+\s+inv\b[^:]*:\s*", re.IGNORECASE)


def invariant_body(expression: str) -> str:
    """An OCL invariant without its ``context X inv name:`` header, which ``info`` prints already."""
    return _OCL_HEADER.sub("", expression, count=1)


# The routes of a plain class X, as ``compact_routes`` writes them; a class with other routes gets its own line.
_PLAIN_ROUTES = ("POST|GET /x/, GET|PUT|DELETE /x/{x_id}/, POST|DELETE /x/bulk/, GET /x/count/, GET /x/paginated/, "
                 "GET /x/search/")


def compact_routes(routes: list[str]) -> list[str]:
    """``routes`` with the methods of one path joined, in first-seen order: ``POST|GET /book/``."""
    verbs: dict[str, list[str]] = {}
    for route in routes:
        verb, path = route.split(" ", 1)
        verbs.setdefault(path, []).append(verb)
    return [f"{'|'.join(methods)} {path}" for path, methods in verbs.items()]


# A domain model with more classes than this is summarised by 'besser info' as an overview (names only):
# its full report grows with it, and whoever changes one class reads it all in every session.
OVERVIEW_OVER = 20


def screen_text(screen: dict[str, Any]) -> str:
    """What a screen shows, as the report says it: its tables, cards, charts, links, menu and buttons."""
    shown = [f"{table['class'] or 'unbound'} table {table['name']} ({', '.join(table['columns'])})"
             for table in screen["tables"]]
    shown += [f"card '{card['title']}' ({card['shows']})" for card in screen["cards"]]
    shown += [f"{chart['kind']} chart '{chart['title']}' ({chart['shows']})" for chart in screen["charts"]]
    shown += [f"link '{link['label']}' ({link['url']})" for link in screen["links"]]
    if screen["menu"]:
        shown.append("menu")
    if screen["opens"]:
        shown.append(f"buttons to {', '.join(screen['opens'])}")
    main = " (main page)" if screen["main"] else ""
    return f"screen {screen['name']}{main}, {screen['route']}: {'; '.join(shown) or 'no table'}"


def summary_lines(entry: dict[str, Any], overview: bool = False) -> list[str]:
    """The lines of the text report for one ``describe_model`` entry. ``overview``: the names only (the
    classes and enumerations of a domain model, the screens of a GUI model with their routes)."""
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
    if overview and entry.get("kind") in ("domain", "gui"):
        for label, key in (("classes", "classes"), ("enumerations", "enumerations")):
            if entry.get(key):
                lines.append(f"    {label}: {', '.join(item['name'] for item in entry[key])}")
        if entry.get("screens"):
            lines.append("    screens: " + ", ".join(
                f"{screen['name']} {screen['route']}{' (main page)' if screen['main'] else ''}"
                for screen in entry["screens"]))
        return lines
    for collection, names in entry.get("elements", {}).items():
        if collection == "screens" and "screens" in entry:
            continue  # a line per screen, below
        lines.append(f"    {collection.replace('_', ' ')}: {', '.join(names)}")
    for screen in entry.get("screens", []):
        lines.append(f"    {screen_text(screen)}")
    for cls in entry.get("classes", []):
        header = f"{'abstract ' if cls['is_abstract'] else ''}class {cls['name']}"
        if cls["parents"]:
            header += f"({', '.join(cls['parents'])})"
        members = [attribute_text(attribute) for attribute in cls["attributes"]]
        members.extend(f"{method}()" for method in cls["methods"])
        lines.append(f"    {header}{': ' + ', '.join(members) if members else ''}")
    for enumeration in entry.get("enumerations", []):
        lines.append(f"    enumeration {enumeration['name']}: {', '.join(enumeration['literals'])}")
    for association in entry.get("associations", []):
        lines.append(f"    association {association['name']}: {' '.join(map(end_text, association['ends']))}")
    for constraint in entry.get("constraints", []):
        lines.append(f"    constraint {constraint['name']} on {constraint['context']}: "
                     f"{invariant_body(constraint['expression'])}")
    if entry.get("backend_endpoints"):
        lines.append(f"    backend routes per class X: {_PLAIN_ROUTES}")
        for cls, routes in entry["backend_endpoints"].items():
            crud = ", ".join(compact_routes([route for route in routes if "/methods/" not in route]))
            if crud != _PLAIN_ROUTES.replace("/x/", f"/{cls.lower()}/").replace("{x_id}", f"{{{cls.lower()}_id}}"):
                lines.append(f"    {cls} routes: {crud}")
            methods = [route for route in routes if "/methods/" in route]
            if methods:
                lines.append(f"    {cls} methods: {', '.join(methods)}")
    return lines


def _signature(method: Any) -> str:
    parameters = ", ".join(f"{parameter.name}:{_name(parameter.type)}" for parameter in method.parameters)
    returns = f" -> {_name(method.type)}" if method.type is not None else ""
    return f"{method.name}({parameters}){returns}"


def describe_class(model: DomainModel, cls: Any) -> dict[str, Any]:
    """One class as ``besser info --class`` shows it: its members, inherited ones, association ends,
    constraints, and what the backend generator writes for it."""
    from besser.generators.backend.endpoints import class_endpoints

    association_of = {id(end): association.name for association in model.associations for end in association.ends}
    parents = sort_by_timestamp(cls.all_parents())
    inherited = [attribute for parent in parents for attribute in sort_by_timestamp(parent.attributes)]
    return {
        "name": cls.name,
        "is_abstract": cls.is_abstract,
        "parents": sorted(_name(parent) for parent in cls.parents()),
        "attributes": [attribute_text(_attribute(attribute)) for attribute in sort_by_timestamp(cls.attributes)],
        "inherited_attributes": [attribute_text(_attribute(attribute)) for attribute in inherited],
        "methods": [_signature(method) for method in sort_by_timestamp(cls.methods)],
        "ends": [f"{end_text({'name': end.name, 'type': _name(end.type), 'multiplicity': _multiplicity(end)})} "
                 f"({association_of.get(id(end), '?')})"
                 for end in sorted(cls.association_ends(), key=lambda end: end.name)],
        "inherited_ends": [f"{end_text({'name': end.name, 'type': _name(end.type), 'multiplicity': _multiplicity(end)})} "
                           f"({association_of.get(id(end), '?')}, from {parent.name})"
                           for parent in parents for end in sorted(parent.association_ends(), key=lambda end: end.name)],
        "constraints": [f"{constraint.name}: {invariant_body(constraint.expression)}"
                        for constraint in sorted(model.constraints, key=lambda c: c.name) if constraint.context is cls],
        "backend_endpoints": class_endpoints(cls, model),
        "backend_code": [f"sql_alchemy.{cls.name} (table {cls.name.lower()})", f"pydantic_classes.{cls.name}Create",
                         "database.get_db"],
    }


def _attribute(attribute: Any) -> dict[str, Any]:
    return {"name": attribute.name, "type": _name(attribute.type), "optional": attribute.is_optional,
            "default": _plain(attribute.default_value), "derived": bool(attribute.is_derived)}


def class_lines(entry: dict[str, Any]) -> list[str]:
    """The lines of ``besser info --class`` for one ``describe_class`` entry."""
    header = f"{'abstract ' if entry['is_abstract'] else ''}class {entry['name']}"
    if entry["parents"]:
        header += f"({', '.join(entry['parents'])})"
    lines = [header]
    for label, key in (("attributes", "attributes"), ("inherited", "inherited_attributes"), ("methods", "methods"),
                       ("ends", "ends"), ("inherited ends", "inherited_ends"), ("constraints", "constraints"),
                       ("backend routes", "backend_endpoints"), ("backend code", "backend_code")):
        values = compact_routes(entry[key]) if key == "backend_endpoints" else entry[key]
        if values:
            lines.append(f"  {label}: {', '.join(values)}")
    lines.extend(f"  page: {page}" for page in entry.get("pages", []))
    return lines


def class_pages(gui_model: Any, class_name: str) -> list[str]:
    """The screens of a GUI model that show the class ``class_name`` in a table, as the report says them."""
    return [screen_text(screen) for screen in _gui_screens(gui_model)
            if any(table["class"] == class_name for table in screen["tables"])]
