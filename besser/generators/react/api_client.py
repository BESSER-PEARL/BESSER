"""The typed API client of the generated frontend (``src/api/generated/``).

``types.ts`` and ``client.ts`` are rendered from ``build_api_contract`` (the backend's own route
list, also written as ``api_contract.json``), so the client calls exactly the routes the backend
serves. These filters turn the contract's model-level type names into TypeScript.
"""

from __future__ import annotations

import re

from jinja2 import Environment

from besser.BUML.metamodel.structural import DomainModel
from besser.generators.backend.api_contract import build_api_contract

_SCALARS = {
    "int": "number", "integer": "number", "float": "number", "timedelta": "number",
    "str": "string", "string": "string", "date": "string", "datetime": "string", "time": "string",
    "bool": "boolean", "boolean": "boolean", "any": "unknown",
}
_PATH_PARAM_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def ts_type(type_name: str) -> str:
    """A contract type as TypeScript: a scalar, or the enum or record type of that name."""
    return _SCALARS.get(type_name, type_name)


def ts_field(field: dict, type_names: dict) -> str:
    """The TypeScript type of a contract field (list, link payload and ``null`` included)."""
    value = ts_type(field["type"])
    if field.get("link"):
        value = f"{value} | {type_names[field['link']]['link']}"
    if field.get("many"):
        value = f"({value})[]" if " " in value else f"{value}[]"
    return f"{value} | null" if field.get("nullable") else value


def ts_path(path: str) -> str:
    """A route path as a TypeScript template literal, each path parameter URL-encoded."""
    return "`" + _PATH_PARAM_RE.sub(lambda m: "${apiSegment(" + m.group(1) + ")}", path) + "`"


def api_name(entity_name: str) -> str:
    """Name of a class's API object: ``Task`` -> ``taskApi``."""
    return entity_name[:1].lower() + entity_name[1:] + "Api"


# Types derived from a class, by role, and the suffix each adds to the class name.
_DERIVED_TYPES = (("create", "Create"), ("detailed", "Detailed"), ("search", "Search"),
                  ("link", "Link"), ("link_attributes", "LinkAttributes"))


def type_names(contract: dict) -> dict:
    """Per class, the names of its types in ``types.ts``, by role (``record``, ``create``, ...).

    A derived name that is already a class or enumeration of the model, or that another class
    derived first, gets the first free number (``TaskCreate2``); classes are taken by name order.
    """
    taken = {entity["name"] for entity in contract["entities"]} | set(contract["enums"])
    names = {}
    for entity in contract["entities"]:
        names[entity["name"]] = {"record": entity["name"]}
        for role, suffix in _DERIVED_TYPES:
            name, number = entity["name"] + suffix, 2
            while name in taken:
                name, number = f"{entity['name']}{suffix}{number}", number + 1
            taken.add(name)
            names[entity["name"]][role] = name
    return names


def api_client_context(model: DomainModel) -> dict:
    """Template context of the client: the contract, and each class's routes grouped by kind."""
    contract = build_api_contract(model)
    by_entity = {entity["name"]: {"crud": {}, "relations": {}, "methods": []} for entity in contract["entities"]}
    for route in contract["routes"]:
        if route["entity"] is None:
            continue
        routes = by_entity[route["entity"]]
        if route["group"] == "crud":
            routes["crud"][route["op"]] = route
        elif route["group"] == "relation":
            routes["relations"].setdefault(route["end"], {})[route["op"]] = route
        else:
            routes["methods"].append(route)
    return {"api_contract": contract, "api_routes": by_entity, "type_names": type_names(contract)}


# Client functions that call the same route as another one.
_SAME_ROUTE = {"listDetailed": "list"}


def client_functions(contract: dict) -> dict:
    """Each function of the generated client by the expression that calls it, with its route.

    ``{"taskApi.create": route, "taskApi.relations.tags.list": route, "taskApi.methods.close": route, ...}``,
    for checks that read code written against the client.
    """
    functions = {}
    for route in contract["routes"]:
        if route["entity"] is None:
            continue
        api = api_name(route["entity"])
        if route["group"] == "crud":
            functions[f"{api}.{route['op']}"] = route
            for alias, op in _SAME_ROUTE.items():
                if op == route["op"]:
                    functions[f"{api}.{alias}"] = route
        elif route["group"] == "relation":
            functions[f"{api}.relations.{route['end']}.{route['op']}"] = route
            if route["op"] == "list" and route["response"].get("links_key"):
                functions[f"{api}.relations.{route['end']}.links"] = route
        else:
            functions[f"{api}.methods.{route['op']}"] = route
    return functions


def register_api_client_filters(env: Environment) -> None:
    env.filters.update(ts_type=ts_type, ts_field=ts_field, ts_path=ts_path, api_name=api_name)
