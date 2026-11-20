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


def ts_field(field: dict) -> str:
    """The TypeScript type of a contract field (list, link payload and ``null`` included)."""
    value = ts_type(field["type"])
    if field.get("link"):
        value = f"{value} | {field['link']}Link"
    if field.get("many"):
        value = f"({value})[]" if " " in value else f"{value}[]"
    return f"{value} | null" if field.get("nullable") else value


def ts_path(path: str) -> str:
    """A route path as a TypeScript template literal, each path parameter URL-encoded."""
    return "`" + _PATH_PARAM_RE.sub(lambda m: "${seg(" + m.group(1) + ")}", path) + "`"


def api_name(entity_name: str) -> str:
    """Name of a class's API object: ``Task`` -> ``taskApi``."""
    return entity_name[:1].lower() + entity_name[1:] + "Api"


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
    return {"api_contract": contract, "api_routes": by_entity}


def register_api_client_filters(env: Environment) -> None:
    env.filters.update(ts_type=ts_type, ts_field=ts_field, ts_path=ts_path, api_name=api_name)
