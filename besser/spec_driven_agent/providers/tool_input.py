"""Coerce tool-call inputs back to the shapes their schema declares.

Models sometimes send an array or object argument as a JSON string, often
wrapping the whole input again: Sonnet 5 did it in 16 of 30 forced
``submit_requirements`` calls (2026-09-24), Haiku 4.5 in 0 of 10. Decoding is
schema-guided and provider-neutral: only a string where the schema allows no
string is touched, and only when it decodes to the declared shape.
"""

import json
from typing import Any

_CONTAINERS = {"array": list, "object": dict}


def _types(schema: dict) -> set[str]:
    declared = schema.get("type")
    types = set(declared if isinstance(declared, list) else [declared] if declared else [])
    for option in (schema.get("anyOf") or []) + (schema.get("oneOf") or []):
        if isinstance(option, dict):
            types |= _types(option)
    return types


def _decode(value: str, types: set[str], key: str | None) -> Any:
    try:
        decoded = json.loads(value)
    except ValueError:
        return None
    # '{"requirements": [...]}' sent as the value of "requirements".
    if ("array" in types and isinstance(decoded, dict) and len(decoded) == 1
            and isinstance(decoded.get(key), list)):
        decoded = decoded[key]
    if any(isinstance(decoded, _CONTAINERS[t]) for t in types & _CONTAINERS.keys()):
        return decoded
    return None


def coerce_to_schema(value: Any, schema: Any, key: str | None = None) -> Any:
    """``value`` with stringified arrays/objects decoded wherever ``schema``
    requires one; everything else is returned unchanged."""
    if not isinstance(schema, dict):
        return value
    types = _types(schema)
    if isinstance(value, str) and "string" not in types and types & _CONTAINERS.keys():
        decoded = _decode(value, types, key)
        if decoded is not None:
            value = decoded
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        return {k: coerce_to_schema(v, properties.get(k), k) for k, v in value.items()}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [coerce_to_schema(v, schema["items"]) for v in value]
    return value


def normalize_tool_blocks(content: list, tools: list[dict] | None) -> list:
    """Coerce every ``tool_use`` block's input against the tool it names."""
    schemas = {t.get("name"): t.get("input_schema") for t in tools or []}
    for block in content or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        schema = schemas.get(getattr(block, "name", None))
        current = getattr(block, "input", None)
        if isinstance(schema, dict) and isinstance(current, dict):
            fixed = coerce_to_schema(current, schema)
            if fixed != current:
                block.input = fixed
    return content
