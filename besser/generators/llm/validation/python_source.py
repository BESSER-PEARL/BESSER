"""Source-only Python contract checks, independent of run orchestration."""

import os
import re as _re

from besser.generators.llm.checkpoint import _SNAPSHOT_DIR


_CREATE_MODEL_RE = _re.compile(r"^class\s+(\w+Create)\s*\(([^)]*)\)\s*:", _re.M)
_CREATE_FIELD_RE = _re.compile(r"^\s{4}(\w+)\s*:", _re.M)
_ROUTER_READ_RE = _re.compile(r"\b(\w+)_data\.(\w+)\b")


def _create_schema_router_mismatches(output_dir: str) -> list[str]:
    """Fields a router reads off a create schema that does not define them.

    Every occurrence is a guaranteed 500 on that endpoint, and the shape
    recurs because two authors own the two halves: the deterministic
    generator writes the router, the LLM edits the schema, and nothing
    reconciles them. Three instances on 2026-09-17 alone --
    ``createdAt``/``updatedAt`` read off a schema that excludes them, a
    1:1 relationship field, and finally ``PersonCreate`` losing
    ``lastName`` when the model rewrote the class to add an email
    validator. That last one returned 500 on person, guest AND employee,
    which is every way to get a row into the system.

    Purely structural, so it cannot fire on a schema that merely looks
    unusual: the field is either declared on the class (or one of its
    Create bases) or it is not.
    """
    root = os.path.join(output_dir, "")
    schema_bodies: dict[str, str] = {}
    schema_bases: dict[str, list[str]] = {}
    for path in _python_files(output_dir):
        try:
            text = open(path, "r", encoding="utf-8").read()
        except OSError:
            continue
        for match in _CREATE_MODEL_RE.finditer(text):
            name = match.group(1)
            bases = [b.strip() for b in match.group(2).split(",")
                     if b.strip().endswith("Create")]
            end = text.find("\nclass ", match.end())
            schema_bodies[name] = text[match.end(): end if end != -1 else len(text)]
            schema_bases[name] = bases
    if not schema_bodies:
        return []

    def declared(name: str, seen: frozenset = frozenset()) -> set:
        if name in seen or name not in schema_bodies:
            return set()
        fields = set(_CREATE_FIELD_RE.findall(schema_bodies[name]))
        for base in schema_bases.get(name, ()):
            fields |= declared(base, seen | {name})
        return fields

    lowered = {n.lower(): n for n in schema_bodies}
    problems: list[str] = []
    seen_pairs: set = set()
    for path in _python_files(output_dir):
        rel = os.path.relpath(path, root).replace("\\", "/")
        if "/routers/" not in f"/{rel}" and not rel.startswith("routers/"):
            continue
        try:
            text = open(path, "r", encoding="utf-8").read()
        except OSError:
            continue
        for match in _ROUTER_READ_RE.finditer(text):
            entity, field = match.group(1), match.group(2)
            schema = lowered.get(f"{entity}create")
            if not schema:
                continue
            if field in declared(schema):
                continue
            key = (schema, field)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            problems.append(
                f"data contract: {rel} "
                f"line {text.count(chr(10), 0, match.start()) + 1} "
                f"reads `{entity}_data.{field}` but "
                f"{schema} does not define `{field}` - this endpoint "
                f"returns 500 on every request"
            )
    return problems


def _python_files(output_dir: str) -> list[str]:
    """Generated .py files, skipping snapshots and vendored trees."""
    try:
        return [
            os.path.join(root, name)
            for root, dirs, files in os.walk(output_dir)
            for name in files
            if name.endswith(".py")
            if not any(part in ("node_modules", _SNAPSHOT_DIR, "__pycache__")
                       for part in root.split(os.sep))
        ]
    except OSError:
        return []
