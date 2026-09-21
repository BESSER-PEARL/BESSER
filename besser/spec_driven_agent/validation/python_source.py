"""Source-only Python contract checks, independent of run orchestration."""

import ast
import os
import re as _re

from besser.spec_driven_agent.checkpoint import _SNAPSHOT_DIR


# Kept only because ``orchestrator`` re-exports them for external callers.
# The check itself reads the syntax tree; these text patterns no longer
# decide anything.
_CREATE_MODEL_RE = _re.compile(r"^class\s+(\w+Create)\s*\(([^)]*)\)\s*:", _re.M)
_CREATE_FIELD_RE = _re.compile(r"^\s{4}(\w+)\s*:", _re.M)
_ROUTER_READ_RE = _re.compile(r"\b(\w+)_data\.(\w+)\b")


# Bases that declare no fields of their own. Any other base has to resolve to a
# class we can read, or the schema is unknown and nothing is reported about it.
_FIELDLESS_BASES = frozenset({
    "ABC", "BaseModel", "BaseSettings", "Generic", "TypedDict", "object",
})
_PAYLOAD_SUFFIX = "_data"


def _base_name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def _own_fields(node: ast.ClassDef) -> set:
    """Names bound in the class body itself, at whatever indent it uses."""
    fields = set()
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            fields.add(statement.target.id)
        elif isinstance(statement, ast.Assign):
            fields.update(t.id for t in statement.targets if isinstance(t, ast.Name))
    return fields


def _class_index(output_dir: str) -> dict:
    """class name -> (fields declared here, base class names)."""
    classes: dict = {}
    for path in _python_files(output_dir):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            fields = _own_fields(node)
            bases = [_base_name(base) for base in node.bases]
            if node.name in classes:
                # Redefined (a duplicated block, or two modules): take the
                # union, never the narrower of the two.
                known_fields, known_bases = classes[node.name]
                classes[node.name] = (known_fields | fields, known_bases + bases)
            else:
                classes[node.name] = (fields, bases)
    return classes


def _declared(name: str, classes: dict, seen: frozenset = frozenset()):
    """Every field ``name`` accepts, or None when a base cannot be read.

    Returning None matters: a base the workspace does not define could
    declare the field, so "absent" is unprovable and nothing is reported.
    """
    if name in seen:
        return set()
    entry = classes.get(name)
    if entry is None:
        return None
    fields, bases = entry
    accepted = set(fields)
    for base in bases:
        if base is None:
            return None
        if base in _FIELDLESS_BASES:
            continue
        inherited = _declared(base, classes, seen | {name})
        if inherited is None:
            return None
        accepted |= inherited
    return accepted


def _annotated_payloads(scope) -> dict:
    """``{parameter name: annotation}`` for this function's arguments.

    The ``<entity>_data`` naming convention was the only way a payload was
    recognised, so a handler written ``payload: BookingCreate`` was invisible
    to this check however wrong its reads were. The annotation is the better
    signal anyway: it NAMES the schema instead of guessing it from a
    variable, so it also cannot mis-resolve when the two disagree.
    """
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    args = scope.args
    annotated = {}
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        name = getattr(arg.annotation, "id", None)
        if name:
            annotated[arg.arg] = name
    return annotated


def _unguarded_payload_reads(tree: ast.AST) -> list:
    """``(variable, annotation, field, line)`` for payload attribute reads.

    ``annotation`` is the declared type when the handler annotated the
    parameter, and ``None`` when the receiver was recognised only by the
    ``_data`` suffix; the caller resolves the schema from whichever it has.

    A read the handler guards with ``hasattr``/``getattr`` on the same
    attribute is skipped: the author already handles the field being
    absent, so the access cannot raise. Live tree ``...-d71pocck`` shipped
    ``booking_data.id if hasattr(booking_data, 'id') ... else True`` and
    served 11/11 workflow checks.
    """
    reads: list = []
    examined: set = set()
    scopes = [node for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scopes.append(tree)
    for scope in scopes:
        annotated = _annotated_payloads(scope)
        guarded = {
            (node.args[0].id, node.args[1].value)
            for node in ast.walk(scope)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("hasattr", "getattr") and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        }
        for node in ast.walk(scope):
            if not (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)):
                continue
            receiver = node.value.id
            annotation = annotated.get(receiver)
            if not (receiver.endswith(_PAYLOAD_SUFFIX) or annotation):
                continue
            position = (node.lineno, node.col_offset)
            if position in examined:
                continue
            examined.add(position)
            if (receiver, node.attr) in guarded:
                continue
            reads.append((receiver, annotation, node.attr, node.lineno))
    return reads


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
    bases) or it is not. Both halves are read from the parsed syntax tree
    rather than matched in the raw text, so a field at an unusual indent
    counts as declared and an access named only in a comment or a string
    is not a read at all.
    """
    classes = _class_index(output_dir)
    create_schemas = {name.lower(): name for name in classes
                      if name.endswith("Create")}
    if not create_schemas:
        return []

    problems: list[str] = []
    seen_pairs: set = set()
    for path in _python_files(output_dir):
        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        if "/routers/" not in f"/{rel}" and not rel.startswith("routers/"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for receiver, annotation, field, line in _unguarded_payload_reads(tree):
            # An annotation names the schema outright; the ``_data`` suffix
            # only guesses it. Prefer the annotation, and fall back so the
            # unannotated handlers this check was written for still work.
            schema = None
            if annotation and annotation in classes and annotation.endswith("Create"):
                schema = annotation
            elif receiver.endswith(_PAYLOAD_SUFFIX):
                entity = receiver[:-len(_PAYLOAD_SUFFIX)]
                schema = create_schemas.get(f"{entity.lower()}create")
            if not schema:
                continue
            accepted = _declared(schema, classes)
            if accepted is None or field in accepted:
                continue
            key = (schema, field)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            problems.append(
                f"data contract: {rel} "
                f"line {line} "
                f"reads `{receiver}.{field}` but "
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
