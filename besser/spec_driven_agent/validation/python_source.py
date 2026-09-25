"""Source-only Python contract checks, independent of run orchestration."""

import ast
import os
import re as _re

from besser.spec_driven_agent.state.checkpoint import _SNAPSHOT_DIR
from besser.spec_driven_agent.parsed_source import parse_source


# Kept only because ``orchestrator`` re-exports them for external callers.
# The check itself reads the syntax tree, not these text patterns.
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
                tree = parse_source(handle.read())
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


_OPTIONAL_WRAPPERS = frozenset({"Optional"})
_SEQUENCE_WRAPPERS = frozenset({"list", "List", "Sequence"})


def _unwrap(annotation, wrappers) -> str | None:
    """``X`` from ``wrapper[X]``, or None."""
    if (isinstance(annotation, ast.Subscript)
            and _base_name(annotation.value) in wrappers
            and isinstance(annotation.slice, ast.Name)):
        return annotation.slice.id
    return None


def _payload_type(annotation) -> str | None:
    """``X`` from ``X``, ``Optional[X]`` or ``X | None``."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    if (isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr)
            and isinstance(annotation.right, ast.Constant)
            and annotation.right.value is None):
        return _payload_type(annotation.left)
    return _unwrap(annotation, _OPTIONAL_WRAPPERS)


def _annotated_payloads(scope) -> dict:
    """``{parameter name: annotation}`` for this function's arguments.

    Complements the ``<entity>_data`` naming convention, so a handler written
    ``payload: BookingCreate`` is checked too. The annotation is the better
    signal: it NAMES the schema instead of guessing it from a variable, so it
    cannot mis-resolve when the two disagree.
    """
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    args = scope.args
    annotated = {}
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        name = _payload_type(arg.annotation)
        if name:
            annotated[arg.arg] = name
    return annotated


def _loop_payload_reads(scope) -> dict:
    """``{(line, col): element type}`` for reads on a loop over a list payload.

    ``items: list[BillCreate]`` iterated as ``for item in items`` or
    ``for i, item in enumerate(items)``: reads on ``item`` inside that loop
    body are reads on a ``BillCreate``. Bound per loop, so the same name in
    another loop is not affected.
    """
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    args = scope.args
    elements = {}
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        name = _unwrap(arg.annotation, _SEQUENCE_WRAPPERS)
        if name:
            elements[arg.arg] = name
    if not elements:
        return {}
    bound = {}
    for loop in ast.walk(scope):
        if not isinstance(loop, (ast.For, ast.AsyncFor)):
            continue
        iterable, target = loop.iter, loop.target
        if (isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name)
                and iterable.func.id == "enumerate" and len(iterable.args) == 1
                and isinstance(target, ast.Tuple) and len(target.elts) == 2):
            iterable, target = iterable.args[0], target.elts[1]
        if not (isinstance(iterable, ast.Name) and iterable.id in elements
                and isinstance(target, ast.Name)):
            continue
        for statement in loop.body:
            for node in ast.walk(statement):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == target.id):
                    bound[(node.lineno, node.col_offset)] = elements[iterable.id]
    return bound


def _unguarded_payload_reads(tree: ast.AST) -> list:
    """``(variable, annotation, field, line)`` for payload attribute reads.

    ``annotation`` is the declared type when the handler annotated the
    parameter (the element type for an item of a ``list[X]`` one), and
    ``None`` when the receiver was recognised only by the
    ``_data`` suffix; the caller resolves the schema from whichever it has.

    A read the handler guards with ``hasattr``/``getattr`` on the same
    attribute is skipped: the author already handles the field being
    absent, so the access cannot raise (e.g.
    ``booking_data.id if hasattr(booking_data, 'id') ... else True``).
    """
    reads: list = []
    examined: set = set()
    scopes = [node for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    scopes.append(tree)
    for scope in scopes:
        annotated = _annotated_payloads(scope)
        loop_reads = _loop_payload_reads(scope)
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
            position = (node.lineno, node.col_offset)
            annotation = annotated.get(receiver) or loop_reads.get(position)
            if not (receiver.endswith(_PAYLOAD_SUFFIX) or annotation):
                continue
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
    reconciles them (e.g. ``createdAt``/``updatedAt`` read off a schema
    that excludes them, or ``PersonCreate`` losing ``lastName`` when the
    model rewrote the class to add a validator).

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
                tree = parse_source(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for receiver, annotation, field, line in _unguarded_payload_reads(tree):
            # An annotation names the schema outright; the ``_data`` suffix
            # only guesses it. Prefer the annotation, and fall back so
            # unannotated handlers are still checked.
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
