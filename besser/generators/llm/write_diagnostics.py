"""Cheap, same-turn diagnostics for LLM-authored file writes.

These checks deliberately avoid importing or executing generated code. They
give the model immediate feedback after ``write_file`` / ``modify_file`` while
the changed region is still in context; the broader Phase-3 sweep remains the
release gate.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import json
import logging
import os
import sqlite3
import sys
import tomllib
from typing import Any


logger = logging.getLogger(__name__)

MAX_WRITE_DIAGNOSTICS = 10


_WARNED_PYFLAKES_MISSING = False


def workspace_uses_sqlite(workspace: str | None) -> bool:
    """Detect the generated database dialect without importing application code."""
    if not workspace:
        return False
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in {
            "node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build",
        } and not d.startswith(".besser_")]
        if "database.py" in files:
            try:
                with open(os.path.join(root, "database.py"), encoding="utf-8-sig") as fh:
                    if "sqlite:" in fh.read():
                        return True
            except OSError:
                continue
    return False


_TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _unreachable_lines(tree: ast.AST) -> set[int]:
    """Lines that follow an unconditional exit in the SAME statement list.

    An undefined name there cannot raise: nothing runs the line. Two apps
    in the 74-app working corpus shipped an orphan block left after a
    ``return`` by a botched edit (``...-3jkm7pib`` reading ``ids``,
    ``...-omtn74nk`` reading ``product_list``), and both were reported as
    blockers claiming a NameError on a tree that passed 10/10 workflow
    checks. Only a terminator at the same nesting level counts, so a
    ``return`` inside an ``if`` leaves the rest of the body live.
    """
    dead: set[int] = set()
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for index, statement in enumerate(block):
                if not isinstance(statement, _TERMINATORS):
                    continue
                for later in block[index + 1:]:
                    end = getattr(later, "end_lineno", None) or later.lineno
                    dead.update(range(later.lineno, end + 1))
                break
    return dead


def _bound_names(statement: ast.stmt) -> list[str]:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [statement.name]
    targets = statement.targets if isinstance(statement, ast.Assign) else (
        [statement.target] if isinstance(statement, ast.AnnAssign) else []
    )
    return [target.id for target in targets if isinstance(target, ast.Name)]


def python_structural_diagnostics(
    tree: ast.Module, *, sqlite: bool = False,
    rel_path: str | None = None, workspace: str | None = None,
) -> list[dict]:
    """Provable declaration errors; no application imports, execution, or guessing.

    SQLite CHECK expressions are compiled against an in-memory table containing
    only the mapped column names. No generated Python or user database is used.

    ``rel_path``/``workspace`` additionally resolve star-imported ORM modules
    to catch ``obj.enum_column != SomeEnum.MEMBER.value`` comparisons; without
    both, that check is skipped rather than guessed at.
    """
    findings: list[dict] = []

    def add(node, code, message):
        finding = _finding("python-contract", message, code=code, line=node.lineno)
        finding["end_line"] = node.end_lineno
        findings.append(finding)

    enums = {}
    enum_bases = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "enum":
            enum_bases.update(a.asname or a.name for a in node.names if a.name in enum_bases)
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if any(getattr(base, "id", getattr(base, "attr", "")) in enum_bases for base in cls.bases):
            if cls in tree.body:
                enums[cls.name] = {name for stmt in cls.body for name in _bound_names(stmt)}
        seen: dict[str, ast.stmt] = {}
        columns = []
        for stmt in cls.body:
            for name in _bound_names(stmt):
                if name in seen:
                    # Property setters and overload signatures intentionally reuse names.
                    decorators = getattr(stmt, "decorator_list", [])
                    intentional = any(
                        (isinstance(d, ast.Attribute) and d.attr in {"setter", "deleter"})
                        or getattr(d, "id", getattr(d, "attr", "")) == "overload"
                        for d in decorators + getattr(seen[name], "decorator_list", [])
                    )
                    if not intentional and (
                        name in {"__table_args__", "__mapper_args__"}
                        or isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ):
                        add(seen[name], "shadowed-declaration", f"{cls.name}.{name} is overwritten by its second declaration at line {stmt.lineno}; merge the declarations instead of silently discarding behavior.")
                        add(stmt, "duplicate-declaration", f"{cls.name}.{name} duplicates line {seen[name].lineno}; only this last declaration takes effect.")
                seen[name] = stmt
            value = getattr(stmt, "value", None)
            if isinstance(value, ast.Call) and getattr(value.func, "id", getattr(value.func, "attr", "")).rstrip("_") in {"mapped_column", "Column"}:
                names = _bound_names(stmt)
                if value.args and isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str):
                    names = [value.args[0].value]
                columns.extend(names)
        # Do not guess inherited columns. A concrete inherited mapping is
        # checked by the isolated application/DDL probe instead.
        local_classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        inherits_columns = any(
            isinstance(base, ast.Name) and base.id in local_classes
            and any(isinstance(n, ast.Call) and getattr(n.func, "id", "").rstrip("_") in {"Column", "mapped_column"}
                    for n in ast.walk(local_classes[base.id]))
            for base in cls.bases
        )
        if sqlite and columns and not inherits_columns:
            table = next((stmt.value.value for stmt in cls.body if isinstance(stmt, ast.Assign)
                          and "__tablename__" in _bound_names(stmt)
                          and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)), cls.name)
            def quote(name):
                return '"' + name.replace('"', '""') + '"'
            definitions = ", ".join(quote(name) + " NUMERIC" for name in dict.fromkeys(columns))
            for call in (n for n in ast.walk(cls) if isinstance(n, ast.Call)
                         and getattr(n.func, "id", getattr(n.func, "attr", "")) == "CheckConstraint"):
                if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
                    continue
                connection = sqlite3.connect(":memory:")
                try:
                    connection.execute(f"CREATE TABLE {quote(table)} ({definitions}, CHECK ({call.args[0].value}))")
                except sqlite3.Error as exc:
                    # Custom SQL functions may be registered by the app; their absence
                    # in this scratch database is not proof of a broken constraint.
                    if "no such function" not in str(exc).lower():
                        add(call, "invalid-sqlite-check", f"SQLite cannot create {cls.name}'s CHECK constraint: {exc}. Cross-row rules must be enforced transactionally in the mutation paths, not in SQLite CHECK subqueries.")
                finally:
                    connection.close()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        members = enums.get(node.value.id)
        if members is not None and node.attr not in members and not node.attr.startswith("_"):
            add(node, "invalid-enum-member", f"{node.value.id}.{node.attr} does not exist; declared members: {', '.join(sorted(members))}. Use an actual member, not an invented spelling.")
    findings.extend(_enum_column_value_misuse(tree, rel_path, workspace))
    findings.extend(_star_import_attribute_misuse(tree, rel_path, workspace))
    return findings


def _finding(
    source: str,
    message: str,
    *,
    code: str,
    line: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "source": source,
        "severity": "error",
        "code": code,
        "message": message,
    }
    if line is not None:
        finding["line"] = line
    if column is not None:
        finding["column"] = column
    return finding


_COROUTINE_SCHEDULERS = {
    "create_task", "ensure_future", "gather", "wait", "wait_for", "run",
    "run_until_complete", "run_coroutine_threadsafe", "shield", "as_completed",
    "to_thread", "start_soon", "spawn",
}


def _is_async_generator(fn: ast.AsyncFunctionDef) -> bool:
    """An ``async def`` that yields returns an iterator, not a coroutine."""
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Yield, ast.YieldFrom)):
            continue
        owner = node
        while owner is not fn:
            owner = _parent_of(fn, owner)
            if owner is None or isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
        if owner is fn:
            return True
    return False


def _parent_of(root: ast.AST, target: ast.AST) -> ast.AST | None:
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            if child is target:
                return node
    return None


def _unawaited_coroutines(tree: ast.Module) -> list[dict[str, Any]]:
    """Calls to a local ``async def`` that are neither awaited nor scheduled.

    Neither ``ast.parse`` nor pyflakes sees this; it surfaces as a 500 at
    runtime (``'coroutine' object is not subscriptable``) once the route is
    actually exercised, which no static gate reaches.
    """
    coroutine_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and not _is_async_generator(node)
    }
    if not coroutine_names:
        return []

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def is_consumed(call: ast.Call) -> bool:
        node: ast.AST = call
        while True:
            parent = parents.get(node)
            if parent is None or isinstance(parent, ast.stmt):
                return False
            if isinstance(parent, ast.Await):
                return True
            if isinstance(parent, ast.Call):
                func = parent.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in _COROUTINE_SCHEDULERS:
                    return True
            node = parent

    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in coroutine_names and not is_consumed(node):
            findings.append(_finding(
                "python",
                f"'{name}' is an async function; calling it without 'await' returns a "
                f"coroutine, not its result. Use 'await {name}(...)', or move the shared "
                f"logic into a plain (non-async) helper both callers use.",
                code="unawaited-coroutine",
                line=node.lineno,
                column=node.col_offset + 1,
            ))
    return findings


def _module_scope_statements(body):
    """Statements bound at module scope: descends into if/for/while/with/try
    blocks (still module scope), never into a def or class body."""
    for node in body:
        yield node
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            yield from _module_scope_statements(node.body)
            yield from _module_scope_statements(getattr(node, "orelse", []))
        elif isinstance(node, ast.Try):
            yield from _module_scope_statements(node.body)
            for handler in node.handlers:
                yield handler
                yield from _module_scope_statements(handler.body)
            yield from _module_scope_statements(node.orelse)
            yield from _module_scope_statements(node.finalbody)


def _resolve_module(module: str, start_dir: str, root: str) -> str | None:
    """Path of a local ``module`` as the generated service imports it.

    A service runs with its own folder as cwd (``python main_api.py`` from
    ``backend/``), so ``routers/x.py`` resolves ``sql_alchemy`` from
    ``backend/``: search the importing file's folder, then each ancestor up
    to the workspace root. Same rule as ``_unresolvable_local_imports``.
    """
    rel = module.replace(".", os.sep)
    directory = start_dir
    while True:
        for candidate in (rel + ".py", os.path.join(rel, "__init__.py")):
            path = os.path.join(directory, candidate)
            if os.path.isfile(path):
                return path
        if os.path.normpath(directory) == os.path.normpath(root):
            return None
        parent = os.path.dirname(directory)
        if not parent or parent == directory:
            return None
        directory = parent


def _star_exports(path: str, root: str, visited: set[str]) -> set[str] | None:
    """Names ``from <module> import *`` binds, read off the module's AST.

    ``__all__`` when declared, else every module-scope binding without a
    leading underscore, including names the module imports itself (star
    imports transitively). ``None`` when the module, or one it star-imports,
    is not on disk to read: nothing can be judged against it.
    """
    if path in visited:
        return set()
    visited.add(path)
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None
    names: set[str] = set()
    for node in _module_scope_statements(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            if (any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
                    and isinstance(node.value, (ast.List, ast.Tuple))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                            for e in node.value.elts)):
                return {e.value for e in node.value.elts}
            for target in node.targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names |= {n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                names.add(node.name)
        elif isinstance(node, ast.Import):
            names |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
                    continue
                if node.level or not node.module:
                    return None
                target = _resolve_module(node.module, os.path.dirname(path), root)
                inner = _star_exports(target, root, visited) if target else None
                if inner is None:
                    return None
                names |= inner
    return {name for name in names if not name.startswith("_")}


def _star_import_scope(
    tree: ast.Module, rel_path: str, workspace: str
) -> tuple[list[str], set[str]] | None:
    """The written file's star-imported module names and the union of what
    they export, or ``None`` when any of them cannot be read off disk."""
    start_dir = os.path.dirname(os.path.join(workspace, rel_path))
    modules: list[str] = []
    exported: set[str] = set()
    for node in _module_scope_statements(tree.body):
        if not isinstance(node, ast.ImportFrom) or not any(a.name == "*" for a in node.names):
            continue
        if node.level or not node.module:
            return None
        target = _resolve_module(node.module, start_dir, workspace)
        exports = _star_exports(target, workspace, set()) if target else None
        if exports is None:
            return None
        modules.append(node.module)
        exported |= exports
    return modules, exported


_VALUE_COMPARABLE_ENUM_MIXINS = {"IntEnum", "StrEnum", "IntFlag", "str", "int"}


def _attr_access(node: ast.expr) -> str | None:
    """``obj.attr`` -> ``"attr"``, else ``None``."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr
    return None


def _enum_member_value_ref(node: ast.expr) -> tuple[str, str] | None:
    """``EnumClass.MEMBER.value`` -> ``("EnumClass", "MEMBER")``, else ``None``."""
    if (isinstance(node, ast.Attribute) and node.attr == "value"
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)):
        return node.value.value.id, node.value.attr
    return None


def _scan_orm_module(path: str) -> tuple[dict[str, str], set[str], set[str]] | None:
    """One module's ``Column(Enum(X))`` / ``mapped_column(Enum(X))`` columns.

    Returns ``(column name -> enum class name, every locally-declared enum
    class name, the subset of those that mix in str/int and so compare equal
    to their own .value)``. ``None`` when the file cannot be parsed - nothing
    can be judged against it.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None

    enum_bases = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
    value_comparable_bases = set(_VALUE_COMPARABLE_ENUM_MIXINS)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "enum":
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name in enum_bases:
                    enum_bases.add(bound)
                if alias.name in value_comparable_bases:
                    value_comparable_bases.add(bound)

    enum_classes: dict[str, bool] = {}
    columns: dict[str, str] = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        base_names = {getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases}
        if base_names & enum_bases:
            enum_classes[cls.name] = bool(base_names & value_comparable_bases)
        for stmt in cls.body:
            value = getattr(stmt, "value", None)
            if not (isinstance(value, ast.Call) and getattr(
                    value.func, "id", getattr(value.func, "attr", "")
            ).rstrip("_") in {"mapped_column", "Column"}):
                continue
            enum_arg = next(
                (arg for arg in value.args if isinstance(arg, ast.Call)
                 and getattr(arg.func, "id", getattr(arg.func, "attr", "")).rstrip("_") == "Enum"
                 and arg.args and isinstance(arg.args[0], ast.Name)),
                None,
            )
            if enum_arg is None:
                continue
            names = _bound_names(stmt)
            if value.args and isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str):
                names = [value.args[0].value]
            for name in names:
                columns[name] = enum_arg.args[0].id
    return columns, set(enum_classes), {name for name, comparable in enum_classes.items() if comparable}


def _star_import_paths(tree: ast.Module, rel_path: str, workspace: str) -> list[str] | None:
    """Resolved file paths of the written file's star-imported modules, or
    ``None`` when any of them cannot be found on disk. Same resolution rule
    as ``_star_import_scope``, but returns paths rather than export names."""
    start_dir = os.path.dirname(os.path.join(workspace, rel_path))
    paths: list[str] = []
    for node in _module_scope_statements(tree.body):
        if not isinstance(node, ast.ImportFrom) or not any(a.name == "*" for a in node.names):
            continue
        if node.level or not node.module:
            return None
        target = _resolve_module(node.module, start_dir, workspace)
        if target is None:
            return None
        paths.append(target)
    return paths


def _enum_column_value_misuse(
    tree: ast.Module, rel_path: str | None, workspace: str | None
) -> list[dict]:
    """``obj.col != SomeEnum.MEMBER.value`` where ``col`` is a ``Column(Enum(...))``
    ORM attribute.

    SQLAlchemy's ORM returns the enum *member* for such a column, not its
    ``.value``, and a plain ``enum.Enum`` (no str/int mixin) never compares
    equal to its own ``.value`` under Python's default equality - confirmed
    live (run iw82zzoc): ``stored == MEMBER`` is True, ``stored ==
    MEMBER.value`` is False. So the guard, or its inverse, can never fire.

    ``col``'s type is established from an actual ``Column(Enum(X))`` /
    ``mapped_column(Enum(X))`` declaration in the file's star-imported ORM
    module(s), never guessed from the attribute name alone; an enum whose
    class mixes in ``str``/``int`` (``StrEnum``, ``IntEnum``, ``IntFlag``,
    or an explicit mixin base) is excluded, since those members do compare
    equal to their own ``.value``. If any star-imported module cannot be
    resolved and parsed, this reports nothing for the file.
    """
    if not rel_path or not workspace:
        return []
    paths = _star_import_paths(tree, rel_path, workspace)
    if not paths:
        return []

    columns: dict[str, str] = {}
    known_enums: set[str] = set()
    value_comparable: set[str] = set()
    for path in paths:
        scanned = _scan_orm_module(path)
        if scanned is None:
            return []
        module_columns, module_enums, module_value_comparable = scanned
        columns.update(module_columns)
        known_enums |= module_enums
        value_comparable |= module_value_comparable

    eligible = {
        col: enum_cls for col, enum_cls in columns.items()
        if enum_cls in known_enums and enum_cls not in value_comparable
    }
    if not eligible:
        return []

    def match(attr_side, value_side):
        col = _attr_access(attr_side)
        if col is None or col not in eligible:
            return None
        ref = _enum_member_value_ref(value_side)
        if ref is None or ref[0] != eligible[col]:
            return None
        return col, eligible[col], ref[1]

    findings: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for i, op in enumerate(node.ops):
            left, right = operands[i], operands[i + 1]
            if isinstance(op, (ast.Eq, ast.NotEq)):
                found = match(left, right) or match(right, left)
                if found is None:
                    continue
                col, enum_cls, member = found
                comparator = "!=" if isinstance(op, ast.NotEq) else "=="
                findings.append(_finding(
                    "python-contract",
                    f"'{col}' is a Column(Enum({enum_cls})) attribute; SQLAlchemy's ORM "
                    f"returns the enum member for it, not its .value, and a plain Enum "
                    f"member never equals its own .value. "
                    f"'{col} {comparator} {enum_cls}.{member}.value' is always "
                    f"{'True' if comparator == '!=' else 'False'}, regardless of the actual "
                    f"state. Compare against '{enum_cls}.{member}' directly, without '.value'.",
                    code="enum-column-compared-to-value",
                    line=node.lineno,
                ))
            elif isinstance(op, (ast.In, ast.NotIn)):
                col = _attr_access(left)
                if col is None or col not in eligible or not isinstance(right, (ast.Tuple, ast.List, ast.Set)):
                    continue
                enum_cls = eligible[col]
                members = [ref[1] for elt in right.elts
                           if (ref := _enum_member_value_ref(elt)) and ref[0] == enum_cls]
                if not members:
                    continue
                comparator = "not in" if isinstance(op, ast.NotIn) else "in"
                findings.append(_finding(
                    "python-contract",
                    f"'{col}' is a Column(Enum({enum_cls})) attribute; SQLAlchemy's ORM "
                    f"returns the enum member for it, not its .value, so it can never be "
                    f"{comparator} a container of .value strings "
                    f"({', '.join(f'{enum_cls}.{m}.value' for m in members)}). Compare against "
                    f"the members directly, without '.value'.",
                    code="enum-column-compared-to-value",
                    line=node.lineno,
                ))
    return findings


# Interactive/GUI/joke stdlib modules that have real import-time side effects
# (``antigravity`` opens a browser tab; ``turtle``/``tkinter`` want a display
# and can hang headless). Excluded even though they are technically stdlib,
# so resolving a star import never executes anything beyond a plain,
# side-effect-free module import - the same guarantee this file's docstring
# makes about generated code, extended to the stdlib modules it introspects.
_UNSAFE_STDLIB_IMPORTS = frozenset({
    "antigravity", "this", "turtle", "tkinter", "idlelib", "test", "lib2to3", "ensurepip",
})
_STDLIB_MODULES = frozenset(
    name for name in getattr(sys, "stdlib_module_names", ())
    if not name.startswith("_") and name not in _UNSAFE_STDLIB_IMPORTS
)


def _scan_star_import_bindings(path: str, workspace: str) -> dict[str, object] | None:
    """Names one star-imported module binds via its own ``import`` /
    ``from ... import`` statements, resolved to the real object.

    Static AST parsing only, never importing ``path`` itself - it is
    generated code, and this module's docstring rules that out. Resolution
    only reaches further when a binding's *source* is a standard-library
    module: :func:`importlib.import_module` on a known stdlib name is
    side-effect-free (barring ``_UNSAFE_STDLIB_IMPORTS``) and always
    available, unlike a third-party dependency the diagnostics environment
    may not have installed, or another workspace file, which is generated
    code just the same.

    A name is dropped rather than guessed at when: its source is a
    workspace file or a non-stdlib package; the stdlib module doesn't
    actually export it; two imports disagree on what it is (e.g. a
    try/except fallback import); or this module also binds it some other
    way (assignment, def, class, a for/with/except target) - any of those
    make the real, final value something this static pass cannot prove.
    Names this module only re-exports via its own ``from x import *`` are
    likewise never added here; nothing is known about what they resolve to.

    Returns ``None`` only when ``path`` itself cannot be parsed - nothing
    can be judged against it, mirroring ``_scan_orm_module``.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None

    module_dir = os.path.dirname(path)
    resolved: dict[str, object] = {}
    shadowed: set[str] = set()

    def invalidate(name: str) -> None:
        shadowed.add(name)
        resolved.pop(name, None)

    def offer(name: str, obj: object) -> None:
        if name in shadowed:
            return
        if name in resolved and resolved[name] is not obj:
            invalidate(name)
            return
        resolved[name] = obj

    for node in _module_scope_statements(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                bound_name = alias.asname or top
                if (_resolve_module(top, module_dir, workspace) is not None
                        or top not in _STDLIB_MODULES):
                    invalidate(bound_name)
                    continue
                try:
                    imported = importlib.import_module(alias.name)
                except Exception:
                    invalidate(bound_name)
                    continue
                offer(bound_name, imported if alias.asname else sys.modules[top])
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                continue
            if node.level or not node.module:
                for alias in node.names:
                    invalidate(alias.asname or alias.name)
                continue
            top = node.module.split(".")[0]
            if (_resolve_module(node.module, module_dir, workspace) is not None
                    or top not in _STDLIB_MODULES):
                for alias in node.names:
                    invalidate(alias.asname or alias.name)
                continue
            try:
                source = importlib.import_module(node.module)
            except Exception:
                for alias in node.names:
                    invalidate(alias.asname or alias.name)
                continue
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if not hasattr(source, alias.name):
                    invalidate(bound_name)
                    continue
                offer(bound_name, getattr(source, alias.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            invalidate(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name_node in ast.walk(target):
                    if isinstance(name_node, ast.Name):
                        invalidate(name_node.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                invalidate(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for name_node in ast.walk(node.target):
                if isinstance(name_node, ast.Name):
                    invalidate(name_node.id)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for name_node in ast.walk(item.optional_vars):
                        if isinstance(name_node, ast.Name):
                            invalidate(name_node.id)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                invalidate(node.name)

    if not resolved:
        return resolved
    # ``__all__`` (or a leading underscore) can keep a name a plain
    # ``import``/``from`` statement binds out of ``from <module> import *``
    # entirely; without this intersection a name resolved above but not
    # actually re-exported would be flagged as broken when it is simply
    # absent from the star-importing file's namespace (a different, already
    # -covered bug: pyflakes' ImportStarUsage / this module's UndefinedName).
    exported = _star_exports(path, workspace, set())
    if exported is None:
        return None
    return {name: obj for name, obj in resolved.items() if name in exported}


def _describe_binding(obj: object) -> str:
    """What a resolved star-import binding actually is, for a finding
    message: its kind and its real, importable name - never the name the
    generated code mistakenly assumes it has."""
    if inspect.ismodule(obj):
        return f"the module '{obj.__name__}'"
    qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
    home = getattr(obj, "__module__", None)
    label = f"{home}.{qualname}" if home else str(qualname)
    if isinstance(obj, type):
        kind = "class"
    elif inspect.isroutine(obj):
        kind = "function"
    else:
        kind = type(obj).__name__
    return f"the {kind} '{label}'"


def _locally_bound_names(tree: ast.Module) -> set[str]:
    """Every name the written file binds itself, at any scope: assignment,
    augmented-assignment, for/with/except and comprehension targets, def/class
    names, parameters, and this file's own imports. Deliberately whole-file
    and scope-blind rather than a precise per-scope (LEGB) resolution: a name
    assigned anywhere shadows the star import for this check everywhere in
    the file, even where Python's actual scoping would still see the star
    import. That can only miss a genuinely-broken case, never invent one -
    the safe side to err on.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {alias.asname or alias.name for alias in node.names if alias.name != "*"}
    return names


def _star_import_attribute_misuse(
    tree: ast.Module, rel_path: str | None, workspace: str | None
) -> list[dict]:
    """``NAME.attr`` (plain access or a call) where ``NAME`` is provided only
    by a star import and the object actually bound to it has no ``attr``.

    Live (run _abcgx9s): ``routers/booking_methods.py`` does ``from
    sql_alchemy import *``; ``sql_alchemy.py`` does ``from datetime import
    ..., time``, so the star import binds ``time`` to the *class*
    ``datetime.time``, not the ``time`` module. ``int(time.time())`` then
    raises ``AttributeError: type object 'datetime.time' has no attribute
    'time'`` on the first request that reaches it. The name is genuinely
    bound, so pyflakes and ruff both pass, and the file ``ast.parse``s
    cleanly too - the same "ships green, boots dead" class as an undefined
    name behind a star import, one layer over.

    ``NAME``'s real object comes only from ``_scan_star_import_bindings``,
    which never imports the star-imported module itself; see its docstring
    for exactly which bindings that leaves resolved. A name this file (the
    one being diagnosed, not the star-imported one) also binds itself,
    anywhere, is skipped as shadowed, and so is any builtin name. If any
    star-imported module cannot be resolved and parsed, this reports
    nothing for the file, same as ``_enum_column_value_misuse``.
    """
    if not rel_path or not workspace:
        return []
    paths = _star_import_paths(tree, rel_path, workspace)
    if not paths:
        return []

    bindings: dict[str, object] = {}
    origin: dict[str, str] = {}
    for path in paths:
        scanned = _scan_star_import_bindings(path, workspace)
        if scanned is None:
            return []
        label = os.path.splitext(os.path.basename(path))[0]
        for name, obj in scanned.items():
            bindings[name] = obj
            origin[name] = label
    if not bindings:
        return []

    local_names = _locally_bound_names(tree)

    findings: list[dict] = []
    for node in ast.walk(tree):
        attr = _attr_access(node)
        if attr is None:
            continue
        name = node.value.id
        if name in local_names or name not in bindings or hasattr(builtins, name):
            continue
        obj = bindings[name]
        if hasattr(obj, attr):
            continue
        findings.append(_finding(
            "python-contract",
            f"'{name}.{attr}' does not exist: the star import from '{origin[name]}' "
            f"binds '{name}' to {_describe_binding(obj)}, which has no '{attr}'. This "
            f"is a standard-library name collision, not an undefined name, so pyflakes "
            f"and ruff both pass while this raises AttributeError the first time the "
            f"line actually runs.",
            code="star-import-attribute-misuse",
            line=node.lineno,
        ))
    return findings


def _python_diagnostics(
    rel_path: str, content: str, workspace: str | None = None
) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(content, filename=rel_path)
    except SyntaxError as exc:
        return [_finding(
            "python",
            exc.msg,
            code="syntax",
            line=exc.lineno,
            column=exc.offset,
        )]

    unawaited = python_structural_diagnostics(
        tree, sqlite=workspace_uses_sqlite(workspace),
        rel_path=rel_path, workspace=workspace,
    ) + _unawaited_coroutines(tree)

    # Pyflakes is a small, in-process AST checker. Keep this collector focused
    # on undefined-name failures; unused-import style feedback is noisy during
    # incremental construction and the full Ruff pass handles it later.
    #
    # Under ``from x import *`` pyflakes reports every unresolved load as
    # ImportStarUsage, never UndefinedName (checker.py, handleNodeLoad), and
    # every scaffold router star-imports sql_alchemy, pydantic_classes and
    # bal_stdlib. Those modules are ours: read what they export and judge the
    # name against it (run 0c537a4e, 2026-09-18, shipped five bodies with
    # undefined names and no diagnostics because of this).
    try:
        from pyflakes.checker import Checker
    except ImportError:
        # pyflakes ships in the backend requirements but a library caller may not
        # have it. Warn once: a silent [] is indistinguishable from a clean file,
        # which is how code that NameErrors on import ships as "no diagnostics".
        global _WARNED_PYFLAKES_MISSING
        if not _WARNED_PYFLAKES_MISSING:
            _WARNED_PYFLAKES_MISSING = True
            logger.warning(
                "pyflakes is not installed - same-turn undefined-name checks on "
                "written files are SKIPPED. Install pyflakes to enable them."
            )
        return unawaited

    try:
        messages = Checker(tree, filename=rel_path).messages
    except Exception:
        logger.debug("pyflakes failed on %s", rel_path, exc_info=True)
        return unawaited

    star_scope = None
    if workspace:
        try:
            star_scope = _star_import_scope(tree, rel_path, workspace)
        except Exception:
            logger.debug("star-import resolution failed on %s", rel_path, exc_info=True)

    findings: list[dict[str, Any]] = list(unawaited)
    undefined_kinds = {"UndefinedName", "UndefinedExport", "UndefinedLocal"}
    dead_lines = _unreachable_lines(tree)
    for item in sorted(messages, key=lambda msg: (msg.lineno, msg.col)):
        if getattr(item, "lineno", None) in dead_lines:
            continue  # unreachable: the name is never looked up at runtime
        kind = type(item).__name__
        if kind == "ImportStarUsage":
            if star_scope is None or item.message_args[0] in star_scope[1]:
                continue
            kind = "UndefinedName"
            message = (
                f"undefined name '{item.message_args[0]}' - not defined in this file "
                f"and not exported by {', '.join(star_scope[0])}"
            )
        elif kind not in undefined_kinds:
            continue
        else:
            try:
                message = item.message % item.message_args
            except Exception:
                message = str(item)
        findings.append(_finding(
            "pyflakes",
            message,
            code=kind,
            line=getattr(item, "lineno", None),
            column=(getattr(item, "col", 0) or 0) + 1,
        ))
    return findings


def _json_diagnostics(content: str) -> list[dict[str, Any]]:
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return [_finding(
            "json", exc.msg, code="parse", line=exc.lineno, column=exc.colno
        )]
    return []


def _yaml_diagnostics(content: str) -> list[dict[str, Any]]:
    try:
        import yaml

        yaml.safe_load(content)
    except Exception as exc:
        mark = getattr(exc, "problem_mark", None)
        message = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        return [_finding(
            "yaml",
            message,
            code="parse",
            line=(getattr(mark, "line", -1) + 1) if mark is not None else None,
            column=(getattr(mark, "column", -1) + 1) if mark is not None else None,
        )]
    return []


def _toml_diagnostics(content: str) -> list[dict[str, Any]]:
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return [_finding(
            "toml",
            str(exc),
            code="parse",
            line=getattr(exc, "lineno", None),
            column=getattr(exc, "colno", None),
        )]
    return []


def diagnose_written_content(
    rel_path: str,
    content: str,
    *,
    limit: int = MAX_WRITE_DIAGNOSTICS,
    workspace: str | None = None,
) -> list[dict[str, Any]]:
    """Return bounded parser/undefined-name findings for supported file types.

    ``workspace`` is the root the file's local star imports resolve under;
    without it a name behind ``from x import *`` is not judged.
    """
    extension = os.path.splitext(rel_path.lower())[1]
    try:
        if extension in {".py", ".pyi"}:
            findings = _python_diagnostics(rel_path, content, workspace)
        elif extension == ".json":
            findings = _json_diagnostics(content)
        elif extension in {".yaml", ".yml"}:
            findings = _yaml_diagnostics(content)
        elif extension == ".toml":
            findings = _toml_diagnostics(content)
        else:
            findings = []
    except Exception:
        # Diagnostics must never turn a successful write into a failed tool
        # call. Phase 3 remains the independent backstop.
        return []
    return findings[:max(0, limit)]
