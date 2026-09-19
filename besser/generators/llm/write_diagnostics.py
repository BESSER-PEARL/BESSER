"""Cheap, same-turn diagnostics for LLM-authored file writes.

These checks deliberately avoid importing or executing generated code. They
give the model immediate feedback after ``write_file`` / ``modify_file`` while
the changed region is still in context; the broader Phase-3 sweep remains the
release gate.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import sqlite3
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


def _bound_names(statement: ast.stmt) -> list[str]:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [statement.name]
    targets = statement.targets if isinstance(statement, ast.Assign) else (
        [statement.target] if isinstance(statement, ast.AnnAssign) else []
    )
    return [target.id for target in targets if isinstance(target, ast.Name)]


def python_structural_diagnostics(tree: ast.Module, *, sqlite: bool = False) -> list[dict]:
    """Provable declaration errors; no application imports, execution, or guessing.

    SQLite CHECK expressions are compiled against an in-memory table containing
    only the mapped column names. No generated Python or user database is used.
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
    for item in sorted(messages, key=lambda msg: (msg.lineno, msg.col)):
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
