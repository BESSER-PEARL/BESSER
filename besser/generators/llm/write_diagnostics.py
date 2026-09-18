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
import tomllib
from typing import Any


logger = logging.getLogger(__name__)

MAX_WRITE_DIAGNOSTICS = 10


_WARNED_PYFLAKES_MISSING = False


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


def _python_diagnostics(rel_path: str, content: str) -> list[dict[str, Any]]:
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

    unawaited = _unawaited_coroutines(tree)

    # Pyflakes is a small, in-process AST checker. Keep this collector focused
    # on undefined-name failures; unused-import style feedback is noisy during
    # incremental construction and the full Ruff pass handles it later.
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

    findings: list[dict[str, Any]] = list(unawaited)
    undefined_kinds = {"UndefinedName", "UndefinedExport", "UndefinedLocal"}
    for item in sorted(messages, key=lambda msg: (msg.lineno, msg.col)):
        if type(item).__name__ not in undefined_kinds:
            continue
        try:
            message = item.message % item.message_args
        except Exception:
            message = str(item)
        findings.append(_finding(
            "pyflakes",
            message,
            code=type(item).__name__,
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
) -> list[dict[str, Any]]:
    """Return bounded parser/undefined-name findings for supported file types."""
    extension = os.path.splitext(rel_path.lower())[1]
    try:
        if extension in {".py", ".pyi"}:
            findings = _python_diagnostics(rel_path, content)
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
