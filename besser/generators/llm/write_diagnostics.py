"""Cheap, same-turn diagnostics for LLM-authored file writes.

These checks deliberately avoid importing or executing generated code. They
give the model immediate feedback after ``write_file`` / ``modify_file`` while
the changed region is still in context; the broader Phase-3 sweep remains the
release gate.
"""

from __future__ import annotations

import ast
import json
import os
import tomllib
from typing import Any


MAX_WRITE_DIAGNOSTICS = 10


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

    # Pyflakes is a small, in-process AST checker. Keep this collector focused
    # on undefined-name failures; unused-import style feedback is noisy during
    # incremental construction and the full Ruff pass handles it later.
    try:
        from pyflakes.checker import Checker

        messages = Checker(tree, filename=rel_path).messages
    except Exception:
        return []

    findings: list[dict[str, Any]] = []
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
