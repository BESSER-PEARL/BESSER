"""Add an import the model forgot, when the answer is not a guess.

Three live Qwen runs shipped the same shape: an edit uses a framework name
the file never imported, the module stops importing, every router that
star-imports it dies, and Phase 3 rolls the whole repair back.

    trilraak  booking_guest = Table(...)   # sqlalchemy.Table never imported
    trilraak  dt_date                      # datetime never imported
    pcovsppe  datetime                     # same

Run trilraak shipped a backend that could not start because of it, and runs
pcovsppe and se7k3zbx lost otherwise-good repairs to the rollback it caused.
Nothing about this needs a language model: the name is exported by exactly
one module the app already depends on.

Deliberately narrow. A name is repaired only when exactly ONE allowlisted
module exports it, checked by importing that module rather than by matching
a hand-typed list. Anything ambiguous, unknown, or locally-defined-later is
left for the LLM and for Phase 3 - a wrong import is worse than a missing
one, because it makes the failure harder to read.
"""

from __future__ import annotations

import ast
import importlib
import logging

logger = logging.getLogger(__name__)

# Modules the generated stack already depends on, so importing one here adds
# no capability the app did not already have. Ordered: the first module that
# exports a name wins ties only when the others do not export it at all.
_CANDIDATE_MODULES = (
    "datetime",
    "decimal",
    "enum",
    "uuid",
    "typing",
    "sqlalchemy",
    "sqlalchemy.orm",
    "pydantic",
    "fastapi",
)

# Never auto-import these: too generic, or shadowing one is a real bug.
_NEVER = frozenset({"id", "type", "list", "dict", "set", "filter", "map",
                    "date", "time", "Any", "self", "cls"})


def _exporting_modules(name: str) -> list[str]:
    """Allowlisted modules that export ``name``, as an importable attribute."""
    found = []
    for module_name in _CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, name):
            found.append(module_name)
    return found


def _undefined_names(tree: ast.AST, path: str) -> set[str]:
    try:
        from pyflakes.checker import Checker
    except ImportError:
        return set()
    try:
        messages = Checker(tree, filename=path).messages
    except Exception:
        return set()
    names = set()
    for item in messages:
        if type(item).__name__ in {"UndefinedName", "UndefinedLocal"}:
            args = getattr(item, "message_args", ())
            if args and isinstance(args[0], str):
                names.add(args[0])
    return names


def _existing_from_imports(tree: ast.AST) -> dict[str, ast.ImportFrom]:
    """``module -> the last 'from module import ...' node`` at top level."""
    found: dict[str, ast.ImportFrom] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found[node.module] = node
    return found


def _insertion_line(tree: ast.AST) -> int:
    """After the last top-level import, else after the module docstring."""
    line = 0
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            line = max(line, getattr(node, "end_lineno", node.lineno))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str) and line == 0:
            line = getattr(node, "end_lineno", node.lineno)
    return line


def repair_missing_imports(path: str, content: str) -> tuple[str, list[str]]:
    """Return ``(content, notes)``; ``content`` is unchanged when unsure.

    The result is re-parsed and re-checked, so a repair that does not
    actually resolve the name is discarded rather than shipped.
    """
    if not path.lower().endswith((".py", ".pyi")):
        return content, []
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError:
        return content, []

    missing = {n for n in _undefined_names(tree, path) if n not in _NEVER}
    if not missing:
        return content, []

    existing = _existing_from_imports(tree)
    add_to: dict[str, list[str]] = {}
    for name in sorted(missing):
        modules = _exporting_modules(name)
        if len(modules) != 1:
            # Ambiguous or unknown: a wrong import is worse than none.
            continue
        add_to.setdefault(modules[0], []).append(name)
    if not add_to:
        return content, []

    lines = content.splitlines(keepends=True)
    notes: list[str] = []
    # Extend an existing "from X import ..." in place where we can; otherwise
    # add one line after the last import. Work bottom-up so earlier line
    # numbers stay valid.
    appended: list[str] = []
    edits: list[tuple[int, str]] = []
    for module, names in add_to.items():
        node = existing.get(module)
        if node is not None and not any(a.name == "*" for a in node.names):
            index = node.end_lineno - 1
            if index < len(lines) and lines[index].rstrip("\r\n").endswith(tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789")):
                edits.append((index, lines[index].rstrip("\r\n") + ", " + ", ".join(names) + "\n"))
                notes.append(f"added {', '.join(names)} to the existing 'from {module} import ...'")
                continue
        appended.append(f"from {module} import {', '.join(names)}\n")
        notes.append(f"added 'from {module} import {', '.join(names)}'")

    for index, replacement in sorted(edits, reverse=True):
        lines[index] = replacement
    if appended:
        at = _insertion_line(tree)
        lines[at:at] = appended

    repaired = "".join(lines)
    try:
        repaired_tree = ast.parse(repaired, filename=path)
    except SyntaxError:
        return content, []
    still_missing = _undefined_names(repaired_tree, path)
    if still_missing & {n for names in add_to.values() for n in names}:
        return content, []
    logger.info("Auto-imported into %s: %s", path, "; ".join(notes))
    return repaired, notes
