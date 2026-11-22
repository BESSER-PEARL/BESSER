"""Add an import the model forgot, when the answer is not a guess.

A common failure: an edit uses a framework name the file never imported
(e.g. ``Table(...)`` without ``from sqlalchemy import Table``), the module
stops importing, every router that star-imports it dies, and Phase 3 rolls
the whole repair back. Nothing about this needs a language model: the name
is exported by exactly one module the app already depends on.

NOT covered: an ALIASED name (e.g. ``dt_date`` for ``date as dt_date``).
No allowlisted module exports it, and recovering it means inferring which
real name the alias stands for, which is the guess this module refuses.

Deliberately narrow. A name is repaired only when exactly ONE allowlisted
module exports it, checked by importing that module rather than by matching
a hand-typed list. Anything ambiguous, unknown, or locally-defined-later is
left for the LLM and for Phase 3 - a wrong import is worse than a missing
one, because it makes the failure harder to read.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import logging
import os
import sys

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


# Stdlib modules a missing name may become ``import <name>`` for: common in generated
# backends, and importing one has no side effect. Never an arbitrary stdlib name: ``email``,
# ``token``, ``code`` and ``string`` are far likelier to be an undefined variable.
_STDLIB_IMPORTABLE = frozenset({
    "base64", "collections", "copy", "csv", "decimal", "enum", "functools", "hashlib", "hmac",
    "io", "itertools", "json", "logging", "math", "os", "pathlib", "random", "re", "secrets",
    "shutil", "statistics", "sys", "textwrap", "traceback", "uuid", "warnings", "zlib",
})


@functools.lru_cache(maxsize=None)
def _explicit_reexports(module_name: str) -> frozenset[str]:
    """Names ``module_name``'s own source re-exports explicitly: ``import X as X``, ``from Y import X as X``."""
    try:
        tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    except Exception:
        return frozenset()
    return frozenset(alias.asname for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                     for alias in node.names if alias.asname and alias.asname == alias.name.split(".")[-1])


def _exporting_modules(name: str, attributes: frozenset[str] = frozenset()) -> list[str]:
    """Allowlisted modules that export ``name``, as an importable attribute.

    A name its ``__all__`` leaves out is not an export. Nor is a module a candidate merely
    holds (``typing.re``, ``uuid.os``, ``sqlalchemy.engine``): a module counts only when the
    candidate's source re-exports it explicitly (``fastapi.status``), it is no stdlib module's
    name, and it has every attribute in ``attributes`` the file reads from it.
    """
    found = []
    for module_name in _CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        if inspect.ismodule(value) and (name in sys.stdlib_module_names
                                        or name not in _explicit_reexports(module_name)
                                        or not all(hasattr(value, a) for a in attributes)):
            continue
        public = getattr(module, "__all__", None)
        if public is not None and name not in public:
            continue
        found.append(module_name)
    return found


def _attributes_read(tree: ast.AST, name: str) -> frozenset[str]:
    """The attributes the file reads from ``name`` (``findall`` for ``re.findall``)."""
    return frozenset(node.attr for node in ast.walk(tree)
                     if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                     and node.value.id == name)


def _used_only_as_module(tree: ast.AST, name: str) -> bool:
    """Whether every load of ``name`` reads an attribute of it (``re.findall``)."""
    attribute_bases = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    uses = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == name]
    return bool(uses) and all(id(node) in attribute_bases for node in uses)


def _local_module(root: str, path: str, name: str) -> bool:
    """Whether a local ``name`` module could shadow the stdlib one for the file at ``path``.

    A ``<name>.py``, or a ``<name>/`` folder holding Python files, in the file's own folder or any
    folder above it up to ``root`` (the workspace and the app folder are import roots). A file
    outside ``root`` counts as shadowed: unknown is not safe.
    """
    from besser.utilities.workspace_fs import walk_plain

    root = os.path.abspath(root)
    folder = os.path.dirname(os.path.abspath(os.path.join(root, path)))
    if os.path.commonpath([folder, root]) != root:
        return True
    while True:
        if os.path.isfile(os.path.join(folder, f"{name}.py")):
            return True
        package = os.path.join(folder, name)
        if os.path.isdir(package) and any(
                file.endswith(".py") for _sub, _dirs, files in walk_plain(package) for file in files):
            return True
        if folder == root:
            return False
        folder = os.path.dirname(folder)


def _stdlib_module_use(tree: ast.AST, name: str, root: str | None, path: str) -> bool:
    """Whether a missing ``name`` is the stdlib module of that name, used as one."""
    if root is None or name not in _STDLIB_IMPORTABLE or not _used_only_as_module(tree, name):
        return False
    module = importlib.import_module(name)
    return all(hasattr(module, a) for a in _attributes_read(tree, name)) and not _local_module(root, path, name)


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


def repair_missing_imports(path: str, content: str, root: str | None = None) -> tuple[str, list[str]]:
    """Return ``(content, notes)``; ``content`` is unchanged when unsure.

    The result is re-parsed and re-checked, so a repair that does not
    actually resolve the name is discarded rather than shipped. ``root`` is
    the workspace: without it no ``import <stdlib module>`` is added, since a
    local module of that name cannot be ruled out.
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
    modules_to_import: list[str] = []
    for name in sorted(missing):
        modules = _exporting_modules(name, _attributes_read(tree, name))
        # A stdlib module used as one (``re.findall``) is imported as itself.
        if not modules and _stdlib_module_use(tree, name, root, path):
            modules_to_import.append(name)
            continue
        if len(modules) != 1:
            # Ambiguous or unknown: a wrong import is worse than none.
            continue
        add_to.setdefault(modules[0], []).append(name)
    if not add_to and not modules_to_import:
        return content, []

    lines = content.splitlines(keepends=True)
    notes: list[str] = []
    # Extend an existing "from X import ..." in place where we can; otherwise
    # add one line after the last import. Work bottom-up so earlier line
    # numbers stay valid.
    appended: list[str] = [f"import {name}\n" for name in modules_to_import]
    notes.extend(f"added 'import {name}'" for name in modules_to_import)
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
    if still_missing & {*modules_to_import, *(n for names in add_to.values() for n in names)}:
        return content, []
    logger.info("Auto-imported into %s: %s", path, "; ".join(notes))
    return repaired, notes
