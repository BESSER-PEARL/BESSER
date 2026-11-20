"""The files a Phase 3 fix attempt may edit.

An attempt edits the files its blockers name, the files those import or that
import them (relative imports, and tsconfig ``paths`` aliases for JS/TS), the
other source files of their folder, and new files; a frontend file also brings
its app's routing and entry files, and a blocker about a missing module brings
the dependency manifest. A blocker that names no file leaves the attempt's
scope open. The scope steers the model; keeping or reverting an attempt is
what protects the tree.
"""

from __future__ import annotations

import json
import os
import posixpath
import re

from besser.spec_driven_agent.planning.requirements_ledger import (
    _JS_RESOLVE_SUFFIXES,
    _JS_SUFFIXES,
    ImportIndex,
    _ancestors,
)
from besser.spec_driven_agent.validation.frontend_resolution import import_specifiers
from besser.spec_driven_agent.validation.write_diagnostics import _jsonc_to_json

# A path-like token and the line it names: ``x.py line 3``, ``x.py:3:1``, ``x.tsx(3,5)``.
_NAMED_PATH_RE = re.compile(r"(?<![\w./\\-])([\w./\\-]*\w\.[A-Za-z]\w*)(?:(?: line |:|\()(\d+))?")
# ``tsc [frontend]: src/App.tsx(3,5): ...``: a toolchain names files relative to its project.
_PROJECT_RE = re.compile(r"^[\w ]+ \[([^\]]+)\]:")
_ENTRY_STEMS = frozenset({"main", "index", "App"})
_ROUTER_RE = re.compile(r"<Route\b|createBrowserRouter|createRouter\(")
# The dependency manifest a missing module is declared in, nearest the file first.
_PYTHON_MANIFESTS = ("requirements.txt", "pyproject.toml")
_JS_MANIFESTS = ("package.json",)


def named_locations(message: str, files: list[str]) -> list[tuple[str, int | None]]:
    """The workspace files ``message`` names, each once, with the line it names.

    A token resolves as a workspace path, then against the toolchain project the
    message names, then as the suffix of exactly one file; anything else is not
    a file this message names.
    """
    known = set(files)
    project = _PROJECT_RE.match(message)
    found: dict[str, int | None] = {}
    for token, line in _NAMED_PATH_RE.findall(message):
        token = token.replace("\\", "/")
        while token.startswith("./"):
            token = token[2:]
        candidates = [token] if token in known else []
        if not candidates and project:
            joined = posixpath.normpath(posixpath.join(project.group(1), token))
            candidates = [joined] if joined in known else []
        if not candidates:
            candidates = [rel for rel in files if rel.endswith("/" + token)]
        if len(candidates) == 1 and candidates[0] not in found:
            found[candidates[0]] = int(line) if line else None
    return list(found.items())


def edit_scope(named: set[str], output_dir: str, files: list[str],
               manifests_for: frozenset[str] = frozenset()) -> set[str]:
    """``named``, the files they import, the files importing them, the other source
    files of their folders, a frontend file's routing and entry files, and the
    dependency manifest of each file in ``manifests_for``."""
    index = ImportIndex(output_dir)
    allowed = set(named)
    for rel in named:
        suffix = os.path.splitext(rel)[1].lower()
        allowed |= index.imports(rel) | index.importers(rel)
        folder = posixpath.dirname(rel)
        if suffix in _JS_SUFFIXES:
            allowed |= {other for other in files if posixpath.dirname(other) == folder}
            allowed |= _routing_files(output_dir, rel, files)
            allowed |= _alias_neighbours(output_dir, rel, files)
        elif suffix == ".py":
            allowed |= {other for other in files if posixpath.dirname(other) == folder and other.endswith(".py")}
        if rel in manifests_for:
            allowed |= _manifest(rel, files, _JS_MANIFESTS if suffix in _JS_SUFFIXES else _PYTHON_MANIFESTS)
    return allowed


def _manifest(rel: str, files: list[str], names: tuple[str, ...]) -> set[str]:
    """The nearest of ``names`` above ``rel``."""
    known = set(files)
    for folder in _ancestors(posixpath.dirname(rel)):
        found = {posixpath.join(folder, name) for name in names} & known
        if found:
            return found
    return set()


def _project(rel: str, known: set[str]) -> str | None:
    """The frontend project (nearest folder with a ``package.json``) holding ``rel``."""
    return next((folder for folder in _ancestors(posixpath.dirname(rel))
                 if posixpath.join(folder, "package.json") in known), None)


def _project_sources(project: str, files: list[str]) -> list[str]:
    prefix = f"{project}/" if project else ""
    return [other for other in files
            if other.startswith(prefix) and os.path.splitext(other)[1].lower() in _JS_SUFFIXES]


def _routing_files(output_dir: str, rel: str, files: list[str]) -> set[str]:
    """The entry files (``main`` / ``index`` / ``App`` at the project root or its ``src``)
    and the files declaring routes, of the frontend project holding ``rel``."""
    project = _project(rel, set(files))
    if project is None:
        return set()
    found = set()
    for other in _project_sources(project, files):
        stem = os.path.splitext(posixpath.basename(other))[0]
        if posixpath.dirname(other) in (project, posixpath.join(project, "src")) and stem in _ENTRY_STEMS:
            found.add(other)
            continue
        try:
            with open(os.path.join(output_dir, other), encoding="utf-8", errors="ignore") as handle:
                if _ROUTER_RE.search(handle.read()):
                    found.add(other)
        except OSError:
            continue
    return found


def _aliases(output_dir: str, project: str) -> list[tuple[str, str, list[str]]]:
    """The project's tsconfig/jsconfig ``paths``: (folder they resolve from, alias prefix, target prefixes)."""
    for name in ("tsconfig.json", "tsconfig.app.json", "jsconfig.json"):
        try:
            with open(os.path.join(output_dir, project, name), encoding="utf-8") as handle:
                options = json.loads(_jsonc_to_json(handle.read())).get("compilerOptions") or {}
        except (OSError, ValueError, AttributeError):
            continue
        paths = options.get("paths") if isinstance(options, dict) else None
        if isinstance(paths, dict) and paths:
            base = posixpath.normpath(posixpath.join(project, str(options.get("baseUrl") or ".")))
            return [(base, alias.split("*")[0], [str(target).split("*")[0] for target in targets])
                    for alias, targets in paths.items() if isinstance(targets, list)]
    return []


def _alias_neighbours(output_dir: str, rel: str, files: list[str]) -> set[str]:
    """Files ``rel`` imports, or that import it, through a tsconfig ``paths`` alias."""
    known = set(files)
    project = _project(rel, known)
    aliases = _aliases(output_dir, project) if project is not None else []
    if not aliases:
        return set()

    def targets(source: str) -> set[str]:
        found = set()
        for spec in import_specifiers(os.path.join(output_dir, source)):
            for base, alias, prefixes in aliases:
                if alias and spec.startswith(alias):
                    for prefix in prefixes:
                        stem = posixpath.normpath(posixpath.join(base, prefix + spec[len(alias):]))
                        found |= {stem + tail for tail in _JS_RESOLVE_SUFFIXES if stem + tail in known}
        return found

    importers = {other for other in _project_sources(project, files) if other != rel and rel in targets(other)}
    return targets(rel) | importers
