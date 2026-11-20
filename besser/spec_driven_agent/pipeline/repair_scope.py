"""The files a Phase 3 fix attempt may edit.

An attempt edits the files its blockers name, the files those import or that
import them, and new files. A frontend file also brings its folder and the
app's routing and entry files, which relative-import resolution can miss (path
aliases). A blocker that names no file leaves the attempt's scope open.
"""

from __future__ import annotations

import os
import posixpath
import re

from besser.spec_driven_agent.planning.requirements_ledger import _JS_SUFFIXES, ImportIndex, _ancestors

# A path-like token and the line it names: ``x.py line 3``, ``x.py:3:1``, ``x.tsx(3,5)``.
_NAMED_PATH_RE = re.compile(r"(?<![\w./\\-])([\w./\\-]*\w\.[A-Za-z]\w*)(?:(?: line |:|\()(\d+))?")
# ``tsc [frontend]: src/App.tsx(3,5): ...``: a toolchain names files relative to its project.
_PROJECT_RE = re.compile(r"^[\w ]+ \[([^\]]+)\]:")
_ENTRY_STEMS = frozenset({"main", "index", "App"})
_ROUTER_RE = re.compile(r"<Route\b|createBrowserRouter|createRouter\(")


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


def edit_scope(named: set[str], output_dir: str, files: list[str]) -> set[str]:
    """``named``, the files they import, the files importing them and, for a frontend
    file, the other files of its folder and its app's routing and entry files."""
    index = ImportIndex(output_dir)
    allowed = set(named)
    for rel in named:
        allowed |= index.imports(rel) | index.importers(rel)
        if os.path.splitext(rel)[1].lower() in _JS_SUFFIXES:
            folder = posixpath.dirname(rel)
            allowed |= {other for other in files if posixpath.dirname(other) == folder}
            allowed |= _routing_files(output_dir, rel, files)
    return allowed


def _routing_files(output_dir: str, rel: str, files: list[str]) -> set[str]:
    """The entry files (``main`` / ``index`` / ``App`` at the project root or its ``src``)
    and the files declaring routes, of the frontend project holding ``rel``."""
    known = set(files)
    project = next((folder for folder in _ancestors(posixpath.dirname(rel))
                    if posixpath.join(folder, "package.json") in known), None)
    if project is None:
        return set()
    prefix = f"{project}/" if project else ""
    found = set()
    for other in files:
        stem, suffix = os.path.splitext(other)
        if not other.startswith(prefix) or suffix.lower() not in _JS_SUFFIXES:
            continue
        if posixpath.dirname(other) in (project, posixpath.join(project, "src")) and \
                posixpath.basename(stem) in _ENTRY_STEMS:
            found.add(other)
            continue
        try:
            with open(os.path.join(output_dir, other), encoding="utf-8", errors="ignore") as handle:
                if _ROUTER_RE.search(handle.read()):
                    found.add(other)
        except OSError:
            continue
    return found
