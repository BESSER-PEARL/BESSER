"""Scaffold map: where the deterministically generated code lives, and where new code goes.

Phase 1 writes a scaffold with a BESSER generator; Phase 2's LLM then has to
extend it without rewriting it or adding code where the framework ignores it
(an ``@app`` route registered after the routers, a static route behind
``/{id}/``, a page with no ``<Route>``). This package reads the scaffold's
files and returns, per application root and per domain entity, every place
that matters: definitions, routes, stubs to implement and named insertion
points, each with a stable id and its current lines.

Entry points:

* :func:`build_workspace_map` — compute the map from disk (cheap, read-only).
* :func:`scaffold_overview` — the line-free overview for the system prompt
  and the gap analyzer.
* :func:`locate` — the ``locate_code`` tool's answer, with current lines and
  ownership.

Extractors (``fastapi``, ``django``, ``react``, ``generic``) each recognise
one scaffold family by its layout, never by which generator was asked for, so
they also cover a scaffold an earlier run produced or the model reshaped.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from besser.spec_driven_agent.planning.scaffold_map.model import (
    Anchor,
    EntityMap,
    Recipe,
    ScaffoldMap,
    WorkspaceMap,
)
from besser.spec_driven_agent.planning.scaffold_map.ownership import OwnershipIndex
from besser.spec_driven_agent.planning.scaffold_map.render import render_locate, render_overview
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex

logger = logging.getLogger(__name__)

__all__ = [
    "Anchor", "EntityMap", "Recipe", "ScaffoldMap", "WorkspaceMap",
    "build_workspace_map", "scaffold_overview", "locate",
]


def _domain_class_names(domain_model: Any) -> list[str]:
    try:
        return [c.name for c in domain_model.get_classes()] if domain_model is not None else []
    except Exception:
        return []


_EXTRACTORS = ("fastapi", "django", "react")


def _build(index: SourceIndex, class_names: list[str]) -> WorkspaceMap:
    """Run every extractor; one that fails (even to import) only loses its own family."""
    ws = WorkspaceMap()
    for name in _EXTRACTORS:
        try:
            module = importlib.import_module(f"{__name__}.{name}")
            ws.maps.extend(module.extract(index, class_names))
        except Exception:
            logger.debug("Scaffold map: %s extractor failed", name, exc_info=True)
            ws.notes.append(f"{name} extractor failed; its scaffold is not mapped")
    try:
        generic = importlib.import_module(f"{__name__}.generic")
        ws.maps.extend(generic.extract(index, class_names, claimed_roots=[m.root for m in ws.maps]))
    except Exception:
        logger.debug("Scaffold map: generic extractor failed", exc_info=True)
        ws.notes.append("generic extractor failed; files outside recognised scaffolds are not mapped")
    ws.broken = [f"{rel}: {err}" for rel, err in sorted(index.syntax_errors.items())]
    ws.notes.extend(index.notes)
    return ws


# Built maps, keyed on the workspace, the model's class names and every
# indexed file's (path, mtime, size): locate_code is called many times per run
# and the files rarely change between two calls. Small, since an entry holds
# the index's file texts (ownership reads them).
_CACHE: "OrderedDict[tuple, tuple[SourceIndex, WorkspaceMap]]" = OrderedDict()
_CACHE_SIZE = 4
_CACHE_LOCK = threading.Lock()
_SETTLE_NS = 2_000_000_000


def _stamp(index: SourceIndex) -> tuple:
    stamps = []
    for rel in index.files:
        try:
            st = os.stat(index.abspath(rel))
        except OSError:
            stamps.append((rel, None))
            continue
        stamps.append((rel, st.st_mtime_ns, st.st_size))
    return tuple(stamps)


def _index_and_map(workspace: str, class_names: list[str]) -> tuple[SourceIndex, WorkspaceMap]:
    """The index and map of ``workspace`` as it is now, rebuilt only when a file changed."""
    index = SourceIndex(workspace)
    stamps = _stamp(index)
    key = (index.workspace, tuple(class_names), stamps)
    # Filesystem timestamps are coarse (a clock tick): two writes inside one
    # tick can leave mtime and size unchanged, so a file changed in the last
    # couple of seconds always forces a rebuild.
    settled = all(s[1] is not None and time.time_ns() - s[1] > _SETTLE_NS for s in stamps)
    with _CACHE_LOCK:
        hit = _CACHE.get(key) if settled else None
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    ws = _build(index, class_names)
    index.drop_trees()
    with _CACHE_LOCK:
        _CACHE[key] = (index, ws)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return index, ws


def build_workspace_map(workspace: str, domain_model: Any = None) -> WorkspaceMap:
    """Every scaffold map in ``workspace``. Never raises; a failing extractor is skipped.

    The result may be shared with later calls (it is cached): do not mutate it.
    """
    try:
        return _index_and_map(workspace, _domain_class_names(domain_model))[1]
    except Exception:
        logger.debug("Scaffold map build failed", exc_info=True)
        return WorkspaceMap()


def scaffold_overview(workspace: str, domain_model: Any = None, max_chars: int = 4000) -> str:
    """Line-free overview, or '' when nothing was recognised. Never raises."""
    try:
        return render_overview(build_workspace_map(workspace, domain_model), max_chars=max_chars)
    except Exception:
        logger.debug("Scaffold overview failed", exc_info=True)
        return ""


def locate(
    workspace: str,
    *,
    entity: Optional[str] = None,
    path: Optional[str] = None,
    query: Optional[str] = None,
    domain_model: Any = None,
    max_chars: int = 7000,
) -> dict:
    """The ``locate_code`` tool's answer, computed from the files as they are now."""
    if path and os.path.isabs(path.strip()):
        # read_file accepts an absolute path inside the workspace; so does this.
        real = os.path.realpath(path.strip())
        root = os.path.realpath(workspace)
        path = os.path.relpath(real, root) if real == root or real.startswith(root + os.sep) else path
    index, ws = _index_and_map(workspace, _domain_class_names(domain_model))
    owner_of = None
    try:
        ownership = OwnershipIndex(workspace, index.lines)
        if ownership.available:
            owner_of = ownership.summarize
    except Exception:
        # The attribution file is model-writable; a bad one only costs the labels.
        logger.debug("locate: ownership unavailable", exc_info=True)
    result = render_locate(
        ws, entity=entity, path=path, query=query, owner_of=owner_of, max_chars=max_chars,
    )
    if ws.notes:
        result["notes"] = ws.notes
    return result
