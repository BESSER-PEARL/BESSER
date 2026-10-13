"""Data model of the scaffold map: where the generated code lives and where new code goes.

A map is computed from the files on disk, never from the generator templates:
the same model yields different files across generator options, BESSER
versions and host tool versions, and Phase 2 edits move everything. Every
location therefore carries two kinds of address:

* a stable, line-free one (``Anchor.id``, a file path and a symbol, route or
  block name) that stays valid while the code is edited, used in the system
  prompt and the gap analyzer, which are built once per run;
* the current line range plus a ``quote`` (one verbatim line, unique in its
  file, to use as ``modify_file``'s ``old_text``), resolved fresh from disk
  each time the ``locate_code`` tool is called.

Extractors (one module per scaffold family) fill these structures; rendering
and ownership live in ``render.py`` and ``ownership.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Anchor kinds.
KIND_FILE = "file"              # a whole file (its role is the label)
KIND_CLASS = "class"            # a class definition
KIND_FUNCTION = "function"      # a function / component definition
KIND_ROUTE = "route"            # an HTTP route handler (decorator through body)
KIND_BLOCK = "block"            # a named block: imports, include_router calls, <Routes>, urlpatterns
KIND_INSERT = "insert"          # an insertion point: new code goes right before/after ``quote``
KIND_STUB = "stub"              # a generated placeholder to implement (501, pass, TODO)


@dataclass
class Anchor:
    """One addressable place in the scaffold.

    ``start_line``/``end_line`` are 1-based and inclusive; a route or
    decorated function starts at its first decorator. ``quote`` is a single
    line copied verbatim from the file (leading whitespace included) that is
    unique in the file, or ``""`` when no line is unique; extractors must not
    invent it.
    """

    id: str
    file: str
    kind: str
    label: str
    start_line: int
    end_line: int
    quote: str = ""
    hint: str = ""
    entity: Optional[str] = None
    status: Optional[str] = None


@dataclass
class EntityMap:
    """Every piece of one domain entity (class) across the scaffold.

    ``missing`` names the pieces the scaffold family normally has for an
    entity but that are absent on disk (e.g. a class added to the model in a
    modify run, which has no router yet). It is derived from code plus the
    domain model, never predicted from templates.
    """

    name: str
    anchors: list[Anchor] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    # Line-free one-liners for the overview, e.g. "routes in backend/routers/book.py:
    # list, read, create, update, delete; links writtenBy". Empty = derived from anchors.
    summary: list[str] = field(default_factory=list)


@dataclass
class Recipe:
    """How to make one common kind of change in this scaffold.

    ``steps`` are short imperative sentences that name files and anchor ids,
    never line numbers, so a recipe stays valid while the code is edited.
    """

    intent: str
    steps: list[str]


@dataclass
class ScaffoldMap:
    """The map of one application root of one scaffold family."""

    family: str                 # "fastapi", "react", "django", "generic"
    root: str                   # workspace-relative app root ("" = workspace root)
    title: str                  # one-line description, e.g. "FastAPI backend (modular routers)"
    entities: dict[str, EntityMap] = field(default_factory=dict)
    anchors: list[Anchor] = field(default_factory=list)       # app-level anchors
    recipes: list[Recipe] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)         # pitfalls found in THIS code
    do_not_edit: list[str] = field(default_factory=list)      # shared runtime files, configured via props
    summary: list[str] = field(default_factory=list)          # line-free app-level one-liners (file roles)

    def all_anchors(self) -> list[Anchor]:
        out = list(self.anchors)
        for entity in self.entities.values():
            out.extend(entity.anchors)
        return out


@dataclass
class WorkspaceMap:
    """Every scaffold map found in a workspace, plus what could not be read."""

    maps: list[ScaffoldMap] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)   # "path: SyntaxError line N: msg" — fix first
    notes: list[str] = field(default_factory=list)    # coverage limits (caps hit, files skipped)

    def all_anchors(self) -> list[Anchor]:
        return [a for m in self.maps for a in m.all_anchors()]

    def entity_names(self) -> list[str]:
        names: list[str] = []
        for m in self.maps:
            for name in m.entities:
                if name not in names:
                    names.append(name)
        return names
