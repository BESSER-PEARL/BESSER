"""Scaffold map extractor for FastAPI applications.

An application root is the directory of a ``.py`` file that assigns
``FastAPI(...)`` at module level; ``backend/`` and ``web_app/backend/`` (the
modular BackendGenerator layout) and ``rest_api/`` (the single-file
RESTAPIGenerator layout) are all found that way, never by the generator that
was asked for. Per root it reads, from the AST only:

* the app file: router imports, the ``include_router`` block, middleware,
  exception handlers, startup hooks, ``@app`` routes, the ``__main__`` guard;
* every ``APIRouter``/``FastAPI`` binding and its decorated routes, resolved
  through the ``include_router`` mounts into the routes actually served
  (first registration wins), so unmounted routers, dead duplicates and
  static routes captured by an earlier ``/{x_id}/`` route are reported;
  module-level mounts and loops over a literal tuple are placed in order,
  and any other ``include_router`` call (in an if/try block or a function)
  makes the mount table unknown, so nothing is then claimed unmounted;
* ORM classes (``__tablename__``, a declarative base, or SQLModel
  ``table=True``), their columns and module-level ``relationship()`` lines;
* Pydantic schemas (``BaseModel`` subclasses), their fields and validators;
* generated method stubs (HTTP 501, ``NotImplementedError``, invalid-code
  wrappers) with the request-body contract their implementation must keep;
* routes whose first statement refuses every non-empty payload (the generated
  bulk guards ``if items: raise HTTPException(422, ...)``), so they are never
  presented as working.

Everything is keyed to domain entities by decorator tags (``"Book"``,
``"Book Relationships"``, ``"Book Methods"``) and path segments, never by file
names, so a router the model wrote itself is attributed the same way.
Missing pieces are judged across every FastAPI root of the workspace, and
the per-entity overview lines leave conventional paths out (the recipes state
the convention once) so a large model still fits the prompt budget.
"""

from __future__ import annotations

import ast
import logging
import posixpath
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from besser.spec_driven_agent.planning.action_inventory import _placeholder_reason
from besser.spec_driven_agent.planning.scaffold_map.model import (
    KIND_BLOCK,
    KIND_CLASS,
    KIND_FILE,
    KIND_FUNCTION,
    KIND_INSERT,
    KIND_ROUTE,
    KIND_STUB,
    Anchor,
    EntityMap,
    Recipe,
    ScaffoldMap,
)
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex, node_start

logger = logging.getLogger(__name__)

FAMILY = "fastapi"

_VERBS = ("get", "post", "put", "patch", "delete", "head", "options")
_APP_FILE_PREFERENCE = ("main_api.py", "main.py", "app.py", "server.py", "api.py")
_TEST_DIRS = frozenset({"tests", "test"})
_APP_ASSIGN_RE = re.compile(r"^(\w+)\s*(?::\s*[\w.]+\s*)?=\s*(?:\w+\.)*FastAPI\(", re.M)
_IDENTITY_COMMENT_RE = re.compile(r"^\s*#\s*(\w+)\.(\w+)\s*$")
_RELATIONSHIP_BANNER_RE = re.compile(r"^#-+\s*Relationships of the (\w+) table\s*$")
_PARAM_RE = re.compile(r"^\{(\w+)(?::(\w+))?\}$")
# ``- percent: float`` in a generated docstring. The type is checked on the stripped
# rest of the line: a lazy group next to ``\s*`` (both match spaces) backtracks
# cubically on a long line of spaces that does not end the way it expects.
_DOC_PARAM_RE = re.compile(r"^\s*-\s*(\w+)\s*:(.*)$")
_DOC_TYPE_RE = re.compile(r"[\w\[\], .|]+")
_HTTP_STATUS_NAME_RE = re.compile(r"^HTTP_(\d{3})_")
_TAG_SUFFIXES = ("Relationships", "Methods")
_SCHEMA_SUFFIXES = ("LinkCreate", "Create", "Update", "Patch", "Read", "Response", "Out", "In", "Base", "Schema")
_VALIDATOR_DECORATORS = frozenset({"field_validator", "model_validator", "validator", "root_validator"})
_COLUMN_CALLS = frozenset({"Column", "Column_", "mapped_column", "Field"})
_RELATIONSHIP_CALLS = frozenset({"relationship", "Relationship"})
_ENGINE_CALLS = frozenset({"create_engine", "create_async_engine"})
# A path parameter with one of these converters never captures a word segment.
_NUMERIC_CONVERTERS = frozenset({"int", "float"})
_MAX_WARNINGS = 8
_MAX_LABEL_NAMES = 8
_STANDARD_CRUD = ("list", "read", "create", "update", "delete")
# Entity anchors are listed in this order (locate_code truncates long answers):
# definitions and the places to edit first, the full route list last.
_ANCHOR_ORDER = {
    "orm": 0, "orm_add_column": 1, "orm_relationships": 2, "schema": 3, "stub": 4, "insert_static": 5,
    "router": 6, "methods_router": 6, "ctor": 7, "setattr": 7, "route": 8,
}
_ROUTE_KIND_ORDER = (
    "list", "count", "paginated", "search", "read", "create", "bulk create", "bulk delete", "update", "delete",
)


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------

def _name(node: Optional[ast.AST]) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _dotted(node: ast.AST) -> str:
    """``a.b.c`` for a Name/Attribute chain, else ''."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _string(node: Optional[ast.AST]) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _keyword(call: ast.Call, name: str) -> Optional[ast.AST]:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _assigned_call(statement: ast.stmt) -> Optional[ast.Call]:
    value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
    return value if isinstance(value, ast.Call) else None


def _targets(statement: ast.stmt) -> list[ast.AST]:
    if isinstance(statement, ast.Assign):
        return list(statement.targets)
    return [statement.target] if isinstance(statement, ast.AnnAssign) else []


def _prefix_of(call: ast.Call) -> Optional[str]:
    """The literal ``prefix=`` of an APIRouter/include_router call; None when not knowable."""
    if any(kw.arg is None for kw in call.keywords):
        return None
    node = _keyword(call, "prefix")
    return "" if node is None else _string(node)


def _join_path(prefix: str, path: str) -> str:
    if not prefix:
        return path
    return prefix.rstrip("/") + ("/" + path.lstrip("/") if path else "")


def _segments(path: str) -> tuple[list[str], bool]:
    return [s for s in path.split("/") if s], path.endswith("/") and path != "/"


def _is_param(segment: str) -> bool:
    return bool(_PARAM_RE.match(segment))


def _captures(earlier: str, later: str) -> bool:
    """Does the earlier route's segment match whatever the later route's segment accepts?"""
    if earlier == later:
        return True
    m = _PARAM_RE.match(earlier)
    if not m:
        return False
    converter = m.group(2) or ""
    if _is_param(later):
        later_conv = _PARAM_RE.match(later).group(2) or ""
        return converter == later_conv or not converter
    if converter in _NUMERIC_CONVERTERS:
        return later.replace(".", "", 1).isdigit()
    return True


def _include_call(statement: ast.stmt) -> Optional[ast.Call]:
    """The ``x.include_router(...)`` call a statement consists of, else None."""
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) \
            and isinstance(statement.value.func, ast.Attribute) and statement.value.func.attr == "include_router":
        return statement.value
    return None


def _substitute(node: Optional[ast.AST], name: str, value: ast.AST) -> Optional[ast.AST]:
    """``node`` with the loop variable ``name`` replaced by ``value`` (Name/Attribute chains only)."""
    if isinstance(node, ast.Name) and node.id == name:
        return value
    if isinstance(node, ast.Attribute):
        inner = _substitute(node.value, name, value)
        return node if inner is node.value else ast.Attribute(value=inner, attr=node.attr, ctx=ast.Load())
    return node


def _function_defs(body: list[ast.stmt]) -> list[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    return [st for st in body if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _docstring(fn: ast.AST) -> str:
    try:
        return ast.get_docstring(fn) or ""
    except TypeError:
        return ""


def _short_list(names: list[str], limit: int = _MAX_LABEL_NAMES) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" +{len(names) - limit}" if len(names) > limit else "")


def _is_under(path: str, root: str) -> bool:
    return not root or path == root or path.startswith(root.rstrip("/") + "/")


def _tag_entity(tag: str) -> str:
    """``"Book Relationships"`` / ``"Book Methods"`` -> ``"Book"``; any other tag stripped.

    String splitting, not ``re.sub(r"\\s+(...)$")``: that regex retries its
    ``\\s+`` from every position of a long whitespace run (quadratic).
    """
    words = tag.strip().rsplit(None, 1)
    return words[0] if len(words) == 2 and words[1] in _TAG_SUFFIXES else tag.strip()


# ---------------------------------------------------------------------------
# Collected facts
# ---------------------------------------------------------------------------

@dataclass
class _Route:
    """One route decorator on one top-level function."""

    file: str
    receiver: str
    verbs: tuple[str, ...]
    path: str
    tag: Optional[str]
    fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    decorator_line: int
    entity: Optional[str] = None
    full_path: Optional[str] = None     # with resolved prefixes; None = not knowable
    mounted: Optional[bool] = None      # None = mount table not knowable
    dead: str = ""                      # why it is never served ('' = served or unknown)

    @property
    def label_path(self) -> str:
        return self.full_path if self.full_path is not None else self.path


@dataclass
class _Include:
    file: str
    statement: ast.stmt
    receiver: str
    target: Optional[tuple[str, str]]   # (file, binding), None = unresolved
    prefix: Optional[str]


@dataclass
class _Binding:
    file: str
    name: str
    kind: str                           # "FastAPI" | "APIRouter"
    prefix: Optional[str]
    statement: ast.stmt
    events: list = field(default_factory=list)   # _Route / _Include in registration order


@dataclass
class _OrmClass:
    file: str
    node: ast.ClassDef
    bases: list[str]
    columns: list[ast.stmt]
    relationships: list[str]            # in-class relationship() names
    table: Optional[str]


@dataclass
class _Schema:
    file: str
    node: ast.ClassDef
    bases: list[str]
    fields: list[str]
    validators: list[str]


# ---------------------------------------------------------------------------
# Root discovery
# ---------------------------------------------------------------------------

def _app_bindings(index: SourceIndex, rel: str) -> list[str]:
    """Names bound to ``FastAPI(...)`` at module level in ``rel`` (regex fallback on broken files)."""
    text = index.text(rel)
    if not text or "FastAPI" not in text:
        return []
    tree = index.python_ast(rel)
    if tree is None:
        return sorted(set(_APP_ASSIGN_RE.findall(text)))
    names: list[str] = []
    for statement in tree.body:
        call = _assigned_call(statement)
        if call is not None and _name(call.func) == "FastAPI":
            names.extend(t.id for t in _targets(statement) if isinstance(t, ast.Name))
    return names


def _find_roots(index: SourceIndex) -> list[tuple[str, str]]:
    """(root, app file) pairs, sorted; one app file per root directory."""
    candidates: dict[str, list[str]] = {}
    for rel in index.with_suffix((".py",)):
        parts = rel.split("/")
        if any(p in _TEST_DIRS for p in parts[:-1]) or parts[-1].startswith("test_"):
            continue
        if _app_bindings(index, rel):
            candidates.setdefault(posixpath.dirname(rel), []).append(rel)
    roots = []
    for root, files in sorted(candidates.items()):
        by_name = {posixpath.basename(f): f for f in files}
        preferred = next((by_name[n] for n in _APP_FILE_PREFERENCE if n in by_name), sorted(files)[0])
        roots.append((root, preferred))
    return roots


def extract(index: SourceIndex, class_names: list[str]) -> list[ScaffoldMap]:
    """One :class:`ScaffoldMap` per FastAPI application root in ``index``.

    Read-only and deterministic; code it cannot parse is skipped (and is
    already reported through ``index.syntax_errors``).
    """
    roots = _find_roots(index)
    used_ids: dict[str, str] = {}      # anchor id -> file, shared so ids stay unique across roots
    mappers: list[_RootMapper] = []
    family_ids: list[str] = []
    for root, app_file in roots:
        nested = [other for other, _ in roots if other != root and _is_under(other, root)]
        files = [f for f in index.with_suffix((".py",), root)
                 if not any(_is_under(posixpath.dirname(f), n) for n in nested)]
        family_id = FAMILY if len(roots) == 1 else f"{FAMILY}@{re.sub(r'[^0-9A-Za-z]+', '_', root) or 'root'}"
        if family_id in family_ids:          # `a_b/` and `a/b/` slug alike
            family_id = next(f"{family_id}_{n}" for n in range(2, len(roots) + 2)
                             if f"{family_id}_{n}" not in family_ids)
        family_ids.append(family_id)
        try:
            mapper = _RootMapper(index, root, app_file, files, class_names, family_id, used_ids)
            mapper.prepare()
            mappers.append(mapper)
        except Exception:  # one unexpected layout must not hide the other roots
            logger.debug("Scaffold map: FastAPI root %r not mapped", root, exc_info=True)
    # "Missing" is a workspace-wide question: a small second app (a proxy, a demo)
    # must not report every entity the main backend serves as missing.
    presence = [mapper.presence() for mapper in mappers]
    built: list[tuple[ScaffoldMap, str]] = []
    for mapper in mappers:
        try:
            built.append((mapper.finish(presence), mapper.family_id))
        except Exception:
            logger.debug("Scaffold map: FastAPI root %r not mapped", mapper.root, exc_info=True)
    _share_recipes(built)
    return [smap for smap, _ in built]


def _share_recipes(built: list[tuple[ScaffoldMap, str]]) -> None:
    """Say a root's recipes once: a later root with the same recipes (modulo its id prefix
    and directory) points back to the first one, keeping the shared prompt budget for entities."""
    seen: list[tuple[str, ScaffoldMap, str]] = []      # (normalised text, map, family id)
    for smap, family_id in built:
        text = "\n".join(f"{r.intent}: {' '.join(r.steps)}" for r in smap.recipes)
        normal = text.replace(f"`{family_id}:", "`<id>:")
        if smap.root:
            normal = normal.replace(f"{smap.root}/", "<root>/")
        earlier = next(((m, fid) for n, m, fid in seen if n == normal), None)
        if earlier is None or not smap.recipes:
            seen.append((normal, smap, family_id))
            continue
        first, first_id = earlier
        smap.recipes = [Recipe(f"Same recipes as `{first.root or '.'}/`", [
            f"with files under `{smap.root or '.'}/` and anchor ids `{family_id}:…` instead of `{first_id}:…`."])]


# ---------------------------------------------------------------------------
# One application root
# ---------------------------------------------------------------------------

class _RootMapper:
    """Collects one root's facts, resolves its mounts and renders its ScaffoldMap."""

    def __init__(self, index: SourceIndex, root: str, app_file: str, files: list[str],
                 class_names: list[str], family_id: str, used_ids: dict[str, str]) -> None:
        self.index = index
        self.root = root
        self.app_file = app_file
        self.files = files
        self.class_names = [n for n in class_names if isinstance(n, str) and n.strip()]
        self.family_id = family_id
        self.used_ids = used_ids
        self.file_set = set(index.files)
        self.trees: dict[str, ast.Module] = {}
        self.broken: list[str] = []
        self.aliases: dict[str, dict[str, tuple[str, Optional[str], int]]] = {}
        self.bindings: dict[tuple[str, str], _Binding] = {}
        self.routes: list[_Route] = []
        self.unbound_routes: list[_Route] = []
        self.includes: list[_Include] = []
        self.orm: list[_OrmClass] = []
        # (file, owner class, statement, position in the module body)
        self.orm_relationships: list[tuple[str, str, ast.stmt, int]] = []
        self.schemas: list[_Schema] = []
        self.canon: dict[str, str] = {}
        self.mounts_known = False
        # include_router calls the mount table cannot place (in an if/try/with block or a
        # function body, on an unknown receiver): with any of them, no "never mounted" claims.
        self.uncertain_mounts: list[str] = []
        self.served: list[_Route] = []
        self.warnings: list[str] = []
        self.extra_warnings = 0
        self.entities: dict[str, EntityMap] = {}
        self.orphans: list[Anchor] = []          # app-level route/file anchors with no entity
        self._by_file: Optional[dict[str, list[_Route]]] = None
        self.common_kinds = ""                   # the route set most entities share
        self.modular = False
        self.app_level: list[Anchor] = []
        self._roles: Optional[dict[str, set[str]]] = None

    # -- ids and quotes ----------------------------------------------------

    def _id(self, *parts: str, file: str = "") -> str:
        """A workspace-unique id; a clash with another file's anchor is told apart by this file's stem."""
        base = ":".join([self.family_id, *[p for p in parts if p]])
        if base in self.used_ids and file and self.used_ids[base] != file:
            base = f"{base}:{posixpath.splitext(posixpath.basename(file))[0]}"
        anchor_id, n = base, 2
        while anchor_id in self.used_ids:
            anchor_id = f"{base}:{n}"
            n += 1
        self.used_ids[anchor_id] = file
        return anchor_id

    def _quote(self, rel: str, *preferred: int, span: Optional[tuple[int, int]] = None) -> str:
        for line_no in preferred:
            quote = self.index.unique_quote(rel, line_no)
            if quote:
                return quote
        return self.index.first_unique_quote(rel, *span) if span else ""

    def _entity_key(self, name: Optional[str]) -> str:
        return (name or "").lower()

    # -- entities ----------------------------------------------------------

    def _lookup(self, name: str) -> Optional[str]:
        return self.canon.get(name.lower()) if name else None

    def _schema_entity(self, name: str) -> Optional[str]:
        """The entity a schema class belongs to: its own name when that is an entity
        (``SurveyResponse``, ``KnowledgeBase``), else the longest suffix-stripped base that is one."""
        for candidate in [name] + _schema_bases(name):
            found = self._lookup(candidate)
            if found:
                return found
        return None

    def _from_segment(self, segment: str) -> Optional[str]:
        found = self._lookup(segment)
        if found is None and segment.endswith("es"):
            found = self._lookup(segment[:-2])
        if found is None and segment.endswith("s"):
            found = self._lookup(segment[:-1])
        return found

    def _entity_of_route(self, route: _Route) -> Optional[str]:
        if route.tag:
            found = self._lookup(_tag_entity(route.tag))
            if found:
                return found
        for segment in _segments(route.label_path)[0]:
            if _is_param(segment):
                break
            found = self._from_segment(segment)
            if found:
                return found
        return None

    # -- collection --------------------------------------------------------

    def _collect(self) -> None:
        for rel in self.files:
            tree = self.index.python_ast(rel)
            if tree is None:
                if rel in self.index.syntax_errors:
                    self.broken.append(rel)
                continue
            if isinstance(tree, ast.Module):
                self.trees[rel] = tree
        for rel, tree in self.trees.items():
            self.aliases[rel] = self._import_aliases(tree)
            self._collect_bindings(rel, tree)
        for rel, tree in self.trees.items():
            self._collect_routes(rel, tree)
        self._collect_orm()
        self._collect_schemas()

    @staticmethod
    def _import_aliases(tree: ast.Module) -> dict[str, tuple[str, Optional[str], int]]:
        """alias -> (module, imported name or None for a module import, relative level)."""
        aliases: dict[str, tuple[str, Optional[str], int]] = {}
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    if alias.name != "*":
                        aliases[alias.asname or alias.name] = (statement.module or "", alias.name, statement.level)
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    if alias.asname:
                        aliases[alias.asname] = (alias.name, None, 0)
                    else:
                        head = alias.name.split(".")[0]
                        aliases[head] = (head, None, 0)
        return aliases

    def _module_file(self, dotted: str, level: int, from_rel: str) -> Optional[str]:
        parts = [p for p in dotted.split(".") if p]
        if not parts:
            return None
        base = posixpath.dirname(from_rel)
        if level:
            for _ in range(level - 1):
                base = posixpath.dirname(base)
            bases = [base]
        else:
            bases = [base]
            while base:
                base = posixpath.dirname(base)
                bases.append(base)
        for directory in bases:
            stem = posixpath.join(directory, *parts) if directory else posixpath.join(*parts)
            for candidate in (stem + ".py", stem + "/__init__.py"):
                if candidate in self.file_set:
                    return candidate
        if not level:
            # A package-absolute import (``myproject.backend.routers.book``) when the
            # workspace starts inside that package: match the longest tail in this root.
            for start in range(1, len(parts)):
                stem = posixpath.join(*parts[start:])
                for candidate in (stem + ".py", stem + "/__init__.py"):
                    for directory in ([self.root, posixpath.dirname(self.root)] if self.root else [""]):
                        found = posixpath.join(directory, candidate) if directory else candidate
                        if found in self.file_set and _is_under(found, self.root):
                            return found
        return None

    def _resolve_router(self, node: Optional[ast.AST], rel: str) -> Optional[tuple[str, str]]:
        """The (file, binding) an ``include_router`` argument refers to, or None."""
        return self._follow_reexports(self._resolve_reference(node, rel))

    def _follow_reexports(self, target: Optional[tuple[str, str]]) -> Optional[tuple[str, str]]:
        """Follow ``from .x import router as x_router`` re-exports (e.g. in a package __init__)."""
        for _ in range(4):
            if target is None or target in self.bindings:
                return target
            file, name = target
            module, imported, level = self.aliases.get(file, {}).get(name, ("", None, 0))
            following = self._module_file(module, level, file) if imported else None
            if following is None:
                return target
            target = (following, imported)
        return target

    def _resolve_reference(self, node: Optional[ast.AST], rel: str) -> Optional[tuple[str, str]]:
        aliases = self.aliases.get(rel, {})
        if isinstance(node, ast.Name):
            if (rel, node.id) in self.bindings:
                return rel, node.id
            if node.id in aliases:
                module, imported, level = aliases[node.id]
                target = self._module_file(module, level, rel) if imported else None
                if target:
                    return target, imported
            return None
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node.value)
            if not dotted:
                return None
            head, _, rest = dotted.partition(".")
            if head in aliases:
                module, imported, level = aliases[head]
                full = ".".join(p for p in (module, imported, rest) if p)
                target = self._module_file(full, level, rel)
            else:
                target = self._module_file(dotted, 0, rel)
            return (target, node.attr) if target else None
        return None

    def _collect_bindings(self, rel: str, tree: ast.Module) -> None:
        for statement in tree.body:
            call = _assigned_call(statement)
            kind = _name(call.func) if call is not None else ""
            if kind not in ("FastAPI", "APIRouter"):
                continue
            prefix = _prefix_of(call) if kind == "APIRouter" else ""
            for target in _targets(statement):
                if isinstance(target, ast.Name):
                    self.bindings[(rel, target.id)] = _Binding(rel, target.id, kind, prefix, statement)

    def _route_decorators(self, rel: str, fn) -> list[_Route]:
        found: list[_Route] = []
        for decorator in fn.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            receiver = decorator.func.value
            if not isinstance(receiver, ast.Name):
                continue
            attr = decorator.func.attr
            if attr in _VERBS:
                verbs: tuple[str, ...] = (attr.upper(),)
            elif attr == "api_route":
                methods = _keyword(decorator, "methods")
                elts = methods.elts if isinstance(methods, (ast.List, ast.Tuple)) else []
                verbs = tuple(sorted({(_string(e) or "").upper() for e in elts} - {""})) or ("GET",)
            else:
                continue
            path = _string(decorator.args[0]) if decorator.args else _string(_keyword(decorator, "path"))
            if path is None:
                continue
            tags = _keyword(decorator, "tags")
            tag = _string(tags.elts[0]) if isinstance(tags, (ast.List, ast.Tuple)) and tags.elts else None
            found.append(_Route(rel, receiver.id, verbs, path, tag, fn, decorator.lineno))
        return found

    def _collect_routes(self, rel: str, tree: ast.Module) -> None:
        placed: set[int] = set()     # include_router calls registered in module order
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                routes = self._route_decorators(rel, statement)
                self.routes.extend(routes)
                # The decorator closest to the def registers first.
                for route in reversed(routes):
                    binding = self.bindings.get((rel, route.receiver))
                    if binding is not None:
                        binding.events.append(route)
                    else:
                        self.unbound_routes.append(route)
            elif _include_call(statement) is not None:
                call = _include_call(statement)
                placed.add(id(call))
                self._add_include(rel, statement, call, call.args[0] if call.args else _keyword(call, "router"))
            elif isinstance(statement, ast.For):
                # `for r in (a_router, b_router): app.include_router(r.router)` mounts each, in order.
                calls = [_include_call(st) for st in statement.body]
                if (calls and None not in calls and not statement.orelse and isinstance(statement.target, ast.Name)
                        and isinstance(statement.iter, (ast.Tuple, ast.List))):
                    for element in statement.iter.elts:
                        for call in calls:
                            argument = call.args[0] if call.args else _keyword(call, "router")
                            self._add_include(rel, statement, call,
                                              _substitute(argument, statement.target.id, element))
                    placed.update(id(call) for call in calls)
        for node in ast.walk(tree):
            if id(node) not in placed and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "include_router":
                self.uncertain_mounts.append(f"{rel}:{node.lineno}")

    def _add_include(self, rel: str, statement: ast.stmt, call: ast.Call, argument: Optional[ast.AST]) -> None:
        receiver = _name(call.func.value)
        include = _Include(rel, statement, receiver, self._resolve_router(argument, rel), _prefix_of(call))
        self.includes.append(include)
        binding = self.bindings.get((rel, receiver)) if isinstance(call.func.value, ast.Name) else None
        if binding is not None:
            binding.events.append(include)
        else:
            self.uncertain_mounts.append(f"{rel}:{call.lineno}")

    def _collect_orm(self) -> None:
        declarative: set[str] = set()
        candidates: list[tuple[str, ast.ClassDef, list[str]]] = []
        for rel, tree in self.trees.items():
            for statement in tree.body:
                call = _assigned_call(statement)
                if call is not None and _name(call.func) in ("declarative_base", "generate_base"):
                    declarative.update(t.id for t in _targets(statement) if isinstance(t, ast.Name))
                if isinstance(statement, ast.ClassDef):
                    bases = [_name(b) for b in statement.bases]
                    if any(b in ("DeclarativeBase", "DeclarativeBaseNoMeta") for b in bases):
                        declarative.add(statement.name)
                    else:
                        candidates.append((rel, statement, bases))
        orm_names: set[str] = set()
        selected: dict[int, tuple[str, ast.ClassDef, list[str]]] = {}
        for _ in range(6):   # fixpoint over base chains (subclasses of mapped classes)
            grew = False
            for rel, node, bases in candidates:
                if id(node) in selected:
                    continue
                table_kw = next((kw.value for kw in node.keywords if kw.arg == "table"), None)
                is_table = isinstance(table_kw, ast.Constant) and table_kw.value is True   # SQLModel
                if _tablename(node) is not None or is_table or any(b in declarative or b in orm_names for b in bases):
                    selected[id(node)] = (rel, node, bases)
                    orm_names.add(node.name)
                    grew = True
            if not grew:
                break
        for rel, node, bases in sorted(selected.values(), key=lambda item: (item[0], item[1].lineno)):
            columns: list[ast.stmt] = []
            relationships: list[str] = []
            for item in node.body:
                target = _targets(item)[0] if _targets(item) else None
                if not isinstance(target, ast.Name) or target.id.startswith("__"):
                    continue
                call = _assigned_call(item)
                if call is not None and _name(call.func) in _RELATIONSHIP_CALLS:
                    relationships.append(target.id)
                elif isinstance(item, ast.AnnAssign) or (call is not None and _name(call.func) in _COLUMN_CALLS):
                    columns.append(item)
            self.orm.append(_OrmClass(rel, node, bases, columns, relationships, _tablename(node)))
        for rel, tree in self.trees.items():
            for position, statement in enumerate(tree.body):
                call = _assigned_call(statement)
                targets = _targets(statement)
                if call is None or _name(call.func) not in _RELATIONSHIP_CALLS or len(targets) != 1:
                    continue
                target = targets[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id in orm_names:
                    self.orm_relationships.append((rel, target.value.id, statement, position))

    def _collect_schemas(self) -> None:
        orm_nodes = {id(c.node) for c in self.orm}
        known: set[str] = {"BaseModel", "SQLModel"}
        candidates = [(rel, st) for rel, tree in self.trees.items() for st in tree.body
                      if isinstance(st, ast.ClassDef) and id(st) not in orm_nodes]
        chosen: dict[int, tuple[str, ast.ClassDef]] = {}
        for _ in range(6):
            grew = False
            for rel, node in candidates:
                if id(node) not in chosen and any(_name(b) in known for b in node.bases):
                    chosen[id(node)] = (rel, node)
                    known.add(node.name)
                    grew = True
            if not grew:
                break
        for rel, node in sorted(chosen.values(), key=lambda item: (item[0], item[1].lineno)):
            fields = [item.target.id for item in node.body
                      if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)]
            validators = []
            for fn in _function_defs(node.body):
                names = {_name(d.func) if isinstance(d, ast.Call) else _name(d) for d in fn.decorator_list}
                if names & _VALIDATOR_DECORATORS:
                    validators.append(fn.name)
            self.schemas.append(_Schema(rel, node, [_name(b) for b in node.bases], fields, validators))

    def _resolve_entities(self) -> None:
        for name in self.class_names:
            self.canon.setdefault(name.lower(), name)
        for cls in self.orm:
            self.canon.setdefault(cls.node.name.lower(), cls.node.name)
        # A schema named after a route segment or tag (the single-file layout keeps
        # one pydantic class per entity and no ORM) names an entity too.
        words: set[str] = set()
        for route in self.routes:
            words.update(s.lower() for s in _segments(route.label_path)[0] if not _is_param(s))
            if route.tag:
                words.add(route.tag.lower())
        for schema in self.schemas:
            # The full name first: ``SurveyResponse`` is an entity of its own, not Survey's response.
            for candidate in [schema.node.name] + _schema_bases(schema.node.name):
                if candidate.lower() in words:
                    self.canon.setdefault(candidate.lower(), candidate)
                    break
        for route in self.routes + self.unbound_routes:
            route.entity = self._entity_of_route(route)

    # -- mounts and the effective route table ------------------------------

    def _app_binding(self) -> Optional[_Binding]:
        apps = [b for (rel, _), b in sorted(self.bindings.items()) if rel == self.app_file and b.kind == "FastAPI"]
        return next((b for b in apps if b.name == "app"), apps[0] if apps else None)

    def _resolve_mounts(self) -> None:
        visited: set[tuple[str, str]] = set()
        app = self._app_binding()
        if app is not None:
            self.mounts_known = True
            self._walk_mounts(app, visited)
        if self.uncertain_mounts:
            self.mounts_known = False
        for binding in self.bindings.values():
            if binding.kind != "APIRouter" or (binding.file, binding.name) in visited:
                continue
            for event in binding.events:
                if isinstance(event, _Route):
                    event.mounted = False if self.mounts_known else None
                    if binding.prefix is not None:
                        event.full_path = _join_path(binding.prefix, event.path)

    def _walk_mounts(self, app: _Binding, visited: set[tuple[str, str]]) -> None:
        """Register the app's routes and mounted routers in order, resolving prefixes."""

        def walk(binding: _Binding, outer: Optional[str], depth: int) -> None:
            key = (binding.file, binding.name)
            if key in visited or depth > 6:
                return
            visited.add(key)
            own = binding.prefix if binding.kind == "APIRouter" else ""
            base = None if outer is None or own is None else outer + own
            for event in binding.events:
                if isinstance(event, _Route):
                    event.mounted = True
                    event.full_path = None if base is None else _join_path(base, event.path)
                    self.served.append(event)
                    continue
                child = self.bindings.get(event.target) if event.target else None
                if child is None:
                    self.mounts_known = False
                    continue
                inner = None if base is None or event.prefix is None else base + event.prefix
                walk(child, inner, depth + 1)

        walk(app, "", 0)

    def _detect_conflicts(self) -> None:
        """First registration wins: mark later duplicates and captured static routes as dead."""
        registered: list[tuple[str, list[str], bool, _Route]] = []
        for route in self.served:
            if route.full_path is None:
                continue
            segments, slash = _segments(route.full_path)
            for verb in route.verbs:
                winner = None
                for w_verb, w_segments, w_slash, w_route in registered:
                    if (w_verb == verb and w_slash == slash and len(w_segments) == len(segments)
                            and (w_route.file, w_route.fn.lineno) != (route.file, route.fn.lineno)
                            and all(_captures(a, b) for a, b in zip(w_segments, segments))):
                        winner = (w_segments, w_route)
                        break
                if winner is None:
                    registered.append((verb, segments, slash, route))
                    continue
                w_segments, w_route = winner
                duplicate = all(a == b or (_is_param(a) and _is_param(b)) for a, b in zip(w_segments, segments))
                where = f"{w_route.file} {w_route.fn.name}"
                if duplicate:
                    route.dead = f"never served: {where} registered earlier serves {verb} {route.full_path}"
                    self._warn(f"{route.file} {route.fn.name} ({verb} {route.full_path}) is dead code: {where} "
                               "is registered earlier for the same route and wins; edit that handler instead.")
                else:
                    outcome = self._capture_outcome(w_route, w_segments, segments)
                    route.dead = f"never served: captured by {verb} {w_route.full_path} ({where})"
                    if w_route.file == route.file:
                        fix = f"declare it above {w_route.fn.name}"
                    elif route.file == self.app_file:
                        fix = f"move it into {w_route.file} above {w_route.fn.name}"
                    else:
                        fix = f"mount its router before {w_route.file}'s or change the path"
                    self._warn(f"{route.file} {route.fn.name} ({verb} {route.full_path}) is unreachable: "
                               f"{verb} {w_route.full_path} in {where} is registered earlier and captures it "
                               f"({outcome}); {fix}.")
                break

    @staticmethod
    def _capture_outcome(winner: _Route, w_segments: list[str], segments: list[str]) -> str:
        for a, b in zip(w_segments, segments):
            m = _PARAM_RE.match(a)
            if m and a != b and not _is_param(b):
                kind = _annotation_of(winner.fn, m.group(1))
                if kind in ("int", "float"):
                    return f"requests get 422: '{b}' is not {_article(kind)}"
                if kind == "str":
                    return f"requests run {winner.fn.name} with '{b}' as the key, typically a misleading 404"
                return f"requests run {winner.fn.name}"
        return f"requests run {winner.fn.name}"

    def _warn(self, text: str) -> None:
        if text in self.warnings:
            return
        if len(self.warnings) < _MAX_WARNINGS:
            self.warnings.append(text)
        else:
            self.extra_warnings += 1

    def _unmounted_warnings(self) -> None:
        for (rel, name), binding in sorted(self.bindings.items()) if self.mounts_known else []:
            routes = [e for e in binding.events if isinstance(e, _Route)]
            if binding.kind != "APIRouter" or not routes or routes[0].mounted is not False:
                continue
            sample = f"{routes[0].verbs[0]} {routes[0].label_path}"
            self._warn(f"{rel}: APIRouter `{name}` is never mounted with include_router in {self.app_file}, "
                       f"so its {len(routes)} route(s) (e.g. {sample}) return 404.")
        # Independent of the mount table: `app` is not even a name in that file.
        for route in self.unbound_routes:
            if route.file == self.app_file or route.receiver != "app":
                continue
            if route.receiver in self.aliases.get(route.file, {}):
                continue
            self._warn(f"{route.file} {route.fn.name} is decorated with @{route.receiver}, which "
                       f"{route.file} does not define: use the file's own @router.")

    # -- anchors -----------------------------------------------------------

    def prepare(self) -> None:
        """Collect this root's facts and anchors (everything but the cross-root parts)."""
        self._collect()
        self._resolve_mounts()
        self._resolve_entities()
        self._detect_conflicts()
        self._unmounted_warnings()
        if self.extra_warnings:
            self.warnings.append(f"… and {self.extra_warnings} more route problem(s): "
                                 "locate_code query=\"never served\" lists them.")
        self.modular = any(b.kind == "APIRouter" for b in self.bindings.values()) or bool(self.includes)
        self._entity_anchors()
        self.app_level = self._app_anchors() + self.orphans

    def presence(self) -> dict[str, set[str]]:
        """Entities that have each per-entity piece in this root ("orm", "schema", "routes")."""
        return {
            "orm": {self._lookup(c.node.name) or c.node.name for c in self.orm},
            "schema": {self._schema_entity(s.node.name) for s in self.schemas} - {None},
            "routes": {r.entity for r in self.routes if r.entity},
        }

    def finish(self, presence: list[dict[str, set[str]]]) -> ScaffoldMap:
        """The ScaffoldMap, with missing pieces judged against every root's ``presence``."""
        modular = self.modular
        smap = ScaffoldMap(
            family=FAMILY,
            root=self.root,
            title=("FastAPI backend (modular routers)" if modular
                   else "FastAPI app (single file, @app routes)"),
        )
        smap.anchors = list(self.app_level)
        self._missing(modular, presence)
        common = self.common_kinds = self._common_kinds()
        for name in sorted(self.entities, key=lambda n: (n.lower(), n)):
            entity = self.entities[name]
            entity.anchors = [a for _, _, a in sorted(
                (_ANCHOR_ORDER.get(a.id[len(self.family_id) + 1:].split(":", 1)[0], 9), n, a)
                for n, a in enumerate(entity.anchors))]
            entity.summary = self._entity_summary(name, common)
            smap.entities[name] = entity
        smap.warnings = list(self.warnings)
        smap.summary = self._map_summary(modular, smap)
        smap.recipes = self._recipes(modular, smap)
        return smap

    def build(self) -> ScaffoldMap:
        """This root alone (as if it were the only one in the workspace)."""
        self.prepare()
        return self.finish([self.presence()])

    def _entity(self, name: str) -> EntityMap:
        if name not in self.entities:
            self.entities[name] = EntityMap(name=name)
        return self.entities[name]

    def _add(self, entity: Optional[str], anchor: Anchor, app_level: list[Anchor]) -> None:
        if entity:
            self._entity(entity).anchors.append(anchor)
        else:
            app_level.append(anchor)

    def _entity_anchors(self) -> None:
        """ORM, schema, router, stub and insertion anchors per entity, in a fixed order."""
        for cls in self.orm:
            self._orm_anchors(cls)
        self._relationship_anchors()
        for schema in self.schemas:
            entity = self._schema_entity(schema.node.name)
            if entity:
                self._schema_anchor(schema, entity)
        self._router_anchors()

    # ORM -------------------------------------------------------------------

    def _orm_anchors(self, cls: _OrmClass) -> None:
        entity = self._lookup(cls.node.name) or cls.node.name
        key = self._entity_key(entity)
        rel, node = cls.file, cls.node
        start = node_start(node)
        column_names = [_targets(c)[0].id for c in cls.columns]
        parent = next((b for b in cls.bases if b in {c.node.name for c in self.orm}), None)
        hint_parts = []
        if parent:
            hint_parts.append(f"inherits {parent}: inherited fields are re-emitted in this class's router")
        if _function_defs(node.body) and self._methods_router_of(entity):
            hint_parts.append("the methods defined here are copies; the HTTP endpoints live in "
                              f"{self._methods_router_of(entity)}")
        self._entity(entity).anchors.append(Anchor(
            id=self._id("orm", key, file=rel), file=rel, kind=KIND_CLASS,
            label=f"ORM class {node.name} ({_short_list(column_names) or 'no columns'})",
            start_line=start, end_line=node.end_lineno or start,
            quote=self._quote(rel, node.lineno, span=(start, node.end_lineno or start)),
            hint="; ".join(hint_parts), entity=entity,
        ))
        # Where a new column goes: after the last single-line column, else after
        # __tablename__, else after the class header.
        candidates: list[int] = []
        if cls.columns and cls.columns[-1].lineno == cls.columns[-1].end_lineno:
            candidates.append(cls.columns[-1].lineno)
        table_stmt = next((item for item in node.body if _tablename_stmt(item)), None)
        if table_stmt is not None:
            candidates.append(table_stmt.lineno)
        candidates.append(node.lineno)
        for line_no in candidates:
            quote = self.index.unique_quote(rel, line_no)
            if quote:
                self._entity(entity).anchors.append(Anchor(
                    id=self._id("orm_add_column", key, file=rel), file=rel, kind=KIND_INSERT,
                    label=f"new {node.name} columns go after this line",
                    start_line=line_no, end_line=line_no, quote=quote,
                    hint="insert new columns AFTER this line, before any method; "
                         "no migrations: create_all never alters an existing table",
                    entity=entity,
                ))
                break

    def _relationship_anchors(self) -> None:
        """One block per run of consecutive ``Owner.role = relationship(...)`` statements.

        An association class interleaves its owners' lines (``Reservation.seats``,
        ``Seat.reservations``, ``Reservation.trips``…), so one min..max range per
        owner would cover other entities' relationships; each run is its own
        anchor, the first keeps the plain id and later ones add their first role.
        """
        runs: list[tuple[str, str, list[ast.stmt]]] = []     # (file, owner, statements)
        previous: Optional[tuple[str, str, int]] = None
        for rel, owner, statement, position in sorted(self.orm_relationships, key=lambda item: (item[0], item[3])):
            if previous == (rel, owner, position - 1):
                runs[-1][2].append(statement)
            else:
                runs.append((rel, owner, [statement]))
            previous = (rel, owner, position)
        tables = {c.table: c.node.name for c in self.orm if c.table}
        banners: dict[tuple[str, str], int] = {}
        for rel in sorted({rel for rel, _, _ in runs}):
            for n, line in enumerate(self.index.lines(rel), 1):
                m = _RELATIONSHIP_BANNER_RE.match(line)
                if m and m.group(1) in tables:
                    banners.setdefault((rel, tables[m.group(1)]), n)
        first_run: set[tuple[str, str]] = set()
        for rel, owner, statements in runs:
            entity = self._lookup(owner) or owner
            start = statements[0].lineno
            end = statements[-1].end_lineno or statements[-1].lineno
            banner = banners.get((rel, owner))
            if banner is not None and banner < start and start - banner <= 3:
                start = banner
            roles = [_targets(st)[0].attr for st in statements]
            later = (rel, owner) in first_run
            first_run.add((rel, owner))
            self._entity(entity).anchors.append(Anchor(
                id=self._id("orm_relationships", self._entity_key(entity), roles[0] if later else "", file=rel),
                file=rel, kind=KIND_BLOCK,
                label=f"relationships of {owner}: {_short_list(roles)}",
                start_line=start, end_line=end, quote=self._quote(rel, span=(start, end)),
                hint="module-level relationship() lines; N:M links are rows of the secondary table",
                entity=entity,
            ))

    # Schemas ---------------------------------------------------------------

    def _schema_anchor(self, schema: _Schema, entity: str) -> None:
        rel, node = schema.file, schema.node
        start = node_start(node)
        label = f"schema {node.name} ({_short_list(schema.fields) or 'no fields'}"
        label += f"; validators: {_short_list(schema.validators, 4)})" if schema.validators else ")"
        hint = "add fields after the last field, validators after the fields"
        usage = self._schema_usage(node.name)
        if usage:
            hint += f"; {usage}"
        imports = " ".join(self.index.lines(rel)[:40])
        if "model_validator" not in imports and "pydantic" in imports:
            hint += "; @model_validator needs `model_validator` added to the pydantic import"
        self._entity(entity).anchors.append(Anchor(
            id=self._id("schema", self._entity_key(entity), node.name, file=rel), file=rel, kind=KIND_CLASS,
            label=label, start_line=start, end_line=node.end_lineno or start,
            quote=self._quote(rel, node.lineno, span=(start, node.end_lineno or start)),
            hint=hint, entity=entity,
        ))

    def _schema_usage(self, name: str) -> str:
        """Which handlers take the schema as a parameter, else which schemas nest it ('' = neither).

        Derived from the code, not the layout: a read-only entity has no PUT,
        and an ``<Assoc>LinkCreate`` is only nested in other schemas' fields.
        """
        handlers: list[str] = []
        seen: set[int] = set()
        for route in self.routes:
            if id(route.fn) in seen:
                continue
            seen.add(id(route.fn))
            args = route.fn.args.args + route.fn.args.kwonlyargs
            if any(arg.annotation is not None and _mentions(arg.annotation, name) for arg in args):
                handlers.append(route.fn.name)
        if handlers:
            return f"request body of {_short_list(handlers, 4)}"
        nested = [f"{schema.node.name}.{item.target.id}" for schema in self.schemas for item in schema.node.body
                  if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                  and schema.node.name != name and _mentions(item.annotation, name)]
        return f"nested in {_short_list(nested, 3)}" if nested else ""

    # Routers ---------------------------------------------------------------

    def _router_files(self) -> dict[str, list[_Route]]:
        if self._by_file is None:
            self._by_file = {}
            for route in self.routes:
                self._by_file.setdefault(route.file, []).append(route)
        return self._by_file

    def _file_entity(self, routes: list[_Route]) -> Optional[str]:
        counts: dict[str, int] = {}
        for route in routes:
            if route.entity:
                counts[route.entity] = counts.get(route.entity, 0) + 1
        if not counts:
            return None
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def _is_methods_router(self, routes: list[_Route]) -> bool:
        return bool(routes) and all(
            (r.tag or "").endswith(" Methods") or "/methods/" in r.path for r in routes)

    def _methods_router_of(self, entity: str) -> Optional[str]:
        for rel, routes in sorted(self._router_files().items()):
            if rel != self.app_file and self._file_entity(routes) == entity and self._is_methods_router(routes):
                return rel
        return None

    def _router_anchors(self) -> None:
        enum_clash = self._enum_clash()

        def claim_order(item: tuple[str, list[_Route]]) -> tuple[int, str]:
            rel, routes = item
            stem = posixpath.splitext(posixpath.basename(rel))[0].lower()
            entity = (self._file_entity(routes) or "").lower()
            return (0 if stem in (entity, f"{entity}_methods") else 1), rel

        for rel, routes in sorted(self._router_files().items(), key=claim_order):
            functions: dict[int, list[_Route]] = {}
            for route in routes:
                functions.setdefault(route.fn.lineno, []).append(route)
            file_entity = self._file_entity(routes) if rel != self.app_file else None
            methods = self._is_methods_router(routes)
            if rel != self.app_file:
                self._router_file_anchor(rel, routes, file_entity, methods)
            if rel != self.app_file or not any(b.kind == "APIRouter" for b in self.bindings.values()):
                self._insert_static_anchors(rel, routes)
            for _, group in sorted(functions.items()):
                self._route_anchor(rel, group, enum_clash)

    def _router_file_anchor(self, rel: str, routes: list[_Route], entity: Optional[str], methods: bool) -> None:
        mounted = [r for r in routes if r.mounted]
        binding = next((self.bindings[(rel, r.receiver)] for r in routes if (rel, r.receiver) in self.bindings), None)
        what = "method endpoints" if methods else "routes"
        state = "" if not routes or routes[0].mounted is None else (
            " (mounted)" if mounted else " (NOT MOUNTED: its routes 404)")
        quote = self._quote(rel, binding.statement.lineno) if binding is not None else ""
        key = self._entity_key(entity) if entity else posixpath.splitext(posixpath.basename(rel))[0]
        anchor = Anchor(
            id=self._id("methods_router" if methods else "router", key, file=rel), file=rel, kind=KIND_FILE,
            label=f"{'methods router' if methods else 'router'} {rel}: {len({id(r.fn) for r in routes})} {what}{state}",
            start_line=1, end_line=max(1, len(self.index.lines(rel))), quote=quote,
            hint="use this file's @router (never @app); router-wide dependencies go in its APIRouter(...)",
            entity=entity,
        )
        self._add(entity, anchor, self.orphans)

    def _insert_static_anchors(self, rel: str, routes: list[_Route]) -> None:
        """Per entity, the first ``/<entity>/{x}/`` route: static siblings must be declared above it."""
        seen: set[str] = set()
        for route in sorted(routes, key=lambda r: node_start(r.fn)):
            entity = route.entity
            if not entity or entity in seen:
                continue
            segments, _ = _segments(route.label_path)
            if len(segments) != 2 or _is_param(segments[0]):
                continue
            m = _PARAM_RE.match(segments[1])
            if not m or (m.group(2) or "") in _NUMERIC_CONVERTERS:
                continue
            seen.add(entity)
            # New code goes above the handler's FIRST decorator: inserting between an
            # extra decorator (e.g. @audit_log) and the route decorator would re-target it.
            line = node_start(route.fn)
            quote = self.index.unique_quote(rel, line)
            end = line
            where = "BEFORE this line"
            if not quote:
                end = max(line, route.decorator_line)
                quote = self.index.first_unique_quote(rel, line, end)
                where = f"BEFORE this handler's first decorator (`{self.index.lines(rel)[line - 1].strip()}`)"
            kind = _annotation_of(route.fn, m.group(1))
            if kind in ("int", "float"):
                outcome = f"its {{{m.group(1)}}} is {_article(kind)}, so the static path fails with 422"
            elif kind == "str":
                outcome = (f"its {{{m.group(1)}}} is a str, so {route.fn.name} answers instead, "
                           "typically with a misleading 404")
            else:
                outcome = f"{route.fn.name} answers instead"
            example = f"/{segments[0]}/stats/"
            self._entity(entity).anchors.append(Anchor(
                id=self._id("insert_static", self._entity_key(entity), file=rel), file=rel, kind=KIND_INSERT,
                label=f"static {segments[0]} routes go above {route.verbs[0]} {route.label_path}",
                start_line=line, end_line=end, quote=quote,
                hint=(f"insert new static routes (e.g. GET {example}) {where}: one placed after "
                      f"{route.fn.name} is captured by {route.label_path}; {outcome}"),
                entity=entity,
            ))

    def _route_anchor(self, rel: str, group: list[_Route], enum_clash: list[str]) -> None:
        route = group[0]
        fn = route.fn
        start, end = node_start(fn), fn.end_lineno or fn.lineno
        entity = route.entity
        key = self._entity_key(entity) if entity else ""
        label = f"{','.join(route.verbs)} {route.label_path} -> {fn.name}"
        if len(group) > 1:
            label += f" (+{len(group) - 1} more path)"
        stub = _placeholder_reason(fn)
        refusal = _refusal(fn)
        status = None
        hint = ""
        dead = next((r.dead for r in group if r.dead), "")
        if dead:
            status, hint = "never served", dead
        elif route.mounted is False:
            status = "not mounted (404)"
        elif stub:
            status = f"stub ({stub})"
        elif refusal:
            code, param = refusal
            status = f"refuses every payload: HTTP {code}"
            hint = (f"its first statement raises HTTP {code} for any non-empty `{param}`, so the code below "
                    "never runs; supporting this route means replacing that guard")
        quote = self._quote(rel, fn.lineno, *[d.lineno for d in fn.decorator_list], span=(start, end))
        anchor = Anchor(id=self._id("route", key, fn.name, file=rel), file=rel, kind=KIND_ROUTE, label=label,
                        start_line=start, end_line=end, quote=quote, hint=hint, entity=entity, status=status)
        anchors = [anchor]
        stubs = self._stub_anchors(rel, route, stub, enum_clash)
        if stubs and anchor.status is None:
            anchor.status = f"stub ({stubs[0].status})"
        anchors.extend(stubs)
        anchors.extend(self._constructor_anchors(rel, route))
        for item in anchors:
            self._add(entity, item, self.orphans)

    def _stub_anchors(self, rel: str, route: _Route, stub: Optional[str], enum_clash: list[str]) -> list[Anchor]:
        fn = route.fn
        segments = _segments(route.path)[0]
        method = segments[segments.index("methods") + 1] if "methods" in segments[:-1] else fn.name
        key = self._entity_key(route.entity) if route.entity else ""
        out: list[Anchor] = []
        if stub:
            terminal = _terminal_statement(fn.body)
            if terminal is None:
                return out
            start, end = terminal.lineno, terminal.end_lineno or terminal.lineno
            lines = self.index.lines(rel)
            comment = start - 1 if start >= 2 and _IDENTITY_COMMENT_RE.match(lines[start - 2]) else None
            start = comment or start
            quote = self._quote(rel, *(n for n in (comment,) if n), span=(start, end))
            if not quote:   # e.g. a bare `pass`: anchor on the handler itself
                start = node_start(fn)
                quote = self._quote(rel, fn.lineno, span=(start, end))
            out.append(Anchor(
                id=self._id("stub", key, method, file=rel), file=rel, kind=KIND_STUB,
                label=f"{route.verbs[0]} {route.label_path} -> {fn.name}: replace the placeholder",
                start_line=start, end_line=end, quote=quote,
                hint=self._stub_hint(fn, method, route.entity, enum_clash),
                entity=route.entity, status=_stub_status(stub),
            ))
            return out
        inner = _invalid_code_stub(fn)
        if inner is not None:
            raise_line = inner.body[-1].lineno
            out.append(Anchor(
                id=self._id("stub", key, method, file=rel), file=rel, kind=KIND_STUB,
                label=f"{route.verbs[0]} {route.label_path} -> {fn.name}: method body did not compile",
                start_line=node_start(inner), end_line=inner.end_lineno or raise_line,
                quote=self._quote(rel, raise_line, span=(node_start(inner), inner.end_lineno or raise_line)),
                hint="the modelled code was invalid and was commented out; write the body here "
                     "(the endpoint answers 500 until then)",
                entity=route.entity, status="invalid method code (NotImplementedError)",
            ))
        return out

    def _entity_enums(self, entity: Optional[str], enum_clash: list[str]) -> list[str]:
        """The colliding enums the entity's ORM class actually uses."""
        used: set[str] = set()
        for cls in self.orm:
            if (self._lookup(cls.node.name) or cls.node.name) == entity:
                used.update(n.id for n in ast.walk(cls.node) if isinstance(n, ast.Name))
        return [name for name in enum_clash if name in used]

    def _stub_hint(self, fn, method: str, entity: Optional[str], enum_clash: list[str]) -> str:
        args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        parts = []
        instance = next((t.id for st in ast.walk(fn) if isinstance(st, ast.Assign) for t in st.targets
                         if isinstance(t, ast.Name) and t.id.startswith("_") and t.id.endswith("_object")), None)
        if instance:
            parts.append(f"the instance is `{instance}`")
        session = next((a for a in args if a in ("database", "db", "session")), None)
        if session:
            parts.append(f"the session `{session}`")
        if "params" in args:
            doc = _docstring(fn)
            declared = _declared_params(doc)
            if declared:
                names = ", ".join(f"{n}: {t}" for n, t in declared)
                first = declared[0][0]
                parts.append(f'the body is {{"params": {{...}}}} (declared {names}): read '
                             f"`(params or {{}}).get('{first}')`, params may be None")
            elif "no parameters" in doc:
                parts.append(f"{method} declares no parameters, so it must succeed on an empty body {{}}")
            else:
                parts.append('the body is {"params": {...}}: read `(params or {}).get(...)`; '
                             "a zero-parameter method must succeed on {}")
        parts.append("commit after changes and return a JSON-serialisable dict; keep the route and function name")
        enums = self._entity_enums(entity, enum_clash)
        if enums:
            parts.append(f"{', '.join(enums[:3])} here is the ORM enum (star imports): compare `.name`")
        return "; ".join(parts)

    def _constructor_anchors(self, rel: str, route: _Route) -> list[Anchor]:
        """Where a new field goes inside create/bulk-create (ORM kwargs) and update (setattr) handlers."""
        kind = self._route_kind(route)[0]
        if kind not in ("create", "bulk create", "update") or not route.entity or _refusal(route.fn):
            return []      # a handler refusing every payload never reaches its constructor
        fn = route.fn
        key = self._entity_key(route.entity)
        orm_names = {c.node.name for c in self.orm}
        out: list[Anchor] = []
        if kind in ("create", "bulk create"):
            for node in ast.walk(fn):
                call = node.value if isinstance(node, ast.Assign) else None
                if isinstance(call, ast.Call) and _name(call.func) in orm_names and call.keywords:
                    kw_lines = sorted({kw.value.lineno for kw in call.keywords if kw.arg})
                    one_line = len(kw_lines) == 1
                    quote = self._quote(rel, *kw_lines, span=(call.lineno, call.end_lineno or call.lineno))
                    hint = ("every kwarg sits on this ONE physical line: add `<field>=<data>.<field>,` inside it"
                            if one_line else "add a `<field>=<data>.<field>,` kwarg next to the quoted one")
                    out.append(Anchor(
                        id=self._id("ctor", key, fn.name, file=rel), file=rel, kind=KIND_INSERT,
                        label=f"{_name(call.func)}(...) constructor in {fn.name}",
                        start_line=call.lineno, end_line=call.end_lineno or call.lineno, quote=quote,
                        hint=hint, entity=route.entity,
                    ))
                    break
        else:
            # Only direct statements of the handler: the FK / N:M reconciliation sits in if-blocks.
            setattrs = [st.value for st in fn.body
                        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                        and _name(st.value.func) == "setattr" and len(st.value.args) >= 2]
            if setattrs:
                last = max(setattrs, key=lambda node: node.lineno)
                target = ast.unparse(last.args[0])
                quote = self.index.unique_quote(rel, last.lineno)
                if quote:
                    out.append(Anchor(
                        id=self._id("setattr", key, fn.name, file=rel), file=rel, kind=KIND_INSERT,
                        label=f"field updates in {fn.name}",
                        start_line=last.lineno, end_line=last.lineno, quote=quote,
                        hint=f"add `setattr({target}, '<field>', <data>.<field>)` AFTER this line, "
                             "or PUT silently drops the field",
                        entity=route.entity,
                    ))
        return out

    # App-level -------------------------------------------------------------

    def _app_anchors(self) -> list[Anchor]:
        rel = self.app_file
        tree = self.trees.get(rel)
        out: list[Anchor] = []
        if tree is not None:
            out.extend(self._app_file_anchors(rel, tree))
        out.extend(self._database_anchors())
        requirements = posixpath.join(self.root, "requirements.txt") if self.root else "requirements.txt"
        if requirements in self.file_set:
            out.extend(self._requirements_anchor(requirements))
        return out

    def _app_file_anchors(self, rel: str, tree: ast.Module) -> list[Anchor]:
        out: list[Anchor] = []
        app = self._app_binding()
        app_name = app.name if app is not None else "app"
        router_files = {inc.target[0] for inc in self.includes if inc.target and inc.file == rel}
        router_imports = [
            statement for statement in tree.body
            if isinstance(statement, (ast.Import, ast.ImportFrom))
            and any(self._import_files(rel, a.asname or a.name.split(".")[0]) & router_files
                    for a in statement.names)
        ]
        if router_imports:
            last = router_imports[-1]
            start, end = last.lineno, last.end_lineno or last.lineno
            out.append(Anchor(
                id=self._id("router_imports_end"), file=rel, kind=KIND_INSERT,
                label="router imports end", start_line=start, end_line=end,
                quote=self._quote(rel, end, span=(start, end)),
                hint=("add a new router import AFTER this line, in the same form" if start == end
                      else "add a new router import AFTER this import statement (or inside its parentheses)"),
            ))
        if app is not None:
            statement = app.statement
            start, end = statement.lineno, statement.end_lineno or statement.lineno
            call = _assigned_call(statement)
            lifespan = call is not None and _keyword(call, "lifespan") is not None
            out.append(Anchor(
                id=self._id("app"), file=rel, kind=KIND_BLOCK, label=f"{app_name} = FastAPI(...)",
                start_line=start, end_line=end, quote=self._quote(rel, start, span=(start, end)),
                hint=("lifespan= is set" if lifespan
                      else "no lifespan= yet: startup work goes in an on_event('startup') hook or a lifespan= here"),
            ))
        own_includes = [inc for inc in self.includes if inc.file == rel and inc.receiver == app_name]
        if own_includes:
            first, last = own_includes[0].statement, own_includes[-1].statement
            start, end = first.lineno, last.end_lineno or last.lineno
            out.append(Anchor(
                id=self._id("include_routers"), file=rel, kind=KIND_BLOCK,
                label=f"{len(own_includes)} include_router call(s)", start_line=start, end_line=end,
                quote=self._quote(rel, span=(start, end)),
                hint="mount order decides which handler serves a duplicated or captured path: the first wins; "
                     "router-wide auth: include_router(..., dependencies=[Depends(...)])",
            ))
            if isinstance(last, ast.For):
                # The mounts run in a loop: its last line is indented loop body.
                loop_start = last.lineno
                out.append(Anchor(
                    id=self._id("include_routers_end"), file=rel, kind=KIND_INSERT,
                    label="the include_router loop", start_line=loop_start, end_line=end,
                    quote=self._quote(rel, loop_start, span=(loop_start, end)),
                    hint="mount a new router by adding it to this loop's tuple, in mount order; one serving "
                         "static paths under an existing /<entity>/ prefix must come BEFORE that entity's router",
                ))
            else:
                out.append(Anchor(
                    id=self._id("include_routers_end"), file=rel, kind=KIND_INSERT,
                    label="after the last include_router", start_line=end, end_line=end,
                    quote=self._quote(rel, end),
                    hint="mount a new router AFTER this line; one serving static paths under an existing "
                         "/<entity>/ prefix must be mounted BEFORE that entity's router",
                ))
        middleware: list[tuple[int, int, str]] = []
        handlers: list[tuple[int, int, str]] = []
        hooks: list[ast.AST] = []
        for statement in tree.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) \
                    and _name(statement.value.func) == "add_middleware":
                cls = _name(statement.value.args[0]) if statement.value.args else "middleware"
                middleware.append((statement.lineno, statement.end_lineno or statement.lineno, cls))
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in statement.decorator_list:
                attr = _name(decorator.func) if isinstance(decorator, ast.Call) else _name(decorator)
                if attr == "middleware":
                    middleware.append((node_start(statement), statement.end_lineno or statement.lineno,
                                       statement.name))
                elif attr == "exception_handler":
                    exc = decorator.args[0] if isinstance(decorator, ast.Call) and decorator.args else None
                    handlers.append((node_start(statement), statement.end_lineno or statement.lineno,
                                     _name(exc) or statement.name))
                elif attr == "on_event":
                    hooks.append(statement)
        if middleware:
            start, end = min(m[0] for m in middleware), max(m[1] for m in middleware)
            names = [m[2] for m in middleware]
            hint = "Starlette runs the last-added middleware outermost"
            if "CORSMiddleware" in names:
                hint += "; CORS origins are set in add_middleware(CORSMiddleware, ...)"
            out.append(Anchor(
                id=self._id("middleware"), file=rel, kind=KIND_BLOCK, label=f"middleware: {_short_list(names)}",
                start_line=start, end_line=end, quote=self._quote(rel, span=(start, end)), hint=hint,
            ))
        if handlers:
            start, end = min(h[0] for h in handlers), max(h[1] for h in handlers)
            out.append(Anchor(
                id=self._id("exception_handlers"), file=rel, kind=KIND_BLOCK,
                label=f"exception handlers: {_short_list([h[2] for h in handlers])}",
                start_line=start, end_line=end, quote=self._quote(rel, span=(start, end)),
                hint="add new @app.exception_handler(...) functions after the last one",
            ))
        for hook in hooks:
            start = node_start(hook)
            out.append(Anchor(
                id=self._id("startup", hook.name), file=rel, kind=KIND_FUNCTION, label=f"startup hook {hook.name}",
                start_line=start, end_line=hook.end_lineno or start,
                quote=self._quote(rel, hook.lineno, span=(start, hook.end_lineno or start)),
                hint="runs before the first request (also under the runtime probe); keep it idempotent",
            ))
        call = _assigned_call(app.statement) if app is not None else None
        lifespan_name = _name(_keyword(call, "lifespan")) if call is not None else ""
        lifespan_fn = next((fn for fn in _function_defs(tree.body) if fn.name == lifespan_name), None)
        if lifespan_fn is not None:
            start = node_start(lifespan_fn)
            out.append(Anchor(
                id=self._id("lifespan"), file=rel, kind=KIND_FUNCTION, label=f"lifespan {lifespan_fn.name}",
                start_line=start, end_line=lifespan_fn.end_lineno or start,
                quote=self._quote(rel, lifespan_fn.lineno, span=(start, lifespan_fn.end_lineno or start)),
                hint="startup work goes before its yield, shutdown work after",
            ))
        for statement in tree.body:
            if isinstance(statement, ast.If) and "__main__" in ast.dump(statement.test):
                start, end = statement.lineno, statement.end_lineno or statement.lineno
                out.append(Anchor(
                    id=self._id("main_guard"), file=rel, kind=KIND_BLOCK, label="if __name__ == '__main__' (uvicorn)",
                    start_line=start, end_line=end, quote=self._quote(rel, start, span=(start, end)),
                    hint="local entry point only; servers and the runtime probe import the module-level app",
                ))
                break
        return out

    def _import_files(self, rel: str, alias: str) -> set[str]:
        """Files an imported name may come from: the module itself, or the module it is a member of."""
        module, imported, level = self.aliases.get(rel, {}).get(alias, ("", None, 0))
        found = {self._module_file(".".join(p for p in (module, imported) if p), level, rel),
                 self._module_file(module, level, rel) if imported else None}
        return {f for f in found if f}

    def _database_anchors(self) -> list[Anchor]:
        out: list[Anchor] = []
        orm_engine_files = set()
        for cls_file in {c.file for c in self.orm}:
            tree = self.trees.get(cls_file)
            if tree is not None and any(
                    (call := _assigned_call(st)) is not None and _name(call.func) in _ENGINE_CALLS
                    for st in tree.body):
                orm_engine_files.add(cls_file)
        init_fn, get_db = None, None
        for rel, tree in sorted(self.trees.items()):
            for fn in _function_defs(tree.body):
                if get_db is None and fn.name in ("get_db", "get_session") and any(
                        isinstance(n, (ast.Yield, ast.Return)) for n in ast.walk(fn)):
                    get_db = (rel, fn)
                if init_fn is None and any(isinstance(n, ast.Call) and _name(n.func) in _ENGINE_CALLS
                                           for n in ast.walk(fn)):
                    init_fn = (rel, fn)
        if init_fn is not None:
            rel, fn = init_fn
            start = node_start(fn)
            hint = "the app's engine and create_all (tables are created, never altered: no migrations)"
            stray = sorted(orm_engine_files - {rel})
            if stray and get_db is not None and get_db[0] == rel:
                hint += f"; the module-level engine in {stray[0]} is NOT the one the app uses"
            out.append(Anchor(
                id=self._id("init_db"), file=rel, kind=KIND_FUNCTION, label=f"{fn.name}: database engine setup",
                start_line=start, end_line=fn.end_lineno or start,
                quote=self._quote(rel, fn.lineno, span=(start, fn.end_lineno or start)), hint=hint,
            ))
        elif get_db is not None:
            # A hand-written database module often creates the engine at module level.
            rel = get_db[0]
            statement = next((st for st in self.trees[rel].body if (call := _assigned_call(st)) is not None
                              and _name(call.func) in _ENGINE_CALLS), None)
            if statement is not None:
                start, end = statement.lineno, statement.end_lineno or statement.lineno
                out.append(Anchor(
                    id=self._id("init_db"), file=rel, kind=KIND_BLOCK, label="database engine (module level)",
                    start_line=start, end_line=end, quote=self._quote(rel, start, span=(start, end)),
                    hint="the app's engine; tables are created, never altered (no migrations)",
                ))
        if get_db is not None:
            rel, fn = get_db
            start = node_start(fn)
            out.append(Anchor(
                id=self._id("get_db"), file=rel, kind=KIND_FUNCTION, label=f"{fn.name}: per-request session",
                start_line=start, end_line=fn.end_lineno or start,
                quote=self._quote(rel, fn.lineno, span=(start, fn.end_lineno or start)),
                hint=f"handlers take it as `database: Session = Depends({fn.name})`",
            ))
        return out

    def _requirements_anchor(self, rel: str) -> list[Anchor]:
        lines = self.index.lines(rel)
        text = self.index.text(rel) or ""
        last = max((n for n, line in enumerate(lines, 1) if line.strip()), default=0)
        if not last:
            return []
        newline = text.endswith("\n")
        return [Anchor(
            id=self._id("requirements_end"), file=rel, kind=KIND_INSERT,
            label=f"requirements.txt ({sum(1 for line in lines if line.strip())} packages)",
            start_line=last, end_line=last, quote=self._quote(rel, last),
            hint=("append new packages AFTER this line" + ("" if newline else
                  "; the file has NO trailing newline, so start the appended text with a newline")),
        )]

    # -- missing pieces, summaries, recipes --------------------------------

    def _mentioned_in_broken(self, name: str) -> bool:
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.I)
        return any(pattern.search(self.index.text(rel) or "") for rel in self.broken)

    def _missing(self, modular: bool, presence: list[dict[str, set[str]]]) -> None:
        """Per model class, the pieces this root's layout has for other entities but not for it.

        ``presence`` holds every FastAPI root's pieces. A piece is claimed missing
        here only when no root has it for the class, and only by a root that
        carries that piece for at least half of the model's classes (or for the
        most of them): a small second app (a proxy, a demo) never reports the
        main backend's entities as missing.
        """
        if not self.class_names:
            return
        own = self.presence()
        wanted = {self._lookup(name) or name for name in self.class_names}
        labels = {"orm": "ORM class", "schema": "Create schema" if modular else "schema",
                  "routes": "router" if modular else "routes"}
        reporting: dict[str, bool] = {}
        for piece, entities in own.items():
            coverage = len(entities & wanted)
            best = max((len(p.get(piece, set()) & wanted) for p in presence), default=coverage)
            reporting[piece] = bool(entities) and (2 * coverage >= len(wanted) or coverage >= best)
        anywhere = {piece: set().union(*(p.get(piece, set()) for p in presence), own[piece]) for piece in own}
        mounted_entities = {r.entity for r in self.routes if r.entity and r.mounted is not False}
        for name in self.class_names:
            entity = self._lookup(name) or name
            if self._mentioned_in_broken(entity):
                continue
            missing = [labels[piece] for piece in ("orm", "schema", "routes")
                       if reporting[piece] and entity not in anywhere[piece]]
            if entity in own["routes"] and entity not in mounted_entities and self.mounts_known:
                missing.append("include_router mount")
            if missing or entity in self.entities:
                self._entity(entity).missing = missing

    def _route_kind(self, route: _Route) -> tuple[str, str]:
        """(kind, detail) of a route, from its path shape relative to the entity segment."""
        segments, _ = _segments(route.label_path)
        if not route.entity:
            return "custom", ""
        lowered = [s.lower() for s in segments]
        anchor_at = next((i for i, s in enumerate(lowered) if not _is_param(s) and self._from_segment(s)
                          == route.entity), None)
        if anchor_at is None:
            return "custom", ""
        rest = segments[anchor_at + 1:]
        verb = route.verbs[0]
        n = len(rest)
        if n == 0:
            return {"GET": ("list", ""), "POST": ("create", "")}.get(verb, ("custom", ""))
        if n == 1 and _is_param(rest[0]):
            return {"GET": ("read", ""), "PUT": ("update", ""), "PATCH": ("update", ""),
                    "DELETE": ("delete", "")}.get(verb, ("custom", ""))
        if n == 1:
            word = rest[0]
            if verb == "GET" and word in ("count", "paginated", "search"):
                return word, ""
            if word == "bulk" and verb in ("POST", "DELETE"):
                return ("bulk create" if verb == "POST" else "bulk delete"), ""
        if n == 2 and all(_is_param(s) for s in rest):
            return {"GET": ("read", ""), "PUT": ("update", ""), "DELETE": ("delete", "")}.get(verb, ("custom", ""))
        if n == 3 and _is_param(rest[0]) and rest[1] == "methods":
            return "method", rest[2]
        if n == 2 and rest[0] == "methods":
            return "class method", rest[1]
        is_role = n >= 2 and _is_param(rest[0]) and not _is_param(rest[1]) and self._is_role(route.entity, rest[1])
        if is_role and n == 2 and verb == "GET":
            return "link list", rest[1]
        if is_role and n == 3 and _is_param(rest[2]):
            if verb == "POST":
                return "link add", rest[1]
            if verb == "DELETE":
                return "link remove", rest[1]
        return "custom", ""

    def _is_role(self, entity: str, segment: str) -> bool:
        """Is ``segment`` a relationship role of ``entity``? Without an ORM, trust the path shape."""
        if not self.orm:
            return True
        if self._roles is None:
            own: dict[str, set[str]] = {}
            for cls in self.orm:
                own.setdefault(cls.node.name, set()).update(cls.relationships)
            for _, owner, statement, _ in self.orm_relationships:
                own.setdefault(owner, set()).add(_targets(statement)[0].attr)
            # A subclass router re-emits its ORM ancestors' link routes: their roles count too.
            by_name = {cls.node.name: cls for cls in self.orm}
            self._roles = {}
            for cls in self.orm:
                roles: set[str] = set()
                pending, seen = [cls.node.name], set()
                while pending:
                    name = pending.pop()
                    if name in seen or name not in by_name:
                        continue
                    seen.add(name)
                    roles |= own.get(name, set())
                    pending.extend(by_name[name].bases)
                self._roles.setdefault(self._lookup(cls.node.name) or cls.node.name, set()).update(roles)
        return segment in self._roles.get(entity, set())

    def _file_digest(self, routes: list[_Route]) -> tuple[str, str, list[str], list[str]]:
        """(route kinds text, served kinds text, other parts, method parts) of one entity's
        routes in one file.

        The kinds text names every standard route the file declares (it is what
        entities are compared on); the served text leaves out the kinds whose
        handler refuses every payload, which the first other part names instead.
        """
        kinds: list[str] = []
        links: dict[str, list[str]] = {}
        methods: dict[str, list[str]] = {}       # "methods" / "501 stubs" / "stubs" -> names
        custom: list[str] = []
        seen_fns: set[int] = set()
        for route in routes:
            if id(route.fn) in seen_fns:
                continue
            seen_fns.add(id(route.fn))
            kind, detail = self._route_kind(route)
            if kind in ("method", "class method"):
                stub = _placeholder_reason(route.fn) or ("invalid code" if _invalid_code_stub(route.fn) else None)
                group = "501 stubs" if stub == "HTTP 501" else "stubs" if stub else "methods"
                methods.setdefault(group, []).append(detail)
            elif kind.startswith("link "):
                links.setdefault(detail, []).append(kind.split()[1])
            elif kind == "custom":
                custom.append(f"{route.verbs[0]} {route.label_path}")
            elif kind not in kinds:
                kinds.append(kind)
        refused = self._refused_kinds(routes)
        kinds_text = _kinds_text([k for k in _ROUTE_KIND_ORDER if k in kinds])
        served_text = _kinds_text([k for k in _ROUTE_KIND_ORDER if k in kinds and k not in refused])
        parts = []
        if refused:
            parts.append(_refusal_note(refused))
        if links:
            parts.append("links " + ", ".join(f"{role} ({'/'.join(ops)})" for role, ops in links.items()))
        if custom:
            parts.append("custom " + _short_list(custom, 3))
        method_parts = [f"{group}: {_short_list(methods[group], 6)}"
                        for group in ("methods", "501 stubs", "stubs") if group in methods]
        return kinds_text, served_text, parts, method_parts

    def _refused_kinds(self, routes: list[_Route]) -> dict[str, str]:
        """Standard route kind -> HTTP status, for kinds whose every handler in ``routes``
        refuses every payload (a kind another handler serves is not refused)."""
        refused: dict[str, str] = {}
        served: set[str] = set()
        for route in routes:
            kind = self._route_kind(route)[0]
            if kind not in _ROUTE_KIND_ORDER:
                continue
            refusal = _refusal(route.fn)
            if refusal is None:
                served.add(kind)
            else:
                refused.setdefault(kind, refusal[0])
        return {kind: code for kind, code in refused.items() if kind not in served}

    def _refusals(self) -> dict[str, list[str]]:
        """Refused standard route kind -> the entities whose routes refuse every payload for it."""
        by_kind: dict[str, list[str]] = {}
        for entity in sorted(self.entities, key=lambda n: (n.lower(), n)):
            routes = [r for rs in self._entity_files(entity).values() for r in rs]
            for kind in self._refused_kinds(routes):
                by_kind.setdefault(kind, []).append(entity)
        return {kind: by_kind[kind] for kind in _ROUTE_KIND_ORDER if kind in by_kind}

    def _entity_files(self, entity: str) -> dict[str, list[_Route]]:
        by_file: dict[str, list[_Route]] = {}
        for route in self.routes:
            if route.entity == entity:
                by_file.setdefault(route.file, []).append(route)
        return dict(sorted(by_file.items()))

    def _common_kinds(self) -> str:
        """The route set most entities share; stated once in the map summary, not per entity.

        Routes that refuse every payload count as declared here, so a guarded
        bulk route does not split the vote; the recipe names those exceptions.
        """
        counts: dict[str, int] = {}
        for entity in self.entities:
            for routes in self._entity_files(entity).values():
                kinds_text = self._file_digest(routes)[0]
                if kinds_text:
                    counts[kinds_text] = counts.get(kinds_text, 0) + 1
        best = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        return best[0][0] if best and best[0][1] >= 2 else ""

    def _conventional_file(self, rel: str, entity: str) -> bool:
        """Is ``rel`` where the recipes say this entity's routes live (so its line need not name it)?"""
        if not self.modular:
            return rel == self.app_file
        stem = posixpath.splitext(posixpath.basename(rel))[0].lower()
        return stem in (entity.lower(), f"{entity.lower()}_methods")

    def _entity_summary(self, entity: str, common: str) -> list[str]:
        """Short, line-free parts of one entity's overview line.

        Paths that follow the convention (``<routers>/<entity>.py`` and
        ``<entity>_methods.py``, or the single app file) are left out; the
        recipes state the convention once. Only what sets this entity apart is
        listed: routes that refuse every payload, links, custom routes, methods
        and stubs, routes missing from the common set, and files that break
        the convention.
        """
        lines: list[str] = []
        orm = next((c for c in self.orm if (self._lookup(c.node.name) or c.node.name) == entity), None)
        if orm is not None:
            parent = next((b for b in orm.bases if b in {c.node.name for c in self.orm}), None)
            if parent:
                lines.append(f"extends {parent}")
        conventional: list[str] = []
        elsewhere: list[str] = []      # files that break the convention: named, listed last
        files = self._entity_files(entity)
        for rel, routes in files.items():
            kinds_text, served_text, parts, method_parts = self._file_digest(routes)
            if served_text and kinds_text != common:
                parts.insert(0, served_text)
            parts += method_parts
            unmounted = routes[0].mounted is False
            if self._conventional_file(rel, entity) and not unmounted:
                conventional.extend(parts)
            else:
                head = f"in {rel}" + (" (NOT MOUNTED)" if unmounted else "")
                elsewhere.append(f"{head}: {'; '.join(parts)}" if parts else head)
        if not files:
            pieces = [label for label, present in (
                ("ORM class", orm is not None),
                ("schema", any(self._schema_entity(sc.node.name) == entity for sc in self.schemas)),
            ) if present]
            conventional.append(f"{' + '.join(pieces)}, no routes" if pieces else "no routes")
        elif not conventional and not elsewhere:
            conventional.append("standard routes only")
        return lines + conventional + elsewhere

    def _map_summary(self, modular: bool, smap: ScaffoldMap) -> list[str]:
        """App-level file roles. Seeding and packages live here rather than in the recipes:
        cheap to say, rarely the point of a request, and dropped first when the budget is tight."""
        lines = []
        app_bits = []
        if any(a for a in self.includes if a.file == self.app_file):
            app_bits.append("router mounts")
        system = [f"{r.verbs[0]} {r.label_path}" for r in self.routes if r.file == self.app_file and not r.entity]
        if system:
            app_bits.append("own routes " + _short_list(system, 4))
        lines.append(f"app {self.app_file}" + (": " + "; ".join(app_bits) if app_bits else ""))
        orm_files = sorted({c.file for c in self.orm})
        schema_files = sorted({s.file for s in self.schemas})
        data = []
        if orm_files:
            data.append(f"ORM {_short_list(orm_files, 3)}")
        if schema_files:
            data.append(f"schemas {_short_list(schema_files, 3)}")
        if data:
            lines.append("; ".join(data))
        # The entity lines leave conventional router paths out; when the budget allows
        # (this summary is the first thing dropped), name them once here.
        entity_routers = sorted(
            rel for rel, routes in self._router_files().items()
            if rel != self.app_file and not self._is_methods_router(routes)
            and self._conventional_file(rel, self._file_entity(routes) or ""))
        if modular and entity_routers:
            lines.append(f"entity routers: {_short_list(entity_routers, 6)}")
        if not modular:
            stores = self._in_memory_stores()
            if stores:
                lines.append(f"data lives in in-memory lists ({_short_list(stores, 4)}), lost on restart")
        fid = self.family_id
        ids = {a.id for a in smap.anchors}
        startup = next((a.id for a in smap.anchors if a.id.startswith(f"{fid}:startup:")), "")
        if f"{fid}:lifespan" in ids:
            lines.append(f"seed data (idempotently) in `{fid}:lifespan` before its yield")
        elif startup:
            lines.append(f"seed data (idempotently) in the startup hook `{startup}`")
        elif f"{fid}:include_routers_end" in ids:
            lines.append(f"seed data in an idempotent @app.on_event(\"startup\") hook "
                         f"after `{fid}:include_routers_end`")
        requirements = next((a for a in smap.anchors if a.id == f"{fid}:requirements_end"), None)
        if requirements is not None:
            lines.append(f"new packages at `{requirements.id}`"
                         + (", starting with a newline (the file has none at the end)"
                            if "NO trailing newline" in requirements.hint else ""))
        return lines

    def _in_memory_stores(self) -> list[str]:
        tree = self.trees.get(self.app_file)
        if tree is None:
            return []
        return [t.id for st in tree.body if isinstance(st, ast.Assign) and isinstance(st.value, (ast.List, ast.Dict))
                for t in st.targets if isinstance(t, ast.Name) and t.id.endswith(("_list", "_db", "_store"))]

    def _enum_clash(self) -> list[str]:
        """Enum classes defined in both the ORM and a schema module: routers star-import both."""
        orm_files = {c.file for c in self.orm}
        schema_files = {s.file for s in self.schemas}

        def enums(files: set[str]) -> set[str]:
            names = set()
            for rel in files:
                tree = self.trees.get(rel)
                for st in tree.body if tree is not None else []:
                    if isinstance(st, ast.ClassDef) and any(_name(b).endswith("Enum") for b in st.bases):
                        names.add(st.name)
            return names

        return sorted(enums(orm_files) & enums(schema_files))

    def _recipes(self, modular: bool, smap: ScaffoldMap) -> list[Recipe]:
        ids = {a.id for a in smap.all_anchors()}
        fid = self.family_id

        def has(prefix: str) -> bool:
            return any(i == f"{fid}:{prefix}" or i.startswith(f"{fid}:{prefix}:") for i in ids)

        router_dirs = sorted({posixpath.dirname(r.file) for r in self.routes if r.file != self.app_file})
        routers = router_dirs[0] if router_dirs else (self.root or ".")
        orm_file = self.orm[0].file if self.orm else ""
        create_schemas = [s for s in self.schemas
                          if s.node.name.endswith("Create") and self._schema_entity(s.node.name)]
        schema_file = create_schemas[0].file if create_schemas else (self.schemas[0].file if self.schemas else "")
        methods_files = sorted(rel for rel, rs in self._router_files().items() if self._is_methods_router(rs))
        auth_file = posixpath.join(self.root, "auth.py") if self.root else "auth.py"
        # Recipes go into a prompt budget shared by every map: one short sentence each.
        recipes: list[Recipe] = []
        # The entity lines omit the route set they share; it is stated here, where it
        # survives the overview budget, to stop re-implementing generated routes.
        # Routes that refuse every payload are named here too: they outlive the entity
        # lines, and "has bulk create" must not read as done for an entity that refuses it.
        refusals = self._refusals()
        existing = ""
        if self.common_kinds:
            refused = "; ".join(f"{kinds} for {_short_list(names, 6)}"
                                for kinds, names in _merge_bulk(refusals).items())
            existing = (f"Each entity has {self.common_kinds} (unless noted"
                        + (f"; refusing every payload: {refused}" if refused else "") + "). ")
        static = f"static paths BEFORE `{fid}:insert_static:<entity>`" if has("insert_static") else ""
        if not modular:
            recipes.append(Recipe("Add an endpoint", [
                f"{existing}`@app.<verb>` in {self.app_file} next to the entity's routes"
                + (f"; {static}." if static else ".")]))
            if self.schemas and self.entities:
                recipes.append(Recipe("Add a field", [f"The entity's class in {schema_file}."]))
            recipes.append(Recipe("Add auth", [
                f"New {auth_file}; `dependencies=[Depends(...)]` on the routes or on FastAPI(...)."]))
            return recipes
        # The entity lines leave conventional paths out, so the convention is stated here.
        entity_router = f"{routers}/<entity>.py" if self._routers_conventional() else "the entity's router file"
        recipes.append(Recipe("Add an endpoint for an entity", [
            f"{existing}@router (never @app) in {entity_router}, trailing slash"
            + (f"; {static}." if static else ", at the end of the file.")]))
        if has("stub") or methods_files:
            conventional = methods_files and all(
                rel == f"{routers}/{(self._file_entity(self._router_files()[rel]) or '').lower()}_methods.py"
                for rel in methods_files)
            # The routers directory is named by the endpoint recipe just above.
            where = "<entity>_methods.py" if conventional else "the entity's methods router"
            recipes.append(Recipe("Implement a model method", [
                f"Replace `{fid}:stub:<entity>:<method>` in {where}"
                + (", not the ORM copy" if orm_file else "")
                + '; body {"params": {...}}: read `(params or {}).get(..)`; zero-parameter methods must accept {}.']))
        field_steps = []
        if orm_file:
            field_steps.append(f"`{fid}:orm_add_column:<entity>`")
        if create_schemas:
            field_steps.append("<Entity>Create")
        if has("ctor"):
            # A bulk create that refuses every payload never reaches its constructor (and has no anchor).
            bulk = any(i.startswith(f"{fid}:ctor:") and ":bulk_create_" in i for i in ids)
            field_steps.append(f"kwargs at `{fid}:ctor:<entity>:create_<entity>`"
                               + ("" if not bulk else " (+ bulk_create_<entity> unless refused)"
                                  if "bulk create" in refusals else " (+ bulk_create_<entity>)"))
        if has("setattr"):
            field_steps.append(f"setattr at `{fid}:setattr:<entity>:update_<entity>`")
        pairs = self._inheritance_pairs()
        if field_steps and self.entities:
            # The entity lines say "extends X"; naming the pairs again would cost the budget twice.
            recipes.append(Recipe("Add a field", [
                ", ".join(field_steps) + ("; subclass routers re-emit inherited fields." if pairs else ".")]))
        mount = (f"import at `{fid}:router_imports_end`, mount at `{fid}:include_routers_end`."
                 if has("router_imports_end") and has("include_routers_end")
                 else f"app.include_router it in {self.app_file}.")
        recipes.append(Recipe("Add a new entity", [
            ("ORM class + <X>Create, " if orm_file and create_schemas else "")
            + f"new {routers}/<x>.py, {mount}"]))
        recipes.append(Recipe("Add auth", [
            f"New {auth_file}; `dependencies=[Depends(...)]` on include_router or APIRouter()"
            # Generated method bodies call the CRUD handlers positionally.
            + ("; no handler parameter before `database` (method bodies pass it positionally)."
               if methods_files else ".")]))
        return recipes

    def _routers_conventional(self) -> bool:
        """Does every entity with a router file have one named after it (``<entity>.py``)?"""
        by_entity: dict[str, list[str]] = {}
        for rel, routes in self._router_files().items():
            entity = self._file_entity(routes)
            if rel != self.app_file and entity:
                by_entity.setdefault(entity, []).append(rel)
        return all(any(self._conventional_file(rel, entity) for rel in files) for entity, files in by_entity.items())

    def _inheritance_pairs(self) -> list[str]:
        names = {c.node.name for c in self.orm}
        pairs = []
        for cls in self.orm:
            for base in cls.bases:
                if base in names:
                    pairs.append(f"{base}->{cls.node.name}")
        return pairs


# ---------------------------------------------------------------------------
# Module-level helpers used by the mapper
# ---------------------------------------------------------------------------

def _tablename_stmt(statement: ast.stmt) -> bool:
    return any(isinstance(t, ast.Name) and t.id == "__tablename__" for t in _targets(statement))


def _tablename(node: ast.ClassDef) -> Optional[str]:
    for statement in node.body:
        if _tablename_stmt(statement):
            value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
            return _string(value) or "?"
    return None


def _schema_bases(name: str) -> list[str]:
    """Entity names a schema class may be named after, longest first.

    ``BookLinkCreate`` -> ``BookLink``, ``Book``; the full name itself is not
    included (a schema may be named exactly like a domain class, e.g.
    ``SurveyResponse`` or ``CheckIn``, so callers try it first).
    """
    bases = {name[: -len(suffix)] for suffix in _SCHEMA_SUFFIXES
             if name.endswith(suffix) and len(name) > len(suffix)}
    return sorted(bases, key=lambda base: (-len(base), base))


def _invalid_code_stub(fn: ast.AST) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    """The nested function standing in for modelled code that did not compile, else None.

    The generator comments the invalid body out and leaves only
    ``raise NotImplementedError("... invalid code ...")`` (after an optional docstring).
    """
    for inner in ast.walk(fn):
        if inner is fn or not isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [st for st in inner.body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))]
        if len(body) == 1 and isinstance(body[0], ast.Raise) and isinstance(body[0].exc, ast.Call) \
                and _name(body[0].exc.func) == "NotImplementedError":
            message = _string(body[0].exc.args[0]) if body[0].exc.args else ""
            if message and "invalid code" in message:
                return inner
    return None


def _refusal(fn: ast.AST) -> Optional[tuple[str, str]]:
    """``(status, parameter)`` when the handler refuses every non-empty payload, else None.

    The generator guards bulk routes it cannot run safely (an entity with
    relationships) with a first statement ``if items: raise HTTPException(422, ...)``
    or ``if ids: raise HTTPException(409, ...)``: the route exists, but every
    request that carries data fails. ``status`` is the code ("422"), or
    "error" when it is not a literal; ``len(items)`` and ``len(items) > 0`` count too.
    """
    body = list(getattr(fn, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]                          # the docstring
    guard = body[0] if body else None
    if not isinstance(guard, ast.If) or guard.orelse:
        return None
    test = guard.test
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Gt, ast.NotEq)) \
            and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == 0:
        test = test.left
    if isinstance(test, ast.Call) and _name(test.func) == "len" and len(test.args) == 1 and not test.keywords:
        test = test.args[0]
    args = fn.args
    params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if not isinstance(test, ast.Name) or test.id not in params:
        return None
    # Only side-effect statements (a log call) may precede the raise.
    if not all(isinstance(st, ast.Expr) for st in guard.body[:-1]):
        return None
    last = guard.body[-1]
    exc = last.exc if isinstance(last, ast.Raise) else None
    if not isinstance(exc, ast.Call) or not _name(exc.func).endswith("HTTPException"):
        return None
    status = _keyword(exc, "status_code") or (exc.args[0] if exc.args else None)
    code = "error"
    if isinstance(status, ast.Constant) and isinstance(status.value, int) and not isinstance(status.value, bool):
        code = str(status.value)
    elif isinstance(status, ast.Attribute) and _HTTP_STATUS_NAME_RE.match(status.attr):
        code = _HTTP_STATUS_NAME_RE.match(status.attr).group(1)
    return code, test.id


def _mentions(annotation: ast.AST, name: str) -> bool:
    """Does a type annotation refer to ``name`` (``X``, ``List[X]``, ``Optional[m.X]``, ``"X"``)?"""
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and re.search(rf"\b{re.escape(name)}\b", node.value):
            return True
    return False


def _annotation_of(fn: ast.AST, arg_name: str) -> str:
    args = getattr(fn, "args", None)
    for arg in (args.args + args.kwonlyargs) if args is not None else []:
        if arg.arg == arg_name and arg.annotation is not None:
            return _name(arg.annotation) or (_string(arg.annotation) or "")
    return ""


def _terminal_statement(body: list[ast.stmt]) -> Optional[ast.stmt]:
    if not body:
        return None
    last = body[-1]
    if isinstance(last, ast.Try):
        return _terminal_statement(last.body)
    return last


def _declared_params(doc: str) -> list[tuple[str, str]]:
    """``[(name, type)]`` from a generated docstring's ``Parameters:`` list."""
    if "Parameters:" not in doc:
        return []
    out = []
    for line in doc.split("Parameters:", 1)[1].splitlines():
        m = _DOC_PARAM_RE.match(line)
        kind = m.group(2).strip() if m else ""
        if kind and _DOC_TYPE_RE.fullmatch(kind):
            out.append((m.group(1), kind))
        elif line.strip() and out:
            break
    return out


def _article(word: str) -> str:
    return f"{'an' if word[:1] in 'aeiou' else 'a'} {word}"


def _stub_status(reason: str) -> str:
    return {
        "HTTP 501": "HTTP 501 stub",
        "NotImplementedError": "NotImplementedError stub",
        "an empty action body": "empty body stub",
    }.get(reason, f"stub ({reason})")


def _kinds_text(ordered: list[str]) -> str:
    """``CRUD, count, bulk create/delete`` for route kinds in ``_ROUTE_KIND_ORDER`` order."""
    if not all(k in ordered for k in _STANDARD_CRUD):
        return ", ".join(ordered)
    extras = [k for k in ordered if k not in _STANDARD_CRUD]
    if "bulk create" in extras and "bulk delete" in extras:
        extras = [k for k in extras if k != "bulk delete"]
        extras[extras.index("bulk create")] = "bulk create/delete"
    return ", ".join(["CRUD"] + extras)


def _refusal_note(refused: dict[str, str]) -> str:
    """``bulk create/delete refused (422/409)`` for ``{kind: status}``."""
    kinds = [k for k in _ROUTE_KIND_ORDER if k in refused]
    codes = [refused[k] for k in kinds]
    if "bulk create" in kinds and "bulk delete" in kinds:      # adjacent in _ROUTE_KIND_ORDER
        at = kinds.index("bulk create")
        kinds[at:at + 2] = ["bulk create/delete"]
    return f"{', '.join(kinds)} refused ({'/'.join(codes)})"


def _merge_bulk(by_kind: dict[str, list[str]]) -> dict[str, list[str]]:
    """Name ``bulk create`` and ``bulk delete`` once when the same entities refuse both."""
    if "bulk create" not in by_kind or by_kind.get("bulk create") != by_kind.get("bulk delete"):
        return by_kind
    return {("bulk create/delete" if k == "bulk create" else k): v for k, v in by_kind.items() if k != "bulk delete"}
