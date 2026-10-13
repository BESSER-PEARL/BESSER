"""Scaffold map extractor for React frontends (Vite + react-router).

Covers three shapes with one detection: the GUI scaffold ``generate_web_app``
writes to ``web_app/frontend`` and ``generate_react`` to ``react/`` (the two
are byte-identical except a Dockerfile), and a frontend the model authored on
the class-diagram path (lenient: same detection, fewer guarantees).

A frontend root is a directory holding ``package.json`` (declaring React, or
with an entry that imports it) and a ``src/`` entry (``main``/``index``/``App``
with a JS/TS extension). Per root the map gives:

* the entry chain (the ``render`` call with its router and providers) and the
  route table (``<Routes>``/``<Route>`` JSX or ``createBrowserRouter``
  objects), each route resolved to its page file through the imports, plus
  insertion points for a new ``<Route>`` and a new page import;
* the pages (``src/pages/*`` and every routed component) and, per page, the
  runtime blocks bound to an entity (``TableBlock``, ``FormBlock``,
  ``MethodButton``...). Pages follow GUI screens, not classes: a page is
  linked to an entity only through a bound block, and nothing is ever
  reported missing;
* the API base declarations (the generated scaffold has no central client),
  the theme files and the build configuration;
* pitfalls in the code as it is: preventDefault-only forms (a Phase 3
  blocker, folded into one line when there are several), imports of files that
  do not exist, pages nothing routes to and an API base re-declared in several
  editable files (the runtime blocks' own copies are the convention, not a pitfall).

Nothing here parses TypeScript. JSX is scanned with the validators' own
brace-aware helpers (``frontend_forms._jsx_tags``/``_attr``) on a copy of each
file whose comments are blanked in place (offsets and lines unchanged), so a
malformed file yields fewer anchors, never an error.
"""

from __future__ import annotations

import bisect
import json
import logging
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from besser.spec_driven_agent.planning.scaffold_map.model import (
    KIND_BLOCK,
    KIND_FILE,
    KIND_FUNCTION,
    KIND_INSERT,
    KIND_STUB,
    Anchor,
    EntityMap,
    Recipe,
    ScaffoldMap,
)
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex
from besser.spec_driven_agent.repair.scaffold_repair import _API_BASE_DECL_RE, _API_BASE_OPTION_RE
from besser.spec_driven_agent.validation.frontend_forms import _attr, _jsx_tags
from besser.spec_driven_agent.validation.frontend_resolution import _RESOLVE_EXT, _SPEC_RE

logger = logging.getLogger(__name__)

FAMILY = "react"

_SOURCE_EXT = (".tsx", ".jsx", ".ts", ".js", ".mjs")
_ENTRY_NAMES = tuple(f"{stem}{ext}" for stem in ("main", "index") for ext in (".tsx", ".jsx", ".ts", ".js"))
_APP_NAMES = ("App.tsx", "App.jsx", "App.ts", "App.js")
_STYLE_EXT = (".css", ".scss")

# Runtime blocks of the GUI scaffold, configured through literal JSON props.
_RUNTIME_BLOCKS = (
    "TableBlock", "FormBlock", "MethodButton", "CrudButton", "ChartBlock", "MetricCardBlock",
    "DataListBlock", "MapBlock", "AlertBlock", "AgentComponent",
)
# A capitalised tag carrying one of these props is a bound block under any name.
_BINDING_PROPS = ("entity", "endpoint", "dataBinding", "series", "dataSources")
# The GUI scaffold's shared runtime (verified on generated output): configured
# through page props, never edited. Directories are reported as globs.
_RUNTIME_DIRS = ("src/components/runtime", "src/components/table", "src/components/charts")
_RUNTIME_FILES = ("src/components/MethodButton.tsx", "src/contexts/TableContext.tsx")
_MAX_LIST = 12          # routes named in one overview line
_MAX_WARNINGS = 4       # warnings of one kind, then a count

# Comments and string literals in source order; only comments get blanked.
# Quoted strings stop at a newline so an apostrophe in JSX text cannot swallow
# the rest of the file (frontend_bindings._NON_CODE uses the same bound). A quote
# right after a backslash opens no string: inside a string that never closes,
# every escaped quote would otherwise rescan it to its end (quadratic).
_STRING = r"(?<!\\)\"(?:\\.|[^\"\\\n])*\"|(?<!\\)'(?:\\.|[^'\\\n])*'|(?<!\\)`(?:\\.|[^`\\])*`"
# An unterminated block comment runs to the end of the text in ONE match: the
# lazy ``/\*[\s\S]*?\*/`` rescans from every later ``/*`` and is quadratic.
_COMMENT_OR_STRING = re.compile(r"//[^\n]*|/\*(?:[^*]|\*(?!/))*(?:\*/|\Z)|" + _STRING)
_STATIC_IMPORT_RE = re.compile(
    r"\bimport\s+(?:type\s+)?(?P<clause>[\w$]+(?:\s*,\s*\{[^}]*\})?|\{[^}]*\}|\*\s*as\s+[\w$]+)"
    r"\s*from\s*(?P<q>['\"])(?P<spec>[^'\"\n]+)(?P=q)"
)
_LAZY_IMPORT_RE = re.compile(
    r"\b(?:const|let|var)\s+(?P<name>[A-Z][\w$]*)\s*=\s*(?:React\s*\.\s*)?lazy\s*\(\s*\(\s*\)\s*=>\s*"
    r"import\s*\(\s*(?P<q>['\"])(?P<spec>[^'\"\n]+)(?P=q)"
)
_DEFAULT_EXPORT_RE = re.compile(
    r"^export\s+default\s+(?:async\s+)?(?:function|class)\s+(?P<a>[A-Z][\w$]*)"
    r"|^export\s+default\s+(?:[\w$.]+\s*\(\s*)*(?P<b>[A-Z][\w$]*)\b",
    re.M,
)
_COMPONENT_DEF_RE = re.compile(
    r"^(?P<export>export\s+)?(?:default\s+)?(?:async\s+)?(?P<kw>function\s*(?:\*\s*)?|(?:const|let|var)\s+|class\s+)"
    r"(?P<name>[A-Z][\w$]*)\b(?P<rest>[^\n]*)",
    re.M,
)
# What follows ``const X`` (an optional type annotation, then '='): a literal here is data, not a component.
_INITIALISER_RE = re.compile(r"\s*(?::[^=\n]*)?=\s*(?P<first>\S?)")
_DATA_START = tuple("[{'\"`0123456789-") + ("new ", "true", "false", "null")
_RENDER_RE = re.compile(r"\.render\s*\(")
# No two optional whitespace runs are ever adjacent in these patterns (``\s*\(?\s*`` is
# quadratic on a long blank run, and comments are blanked to spaces): ``\s*(?:\(\s*)?``.
_RETURN_JSX_RE = re.compile(r"\breturn\s*(?:\(\s*)?<\s*(?P<tag>[A-Za-z][\w.]*)?")
_ARROW_JSX_RE = re.compile(r"=>\s*(?:\(\s*)?<\s*(?P<tag>[A-Za-z][\w.]*)?")
_ROUTES_OPEN_RE = re.compile(r"<Routes\b[^>]*>")
_ROUTES_CLOSE_RE = re.compile(r"</Routes\s*>")
_ROUTE_CLOSE_RE = re.compile(r"</Route\s*>")
_ROUTER_CALL_RE = re.compile(r"\b(?P<fn>createBrowserRouter|createHashRouter|createMemoryRouter|useRoutes)\s*\(")
_ROUTER_ARG_RE = re.compile(r"\s*(?:(?P<array>\[)|(?P<name>[A-Za-z_$][\w$]*)\s*[,)])")
# Keys of one route object, searched in its top-level text (nested brackets blanked).
_OBJ_PATH_RE = re.compile(r"(?<![\w$.])path\s*:\s*(?P<q>['\"`])(?P<path>[^'\"`\n]*)(?P=q)")
_OBJ_INDEX_RE = re.compile(r"(?<![\w$.])index\s*:\s*true\b")
_OBJ_ELEMENT_RE = re.compile(r"(?<![\w$.])element\s*:")
_OBJ_COMPONENT_RE = re.compile(r"(?<![\w$.])Component\s*:\s*(?P<name>[A-Z][\w$]*)")
_OBJ_CHILDREN_RE = re.compile(r"(?<![\w$.])children\s*:\s*\[")
_STRING_AT_RE = re.compile(_STRING)
_MAX_ROUTE_DEPTH = 24
_ROUTER_KIND_RE = re.compile(r"<(BrowserRouter|HashRouter|MemoryRouter|RouterProvider)\b")
_TOP_LEVEL_RE = re.compile(r"[A-Za-z_$@]")
_TAG_NAME_RE = re.compile(r"<([A-Za-z][\w.]*)")
_ID_ATTR_RE = re.compile(r"""\bid\s*=\s*(?:\{\s*)?(['"])([^'"]+)\1""")
_NAV_RE = re.compile(r"<(?:nav|NavLink|Link)\b")
# frontend_contract's two no-op onSubmit shapes: an empty or preventDefault-only handler. The
# parameter is ``()``, ``(e)`` or a bare ``e``. No two optional whitespace runs are adjacent (its
# ``\(?\s*\w*\s*\)?\s*`` is polynomial): a long comment in the handler, blanked to spaces, stalled every map.
_NOOP_SUBMIT_RES = (
    re.compile(r"onSubmit\s*=\s*\{\s*(?:async\s*)?(?:\(\s*(?:\w+\s*)?\)|\w+)\s*=>\s*"
               r"\{\s*(?:[\w.]*\.preventDefault\(\)\s*(?:;\s*)?)?\}\s*\}"),
    re.compile(r"onSubmit\s*=\s*\{\s*(?:\(\s*\w+\s*\)|\w+)\s*=>\s*[\w.]*\.preventDefault\(\)\s*\}"),
)
# API base declarations: an env read (with or without a fallback), a base-like
# constant holding a URL, or the localhost shapes ensure_frontend_scaffold rewrites.
_ENV_BASE_RE = re.compile(
    r"(?:import\.meta\.env|process\.env)\.(?P<var>(?:VITE|REACT_APP)_\w*?(?:API|BACKEND|SERVER)\w*)"
)
_FALLBACK_RE = re.compile(r"\s*(?:\|\||\?\?)\s*(?P<q>['\"`])(?P<url>https?://[^'\"`\s$]+)(?P=q)")
_HARDCODED_BASE_RE = re.compile(
    r"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+(?P<name>[A-Za-z_$][\w$]*)[ \t]*=[ \t]*(?:\([ \t]*)?"
    r"(?P<q>['\"`])(?P<url>https?://[^'\"`\s$]*)(?P=q)",
    re.M,
)
_BASE_NAME_RE = re.compile(r"(?i)api|base|backend|server")
_EXPORTED_FN_RE = re.compile(
    r"^export\s+(?:async\s+)?function\s+(?P<a>[\w$]+)"
    r"|^export\s+const\s+(?P<b>[\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>",
    re.M,
)
_CLIENT_STEMS = {"api", "client", "apiclient", "http", "axios", "request", "requests", "apiservice", "fetcher"}
_CLIENT_DIRS = {"api", "services", "service", "lib"}
_HTTP_CALL_RE = re.compile(r"\baxios\s*[.(]|\bfetch\s*\(")
_CSS_IMPORT_RE = re.compile(r"""@import\s+(?:url\(\s*)?(['"])(?P<spec>[^'"]+)\1""")
_HTML_LINK_RE = re.compile(r"""<link\b[^>]*\bhref\s*=\s*(['"])(?P<href>[^'"]+\.css)\1""")
_HTML_SCRIPT_RE = re.compile(r"""<script\b[^>]*\bsrc\s*=\s*(['"])(?P<src>[^'"]+)\1""")
_NOT_A_BREAK = re.compile("[^\n]")


# --------------------------------------------------------------------------- source helpers

def _code_view(text: str) -> str:
    """``text`` with comments blanked to spaces, so offsets and line numbers still match.

    Newlines are kept, so the code's lines pair one-to-one with ``SourceIndex.lines``
    (which splits on ``\\n`` only, like ``read_file``).
    """
    # A '/*' after the last '*/' never closes, so it is no comment; yet each one made the
    # pattern scan to the end of the text again. They are masked in the copy it runs on.
    last = text.rfind("*/")
    cut = last + 2 if last >= 0 else 0
    scan = text[:cut] + text[cut:].replace("/*", "/\0")
    out, pos = [], 0
    for match in _COMMENT_OR_STRING.finditer(scan):
        start, end = match.span()
        if text.startswith(("//", "/*"), start):
            out += [text[pos:start], _NOT_A_BREAK.sub(" ", text[start:end])]
            pos = end
    out.append(text[pos:])
    return "".join(out)


class _Src:
    """One source file: its text, the comment-blanked code and an offset-to-line table."""

    def __init__(self, rel: str, text: str) -> None:
        self.rel = rel
        self.text = text
        self.code = _code_view(text)
        # Lines end at '\n' only, as in SourceIndex.lines(): a U+2028 or a form feed is no line break.
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]
        if self.starts[-1] < len(text):
            self.starts.append(len(text))
        self.line_count = max(1, len(self.starts) - 1)
        self.code_lines = self.code.split("\n")
        if self.code_lines[-1] == "":
            self.code_lines.pop()

    def code_line(self, line: int) -> str:
        """Comment-blanked text of 1-based ``line`` ('' out of range)."""
        return self.code_lines[line - 1] if 1 <= line <= len(self.code_lines) else ""

    def line(self, offset: int) -> int:
        return min(max(1, bisect.bisect_right(self.starts, offset)), self.line_count)

    def offset(self, line: int) -> int:
        """Offset of the start of 1-based ``line`` (end of text past the last line)."""
        return self.starts[min(max(0, line - 1), len(self.starts) - 1)]

    def block_end(self, start_line: int) -> int:
        """Last code line of the top-level statement starting at ``start_line``.

        It ends before the next column-0 statement; lines that are blank once
        comments are removed do not extend it. This holds for generated and
        formatter-shaped code and, unlike a brace counter, survives an
        apostrophe in JSX text.
        """
        end = start_line
        for idx in range(start_line, len(self.code_lines)):
            line = self.code_lines[idx]
            if line[:1] and _TOP_LEVEL_RE.match(line):
                break
            if line.strip():
                end = idx + 1
        return end


def _balanced_end(code: str, open_idx: int) -> Optional[int]:
    """Offset just past the bracket closing ``code[open_idx]`` (strings skipped), or None."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    quote = None
    idx = open_idx
    while idx < len(code):
        char = code[idx]
        if quote:
            if char == "\\":
                idx += 1
            elif char == quote or (char == "\n" and quote != "`"):
                quote = None
        elif char in "\"'`":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in ")]}":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return idx + 1
        idx += 1
    return None


def _tags(code: str, tag: str):
    """``_jsx_tags(code, tag)``, stopped once the scanned attribute text exceeds a budget linear in ``code``.

    Each tag is scanned up to its closing ``>`` at brace depth 0, so an unclosed
    ``{`` or quote made every later tag scan to the end of the file (quadratic:
    minutes on a large page). Past the budget the remaining tags are not read: a
    broken file yields fewer anchors, never a stall.
    """
    budget = 3 * len(code) + 50_000
    for start, attrs, top in _jsx_tags(code, tag):
        yield start, attrs, top
        budget -= len(attrs)
        if budget < 0:
            return


def _scalar(raw: Optional[str]) -> Optional[str]:
    """A JSX attribute value (``_attr`` output) as text when it is a literal string."""
    if raw is None:
        return None
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"`" and "${" not in raw:
        return raw[1:-1]
    return None


def _has_prop(top: str, name: str) -> bool:
    return re.search(rf"(?<![\w.-]){re.escape(name)}\s*=", top) is not None


def _prop(attrs: str, top: str, name: str) -> Any:
    """Literal value of the top-level JSX prop ``name``: a string, decoded JSON, or None."""
    if not _has_prop(top, name):
        return None
    raw = _attr(attrs, name)
    text = _scalar(raw)
    if text is not None or raw is None:
        return text
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _first_segment(endpoint: str) -> str:
    """``/book/{book_id}/methods/x/`` -> ``book``; '' when there is no literal segment."""
    path = re.sub(r"^https?://[^/]+", "", endpoint or "").split("?")[0]
    for part in path.split("/"):
        if part and not part.startswith(("{", ":", "$")):
            return part
    return ""


def _names(items: list[str], limit: int) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f", +{len(items) - limit} more" if len(items) > limit else "")


def _is_component_def(match: re.Match) -> bool:
    """Whether a ``_COMPONENT_DEF_RE`` match can be a component (not ``const COLUMNS = [...]``)."""
    if match.group("name").isupper():           # SCREAMING_CASE: a constant, never a component
        return False
    if not match.group("kw").startswith(("const", "let", "var")):
        return True
    init = _INITIALISER_RE.match(match.group("rest"))
    return not (init and init.group("first") and match.group("rest")[init.start("first"):].startswith(_DATA_START))


def _top_level(code: str, lo: int, hi: int) -> str:
    """``code[lo:hi]`` with everything inside a nested bracket blanked; offsets and newlines kept.

    The brackets opened at the top level stay, so a route object's own keys
    (``path``, ``element``, ``children: [``) can be read without its children's.
    """
    out = list(code[lo:hi])
    depth = 0
    idx = lo
    while idx < hi:
        char = code[idx]
        if char in "\"'`":
            match = _STRING_AT_RE.match(code, idx)
            end = min(match.end() if match else idx + 1, hi)
            if depth:
                for pos in range(idx, end):
                    out[pos - lo] = out[pos - lo] if code[pos] == "\n" else " "
            idx = end
            continue
        if char in "([{":
            if depth:
                out[idx - lo] = " "
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
            if depth:
                out[idx - lo] = " "
        elif depth and char != "\n":
            out[idx - lo] = " "
        idx += 1
    return "".join(out)


# --------------------------------------------------------------------------- data

@dataclass
class _Route:
    path: str                   # full path (nested paths joined); a pathless layout keeps its parent's
    component: str              # rendered component; 'Navigate' for a redirect, '' when dynamic
    target: str                 # redirect target of a <Navigate>
    file: str                   # route-table file
    line: int
    kind: str = "path"          # "path" (a literal path of its own), "index" or "layout" (pathless)
    guarded: bool = False       # `cond ? <Page /> : <Navigate />`
    grouping: bool = False      # children but no element: renders only an <Outlet>
    one_line: bool = True       # the whole route (object) sits on one line
    parent: bool = False        # has child routes (a layout renders them through <Outlet>)
    resolved: Optional[str] = None   # file defining ``component``

    @property
    def declared(self) -> bool:
        return self.kind == "path"


@dataclass
class _Table:
    file: str
    kind: str                   # "jsx" or the router function name
    start: int
    end: int
    routes: list[_Route]
    close_line: Optional[int] = None


@dataclass
class _Block:
    tag: str
    block_id: str
    start: int
    end: int
    detail: str                 # method name, CRUD action or the block id
    endpoint: str = ""          # first literal endpoint prop, as written
    names: list[str] = field(default_factory=list)       # entity props as written
    segments: list[str] = field(default_factory=list)    # first segments of endpoint props
    entities: list[str] = field(default_factory=list)    # resolved, after every page is scanned


@dataclass
class _Page:
    file: str
    component: str
    start: int
    end: int
    routes: list[str] = field(default_factory=list)
    root_line: int = 0
    blocks: list[_Block] = field(default_factory=list)
    layout: bool = False        # rendered by a pathless layout route around other routes


@dataclass
class _Api:
    client: Optional[str] = None
    env_var: str = ""
    fallback: str = ""
    sites: dict[str, list[int]] = field(default_factory=dict)


# --------------------------------------------------------------------------- detection

def _join(root: str, rel: str) -> str:
    return f"{root}/{rel}" if root else rel


def _declares_react(index: SourceIndex, root: str, files: set[str]) -> bool:
    try:
        manifest = json.loads(index.text(_join(root, "package.json")) or "null")
    except ValueError:
        manifest = None
    if isinstance(manifest, dict):
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            deps = manifest.get(key)
            if isinstance(deps, dict) and ("react" in deps or "react-dom" in deps):
                return True
    # Lenient: the manifest is incomplete or broken, but the entry imports React.
    for name in _ENTRY_NAMES + _APP_NAMES:
        rel = _join(root, f"src/{name}")
        if rel in files and re.search(r"""from\s+['"]react(?:-dom)?(?:/client)?['"]""", index.text(rel) or ""):
            return True
    return False


def _frontend_roots(index: SourceIndex) -> list[str]:
    """Directories with a package.json, a src/ entry and React; sorted."""
    files = set(index.files)
    roots = []
    for rel in index.files:
        if posixpath.basename(rel) != "package.json":
            continue
        root = posixpath.dirname(rel)
        if any(_join(root, f"src/{name}") in files for name in _ENTRY_NAMES + _APP_NAMES) \
                and _declares_react(index, root, files):
            roots.append(root)
    return sorted(roots)


# --------------------------------------------------------------------------- one root

class _RootMapper:
    """Builds the :class:`ScaffoldMap` of one frontend root."""

    def __init__(self, index: SourceIndex, root: str, class_names: list[str], all_roots: list[str],
                 fam: str = FAMILY, used_ids: Optional[set[str]] = None) -> None:
        self.index = index
        self.root = root
        # Sorted: the model's classes come from a set, and the map must not depend on hash order.
        self.class_names = sorted(class_names)
        nested = [r for r in all_roots if r != root and (not root or r.startswith(root + "/"))]
        self.files = [f for f in index.under(root) if not any(f.startswith(n + "/") for n in nested)]
        self.file_set = set(self.files)
        self.fam = fam
        self._partial: Optional[bool] = None
        self.canon: dict[str, str] = {c.lower(): c for c in self.class_names}
        src_dir = self.p("src") + "/"
        self.sources = [f for f in self.files
                        if f.startswith(src_dir) and f.endswith(_SOURCE_EXT) and not f.endswith(".d.ts")]
        self.runtime = self._runtime_files()
        self.editable = [f for f in self.sources if f not in self.runtime]
        self.anchors: list[Anchor] = []
        self.entities: dict[str, EntityMap] = {}
        self._warnings: list[tuple[int, str]] = []      # (rank, text): build breakers first
        self._srcs: dict[str, Optional[_Src]] = {}
        self._imports: dict[str, dict[str, tuple[str, str, int]]] = {}
        self._specs: dict[str, list[tuple[str, int]]] = {}
        self._importers: Optional[dict[str, set[str]]] = None
        self._ids: set[str] = used_ids if used_ids is not None else set()   # shared by every root
        self.pages_dir = self.p("src/pages")

    # -- access ----------------------------------------------------------

    def p(self, rel: str) -> str:
        return _join(self.root, rel)

    def short(self, rel: str) -> str:
        """``rel`` relative to the frontend root, for compact overview lines."""
        prefix = self.root + "/" if self.root else ""
        return rel[len(prefix):] if prefix and rel.startswith(prefix) else rel

    def src(self, rel: Optional[str]) -> Optional[_Src]:
        if not rel:
            return None
        if rel not in self._srcs:
            text = self.index.text(rel) if rel in self.file_set else None
            self._srcs[rel] = _Src(rel, text) if text is not None else None
        return self._srcs[rel]

    def anchor(self, what: str, rel: str, kind: str, label: str, start: int, end: int, *,
               quote_line: Optional[int] = None, hint: str = "", entity: Optional[str] = None,
               status: Optional[str] = None) -> Anchor:
        """An anchor with a workspace-unique id and a quote taken only from the index."""
        anchor_id, n = f"{self.fam}:{what}", 2
        while anchor_id in self._ids:
            anchor_id, n = f"{self.fam}:{what}#{n}", n + 1
        self._ids.add(anchor_id)
        end = max(start, end)
        quote = self.index.unique_quote(rel, quote_line) if quote_line else ""
        if not quote and kind != KIND_INSERT:
            quote = self.index.first_unique_quote(rel, start, end)
        return Anchor(id=anchor_id, file=rel, kind=kind, label=label, start_line=start, end_line=end,
                      quote=quote, hint=hint, entity=entity, status=status)

    def ref(self, what: str) -> Optional[str]:
        """This root's anchor id ``<fam>:<what>`` when that anchor was emitted, else None."""
        anchor_id = f"{self.fam}:{what}"
        return anchor_id if any(a.id == anchor_id for a in self.anchors) else None

    def _candidates(self, importer: str, spec: str) -> list[str]:
        """The files a relative or ``@/`` import may name, in resolution order ([] for any other spec)."""
        if spec.startswith("."):
            base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
        elif spec.startswith("@/"):
            base = self.p("src/" + spec[2:])
        else:
            return []
        candidates = [base + ext for ext in _RESOLVE_EXT] + [f"{base}/index{ext}" for ext in _RESOLVE_EXT[1:]]
        stem, ext = posixpath.splitext(base)
        if ext in (".js", ".jsx", ".mjs"):     # TS ESM style: './util.js' names util.ts
            candidates += [stem + ".ts", stem + ".tsx"]
        return candidates

    def resolve(self, importer: str, spec: str) -> Optional[str]:
        """The mapped file a local import names; '' when it is local but not mapped; None for a package."""
        spec = spec.split("?")[0]
        if spec.startswith("/") and not spec.startswith("//"):
            # Root-relative: the project root (index.html's entry) or Vite's public/ folder.
            found = next((c for c in (self.p(spec[1:]), self.p("public" + spec)) if c in self.file_set), None)
            return found
        candidates = self._candidates(importer, spec)
        if not candidates:
            return None
        found = next((c for c in candidates if c in self.file_set), None)
        if found:
            return found
        return None if spec.startswith("@/") else ""   # an alias may point elsewhere: not reported

    def missing(self, importer: str, spec: str) -> bool:
        """Whether a local import names no file at all: none mapped, and none on disk either.

        The index leaves files out (past its caps; images and other unmappable
        files), so a file it lacks is looked for on disk before the map says it
        does not exist. A path leaving the workspace is never called missing.
        """
        if self.resolve(importer, spec) != "":
            return False
        candidates = self._candidates(importer, spec.split("?")[0])
        if any(c == ".." or c.startswith("../") for c in candidates):
            return False
        return not any(self.index.exists(c) for c in candidates)

    def partial(self) -> bool:
        """Whether some of this root's code is unseen: files left out by the index's caps, or unreadable.

        Then a file nothing seems to import or route may be used by one the map
        never read, so no "unreachable" or "imported nowhere" claim is made.
        """
        if self._partial is None:
            self._partial = bool(self.index.capped) or any(
                self.index.text(rel) is None for rel in self.files if rel.endswith(_SOURCE_EXT + _STYLE_EXT))
        return self._partial

    @staticmethod
    def _spec_to(importer: str, folder: str) -> str:
        """Relative import specifier from ``importer``'s folder to ``folder`` ('./pages', '../views')."""
        rel = posixpath.relpath(folder, posixpath.dirname(importer) or ".")
        return rel if rel.startswith("..") else "./" + rel

    def imports(self, rel: str) -> dict[str, tuple[str, str, int]]:
        """Local name -> (spec, imported name or 'default', last line of the import statement)."""
        if rel not in self._imports:
            out: dict[str, tuple[str, str, int]] = {}
            src = self.src(rel)
            if src is not None:
                for match in _STATIC_IMPORT_RE.finditer(src.code):
                    spec, line = match.group("spec"), src.line(match.end())
                    head, _, braces = match.group("clause").partition("{")
                    default = head.strip().rstrip(",").strip()
                    if default and not default.startswith("*"):
                        out.setdefault(default, (spec, "default", line))
                    for item in braces.rstrip("}").split(","):
                        parts = item.replace("type ", " ").split()
                        if len(parts) == 3 and parts[1] == "as":
                            out.setdefault(parts[2], (spec, parts[0], line))
                        elif len(parts) == 1:
                            out.setdefault(parts[0], (spec, parts[0], line))
                for match in _LAZY_IMPORT_RE.finditer(src.code):
                    out.setdefault(match.group("name"), (match.group("spec"), "default", src.line(match.end())))
            self._imports[rel] = out
        return self._imports[rel]

    def importers(self) -> dict[str, set[str]]:
        """Workspace path -> the source files importing it (static, dynamic or side-effect)."""
        if self._importers is None:
            self._importers = {}
            for rel in self.sources:
                for spec, _ in self.specs(rel):
                    target = self.resolve(rel, spec)
                    if target:
                        self._importers.setdefault(target, set()).add(rel)
        return self._importers

    def specs(self, rel: str) -> list[tuple[str, int]]:
        """Every import specifier of ``rel`` (static, side-effect, dynamic, require) with its line."""
        if rel not in self._specs:
            src = self.src(rel)
            self._specs[rel] = [(m.group(2), src.line(m.start())) for m in _SPEC_RE.finditer(src.code)] \
                if src is not None else []
        return self._specs[rel]

    def _runtime_files(self) -> set[str]:
        runtime_dir = self.p("src/components/runtime") + "/"
        if not any(f.startswith(runtime_dir) and posixpath.basename(f)[:-len(".tsx")] in _RUNTIME_BLOCKS
                   for f in self.files if f.endswith(".tsx")):
            return set()
        dirs = tuple(self.p(d) + "/" for d in _RUNTIME_DIRS)
        out = {f for f in self.files if f.startswith(dirs)}
        out.update(self.p(f) for f in _RUNTIME_FILES if self.p(f) in self.file_set)
        return out

    def _vite_config(self) -> Optional[str]:
        return next((self.p(f"vite.config.{ext}") for ext in ("ts", "js", "mjs", "cjs", "mts")
                     if self.p(f"vite.config.{ext}") in self.file_set), None)

    # -- build -----------------------------------------------------------

    def build(self) -> ScaffoldMap:
        entry = self._entry_file()
        app = next((self.p(f"src/{n}") for n in _APP_NAMES if self.p(f"src/{n}") in self.file_set), None)
        tables = self._route_tables([f for f in (app, entry) if f])
        primary = tables[0] if tables else None

        pages = self._pages(tables)
        self._resolve_entities(pages)
        self.pages_dir = self._pages_dir(pages, primary)
        if entry:
            self._entry_anchor(entry)
        if primary:
            self._route_anchors(primary, tables[1:])
        self._page_anchors(pages)
        self._form_stubs(pages)
        api = self._api_layer()
        theme = self._theme(entry)
        self._config_anchors()
        self._missing_import_warnings(primary)
        self._route_warnings(tables, primary)
        self._unrouted_warning(pages, tables, primary)

        title = ("React + Vite frontend (GUI scaffold, runtime blocks)" if self.runtime
                 else "React frontend" + (" (Vite)" if self._vite_config() else ""))
        smap = ScaffoldMap(family=FAMILY, root=self.root, title=title)
        smap.anchors = self.anchors
        smap.entities = {name: self.entities[name] for name in sorted(self.entities)}
        smap.warnings = [text for _, text in sorted(self._warnings, key=lambda w: w[0])]
        smap.do_not_edit = self._do_not_edit()
        nav = self._nav_file(entry, tables, pages)
        smap.summary = self._summary(entry, app, primary, pages, api, theme, nav)
        smap.recipes = self._recipes(entry, primary, pages, api, theme, nav)
        return smap

    def _pages_dir(self, pages: list[_Page], primary: Optional[_Table]) -> str:
        """Where pages live: src/pages, else the folder holding most routed pages."""
        default = self.p("src/pages")
        if any(p.file.startswith(default + "/") for p in pages):
            return default
        folders = [posixpath.dirname(p.file) for p in pages if not primary or p.file != primary.file]
        return max(sorted(set(folders)), key=folders.count) if folders else default

    # -- entry -----------------------------------------------------------

    def _entry_file(self) -> Optional[str]:
        html = self.p("index.html")
        if html in self.file_set:
            for match in _HTML_SCRIPT_RE.finditer(self.index.text(html) or ""):
                target = self.resolve(html, match.group("src"))
                if target and target.endswith(_SOURCE_EXT):
                    return target
        return next((self.p(f"src/{n}") for n in _ENTRY_NAMES if self.p(f"src/{n}") in self.file_set), None)

    def _entry_anchor(self, entry: str) -> None:
        src = self.src(entry)
        if src is None:
            return
        match = _RENDER_RE.search(src.code)
        if not match:
            self.anchors.append(self.anchor("entry", entry, KIND_FILE, "app entry (no render call found)",
                                            1, src.line_count))
            return
        start = src.line(match.start())
        end = src.block_end(start)
        chain: list[str] = []
        for name in re.findall(r"<([A-Z][\w.]*)", src.code[match.start():src.offset(end + 1)]):
            if name not in chain:
                chain.append(name)
        app_line = next((n for n in range(start, end + 1) if re.search(r"<App\b", src.code_line(n))), None)
        self.anchors.append(self.anchor(
            "entry", entry, KIND_BLOCK, "app entry: " + (" > ".join(chain) if chain else "render call"),
            start, end, quote_line=app_line,
            hint="wrap <App /> here for an app-wide provider (auth, data); import side-effect modules "
                 "(axios interceptors, CSS) at the top of this file"))

    # -- routes ----------------------------------------------------------

    def _route_tables(self, preferred: list[str]) -> list[_Table]:
        order = [f for f in preferred if f in self.sources]
        order += [f for f in self.editable if f not in order]
        tables = []
        for rel in order:
            src = self.src(rel)
            if src is None:
                continue
            table = self._jsx_table(src) if "<Route" in src.code else None
            table = table or self._object_table(src)
            if table is not None:
                tables.append(table)
        return tables

    def _jsx_table(self, src: _Src) -> Optional[_Table]:
        events = [(start, attrs, top) for start, attrs, top in _tags(src.code, "Route")]
        events += [(m.start(), None, None) for m in _ROUTE_CLOSE_RE.finditer(src.code)]
        events.sort(key=lambda e: e[0])
        stack: list[str] = []
        routes: list[_Route] = []
        for offset, attrs, top in events:
            if attrs is None:       # </Route>
                if stack:
                    stack.pop()
                continue
            parent = stack[-1] if stack else ""
            raw_path = _attr(attrs, "path") if _has_prop(top, "path") else None
            path = _scalar(raw_path)
            if path is None and raw_path is not None:
                path = "{" + raw_path.strip()[:40] + "}"
            is_index = re.search(r"(?<![\w.-])index\b(?!\s*=\s*\{\s*false)", top) is not None
            full = self._join_path(parent, path, is_index)
            element = _attr(attrs, "element") if _has_prop(top, "element") else ""
            component, target, guarded = self._element_component(element or "")
            if not component:
                named = _attr(attrs, "Component") or _attr(attrs, "component") or ""
                component = named.strip() if re.fullmatch(r"\s*[A-Z][\w$]*\s*", named) else ""
            opens = not top.rstrip().endswith("/")
            kind = "index" if is_index else "path" if path is not None else "layout"
            routes.append(_Route(path=full, component=component, target=target, file=src.rel,
                                 line=src.line(offset), kind=kind, guarded=guarded, parent=opens,
                                 grouping=opens and not component and not element))
            if opens:
                stack.append(full if path is not None else parent)
        if not routes:
            return None
        opened = _ROUTES_OPEN_RE.search(src.code)
        close_line = None
        if opened:
            closed = _ROUTES_CLOSE_RE.search(src.code, opened.end())
            close_line = src.line(closed.start()) if closed else None
            start = src.line(opened.start())
        else:
            start = min(r.line for r in routes)
        end = close_line or max(r.line for r in routes)
        return _Table(file=src.rel, kind="jsx", start=start, end=max(start, end), routes=routes,
                      close_line=close_line)

    def _object_table(self, src: _Src) -> Optional[_Table]:
        """A ``createBrowserRouter([...])`` / ``useRoutes(routes)`` table, children joined to their parents.

        The routes array is the call's literal argument or, for ``useRoutes(routes)``,
        the ``const routes = [...]`` of the same file; the table spans that array.
        """
        match = _ROUTER_CALL_RE.search(src.code)
        if not match:
            return None
        code = src.code
        open_idx = match.end() - 1
        close = _balanced_end(code, open_idx) or len(code)
        start, end_offset = src.line(match.start()), max(open_idx, close - 1)
        array_idx, apart = None, False
        arg = _ROUTER_ARG_RE.match(code, open_idx + 1)
        if arg and arg.group("array"):
            array_idx = arg.start("array")
        elif arg and arg.group("name"):
            decl = re.search(rf"^[ \t]*(?:export\s+)?(?:const|let|var)\s+{re.escape(arg.group('name'))}\b"
                             rf"[^=\n]*=\s*\[", code, re.M)
            if decl:
                array_idx, apart, start = decl.end() - 1, True, src.line(decl.start())
        routes: list[_Route] = []
        close_line = None
        if array_idx is not None:
            array_end = _balanced_end(code, array_idx)        # just past the closing ']', None when unclosed
            items_end = array_end - 1 if array_end else len(code)
            if apart:
                end_offset = max(array_idx, items_end)
            self._object_routes(src, array_idx + 1, max(array_idx + 1, items_end), "", routes, 0)
            if array_end:
                closing = src.line(items_end)
                # Only a ']' opening its own line is a line to insert before.
                close_line = closing if src.code_line(closing).lstrip().startswith("]") else None
        return _Table(file=src.rel, kind=match.group("fn"), start=start,
                      end=max(start, src.line(end_offset)), routes=routes, close_line=close_line)

    def _object_routes(self, src: _Src, lo: int, hi: int, parent: str, routes: list[_Route], depth: int) -> None:
        """Append the route objects of the array whose items lie in ``src.code[lo:hi]``, in source order."""
        code = src.code
        idx = lo
        while idx < hi and depth <= _MAX_ROUTE_DEPTH:
            char = code[idx]
            if char in "\"'`":
                string = _STRING_AT_RE.match(code, idx)
                idx = string.end() if string else idx + 1
                continue
            if char in "([{":
                end = _balanced_end(code, idx)
                stop = min(end, hi) if end else hi
                if char == "{":
                    self._object_route(src, idx, stop, parent, routes, depth)
                idx = max(stop, idx + 1)
                continue
            idx += 1

    def _object_route(self, src: _Src, start: int, end: int, parent: str, routes: list[_Route], depth: int) -> None:
        """One route object ``src.code[start:end]`` (braces included), then its children."""
        lo, hi = start + 1, max(start + 1, end - 1)
        top = _top_level(src.code, lo, hi)
        path_key, index_key = _OBJ_PATH_RE.search(top), _OBJ_INDEX_RE.search(top)
        element_key, named = _OBJ_ELEMENT_RE.search(top), _OBJ_COMPONENT_RE.search(top)
        children = _OBJ_CHILDREN_RE.search(top)
        if not (path_key or index_key or element_key or named or children):
            return                  # not a route object (a loader's options, a handle...)
        path = path_key.group("path") if path_key else None
        is_index = index_key is not None and path is None
        full = self._join_path(parent, path, is_index)
        component, target, guarded = "", "", False
        if element_key:
            # The value runs to the next top-level comma (a comma inside JSX braces is blanked).
            comma = top.find(",", element_key.end())
            component, target, guarded = self._element_component(
                src.code[lo + element_key.end():lo + (comma if comma >= 0 else len(top))])
        if not component and named:
            component = named.group("name")
        key = path_key or index_key
        routes.append(_Route(
            path=full, component=component, target=target, file=src.rel,
            line=src.line(lo + key.start()) if key else src.line(start),
            kind="index" if is_index else "path" if path is not None else "layout", guarded=guarded,
            grouping=children is not None and not element_key and not named, parent=children is not None,
            one_line=src.line(start) == src.line(max(start, end - 1))))
        if children:
            bracket = lo + children.end() - 1
            child_end = _balanced_end(src.code, bracket)
            child_hi = child_end - 1 if child_end and child_end <= hi else hi
            self._object_routes(src, bracket + 1, max(bracket + 1, child_hi), full, routes, depth + 1)

    @staticmethod
    def _join_path(parent: str, path: Optional[str], is_index: bool) -> str:
        """Full path of a nested JSX route (relative paths join their parent's)."""
        if path is None:
            return (parent or "/") if is_index else parent
        if path.startswith(("/", "{")) or (path == "*" and not parent):
            return path
        return f"{parent.rstrip('/')}/{path}"

    @staticmethod
    def _element_component(element: str) -> tuple[str, str, bool]:
        """(component, redirect target, guarded) of a route element expression.

        Only an element whose components are all ``<Navigate>`` is a redirect; in
        ``cond ? <Page /> : <Navigate to="/login" />`` the page is guarded.
        """
        tags = [t for t in re.findall(r"<\s*([A-Z][\w.]*)", element) if t not in ("Route", "Routes")]
        if not tags:
            return "", "", False
        others = [t for t in tags if t != "Navigate"]
        if not others:
            to = re.search(r"""\bto\s*=\s*(?:\{\s*)?(['"`])([^'"`]*)\1""", element)
            return "Navigate", to.group(2) if to else "?", False
        # A guard or <Suspense> wraps the page: the innermost (last) tag is the page.
        return others[-1], "", len(others) < len(tags)

    def _route_anchors(self, table: _Table, nested: list[_Table]) -> None:
        src = self.src(table.file)
        if src is None:
            return
        routes = table.routes
        jsx = table.kind == "jsx"
        redirects = [r for r in routes if r.component == "Navigate"]
        page_routes = [r for r in routes if r.component and r.component != "Navigate" and r.kind != "layout"]
        label = f"route table: {len(page_routes)} page route(s)"
        if redirects:
            label += "; " + ", ".join(f"{r.path} -> {r.target}" for r in redirects[:3])
        self.anchors.append(self.anchor(
            "routes", table.file, KIND_BLOCK, label, table.start, table.end,
            quote_line=table.start if jsx else None,
            hint="one entry per page; paths are case-sensitive"))
        # New routes go before the '/' redirect or the '*' catch-all, else before </Routes> (or the
        # array's closing ']'). A route object spread over several lines has no single line to insert before.
        fallback = [r for r in routes if (r.path in ("*", "/*") or (r.component == "Navigate" and r.path == "/"))
                    and (jsx or r.one_line)]
        insert_line = min((r.line for r in fallback), default=None)
        entry = ("<Route path=\"/x\" element={<Page />} /> lines" if jsx
                 else "`{ path: \"/x\", element: <Page /> },` route objects")
        if insert_line is not None:
            what = ("the catch-all" if any(r.line == insert_line and r.path in ("*", "/*") for r in routes)
                    else "the '/' redirect")
            self.anchors.append(self.anchor(
                "route_insert", table.file, KIND_INSERT, f"new routes go before {what}",
                insert_line, insert_line, quote_line=insert_line,
                hint=f"insert new {entry} BEFORE this line ({what})"))
        elif table.close_line:
            self.anchors.append(self.anchor(
                "route_insert", table.file, KIND_INSERT,
                "new routes go before " + ("</Routes>" if jsx else "the end of the routes array"),
                table.close_line, table.close_line, quote_line=table.close_line,
                hint=f"insert new {entry} BEFORE this line"))
        # New page imports go after the last import of a routed component.
        imports = self.imports(table.file)
        routed = {r.component for r in routes}
        lines = [line for name, (_, _, line) in imports.items() if name in routed] \
            or [line for (_, _, line) in imports.values()]
        if lines:
            self.anchors.append(self.anchor(
                "page_imports", table.file, KIND_INSERT, "new page imports go after the last one",
                max(lines), max(lines), quote_line=max(lines),
                hint=f"add `import NewPage from \"{self._spec_to(table.file, self.pages_dir)}/NewPage\";` "
                     f"AFTER this line"))
        for other in nested:
            stem = posixpath.splitext(posixpath.basename(other.file))[0]
            self.anchors.append(self.anchor(
                f"routes:{stem}", other.file, KIND_BLOCK, f"nested route table: {len(other.routes)} route(s)",
                other.start, other.end, quote_line=other.start if other.kind == "jsx" else None))

    # -- pages -----------------------------------------------------------

    def _resolve_route(self, route: _Route) -> Optional[tuple[str, str]]:
        """(file, component name in that file or '' for its default export) of a routed component."""
        if not route.component or route.component == "Navigate" or "." in route.component:
            return None
        imported = self.imports(route.file).get(route.component)
        if imported:
            spec, name, _ = imported
            target = self.resolve(route.file, spec)
            if not target or not target.endswith(_SOURCE_EXT):
                return None
            route.resolved = target
            return target, ("" if name == "default" else name)
        src = self.src(route.file)
        if src is not None and re.search(
                rf"\b(?:function|const|let|var|class)\s+{re.escape(route.component)}\b", src.code):
            route.resolved = route.file
            return route.file, route.component
        return None

    def _component_span(self, rel: str, wanted: str = "") -> Optional[tuple[str, int, int]]:
        """(name, first line, last line) of component ``wanted`` in ``rel``, or of the one the file exports.

        Without a default export the first exported component wins, then the
        first component; constants (``const COLUMNS = [...]``) never do. None when
        the file defines no component and holds no JSX (a constants or helper module).
        """
        src = self.src(rel)
        if src is None:
            return None
        name = wanted
        if not name:
            match = _DEFAULT_EXPORT_RE.search(src.code)
            name = (match.group("a") or match.group("b")) if match else ""
        defs = list(_COMPONENT_DEF_RE.finditer(src.code))
        if name:
            definition = next((d for d in defs if d.group("name") == name), None)
        else:
            components = [d for d in defs if _is_component_def(d)]
            definition = next((d for d in components if d.group("export")), None) \
                or (components[0] if components else None)
        if definition is None:
            if not name and not (_RETURN_JSX_RE.search(src.code) or _ARROW_JSX_RE.search(src.code)):
                return None
            return (name or posixpath.splitext(posixpath.basename(rel))[0]), 1, src.line_count
        start = src.line(definition.start())
        return name or definition.group("name"), start, src.block_end(start)

    def _pages(self, tables: list[_Table]) -> list[_Page]:
        pages_dir = self.p("src/pages") + "/"
        found: dict[tuple[str, str], _Page] = {}

        def page_of(rel: str, wanted: str = "") -> Optional[_Page]:
            # Keyed by (file, component): a file routed under several names is still one page.
            span = self._component_span(rel, wanted)
            if not span:
                return None
            if (rel, span[0]) not in found:
                found[(rel, span[0])] = _Page(file=rel, component=span[0], start=span[1], end=span[2])
            return found[(rel, span[0])]

        for rel in self.editable:
            if rel.startswith(pages_dir) and not re.search(r"\.(?:test|spec|stories)\.", rel):
                page_of(rel)
        for table in tables:
            for route in table.routes:
                target = self._resolve_route(route)
                if target is None or target[0] in self.runtime:
                    continue
                if route.kind == "layout":
                    # A pathless layout or guard wraps other routes; it is a page only if it already is one.
                    span = self._component_span(*target)
                    page = found.get((target[0], span[0])) if span else None
                    if page is not None:
                        page.layout = True
                    continue
                page = page_of(*target)
                if page is not None and route.path not in page.routes:
                    page.routes.append(route.path)
        pages = [found[k] for k in sorted(found)]
        for page in pages:
            try:
                self._scan_page(page)
            except Exception:   # an unexpected shape costs this page's details, not the map
                logger.debug("Scaffold map: could not scan page %s", page.file, exc_info=True)
        return pages

    def _scan_page(self, page: _Page) -> None:
        src = self.src(page.file)
        if src is None:
            return
        lo, hi = src.offset(page.start), src.offset(page.end + 1)
        # The component's own return is the least indented one (helpers nest deeper).
        returns = list(_RETURN_JSX_RE.finditer(src.code, lo, hi)) or list(_ARROW_JSX_RE.finditer(src.code, lo, hi))
        if returns:
            def column(m: re.Match) -> int:
                return m.start() - src.offset(src.line(m.start()))
            root = min(returns, key=column)
            page.root_line = src.line(src.code.rfind("<", lo, root.end()))
        counts: dict[str, int] = {}
        for start, attrs, top in _tags(src.code, r"[A-Z][\w.]*"):
            if not lo <= start < hi:
                continue
            tag_match = _TAG_NAME_RE.match(src.code, start)
            if not tag_match:
                continue
            tag = tag_match.group(1)
            if tag not in _RUNTIME_BLOCKS and not any(_has_prop(top, p) for p in _BINDING_PROPS):
                continue
            counts[tag] = counts.get(tag, 0) + 1
            block_id = _prop(attrs, top, "id")
            if not isinstance(block_id, str) or not block_id:
                block_id = f"{tag}{counts[tag]}"
            end = start + len(tag_match.group(0)) + len(attrs)
            if not top.rstrip().endswith("/"):
                closing = src.code.find(f"</{tag}>", end)
                end = closing if closing >= 0 else end
            block = _Block(tag=tag, block_id=block_id, start=src.line(start), end=src.line(end), detail=block_id)
            self._block_binding(block, attrs, top)
            page.blocks.append(block)

    @staticmethod
    def _block_binding(block: _Block, attrs: str, top: str) -> None:
        """Fill the block's entity names and endpoint segments from literal props only.

        A TableBlock's lookup columns name other entities: never read them.
        """
        entity = _prop(attrs, top, "entity")
        if isinstance(entity, str) and entity:
            block.names.append(entity)
        endpoint = _prop(attrs, top, "endpoint")
        if isinstance(endpoint, str):
            block.endpoint = endpoint
            block.segments.append(_first_segment(endpoint))
            method = re.search(r"/methods/([\w-]+)", endpoint)
            if method:
                block.detail = method.group(1)
        binding = _prop(attrs, top, "dataBinding")
        if isinstance(binding, dict):
            if isinstance(binding.get("entity"), str) and binding["entity"]:
                block.names.append(binding["entity"])
            if isinstance(binding.get("endpoint"), str):
                block.endpoint = block.endpoint or binding["endpoint"]
                block.segments.append(_first_segment(binding["endpoint"]))
        for key, fields in (("series", ("endpoint", "dataSource")), ("dataSources", ("endpoint", "domain"))):
            value = _prop(attrs, top, key)
            for item in value if isinstance(value, list) else []:
                if isinstance(item, dict):
                    block.segments += [_first_segment("/" + item[f].lstrip("/")) for f in fields
                                       if isinstance(item.get(f), str)]
        action = _prop(attrs, top, "action")
        if block.tag == "CrudButton" and isinstance(action, str) and action:
            block.detail = action

    def _resolve_entities(self, pages: list[_Page]) -> None:
        """Map every block to entity names: the model's spelling first, then the code's."""
        for page in pages:
            for block in page.blocks:
                for name in block.names:
                    self.canon.setdefault(name.lower(), name)
        for page in pages:
            for block in page.blocks:
                out: list[str] = [self.canon[name.lower()] for name in block.names]
                for seg in filter(None, block.segments):
                    low = seg.lower()
                    name = self.canon.get(low) or (self.canon.get(low[:-1]) if low.endswith("s") else None) or seg
                    out.append(name)
                block.entities = list(dict.fromkeys(out))

    def warn(self, rank: int, text: str) -> None:
        """Record a pitfall; rank 0 breaks the build, 1 routing, 2 Phase 3 blockers, 3 reachability, 4 advice."""
        self._warnings.append((rank, text))

    def _entity(self, name: str) -> EntityMap:
        if name not in self.entities:
            self.entities[name] = EntityMap(name=name)
        return self.entities[name]

    def _page_anchors(self, pages: list[_Page]) -> None:
        components = [p.component for p in pages]
        for page in pages:
            what = page.component if components.count(page.component) == 1 \
                else posixpath.splitext(self.short(page.file))[0]
            bound = list(dict.fromkeys(e for b in page.blocks for e in b.entities))
            at = ", ".join(page.routes) if page.routes else "a pathless layout route" if page.layout else "no route"
            label = f"page {page.component} at {at}"
            unbound = [f"{b.tag} {b.detail}" for b in page.blocks if not b.entities]
            if unbound:
                label += "; " + ", ".join(unbound[:4])
            hint = ""
            root_quote = self.index.unique_quote(page.file, page.root_line) if page.root_line else ""
            if root_quote:
                hint = f"new JSX goes right after the root line `{root_quote.strip()}`"
                if self.runtime:
                    hint += " (the root's closing tag shares a line with the last child)"
            # A page bound to exactly one entity is that entity's page; others stay app-level.
            entity = bound[0] if len(bound) == 1 else None
            anchor = self.anchor(f"page:{what}", page.file, KIND_FUNCTION, label, page.start, page.end,
                                 quote_line=page.start, hint=hint, entity=entity)
            (self._entity(entity).anchors if entity else self.anchors).append(anchor)
            for block in page.blocks:
                if not block.entities:
                    continue
                name = block.entities[0]
                also = f" (also {', '.join(block.entities[1:])})" if len(block.entities) > 1 else ""
                self._entity(name).anchors.append(self.anchor(
                    f"block:{name.lower()}:{what}.{block.block_id}", page.file, KIND_BLOCK,
                    f"{block.tag} {block.detail} on page {page.component}{also}"
                    + (f", endpoint {block.endpoint}" if block.endpoint else ""), block.start, block.end,
                    hint="configure it through its props here, never in the runtime component"
                    if self.runtime else "", entity=name))
            for name in bound:
                # Tag plus method/action; block ids stay in the anchors (locate_code entity=).
                blocks = [b.tag + (f" {b.detail}" if b.detail != b.block_id else "")
                          for b in page.blocks if name in b.entities]
                more = f", +{len(blocks) - 4}" if len(blocks) > 4 else ""
                where = page.file[len(self.pages_dir) + 1:] if page.file.startswith(self.pages_dir + "/") \
                    else self.short(page.file)
                self._entity(name).summary.append(f"{where} {at} ({', '.join(blocks[:4])}{more})")

    def _form_stubs(self, pages: list[_Page]) -> None:
        owner = {p.file: p.component for p in pages}
        stubs: dict[str, list[str]] = {}        # file -> its no-op form ids, in source order
        where_of: dict[str, str] = {}
        for rel in self.editable:
            src = self.src(rel)
            if src is None or "<form" not in src.code:
                continue
            for n, (start, attrs, top) in enumerate(_tags(src.code, "form"), 1):
                if not any(rx.search(attrs) for rx in _NOOP_SUBMIT_RES):
                    continue
                form_id = _prop(attrs, top, "id")
                form_id = form_id if isinstance(form_id, str) and form_id else f"form{n}"
                close = src.code.find("</form>", start)
                where = owner.get(rel) or posixpath.splitext(posixpath.basename(rel))[0]
                self.anchors.append(self.anchor(
                    f"form:{where}.{form_id}", rel, KIND_STUB, f"form {form_id} on {where}: submit does nothing",
                    src.line(start), src.line(close) if close >= 0 else src.line(start),
                    quote_line=src.line(start), status="no-op onSubmit",
                    hint="post its inputs to the backend (axios) or replace it with a bound "
                         "<FormBlock endpoint entity fields> whose inputs carry name="))
                stubs.setdefault(rel, []).append(form_id)
                where_of[rel] = where
        count = sum(len(ids) for ids in stubs.values())
        if count == 1:
            rel, (form_id,) = next(iter(stubs.items()))
            self.warn(2, f"{rel}: form {form_id} in {where_of[rel]} only calls preventDefault on submit, so "
                         f"it saves nothing (a Phase 3 blocker): post its inputs or make it a bound <FormBlock>.")
        elif count:
            # One line for all of them: the overview budget is shared with the backend's.
            files = [f"{rel} ({', '.join(ids)})" for rel, ids in stubs.items()]
            self.warn(2, f"{count} forms only call preventDefault on submit, so they save nothing (a Phase 3 "
                         f"blocker): {_names(files, 2)}; post their inputs or make them bound <FormBlock>s "
                         f"(locate_code query='no-op onSubmit').")

    # -- API layer -------------------------------------------------------

    def _api_layer(self) -> _Api:
        api = _Api()
        for rel in self.sources:
            src = self.src(rel)
            if src is None:
                continue
            lines: set[int] = set()
            for match in _ENV_BASE_RE.finditer(src.code):
                lines.add(src.line(match.start()))
                api.env_var = api.env_var or match.group("var")
                fallback = _FALLBACK_RE.match(src.code, match.end())
                if fallback and not api.fallback:
                    api.fallback = fallback.group("url")
            for regex in (_API_BASE_DECL_RE, _API_BASE_OPTION_RE, _HARDCODED_BASE_RE):
                for match in regex.finditer(src.code):
                    if regex is _HARDCODED_BASE_RE and not _BASE_NAME_RE.search(match.group("name")):
                        continue
                    lines.add(src.line(match.start("url")))
                    api.fallback = api.fallback or match.group("url")
            if lines:
                api.sites[rel] = sorted(lines)
        api.client = self._client_file(api.sites)
        if api.client:
            src = self.src(api.client)
            helpers = [m.group("a") or m.group("b") for m in _EXPORTED_FN_RE.finditer(src.code)] if src else []
            shown = ", ".join(helpers[:6]) + (f", +{len(helpers) - 6}" if len(helpers) > 6 else "")
            self.anchors.append(self.anchor(
                "api_client", api.client, KIND_FILE, "shared API client" + (f": {shown}" if shown else ""),
                1, src.line_count if src else 1, quote_line=api.sites.get(api.client, [None])[0],
                hint="add request helpers here and import them in pages; copy backend paths exactly"))
            outside = sorted(f for f in api.sites if f != api.client and f not in self.runtime)
            if outside:
                self.warn(4, f"{len(outside)} file(s) re-declare the API base instead of importing it "
                             f"from {api.client}: {_names(outside, 3)}.")
            return api
        editable = sorted(f for f in api.sites if f not in self.runtime)
        if editable:
            first = editable[0]
            self.anchors.append(self.anchor(
                "api_base", first, KIND_BLOCK, "API base URL declaration", api.sites[first][0], api.sites[first][0],
                quote_line=api.sites[first][0],
                hint="there is no shared client: move this into one src/api/client module and import it"))
        # The runtime blocks' own copies are the scaffold's convention (they are not edited): the
        # summary and the 'Call a backend endpoint' recipe say so; only editable copies are a pitfall.
        if len(editable) >= 2:
            client = self.p("src/api/client.ts" if any(f.endswith(".tsx") for f in self.sources)
                            else "src/api/client.js")
            self.warn(4, f"{len(editable)} editable files declare their own API base ({_names(editable, 2)}) "
                         f"and there is no shared client: move it into {client} and import it.")
        return api

    def _client_file(self, sites: dict[str, list[int]]) -> Optional[str]:
        """The shared API module: an api/client-named file with the base or helpers, or an exported axios.create().

        A file that only lives in an api/services/lib folder must also talk HTTP
        (declare the base, create an axios instance, or call axios/fetch): a
        ``src/lib/format.js`` of date helpers is not a client.
        """
        candidates = []
        for rel in self.editable:
            src = self.src(rel)
            if src is None:
                continue
            inner = self.short(rel)[len("src/"):]
            stem = posixpath.splitext(posixpath.basename(inner))[0].lower().replace("-", "").replace("_", "")
            creates = re.search(r"\baxios\s*\.\s*create\s*\(", src.code) is not None
            exported_fn = _EXPORTED_FN_RE.search(src.code) is not None
            if stem in _CLIENT_STEMS:
                named = creates or rel in sites or exported_fn
            elif inner.split("/")[0] in _CLIENT_DIRS:
                named = creates or rel in sites or (exported_fn and _HTTP_CALL_RE.search(src.code) is not None)
            else:
                named = False
            exported = re.search(r"^export\b", src.code, re.M) is not None
            # A page building a private axios instance is not a shared client.
            if named or (creates and exported
                         and not any(_is_component_def(d) for d in _COMPONENT_DEF_RE.finditer(src.code))):
                candidates.append((0 if creates else 1, rel))
        return min(candidates)[1] if candidates else None

    # -- styles and config ----------------------------------------------

    def _theme(self, entry: Optional[str]) -> dict:
        src_dir = self.p("src") + "/"
        css = [f for f in self.files if f.startswith(src_dir) and f.endswith(_STYLE_EXT)]
        imported: set[str] = set()
        for rel in self.sources:
            for spec, _ in self.specs(rel):
                target = self.resolve(rel, spec) if spec.split("?")[0].endswith(_STYLE_EXT) else None
                if target:
                    imported.add(target)
        for rel in css:
            for match in _CSS_IMPORT_RE.finditer(self.index.text(rel) or ""):
                target = self.resolve(rel, match.group("spec"))
                if target:
                    imported.add(target)
        html = self.p("index.html")
        html_text = (self.index.text(html) or "") if html in self.file_set else ""
        for match in _HTML_LINK_RE.finditer(html_text):
            target = self.resolve(html, match.group("href"))
            if target:
                imported.add(target)
        entry_css = [t for spec, _ in (self.specs(entry) if entry else [])
                     for t in [self.resolve(entry, spec)] if t and t.endswith(_STYLE_EXT)]
        main = next((f for f in entry_css if not f.endswith(("design.css", "design-overrides.css"))), None) \
            or next((f for f in (self.p("src/index.css"), self.p("src/App.css")) if f in imported), None)
        theme = {"main": main, "tailwind": "", "design": None,
                 "unimported": [] if self.partial()
                 else sorted(f for f in css if f not in imported and f not in self.runtime),
                 "inline": False}
        if main:
            lines = self.index.lines(main)
            tw_line = next((i for i, line in enumerate(lines, 1)
                            if re.match(r"""\s*@import\s+['"]tailwindcss['"]""", line)), None)
            if tw_line:
                theme["tailwind"] = "v4"
            elif any(re.match(r"\s*@tailwind\s+base", line) for line in lines):
                theme["tailwind"] = "v3"
            self.anchors.append(self.anchor(
                "theme", main, KIND_FILE,
                "global stylesheet" + (f" (Tailwind {theme['tailwind']})" if theme["tailwind"] else ""),
                1, max(1, len(lines)), quote_line=tw_line,
                hint=("add @theme { ... } tokens and global rules after @import \"tailwindcss\"" if tw_line
                      else "global rules and CSS variables")))
        if self.p("src/design.css") in self.file_set:
            theme["design"] = self.p("src/design.css")
        overrides = self.p("src/design-overrides.css")
        if overrides in self.file_set:
            self.anchors.append(self.anchor("design_overrides", overrides, KIND_FILE,
                                            "design-system overrides (--ds-* variables)",
                                            1, max(1, len(self.index.lines(overrides)))))
        theme["inline"] = any("style={{" in src.code for src in map(self.src, self.editable)
                              if src is not None and src.rel.startswith(self.pages_dir + "/"))
        return theme

    def _config_anchors(self) -> None:
        package = self.p("package.json")
        lines = self.index.lines(package)
        deps = next((i for i, line in enumerate(lines, 1) if '"dependencies"' in line), None)
        self.anchors.append(self.anchor(
            "package_json", package, KIND_FILE, "package.json (dependencies, scripts)", 1, max(1, len(lines)),
            quote_line=deps, hint="add a dependency as one line; never rewrite the file"))
        vite = self._vite_config()
        if vite:
            text = self.index.text(vite) or ""
            port = re.search(r"\bport\s*:\s*(\d+)", text)
            out_dir = re.search(r"""\boutDir\s*:\s*['"]([^'"]+)""", text)
            facts = [f"dev port {port.group(1)}" if port else "", f"outDir {out_dir.group(1)}" if out_dir else "",
                     "dev proxy" if re.search(r"\bproxy\s*:", text) else ""]
            facts = [f for f in facts if f]
            vlines = self.index.lines(vite)
            plugins = next((i for i, line in enumerate(vlines, 1) if re.search(r"\bplugins\s*:", line)), None)
            self.anchors.append(self.anchor(
                "vite_config", vite, KIND_FILE, "Vite config" + (f" ({', '.join(facts)})" if facts else ""),
                1, max(1, len(vlines)), quote_line=plugins))
        tsconfig = self.p("tsconfig.json")
        if tsconfig in self.file_set:
            self.anchors.append(self.anchor("tsconfig", tsconfig, KIND_FILE, "TypeScript config",
                                            1, max(1, len(self.index.lines(tsconfig)))))
        dockerfile = self.p("Dockerfile")
        if dockerfile in self.file_set:
            dlines = self.index.lines(dockerfile)
            build = next((i for i, line in enumerate(dlines, 1) if re.match(r"\s*RUN\s+npm\s+run\s+build", line)),
                         None)
            self.anchors.append(self.anchor(
                "dockerfile", dockerfile, KIND_FILE, "frontend Dockerfile", 1, max(1, len(dlines)),
                quote_line=build,
                hint="VITE_* values are inlined at build time: add ARG + ENV before this line" if build else ""))

    def _do_not_edit(self) -> list[str]:
        if not self.runtime:
            return []
        out = [self.p(d) + "/*" for d in _RUNTIME_DIRS if any(f.startswith(self.p(d) + "/") for f in self.runtime)]
        out += [self.p(f) for f in _RUNTIME_FILES if self.p(f) in self.runtime]
        if self.p("src/design.css") in self.file_set:
            out.append(self.p("src/design.css"))
        return out

    # -- warnings --------------------------------------------------------

    def _missing_import_warnings(self, primary: Optional[_Table]) -> None:
        reported = extra = 0
        for rel in self.sources:
            missing = sorted({spec for spec, _ in self.specs(rel) if self.missing(rel, spec)})
            if not missing:
                continue
            if reported >= _MAX_WARNINGS:
                extra += 1
                continue
            reported += 1
            shown = ", ".join(f"'{s}'" for s in missing[:3]) + (f" (+{len(missing) - 3})" if len(missing) > 3 else "")
            one = len(missing) == 1
            if rel in self.runtime and not self.importers().get(rel) and not self.partial():
                fix = "nothing imports this runtime file, so delete it"
            else:
                fix = (f"create {'it' if one else 'them'} or remove the import"
                       + (" and its <Route>" if primary and rel == primary.file else ""))
            self.warn(0, f"{rel} imports {shown}, which {'does' if one else 'do'} not exist, so the build "
                         f"fails: {fix}.")
        if extra:
            self.warn(0, f"{extra} more file(s) import modules that do not exist.")

    def _route_warnings(self, tables: list[_Table], primary: Optional[_Table]) -> None:
        for table in tables:
            counts: dict[str, int] = {}
            for route in table.routes:
                if route.declared:
                    counts[route.path] = counts.get(route.path, 0) + 1
            for path, count in sorted(counts.items()):
                if count > 1 and path:
                    self.warn(1, f"{table.file} declares path {path} {count} times; only one of those "
                                 f"<Route>s ever renders.")
            for route in table.routes:
                comp = route.component
                if not comp or comp == "Navigate" or "." in comp or route.resolved is not None:
                    continue
                if comp in self.imports(table.file):
                    continue    # a package component, or a missing module reported as such
                self.warn(1, f"{table.file}: the route {route.path} renders <{comp} />, which is "
                             f"neither imported nor defined in that file.")
        if primary and primary.routes:
            # A pathless layout or a bare grouping route renders nothing at '/' by itself; '*' matches '/'.
            has_home = any(r.kind != "layout" and not r.grouping
                           and (r.path in ("/", "", "*", "/*") or r.path.startswith("{"))
                           for r in primary.routes)
            if not has_home:
                self.warn(1, f"{primary.file} has no route for '/' (no index route or <Navigate>), so "
                             f"the app renders blank on first load.")

    def _unrouted_warning(self, pages: list[_Page], tables: list[_Table], primary: Optional[_Table]) -> None:
        if not primary or self.partial():
            return
        table_files = {t.file for t in tables}
        orphans = [page.file for page in pages
                   if not page.routes and not page.layout and not self._rendered_elsewhere(page.file, table_files)]
        if not orphans:
            return
        where = self.ref("route_insert") or self.ref("routes") or primary.file
        if len(orphans) == 1:
            self.warn(3, f"{orphans[0]} has no <Route>, so it is unreachable: import it in "
                         f"{primary.file} and add a <Route> ({where}).")
        else:
            self.warn(3, f"{len(orphans)} pages have no <Route> and are unreachable "
                         f"({_names(orphans, 3)}): register them in {primary.file} ({where}).")

    def _rendered_elsewhere(self, page_file: str, table_files: set[str]) -> bool:
        """Whether a file other than the page renders it: any importer, or a route file using its tag."""
        for importer in self.importers().get(page_file, set()) - {page_file}:
            if importer not in table_files:
                return True
            src = self.src(importer)
            names = [name for name, (spec, _, _) in self.imports(importer).items()
                     if self.resolve(importer, spec) == page_file]
            if src is not None and any(re.search(rf"<{re.escape(n)}\b", src.code) for n in names):
                return True
        return False

    # -- overview text ---------------------------------------------------

    def _nav_file(self, entry: Optional[str], tables: list[_Table], pages: list[_Page]) -> Optional[str]:
        """The file holding the app-wide nav (<nav>, <NavLink>, <Link>), or None.

        Looked for in the shell: the route file, the entry, the layouts routes
        nest children in, every other file with an <Outlet>, then one hop further
        in the local components the shell renders (a <NavBar /> in App.tsx).
        Pages are never the nav through that hop: their links are content.
        """
        hosts = [f for f in (tables[0].file if tables else None, entry) if f]
        hosts += [r.resolved for t in tables for r in t.routes if r.parent and r.resolved in self.editable]
        hosts += [rel for rel in self.editable if "<Outlet" in self._code(rel)]
        hosts = list(dict.fromkeys(hosts))
        page_files = {p.file for p in pages}
        candidates = list(hosts)
        for host in hosts:
            code = self._code(host)
            for name, (spec, _, _) in sorted(self.imports(host).items(), key=lambda item: item[1][2]):
                target = self.resolve(host, spec)
                if target and target in self.editable and target not in candidates and target not in page_files \
                        and re.search(rf"<{re.escape(name)}\b", code):
                    candidates.append(target)
        return next((rel for rel in candidates if _NAV_RE.search(self._code(rel))), None)

    def _code(self, rel: Optional[str]) -> str:
        src = self.src(rel)
        return src.code if src is not None else ""

    def _summary(self, entry: Optional[str], app: Optional[str], primary: Optional[_Table],
                 pages: list[_Page], api: _Api, theme: dict, nav: Optional[str]) -> list[str]:
        """Line-free one-liners; paths are relative to the root the section header names."""
        out = []
        parts = []
        router = next((m.group(1) for rel in (entry, primary.file if primary else None)
                       for m in [_ROUTER_KIND_RE.search(self.src(rel).code if self.src(rel) else "")] if m), None)
        if entry:
            parts.append(f"entry {self.short(entry)}" + (f" ({router})" if router else ""))
        if primary:
            src = self.src(primary.file)
            providers = sorted(set(re.findall(r"<([A-Z]\w*Provider)\b", src.code)) - {"RouterProvider"}) \
                if src else []
            parts.append(f"routes in {self.short(primary.file)}"
                         + (f" inside {', '.join(providers)}" if providers else ""))
        elif app:
            parts.append(f"no router: {self.short(app)} renders the UI")
        parts.append(f"nav in {self.short(nav)}" if nav else "no global nav")
        out.append("; ".join(parts))
        listed = [r for r in primary.routes if not r.grouping] if primary else []
        if listed:
            shown = []
            for r in listed[:_MAX_LIST]:
                component = r.component or "(inline JSX)"
                if r.component == "Navigate":
                    shown.append(f"{r.path} -> {r.target}")
                elif r.kind == "layout":
                    shown.append(f"(layout) {component}")
                else:
                    shown.append(f"{r.path}{' (index)' if r.kind == 'index' else ''} {component}"
                                 + (" (guarded)" if r.guarded else ""))
            more = f", +{len(listed) - _MAX_LIST} more" if len(listed) > _MAX_LIST else ""
            out.append("routes: " + ", ".join(shown) + more)
        if any(p.file.startswith(self.pages_dir + "/") for p in pages):
            out.append(f"pages: {self.short(self.pages_dir)}/*, one component per file"
                       + ("; widgets are runtime blocks set up by literal JSON props" if self.runtime else ""))
        data = [f"shared API client {self.short(api.client)}" if api.client
                else "axios in the runtime blocks" if self.runtime else ""]
        if api.env_var or api.fallback:
            data.append("base " + " || ".join(filter(None, (api.env_var, api.fallback))))
        if not api.client and api.sites:
            data.append("no shared client")
        data = [d for d in data if d]
        if data:
            out.append("data: " + ", ".join(data))
        config = [self.short(a.file) for a in self.anchors
                  if a.id.split(":", 1)[1] in ("package_json", "vite_config", "tsconfig", "dockerfile")]
        if theme.get("main"):
            config.append(f"styles {self.short(theme['main'])}"
                          + (f" (Tailwind {theme['tailwind']})" if theme["tailwind"] else ""))
        out.append("config: " + ", ".join(config))
        return out

    def _recipes(self, entry: Optional[str], primary: Optional[_Table], pages: list[_Page],
                 api: _Api, theme: dict, nav: Optional[str]) -> list[Recipe]:
        """Common requests for this frontend, terse: they share the overview budget with the backend's.

        A step names an anchor id only when this map emitted it (``ref``), else the file.
        """
        ts = any(f.endswith(".tsx") for f in self.sources)
        pages_dir = self.pages_dir
        example = next((p for p in pages if p.file.startswith(pages_dir + "/")), None)
        recipes = []

        steps = [f"New {pages_dir}/<Name>{'.tsx' if ts else '.jsx'}"
                 + (f" (copy {posixpath.basename(example.file)})" if example else "") + ";"]
        if primary:
            imports = self.ref("page_imports")
            insert = self.ref("route_insert") or self.ref("routes")
            steps.append(f"import it {f'at {imports}' if imports else f'in {primary.file}'}, add its "
                         f"{'<Route>' if primary.kind == 'jsx' else 'route object'} "
                         f"{f'at {insert}' if insert else f'in {primary.file}'};")
        else:
            steps.append("no router yet: add react-router-dom <Routes> for a second page;")
        steps.append(f"link it from {nav}." if nav else "no global nav: link it from a page.")
        recipes.append(Recipe("Add a page", steps))

        if self.runtime:
            recipes.append(Recipe("Add a field to a table or form", [
                "Set it in the page's TableBlock options.formColumns / FormBlock fields props, never the runtime; "
                "the backend <Entity>Create and handlers need it too."]))
        else:
            recipes.append(Recipe("Add a field to a form", [
                "Add the input and send it in the submit payload; the backend create schema must accept it."]))

        if api.client:
            client = self.ref("api_client")
            call = [f"Add a helper to {api.client}" + (f" ({client});" if client else ";")]
        else:
            expr = (f" = `import.meta.env.{api.env_var} || \"{api.fallback}\"`"
                    if api.env_var.startswith("VITE_") and api.fallback else "")
            call = [f"export API_BASE{expr} once from {self.p('src/api/client.ts' if ts else 'src/api/client.js')} "
                    f"and import it;"]
        methods = any(b.tag == "MethodButton" for p in pages for b in p.blocks)
        call.append("copy backend paths exactly (slashes kept, no /api, no plurals)" + (";" if methods else "."))
        if methods:
            call.append("methods: a <MethodButton> on that entity's table.")
        recipes.append(Recipe("Call a backend endpoint", call))

        root, routes = self.ref("entry"), self.ref("routes")
        auth = [f"Login page + auth provider wrapping the app {f'at {root}' if root else 'at its root'};"]
        if self.runtime:
            auth.append("one axios.interceptors.request.use (the runtime blocks use the default axios);")
        elif api.client:
            auth.append(f"attach the token in {api.client};")
        auth.append(f"guard routes at {routes}." if routes else "guard pages where they render.")
        recipes.append(Recipe("Add auth / login", auth))

        theme_steps = []
        if theme.get("main"):
            tokens = ", Tailwind v4 @theme" if theme["tailwind"] == "v4" else ""
            anchor = self.ref("theme")
            theme_steps.append(f"{theme['main']}" + (f" ({anchor}{tokens});" if anchor else ";"))
        if theme.get("design"):
            theme_steps.append(f"never edit {theme['design']}: override its --ds-* variables in "
                               f"{self.p('src/design-overrides.css')}, imported after it;")
        if theme.get("unimported"):
            files = [self.short(f) for f in theme["unimported"]]
            theme_steps.append(f"{_names(files, 2)} {'is' if len(files) == 1 else 'are'} imported nowhere;")
        if theme.get("inline"):
            theme_steps.append("inline style={{...}} on pages beats CSS;")
        if theme_steps:
            theme_steps[-1] = theme_steps[-1][:-1] + "."
            recipes.append(Recipe("Change the theme", theme_steps))
        for recipe in recipes:      # steps are clauses of one sentence: capitalise the first, unless a path
            first = recipe.steps[0]
            if "/" not in first.split(" ", 1)[0]:
                recipe.steps[0] = first[0].upper() + first[1:]
        return recipes


# --------------------------------------------------------------------------- entry point

def extract(index: SourceIndex, class_names: list[str]) -> list[ScaffoldMap]:
    """One :class:`ScaffoldMap` per React frontend root in the workspace ([] when there is none)."""
    roots = _frontend_roots(index)
    maps = []
    used_ids: set[str] = set()          # shared, so ids stay unique across roots
    families: set[str] = set()
    for root in roots:
        # Same slug rule as the other families ('react@web_app_frontend' when several roots exist);
        # 'front/end' and 'front_end' share a slug, so a later one gets a numbered family of its own.
        base = FAMILY if len(roots) == 1 else f"{FAMILY}@{re.sub(r'[^0-9A-Za-z]+', '_', root) or 'root'}"
        fam, n = base, 2
        while fam in families:
            fam, n = f"{base}_{n}", n + 1
        families.add(fam)
        try:
            maps.append(_RootMapper(index, root, class_names, roots, fam, used_ids).build())
        except Exception:   # one unmappable root must not hide the others
            logger.debug("Scaffold map: React root %r could not be mapped", root, exc_info=True)
    return maps
