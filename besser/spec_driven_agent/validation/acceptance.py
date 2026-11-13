"""Model-derived acceptance matrix for generated apps.

For every class in the domain model, three static discovery signals
indicate whether the entity appears in the app (not executed acceptance):

* ``route``  — the backend exposes REST routes for it (checked against
  the same static route parse the endpoint manifest uses). That parse
  reads FastAPI decorators in ``.py`` files only, so on any other stack
  the cell is ``None`` (unmeasured) rather than ``False``;
* ``page``   — some frontend file is about it (name or content match);
* ``create`` — a frontend file about it contains a POST, OR calls a
  shared api-client helper or React hook that issues one (a generic form is about the
  entities its imported config module declares), OR a ``<TableBlock>`` bound
  to it: ``dataBinding`` naming this entity and an endpoint, plus a
  non-empty ``options.formColumns``. The GUI-model scaffold's only
  generated create path issues its POST from a shared runtime component
  (``TableComponent``), never a literal call in the page file itself, so
  the binding — not the call site — is the signal.

Most generated React frontends put every request behind one api module
(``api.create('carpark', payload)`` → ``fetch(url, {method:'POST'})``),
so the literal POST never appears in the file that names the entity, and a
``.post(`` scan alone would report "no create path" for working apps.
``_client_create_calls`` resolves those calls instead of demanding a
literal in one file.

The matrix is deliberately REPORT-ONLY (warnings + a recipe field, never
blockers): a GUI-model-driven run may legitimately scope the UI to a
subset of entities, so "entity X has no page" is a visibility signal for
the checklist and the recipe, not a defect the fix loop must burn turns
on. Blocker-level enforcement stays with ``contract_checks``.
"""

from __future__ import annotations

import functools
import os
import re
from besser.utilities.workspace_fs import walk_plain

_FRONTEND_EXTS = (".js", ".jsx", ".ts", ".tsx")
_SKIP_DIRS = ("node_modules", "dist", "build", "__pycache__")

# A POST issued from frontend code — axios/api `.post(`, the method as an
# argument (`request('POST', path)`), or fetch with method POST; `url` is the
# literal path a fetch-style call names before its options.
_POST_RE = re.compile(
    r"\.post\s*\(|\(\s*['\"]POST['\"]\s*,"
    r"|(?:\(\s*['\"](?P<url>[^'\"]{1,200})['\"]\s*,\s*\{[^{}]{0,200}?)?method\s*:\s*['\"]POST['\"]",
    re.IGNORECASE)
_FIRST_ARG_RE = re.compile(r"\s*['\"]([^'\"]{1,200})['\"]")
# The rest of a one-expression arrow helper up to its call: `(r, b) => api`.
_ARROW_BODY_RE = re.compile(r"\s*(?:async\s+)?\([^()]*\)\s*=>\s*[\w$.\s]*")

# The GUI-model scaffold's runtime-bound create path: a <TableBlock> whose
# dataBinding names the entity + endpoint and whose options carry at least
# one editable formColumns entry. Its POST is issued by the shared
# TableComponent it delegates to (axios.post gated on modalMode === 'add'),
# not by a literal call in the page file — a `.post(` scan is structurally
# blind to it. DOTALL: the scaffold sometimes pretty-prints this tag across
# many lines; dataBinding's own body never contains `}` so `[^}]*` closes on
# the real end without needing DOTALL there.
_TABLE_BLOCK_RE = re.compile(r"<TableBlock\b.*?/>", re.DOTALL)
_DATA_BINDING_RE = re.compile(r"dataBinding\s*=\s*\{\{([^}]*)\}\}")
_ENTITY_RE = re.compile(r'"entity"\s*:\s*"([^"]*)"')
_ENDPOINT_RE = re.compile(r'"endpoint"\s*:\s*"([^"]*)"')
_NONEMPTY_FORM_COLUMNS_RE = re.compile(r'"formColumns"\s*:\s*\[\s*[^\]\s]')

# Resolution of a create issued through a shared api-client module.
_IMPORT_RE = re.compile(
    r"\bimport\s+(?P<clause>[^'\";]+?)\s+from\s+['\"](?P<spec>\.[^'\"]*)['\"]")
_NAMESPACE_IMPORT_RE = re.compile(r"\*\s+as\s+([A-Za-z_$][\w$]*)")
_NAMED_IMPORT_RE = re.compile(r"\{([^}]*)\}")
_LEADING_NAME_RE = re.compile(r"^([A-Za-z_$][\w$]*)")
# A named definition whose value is itself callable. The second branch must
# require a function on the right: `const response = await fetch(...)` would
# otherwise read as the helper enclosing the POST and the real one — the
# exported `createItem` around it — would never be found.
_HELPER_DEF_RE = re.compile(
    r"\b(?P<fn>[A-Za-z_$][\w$]*)\s*(?=\([^()]*\)\s*\{)"
    r"|(?:^|[,{;(]|\bconst\s+|\blet\s+|\bvar\s+|\bexport\s+)[ \t]*"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*[:=](?![=>])"
    r"(?=\s*(?:async\s+)?(?:function\b|\())",
    re.MULTILINE,
)
_NOT_A_HELPER = frozenset({
    "if", "for", "while", "switch", "catch", "function", "return", "await",
    "typeof", "new", "delete", "void", "do", "else",
})
_STRING_ARG_RE = re.compile(r"['\"]([^'\"]{1,60})['\"]")
_BLANKABLE_RE = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/"
    r"|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`")
_KEEP_NEWLINES_RE = re.compile(r"[^\n]")
# A string compared against a value (`key === 'booking'`, `case 'booking':`).
_COMPARED_STRING_RE = re.compile(
    r"(?:[!=]==?|\bcase)\s*(['\"`])[^'\"`\n]*\1|(['\"`])[^'\"`\n]*\2\s*[!=]==?")
# A property read (`err.message`, `e?.message`, `row['message']`), not a call.
_PROPERTY_READ_RE = re.compile(
    r"(?:\?\.|(?<!\.)\.)\s*[A-Za-z_$][\w$]*(?![\w$])(?!\s*\()"
    r"|(?<=[\w$)\]])\s*(?:\?\.)?\[\s*(['\"`])[^'\"`\n]*\1\s*\]")


def _blank_literals(text: str) -> str:
    """Blank comments and string bodies, keeping every offset in place.

    Offsets and newlines are preserved so `_HELPER_DEF_RE` positions still
    index into the original source. Substituting only the matched spans keeps
    memory proportional to the literals, not to the file.
    """
    return _BLANKABLE_RE.sub(lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group()), text)


def _blank_comments(text: str) -> str:
    """Blank comments only: a doc comment's example path names no resource."""
    return _BLANKABLE_RE.sub(
        lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group()) if m.group().startswith("/") else m.group(), text)


def _post_path(source: str, post: re.Match) -> str | None:
    """The literal path a POST targets, or None when it is built at run time."""
    if post.group("url"):
        return post.group("url")
    first = _FIRST_ARG_RE.match(source, post.end()) if post.group().endswith(("(", ",")) else None
    return first.group(1) if first else None


def _direct_create_calls(source: str) -> list:
    """Creates ``source`` issues itself, shaped like ``_client_create_calls``."""
    code = _blank_comments(source)
    return [[path] if (path := _post_path(code, m)) else None for m in _POST_RE.finditer(code)]


def _bracket_depth(blanked: str, start: int, end: int) -> int:
    depth = 0
    for char in blanked[start:end]:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth


def _post_helper_paths(source: str) -> dict[str, list[str] | None]:
    """{helper name: the literal paths its POSTs target, or None if any is not literal}.

    For each POST literal, the enclosing helper is the last named callable
    definition before it that we are still syntactically inside, or whose
    one-expression arrow body it is. Only those names count as a create;
    `list`/`update`/`remove` next to them do not.
    """
    blanked = _blank_literals(source)
    definitions = [
        (match.end(), match.group("fn") or match.group("name"))
        for match in _HELPER_DEF_RE.finditer(blanked)
        if (match.group("fn") or match.group("name")) not in _NOT_A_HELPER
    ]
    helpers: dict[str, list[str] | None] = {}
    for post in _POST_RE.finditer(source):
        for end, name in reversed(definitions):
            if end <= post.start() and (_bracket_depth(blanked, end, post.start()) >= 1
                                        or _ARROW_BODY_RE.fullmatch(blanked, end, post.start())):
                path, known = _post_path(source, post), helpers.get(name, [])
                helpers[name] = None if path is None or known is None else known + [path]
                break
    return helpers


def _definition_end(blanked: str, start: int) -> int:
    """Where a definition starting at ``start`` ends: after its parameters, then its body."""
    depth, closed = 0, 0
    for i in range(start, len(blanked)):
        if blanked[i] in "([{":
            depth += 1
        elif blanked[i] in ")]}":
            depth -= 1
            if depth < 0:
                return i
            if depth == 0:
                closed += 1
                if closed == 2:
                    return i + 1
    return len(blanked)


def _hook_post_paths(source: str) -> dict[str, list[str] | None]:
    """{React hook name: the literal paths of the POSTs inside its body, or None if any is not literal}.

    A page calling ``useBookings()`` creates through whatever the hook returns
    (``{ add }``), so the hook call is that page's create.
    """
    blanked = _blank_literals(source)
    posts = list(_POST_RE.finditer(source))
    hooks: dict[str, list[str] | None] = {}
    for match in _HELPER_DEF_RE.finditer(blanked):
        name = match.group("fn") or match.group("name")
        if not re.fullmatch(r"use[A-Z][\w$]*", name):
            continue
        end = _definition_end(blanked, match.end())
        for post in posts:
            if match.end() <= post.start() < end:
                path, known = _post_path(source, post), hooks.get(name, [])
                hooks[name] = None if path is None or known is None else known + [path]
    return hooks


def _post_helpers(source: str) -> set[str]:
    """Names in an api-client module whose own definition issues a POST."""
    return set(_post_helper_paths(source))


def _import_bindings(source: str):
    """Yield (local name, relative specifier) for each relative import."""
    for match in _IMPORT_RE.finditer(source):
        clause, spec = match.group("clause"), match.group("spec")
        namespace = _NAMESPACE_IMPORT_RE.search(clause)
        if namespace:
            yield namespace.group(1), spec
        named = _NAMED_IMPORT_RE.search(clause)
        if named:
            for item in named.group(1).split(","):
                item = item.strip()
                if item:
                    yield item.split(" as ")[-1].strip(), spec
        default = clause.split("{")[0].split("*")[0].strip().rstrip(",").strip()
        if default:
            leading = _LEADING_NAME_RE.match(default)
            if leading:
                yield leading.group(1), spec


def _resolve_import(rel: str, spec: str) -> list[str]:
    parts = rel.split("/")[:-1]
    for segment in spec.split("/"):
        if segment == "..":
            if parts:
                parts.pop()
        elif segment not in ("", "."):
            parts.append(segment)
    stem = "/".join(parts)
    candidates = [stem] if stem.endswith(_FRONTEND_EXTS) else []
    return (candidates + [stem + ext for ext in _FRONTEND_EXTS]
            + [f"{stem}/index{ext}" for ext in _FRONTEND_EXTS])


def _client_create_calls(rel: str, source: str, files: dict[str, str],
                         helper_cache: dict[str, dict], used_clients: set[str]) -> list:
    """Creates ``rel`` issues through an imported api-client module.

    One entry per call site: the helper's own literal path, else the call's
    literal string arguments, or None when the call names its resource
    through a variable (the generic ``<EntityList entity={...}/>`` page).
    None means "a create happens here but this call does not say which
    entity", which is exactly the evidence the file-mention test already
    stands on. Each module a call resolves into is added to ``used_clients``.

    ``helper_cache`` is keyed by module path: one api client is imported by
    every page, so it is scanned once rather than once per importer.
    """
    calls = []
    for binding, spec in _import_bindings(source):
        target = next((c for c in _resolve_import(rel, spec) if c in files), None)
        if target is None or not _POST_RE.search(files[target]):
            continue
        if target not in helper_cache:
            helper_cache[target] = {**_post_helper_paths(files[target]), **_hook_post_paths(files[target])}
        helpers = helper_cache[target]
        if not helpers:
            continue
        pattern = re.compile(
            r"\b" + re.escape(binding) + r"\s*(?:\.\s*(?P<member>[\w$]+)\s*)?\("
            r"(?P<args>[^()]{0,200})")
        for call in pattern.finditer(source):
            name = call.group("member") or binding
            if name in helpers:
                calls.append(helpers[name] or _STRING_ARG_RE.findall(call.group("args")) or None)
                used_clients.add(target)
    return calls


def _table_block_create_wired(content: str, cls: str) -> bool:
    """True if a <TableBlock> in ``content`` is a wired create form for ``cls``.

    dataBinding.entity must equal ``cls`` (case-insensitive) with a real
    endpoint, and options.formColumns must carry at least one entry. A
    file can embed TableBlocks for other entities (a lookup column's own
    ``entity`` key, or an unrelated table on the same page) — scoping to
    each tag's own dataBinding, rather than a whole-file search, is what
    keeps this attributed to the right entity.
    """
    for tag in _TABLE_BLOCK_RE.findall(content):
        db_match = _DATA_BINDING_RE.search(tag)
        if not db_match:
            continue
        entity_match = _ENTITY_RE.search(db_match.group(1))
        endpoint_match = _ENDPOINT_RE.search(db_match.group(1))
        if not entity_match or not endpoint_match:
            continue
        if entity_match.group(1).strip().lower() != cls.lower():
            continue
        if not endpoint_match.group(1).strip():
            continue
        if _NONEMPTY_FORM_COLUMNS_RE.search(tag):
            return True
    return False


def _entity_forms(name: str) -> list[str]:
    """Lowercase spellings a generated app plausibly uses for a class."""
    low = name.lower()
    forms = {low, f"{low}s"}
    if low.endswith("y"):
        forms.add(f"{low[:-1]}ies")
    if low.endswith("s"):
        forms.add(f"{low}es")
    return sorted(forms, key=len, reverse=True)


@functools.lru_cache(maxsize=1024)
def _forms_re(forms: tuple[str, ...], whole_word: bool) -> re.Pattern:
    # A form must end its word: "running" is not Run, "username" is not User.
    # A whole word also keeps `RunStatus` / `RUN_STATUS` from declaring Run.
    alternatives = "(?:" + "|".join(map(re.escape, forms)) + ")"
    if whole_word:
        return re.compile(r"(?<![A-Za-z0-9_])" + alternatives + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    return re.compile(alternatives + r"(?-i:(?![a-z0-9]))", re.IGNORECASE)


def _mentions(text: str, forms: list[str], whole_word: bool = False) -> bool:
    return _forms_re(tuple(forms), whole_word).search(text) is not None


def _config_declarations(source: str) -> str:
    """The part of an entity config module that declares entities.

    An entry's own key, path and label declare it. Comments, multi-word labels
    (`'Dataset Version'`) and every list nested inside an entry (`fields: [...]`,
    whose `ref`s name other entities) are blanked.
    """
    source = _BLANKABLE_RE.sub(
        lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group())
        if m.group().startswith("/") or re.search(r"\s", m.group()) else m.group(), source)
    blanked = _blank_literals(source)
    spans, depth, start, outer = [], 0, None, 0
    for i, char in enumerate(blanked):
        if char in "([{":
            if char == "[" and depth >= 2 and start is None:
                start, outer = i, depth
            depth += 1
        elif char in ")]}":
            depth -= 1
            if start is not None and depth == outer:
                spans.append((start, i + 1))
                start = None
    for s, e in reversed(spans):
        source = source[:s] + _KEEP_NEWLINES_RE.sub(" ", source[s:e]) + source[e:]
    return source


def _creates(args: list[str] | None, forms: list[str]) -> bool:
    """Whether a create call can be this entity's.

    An unnamed target (a variable) can be; a literal one must name the entity:
    `api.post('/stockcheck/', ...)` on a page that lists products in a dropdown
    is not Product's create. Separators are ignored, so `/order-lines/` names
    OrderLine.
    """
    return args is None or _mentions(re.sub(r"[-_\s]", "", " ".join(args)), forms)


def build_acceptance_matrix(
    output_dir: str,
    domain_model,
    endpoint_manifest: str = "",
    snapshot_dir: str = ".besser_snapshot",
) -> dict[str, dict[str, bool | None]] | None:
    """Compute {class name: {route, page, create}} for the workspace.

    ``endpoint_manifest`` is the statically parsed route listing (from
    ``prompt_builder.build_endpoint_manifest``); pass it in when the
    caller already built one, else it is rebuilt here.

    Returns None when there is no domain model / no classes.
    """
    if domain_model is None:
        return None
    try:
        classes = [c.name for c in domain_model.get_classes() if getattr(c, "name", None)]
    except Exception:
        return None
    if not classes:
        return None

    if not endpoint_manifest:
        try:
            from besser.spec_driven_agent.agent.prompt_builder import build_endpoint_manifest
            endpoint_manifest = build_endpoint_manifest(output_dir) or ""
        except Exception:
            endpoint_manifest = ""
    manifest_low = endpoint_manifest.lower()

    # Collect frontend files once: (rel path, content).
    frontend_files: list[tuple[str, str]] = []
    for root, dirs, files in walk_plain(output_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith(_FRONTEND_EXTS):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, output_dir).replace("\\", "/")
            if rel.startswith(snapshot_dir) or rel.startswith(".besser_"):
                continue
            try:
                if os.path.getsize(fpath) > 1_000_000:
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    frontend_files.append((rel, f.read()))
            except Exception:
                continue

    by_path = dict(frontend_files)
    helper_cache: dict[str, dict] = {}
    used_clients: set[str] = set()
    client_calls = {rel: _client_create_calls(rel, content, by_path, helper_cache, used_clients)
                    for rel, content in frontend_files}
    # A shared client's own POSTs count where its helpers are called, not as
    # creates of whatever its other helpers (`getMetrics`) happen to name.
    # A module other files call into is such a client even when none of those
    # calls is a create; a page imported to be rendered or routed is not.
    code = {rel: _blank_literals(content) for rel, content in frontend_files}
    clients = used_clients | {
        target for rel, content in frontend_files for binding, spec in _import_bindings(content)
        for target in [next((c for c in _resolve_import(rel, spec) if c in by_path), None)]
        if target in helper_cache
        and not re.search(r"</|/>", code[target])  # a closing tag: renders JSX, a component
        and re.search(r"(?<![\w$.])" + re.escape(binding) + r"\s*[.(]", code[rel])
        and not re.search(r"<\s*" + re.escape(binding) + r"\b", code[rel])
    }
    create_calls = {rel: client_calls[rel] + ([] if rel in clients else _direct_create_calls(content))
                    for rel, content in frontend_files}
    # Generic pages (``EntityList entity={key}``): the create names its resource
    # through a variable and the component's code names no class of the model
    # (its comments may: ``e.g. "/brand/"``). A file that mentions an entity and
    # renders one as JSX creates that entity through it; importing the api
    # client is not rendering a page. ``LoanList`` with a variable-argument
    # create is not generic: crediting it to every entity App.jsx lists would
    # hide their missing creates. Testing which entity it was given
    # (`config.key === 'booking'`) is what a shared form does, so it names none;
    # neither does reading `err.message` when the model has a Message.
    all_forms = [f for name in classes for f in _entity_forms(name)]
    generic_creators = {
        rel for rel, calls in create_calls.items()
        if any(args is None for args in calls)
        and not _mentions(rel.rsplit("/", 1)[-1], all_forms)
        and not _mentions(_PROPERTY_READ_RE.sub(" ", _COMPARED_STRING_RE.sub(" ", _blank_comments(by_path[rel]))),
                          all_forms)
    }
    # A generic creator may read its entities from a config module it imports
    # (`entities.js`); that module, not the creator's own code, declares them.
    config_text = {
        rel: " ".join(_config_declarations(by_path[target]) for target in (
            next((c for c in _resolve_import(rel, spec) if c in by_path), None)
            for _, spec in _import_bindings(by_path[rel]))
            if target and not _POST_RE.search(by_path[target]) and not re.search(r"\bfetch\s*\(", by_path[target]))
        for rel in generic_creators
    }
    # A configured creator is credited through its config alone: the page that
    # renders it mentioning `user` (the signed-in one) declares nothing.
    unconfigured = {rel for rel in generic_creators if not _mentions(config_text[rel], all_forms, whole_word=True)}
    renders_generic_creator = {
        rel: any(next((c for c in _resolve_import(rel, spec) if c in by_path), None) in unconfigured
                 and re.search(r"<\s*" + re.escape(binding) + r"\b", content)
                 for binding, spec in _import_bindings(content))
        for rel, content in frontend_files
    }

    matrix: dict[str, dict[str, bool | None]] = {}
    for cls in sorted(classes):
        forms = _entity_forms(cls)
        # None, not False, when the manifest is empty: build_endpoint_manifest
        # parses FastAPI decorators out of .py files only, so an Express,
        # Django, Spring or axum backend yields "" and every entity would read
        # "no backend REST route" for a route the app genuinely serves. An
        # unparsed stack is unmeasured, not empty.
        route = any(f in manifest_low for f in forms) if manifest_low else None
        page = False
        create = False
        for rel, content in frontend_files:
            about = (_mentions(rel.rsplit("/", 1)[-1], forms) or _mentions(content, forms)
                     or _mentions(config_text.get(rel, ""), forms, whole_word=True))
            if not about:
                continue
            page = True
            # A create that names another resource literally is that resource's
            # create, not this entity's: `api.create('order', ...)` on a page
            # that merely lists clerks in a dropdown proves nothing about Clerk.
            if (any(_creates(args, forms) for args in create_calls[rel])
                    or renders_generic_creator[rel]
                    or _table_block_create_wired(content, cls)):
                create = True
                break
        matrix[cls] = {"route": route, "page": page, "create": create}
    return matrix


def matrix_issues(matrix: dict[str, dict[str, bool | None]] | None) -> list[str]:
    """Render missing matrix cells as advisory issue strings."""
    if not matrix:
        return []
    issues: list[str] = []
    for cls, cells in matrix.items():
        # `is False` and not falsiness: None means the signal was never
        # collected, and "we did not look" is not a finding.
        missing = [k for k in ("route", "page", "create") if cells.get(k) is False]
        if not missing:
            continue
        detail = {
            "route": "no backend REST route",
            "page": "no frontend page/component references it",
            "create": "no frontend create path found (no POST call, no api-client create resolved to it, and no TableBlock bound to it with editable form fields)",
        }
        issues.append(
            "acceptance: entity "
            + cls
            + " — "
            + "; ".join(detail[m] for m in missing)
        )
    return issues
