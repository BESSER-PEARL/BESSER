"""Model-derived acceptance matrix for generated apps.

For every class in the domain model, three static discovery signals
indicate whether the entity appears in the app (not executed acceptance):

* ``route``  — the backend exposes REST routes for it (checked against
  the same static route parse the endpoint manifest uses). That parse
  reads FastAPI decorators in ``.py`` files only, so on any other stack
  the cell is ``None`` (unmeasured) rather than ``False``;
* ``page``   — some frontend file is about it (name or content match);
* ``create`` — a frontend file about it contains a POST, OR calls a
  shared api-client helper that issues one, OR a ``<TableBlock>`` bound
  to it: ``dataBinding`` naming this entity and an endpoint, plus a
  non-empty ``options.formColumns``. The GUI-model scaffold's only
  generated create path issues its POST from a shared runtime component
  (``TableComponent``), never a literal call in the page file itself, so
  the binding — not the call site — is the signal.

Most generated React frontends put every request behind one api module
(``api.create('carpark', payload)`` → ``fetch(url, {method:'POST'})``),
so the literal POST never appears in the file that names the entity. A
``.post(`` scan alone reported "no create path" for 302 entities across
apps a live probe had just driven end to end. ``_client_create_calls``
resolves those calls instead of demanding a literal in one file.

The matrix is deliberately REPORT-ONLY (warnings + a recipe field, never
blockers): a GUI-model-driven run may legitimately scope the UI to a
subset of entities, so "entity X has no page" is a visibility signal for
the checklist and the recipe, not a defect the fix loop must burn turns
on. Blocker-level enforcement stays with ``contract_checks``.
"""

from __future__ import annotations

import os
import re

_FRONTEND_EXTS = (".js", ".jsx", ".ts", ".tsx")
_SKIP_DIRS = ("node_modules", "dist", "build", "__pycache__")

# A POST issued from frontend code — axios/api `.post(`, fetch with
# method POST, or a generated api-layer helper.
_POST_RE = re.compile(r"\.post\s*\(|method\s*:\s*['\"]POST['\"]", re.IGNORECASE)

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


def _blank_literals(text: str) -> str:
    """Blank comments and string bodies, keeping every offset in place.

    Offsets and newlines are preserved so `_HELPER_DEF_RE` positions still
    index into the original source. A per-character rewrite allocated a list
    the size of the file on every call and exhausted memory on the sweep;
    substituting only the matched spans keeps it proportional to the literals.
    """
    return _BLANKABLE_RE.sub(lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group()), text)


def _bracket_depth(blanked: str, start: int, end: int) -> int:
    depth = 0
    for char in blanked[start:end]:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth


def _post_helpers(source: str) -> set[str]:
    """Names in an api-client module whose own definition issues a POST.

    For each POST literal, the enclosing helper is the last named callable
    definition before it that we are still syntactically inside. Only those
    names count as a create; `list`/`update`/`remove` next to them do not.
    """
    blanked = _blank_literals(source)
    definitions = [
        (match.end(), match.group("fn") or match.group("name"))
        for match in _HELPER_DEF_RE.finditer(blanked)
        if (match.group("fn") or match.group("name")) not in _NOT_A_HELPER
    ]
    names = set()
    for post in _POST_RE.finditer(source):
        for end, name in reversed(definitions):
            if end <= post.start() and _bracket_depth(blanked, end, post.start()) >= 1:
                names.add(name)
                break
    return names


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
                         helper_cache: dict[str, set[str]]) -> list:
    """Creates ``rel`` issues through an imported api-client module.

    One entry per call site: its literal string arguments, or None when the
    call names its resource through a variable (the generic
    ``<EntityList entity={...}/>`` page). None means "a create happens here
    but this call does not say which entity", which is exactly the evidence
    the file-mention test already stands on.

    ``helper_cache`` is keyed by module path: one api client is imported by
    every page, and scanning it once per importer is what made this sweep
    run out of memory.
    """
    calls = []
    for binding, spec in _import_bindings(source):
        target = next((c for c in _resolve_import(rel, spec) if c in files), None)
        if target is None or not _POST_RE.search(files[target]):
            continue
        if target not in helper_cache:
            helper_cache[target] = _post_helpers(files[target])
        helpers = helper_cache[target]
        if not helpers:
            continue
        pattern = re.compile(
            r"\b" + re.escape(binding) + r"\s*(?:\.\s*(?P<member>[\w$]+)\s*)?\("
            r"(?P<args>[^()]{0,200})")
        for call in pattern.finditer(source):
            if (call.group("member") or binding) in helpers:
                calls.append(_STRING_ARG_RE.findall(call.group("args")) or None)
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


def _mentions(text: str, forms: list[str]) -> bool:
    low = text.lower()
    return any(f in low for f in forms)


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
    for root, dirs, files in os.walk(output_dir):
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
    helper_cache: dict[str, set[str]] = {}
    client_calls = {rel: _client_create_calls(rel, content, by_path, helper_cache)
                    for rel, content in frontend_files}

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
            about = _mentions(rel.rsplit("/", 1)[-1], forms) or _mentions(content, forms)
            if not about:
                continue
            page = True
            # A resolved create that names another resource literally is that
            # resource's create, not this entity's: `api.create('order', ...)`
            # on a page that merely lists clerks in a dropdown proves nothing
            # about Clerk.
            resolved = any(args is None or _mentions(" ".join(args), forms)
                           for args in client_calls.get(rel, ()))
            if (_POST_RE.search(content) or resolved
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
