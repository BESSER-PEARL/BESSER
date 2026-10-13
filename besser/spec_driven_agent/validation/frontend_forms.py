"""Static checks that an LLM-authored frontend's create forms can save a record.

The backend probes (``validate_app``, ``test_api``, the runbook) never render a
page, so a UI whose every create form is dead still scores as working. These
rules read the frontend source and the backend's ``*Create`` Pydantic schemas;
nothing is installed or executed. Each rule reports only shapes whose outcome
is decidable from the text; anything dynamic is skipped, never guessed.

- ``api url:`` a ``BASE + path`` fetch helper reached with a resource path that
  has no leading ``/`` (``http://localhost:8000expense/``).
- ``form create target:`` Create reuses the edit handler keyed on an ``'new'``
  placeholder whose truthiness picks update, so it sends ``PUT /x/new/``.
- ``form route:`` a New/Create link to a route the router never declares.
- ``form field type:`` a text input for a number/date/boolean schema field.
- ``form json textarea:`` the form is a raw JSON textarea.
"""

from __future__ import annotations

import ast
import os
import re

from besser.spec_driven_agent.validation.frontend_resolution import (
    _read, _strip_comments, _walk,
)

_SOURCE_EXT = (".js", ".jsx", ".ts", ".tsx")
_TYPED_KINDS = {"number", "date", "datetime", "bool"}
_INPUT_FOR_KIND = {"number": 'type="number"', "date": 'type="date"',
                   "datetime": 'type="datetime-local"', "bool": 'type="checkbox"'}
_KIND_OF = {"int": "number", "float": "number", "Decimal": "number",
            "conint": "number", "confloat": "number", "date": "date",
            "datetime": "datetime", "bool": "bool", "str": "text",
            "EmailStr": "text", "constr": "text"}


# ---------------------------------------------------------------- backend

def _annotation_kind(node: ast.AST | None) -> str:
    if isinstance(node, ast.Subscript):
        base = getattr(node.value, "id", getattr(node.value, "attr", ""))
        if base == "Optional":
            return _annotation_kind(node.slice)
        if base == "Union":
            parts = [p for p in getattr(node.slice, "elts", [])
                     if not (isinstance(p, ast.Constant) and p.value is None)]
            return _annotation_kind(parts[0]) if len(parts) == 1 else "other"
        if base in ("List", "list", "Set", "set", "Tuple", "tuple"):
            return "list"
        return "other"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        sides = [s for s in (node.left, node.right)
                 if not (isinstance(s, ast.Constant) and s.value is None)]
        return _annotation_kind(sides[0]) if len(sides) == 1 else "other"
    if isinstance(node, ast.Call):
        return _KIND_OF.get(getattr(node.func, "id", ""), "other")
    name = getattr(node, "id", getattr(node, "attr", ""))
    return _KIND_OF.get(name, "other")


def _create_schemas(workspace: str) -> dict[str, dict[str, str]]:
    """``{"Expense": {"amount": "number", ...}}`` from every ``XCreate`` model."""
    classes: dict[str, tuple[list[str], dict[str, str]]] = {}
    for folder, files in _walk(workspace):
        for name in files:
            if not name.endswith(".py"):
                continue
            text = _read(os.path.join(folder, name))
            if "Create" not in text or "BaseModel" not in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                fields = {stmt.target.id: _annotation_kind(stmt.annotation)
                          for stmt in node.body
                          if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)}
                bases = [getattr(b, "id", "") for b in node.bases]
                classes[node.name] = (bases, fields)

    def resolved(cls: str, seen: frozenset = frozenset()) -> dict[str, str]:
        bases, fields = classes[cls]
        merged: dict[str, str] = {}
        for base in bases:
            if base in classes and base not in seen:
                merged.update(resolved(base, seen | {cls}))
        merged.update(fields)
        return merged

    return {cls[:-len("Create")]: resolved(cls) for cls in classes
            if cls.endswith("Create") and len(cls) > len("Create")}


def _backend_resources(workspace: str, schemas: dict) -> set[str]:
    """First path segments the backend serves (``/expense/`` -> ``expense``)."""
    found = {entity.lower() for entity in schemas}
    route = re.compile(r"""(?:\.(?:get|post|put|patch|delete)\(|prefix\s*=)\s*['"]/([\w-]+)""")
    for folder, files in _walk(workspace):
        for name in files:
            if name.endswith(".py"):
                found.update(m.lower() for m in route.findall(_read(os.path.join(folder, name))))
    return found


# --------------------------------------------------------------- frontend

def _frontend_files(workspace: str) -> list[tuple[str, str]]:
    out = []
    for folder, files in _walk(workspace):
        for name in files:
            if name.endswith(_SOURCE_EXT) and not name.endswith(".d.ts"):
                path = os.path.join(folder, name)
                text = _read(path)
                if len(text) < 1_000_000:
                    rel = os.path.relpath(path, workspace).replace("\\", "/")
                    out.append((rel, _strip_comments(text)))
    return out


def _line(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _jsx_tags(text: str, tag: str):
    """Yield ``(start, attrs, top)`` for each ``<tag ...>``.

    ``top`` is the attribute text with every ``{...}`` emptied, so a ``type=``
    inside an expression is never mistaken for the element's own attribute.
    """
    for match in re.finditer(rf"<{tag}\b", text):
        depth, quote, i = 0, None, match.end()
        top: list[str] = []
        while i < len(text):
            char = text[i]
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'`":
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == ">" and depth == 0:
                break
            if depth == 0 and char != "}":
                top.append(char)
            i += 1
        yield match.start(), text[match.end():i], "".join(top)


# ------------------------------------------------------------------ rules

_FETCH_CONCAT = re.compile(
    r"""\bfetch\(\s*(?:`\$\{\s*(\w+)\s*\}\$\{\s*(\w+)\s*\}`|(\w+)\s*\+\s*(\w+)\s*[,)])""")


def _api_url_issues(files, resources) -> list[str]:
    """T3b: ``fetch(BASE + path)`` reached with ``path`` lacking a leading slash."""
    helpers: list[tuple[str, str]] = []  # (file, base name)
    for rel, text in files:
        for match in _FETCH_CONCAT.finditer(text):
            base = match.group(1) or match.group(3)
            value = re.search(rf"\b{base}\s*=\s*\(?([^;\n]*)", text)
            # Only a base with no trailing slash: an origin or an env override.
            if value and re.search(r"""['"]https?://[^'"/]+(:\d+)?['"]|import\.meta\.env""",
                                   value.group(1)):
                helpers.append((rel, base))
    if not helpers:
        return []
    # Property values naming a backend resource without the "/", per file.
    literal = re.compile(r"""\b(path|endpoint|resource|url|apiPath)\s*:\s*['"]([\w-]+)/?['"]""")
    blob = "\n".join(text for _, text in files)
    found: dict[str, tuple[int, str, list[str]]] = {}
    for rel, text in files:
        for match in literal.finditer(text):
            key, value = match.groups()
            if value.lower() not in resources:
                continue
            # Evidence the value starts a request URL as-is: api(`${path}/`)
            # or api(path + '/'); a router call would be navigate(...).
            if not re.search(rf"""(?<!navigate)\(\s*(?:`\$\{{[\w.]*\b{key}\s*\}}|[\w.]*\b{key}\s*\+)""", blob):
                continue
            entry = found.setdefault(rel, (_line(text, match.start()), key, []))
            if value not in entry[2]:
                entry[2].append(value)
    helper_file, base = helpers[0]
    return [
        f"api url: {rel} line {line}: resource paths {', '.join(repr(v) for v in values)} "
        f"({key}: ...) have no leading '/', and {helper_file} builds request URLs as "
        f"{base} + path, so they are requested at e.g. http://localhost:8000{values[0]}/ "
        f"and every list/create call fails. Write them as '/{values[0]}' (or join "
        f"base and path with a '/')."
        for rel, (line, key, values) in found.items()]


def _create_target_issues(files) -> list[str]:
    """T2a: a ``'new'`` placeholder whose truthiness selects the update call."""
    issues = []
    for rel, text in files:
        for var in set(re.findall(r"""\b(\w+)\s*===?\s*['"]new['"]""", text)) | set(
                re.findall(r"""\bset(\w+)\(\s*['"]new['"]\s*\)""", text)):
            names = {var, var[:1].lower() + var[1:]}
            for name in names:
                branch = re.search(
                    rf"""(?:if\s*\(\s*{name}\s*\)\s*(?:await\s+)?[\w.]*update"""
                    rf"""|\b{name}\s*\?\s*(?:await\s+)?[\w.]*update)""", text)
                if branch:
                    issues.append(
                        f"form create target: {rel} line {_line(text, branch.start())}: "
                        f"'{name}' holds the placeholder 'new' while creating, and the "
                        f"save handler picks update whenever '{name}' is truthy - so "
                        f"Create sends PUT /<entity>/new/ instead of POST /<entity>/ and "
                        f"no record is ever created. Branch on {name} === 'new' (or a "
                        f"null id) to call create.")
                    break
    return issues


_DYNAMIC = "\0"


def _attr(attrs: str, name: str) -> str | None:
    """The raw value of JSX attribute ``name``: a quoted string or a ``{...}`` body."""
    match = re.search(rf"""(?<![\w.-]){name}\s*=\s*(?:"([^"]*)"|'([^']*)'|\{{)""", attrs)
    if not match:
        return None
    if match.group(1) is not None or match.group(2) is not None:
        return "'" + (match.group(1) if match.group(1) is not None else match.group(2)) + "'"
    depth, i = 1, match.end()
    while i < len(attrs) and depth:
        depth += {"{": 1, "}": -1}.get(attrs[i], 0)
        i += 1
    return attrs[match.end():i - 1]


def _path_text(expr: str) -> str | None:
    """A route/link expression as text, each dynamic part replaced by ``_DYNAMIC``.

    ``'/' + k`` and a template ``/${k}/new`` become ``/<dyn>`` and
    ``/<dyn>/new``; anything not starting with a literal ``/`` is unknown.
    """
    out = []
    for _, quoted, template, code in re.findall(
            r"""(['"])(.*?)\1|`([^`]*)`|([\w.$()\[\]]+)""", expr):
        if code:
            out.append(_DYNAMIC)
        else:
            out.append(re.sub(r"\$\{[^}]*\}", _DYNAMIC, quoted or template))
    text = "".join(out)
    return text if text.startswith("/") else None


def _route_patterns(files) -> list[re.Pattern] | None:
    """Declared ``<Route>`` paths as regexes; ``None`` when any is unknown."""
    patterns = []
    for _, text in files:
        for _, attrs, _ in _jsx_tags(text, "Route"):
            expr = _attr(attrs, "path")
            if expr is None:
                continue  # index / layout route
            path = _path_text(expr)
            if path is None:
                return None
            if path == "/*":
                continue  # a catch-all is the not-found page, not the form
            body = re.sub(r":\w+", _DYNAMIC, path.rstrip("/") or "/")
            body = re.escape(body).replace(re.escape(_DYNAMIC), "[^/]+")
            patterns.append(re.compile(body.replace(r"/\*", "(/.*)?") + "/?$"))
    return patterns


def _route_issues(files) -> list[str]:
    """T2b: a New/Create link whose target matches no declared ``<Route>``."""
    patterns = _route_patterns(files)
    if not patterns:
        return []
    issues = []
    for rel, text in files:
        targets = []
        for tag in ("Link", "NavLink"):
            for start, attrs, _ in _jsx_tags(text, tag):
                expr = _attr(attrs, "to")
                label = re.match(r"\s*([^<{]{0,60})", text[start + len(tag) + 1 + len(attrs) + 1:])
                targets.append((start, expr, label.group(1).strip() if label else ""))
        targets += [(m.start(), m.group(1), "")
                    for m in re.finditer(r"""\bnavigate\(\s*((['"`])[^'"`]*\2)""", text)]
        for start, expr, label in targets:
            target = _path_text(expr or "")
            if not target:
                continue
            target = target.split("?")[0].split("#")[0]
            if not (re.search(r"/(new|create|add)/?$", target)
                    or re.match(r"[+＋]?\s*(new|create|add)\b", label, re.I)):
                continue
            if any(p.match(target.replace(_DYNAMIC, "x")) for p in patterns):
                continue
            shown = target.replace(_DYNAMIC, "${...}")
            issues.append(
                f"form route: {rel} line {_line(text, start)}: the "
                f"'{label or 'create'}' link goes to {shown}, but no <Route> declares "
                f"that path, so the create form is unreachable. Declare the route or "
                f"point the link at the page that renders the form.")
    return issues


def _json_textarea_issues(files) -> list[str]:
    """T2c: a textarea whose text is ``JSON.parse``d as the record body."""
    issues = []
    for rel, text in files:
        for start, attrs, _ in _jsx_tags(text, "textarea"):
            bound = re.search(r"\bvalue\s*=\s*\{\s*(\w+)\s*\}", attrs)
            if bound and re.search(rf"JSON\.parse\(\s*{bound.group(1)}\s*\)", text):
                issues.append(
                    f"form json textarea: {rel} line {_line(text, start)}: the form is a "
                    f"raw JSON textarea ('{bound.group(1)}' is JSON.parse'd on submit), so "
                    f"saving needs hand-written JSON matching the API schema. Render one "
                    f"typed input per field instead.")
    return issues


def _field_lists(text: str) -> list[list[str]]:
    """String arrays that read as field lists: ``['a','b']`` or ``[['a','A'],...]``."""
    out = []
    for match in re.finditer(r"\[((?:\s*['\"]\w+['\"]\s*,?){2,})\]", text):
        out.append(re.findall(r"['\"](\w+)['\"]", match.group(1)))
    for match in re.finditer(r"\[((?:\s*\[\s*['\"]\w+['\"][^\[\]]*\]\s*,?){2,})\]", text):
        out.append(re.findall(r"\[\s*['\"](\w+)['\"]", match.group(1)))
    return out


def _field_type_issues(files, schemas) -> list[str]:
    """T1: a text input for a field whose create schema type is number/date/bool.

    Two unambiguous shapes only: an ``<input>`` with no ``type`` at all that
    renders a field list by dynamic key (``value={form[f]}``), where the list
    belongs to exactly one create schema; and an ``<input>`` with no type or
    ``type="text"`` bound to one named field (``value={form.amount}``) whose
    type is the same in every schema that declares it.
    """
    issues = []
    for rel, text in files:
        generic = []
        for start, attrs, top in _jsx_tags(text, "input"):
            if re.search(r"\btype\s*=", top) and not re.search(r"""\btype\s*=\s*['"]text['"]""", top):
                continue
            value = re.search(r"\bvalue\s*=\s*\{([^}]*)\}", attrs)
            if not value:
                continue
            named = re.match(r"\s*\w+(?:\.(\w+)|\[\s*['\"](\w+)['\"]\s*\])\s*(?:\?\?|\|\||$)", value.group(1))
            if named:
                field = named.group(1) or named.group(2)
                kinds = {fields[field] for fields in schemas.values() if field in fields}
                if len(kinds) == 1 and (kind := kinds.pop()) in _TYPED_KINDS:
                    issues.append(
                        f"form field type: {rel} line {_line(text, start)}: the input for "
                        f"'{field}' is a text box, but the create schema types it as "
                        f"{kind}; text that is not a valid {kind} is rejected with 422 and the form "
                        f"gives no hint of the format. Use "
                        f"{_INPUT_FOR_KIND[kind]} and send a {kind} value.")
            elif not re.search(r"\btype\s*=", top) and re.match(r"\s*\w+\[\s*\w+\s*\]", value.group(1)):
                generic.append(start)
        if not generic:
            continue
        reported: set[str] = set()
        for names in _field_lists(text):
            owners = [entity for entity, fields in schemas.items()
                      if all(n in fields for n in names)]
            if len(owners) != 1:
                continue
            entity = owners[0]
            typed = [f"{n} ({schemas[entity][n]})" for n in names
                     if schemas[entity][n] in _TYPED_KINDS]
            if typed and entity not in reported:
                reported.add(entity)
                issues.append(
                    f"form field type: {rel} line {_line(text, generic[0])}: the {entity} "
                    f"form renders every field as an <input> with no type, but the create "
                    f"schema types {', '.join(typed)}; text in those boxes that is not a valid "
                    f"value of that type is rejected with 422. Give each field its input type "
                    f"(number/date/datetime-local/checkbox) and convert the value before "
                    f"sending.")
    return issues


def collect_frontend_form_issues(output_dir: str) -> list[str]:
    """Findings that a generated UI cannot save a record through its forms."""
    workspace = os.path.realpath(output_dir)
    files = _frontend_files(workspace)
    if not any(re.search(r"<\w", text) for _, text in files):
        return []
    schemas = _create_schemas(workspace)
    issues = []
    issues += _api_url_issues(files, _backend_resources(workspace, schemas))
    issues += _create_target_issues(files)
    issues += _route_issues(files)
    issues += _json_textarea_issues(files)
    if schemas:
        issues += _field_type_issues(files, schemas)
    return issues


if __name__ == "__main__":  # manual probe: python frontend_forms.py <dir>
    import sys
    for finding in collect_frontend_form_issues(sys.argv[1]):
        print(finding)
