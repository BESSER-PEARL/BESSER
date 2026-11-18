"""Frontend API calls checked against what the backend route reads and returns.

The contract is read from the route handlers, statically (nothing is imported):
the JSON body FastAPI reads - a Pydantic model's fields for one model parameter,
otherwise one top-level key per body parameter, so ``params: dict =
Body(embed=True)`` reads ``{"params": {...}}`` - and, when every ``return``
sends a dict literal, the object's top-level keys (``route_returns``). The app's
OpenAPI states the same body, and no response at all for a route declared with
``response_model=None``, which every generated route is.

Precision over recall, as in ``contract_checks``. A call is checked only when
its literal or template URL resolves to exactly one route of the endpoint
manifest (``endpoint_coherence`` reads the calls), a body only when it is an
object literal with no spread or computed key, and a response only when every
use of its data in the handler is an array operation or a read of a key the
object does not have: data passed on, returned, or read by one of its real keys
is left alone.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re

from besser.spec_driven_agent.agent.prompt_builder import build_endpoint_manifest
from besser.spec_driven_agent.validation.endpoint_coherence import (
    _ROUTE_VALUE_RE,
    _calls_in_file,
    _parse_manifest,
)
from besser.spec_driven_agent.validation.frontend_schema import (
    _FRONTEND_SUFFIXES,
    _JS_NONCODE,
    _model_resolver,
    _name,
    _sources,
)
from besser.spec_driven_agent.validation.route_returns import response_shape, route_handlers

PREFIX = "api call: "
_MAX_FINDINGS = 10

# Parameter markers FastAPI never reads from the JSON body; those that make it a form
# or let the handler read the raw body itself; and injected parameters.
_NOT_BODY = frozenset({"Depends", "Security", "Query", "Path", "Header", "Cookie"})
_OPEN_BODY = frozenset({"File", "Form", "UploadFile", "Request"})
_INJECTED = frozenset({"Response", "BackgroundTasks", "WebSocket", "HTTPConnection",
                       "Session", "AsyncSession"})
_SCALARS = frozenset({"str", "int", "float", "bool", "date", "datetime", "time", "UUID", "Decimal"})

_AWAITED = re.compile(
    r"(?:const|let|var)\s+(?:(?P<response>[A-Za-z_$][\w$]*)|\{\s*data\s*(?::\s*(?P<data>[A-Za-z_$][\w$]*))?\s*\})"
    r"\s*(?::[^=;]*)?=\s*await\s*$")
_THEN = re.compile(
    r"\s*\.\s*then\s*\(\s*(?:async\s*)?(?:\(\s*(?:(?P<p1>[A-Za-z_$][\w$]*)\s*(?::[^()]*)?|(?P<data>\{\s*data\s*\}))\s*\)"
    r"|(?P<p2>[A-Za-z_$][\w$]*))\s*=>")
_ARRAY_USE = re.compile(
    r"\s*(?:(?:\?\.|\.)\s*(?:(?P<prop>length)(?![\w$])(?!\s*\()"
    r"|(?P<method>map|filter|forEach|reduce|find|findIndex|some|every|flatMap|slice|sort|includes|indexOf|join)"
    r"\s*\()|(?:\?\.)?\[\s*\d+\s*\])")
_MEMBER = re.compile(r"\s*(?:\?\.|\.)\s*([A-Za-z_$][\w$]*)")
_STRING_MEMBER = re.compile(r"\s*(?:\?\.)?\[\s*(['\"])([\w$]+)\1\s*\]")
_IS_ARRAY = re.compile(r"Array\s*\.\s*isArray\s*\(\s*$")
_CLOSE = re.compile(r"\s*\)")
_KEY = re.compile(r"\s*(?:([A-Za-z_$][\w$]*)|(['\"])([^'\"\n]*)\2)\s*(?::|$)")
_STRINGIFY = re.compile(r"\bbody\s*:\s*JSON\s*\.\s*stringify\s*\(\s*\{")
_AXIOS_IMPORT = re.compile(r"""\bimport\s+axios\b[^;\n]*\bfrom\s*['"]axios['"]|\brequire\s*\(\s*['"]axios['"]\s*\)""")


@dataclass(frozen=True)
class _Body:
    keys: frozenset
    required: frozenset
    schema: str  # the model's name, or "" for one key per body parameter


@dataclass(frozen=True)
class _Route:
    file: str
    line: int
    body: _Body | None
    returns: frozenset | None  # top-level keys of an object response


def _is_scalar(annotation) -> bool:
    """A query-parameter type: a scalar, alone or in ``Optional[...]`` / ``X | None``."""
    if isinstance(annotation, ast.Subscript) and _name(annotation.value) in {"Optional", "Union"}:
        parts = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
    elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        parts = [annotation.left, annotation.right]
    else:
        return _name(annotation) in _SCALARS
    return all(_is_scalar(part) or (isinstance(part, ast.Constant) and part.value is None) for part in parts)


def _body(function, path: str, imports: dict, resolve) -> _Body | None:
    """The JSON body FastAPI reads for this handler; None when it reads none or that is not certain."""
    path_params = set(re.findall(r"\{(\w+)", path))
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    params = []
    for arg, default in [*zip(positional, defaults), *zip(arguments.kwonlyargs, arguments.kw_defaults)]:
        annotation, marker = arg.annotation, default
        if (isinstance(annotation, ast.Subscript) and _name(annotation.value) == "Annotated"
                and isinstance(annotation.slice, ast.Tuple)):
            annotation, *metadata = annotation.slice.elts
            marker = next((item for item in metadata if isinstance(item, ast.Call)), marker)
        kind = _name(marker.func) if isinstance(marker, ast.Call) else ""
        type_name = (annotation.value if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)
                     else _name(annotation))
        type_name = imports.get(type_name, type_name)
        if kind in _NOT_BODY or type_name in _INJECTED or (arg.arg in path_params and kind != "Body"):
            continue
        if kind in _OPEN_BODY or type_name in _OPEN_BODY:
            return None
        model = resolve(type_name) if type_name else None
        if kind == "Body":
            keywords = {keyword.arg: keyword.value for keyword in marker.keywords}
            alias = keywords.get("alias")
            embed = keywords.get("embed")
            required = not ({"default", "default_factory"} & keywords.keys()) and (
                not marker.args or (isinstance(marker.args[0], ast.Constant) and marker.args[0].value is Ellipsis))
            name = alias.value if isinstance(alias, ast.Constant) and isinstance(alias.value, str) else arg.arg
            params.append((name, isinstance(embed, ast.Constant) and embed.value is True, required, model, type_name))
        elif model is not None:
            params.append((arg.arg, False, default is None, model, type_name))
        elif not _is_scalar(annotation):
            return None
    if not params:
        return None
    if len(params) == 1 and not params[0][1]:
        _, _, _, model, type_name = params[0]
        if model is None or model.allows_extra:
            return None
        return _Body(model.fields, model.required, type_name)
    return _Body(frozenset(p[0] for p in params), frozenset(p[0] for p in params if p[2]), "")


def _routes(sources: list) -> dict:
    """``(METHOD, path) -> _Route`` for the routes exactly one handler serves."""
    modules, resolve = _model_resolver(sources)
    found: dict = {}
    for rel, (tree, imports) in modules.items():
        for method, path, function, decorator in route_handlers(tree):
            path = re.sub(r"/{2,}", "/", path)
            shape = response_shape(function, decorator)
            found.setdefault((method, path), []).append(_Route(
                rel, function.lineno, _body(function, path, imports, resolve),
                shape[1] if shape and shape[0] == "object" else None))
    return {key: routes[0] for key, routes in found.items() if len(routes) == 1}


def _resolve(path: str, method: str, manifest: dict) -> str | None:
    """The one manifest route a call names. Stricter than ``_paths_match``: an
    interpolated segment must meet a route parameter and a literal one the same
    literal, so ``/order/${id}/`` is not also ``/order/count/``."""
    parts = path.split("/")
    matches = [route for route, methods in manifest.items() if method in methods
               and len(route.split("/")) == len(parts)
               and all((part == "{param}") == bool(_ROUTE_VALUE_RE.fullmatch(segment))
                       and (part == "{param}" or part == segment)
                       for part, segment in zip(parts, route.split("/")))]
    return matches[0] if len(matches) == 1 else None


def _blank(text: str) -> str:
    """Comments and string literals as spaces, every offset and newline kept; the
    ``${...}`` of a template literal is code and stays."""
    def blank(match):
        literal = match.group()
        kept = list(re.sub(r"[^\n]", " ", literal))
        index = literal.find("${") if literal.startswith("`") else -1
        while index >= 0:
            depth, end = 1, index + 2
            while end < len(literal) and depth:
                depth += {"{": 1, "}": -1}.get(literal[end], 0)
                end += 1
            kept[index + 2:end - 1] = literal[index + 2:end - 1]
            index = literal.find("${", end)
        return "".join(kept)
    return _JS_NONCODE.sub(blank, text)


def _group_end(blanked: str, start: int) -> int | None:
    """The index just past the bracket group that opens at ``start``."""
    depth = 0
    for index in range(start, len(blanked)):
        char = blanked[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _enclosing_block(blanked: str, at: int) -> int | None:
    """The ``{`` of the innermost bracket still open at ``at``, if it is a brace."""
    stack = []
    for index in range(at):
        if blanked[index] in "([{":
            stack.append(index)
        elif blanked[index] in ")]}" and stack:
            stack.pop()
    return stack[-1] if stack and blanked[stack[-1]] == "{" else None


def _skip_space(blanked: str, index: int) -> int:
    while index < len(blanked) and blanked[index].isspace():
        index += 1
    return index


def _object_keys(text: str, blanked: str, start: int, end: int) -> list | None:
    """The top-level keys of the object literal ``text[start:end]``; None with a
    spread, a computed or method key, or anything else that is not ``key: value``."""
    keys, depth, piece = [], 0, start + 1
    for index in range(start + 1, end):
        char = blanked[index]
        if index == end - 1 or (char == "," and depth == 0):
            entry = text[piece:index]
            if entry.strip():
                match = _KEY.match(entry)
                if match is None:
                    return None
                keys.append(match.group(1) or match.group(3))
            piece = index + 1
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return keys


def _sent_object(blanked: str, call) -> tuple[int, int] | None:
    """Where the JSON body object literal of a call is: ``axios.post(url, {...})``
    or ``fetch(url, {..., body: JSON.stringify({...})})``."""
    index = _skip_space(blanked, call.url_end)
    if not blanked.startswith(",", index):
        return None
    index = _skip_space(blanked, index + 1)
    if not blanked.startswith("{", index):
        return None
    end = _group_end(blanked, index)
    if end is None:
        return None
    if call.client == "axios":
        return index, end
    for match in _STRINGIFY.finditer(blanked, index, end):
        inner = blanked[index + 1:match.start()]
        if sum(inner.count(c) for c in "([{") == sum(inner.count(c) for c in ")]}"):
            literal_end = _group_end(blanked, match.end() - 1)
            return (match.end() - 1, literal_end) if literal_end else None
    return None


def _body_message(call, route_path: str, route: _Route, sent: list) -> str | None:
    body = route.body
    unknown = [key for key in sent if key not in body.keys]
    missing = sorted(body.required - set(sent))
    if not (sent and len(unknown) == len(sent)) and not missing:
        return None
    accepted = ", ".join(sorted(body.keys))
    if body.schema:
        shape = f"{body.schema}, whose fields are {{{accepted}}} (handler: {route.file} line {route.line})"
    else:
        shape = (f"one top-level key per Body() parameter, {{{accepted}}} "
                 f"(handler: {route.file} line {route.line})")
    problems = []
    if sent and len(unknown) == len(sent):
        problems.append(f"none of the keys it sends ({', '.join(sent)}) is one of them, so they never reach "
                        "the handler")
    if missing:
        problems.append(f"it leaves out the required {', '.join(missing)}, so every call is refused with 422")
    wrapper = next(iter(body.keys)) if not body.schema and len(body.keys) == 1 else None
    advice = (f' Send {{"{wrapper}": {{...}}}} with the values inside.' if wrapper
              else " Send exactly those keys.")
    return (f"{call.file} line {call.line}: {call.method} {route_path} reads its JSON body as {shape}; "
            + "; ".join(problems) + "." + advice)


def _response_data(blanked: str, call):
    """``(pattern, expression, guarded name, scope start, scope end)`` of the value a
    call's response data is read from, or None when the code does not bind it plainly."""
    open_paren = blanked.find("(", call.start)
    close = _group_end(blanked, open_paren) if open_paren >= 0 else None
    if close is None:
        return None
    head = _AWAITED.search(blanked, max(0, call.start - 300), call.start)
    if head and blanked[_skip_space(blanked, close):][:1] in {".", "?", "["}:
        return None  # ``await axios.get(u).then(...)`` binds what the chain returns
    if head:
        block = _enclosing_block(blanked, head.start())
        if block is None:
            return None
        start, end = close, _group_end(blanked, block) or len(blanked)
        response, data = head.group("response"), head.group("data")
        if response is None:
            data = data or "data"
    else:
        then = _THEN.match(blanked, close)
        if then is None:
            return None
        start, end = then.end(), (_group_end(blanked, blanked.index("(", then.start())) or len(blanked)) - 1
        response, data = then.group("p1") or then.group("p2"), "data" if then.group("data") else None
    if call.client == "axios":
        if response is not None:
            return (rf"(?<![\w$.]){re.escape(response)}\s*(?:\?\.|\.)\s*data(?![\w$])",
                    f"{response}.data", response, start, end)
        return rf"(?<![\w$.]){re.escape(data)}(?![\w$])", data, None, start, end
    if call.client != "fetch" or response is None or not head:
        return None
    declared = re.compile(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=;]*)?=\s*await\s+"
                          + re.escape(response) + r"\s*\.\s*json\s*\(\s*\)").search(blanked, start, end)
    if declared is None:
        return None
    name = declared.group(1)
    return rf"(?<![\w$.]){re.escape(name)}(?![\w$])", name, None, declared.end(), end


def _data_uses(text: str, blanked: str, pattern: str, guarded: str | None, start: int, end: int) -> list | None:
    """Each use of the data in ``start:end``: ``("array", how)``, ``("key", name)`` or
    ``("other", "")``; None when the response variable is bound again or used whole."""
    if guarded and re.search(rf"(?<![\w$.]){re.escape(guarded)}(?![\w$])(?!\s*(?:\?\.|\.))",
                             blanked[start:end]):
        return None
    uses = []
    for match in re.compile(pattern).finditer(blanked, start, end):
        after = match.end()
        array = _ARRAY_USE.match(blanked, after)
        if array:
            uses.append(("array", "." + array.group("prop") if array.group("prop")
                         else f".{array.group('method')}()" if array.group("method") else "[index]"))
        elif _IS_ARRAY.search(blanked, max(0, match.start() - 40), match.start()) and _CLOSE.match(blanked, after):
            uses.append(("array", "Array.isArray"))
        elif member := (_MEMBER.match(blanked, after) or _STRING_MEMBER.match(text, after)):
            uses.append(("key", member.group(member.lastindex)))
        else:
            uses.append(("other", ""))
    return uses


def _response_message(call, route_path: str, route: _Route, expression: str, uses: list) -> str | None:
    keys = route.returns
    if call.client == "fetch":
        # ``await response.json()`` is the error body too when the status is not 2xx.
        uses = [(kind, value) for kind, value in uses if kind != "key" or value in keys]
    if not uses or any(kind == "other" or (kind == "key" and value in keys) for kind, value in uses):
        return None
    listed = ", ".join(sorted(keys))
    array = list(dict.fromkeys(value for kind, value in uses if kind == "array"))
    absent = list(dict.fromkeys(value for kind, value in uses if kind == "key"))
    wrong = []
    if array:
        wrong.append(f"uses {expression} as an array ({', '.join(array)})")
    if absent:
        wrong.append("reads " + ", ".join(f"{expression}.{key}" for key in absent))
    return (f"{call.file} line {call.line}: {call.method} {route_path} returns an object with the keys "
            f"{{{listed}}} (handler: {route.file} line {route.line}), but this code "
            + " and ".join(wrong) + " and never reads one of those keys, so it never gets the data. "
            f"Read the value you need from its key, e.g. {expression}.<key>.")


def collect_api_call_diagnostics(workspace, changed_path=None, content=None) -> list[dict]:
    """Frontend calls whose body or response use contradicts the route, for tool feedback.

    ``changed_path``/``content`` optionally overlay the current write.
    """
    sources = list(_sources(workspace, changed_path, content))
    frontend = [(path, text) for path, text in sources if Path(path).suffix in _FRONTEND_SUFFIXES]
    if not frontend:
        return []
    manifest = _parse_manifest(build_endpoint_manifest(str(workspace)))
    routes = _routes(sources) if manifest else {}
    if not routes:
        return []
    # An interceptor can hand the code ``response.data`` as the response itself.
    axios_reads = not any("interceptors.response.use" in text for _, text in frontend)
    findings, seen = [], set()
    for path, text in frontend:
        calls = _calls_in_file(path, text)
        if not calls:
            continue
        blanked = _blank(text)
        imports_axios = bool(_AXIOS_IMPORT.search(text))
        for call in calls:
            route_path = _resolve(call.path, call.method, manifest)
            route = routes.get((call.method, route_path)) if route_path else None
            if route is None or not call.client:
                continue
            messages = []
            if route.body is not None and call.method in {"POST", "PUT", "PATCH"}:
                literal = _sent_object(blanked, call)
                sent = _object_keys(text, blanked, *literal) if literal else None
                if sent is not None:
                    messages.append(_body_message(call, route_path, route, sent))
            if route.returns is not None and (call.client == "fetch" or (axios_reads and imports_axios)):
                data = _response_data(blanked, call)
                if data is not None:
                    pattern, expression, guarded, start, end = data
                    uses = _data_uses(text, blanked, pattern, guarded, start, end)
                    if uses is not None:
                        messages.append(_response_message(call, route_path, route, expression, uses))
            for message in filter(None, messages):
                if (path, call.line, message) in seen:
                    continue
                seen.add((path, call.line, message))
                findings.append({
                    "source": "api-call", "severity": "error", "code": "api-call-shape",
                    "path": path, "line": call.line, "message": message,
                })
                if len(findings) >= _MAX_FINDINGS:
                    return findings
    return findings


def collect_api_call_issues(workspace) -> list[str]:
    """Blocker strings for validate_app and the Phase 3 sweep."""
    return [PREFIX + finding["message"] for finding in collect_api_call_diagnostics(workspace)]
