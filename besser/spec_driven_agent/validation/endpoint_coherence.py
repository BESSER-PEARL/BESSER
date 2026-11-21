"""Static coherence checks between frontend HTTP calls and backend routes.

This validator is intentionally report-only. It handles literal
``fetch`` and Axios URLs with optional template parameters, which covers the
high-confidence runtime-404 class without guessing about fully dynamic URLs,
and the generated API client's calls: ``apiRequest(<METHOD>, <path>)`` and its
typed functions (``bookApi.create(...)``), resolved through ``api_contract.json``.

A URL built entirely out of interpolation is NOT such a call. The shared
api-client module most generated frontends use issues every request from one
wrapper, ``fetch(`${API_BASE}${path}`, options)``, which normalises to
``/{param}`` — no literal segment, nothing to compare (it would match
``/health`` and be misreported). Those calls are skipped: this validator
says nothing about a route the caller resolves at runtime. Reporting on the
resolved call sites instead (``api.create('carpark', ...)`` → ``POST
/carpark/``) would need the helper resolution ``acceptance._post_helpers``
does, and is deliberately left out rather than guessed at here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from besser.generators.react.api_client import client_functions
from besser.spec_driven_agent.agent.prompt_builder import build_endpoint_manifest
from besser.utilities.workspace_fs import walk_plain


_FRONTEND_EXTENSIONS = (".js", ".jsx", ".ts", ".tsx")
_SKIP_DIRS = {"node_modules", "dist", "build", ".next", ".git", ".besser_snapshot"}
# ``build_endpoint_manifest`` writes "  " + methods padded to 20 + " " + path.
# The padding disappears once the method list is longer than 20 characters
# ("GET, PUT, PATCH, DELETE" is 23), so the separator may be a single space;
# and the root route's path is just "/". Both must parse, or a call to a
# dropped route (e.g. "GET /") reads as a 404.
_MANIFEST_ROUTE_RE = re.compile(
    r"^ {2}([A-Z][A-Z, ]*?)[ \t]+(/\S*)[ \t]*$", re.MULTILINE
)
# The URL ends at its own closing quote, rather than running on to a later quote.
_FETCH_RE = re.compile(
    r"\bfetch\s*\(\s*(?P<quote>[`\"'])(?P<url>(?:(?!(?P=quote))[^\n])*)(?P=quote)")
# Comments and string literals, blanked before brackets are counted.
_NONCODE_RE = re.compile(
    r'''"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|//[^\r\n]*|/\*[\s\S]*?\*/''')
_METHOD_KEY_RE = re.compile(r"(?<![\w$.])method(?![\w$])")
_METHOD_VALUE_RE = re.compile(r"method\s*:\s*([\"'])([A-Za-z]+)\1")
# A TypeScript type argument (``axios.get<Order[]>(``), nested one level at most.
_TYPE_ARGUMENT = r"(?:\s*<[^<>()]*(?:<[^<>()]*>[^<>()]*)*>)?"


def _axios_method_re(clients) -> re.Pattern:
    """``<client>.<method>(<literal url>`` for ``axios`` or the given axios instances."""
    return re.compile(
        r"(?<![\w$])(?:" + "|".join(map(re.escape, clients)) + r")\s*\.\s*"
        r"(?P<method>(?i:get|post|put|patch|delete|head|options))" + _TYPE_ARGUMENT
        + r"\s*\(\s*(?P<quote>[`\"'])(?P<url>.*?)(?P=quote)",
        re.DOTALL,
    )


_AXIOS_METHOD_RE = _axios_method_re(("axios",))
_AXIOS_CONFIG_RE = re.compile(r"\baxios\s*\(\s*\{(?P<body>.*?)\}\s*\)", re.DOTALL)
_METHOD_FIELD_RE = re.compile(r"\bmethod\s*:\s*[\"']([A-Za-z]+)[\"']", re.IGNORECASE)
_URL_FIELD_RE = re.compile(r"\burl\s*:\s*(?P<quote>[`\"'])(?P<url>.*?)(?P=quote)", re.DOTALL)
# The generated client's request function: ``apiRequest("POST", "/book/", { body })``.
_API_REQUEST_RE = re.compile(
    r"(?<![\w$.])apiRequest" + _TYPE_ARGUMENT + r"\s*\(\s*(?P<mq>[\"'])(?P<method>[A-Za-z]+)(?P=mq)\s*,\s*"
    r"(?P<quote>[`\"'])(?P<url>(?:(?!(?P=quote))[^\n])*)(?P=quote)")
# A call of the generated typed client, ``taskApi.create(``, ``(taskApi as any).relations.tags.list(``.
_TYPED_CALL_RE = re.compile(
    r"(?<![\w$.])(?:\(\s*)?(?P<api>[a-z][\w$]*Api)(?:\s+as\s+[\w$<>\[\]]+\s*\))?"
    r"(?P<chain>(?:\s*\.\s*(?:relations\s*\.\s*[\w$]+|methods)(?![\w$]))?)\s*\.\s*(?P<function>[\w$]+)\s*\(")
# The typed client the React generator writes (src/api/generated/client.ts).
_GENERATED_CLIENT_RE = re.compile(r"(?:^|/)api/generated/client\.[jt]sx?$")
# Calls that bypass the generated client.
_RAW_CALLS = frozenset({"fetch", "axios", "axios-config", "fetch-helper"})
# Warnings about such calls: at most this many, each naming this many lines.
_MAX_BYPASS_ISSUES = 5
_MAX_BYPASS_LINES = 5
_TEMPLATE_VALUE_RE = re.compile(r"\$\{[^}]+\}")
_ROUTE_VALUE_RE = re.compile(r"\{[^}/]+\}")


@dataclass(frozen=True)
class FrontendCall:
    method: str
    path: str
    file: str
    line: int
    # Where the call starts and its URL literal ends (for a typed client call: its opening
    # parenthesis), and "fetch" / "axios" ("axios-config" for the axios config form),
    # "api-request" or "typed", for checks that read its arguments.
    start: int = 0
    url_end: int = 0
    client: str = ""
    # A typed client call: the function called (``taskApi.create``).
    function: str = ""


def _parse_manifest(manifest: str) -> dict[str, set[str]]:
    routes: dict[str, set[str]] = {}
    for match in _MANIFEST_ROUTE_RE.finditer(manifest):
        path = match.group(2)
        routes.setdefault(path, set()).update(
            method.strip() for method in match.group(1).split(",") if method.strip()
        )
    return routes


def _normalize_frontend_url(raw_url: str) -> str | None:
    value = raw_url.strip()
    # A template base URL is deployment configuration, not a route segment.
    value = re.sub(r"^\$\{[^}]+\}", "", value)
    value = _TEMPLATE_VALUE_RE.sub("{param}", value)
    if value.startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
            return None
        value = parsed.path
    else:
        value = value.split("?", 1)[0].split("#", 1)[0]
    if not value or value.startswith(("data:", "blob:")):
        return None
    if not value.startswith("/"):
        value = "/" + value
    value = re.sub(r"/{2,}", "/", value)
    # Treat obvious static-asset reads as frontend concerns, not API routes.
    if re.search(r"\.[A-Za-z0-9]{1,8}$", value):
        return None
    # No literal segment survived the interpolation: the route is chosen at
    # runtime and there is nothing here to check against the manifest.
    segments = [segment for segment in value.split("/") if segment]
    if segments and all(segment == "{param}" for segment in segments):
        return None
    return value


def _line_number(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _fetch_method(content: str, blanked: str, index: int) -> str | None:
    """The method of ``fetch(url)`` or ``fetch(url, {...})``: the options' own
    top-level ``method``, not one inside their body. None for options held in a
    variable, or a ``method`` that is not a literal."""
    index = _skip_space(blanked, index)
    if blanked.startswith(")", index):
        return "GET"
    if not blanked.startswith(",", index):
        return None
    index = _skip_space(blanked, index + 1)
    if not blanked.startswith("{", index):
        return None
    depth, method = 0, "GET"
    for position in range(index, len(blanked)):
        char = blanked[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return method if blanked.startswith(")", _skip_space(blanked, position + 1)) else None
        elif depth == 1 and _METHOD_KEY_RE.match(blanked, position):
            literal = _METHOD_VALUE_RE.match(content, position)
            if literal is None:
                return None
            method = literal.group(2).upper()
    return None


def typed_client_functions(output_dir: str) -> dict:
    """The generated typed client's functions and their routes, from the workspace's
    ``api_contract.json``; empty without one."""
    for root, dirs, files in walk_plain(output_dir):
        dirs[:] = sorted(directory for directory in dirs if directory not in _SKIP_DIRS)
        if "api_contract.json" in files:
            try:
                with open(os.path.join(root, "api_contract.json"), encoding="utf-8") as handle:
                    return client_functions(json.load(handle))
            except (OSError, ValueError, KeyError, TypeError):
                return {}
    return {}


def _typed_calls(rel_path: str, content: str, typed: dict) -> list[FrontendCall]:
    """Calls of the typed client; a function the client does not have has no method or path."""
    apis = {key.split(".", 1)[0] for key in typed}
    calls = []
    for match in _TYPED_CALL_RE.finditer(content):
        if match.group("api") not in apis:
            continue
        chain = re.sub(r"\s+", "", match.group("chain"))
        function = match.group("api") + chain + "." + match.group("function")
        route = typed.get(function)
        calls.append(FrontendCall(route["method"] if route else "", route["path"] if route else "", rel_path,
                                  _line_number(content, match.start()), match.start(), match.end() - 1,
                                  "typed", function))
    return calls


def _calls_in_file(rel_path: str, content: str, clients=("axios",), helpers=(), typed=None) -> list[FrontendCall]:
    """Literal-URL calls; ``clients`` names the axios instances besides ``axios``, ``helpers``
    the functions that call ``fetch(BASE + path, options)`` for their caller, and ``typed`` the
    generated typed client's functions (``typed_client_functions``)."""
    calls: list[FrontendCall] = list(_typed_calls(rel_path, content, typed) if typed else ())
    for match in _API_REQUEST_RE.finditer(content):
        path = _normalize_frontend_url(match.group("url"))
        if path is not None:
            calls.append(FrontendCall(match.group("method").upper(), path, rel_path,
                                      _line_number(content, match.start()), match.start(),
                                      match.end("url") + 1, "api-request"))
    blanked = None
    if helpers:
        blanked = _NONCODE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), content)
        helper_calls = re.compile(
            r"(?<![\w$.])(?:" + "|".join(map(re.escape, helpers)) + r")" + _TYPE_ARGUMENT
            + r"\s*\(\s*(?P<quote>[`\"'])(?P<url>(?:(?!(?P=quote))[^\n])*)(?P=quote)")
        for match in helper_calls.finditer(content):
            path = _normalize_frontend_url(match.group("url"))
            method = _fetch_method(content, blanked, match.end()) if path is not None else None
            if method is not None:
                calls.append(FrontendCall(method, path, rel_path, _line_number(content, match.start()),
                                          match.start(), match.end("url") + 1, "fetch-helper"))
    for match in _FETCH_RE.finditer(content):
        path = _normalize_frontend_url(match.group("url"))
        if path is None:
            continue
        if blanked is None:
            blanked = _NONCODE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), content)
        method = _fetch_method(content, blanked, match.end())
        if method is None:
            continue
        calls.append(FrontendCall(method, path, rel_path, _line_number(content, match.start()),
                                  match.start(), match.end("url") + 1, "fetch"))

    axios_calls = _AXIOS_METHOD_RE if tuple(clients) == ("axios",) else _axios_method_re(clients)
    for match in axios_calls.finditer(content):
        path = _normalize_frontend_url(match.group("url"))
        if path is not None:
            calls.append(FrontendCall(
                match.group("method").upper(),
                path,
                rel_path,
                _line_number(content, match.start()),
                match.start(),
                match.end("url") + 1,
                "axios",
            ))

    for match in _AXIOS_CONFIG_RE.finditer(content):
        body = match.group("body")
        url_match = _URL_FIELD_RE.search(body)
        if not url_match:
            continue
        path = _normalize_frontend_url(url_match.group("url"))
        if path is None:
            continue
        method_match = _METHOD_FIELD_RE.search(body)
        method = method_match.group(1).upper() if method_match else "GET"
        calls.append(FrontendCall(method, path, rel_path, _line_number(content, match.start()),
                                  match.start(), 0, "axios-config"))
    return calls


def _paths_match(frontend_path: str, backend_path: str) -> bool:
    # Preserve trailing slash semantics while treating route-variable names and
    # concrete IDs as equivalent path segments.
    frontend_parts = frontend_path.split("/")
    backend_parts = backend_path.split("/")
    if len(frontend_parts) != len(backend_parts):
        return False
    for frontend_part, backend_part in zip(frontend_parts, backend_parts):
        if _ROUTE_VALUE_RE.fullmatch(backend_part):
            if not frontend_part:
                return False
            continue
        if _ROUTE_VALUE_RE.fullmatch(frontend_part):
            if not backend_part:
                return False
            continue
        if frontend_part != backend_part:
            return False
    return True


def _covering_function(call: FrontendCall, by_route: dict) -> str | None:
    """The client function of the route the call reaches; an interpolated segment
    prefers a route parameter (``/task/${id}/`` is ``get``, not ``count``)."""
    matches = []
    for (method, path), function in by_route.items():
        if method == call.method and _paths_match(call.path, path):
            literal = sum(1 for mine, theirs in zip(call.path.split("/"), path.split("/"))
                          if _ROUTE_VALUE_RE.fullmatch(mine) and not _ROUTE_VALUE_RE.fullmatch(theirs))
            matches.append((literal, function))
    return min(matches)[1] if matches else None


def collect_endpoint_coherence_issues(
    output_dir: str,
    *,
    max_issues: int = 10,
) -> list[str]:
    """Report literal frontend calls that have no matching generated route."""
    routes = _parse_manifest(build_endpoint_manifest(output_dir))
    if not routes:
        return []

    calls: list[FrontendCall] = []
    typed = typed_client_functions(output_dir)
    has_client = False
    for root, dirs, files in walk_plain(output_dir):
        dirs[:] = [directory for directory in dirs if directory not in _SKIP_DIRS]
        for filename in files:
            if not filename.endswith(_FRONTEND_EXTENSIONS):
                continue
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, output_dir).replace("\\", "/")
            rel_lower = rel_path.lower()
            if not (
                "frontend/" in rel_lower
                or rel_lower.startswith("src/")
                or "/src/" in rel_lower
            ) or _GENERATED_CLIENT_RE.search(rel_lower):
                # Regeneration rewrites the generated client; its functions are checked where code calls them.
                has_client = has_client or bool(_GENERATED_CLIENT_RE.search(rel_lower))
                continue
            try:
                if os.path.getsize(full_path) > 1_000_000:
                    continue
                with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except OSError:
                continue
            calls.extend(_calls_in_file(rel_path, content, typed=typed))

    # The generated client's function for each route it calls, when the app has the client.
    by_route = {}
    for function, route in sorted(typed.items()) if has_client else ():
        by_route.setdefault((route["method"], route["path"]), function)
    issues: list[str] = []
    # Raw calls to covered routes, by (file, client function): reported after the mismatches, under their own cap.
    bypassing: dict[tuple[str, str], list[FrontendCall]] = {}
    seen: set[tuple[str, str]] = set()
    for call in calls:
        covering = _covering_function(call, by_route) if call.client in _RAW_CALLS else None
        if covering:
            bypassing.setdefault((call.file, covering), []).append(call)
            continue
        if len(issues) >= max_issues:
            continue
        if call.client == "typed" and not call.method:
            issues.append(
                f"endpoint coherence: {call.file} line {call.line} calls {call.function}(...), which the "
                "generated API client does not have. Use a function of src/api/generated/client.ts."
            )
            continue
        key = (call.method, call.path)
        if key in seen:
            continue
        seen.add(key)
        matching_paths = [path for path in routes if _paths_match(call.path, path)]
        if any(call.method in routes[path] for path in matching_paths):
            continue
        if matching_paths:
            served_methods = sorted({method for path in matching_paths for method in routes[path]})
            detail = f"the path exists only for {', '.join(served_methods)}"
        else:
            sample = ", ".join(list(routes)[:4])
            detail = f"no generated backend route matches it (examples: {sample})"
        fix = ("Call the route through its function in src/api/generated/client.ts; a route the client does "
               "not have is listed with its exact path and method in the backend endpoint manifest."
               if has_client else "Copy the exact path and method from the backend endpoint manifest.")
        issues.append(
            f"endpoint coherence: {call.file} line {call.line} calls "
            f"{call.method} {call.path}, but {detail}. {fix}"
        )
    for (file, covering), group in list(bypassing.items())[:_MAX_BYPASS_ISSUES]:
        lines = ", ".join(str(call.line) for call in group[:_MAX_BYPASS_LINES])
        if len(group) > _MAX_BYPASS_LINES:
            lines += f" (+{len(group) - _MAX_BYPASS_LINES} more)"
        first = group[0]
        issues.append(
            f"endpoint coherence: {file} {'lines' if len(group) > 1 else 'line'} {lines} "
            f"{'call' if len(group) > 1 else 'calls'} {first.method} {first.path} with {first.client}, which the "
            f"generated API client has as {covering}(...). Call that function of "
            "src/api/generated/client.ts: it returns the plain value and throws ApiError."
        )
    return issues
