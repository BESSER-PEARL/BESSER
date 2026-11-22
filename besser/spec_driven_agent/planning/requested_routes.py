"""HTTP routes the user named verbatim, checked against the routes the app declares.

The gap planner and the requirements judge both read ``GET /stats`` as covered
by the scaffold's ``GET /statistics``. A client calling ``/stats`` still gets a
404, so a path the user wrote is checked here, from the source, not by an LLM.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from besser.spec_driven_agent.planning.mutation_inventory import (
    include_prefixes,
    router_bindings,
    source_files,
)

_VERBS = {"get", "post", "put", "patch", "delete"}
_ANY_VERB = {"api_route", "route"}
# FastAPI's defaults: served unless the app sets the keyword to None.
_BUILTIN_PAGES = {"openapi_url": "/openapi.json", "docs_url": "/docs", "redoc_url": "/redoc"}

_PATH = r"(/[A-Za-z0-9_\-./{}:]*[A-Za-z0-9_}/])"
# "GET /stats", "`POST /orders/{id}/cancel`", or "a /stats endpoint".
_WITH_METHOD_RE = re.compile(rf"(?<![\w/])(GET|POST|PUT|PATCH|DELETE)\s+[`'\"]?{_PATH}")
_WITH_NOUN_RE = re.compile(rf"(?<![\w/:.])[`'\"]?{_PATH}[`'\"]?\s+endpoint")
# "without a /stats endpoint", "do not add GET /admin": not a request for it.
_NEGATED_RE = re.compile(r"\b(?:no|not|never|without|remove|drop)\b|n't\b", re.I)
_PARAM_RE = re.compile(r"\{[^/{}]*\}|(?<=/):[^/]+")


def requested_routes(text: str) -> list[tuple[str | None, str]]:
    """``(method or None, path)`` for each HTTP path written in the request."""
    found = {match.start(2): (match.start(), match.group(1), match.group(2))
             for match in _WITH_METHOD_RE.finditer(text)}
    for match in _WITH_NOUN_RE.finditer(text):
        found.setdefault(match.start(1), (match.start(), None, match.group(1)))
    routes: list[tuple[str | None, str]] = []
    for start, method, path in sorted(found.values()):
        clause = re.split(r"[.;,:!?\n]", text[max(0, start - 25):start])[-1]
        if _NEGATED_RE.search(clause) or _normalise(path) == "/":
            continue
        if (method, path) not in routes:
            routes.append((method, path))
    return routes


def _normalise(path: str) -> str:
    return "/" + "/".join(part for part in _PARAM_RE.sub("{}", path).split("/") if part)


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _methods(call: ast.Call, verb: str) -> frozenset[str] | None:
    """The methods a route call serves; ``None`` (any) when not listed."""
    if verb in _VERBS:
        return frozenset({verb.upper()})
    listed = next((kw.value for kw in call.keywords if kw.arg == "methods"), None)
    names = [_literal(item) for item in getattr(listed, "elts", [])]
    return frozenset(n.upper() for n in names if n) or None


def _builtin_pages(app: ast.Call) -> list[str]:
    """The docs pages a ``FastAPI(...)`` serves: ``docs_url=None`` turns one off,
    ``openapi_url=None`` all three."""
    if any(kw.arg is None for kw in app.keywords):
        return []
    pages = [_literal(node) if (node := next((kw.value for kw in app.keywords if kw.arg == name), None))
             else default for name, default in _BUILTIN_PAGES.items()]
    return [page for page in pages if page] if pages[0] else []


def declared_routes(workspace: str | Path) -> list[tuple[frozenset[str] | None, frozenset[str]]]:
    """Every route the app's Python declares: its methods (``None``: any) and
    the normalised paths it may be served at.

    Routes come from method decorators, ``add_api_route`` and FastAPI's own
    docs pages. Their path is resolved through the router's prefix, the
    ``include_router`` chain and an app's ``mount`` (``mutation_inventory``'s
    resolution). A route also matches alone or under any literal ``prefix=`` in
    the app, which covers a chain that is not literal.
    """
    root = Path(workspace).resolve()
    parsed, unknown = source_files(root)
    bindings = {path: router_bindings(tree) for path, tree in parsed.items()}
    served = include_prefixes(parsed, bindings, unknown)
    found: list[tuple[frozenset[str] | None, tuple[Path, str] | None, str]] = []
    prefixes: set[str] = set()
    for path, tree in parsed.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                prefix = _literal(next((kw.value for kw in node.keywords if kw.arg == "prefix"), None))
                if prefix:
                    prefixes.add(prefix)
            calls = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = [d for d in node.decorator_list if isinstance(d, ast.Call)
                         and isinstance(d.func, ast.Attribute) and d.func.attr in _VERBS | _ANY_VERB]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "add_api_route"):
                calls = [node]
            for call in calls:
                route = _literal(call.args[0] if call.args else next(
                    (kw.value for kw in call.keywords if kw.arg == "path"), None))
                if route is not None:
                    receiver = getattr(call.func.value, "id", None)
                    key = (path, receiver) if receiver in bindings[path] else None
                    found.append((_methods(call, call.func.attr), key, route))
        for name, (kind, _prefix, app) in bindings[path].items():
            if kind == "FastAPI":
                found.extend((frozenset({"GET"}), (path, name), page) for page in _builtin_pages(app))

    routes = []
    for methods, key, route in found:
        paths = {_normalise(route), *(_normalise(prefix + "/" + route) for prefix in prefixes)}
        if key is not None:
            own = bindings[key[0]][key[1]][1]
            paths |= {_normalise(f"{base}/{own}/{route}") for base in served[key]
                      if base is not None and own is not None}
        routes.append((methods, frozenset(paths)))
    return routes


def route_declared(workspace: str | Path, method: str | None, path: str) -> bool:
    """A route serves ``path`` for ``method`` (see :func:`declared_routes`)."""
    wanted = _normalise(path)
    return any(wanted in paths and not (method and methods is not None and method not in methods)
               for methods, paths in declared_routes(workspace))


def requested_route_tasks(workspace: str | Path, request: str) -> list[dict]:
    """Harness-owned tasks for each path the user wrote that the app does not declare."""
    tasks = []
    for method, path in requested_routes(request):
        if route_declared(workspace, method, path):
            continue
        label = f"{method} {path}" if method else path
        tasks.append({
            "text": (
                f"Add the requested endpoint {label}: the user asked for this exact "
                "path and the app does not declare it yet. A route with a different "
                "path does not satisfy it; keep the existing routes and add this one."
            ),
            "verify": lambda method=method, path=path: route_declared(workspace, method, path),
        })
    return tasks
