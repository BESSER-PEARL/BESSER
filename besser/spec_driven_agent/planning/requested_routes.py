"""HTTP routes the user named verbatim, checked against the app's route decorators.

The gap planner and the requirements judge both read ``GET /stats`` as covered
by the scaffold's ``GET /statistics``. A client calling ``/stats`` still gets a
404, so a path the user wrote is checked here, from the source, not by an LLM.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from besser.spec_driven_agent.parsed_source import parse_source
from besser.utilities.workspace_fs import walk_plain

_SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", "env", "dist", "build",
    "tests", "test", "fixtures", "verification",
}
_VERBS = {"get", "post", "put", "patch", "delete"}
_ANY_VERB = {"api_route", "route"}

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


def declared_routes(workspace: str | Path) -> tuple[list[tuple[frozenset[str] | None, str]], set[str]]:
    """Every decorator route and every literal router/mount prefix in the app's Python."""
    workspace = Path(workspace)
    routes: list[tuple[frozenset[str] | None, str]] = []
    prefixes: set[str] = set()
    for root, dirs, files in walk_plain(workspace):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if not name.endswith(".py") or name.startswith("."):
                continue
            try:
                tree = parse_source((Path(root) / name).read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, ValueError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    prefix = _literal(next((kw.value for kw in node.keywords if kw.arg == "prefix"), None))
                    if prefix:
                        prefixes.add(prefix)
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
                        continue
                    verb = decorator.func.attr
                    if verb not in _VERBS | _ANY_VERB:
                        continue
                    path = _literal(decorator.args[0] if decorator.args else next(
                        (kw.value for kw in decorator.keywords if kw.arg == "path"), None))
                    if path is None:
                        continue
                    methods: frozenset[str] | None = frozenset({verb.upper()})
                    if verb in _ANY_VERB:
                        listed = next((kw.value for kw in decorator.keywords if kw.arg == "methods"), None)
                        names = [_literal(item) for item in getattr(listed, "elts", [])]
                        methods = frozenset(n.upper() for n in names if n) or None
                    routes.append((methods, path))
    return routes, prefixes


def route_declared(workspace: str | Path, method: str | None, path: str) -> bool:
    """A decorator serves ``path`` (alone or under a literal prefix) for ``method``."""
    routes, prefixes = declared_routes(workspace)
    wanted = _normalise(path)
    for methods, route in routes:
        if method and methods is not None and method not in methods:
            continue
        if wanted in {_normalise(route), *(_normalise(prefix + "/" + route) for prefix in prefixes)}:
            return True
    return False


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
