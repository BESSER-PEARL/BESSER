"""Read-only source contracts for generated FastAPI model actions.

The model declares an operation; its generated ``/methods/`` handler is the
executable extension point. An ORM method with the same name does not implement
that handler automatically. These checks establish presence and absence of known
placeholders only, not behavioral correctness (which still needs runtime tests).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path


_SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", "env", "dist", "build",
    "tests", "test", "fixtures", "verification",
}
_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


@dataclass(frozen=True)
class ActionEndpoint:
    path: str
    function: str
    http_method: str
    route: str
    line: int
    stub_reason: str | None
    router_binding: str = ""
    router_prefix: str | None = None

    @property
    def action(self) -> str:
        return self.route.rstrip("/").rsplit("/", 1)[-1]


def _placeholder_reason(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if not body or all(
        isinstance(item, ast.Pass)
        or (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
            and item.value.value is Ellipsis)
        for item in body
    ):
        return "an empty action body"
    return _block_placeholder(body)


def _contains_return(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return False
    return isinstance(node, ast.Return) or any(_contains_return(child) for child in ast.iter_child_nodes(node))


def _block_placeholder(body: list[ast.stmt]) -> str | None:
    if not body or any(_contains_return(item) for item in body[:-1]):
        return None
    return _terminal_placeholder(body[-1])


def _terminal_placeholder(statement: ast.stmt) -> str | None:
    """Prove a terminal placeholder, not merely a possible unsupported branch.

    Generated handlers end in a try block which unconditionally raises 501;
    their exception handlers only re-raise/wrap it. Do not descend into nested
    helpers or conditional error paths in otherwise implemented operations.
    """
    if isinstance(statement, ast.Try):
        if statement.finalbody or statement.orelse:
            return None
        if any(not handler.body or not isinstance(handler.body[-1], ast.Raise)
               or any(_contains_return(item) for item in handler.body)
               for handler in statement.handlers):
            return None
        return _block_placeholder(statement.body)
    if not isinstance(statement, (ast.Raise, ast.Return)):
        return None
    value = statement.exc if isinstance(statement, ast.Raise) else statement.value
    if isinstance(statement, ast.Raise) and isinstance(value, ast.Name):
        return "NotImplementedError" if value.id == "NotImplementedError" else None
    if not isinstance(value, ast.Call):
        return None
    called = value.func.id if isinstance(value.func, ast.Name) else getattr(value.func, "attr", "")
    if isinstance(statement, ast.Raise) and called == "NotImplementedError":
        return "NotImplementedError"
    if called in {"HTTPException", "Response", "JSONResponse"}:
        status = next((kw.value for kw in value.keywords if kw.arg == "status_code"), None)
        if status is None and called == "HTTPException" and value.args:
            status = value.args[0]
        if isinstance(status, ast.Constant) and status.value == 501:
            return "HTTP 501"
        if isinstance(status, ast.Attribute) and status.attr == "HTTP_501_NOT_IMPLEMENTED":
            return "HTTP 501"
    return None


def _read_action_file(workspace: Path, path: Path) -> list[ActionEndpoint]:
    try:
        # Never read a source symlink outside the output tree.
        path.resolve().relative_to(workspace.resolve())
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, UnicodeError, ValueError, SyntaxError):
        return []
    endpoints = []
    router_prefixes: dict[str, str | None] = {}
    for function in tree.body:
        if isinstance(function, ast.Assign):
            value = function.value
            prefix = None
            if isinstance(value, ast.Call) and (
                getattr(value.func, "id", None) == "APIRouter"
                or getattr(value.func, "attr", None) == "APIRouter"
            ) and not value.args and not any(kw.arg is None for kw in value.keywords):
                prefix_arg = next((kw.value for kw in value.keywords if kw.arg == "prefix"), None)
                if prefix_arg is None:
                    prefix = ""
                elif isinstance(prefix_arg, ast.Constant) and isinstance(prefix_arg.value, str):
                    prefix = prefix_arg.value
            for target in function.targets:
                if isinstance(target, ast.Name):
                    router_prefixes[target.id] = prefix
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in function.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            verb = decorator.func.attr
            if verb not in _HTTP_METHODS:
                continue
            route = decorator.args[0] if decorator.args else next(
                (kw.value for kw in decorator.keywords if kw.arg == "path"), None,
            )
            if not isinstance(route, ast.Constant) or not isinstance(route.value, str):
                continue
            if "/methods/" not in route.value:
                continue
            binding = ast.unparse(decorator.func.value)
            endpoints.append(ActionEndpoint(
                path=path.relative_to(workspace).as_posix(), function=function.name,
                http_method=verb.upper(), route=route.value, line=function.lineno,
                stub_reason=_placeholder_reason(function),
                router_binding=binding, router_prefix=router_prefixes.get(binding),
            ))
    return endpoints


def collect_action_endpoints(workspace: str | Path) -> list[ActionEndpoint]:
    """Inventory actual model-action routes without importing generated code."""
    workspace = Path(workspace)
    endpoints = []
    for root, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".py") and not name.startswith("."):
                endpoints.extend(_read_action_file(workspace, Path(root) / name))
    return endpoints


def format_action_inventory(endpoints: list[ActionEndpoint]) -> str:
    if not endpoints:
        return "No generated model-action endpoints found."
    lines = [
        "These are executable action extension points, not ORM declarations. "
        "Paths are decorator routes; application/router mount prefixes may apply. "
        "Implement each unresolved handler or wire it to an implemented service. "
        "A body-present result is structural evidence only, not behavioral verification."
    ]
    for endpoint in endpoints:
        state = f"UNIMPLEMENTED: {endpoint.stub_reason}" if endpoint.stub_reason else "body present"
        lines.append(
            f"- {endpoint.http_method} {endpoint.route} -> {endpoint.path} "
            f"line {endpoint.line}, function {endpoint.function} [{state}]"
        )
    return "\n".join(lines)


def _current_endpoint(workspace: str | Path, expected: ActionEndpoint) -> ActionEndpoint | None:
    workspace = Path(workspace)
    candidates = [item for item in _read_action_file(workspace, workspace / expected.path)
                  if (item.http_method, item.route, item.router_binding, item.router_prefix)
                  == (expected.http_method, expected.route, expected.router_binding, expected.router_prefix)]
    exact = [item for item in candidates if item.function == expected.function]
    if exact:
        return exact[0] if len(exact) == 1 else None
    # A Python rename is not a removed HTTP operation. Only accept an
    # unambiguous same-file binding with a statically known unchanged prefix.
    # Matching a route in another module would require resolving include_router
    # mounts; equal decorator strings alone cannot prove an equivalent route.
    if len(candidates) == 1 and expected.router_prefix is not None:
        return candidates[0]
    return None


def action_body_present(workspace: str | Path, expected: ActionEndpoint) -> bool:
    """Fail closed on deletion, route removal, parse/read failure or a known stub."""
    current = _current_endpoint(workspace, expected)
    return current is not None and current.stub_reason is None


def action_gap_tasks(
    workspace: str | Path, endpoints: list[ActionEndpoint] | None = None,
) -> list[dict]:
    """Harness-owned tasks; omitted planner tasks cannot hide generated stubs."""
    if endpoints is None:
        endpoints = collect_action_endpoints(workspace)
    return [
        {
            "text": (
                f"Implement {endpoint.http_method} {endpoint.route} in {endpoint.path}, "
                f"function {endpoint.function}, according to the user's specification; "
                f"it currently contains {endpoint.stub_reason}. Keep the action route "
                "and wire any service/ORM implementation into this handler. Verify "
                "the requested successful and refused outcomes. The checklist's "
                "structural check only rejects missing handlers and known stubs; "
                "it does not prove business behavior."
            ),
            "verify": lambda endpoint=endpoint: action_body_present(workspace, endpoint),
            "_action_contract": _endpoint_key(endpoint),
        }
        for endpoint in endpoints if endpoint.stub_reason
    ]


def _endpoint_key(endpoint: ActionEndpoint) -> tuple[str, str, str, str]:
    return endpoint.path, endpoint.function, endpoint.http_method, endpoint.route


def merge_action_tasks(tasks: list, endpoints: list[ActionEndpoint]) -> list:
    """Fold exact single-action planner duplicates into stable harness tasks.

    Only our explicit source-derived ACTION HANDOFF mapping qualifies, not a
    fuzzy action-name match. Keep all planner detail as ``planning_notes``;
    leave multi-action tasks and unrelated deterministic checks untouched.
    The canonical task text stays stable for checkpoint verifier reattachment.
    """
    known = {_endpoint_key(item): item for item in endpoints}
    copied = [dict(item) if isinstance(item, dict) else item for item in tasks]
    anchors = {}
    for item in copied:
        if isinstance(item, dict):
            key = tuple(item.get("_action_contract", ()))
            if key in known:
                anchors[key] = item
                item["planning_notes"] = list(item.get("planning_notes") or [])
    merged = []
    for item in copied:
        if not isinstance(item, str) or "ACTION HANDOFF: " not in item:
            merged.append(item)
            continue
        # Match the full emitted mapping. In particular Order.cancel and
        # Invoice.cancel are separate operations despite the shared name.
        handoff = item.split("ACTION HANDOFF: ", 1)[1]
        matches = []
        for key, endpoint in known.items():
            marker = (
                f"{endpoint.http_method} {endpoint.route} is served by "
                f"{endpoint.path} function {endpoint.function}"
            )
            if marker + ";" in handoff or marker + "." in handoff:
                matches.append(key)
        if len(matches) != 1 or matches[0] not in anchors:
            merged.append(item)
            continue
        notes = anchors[matches[0]]["planning_notes"]
        if item not in notes:
            notes.append(item)
    return merged


def action_implementation_issues(
    workspace: str | Path, expected: list[ActionEndpoint] | None = None,
) -> list[str]:
    """Stable deterministic blockers, optionally preserving the initial routes."""
    if expected is None:
        expected = collect_action_endpoints(workspace)
    issues = []
    for endpoint in expected:
        current = _current_endpoint(workspace, endpoint)
        reason = current.stub_reason if current else "the expected handler is missing or unreadable"
        if reason:
            issues.append(
                f"action contract: {endpoint.path} line "
                f"{current.line if current else endpoint.line}: "
                f"{endpoint.http_method} {endpoint.route} "
                f"({current.function if current else endpoint.function}): {reason}. "
                "Implement and verify this route; an ORM-only edit does not complete it."
            )
    return issues
