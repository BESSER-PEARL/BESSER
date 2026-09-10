"""Static coherence checks between frontend HTTP calls and backend routes.

This validator intentionally starts as report-only. It handles literal
``fetch`` and Axios URLs with optional template parameters, which covers the
high-confidence runtime-404 class without guessing about fully dynamic URLs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from besser.generators.llm.prompt_builder import build_endpoint_manifest


_FRONTEND_EXTENSIONS = (".js", ".jsx", ".ts", ".tsx")
_SKIP_DIRS = {"node_modules", "dist", "build", ".next", ".git", ".besser_snapshot"}
_MANIFEST_ROUTE_RE = re.compile(r"^\s{2}([A-Z, ]+?)\s{2,}(/\S+)\s*$", re.MULTILINE)
_FETCH_RE = re.compile(
    r"\bfetch\s*\(\s*(?P<quote>[`\"'])(?P<url>.*?)(?P=quote)"
    r"(?P<options>\s*,\s*\{.*?\})?\s*\)",
    re.DOTALL,
)
_AXIOS_METHOD_RE = re.compile(
    r"\baxios\s*\.\s*(?P<method>get|post|put|patch|delete|head|options)"
    r"\s*\(\s*(?P<quote>[`\"'])(?P<url>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)
_AXIOS_CONFIG_RE = re.compile(r"\baxios\s*\(\s*\{(?P<body>.*?)\}\s*\)", re.DOTALL)
_METHOD_FIELD_RE = re.compile(r"\bmethod\s*:\s*[\"']([A-Za-z]+)[\"']", re.IGNORECASE)
_URL_FIELD_RE = re.compile(r"\burl\s*:\s*(?P<quote>[`\"'])(?P<url>.*?)(?P=quote)", re.DOTALL)
_TEMPLATE_VALUE_RE = re.compile(r"\$\{[^}]+\}")
_ROUTE_VALUE_RE = re.compile(r"\{[^}/]+\}")


@dataclass(frozen=True)
class FrontendCall:
    method: str
    path: str
    file: str
    line: int


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
    return value


def _line_number(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _calls_in_file(rel_path: str, content: str) -> list[FrontendCall]:
    calls: list[FrontendCall] = []
    for match in _FETCH_RE.finditer(content):
        path = _normalize_frontend_url(match.group("url"))
        if path is None:
            continue
        method_match = _METHOD_FIELD_RE.search(match.group("options") or "")
        method = method_match.group(1).upper() if method_match else "GET"
        calls.append(FrontendCall(method, path, rel_path, _line_number(content, match.start())))

    for match in _AXIOS_METHOD_RE.finditer(content):
        path = _normalize_frontend_url(match.group("url"))
        if path is not None:
            calls.append(FrontendCall(
                match.group("method").upper(),
                path,
                rel_path,
                _line_number(content, match.start()),
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
        calls.append(FrontendCall(method, path, rel_path, _line_number(content, match.start())))
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
    for root, dirs, files in os.walk(output_dir):
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
            ):
                continue
            try:
                if os.path.getsize(full_path) > 1_000_000:
                    continue
                with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except OSError:
                continue
            calls.extend(_calls_in_file(rel_path, content))

    issues: list[str] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
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
        issues.append(
            f"endpoint coherence: {call.file} line {call.line} calls "
            f"{call.method} {call.path}, but {detail}. Copy the exact path and "
            "method from the backend endpoint manifest."
        )
        if len(issues) >= max_issues:
            break
    return issues
