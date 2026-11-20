"""Phase 3: open every screen of a built frontend in a headless browser.

The other checks build the frontend and drive the backend over HTTP, but none
renders a page: a bundle that builds and then dies on mount, or a screen whose
data call answers 500, passes all of them. After the build has passed, this
check boots the app's backend on a scratch database, serves the build output
and opens each screen in a fresh tab of a headless Chromium
(``ui_probe_driver.mjs``, the DevTools protocol over Node's own WebSocket).

Screens are the router's literal paths read from the source, ``/`` first, plus
same-origin links found on the opened pages that match a declared route, up to
:data:`MAX_SCREENS`. Paths with parameters are opened only when a page links to
them; the database is empty, so usually they are not.

Only observed, deterministic signals block (``screen:``): an uncaught exception
while a declared screen loads, a declared screen whose root is still empty once
the page is idle, and a 5xx from the app's own backend. A 4xx, a console error,
an unhandled promise rejection and anything else is a ``screen warning:``. A
screen whose own backend was unreachable, or whose external script could not
load (the check has no network), only gets warnings: another check owns those.
Nothing is clicked or typed.

Optional by construction: with no browser, no Node.js or no sandbox the result
is one ``screens: not run (...)`` note, never a blocker.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import NamedTuple

from besser.spec_driven_agent.settings import AgentSettings
from besser.spec_driven_agent.validation.frontend_forms import _DYNAMIC, _attr, _jsx_tags, _path_text
from besser.spec_driven_agent.validation.frontend_resolution import _read, _walk

logger = logging.getLogger(__name__)

PREFIX = "screen:"
WARNING_PREFIX = "screen warning:"
NOT_RUN_PREFIX = "screens: not run"

MAX_SCREENS = 25
SCREEN_SECONDS = 15
TOTAL_SECONDS = 150
_BACKEND_BOOT_SECONDS = 30
# Beyond the driver's own deadline: its watchdog prints the report and exits first.
_PROCESS_SLACK_SECONDS = 15
_MIN_SECONDS = 20

_DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_probe_driver.mjs")
_MARKER = "BESSER_SCREEN_REPORT:"
_BUILD_OUTPUT_DIRS = ("dist", "build", "out")
_SOURCE_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_RESOLVE_EXT = (".tsx", ".ts", ".jsx", ".js")
_BACKEND_SKIP = ("node_modules", "__pycache__", ".besser_snapshot", "dist", "build", "data", ".besser_*", "*.db")
_DEFAULT_BACKEND_PORT = 8000
_MAX_BACKEND_PORTS = 4

# Chromium-family browsers, by the names their packages install.
_BROWSER_NAMES = ("chromium-headless-shell", "chromium", "chromium-browser", "google-chrome",
                  "google-chrome-stable", "microsoft-edge", "msedge", "chrome")
_WINDOWS_BROWSERS = (r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe",
                     r"Chromium\Application\chrome.exe")

_HASH_ROUTER = re.compile(r"\b(?:HashRouter|createHashRouter)\b")
_OBJECT_ROUTER = re.compile(r"\b(?:createBrowserRouter|createHashRouter|createMemoryRouter|useRoutes)\b")
_OBJECT_PATH = re.compile(r"""\bpath\s*:\s*(['"`])([^'"`]*)\1""")
_OBJECT_ELEMENT = re.compile(r"""\belement\s*:\s*(<[^,}]*)|\bComponent\s*:\s*([A-Z][\w$]*)""")
_COMPONENT = re.compile(r"<\s*([A-Z][\w$]*)")
_IDENTIFIER = re.compile(r"\s*([A-Z][\w$]*)\s*$")
_LOCAL_ORIGIN = re.compile(r"https?://(?:localhost|127\.0\.0\.1)(?::(\d{2,5}))?(?![\w.])")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")

# Boots the generated FastAPI app as the runbook's ``.besser_probe.py up`` does.
_SERVE = (
    "import os, sys\n"
    "sys.path.insert(0, os.getcwd())\n"
    "import main_api\n"
    "app = getattr(main_api, 'app', None)\n"
    "if app is None:\n"
    "    app = next((v for v in vars(main_api).values()\n"
    "                if hasattr(v, 'openapi') and hasattr(v, 'routes')), None)\n"
    "if app is None:\n"
    "    raise SystemExit('main_api defines no FastAPI app')\n"
    "import uvicorn\n"
    "uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')\n"
)


def not_run(reason: str) -> str:
    """The single note of a screen check that did not run: an unknown, never a blocker."""
    return f"{NOT_RUN_PREFIX} ({reason})"


def find_browser() -> str | None:
    """A Chromium-family browser on this machine, or ``None``. Never downloads one."""
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    if os.name == "nt":
        for base in filter(None, (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"),
                                  os.environ.get("LOCALAPPDATA"))):
            for rel in _WINDOWS_BROWSERS:
                path = os.path.join(base, rel)
                if os.path.isfile(path):
                    return path
    return None


class _Routes(NamedTuple):
    visit: list[str]                  # literal paths, opened in this order
    declared: list[tuple[str, str]]   # (regex source, page file or "") for every declared path
    hash: bool


def _route_pattern(path: str) -> str:
    """A router path as an anchored regex; ``:param`` and computed parts match one segment."""
    parts = []
    for segment in (path.rstrip("/") or "/").split("/")[1:]:
        if segment == "*":
            parts.append("(?:/.*)?")
        elif segment.startswith(":") and segment.endswith("?"):
            parts.append("(?:/[^/]+)?")
        elif segment.startswith(":") or _DYNAMIC in segment:
            parts.append("/[^/]+")
        else:
            parts.append("/" + re.escape(segment))
    return "^" + "".join(parts) + "/?$"


def _resolve_import(spec: str, importer: str, project: str) -> str | None:
    if spec.startswith("@/"):
        base = os.path.join(project, "src", spec[2:])
    elif spec.startswith("."):
        base = os.path.join(os.path.dirname(importer), spec)
    else:
        return None
    base = os.path.normpath(base)
    for candidate in (base, *(base + ext for ext in _RESOLVE_EXT),
                      *(os.path.join(base, "index" + ext) for ext in _RESOLVE_EXT)):
        if os.path.isfile(candidate):
            return candidate
    return None


def _component_file(name: str, text: str, importer: str, project: str) -> str:
    """The file defining component ``name`` as seen from ``importer``; ``importer`` itself when local."""
    quoted = r"""(['"])([^'"]+)\1"""
    for pattern in (rf"\bimport\s+{name}\b[^;'\"]*?\bfrom\s*{quoted}",
                    rf"\bimport\s*(?:[\w$]+\s*,\s*)?\{{[^}}]*\b{name}\b[^}}]*\}}\s*from\s*{quoted}",
                    rf"\b{name}\s*=\s*(?:React\.)?lazy\(\s*(?:async\s*)?\(\s*\)\s*=>\s*import\(\s*{quoted}"):
        found = re.search(pattern, text)
        if found:
            return _resolve_import(found.group(2), importer, project) or ""
    if re.search(rf"\b(?:function|class|const|let|var)\s+{name}\b", text):
        return importer
    return ""


def _element_file(expr: str | None, text: str, importer: str, project: str) -> str:
    """The page an ``element``/``component`` expression renders: its last component, the innermost."""
    if not expr:
        return ""
    names = _COMPONENT.findall(expr) or _IDENTIFIER.findall(expr)
    return _component_file(names[-1], text, importer, project) if names else ""


def discover_routes(project: str) -> _Routes:
    """The screens a frontend's router declares, read from its source."""
    declared: dict[str, tuple[str, str]] = {}
    hash_router = False

    def add(path: str, file: str) -> None:
        if path in ("*", "/*") or not path.startswith("/"):
            return  # a catch-all is the not-found page; a relative path depends on its parent
        rel = os.path.relpath(file, project).replace("\\", "/") if file else ""
        if path not in declared or (rel and not declared[path][1]):
            declared[path] = (_route_pattern(path), rel)

    for folder, files in _walk(project):
        for name in sorted(files):
            full = os.path.join(folder, name)
            if name == "ui_components.json":
                try:
                    pages = json.loads(_read(full)).get("pages") or []
                except (ValueError, AttributeError):
                    pages = []
                for page in pages if isinstance(pages, list) else []:
                    if isinstance(page, dict) and isinstance(page.get("route_path"), str):
                        add(page["route_path"], "")
                continue
            if not name.endswith(_SOURCE_EXT):
                continue
            text = _read(full)
            if "Route" not in text and "path" not in text:
                continue
            hash_router = hash_router or bool(_HASH_ROUTER.search(text))
            for _, attrs, _ in _jsx_tags(text, "Route"):
                expr = _attr(attrs, "path")
                path = _path_text(expr) if expr else None
                if path:
                    element = _attr(attrs, "element") or _attr(attrs, "component")
                    add(path, _element_file(element, text, full, project))
            if _OBJECT_ROUTER.search(text):
                for found in _OBJECT_PATH.finditer(text):
                    element = _OBJECT_ELEMENT.search(text, found.end(), found.end() + 300)
                    expr = (element.group(1) or element.group(2)) if element else None
                    add(found.group(2), _element_file(expr, text, full, project))
    visit = ["/"] + sorted(path for path in declared
                           if path != "/" and _DYNAMIC not in path and ":" not in path and "*" not in path)
    return _Routes(visit, list(declared.values()), hash_router)


def _build_output(project: str) -> str | None:
    """The build's output folder: the newest of dist/build/out holding an index.html."""
    found = [(os.path.getmtime(os.path.join(project, name, "index.html")), os.path.join(project, name))
             for name in _BUILD_OUTPUT_DIRS if os.path.isfile(os.path.join(project, name, "index.html"))]
    return max(found)[1] if found else None


def _bundle_backend_ports(site: str) -> list[int]:
    """The local ports the built bundle names, most named first; ``[8000]`` when it names none.

    A build-time API address and the code's fallback (``VITE_API_URL || "http://localhost:8000"``)
    both stay in the bundle, so the backend answers on each of them.
    """
    counts: Counter = Counter()
    for folder, _, files in os.walk(site):
        for name in files:
            path = os.path.join(folder, name)
            if name.endswith((".js", ".mjs", ".html")) and os.path.getsize(path) < 10_000_000:
                counts.update(int(port) for port in _LOCAL_ORIGIN.findall(_read(path)) if port)
    ports = [port for port, _ in counts.most_common() if 0 < port < 65536]
    return ports[:_MAX_BACKEND_PORTS] or [_DEFAULT_BACKEND_PORT]


def _shown_url(url: str) -> str:
    """A request URL without the check's own origin: ``/books/?page=1``."""
    return re.sub(r"^https?://[^/]+", "", url or "") or "/"


def _first_line(text: str) -> str:
    return (str(text or "").strip().splitlines() or [""])[0][:200]


def _is_local(url: str) -> bool:
    host = re.match(r"^https?://([^/:]+|\[[^\]]+\])", url or "")
    return bool(host) and host.group(1) in _LOCAL_HOSTS


def _listed(items: list[str], limit: int = 3) -> str:
    items = list(dict.fromkeys(items))
    return ", ".join(items[:limit]) + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _screen_findings(screen: dict, declared: bool) -> tuple[list[str], list[str]]:
    """``(blocker parts, warning parts)`` of one opened screen."""
    # The browser asks for /favicon.ico on its own; the app never requested it.
    requests = [r for r in screen.get("requests") or [] if not _shown_url(r.get("url")).startswith("/favicon.ico")]
    exceptions = screen.get("exceptions") or []
    blockers, warnings = [], []
    backend_down = any(r.get("own") and r.get("unreachable") for r in requests)
    script_missing = any(r.get("type") == "Script" and "status" not in r and not _is_local(r.get("url", ""))
                         for r in requests)
    # Another check owns these causes; the screen only shows their effect.
    excuse = ("the app's backend was not reachable" if backend_down
              else "an external script could not load without a network" if script_missing else "")
    render_parts = []
    thrown = [e for e in exceptions if not e.get("promise")]
    if thrown:
        render_parts.append(f"uncaught {_first_line(thrown[0].get('text'))}"
                            + (f" (+{len(thrown) - 1} more)" if len(thrown) > 1 else ""))
    seconds = round((screen.get("ms") or 0) / 1000, 1)
    if screen.get("rendered") is False:
        if screen.get("idle"):
            render_parts.append(f"the page rendered nothing ({screen.get('mount') or 'the root'} "
                                "stayed empty after load)")
        else:
            warnings.append(f"nothing rendered yet after {seconds}s with requests still pending")
    elif not screen.get("idle") and not screen.get("error"):
        # A request still open, or a fetch loop that never stops (the count tells which).
        warnings.append(f"the page did not settle within {seconds}s ({screen.get('request_count', 0)} requests)")
    if render_parts and declared and not excuse:
        blockers.extend(render_parts)
    elif render_parts:
        reason = excuse or "the path matches no declared route"
        warnings.extend(f"{part} (not counted as a blocker: {reason})" for part in render_parts)
    server_errors = [f"{r.get('method')} {_shown_url(r.get('url'))} answered {r['status']}"
                     for r in requests if isinstance(r.get("status"), int) and r["status"] >= 500
                     and r.get("own") and not r.get("unreachable")]
    if server_errors:
        blockers.append(f"{_listed(server_errors)} from the app's backend during load")
    refused = [f"{r.get('method')} {_shown_url(r.get('url'))} answered {r['status']}"
               for r in requests if isinstance(r.get("status"), int) and 400 <= r["status"] < 500]
    if refused:
        warnings.append(_listed(refused))
    others = [f"{r.get('method')} {_shown_url(r.get('url'))} answered {r['status']}"
              for r in requests if isinstance(r.get("status"), int) and r["status"] >= 500
              and not r.get("own")]
    if others:
        warnings.append(_listed(others))
    unreachable = [f"{r.get('method')} {_shown_url(r.get('url'))}" for r in requests
                   if r.get("own") and r.get("unreachable")]
    if unreachable:
        warnings.append(f"the backend was not reachable for {_listed(unreachable)}")
    unanswered = [f"{r.get('method')} {r.get('url')}" for r in requests
                  if "error" in r and _is_local(r.get("url", "")) and not r.get("own")]
    if unanswered:
        warnings.append(f"nothing answered {_listed(unanswered)}")
    rejections = [e for e in exceptions if e.get("promise")]
    if rejections:
        warnings.append(f"unhandled promise rejection: {_first_line(rejections[0].get('text'))}"
                        + (f" (+{len(rejections) - 1} more)" if len(rejections) > 1 else ""))
    console = screen.get("console_errors") or []
    if console:
        warnings.append(f"{len(console)} console error(s), first: {_first_line(console[0])}")
    if screen.get("error"):
        warnings.append(f"the check could not finish this screen: {_first_line(screen['error'])}")
    return blockers, warnings


def _matching_file(route: str, routes: _Routes, rel: str) -> tuple[bool, str]:
    """Whether ``route`` is declared, and the workspace-relative page file serving it ("" if unknown)."""
    declared, file = route == "/", ""
    for pattern, page in routes.declared:
        if re.match(pattern, route):
            declared = True
            if page and not file:
                file = page if rel == "." else f"{rel}/{page}"
    return declared, file


def _remove(folder: str) -> None:
    """Delete the scratch folder; on Windows a closing browser can hold its profile for a moment."""
    for _ in range(20):
        shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            return
        time.sleep(0.25)
    logger.warning("Screen check: could not remove %s", folder)


def _run_driver(config: dict, *, workspace: str, work: str, env: dict, timeout: float) -> dict:
    """Run the Node driver in the run sandbox; its report, or ``{"not_run": reason}``. Never raises."""
    from besser.spec_driven_agent.execution.sandbox import SandboxUnavailable, run_confined

    node = shutil.which("node")
    if not node:
        return {"not_run": "Node.js is not available"}
    config_path = os.path.join(work, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    try:
        result = run_confined([node, _DRIVER, config_path], workspace=workspace, writable=[work],
                              cwd=work, env=env, timeout=timeout)
    except SandboxUnavailable as exc:
        return {"not_run": str(exc)}
    except subprocess.TimeoutExpired:
        return {"not_run": f"timed out after {int(timeout)}s"}
    except OSError as exc:
        return {"not_run": f"could not be launched: {exc}"}
    for line in reversed((result.stdout or "").splitlines()):
        if line.startswith(_MARKER):
            try:
                return json.loads(line[len(_MARKER):])
            except ValueError:
                break
    tail = [line for line in (result.stderr or "").splitlines() if line.strip()]
    return {"not_run": (tail[-1] if tail else "no report")[:300]}


def _probe_project(output_dir: str, rel: str, browser: str, backend: str | None,
                   timeout: float) -> tuple[list[str], list[dict]]:
    from besser.spec_driven_agent.execution.process import _safe_subprocess_env
    from besser.utilities.workspace_fs import copytree_plain

    project = os.path.join(output_dir, rel)
    label = f"[{rel}]"
    site = _build_output(project)
    if site is None:
        return [not_run(f"{label} the build wrote no index.html to serve")], []
    routes = discover_routes(project)
    ports = _bundle_backend_ports(site)
    port = ports[0]
    work = tempfile.mkdtemp(prefix="besser_screens_")
    try:
        copytree_plain(site, os.path.join(work, "site"))
        env = _safe_subprocess_env()
        backend_config = None
        if backend:
            app_dir = os.path.join(work, "app")
            copytree_plain(backend, app_dir, _BACKEND_SKIP)
            env["DATABASE_URL"] = "sqlite:///" + os.path.join(work, "screens.db").replace("\\", "/")
            backend_config = {"argv": [sys.executable, "-u", "-c", _SERVE, str(port)], "cwd": app_dir,
                              "env": {}, "port": port, "ready_path": "/openapi.json",
                              "boot_ms": _BACKEND_BOOT_SECONDS * 1000}
        os.makedirs(os.path.join(work, "profile"))
        config = {
            "site": os.path.join(work, "site"), "browser": browser, "profile": os.path.join(work, "profile"),
            # The run sandbox leaves Chromium no namespace of its own to sandbox renderers in.
            "browser_args": ["--no-sandbox"] if sys.platform.startswith("linux") else [],
            "backend": backend_config, "own_ports": ports, "routes": routes.visit, "hash": routes.hash,
            "route_patterns": [pattern for pattern, _ in routes.declared],
            "max_screens": MAX_SCREENS, "screen_ms": SCREEN_SECONDS * 1000,
            "total_ms": int(max(timeout - _PROCESS_SLACK_SECONDS, 5) * 1000),
        }
        report = _run_driver(config, workspace=output_dir, work=work, env=env, timeout=timeout)
    except OSError as exc:
        report = {"not_run": f"could not prepare the check: {exc}"}
    finally:
        _remove(work)
    if "not_run" in report:
        return [not_run(f"{label} {report['not_run']}")], []
    backend_state = report.get("backend") or {}
    if backend_state.get("status") == "port_in_use":
        return [not_run(f"{label} port {backend_state.get('port')} the build calls is in use")], []
    screens = report.get("screens") or []
    if report.get("error") and not screens:
        return [not_run(f"{label} {report['error']}")], []
    issues, evidence = [], []
    if backend and backend_state.get("status") != "up":
        issues.append(f"{WARNING_PREFIX} {label} the backend did not start for the screen check "
                      f"({backend_state.get('status')}): {_first_line(backend_state.get('log'))}")
    for screen in screens:
        route = screen.get("route") or "/"
        declared, file = _matching_file(route, routes, rel)
        where = f"{label} route {route}" + (f" ({file})" if file else "")
        blockers, warnings = _screen_findings(screen, declared)
        if blockers:
            issues.append(f"{PREFIX} {where}: " + "; ".join(blockers))
        if warnings:
            issues.append(f"{WARNING_PREFIX} {where}: " + "; ".join(warnings))
        evidence.append({"project": rel, "route": route, "file": file, "rendered": screen.get("rendered") is True,
                         "blocker": bool(blockers), "seconds": round((screen.get("ms") or 0) / 1000, 1)})
    left = report.get("unvisited") or []
    if report.get("timed_out") or report.get("error"):
        reason = "the time limit" if report.get("timed_out") else _first_line(report.get("error"))
        issues.append(f"{WARNING_PREFIX} {label} {reason} stopped the check after {len(screens)} screen(s)"
                      + (f"; not opened: {_listed(left, 5)}" if left else ""))
    return issues, evidence


def collect_screen_report(output_dir: str, projects: list[str], *, timeout: float = TOTAL_SECONDS) -> dict:
    """Open the screens of each built frontend in ``projects`` (workspace-relative folders).

    ``{"issues": [...], "screens": [...], "seconds": float}``; ``screens`` is
    what :func:`screen_evidence` reads. Never raises.
    """
    started = time.monotonic()
    issues: list[str] = []
    screens: list[dict] = []
    if not AgentSettings().llm_screen_check:
        issues.append(not_run("turned off on this server"))
    elif not (browser := find_browser()):
        issues.append(not_run("no browser available"))
    else:
        from besser.spec_driven_agent.validation.constructibility import _fastapi_backends

        backends = _fastapi_backends(output_dir)
        for rel in projects:
            left = timeout - (time.monotonic() - started)
            if left < _MIN_SECONDS:
                issues.append(not_run(f"[{rel}] no time left in the run"))
                continue
            try:
                found, opened = _probe_project(output_dir, rel, browser, backends[0] if backends else None,
                                               min(TOTAL_SECONDS, left))
            except Exception as exc:  # a crash of this check must not stop Phase 3
                logger.warning("Screen check crashed on %s", rel, exc_info=True)
                found, opened = [not_run(f"[{rel}] it crashed: {type(exc).__name__}")], []
            issues.extend(found)
            screens.extend(opened)
    seconds = round(time.monotonic() - started, 1)
    logger.info("Screen check: %d screen(s) opened in %.1fs, %d with a blocker",
                len(screens), seconds, sum(1 for s in screens if s["blocker"]))
    return {"issues": issues, "screens": screens, "seconds": seconds}


def screen_evidence(report: dict | None) -> dict[str, bool]:
    """Page file (workspace-relative) -> whether every screen it serves opened, rendered and had no blocker.

    The "screen ran" evidence for the requirements report; a file absent from
    the map was not opened.
    """
    evidence: dict[str, bool] = {}
    for screen in (report or {}).get("screens") or []:
        if screen.get("file"):
            ok = screen["rendered"] and not screen["blocker"]
            evidence[screen["file"]] = evidence.get(screen["file"], True) and ok
    return evidence
