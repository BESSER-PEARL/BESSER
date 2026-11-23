"""Phase 3: open every screen of a built frontend in a headless browser.

The other checks build the frontend and drive the backend over HTTP, but none
renders a page: a bundle that builds and then dies on mount, a screen whose
data call answers 500, or a list that never shows the records the API holds
passes all of them. After the build has passed, this check creates records
through the app's own API (the create probe of ``constructibility.py``, on a
scratch SQLite database), boots the backend on that database, serves the build
output under its base path and opens each screen in a fresh tab of a headless
Chromium (``ui_probe_driver.mjs``, the DevTools protocol over Node's own
WebSocket). Nothing is clicked or typed.

Screens are the router's paths read from the source (nested routes resolved),
``/`` first, plus same-origin links found on the opened pages that match a
declared route, up to :data:`MAX_SCREENS`. Paths with parameters are opened only
when a page links to them.

Every finding is a ``screen warning:`` naming the route and, when the router
names it, the page's source file: an uncaught exception, a page that shows
nothing, React Router's error page, a 4xx or 5xx, a console error, and a
created record the API lists whose text no opened screen's page text or form
fields contain. One starting ``not checked:`` says what the check could not look
at. A screen that throws
and shows nothing is also returned as a blocker candidate, so the precision of
that rule can be measured from real runs before it blocks anything.

Optional by construction: with no browser, no Node.js or no sandbox the result
is one ``screens: not run (...)`` note.
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
from besser.spec_driven_agent.validation.acceptance import _blank_literals
from besser.spec_driven_agent.validation.frontend_forms import _DYNAMIC, _attr, _jsx_tags, _path_text
from besser.spec_driven_agent.validation.frontend_resolution import _read, _walk

logger = logging.getLogger(__name__)

WARNING_PREFIX = "screen warning:"
NOT_RUN_PREFIX = "screens: not run"
# Opens a warning that says what the check could not look at, not what the app does.
COVERAGE = "not checked:"
# The trace key of screens that threw and showed nothing (see the module docstring).
BLOCKER_CANDIDATES = "screen_blocker_candidates"

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
_ROUTE_TOKEN = re.compile(r"<Route\b|</Route\s*>")
_LITERAL = re.compile(r"""\s*(['"`])([^'"`]*)\1\s*""")
_BASENAME = re.compile(r"""\bbasename\s*[=:]\s*\{?\s*(['"`])(/[^'"`]*)\1""")
_COMPONENT = re.compile(r"<\s*([A-Z][\w$]*)")
_IDENTIFIER = re.compile(r"\s*([A-Z][\w$]*)\s*$")
_LOCAL_ORIGIN = re.compile(r"https?://(?:localhost|127\.0\.0\.1)(?::(\d{2,5}))?(?![\w.])")
_ASSET_REF = re.compile(r"""(?:src|href)\s*=\s*["'](/[^"'?#]+)""")
_LOCAL_HOSTS = ("localhost", "127.0.0.1")
# A backend failing because the sandbox has no network: Python's and the HTTP clients' wording.
_NETWORK_ERROR = re.compile(
    r"Name or service not known|Temporary failure in name resolution|getaddrinfo failed|nodename nor servname"
    r"|Network is unreachable|No route to host|ConnectError|ConnectTimeout|NewConnectionError"
    r"|Failed to establish a new connection|URLError|ConnectionRefusedError|gaierror", re.I)
# The create probe's own text samples (``constructibility._sample``): distinctive, so a screen
# showing one shows that record.
_SAMPLE_TEXT = re.compile(r"^(?:p\d+-|probe-\d+-)")

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


# ------------------------------------------------------------------ routes


class _Routes(NamedTuple):
    visit: list[str]                  # literal paths, opened in this order
    declared: list[tuple[str, str]]   # (regex source, page file or "") for every declared path
    hash: bool
    basename: str | None              # the router's literal basename, if it declares one
    computed: bool                    # some route's path is computed, so ``declared`` is incomplete


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


def _route_path(expr: str | None, parent: str) -> str | None:
    """A route's full path: a relative literal is joined to its parent route's path."""
    if not expr:
        return None
    literal = _LITERAL.fullmatch(expr)
    if literal and not literal.group(2).startswith("/"):
        expr = "'" + (parent.rstrip("/") + "/" + literal.group(2)) + "'"
    return _path_text(expr)


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


def _jsx_routes(text: str, full: str, project: str, add) -> bool:
    """``<Route>`` elements, a nested element's relative path joined to its ancestors'.

    True when a route's path is computed (``path={entity.path}``).
    """
    tags = {start: attrs for start, attrs, _ in _jsx_tags(text, "Route")}
    stack: list[str] = []
    computed = False
    for token in _ROUTE_TOKEN.finditer(text):
        if token.group().startswith("</"):
            if stack:
                stack.pop()
            continue
        attrs = tags.get(token.start(), "")
        parent = stack[-1] if stack else ""
        expr = _attr(attrs, "path")
        path = _route_path(expr, parent)
        computed = computed or (expr is not None and path is None)
        element = _element_file(_attr(attrs, "element") or _attr(attrs, "component"), text, full, project)
        if path:
            add(path, element)
        elif re.search(r"(?<![\w-])index(?![\w-])", attrs):
            add(parent or "/", element)  # an index route renders its parent's path
        if not attrs.rstrip().endswith("/"):
            stack.append(path or parent)
    return computed


def _object_routes(text: str, full: str, project: str, add) -> None:
    """Route objects (``createBrowserRouter``), a child's relative path joined to its parents'."""
    matches = list(_OBJECT_PATH.finditer(text))
    if not matches:
        return
    blanked = _blank_literals(text)
    starts = {match.start(): match for match in matches}
    enclosing: dict[int, tuple[int, ...]] = {}
    stack: list[int] = []
    for index, char in enumerate(blanked):
        if index in starts:
            enclosing[index] = tuple(stack)
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            stack.pop()
    owner = {braces[-1]: start for start, braces in enclosing.items() if braces}
    full_path: dict[int, str] = {}
    for start in sorted(enclosing):
        match = starts[start]
        ancestors = [owner[brace] for brace in enclosing[start][:-1] if brace in owner and owner[brace] in full_path]
        parent = full_path[ancestors[-1]] if ancestors else ""
        path = _route_path(match.group(0).split(":", 1)[1], parent)
        if not path:
            continue
        full_path[start] = path
        element = _OBJECT_ELEMENT.search(text, match.end(), match.end() + 300)
        add(path, _element_file((element.group(1) or element.group(2)) if element else None, text, full, project))


def discover_routes(project: str) -> _Routes:
    """The screens a frontend's router declares, read from its source."""
    declared: dict[str, tuple[str, str]] = {}
    hash_router = False
    basename = None
    computed = False

    def add(path: str, file: str) -> None:
        if path in ("*", "/*") or not path.startswith("/"):
            return  # a catch-all is the not-found page
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
            found = _BASENAME.search(text)
            basename = basename or (found.group(2).rstrip("/") if found else None)
            computed = _jsx_routes(text, full, project, add) or computed
            if _OBJECT_ROUTER.search(text):
                _object_routes(text, full, project, add)
    visit = ["/"] + sorted(path for path in declared
                           if path != "/" and _DYNAMIC not in path and ":" not in path and "*" not in path)
    return _Routes(visit, list(declared.values()), hash_router, basename, computed)


# ------------------------------------------------------------------ the build


def _build_output(project: str) -> str | None:
    """The build's output folder: the newest of dist/build/out holding an index.html."""
    found = [(os.path.getmtime(os.path.join(project, name, "index.html")), os.path.join(project, name))
             for name in _BUILD_OUTPUT_DIRS if os.path.isfile(os.path.join(project, name, "index.html"))]
    return max(found)[1] if found else None


def _build_base(site: str) -> str:
    """The base path the build serves its files under (``vite build --base /app/``): ``/`` or ``/app/``.

    Read from the absolute asset URLs in its index.html: the prefix whose rest is a built file.
    """
    counts: Counter = Counter()
    for ref in _ASSET_REF.findall(_read(os.path.join(site, "index.html"))):
        parts = ref.split("/")
        for index in range(1, len(parts)):
            if os.path.isfile(os.path.join(site, *parts[index:])):
                counts["/".join(parts[:index]) + "/"] += 1
                break
    return counts.most_common(1)[0][0] if counts else "/"


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


# --------------------------------------------------------------- the data pass


def _payload_texts(payload) -> list[str]:
    """The create probe's own text samples in a payload it sent, lower-cased."""
    if not isinstance(payload, dict):
        return []
    return [value.lower() for value in payload.values()
            if isinstance(value, str) and len(value) >= 6 and _SAMPLE_TEXT.match(value)]


def _seed(backend: str, database: str, output_dir: str, domain_model, timeout: float) -> tuple[dict, dict]:
    """Create records through the app's API with the create probe, into ``database``.

    ``({entity: {"path": create path, "texts": [...]}}, {entity: why not checked})``.
    """
    from besser.spec_driven_agent.execution.process import _safe_subprocess_env
    from besser.spec_driven_agent.validation.constructibility import _model_actions, _run_probe, abstract_entities

    report = _run_probe(backend, _safe_subprocess_env(), _model_actions(domain_model),
                        abstract_entities(domain_model), workspace=output_dir, database=database,
                        timeout=timeout)
    if report.get("boot") != "ok":
        return {}, {"*": f"the create probe could not run the app ({_first_line(report.get('error') or report.get('boot'))})"}
    created, skipped = {}, {}
    for entity, entry in sorted((report.get("entities") or {}).items()):
        attempts = [a for a in entry.get("attempts") or [] if a.get("outcome") == "created"]
        texts = sorted({text for attempt in attempts for text in _payload_texts(attempt.get("payload"))})
        if not attempts:
            skipped[entity] = _refusal(entry)
        elif "{" in str(entry.get("path")):
            skipped[entity] = "its list route needs a parent record"
        elif not texts:
            skipped[entity] = "it has no text field to look for"
        else:
            created[entity] = {"path": entry["path"], "texts": texts}
    return created, skipped


def _refusal(entry: dict) -> str:
    """Why the create probe has no record of an entity: its guessed sample, not the app, is in question."""
    needs = sorted({str(pair[1]) for pair in entry.get("unresolved") or [] if len(pair) > 1})
    if needs:
        return f"the check could not first create the {' and '.join(needs)} it needs"
    status = next((a.get("status") for a in reversed(entry.get("attempts") or []) if a.get("status")), None)
    return "the check's sample was refused" + (f" ({status})" if status else "")


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _route_serves(route: str, path: str) -> bool:
    from besser.spec_driven_agent.planning.mutation_inventory import route_serves

    return route_serves(route, path)


def _data_findings(label: str, created: dict, skipped: dict, listed: dict, screens: list[dict]) -> list[str]:
    """Records the API lists whose text no opened screen's page text or form fields contain."""
    issues, unchecked = [], dict(skipped)
    for entity, record in sorted(created.items()):
        path = record["path"]
        state = listed.get(entity) or {}
        if not state.get("listed"):
            unchecked[entity] = (f"GET {path} answered {state['status']}" if state.get("status") not in (None, 200)
                                 else f"GET {path} does not return it")
            continue
        if any(set(record["texts"]) & set(screen.get("found") or []) for screen in screens):
            continue
        loading = [screen["route"] for screen in screens
                   if any(call.get("method") == "GET" and _route_serves(path, call.get("path", ""))
                          for call in screen.get("calls") or [])]
        where = (f"screens that load GET {path}: {_listed(loading, 5)}" if loading
                 else f"no opened screen loads GET {path}")
        issues.append(f"{WARNING_PREFIX} {label} created {_article(entity)} {entity} through the API (POST {path}) "
                      f"and the API lists it, but its text ({_listed(record['texts'], 2)}) was not found in the "
                      f"page text or form fields of any of the {len(screens)} opened screen(s); {where}")
    if unchecked:
        issues.append(f"{WARNING_PREFIX} {label} {COVERAGE} records on the screens: "
                      + "; ".join(f"{entity} ({why})" if entity != "*" else why
                                  for entity, why in sorted(unchecked.items())))
    return issues


# ------------------------------------------------------------------ findings


def _shown_url(url: str) -> str:
    """A request URL without the check's own origin: ``/books/?page=1``."""
    return re.sub(r"^https?://[^/]+", "", url or "") or "/"


def _first_line(text) -> str:
    return (str(text or "").strip().splitlines() or [""])[0][:200]


def _is_local(url: str) -> bool:
    host = re.match(r"^https?://([^/:]+)", url or "")
    return bool(host) and host.group(1) in _LOCAL_HOSTS


def _listed(items: list[str], limit: int = 3) -> str:
    items = list(dict.fromkeys(items))
    return ", ".join(items[:limit]) + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _screen_findings(screen: dict, sample_records: bool) -> tuple[list[str], list[str], str]:
    """One opened screen's warning parts, the parts that say what was not checked, and why it is a
    blocker candidate ("" when not). ``sample_records``: the check created records before opening it."""
    # The browser asks for /favicon.ico on its own; the app never requested it.
    requests = [r for r in screen.get("requests") or [] if not _shown_url(r.get("url")).startswith("/favicon.ico")]
    exceptions = screen.get("exceptions") or []
    parts, coverage = [], []
    seconds = round((screen.get("ms") or 0) / 1000, 1)
    backend_down = any(r.get("own") and r.get("unreachable") for r in requests)
    script_missing = any(r.get("type") == "Script" and "status" not in r and not _is_local(r.get("url", ""))
                         for r in requests)
    cause = (" (the app's backend was not reachable)" if backend_down
             else " (an external script could not load without a network)" if script_missing else "")
    shown = screen.get("rendered") is not False
    thrown = [e for e in exceptions if not e.get("promise")]
    candidate = ""
    if thrown:
        first = _first_line(thrown[0].get("text")) + (f" (+{len(thrown) - 1} more)" if len(thrown) > 1 else "")
        if shown:
            parts.append(f"an uncaught {first} was thrown while the page loaded; the page still rendered")
        else:
            parts.append(f"an uncaught {first} was thrown and the page showed nothing{cause}")
            candidate = first
    elif not shown:
        data = "the check's sample records" if sample_records else "an empty database"
        parts.append(f"the page showed nothing with no signed-in user and {data} "
                     f"(no text and no control after {seconds}s){cause}" if screen.get("idle")
                     else f"nothing shown yet after {seconds}s with requests still pending")
    elif not screen.get("idle") and not screen.get("error"):
        # A request still open, or a fetch loop that never stops (the count tells which).
        parts.append(f"the page did not settle within {seconds}s ({screen.get('request_count', 0)} requests)")
    if screen.get("error_page") is not None:
        parts.append("the page shows React Router's error page"
                     + (f": {_first_line(screen['error_page'])}" if screen["error_page"] else ""))
    sandbox_network = bool(_NETWORK_ERROR.search(screen.get("backend_log") or ""))
    server_errors = [f"{r.get('method')} {_shown_url(r.get('url'))} answered {r['status']}"
                     for r in requests if isinstance(r.get("status"), int) and r["status"] >= 500
                     and r.get("own") and not r.get("unreachable")]
    if server_errors and sandbox_network:
        coverage.append(f"{_listed(server_errors)} in the sandbox (no network there; the backend log shows a "
                        "connection error)")
    elif server_errors:
        parts.append(f"{_listed(server_errors)} from the app's backend")
    refused = [f"{r.get('method')} {_shown_url(r.get('url'))} answered {r['status']}"
               for r in requests if isinstance(r.get("status"), int) and 400 <= r["status"] < 500]
    if refused:
        parts.append(_listed(refused))
    others = [f"{r.get('method')} {_shown_url(r.get('url'))} answered {r['status']}"
              for r in requests if isinstance(r.get("status"), int) and r["status"] >= 500 and not r.get("own")]
    if others:
        parts.append(_listed(others))
    unreachable = [f"{r.get('method')} {_shown_url(r.get('url'))}" for r in requests
                   if r.get("own") and r.get("unreachable")]
    if unreachable:
        parts.append(f"the backend was not reachable for {_listed(unreachable)}")
    unanswered = [f"{r.get('method')} {r.get('url')}" for r in requests
                  if "error" in r and _is_local(r.get("url", "")) and not r.get("own")]
    if unanswered:
        parts.append(f"nothing answered {_listed(unanswered)}")
    rejections = [e for e in exceptions if e.get("promise")]
    if rejections:
        parts.append(f"unhandled promise rejection: {_first_line(rejections[0].get('text'))}"
                     + (f" (+{len(rejections) - 1} more)" if len(rejections) > 1 else ""))
    console = screen.get("console_errors") or []
    if console:
        parts.append(f"{len(console)} console error(s), first: {_first_line(console[0])}")
    if screen.get("error"):
        coverage.append(f"the check could not finish this screen: {_first_line(screen['error'])}")
    return parts, coverage, candidate


def _matching_file(route: str, routes: _Routes, rel: str) -> str:
    """The workspace-relative page file serving ``route`` ("" if unknown)."""
    for pattern, page in routes.declared:
        if page and re.match(pattern, route):
            return page if rel == "." else f"{rel}/{page}"
    return ""


# ------------------------------------------------------------------ running it


def _remove(folder: str) -> None:
    """Delete the scratch folder; on Windows a closing browser can hold its profile for a moment."""
    for _ in range(20):
        shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            return
        time.sleep(0.25)
    logger.warning("Screen check: could not remove %s", folder)


def _last_report(stdout: str) -> dict | None:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_MARKER):
            try:
                return json.loads(line[len(_MARKER):])
            except ValueError:
                continue
    return None


def _run_driver(config: dict, *, workspace: str, work: str, env: dict, timeout: float) -> dict:
    """Run the Node driver in the run sandbox; its report, or ``{"not_run": reason}``. Never raises.

    The driver prints its report after every screen, so a run stopped midway keeps
    the screens already opened (``stopped`` says why).
    """
    from besser.spec_driven_agent.execution.process import timeout_cause
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
    except subprocess.TimeoutExpired as exc:  # also the run's own stop (a subclass)
        why = timeout_cause(exc, f"timed out after {int(timeout)}s")
        partial = _last_report(exc.output if isinstance(exc.output, str) else "")
        return {**partial, "stopped": why} if partial and partial.get("screens") else {"not_run": why}
    except OSError as exc:
        return {"not_run": f"could not be launched: {exc}"}
    report = _last_report(result.stdout)
    if report is not None:
        return report
    tail = [line for line in (result.stderr or "").splitlines() if line.strip()]
    return {"not_run": (tail[-1] if tail else "no report")[:300]}


def _probe_project(output_dir: str, rel: str, browser: str, backend: str | None, domain_model,
                   deadline: float) -> dict:
    from besser.spec_driven_agent.execution.process import _safe_subprocess_env
    from besser.utilities.workspace_fs import copytree_plain

    project = os.path.join(output_dir, rel)
    label = f"[{rel}]"
    result = {"issues": [], "screens": [], "candidates": []}
    site = _build_output(project)
    if site is None:
        result["issues"].append(not_run(f"{label} the build wrote no index.html to serve"))
        return result
    routes = discover_routes(project)
    ports = _bundle_backend_ports(site)
    base = _build_base(site)
    work = tempfile.mkdtemp(prefix="besser_screens_")
    created, skipped = {}, {}
    try:
        copytree_plain(site, os.path.join(work, "site"))
        env = _safe_subprocess_env()
        backend_config = None
        if backend:
            app_dir = os.path.join(work, "app")
            copytree_plain(backend, app_dir, _BACKEND_SKIP)
            database = os.path.join(work, "screens.db")
            created, skipped = _seed(backend, database, output_dir, domain_model,
                                     max(5.0, (deadline - time.monotonic()) / 2))
            env["DATABASE_URL"] = "sqlite:///" + database.replace("\\", "/")
            backend_config = {"argv": [sys.executable, "-u", "-c", _SERVE, str(ports[0])], "cwd": app_dir,
                              "env": {}, "port": ports[0], "ready_path": "/openapi.json",
                              "boot_ms": _BACKEND_BOOT_SECONDS * 1000,
                              "seed": [{"entity": entity, "path": record["path"], "texts": record["texts"]}
                                       for entity, record in created.items()]}
        os.makedirs(os.path.join(work, "profile"))
        timeout = deadline - time.monotonic()
        config = {
            "site": os.path.join(work, "site"), "base": base,
            "prefix": routes.basename if routes.basename is not None else base.rstrip("/"),
            "browser": browser, "profile": os.path.join(work, "profile"),
            # The run sandbox leaves Chromium no namespace of its own to sandbox renderers in.
            "browser_args": ["--no-sandbox"] if sys.platform.startswith("linux") else [],
            "backend": backend_config, "own_ports": ports, "routes": routes.visit, "hash": routes.hash,
            # Links are followed only to declared routes, unless some route is computed.
            "route_patterns": [] if routes.computed else [pattern for pattern, _ in routes.declared],
            "markers": sorted({text for record in created.values() for text in record["texts"]}),
            "max_screens": MAX_SCREENS, "screen_ms": SCREEN_SECONDS * 1000,
            "total_ms": int(max(timeout - _PROCESS_SLACK_SECONDS, 5) * 1000),
        }
        report = _run_driver(config, workspace=output_dir, work=work, env=env, timeout=timeout)
    except OSError as exc:
        report = {"not_run": f"could not prepare the check: {exc}"}
    finally:
        _remove(work)
    issues = result["issues"]
    if "not_run" in report:
        issues.append(not_run(f"{label} {report['not_run']}"))
        return result
    backend_state = report.get("backend") or {}
    if backend_state.get("status") == "port_in_use":
        issues.append(not_run(f"{label} port {backend_state.get('port')} the build calls is in use"))
        return result
    screens = report.get("screens") or []
    if report.get("error") and not screens:
        issues.append(not_run(f"{label} {report['error']}"))
        return result
    if backend and backend_state.get("status") != "up":
        issues.append(f"{WARNING_PREFIX} {label} {COVERAGE} the backend did not start for the screen check "
                      f"({backend_state.get('status')}): {_first_line(backend_state.get('log'))}")
    for screen in screens:
        route = screen.get("route") or "/"
        file = _matching_file(route, routes, rel)
        where = f"{label} route {route}" + (f" ({file})" if file else "")
        parts, coverage, candidate = _screen_findings(screen, bool(created))
        if parts:
            issues.append(f"{WARNING_PREFIX} {where}: " + "; ".join(parts))
        if coverage:
            issues.append(f"{WARNING_PREFIX} {where}: {COVERAGE} " + "; ".join(coverage))
        if candidate:
            result["candidates"].append({"project": rel, "route": route, "file": file, "exception": candidate})
        result["screens"].append({"project": rel, "route": route, "file": file,
                                  "rendered": screen.get("rendered") is True,
                                  "seconds": round((screen.get("ms") or 0) / 1000, 1)})
    if backend and screens:
        issues.extend(_data_findings(label, created, skipped, backend_state.get("seeded") or {}, screens))
    left = report.get("unvisited") or []
    stop = report.get("stopped") or ("the time limit" if report.get("timed_out") else report.get("error"))
    if stop:
        issues.append(f"{WARNING_PREFIX} {label} {COVERAGE} {_first_line(stop)} stopped the check after {len(screens)} "
                      "screen(s)" + (f"; not opened: {_listed(left, 5)}" if left else ""))
    return result


def collect_screen_report(output_dir: str, projects: list[str], *, timeout: float = TOTAL_SECONDS,
                          domain_model=None) -> dict:
    """Open the screens of each built frontend in ``projects`` (workspace-relative folders).

    One deadline, ``min(timeout, TOTAL_SECONDS)``, covers every project.
    ``{"issues": [...], "screens": [...], "candidates": [...], "seconds": float}``. Never raises.
    """
    started = time.monotonic()
    deadline = started + min(timeout, TOTAL_SECONDS)
    report = {"issues": [], "screens": [], "candidates": []}
    if not AgentSettings().llm_screen_check:
        report["issues"].append(not_run("turned off on this server"))
    elif not (browser := find_browser()):
        report["issues"].append(not_run("no browser available"))
    else:
        from besser.spec_driven_agent.validation.constructibility import _fastapi_backends

        backends = _fastapi_backends(output_dir)
        for rel in projects:
            if deadline - time.monotonic() < _MIN_SECONDS:
                report["issues"].append(not_run(f"[{rel}] no time left in the run"))
                continue
            try:
                found = _probe_project(output_dir, rel, browser, backends[0] if backends else None,
                                       domain_model, deadline)
            except Exception as exc:  # a crash of this check must not stop Phase 3
                logger.warning("Screen check crashed on %s", rel, exc_info=True)
                found = {"issues": [not_run(f"[{rel}] it crashed: {type(exc).__name__}")], "screens": [],
                         "candidates": []}
            for key in report:
                report[key].extend(found[key])
    report["seconds"] = round(time.monotonic() - started, 1)
    logger.info("Screen check: %d screen(s) opened in %.1fs, %d blocker candidate(s)",
                len(report["screens"]), report["seconds"], len(report["candidates"]))
    return report
