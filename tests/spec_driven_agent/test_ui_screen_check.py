"""Phase 3 opens every screen of a built frontend in a headless browser.

No validation step rendered a page: a frontend that builds and then throws on
mount, stays blank, gets a 500 from its own backend, or never shows the records
the API holds passed every check. ``ui_probe`` creates records through the
app's API, boots the backend, serves the build output and opens each declared
screen (plus the links pages lead to) headless. Every finding is a warning; a
screen that threw and showed nothing is also a blocker candidate for the trace.

The fixture under ``fixtures/screen_app`` is a committed stand-in for a built
app and its backend (no npm install): one screen per behaviour, healthy ones
included. The browser tests drive a Chromium-family browser already installed on
the machine and skip without one, like the Node-dependent tests skip without Node.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time

import pytest

from besser.spec_driven_agent.validation import ui_probe
from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.ui_probe import collect_screen_report, discover_routes, not_run
from besser.spec_driven_agent.validation.user_wording import user_sentence

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "screen_app")
HAVE_BROWSER = ui_probe.find_browser() is not None and shutil.which("node") is not None
needs_browser = pytest.mark.skipif(not HAVE_BROWSER, reason="needs Node.js and a Chromium-family browser")

# Every behaviour of the fixture app, by route and the page file declared for it.
LAB = (("/", "OrderList"), ("/orders/:id", "OrderDetail"), ("/throws", "Broken"), ("/empty", "Blank"),
       ("/server-error", "Report"), ("/missing", "CustomerList"), ("/delayed", "Delayed"), ("/items", "Items"),
       ("/portal", "Portal"), ("/profile", "Profile"), ("/protected", "Protected"), ("/benign", "Benign"),
       ("/quote", "Quote"), ("/webgl", "Scene"), ("/error-page", "Failing"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _copy(root) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    shutil.copytree(FIXTURE, root)


def _declare(root, *routes: tuple[str, str], basename: str = "") -> None:
    """Replace the router with one declaring only ``routes`` (path, page)."""
    lines = [f'import {page} from "./pages/{page}";' for _, page in routes]
    lines += ["export default () => (", f'  <BrowserRouter{f" basename={basename!r}" if basename else ""}>',
              "  <Routes>"]
    lines += [f'    <Route path="{path}" element={{<{page} />}} />' for path, page in routes]
    lines += ["  </Routes>", "  </BrowserRouter>", ");", ""]
    (root / "frontend" / "src" / "App.jsx").write_text("\n".join(lines), encoding="utf-8")


def _run(root, monkeypatch, **kwargs) -> dict:
    """The check on ``root``, its backend on a free port instead of the bundle's default."""
    port = _free_port()
    monkeypatch.setattr(ui_probe, "_bundle_backend_ports", lambda site: [port])
    return collect_screen_report(str(root), ["frontend"], **kwargs)


def _by_route(issues: list[str], route: str) -> list[str]:
    return [issue for issue in issues if f" route {route} " in issue or f" route {route}:" in issue]


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """One run over every screen of the fixture (the browser start is the slow part)."""
    if not HAVE_BROWSER:
        pytest.skip("needs Node.js and a Chromium-family browser")
    root = tmp_path_factory.mktemp("lab") / "workspace"
    _copy(root)
    _declare(root, *LAB)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ui_probe, "SCREEN_SECONDS", 5)
        return _run(root, patch)


# ----------------------------------------------------- every finding is a warning


@needs_browser
def test_nothing_the_screen_check_reports_blocks(lab):
    assert lab["issues"]
    assert all(issue.startswith("screen warning:") for issue in lab["issues"]), lab["issues"]
    assert {_classify_issue(issue).severity for issue in lab["issues"]} == {"warning"}


@needs_browser
def test_a_screen_that_throws_and_shows_nothing_is_a_warning_and_a_blocker_candidate(lab):
    assert _by_route(lab["issues"], "/throws") == [
        "screen warning: [frontend] route /throws (frontend/src/pages/Broken.jsx): an uncaught TypeError: "
        "Cannot read properties of undefined (reading 'length') was thrown and the page showed nothing"]
    assert lab["candidates"] == [{"project": "frontend", "route": "/throws", "file": "frontend/src/pages/Broken.jsx",
                                  "exception": "TypeError: Cannot read properties of undefined (reading 'length')"}]


@needs_browser
def test_an_exception_on_a_page_that_rendered_is_worded_as_such(lab):
    assert _by_route(lab["issues"], "/benign") == [
        "screen warning: [frontend] route /benign: an uncaught Error: telemetry endpoint missing was thrown "
        "while the page loaded; the page still rendered"]


@needs_browser
def test_an_empty_screen_says_the_database_is_empty(lab):
    for route in ("/empty", "/items", "/protected"):
        found = _by_route(lab["issues"], route)
        assert len(found) == 1 and "the page showed nothing with an empty database and no signed-in user" in found[0]
    assert "GET /api/me answered 404" in _by_route(lab["issues"], "/profile")[0]


@needs_browser
def test_late_renders_portals_and_webgl_pages_are_healthy(lab):
    for route in ("/", "/orders/7", "/delayed", "/portal", "/webgl"):
        assert _by_route(lab["issues"], route) == [], route


@needs_browser
def test_a_500_is_told_apart_from_a_backend_that_needs_a_network(lab):
    assert _by_route(lab["issues"], "/server-error") == [
        "screen warning: [frontend] route /server-error (frontend/src/pages/Report.jsx): "
        "GET /api/report answered 500 from the app's backend"]
    assert _by_route(lab["issues"], "/quote") == [
        "screen warning: [frontend] route /quote: GET /api/quote answered 500 in the sandbox "
        "(no network there; the backend log shows a connection error)"]


@needs_browser
def test_react_routers_error_page_is_reported(lab):
    found = _by_route(lab["issues"], "/error-page")
    assert len(found) == 1
    assert "the page shows React Router's error page: rows is undefined" in found[0]
    assert "1 console error(s), first: TypeError: rows is undefined" in found[0]


@needs_browser
def test_a_404_during_load_is_reported(lab):
    assert _by_route(lab["issues"], "/missing") == [
        "screen warning: [frontend] route /missing (frontend/src/pages/CustomerList.jsx): "
        "GET /api/customers answered 404"]


@needs_browser
def test_a_build_served_under_a_base_path_loads(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _copy(root)
    index = root / "frontend" / "dist" / "index.html"
    index.write_text(index.read_text().replace('src="/assets/app.js"', 'src="/app/assets/app.js"'))
    _declare(root, ("/", "OrderList"), ("/orders/:id", "OrderDetail"), ("/admin", "AdminLayout"), basename="/app")
    report = _run(root, monkeypatch)
    assert report["issues"] == []
    assert [screen["route"] for screen in report["screens"]] == ["/", "/admin", "/orders/7"]


@needs_browser
def test_a_healthy_app_has_no_findings(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _copy(root)  # the committed router, with a nested relative route
    report = _run(root, monkeypatch, timeout=120)
    opened = {screen["route"]: screen["file"] for screen in report["screens"]}
    assert opened["/admin/audit"] == "frontend/src/pages/AuditLog.jsx"
    for route in ("/", "/orders/7", "/admin", "/admin/audit"):
        assert _by_route(report["issues"], route) == []


# ----------------------------------------------- records the API lists, on screen


@pytest.fixture
def shop(tmp_path):
    """A generated backend (Order, Product, Customer, Tag) and a frontend that lists orders by
    title, products by price only, and customers nowhere."""
    from besser.BUML.metamodel.structural import Class, DomainModel, FloatType, IntegerType, Property, StringType
    from besser.generators.backend import BackendGenerator

    pytest.importorskip("fastapi")
    root = tmp_path / "workspace"
    model = DomainModel(name="Shop", types={
        Class(name="Order", attributes={Property(name="title", type=StringType)}),
        Class(name="Product", attributes={Property(name="label", type=StringType),
                                          Property(name="price", type=FloatType)}),
        Class(name="Customer", attributes={Property(name="fullName", type=StringType)}),
        Class(name="Tag", attributes={Property(name="weight", type=IntegerType)})})
    BackendGenerator(model=model, output_dir=str(root / "backend")).generate()
    (root / "frontend" / "dist" / "assets").mkdir(parents=True)
    (root / "frontend" / "src").mkdir()
    (root / "frontend" / "dist" / "index.html").write_text(
        '<!doctype html><script type="module" src="/assets/app.js"></script><div id="root"></div>')
    (root / "frontend" / "dist" / "assets" / "app.js").write_text(
        'const root = document.getElementById("root");\n'
        'const route = location.pathname.replace(/\\/+$/, "") || "/";\n'
        'const show = (text) => { const p = document.createElement("p"); p.textContent = text; root.append(p); };\n'
        'const list = (path, field) => fetch(path).then((r) => r.json())\n'
        '  .then((rows) => rows.forEach((row) => show(String(row[field]))));\n'
        'if (route === "/") show("Shop");\n'
        'else if (route === "/orders") { show("Orders"); list("/order/", "title"); }\n'
        'else if (route === "/products") { show("Products"); list("/product/", "price"); }\n')
    (root / "frontend" / "src" / "App.jsx").write_text(
        '<Routes><Route path="/" element={<p />} /><Route path="/orders" element={<p />} />'
        '<Route path="/products" element={<p />} /></Routes>\n')
    return root


@needs_browser
def test_a_created_record_no_opened_screen_shows_is_reported(shop, monkeypatch):
    report = _run(shop, monkeypatch)
    assert report["issues"] == [
        "screen warning: [frontend] created a Customer through the API (POST /customer/) and the API lists it, "
        "but none of its text (p4-customer-fullname) appears on any of the 3 opened screen(s); "
        "no opened screen loads GET /customer/",
        "screen warning: [frontend] created a Product through the API (POST /product/) and the API lists it, "
        "but none of its text (p2-product-label) appears on any of the 3 opened screen(s); "
        "screens that load GET /product/: /products",
        "screen warning: [frontend] records not checked on the screens: Tag (it has no text field to look for)",
    ]
    assert user_sentence(report["issues"][0]) == (
        "A Customer created through the app's API does not appear on any screen that was opened.")


def test_a_record_the_api_does_not_list_is_not_checked():
    created = {"Order": {"path": "/order/", "texts": ["p1-order-title"]}}
    screens = [{"route": "/", "found": [], "calls": []}]
    assert ui_probe._data_findings("[web]", created, {}, {"Order": {"status": 401, "listed": False}}, screens) == [
        "screen warning: [web] records not checked on the screens: Order (GET /order/ answered 401)"]


# ------------------------------------------------------------- optional, bounded


def test_no_browser_is_one_not_run_note(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_probe, "find_browser", lambda: None)
    report = collect_screen_report(str(tmp_path), ["frontend"])
    assert report["issues"] == ["screens: not run (no browser available)"]
    assert _classify_issue(report["issues"][0]).severity == "warning"
    assert user_sentence(report["issues"][0]) == (
        "Not run: opening the app's screens in a browser (no browser available), so they are not covered.")


def test_the_setting_turns_the_check_off(tmp_path, monkeypatch):
    monkeypatch.setenv("BESSER_LLM_SCREEN_CHECK", "0")
    monkeypatch.setattr(ui_probe, "find_browser", lambda: pytest.fail("looked for a browser while off"))
    assert collect_screen_report(str(tmp_path), ["frontend"])["issues"] == [
        not_run("turned off on this server")]


@needs_browser
def test_each_screen_is_time_boxed(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _copy(root)
    _declare(root, ("/", "OrderList"), ("/slow", "Report"), ("/throws", "Broken"))
    monkeypatch.setattr(ui_probe, "SCREEN_SECONDS", 3)
    report = _run(root, monkeypatch)
    # The hanging request ended that screen, not the check: the next screen was still opened.
    assert [screen["route"] for screen in report["screens"]] == ["/", "/slow", "/throws"]
    slow = _by_route(report["issues"], "/slow")
    assert len(slow) == 1 and "the page did not settle within" in slow[0]


@needs_browser
def test_one_deadline_bounds_the_whole_check(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _copy(root)
    _declare(root, ("/", "OrderList"), ("/slow", "Report"), ("/throws", "Broken"))
    monkeypatch.setattr(ui_probe, "SCREEN_SECONDS", 600)
    report = _run(root, monkeypatch, timeout=30)
    assert [screen["route"] for screen in report["screens"]] == ["/", "/slow"]
    assert [issue for issue in report["issues"] if "stopped the check" in issue] == [
        "screen warning: [frontend] the time limit stopped the check after 2 screen(s); not opened: /throws"]


def test_every_frontend_shares_one_deadline(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ui_probe, "find_browser", lambda: "chromium")
    monkeypatch.setattr(ui_probe, "_probe_project", lambda output, rel, browser, backend, model, deadline:
                        seen.append(deadline) or {"issues": [], "screens": [], "candidates": []})
    started = time.monotonic()
    collect_screen_report(str(tmp_path), ["web", "admin"], timeout=10_000)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0] - started <= ui_probe.TOTAL_SECONDS + 1


# -------------------------------------------------------------------- route reading


def test_routes_are_read_from_jsx_object_routers_nesting_and_lazy_pages(tmp_path):
    src = tmp_path / "src"
    (src / "pages").mkdir(parents=True)
    for page in ("Books", "BookForm", "Authors", "Layout", "Loans"):
        (src / "pages" / f"{page}.tsx").write_text(f"export default function {page}() {{ return null; }}\n")
    (src / "App.tsx").write_text(
        'import Books from "./pages/Books";\n'
        'import Layout from "./pages/Layout";\n'
        'import Loans from "./pages/Loans";\n'
        'const BookForm = lazy(() => import("./pages/BookForm"));\n'
        '<BrowserRouter basename="/library">\n'
        "<Route path='/books' element={<Guard><Books /></Guard>} />\n"
        '<Route path="/books/:id/edit" element={<BookForm />} />\n'
        '<Route path="/" element={<Layout />}>\n'
        '  <Route path="loans" element={<Loans />} />\n'
        "</Route>\n"
        '<Route path="*" element={<p>Not found</p>} />\n', encoding="utf-8")
    (src / "router.ts").write_text(
        'import { Authors } from "./pages/Authors";\n'
        "export const router = createHashRouter([{ path: '/catalogue', children: [\n"
        "  { path: 'authors', element: <Authors /> }] }]);\n", encoding="utf-8")
    routes = discover_routes(str(tmp_path))
    assert routes.visit == ["/", "/books", "/catalogue", "/catalogue/authors", "/loans"]
    assert routes.hash is True and routes.basename == "/library"
    files = {pattern: page for pattern, page in routes.declared}
    assert files[ui_probe._route_pattern("/books")] == "src/pages/Books.tsx"
    assert files[ui_probe._route_pattern("/books/:id/edit")] == "src/pages/BookForm.tsx"
    assert files[ui_probe._route_pattern("/loans")] == "src/pages/Loans.tsx"
    assert files[ui_probe._route_pattern("/catalogue/authors")] == "src/pages/Authors.tsx"


def test_the_build_base_is_read_from_its_index(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-1.js").write_text("")
    (tmp_path / "index.html").write_text('<script type="module" src="/app/assets/index-1.js"></script>')
    assert ui_probe._build_base(str(tmp_path)) == "/app/"
    (tmp_path / "index.html").write_text('<script type="module" src="/assets/index-1.js"></script>')
    assert ui_probe._build_base(str(tmp_path)) == "/"


def test_the_backend_answers_on_every_local_port_the_bundle_calls(tmp_path):
    # A build-time API address and the code's fallback both stay in the bundle.
    (tmp_path / "a.js").write_text('const A="http://127.0.0.1:8130"||"http://localhost:8000";'
                                   'fetch(`${A}/x`);fetch("http://127.0.0.1:8130/y")', encoding="utf-8")
    assert ui_probe._bundle_backend_ports(str(tmp_path)) == [8130, 8000]
    (tmp_path / "a.js").write_text('fetch("/api/x")', encoding="utf-8")
    assert ui_probe._bundle_backend_ports(str(tmp_path)) == [8000]


def test_screen_findings_never_block_and_read_for_the_user():
    for message in ("screen: [web] route /x: an uncaught TypeError was thrown and the page showed nothing",
                    "screen warning: [web] route /x: GET /api/x answered 404"):
        assert _classify_issue(message).severity == "warning"
    assert user_sentence("screen warning: [web] route /x: GET /api/x answered 404") == (
        "A screen of the app reported a problem when it was opened in a browser.")


def test_a_crash_caused_elsewhere_names_its_cause():
    crash = {"exceptions": [{"text": "TypeError: data is undefined", "promise": False}],
             "rendered": False, "idle": True}
    down = {**crash, "requests": [{"method": "GET", "url": "http://localhost:8000/api/x", "error": "refused",
                                   "own": True, "unreachable": True}]}
    cdn = {**crash, "requests": [{"method": "GET", "url": "https://cdn.example.org/lib.js", "type": "Script",
                                  "error": "net::ERR_NAME_NOT_RESOLVED", "own": False}]}
    assert ui_probe._screen_findings(down)[0][0].endswith("(the app's backend was not reachable)")
    assert ui_probe._screen_findings(cdn)[0][0].endswith("(an external script could not load without a network)")


# ------------------------------------------------------------------ the sandbox


def _fake_tools(monkeypatch):
    monkeypatch.setattr(ui_probe, "find_browser", lambda: "/usr/bin/chromium-headless-shell")
    monkeypatch.setattr(ui_probe.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)


def _driver_report(**fields) -> str:
    return ui_probe._MARKER + json.dumps({"backend": {"status": "up", "seeded": {}}, "screens": [], **fields})


def test_the_screen_check_runs_confined_with_no_network_on_a_scratch_copy(tmp_path, monkeypatch):
    from besser.spec_driven_agent.execution import sandbox as sandbox_mod

    root = tmp_path / "workspace"
    shutil.copytree(FIXTURE, root)
    _fake_tools(monkeypatch)
    calls = []

    def confined(argv, **kwargs):
        if argv[1] != ui_probe._DRIVER:  # the create probe's child
            return subprocess.CompletedProcess(argv, 0, "", "")
        with open(argv[2], encoding="utf-8") as handle:
            calls.append((argv, kwargs, json.load(handle)))
        return subprocess.CompletedProcess(argv, 0, _driver_report(), "")

    monkeypatch.setattr(sandbox_mod, "run_confined", confined)
    issues = collect_screen_report(str(root), ["frontend"])["issues"]
    assert not any(issue.startswith(ui_probe.NOT_RUN_PREFIX) for issue in issues), issues
    ((argv, kwargs, config),) = calls
    assert argv[:2] == ["/usr/bin/node", ui_probe._DRIVER]
    assert "network" not in kwargs, "run_confined's default is no network"
    assert kwargs["workspace"] == str(root) and kwargs["writable"] == [kwargs["cwd"]]
    assert not kwargs["cwd"].startswith(str(root)), "the check must work on a scratch copy"
    assert config["backend"]["cwd"].startswith(kwargs["cwd"]) and config["site"].startswith(kwargs["cwd"])
    assert config["backend"]["port"] == 8000 and config["routes"][0] == "/" and config["base"] == "/"


def test_a_stopped_check_keeps_the_screens_it_opened(tmp_path, monkeypatch):
    from besser.spec_driven_agent.execution import sandbox as sandbox_mod

    root = tmp_path / "workspace"
    shutil.copytree(FIXTURE, root)
    _fake_tools(monkeypatch)
    opened = {"route": "/throws", "rendered": False, "idle": True, "ms": 900,
              "exceptions": [{"text": "TypeError: x is undefined", "promise": False}]}

    def confined(argv, **kwargs):
        if argv[1] != ui_probe._DRIVER:
            return subprocess.CompletedProcess(argv, 0, "", "")
        stop = subprocess.TimeoutExpired(argv, 30, output=_driver_report(screens=[opened], unvisited=["/empty"]))
        stop.reason = "the run was cancelled"  # what the run's own stop carries
        raise stop

    monkeypatch.setattr(sandbox_mod, "run_confined", confined)
    report = collect_screen_report(str(root), ["frontend"])
    assert [screen["route"] for screen in report["screens"]] == ["/throws"]
    assert report["candidates"][0]["route"] == "/throws"
    assert ("screen warning: [frontend] the run was cancelled stopped the check after 1 screen(s); "
            "not opened: /empty") in report["issues"]


def test_without_a_sandbox_the_screen_check_is_not_run(tmp_path, monkeypatch):
    from besser.spec_driven_agent.execution import sandbox as sandbox_mod
    from besser.spec_driven_agent.execution.sandbox import SandboxUnavailable

    root = tmp_path / "workspace"
    shutil.copytree(FIXTURE, root)
    _fake_tools(monkeypatch)

    def unavailable(*_args, **_kwargs):
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed")

    monkeypatch.setattr(sandbox_mod, "run_confined", unavailable)
    assert collect_screen_report(str(root), ["frontend"])["issues"] == [
        "screens: not run ([frontend] the shell sandbox is unavailable on this server)"]


# ------------------------------------------------------------------- Phase 3 wiring


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"
    usage = _Usage()


class _Trace:
    def __init__(self):
        self.events = []

    def write(self, event, **payload):
        self.events.append((event, payload))


@pytest.fixture
def orch(tmp_path):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

    return LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)


def test_screens_open_only_after_the_build_passed_and_candidates_reach_the_trace(orch, monkeypatch):
    calls = []
    candidate = {"project": "frontend", "route": "/", "file": "", "exception": "Error: x"}

    def report(output_dir, projects, *, timeout, domain_model):
        calls.append((projects, domain_model))
        return {"issues": ["screen warning: [frontend] route /: x"], "screens": [{}], "candidates": [candidate],
                "seconds": 1.0}

    monkeypatch.setattr(ui_probe, "collect_screen_report", report)
    orch._trace = _Trace()
    revision = orch._workspace_revision()
    orch._frontend_build_verdicts[("frontend", revision, (), "vite build")] = "frontend build [frontend]: failed"
    assert orch._collect_screen_issues() == [] and calls == []

    orch._frontend_build_verdicts[("frontend", revision, ("installed",), "vite build")] = None
    assert orch._collect_screen_issues() == ["screen warning: [frontend] route /: x"]
    assert orch._collect_screen_issues() == ["screen warning: [frontend] route /: x"]
    assert calls == [(["frontend"], orch.domain_model)], "an unchanged tree was opened again"
    exits = [payload for event, payload in orch._trace.events if payload.get("phase") == "phase3_screens"
             and event == "phase_exit"]
    assert exits[0][ui_probe.BLOCKER_CANDIDATES] == [candidate]


def test_a_screen_check_that_did_not_run_is_tried_again(orch, monkeypatch):
    outcomes = [not_run("[frontend] timed out after 150s"), not_run("no browser available")]
    calls = []

    def report(output_dir, projects, *, timeout, domain_model):
        calls.append(projects)
        return {"issues": [outcomes[min(len(calls), 2) - 1]], "screens": [], "candidates": [], "seconds": 0.0}

    monkeypatch.setattr(ui_probe, "collect_screen_report", report)
    orch._frontend_build_verdicts[("frontend", orch._workspace_revision(), (), "vite build")] = None
    orch._collect_screen_issues()
    orch._collect_screen_issues()
    orch._collect_screen_issues()
    assert len(calls) == 2, "a timed-out check was cached, or a missing browser was looked for again"


def test_phase3_collects_the_screen_findings_as_warnings(orch, monkeypatch):
    monkeypatch.setattr(type(orch), "_collect_screen_issues", lambda self: ["screen warning: [web] route /: marker"])
    issues = orch._collect_validation_issues()
    assert any(issue.message == "screen warning: [web] route /: marker" and issue.severity == "warning"
               for issue in issues)
