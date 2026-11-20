"""Phase 3 opens every screen of a built frontend in a headless browser.

No validation step rendered a page: a frontend that builds and then throws on
mount, stays blank, or gets a 500 from its own backend for the data it shows
passed every check. ``ui_probe`` boots the backend, serves the build output
and opens each declared screen (plus the links pages lead to) headless.

The fixture under ``fixtures/screen_app`` is a committed stand-in for a built
app and its backend (no npm install): one screen per outcome. The browser
tests drive a Chromium-family browser already installed on the machine and
skip without one, like the Node-dependent tests skip without Node.
"""

from __future__ import annotations

import os
import shutil
import socket
import time

import pytest

from besser.spec_driven_agent.validation import ui_probe
from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.ui_probe import (
    collect_screen_report,
    discover_routes,
    not_run,
    screen_evidence,
)
from besser.spec_driven_agent.validation.user_wording import user_sentence

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "screen_app")

needs_browser = pytest.mark.skipif(
    ui_probe.find_browser() is None or shutil.which("node") is None,
    reason="needs Node.js and a Chromium-family browser installed on this machine")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A copy of the fixture app; its backend gets a free port instead of the bundle's default."""
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    root = tmp_path / "workspace"
    shutil.copytree(FIXTURE, root)
    port = _free_port()
    monkeypatch.setattr(ui_probe, "_bundle_backend_ports", lambda site: [port])
    return root


def _declare(root, *routes: str) -> None:
    """Replace the router with one declaring only ``routes`` (path, page)."""
    lines = [f'import {page} from "./pages/{page}";' for _, page in routes]
    lines += ["export default () => (", "  <Routes>"]
    lines += [f'    <Route path="{path}" element={{<{page} />}} />' for path, page in routes]
    lines += ["  </Routes>", ");", ""]
    (root / "frontend" / "src" / "App.jsx").write_text("\n".join(lines), encoding="utf-8")


def _by_route(issues: list[str], route: str) -> list[str]:
    return [issue for issue in issues if f" route {route} " in issue or issue.endswith(f" route {route}")]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One run over every screen of the fixture (the browser start is the slow part)."""
    if ui_probe.find_browser() is None or shutil.which("node") is None:
        pytest.skip("needs Node.js and a Chromium-family browser installed on this machine")
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    root = tmp_path_factory.mktemp("screens") / "workspace"
    shutil.copytree(FIXTURE, root)
    port = _free_port()
    original = ui_probe._bundle_backend_ports
    ui_probe._bundle_backend_ports = lambda site: [port]
    try:
        return collect_screen_report(str(root), ["frontend"])
    finally:
        ui_probe._bundle_backend_ports = original


# ----------------------------------------------------------- what a screen meets


@needs_browser
def test_a_screen_that_throws_on_load_is_a_blocker_naming_its_route_and_page(full_run):
    found = _by_route(full_run["issues"], "/throws")
    assert len(found) == 1, full_run["issues"]
    assert found[0].startswith("screen: [frontend] route /throws (frontend/src/pages/Broken.jsx): uncaught TypeError")
    assert _classify_issue(found[0]).severity == "blocker"


@needs_browser
def test_a_screen_whose_root_stays_empty_is_a_blocker(full_run):
    found = _by_route(full_run["issues"], "/empty")
    assert found == ["screen: [frontend] route /empty (frontend/src/pages/Blank.jsx): "
                     "the page rendered nothing (#root stayed empty after load)"]
    assert _classify_issue(found[0]).severity == "blocker"


@needs_browser
def test_a_500_from_the_apps_backend_during_load_is_a_blocker(full_run):
    found = _by_route(full_run["issues"], "/server-error")
    assert found == ["screen: [frontend] route /server-error (frontend/src/pages/Report.jsx): "
                     "GET /api/report answered 500 from the app's backend during load"]


@needs_browser
def test_a_404_during_load_is_only_a_warning(full_run):
    found = _by_route(full_run["issues"], "/missing")
    assert found == ["screen warning: [frontend] route /missing (frontend/src/pages/CustomerList.jsx): "
                     "GET /api/customers answered 404"]
    assert _classify_issue(found[0]).severity == "warning"


@needs_browser
def test_healthy_screens_and_linked_screens_have_no_findings(full_run):
    assert _by_route(full_run["issues"], "/") == []
    # /orders/:id is never opened from the router (no id to fill in); the home page links to it.
    assert _by_route(full_run["issues"], "/orders/7") == []
    opened = {screen["route"]: screen for screen in full_run["screens"]}
    assert opened["/orders/7"]["file"] == "frontend/src/pages/OrderDetail.jsx"
    assert opened["/"]["rendered"] and not opened["/"]["blocker"]
    assert screen_evidence(full_run) == {
        "frontend/src/pages/OrderList.jsx": True, "frontend/src/pages/OrderDetail.jsx": True,
        "frontend/src/pages/Broken.jsx": False, "frontend/src/pages/Blank.jsx": False,
        "frontend/src/pages/Report.jsx": False, "frontend/src/pages/CustomerList.jsx": True,
    }


@needs_browser
def test_a_healthy_app_has_no_findings(app):
    _declare(app, ("/", "OrderList"), ("/orders/:id", "OrderDetail"))
    report = collect_screen_report(str(app), ["frontend"])
    assert report["issues"] == []
    assert [screen["route"] for screen in report["screens"]] == ["/", "/orders/7"]


# ------------------------------------------------------------- optional, bounded


def test_no_browser_is_one_not_run_note_and_never_a_blocker(tmp_path, monkeypatch):
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
def test_each_screen_is_time_boxed(app, monkeypatch):
    _declare(app, ("/", "OrderList"), ("/slow", "Report"))
    monkeypatch.setattr(ui_probe, "SCREEN_SECONDS", 3)
    report = collect_screen_report(str(app), ["frontend"])
    slow = next(screen for screen in report["screens"] if screen["route"] == "/slow")
    assert slow["seconds"] < 6
    assert slow["rendered"] and not slow["blocker"]


@needs_browser
def test_the_whole_check_is_time_boxed(app, monkeypatch):
    _declare(app, ("/", "OrderList"), ("/slow", "Report"), ("/throws", "Broken"))
    monkeypatch.setattr(ui_probe, "SCREEN_SECONDS", 120)
    started = time.monotonic()
    report = collect_screen_report(str(app), ["frontend"], timeout=30)
    assert time.monotonic() - started < 30
    assert [screen["route"] for screen in report["screens"]] == ["/", "/slow"]
    stopped = [issue for issue in report["issues"] if "time limit" in issue]
    assert stopped == ["screen warning: [frontend] the time limit stopped the check after 2 screen(s); "
                       "not opened: /throws"]
    assert all(_classify_issue(issue).severity == "warning" for issue in report["issues"])


# ------------------------------------------------------------------ route reading


def test_routes_are_read_from_jsx_object_routers_and_lazy_pages(tmp_path):
    src = tmp_path / "src"
    (src / "pages").mkdir(parents=True)
    for page in ("Books", "BookForm", "Authors"):
        (src / "pages" / f"{page}.tsx").write_text(f"export default function {page}() {{ return null; }}\n")
    (src / "App.tsx").write_text(
        'import Books from "./pages/Books";\n'
        'const BookForm = lazy(() => import("./pages/BookForm"));\n'
        "<Route path='/books' element={<Guard><Books /></Guard>} />\n"
        '<Route path="/books/:id/edit" element={<BookForm />} />\n'
        '<Route path="*" element={<p>Not found</p>} />\n', encoding="utf-8")
    (src / "router.ts").write_text(
        'import { Authors } from "./pages/Authors";\n'
        "export const router = createHashRouter([{ path: '/authors', element: <Authors /> }]);\n",
        encoding="utf-8")
    routes = discover_routes(str(tmp_path))
    assert routes.visit == ["/", "/authors", "/books"]
    assert routes.hash is True
    files = {pattern: page for pattern, page in routes.declared}
    assert files[ui_probe._route_pattern("/books")] == "src/pages/Books.tsx"
    assert files[ui_probe._route_pattern("/books/:id/edit")] == "src/pages/BookForm.tsx"
    assert files[ui_probe._route_pattern("/authors")] == "src/pages/Authors.tsx"


def test_the_backend_answers_on_every_local_port_the_bundle_calls(tmp_path):
    # A build-time API address and the code's fallback both stay in the bundle.
    (tmp_path / "a.js").write_text('const A="http://127.0.0.1:8130"||"http://localhost:8000";'
                                   'fetch(`${A}/x`);fetch("http://127.0.0.1:8130/y")', encoding="utf-8")
    assert ui_probe._bundle_backend_ports(str(tmp_path)) == [8130, 8000]
    (tmp_path / "a.js").write_text('fetch("/api/x")', encoding="utf-8")
    assert ui_probe._bundle_backend_ports(str(tmp_path)) == [8000]


def test_screen_findings_classify_and_read_for_the_user():
    blocker = "screen: [frontend] route /x (frontend/src/X.tsx): uncaught TypeError: a is undefined"
    warning = "screen warning: [frontend] route /x: GET /api/x answered 404"
    assert _classify_issue(blocker).severity == "blocker"
    assert _classify_issue(warning).severity == "warning"
    assert user_sentence(blocker) == "A screen of the app fails when it is opened in a browser."
    assert user_sentence(warning) == "A screen of the app reported errors when it was opened in a browser."


def test_an_unreachable_backend_or_external_script_never_makes_a_blocker():
    crash = {"exceptions": [{"text": "TypeError: data is undefined", "promise": False}],
             "rendered": False, "idle": True, "mount": "#root"}
    down = {**crash, "requests": [{"method": "GET", "url": "http://localhost:8000/api/x", "error": "refused",
                                   "own": True, "unreachable": True}]}
    cdn = {**crash, "requests": [{"method": "GET", "url": "https://cdn.example.org/lib.js", "type": "Script",
                                  "error": "net::ERR_NAME_NOT_RESOLVED", "own": False}]}
    assert ui_probe._screen_findings(crash, declared=True)[0]
    for screen in (down, cdn):
        blockers, warnings = ui_probe._screen_findings(screen, declared=True)
        assert blockers == [] and any("not counted as a blocker" in w for w in warnings)


# ------------------------------------------------------------------- Phase 3 wiring


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"
    usage = _Usage()


@pytest.fixture
def orch(tmp_path):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

    return LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)


def test_screens_open_only_after_the_build_passed_on_this_tree(orch, monkeypatch):
    calls = []

    def report(output_dir, projects, *, timeout):
        calls.append(projects)
        return {"issues": ["screen: [frontend] route /: uncaught Error"], "screens": [], "seconds": 1.0}

    monkeypatch.setattr(ui_probe, "collect_screen_report", report)
    revision = orch._workspace_revision()
    orch._frontend_build_verdicts[("frontend", revision, (), "vite build")] = "frontend build [frontend]: failed"
    assert orch._collect_screen_issues() == [] and calls == []

    orch._frontend_build_verdicts[("frontend", revision, ("installed",), "vite build")] = None
    assert orch._collect_screen_issues() == ["screen: [frontend] route /: uncaught Error"]
    assert orch._collect_screen_issues() == ["screen: [frontend] route /: uncaught Error"]
    assert calls == [["frontend"]], "an unchanged tree was opened again"
    assert orch._screen_report[0] == revision


def test_a_screen_check_that_did_not_run_is_tried_again(orch, monkeypatch):
    outcomes = [not_run("[frontend] timed out after 150s"), not_run("no browser available")]
    calls = []

    def report(output_dir, projects, *, timeout):
        calls.append(projects)
        return {"issues": [outcomes[min(len(calls), 2) - 1]], "screens": [], "seconds": 0.0}

    monkeypatch.setattr(ui_probe, "collect_screen_report", report)
    orch._frontend_build_verdicts[("frontend", orch._workspace_revision(), (), "vite build")] = None
    orch._collect_screen_issues()
    orch._collect_screen_issues()
    orch._collect_screen_issues()
    assert len(calls) == 2, "a timed-out check was cached, or a missing browser was looked for again"


def test_phase3_collects_the_screen_findings(orch, monkeypatch):
    monkeypatch.setattr(type(orch), "_collect_screen_issues", lambda self: ["screen: [web] route /: marker"])
    issues = orch._collect_validation_issues()
    assert any(issue.message == "screen: [web] route /: marker" and issue.severity == "blocker"
               for issue in issues)


# --------------------------------------------------------------------- the sandbox


def _fake_tools(monkeypatch):
    monkeypatch.setattr(ui_probe, "find_browser", lambda: "/usr/bin/chromium-headless-shell")
    monkeypatch.setattr(ui_probe.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)


def test_the_screen_check_runs_confined_with_no_network_on_a_scratch_copy(tmp_path, monkeypatch):
    import json
    import subprocess

    from besser.spec_driven_agent.execution import sandbox as sandbox_mod

    root = tmp_path / "workspace"
    shutil.copytree(FIXTURE, root)
    _fake_tools(monkeypatch)
    calls = []

    def confined(argv, **kwargs):
        calls.append((argv, kwargs))
        with open(argv[2], encoding="utf-8") as handle:
            calls.append(json.load(handle))
        return subprocess.CompletedProcess(argv, 0, ui_probe._MARKER + json.dumps({"screens": [], "backend": {"status": "up"}}), "")

    monkeypatch.setattr(sandbox_mod, "run_confined", confined)
    assert collect_screen_report(str(root), ["frontend"])["issues"] == []
    (argv, kwargs), config = calls
    assert argv[:2] == ["/usr/bin/node", ui_probe._DRIVER]
    assert "network" not in kwargs, "run_confined's default is no network"
    assert kwargs["workspace"] == str(root) and kwargs["writable"] == [kwargs["cwd"]]
    assert not kwargs["cwd"].startswith(str(root)), "the check must work on a scratch copy"
    assert config["backend"]["cwd"].startswith(kwargs["cwd"]) and config["site"].startswith(kwargs["cwd"])
    assert config["backend"]["port"] == 8000 and config["routes"][0] == "/"


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
