"""validate_app runs the checks that decide whether a fix attempt is kept and the run's verdict.

It ran only the startup probes and the static source checks, while the re-check
after a fix attempt also ran tsc, the frontend build and the screen check. A new
``tsc`` error reverted the attempt and ended the repair loop, and the model's own
validate_app had said nothing about it; ``verification setup:`` even told the
model to "call validate_app to run the required checks", which it did not run.
tsc and the build are expensive, so they run only while their inputs changed and
their dependencies are installed; anything that did not run withholds
``verified`` and is never a blocker.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_mod
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.validation import ui_probe

TYPE_ERROR = ("tsc [web]: src/App.tsx(3,7): error TS2322: "
              "Type 'string' is not assignable to type 'number'.")


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"
    usage = _Usage()


def _orch(tmp_path, *, installed=True, shell=True) -> LLMOrchestrator:
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "tsconfig.json").write_text("{}", encoding="utf-8")
    (web / "package.json").write_text('{"name": "orders"}', encoding="utf-8")
    (web / "src" / "App.tsx").write_text("const total: number = 1;\n", encoding="utf-8")
    if installed:
        (web / "node_modules").mkdir()
    (tmp_path / "service.py").write_text("def total(orders):\n    return len(orders)\n", encoding="utf-8")
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
        output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=True, allow_shell_tools=shell, enable_import_smoke_check=False,
        enable_requirements_ledger=False,
    )


@pytest.fixture
def compiler(monkeypatch):
    calls = []

    def tsc(*args, **kwargs):
        calls.append(kwargs)
        return [TYPE_ERROR]

    monkeypatch.setattr(orchestrator_mod, "_collect_tsc_issues", tsc)
    return calls


def test_validate_app_reports_the_type_error_the_recheck_reverts_on(tmp_path, compiler):
    orch = _orch(tmp_path)
    result = orch._validate_app()
    assert TYPE_ERROR in result["issues"] and result["blocker_count"] == 1
    assert result["verified"] is False
    assert TYPE_ERROR in [issue.message for issue in orch._collect_validation_issues()]


def test_a_type_check_reruns_only_when_the_frontend_changed(tmp_path, compiler):
    orch = _orch(tmp_path)
    orch._validate_app()
    orch._validate_app()
    assert len(compiler) == 1, "tsc ran again on an unchanged tree"

    (tmp_path / "service.py").write_text("def total(orders):\n    return 0\n", encoding="utf-8")
    orch._validate_app()
    assert len(compiler) == 1, "a backend-only edit ran tsc again"

    (tmp_path / "web" / "src" / "App.tsx").write_text("const total: number = 2;\n", encoding="utf-8")
    orch._validate_app()
    assert len(compiler) == 2


def test_where_the_agent_can_install_uninstalled_tsc_asks_for_the_install(tmp_path, monkeypatch):
    _host_tsc(monkeypatch, npm=True)

    def no_compiler(*args, **kwargs):
        raise AssertionError("tsc ran on a project whose dependencies the agent can install")

    monkeypatch.setattr(sandbox_mod, "run_confined", no_compiler)
    result = _orch(tmp_path, installed=False)._validate_app()
    assert any(issue.startswith("verification setup: tsc [web] requires installed project dependencies")
               for issue in result["issues"])
    assert result["verified"] is False


def test_a_screen_check_that_did_not_run_withholds_verified(tmp_path, monkeypatch):
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False,
                           enable_import_smoke_check=False, enable_requirements_ledger=False)
    monkeypatch.setattr(ui_probe, "collect_screen_report", lambda *a, **k: {
        "issues": [ui_probe.not_run("no browser available")], "screens": [], "candidates": [], "seconds": 0.0})
    # A build that passed on this tree, as the build check records it.
    orch._frontend_build_verdicts[("web", orch._workspace_revision(), (), "vite build")] = None
    result = orch._validate_app()
    assert "screens: not run (no browser available)" in result["issues"]
    assert result["blocker_count"] == 0 and result["verified"] is False


def _snapshot(root) -> dict:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_validate_app_reports_and_leaves_the_workspace_as_it_was(tmp_path):
    # The sweep's repairs (build config, package.json, Dockerfile) wrote under the model
    # mid-Phase 2: a vite.config.mjs then shadowed the vite.config.ts the model wrote later.
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "package.json").write_text(
        '{"name": "orders", "scripts": {"dev": "vite"}, "dependencies": {"react": "^18.3.1", '
        '"react-dom": "^18.3.1"}, "devDependencies": {"vite": "^5.4.0"}}', encoding="utf-8")
    (web / "index.html").write_text('<div id="root"></div><script type="module" src="/src/main.tsx"></script>',
                                    encoding="utf-8")
    (web / "src" / "main.tsx").write_text(
        "import ReactDOM from 'react-dom/client';\n"
        "ReactDOM.createRoot(document.getElementById('root')!).render(<h1>Orders</h1>);\n", encoding="utf-8")
    (web / "Dockerfile").write_text(
        "FROM node:20\nWORKDIR /app\nCOPY package.json package-lock.json ./\nRUN npm ci\nCOPY . .\n",
        encoding="utf-8")
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False,
                           enable_import_smoke_check=False, enable_requirements_ledger=False)
    before = _snapshot(tmp_path)

    result = orch._validate_app()

    assert _snapshot(tmp_path) == before
    # Its findings are those of the tree the sweep repairs: the lockfile COPY it strips is none.
    assert not any("package-lock.json" in issue for issue in result["issues"])
    orch._collect_validation_issues()
    assert (web / "vite.config.mjs").exists(), "the sweep no longer repairs the build configuration"
    assert "npm ci" not in (web / "Dockerfile").read_text(encoding="utf-8")


@pytest.mark.parametrize("frontend", ["web", "app/web"])
def test_an_edit_the_frontend_compiles_from_outside_its_folder_reruns_tsc_at_the_gate(
        tmp_path, compiler, monkeypatch, frontend):
    orch = _orch(tmp_path)
    monkeypatch.setattr(orch, "_pip_dry_run_issues", lambda rel: [])
    folder = tmp_path / frontend
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tsconfig.json").write_text('{"include": ["src", "../shared"]}', encoding="utf-8")
    if frontend == "app/web":
        # The frontend sits inside a Python project's folder.
        (tmp_path / "app" / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    shared = folder.parent / "shared"
    shared.mkdir()
    (shared / "totals.ts").write_text("export const total: number = 1;\n", encoding="utf-8")
    orch._collect_validation_issues()

    (shared / "totals.ts").write_text("export const total: number = 'x';\n", encoding="utf-8")
    orch._collect_validation_issues()

    assert len(compiler) == 2, "the gate reused a type check made before the edit"


def _host_tsc(monkeypatch, *, npm: bool):
    real_which = shutil.which
    host = {"tsc"} | ({"npm"} if npm else set())
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}" if name in host
                        else real_which(name, *a, **k))


MISSING_MODULE = ("src/App.tsx(1,17): error TS2307: Cannot find module './OrderList' "
                  "or its corresponding type declarations.")


def test_without_shell_validate_app_shows_the_partial_type_check_the_gate_blocks_on(tmp_path, monkeypatch):
    # No install is possible, so the gate decides on the partial check: the model must see it.
    _host_tsc(monkeypatch, npm=False)

    def run(cmd, **kwargs):
        assert "--noEmit" in cmd
        return subprocess.CompletedProcess(cmd, 2, stdout=MISSING_MODULE + "\n", stderr="")

    monkeypatch.setattr(sandbox_mod, "run_confined", run)
    orch = _orch(tmp_path, installed=False, shell=False)

    result = orch._validate_app()

    assert any("TS2307" in issue for issue in result["issues"]) and result["blocker_count"] >= 1
    assert any("TS2307" in issue.message and issue.severity == "blocker"
               for issue in orch._collect_validation_issues())
