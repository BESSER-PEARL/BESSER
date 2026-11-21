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


def test_without_installed_dependencies_tsc_is_listed_not_run(tmp_path, monkeypatch):
    def no_compiler(*args, **kwargs):
        raise AssertionError("tsc ran on a project with no installed dependencies")

    monkeypatch.setattr(sandbox_mod, "run_confined", no_compiler)
    result = _orch(tmp_path, installed=False, shell=False)._validate_app()
    assert any(issue.startswith("validation unverified: tsc [web] did not run completely "
                                "(dependencies are not installed)") for issue in result["issues"])
    assert result["blocker_count"] == 0 and result["verified"] is False


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
