"""Expensive Phase 3 checks run once per source revision, failures included.

Phase 3 re-validates after no-edit repair rounds and after a rollback restore.
Only a SUCCESSFUL ``npm run build`` was cached, and tsc, cargo, kotlinc and the
pip dry-run were not cached at all, so an unchanged tree paid up to 120 s
(build), 60 s (tsc), 180 s (cargo, kotlinc) and 30 s plus network per
requirements file (pip) again on every pass, against ``max_runtime_seconds``.

A check that did not run (timeout, no sandbox, network down) is no verdict:
the next pass must try again.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_mod
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.validation.frontend_build import collect_frontend_build_issues
from besser.spec_driven_agent.validation.issues import _check_did_not_run, required_check_unverified


class _Usage:
    estimated_cost = 0.0


class _Client:
    model = "mock-model"
    usage = _Usage()


@pytest.fixture
def orch(tmp_path) -> LLMOrchestrator:
    model = DomainModel(name="Tiny", types={
        Class(name="Note", attributes={Property(name="body", type=PrimitiveDataType("str"))})})
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=model, output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=True,
    )


# --------------------------------------------------------------- npm run build


def _web_app(root):
    (root / "package.json").write_text(json.dumps({"scripts": {"build": "vite build"}}))
    (root / "index.html").write_text("<div id='root'></div>")
    (root / "node_modules").mkdir()


def test_a_failing_build_runs_once_per_revision(tmp_path, monkeypatch):
    _web_app(tmp_path)
    calls, outcome, revision = [], {"exit": 1}, {"id": "r1"}

    def build(argv, **kwargs):
        calls.append(argv)
        if outcome["exit"] is None:
            raise subprocess.TimeoutExpired(argv, 120)
        return subprocess.CompletedProcess(argv, outcome["exit"], "", "src/App.jsx: Unexpected token")

    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    monkeypatch.setattr(sandbox_mod, "run_confined", build)
    cache: dict = {}

    def check():
        return collect_frontend_build_issues(
            str(tmp_path), enabled=True, allow_shell=True, source_revision=lambda: revision["id"],
            build_verdicts=cache, can_run=lambda: True)

    first = check()
    assert "npm run build failed" in first[0]
    assert check() == first and check() == first
    assert len(calls) == 1, "a failing build re-ran for an unchanged revision"

    revision["id"] = "r2"  # an edit
    outcome["exit"] = None  # ... whose build times out: no verdict
    assert "timed out" in check()[0]
    assert "timed out" in check()[0]
    assert len(calls) == 3, "a timed-out build must be retried, not served from the cache"

    (tmp_path / "node_modules" / ".package-lock.json").write_text("{}")  # dependencies changed
    outcome["exit"] = 0
    assert check() == [] and check() == []
    assert len(calls) == 4


# ---------------------------------------------------- tsc / cargo / kotlinc


@pytest.mark.parametrize("wrapper", ["_collect_tsc_issues", "_collect_cargo_issues", "_collect_kotlinc_issues"])
def test_a_compiler_verdict_is_reused_until_the_tree_changes(orch, tmp_path, monkeypatch, wrapper):
    (tmp_path / "src.ts").write_text("const x: number = 'a';\n")
    calls = []

    def compiler(*args, **kwargs):
        calls.append(args)
        return [f"tsc [.]: src.ts(1,7): error TS2322 (call {len(calls)})"]

    monkeypatch.setattr(orchestrator_mod, wrapper, compiler)
    first = getattr(orch, wrapper)()
    assert getattr(orch, wrapper)() == first
    assert len(calls) == 1, f"{wrapper} re-ran for an unchanged revision"

    (tmp_path / "src.ts").write_text("const x: number = 1;\n")
    assert getattr(orch, wrapper)() != first
    assert len(calls) == 2


@pytest.mark.parametrize("skipped", [
    required_check_unverified("tsc [.]", "timed out after 60s"),
    _check_did_not_run("cargo [.]", "timed out after 180s"),
    _check_did_not_run("kotlinc [src/main/kotlin]", "could not be launched: [Errno 12]"),
    _check_did_not_run("cargo [.]", "its dependencies could not be fetched"),
])
def test_a_check_that_did_not_run_is_retried(orch, monkeypatch, skipped):
    calls = []
    monkeypatch.setattr(orchestrator_mod, "_collect_tsc_issues",
                        lambda *a, **k: calls.append(1) or ["tsc [web]: a.ts(1,1): error TS1", skipped])
    orch._collect_tsc_issues()
    orch._collect_tsc_issues()
    assert len(calls) == 2


def test_tsc_reruns_after_dependencies_are_installed(orch, tmp_path, monkeypatch):
    """node_modules is outside the revision but decides what tsc can resolve."""
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "tsconfig.json").write_text("{}")
    (tmp_path / "web" / "package.json").write_text("{}")
    calls = []
    monkeypatch.setattr(orchestrator_mod, "_collect_tsc_issues",
                        lambda *a, **k: calls.append(1) or ["tsc [web]: a.ts(1,1): error TS2307"])
    orch._collect_tsc_issues()
    orch._collect_tsc_issues()
    (tmp_path / "web" / "node_modules").mkdir()
    (tmp_path / "web" / "node_modules" / ".package-lock.json").write_text("{}")
    orch._collect_tsc_issues()
    assert len(calls) == 2


# ------------------------------------------------------------ pip dry-run


def test_the_pip_dry_run_runs_once_per_revision(orch, tmp_path, monkeypatch):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "requirements.txt").write_text("fastapi==0.1\npydantic==9\n")
    orch.allow_shell_tools = True
    pip_calls, stderr = [], {"text": "ERROR: ResolutionImpossible: fastapi 0.1 needs pydantic<2"}

    def run(argv, **kwargs):
        if "pip" in argv:
            pip_calls.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", stderr["text"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(orchestrator_mod, "run_confined", run)

    def conflicts():
        return [i.message for i in orch._collect_validation_issues() if "requirements.txt" in i.message]

    first = conflicts()
    assert any(m.startswith("Dependency conflict in backend/requirements.txt") for m in first)
    assert conflicts() == first
    assert len(pip_calls) == 1, "the pip dry-run re-ran for an unchanged revision"

    (tmp_path / "backend" / "requirements.txt").write_text("fastapi==0.2\npydantic==9\n")
    stderr["text"] = "ERROR: SSLError(CERTIFICATE_VERIFY_FAILED)"  # network down: no verdict
    assert any("could not verify dependencies" in m for m in conflicts())
    conflicts()
    assert len(pip_calls) == 3, "a dependency check that could not reach the index must be retried"
