"""A validation collector must never report clean when it did not run.

Every one of these collectors returned ``[]`` on a timeout, a tool that failed
to launch, or an internal error — which the Phase 3 verdict renders as
"0 blockers". That is the exact shape of the pilot failure where backends that
NameError on import shipped as verified: the check had not run, and nothing in
the result said so.

The contract is: a skipped check produces a visible *warning* (we can't prove
the code is broken, only that we didn't look), never silence.
"""
import subprocess

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import (
    LLMOrchestrator, _check_did_not_run, _classify_issue,
)


def _model():
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="TestModel", types={user})


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        return {"stop_reason": "end_turn", "content": []}


@pytest.fixture
def orch(tmp_path):
    return LLMOrchestrator(
        llm_client=_MockClient(), domain_model=_model(), output_dir=str(tmp_path),
    )


def _said_it_skipped(issues):
    return any("did not run" in i and "SKIPPED" in i for i in issues)


# --------------------------------------------------------------------------- #
# The note itself
# --------------------------------------------------------------------------- #
def test_a_skipped_check_is_a_warning_not_a_blocker():
    note = _check_did_not_run("ruff", "timed out after 30s")
    assert _classify_issue(note).severity == "warning"
    # It must not look like a passing result to a reader either.
    assert "SKIPPED" in note


# --------------------------------------------------------------------------- #
# ruff
# --------------------------------------------------------------------------- #
def test_ruff_timeout_is_reported(orch, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="ruff", timeout=30)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert _said_it_skipped(orch._collect_ruff_issues())


def test_ruff_failing_to_launch_is_reported(orch, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    def _boom(*a, **kw):
        raise OSError("Exec format error")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert _said_it_skipped(orch._collect_ruff_issues())


def test_ruff_erroring_out_is_reported(orch, monkeypatch):
    """--exit-zero means findings never set a non-zero status, so a non-zero
    code is ruff itself failing. That used to read as a clean workspace."""
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    class _R:
        returncode = 2
        stdout = ""
        stderr = "ruff failed to read pyproject.toml"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())
    issues = orch._collect_ruff_issues()
    assert _said_it_skipped(issues)
    assert any("pyproject.toml" in i for i in issues)


def test_a_clean_ruff_run_stays_clean(orch, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())
    assert orch._collect_ruff_issues() == []


# --------------------------------------------------------------------------- #
# tsc
# --------------------------------------------------------------------------- #
def _with_tsconfig(tmp_path):
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")


def test_tsc_failing_with_no_parseable_errors_is_reported(orch, tmp_path, monkeypatch):
    """A frontend whose toolchain won't even start is not a passing frontend."""
    _with_tsconfig(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/tsc")

    class _R:
        returncode = 1
        stdout = ""
        stderr = "Cannot find module 'typescript'"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())
    issues = orch._collect_tsc_issues()
    assert any("typescript" in i for i in issues)


def test_tsc_timeout_is_reported(orch, tmp_path, monkeypatch):
    _with_tsconfig(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/tsc")

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="tsc", timeout=60)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert _said_it_skipped(orch._collect_tsc_issues())


def test_a_clean_tsc_run_stays_clean(orch, tmp_path, monkeypatch):
    _with_tsconfig(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/tsc")

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())
    assert orch._collect_tsc_issues() == []


# --------------------------------------------------------------------------- #
# data contract
# --------------------------------------------------------------------------- #
def test_a_failed_contract_build_is_reported(orch, monkeypatch):
    import besser.generators.llm.contract_checks as cc

    def _boom(_model):
        raise RuntimeError("contract build exploded")

    monkeypatch.setattr(cc, "build_data_contract", _boom)
    issues = orch._collect_data_contract_issues()
    assert _said_it_skipped(issues)
    assert any("exploded" in i for i in issues)
