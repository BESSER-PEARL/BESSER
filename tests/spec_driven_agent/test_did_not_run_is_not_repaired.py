"""A check that could not run is an unknown, not a defect the fix loop can edit away.

Reproduced on a pristine scaffold with no ``bwrap`` on PATH: the import smoke
check and the constructibility probe did not run ("the shell sandbox is
unavailable on this server"), and Phase 3 re-prefixed both
``runtime unverified:`` - a blocker. After one ``test_api`` call the scenario
whose boot was ``probe_error`` was retained and replayed as
``api scenario: ... failed``, another blocker. Each one bought a paid repair
round (four LLM calls) that could only end on the replay guard: no source edit
starts a sandbox, ends a timeout or installs the harness's ``httpx``.

They are now ``validation unverified:`` warnings, the class the
``ValidationIssue`` docstring reserves for this. The fix loop does not consume
them, and they still refuse verified completion: the run reports the honest
"unverified" outcome, neither complete nor failed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from besser.spec_driven_agent.pipeline.constants import _RUNTIME_UNVERIFIED
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation import api_probe, constructibility
from besser.spec_driven_agent.validation.issues import (
    _check_did_not_run,
    _classify_issue,
    is_completion_issue,
    unresolved_defects,
)

_NO_SANDBOX = "the shell sandbox is unavailable on this server"


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


@pytest.fixture
def orch(simple_library_book_model, tmp_path, monkeypatch):
    orchestrator = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, auto_fix_issues=True,
    )
    monkeypatch.setattr(orchestrator, "_probeable_backends", lambda: ["backend"])
    monkeypatch.setattr(orchestrator, "_workspace_revision", lambda: "rev-1")
    return orchestrator


def _sandbox_unavailable(monkeypatch):
    """Both runtime probes answer as they do on a server with no sandbox."""
    monkeypatch.setattr(
        "besser.spec_driven_agent.pipeline.orchestrator._import_smoke_issues",
        lambda _d: [_check_did_not_run("the import smoke check", _NO_SANDBOX)])
    report = {"boot": "probe_error", "error": _NO_SANDBOX}
    monkeypatch.setattr(
        constructibility, "collect_constructibility_report",
        lambda _d, _m=None: {"issues": constructibility._issues_from_report(report, "backend"),
                             "backends": [{"backend": "backend", **report}]})


def _classified(orchestrator):
    return [_classify_issue(item) for item in orchestrator._collect_execution_issues()]


def test_a_probe_that_did_not_run_is_an_unverified_warning(orch, monkeypatch):
    _sandbox_unavailable(monkeypatch)

    issues = _classified(orch)

    assert issues, "a check that did not run must still be reported"
    assert not [i.message for i in issues if i.severity == "blocker"]
    assert all(i.message.startswith("validation unverified:") for i in issues), issues
    assert all("did not run" in i.message for i in issues)
    # Still an unknown that refuses verified completion, never a pass.
    assert all(is_completion_issue(i) for i in issues)
    assert orch._runtime_probe_verdict == _RUNTIME_UNVERIFIED


@pytest.mark.parametrize("boot,error", [
    ("probe_error", _NO_SANDBOX),
    ("timeout", "API scenario timed out after 60s"),
    ("missing_dependency", "ModuleNotFoundError: No module named 'httpx'"),
])
def test_a_scenario_that_did_not_run_is_not_replayed_as_a_failure(orch, monkeypatch, boot, error):
    monkeypatch.setattr(api_probe, "probe_api_scenario",
                        lambda *_a, **_k: api_probe._error(error, boot, backend="backend"))

    orch._test_api({"requests": [{"method": "POST", "path": "/book/", "json": {"title": "t"}},
                                 {"method": "GET", "path": "/book/"}]})
    [finding] = [_classify_issue(m) for m in orch._collect_api_scenario_issues()]

    assert finding.severity == "warning", finding.message
    assert finding.message.startswith("validation unverified:")
    assert "did not run" in finding.message and error[:30] in finding.message
    assert "fix the application" not in finding.message
    assert is_completion_issue(finding)


def test_a_scenario_that_ran_and_failed_is_still_a_blocker(orch, monkeypatch):
    """The control: an executed workflow that failed is evidence, not an unknown."""
    failed = {"status": "failed", "boot": "ok", "backend": "backend", "responses": [],
              "assertion_failures": [{"request": 1, "expected": 200, "actual": 500}]}
    monkeypatch.setattr(api_probe, "probe_api_scenario", lambda *_a, **_k: dict(failed))

    orch._test_api({"requests": [{"method": "GET", "path": "/book/"}]})
    [finding] = [_classify_issue(m) for m in orch._collect_api_scenario_issues()]

    assert finding.severity == "blocker"
    assert finding.message.startswith("api scenario:")


def test_phase3_spends_no_repair_round_on_checks_that_did_not_run(orch, monkeypatch):
    """The cost a second reviewer measured: each such blocker bought a fix
    round that ended on the replay guard. The run ends unverified instead -
    the gate does not call it complete, and the runner does not call it
    failed."""
    _sandbox_unavailable(monkeypatch)

    with patch.object(orch, "_collect_validation_issues", side_effect=lambda: _classified(orch)), \
            patch.object(orch, "_create_snapshot"), \
            patch.object(orch, "_invoke_phase3_fix_loop") as fix:
        orch._run_phase3_validation()

    fix.assert_not_called()
    assert not [i for i in orch._validation_issues if i.message.startswith("runtime gate:")]
    completion = [i.message for i in orch._validation_issues if is_completion_issue(i)]
    assert completion, "a run whose runtime checks did not run is not verified complete"
    assert unresolved_defects(completion) == [], "an unknown is not a defect"


def test_a_stale_verdict_beside_a_check_that_did_not_run_stays_unverified(orch):
    """Parity with the ``runtime unverified:`` finding this replaces: with no
    verdict for the shipped tree, a did-not-run finding already says why, so
    the gate does not add a "no runtime evidence" refusal that would turn an
    unknown into an incomplete run."""
    from besser.spec_driven_agent.validation.issues import runtime_check_unverified

    orch._runtime_probe_verdict = None
    unrun = _classify_issue(runtime_check_unverified("the constructibility probe", _NO_SANDBOX))
    orch._validation_issues = [unrun]

    orch._apply_runtime_gate()

    assert orch._validation_issues == [unrun]


def test_validate_app_does_not_call_an_unrun_probe_verified(orch, monkeypatch):
    _sandbox_unavailable(monkeypatch)

    result = orch._validate_app()

    assert result["verified"] is False
    assert result["blocker_count"] == 0
    assert "did not run" in result["next_step"]


def test_checks_disabled_by_configuration_are_an_unverified_warning(simple_library_book_model, tmp_path):
    """``enable_import_smoke_check=False`` on an app with a backend used to be a
    ``runtime unverified:`` blocker: every fix round was paid for a finding no
    edit can settle, because the run's own configuration is what skipped the check."""
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "main_api.py").write_text("app = None\n", encoding="utf-8")
    orchestrator = LLMOrchestrator(
        llm_client=_Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
        enable_tracing=False, enable_import_smoke_check=False,
    )

    issues = orchestrator._collect_execution_issues()

    assert len(issues) == 1
    finding = _classify_issue(issues[0])
    assert finding.severity == "warning"
    assert not issues[0].startswith("runtime unverified:")
    assert "disabled" in issues[0]
    assert is_completion_issue(finding), "a disabled check still withholds 'verified'"
    assert unresolved_defects(issues) == []
    assert orchestrator._runtime_probe_verdict is None
