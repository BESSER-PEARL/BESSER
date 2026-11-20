"""The run's success/incomplete verdict must respect Phase-3 blockers.

Before this, `incomplete` keyed ONLY on Phase 2 emitting `end_turn`, so an app
that parsed but had an unfixed blocker-class issue shipped as a green success.
Now unresolved implementation issues mark the run incomplete, without assuming
every blocker means the application cannot start.

A finding that says "we could not CHECK"
no longer sets the verdict. The unverified family carries blocker severity, so a
run reporting "Nothing we checked was found missing from the delivered code",
3 verified and 4 could-not-verify was still headlined "Generated - incomplete".
"Incomplete" is now reserved for output that does not work. A finding that is
not confirmed is not counted in `blockerCount` either; it stays in the report's
could-not-verify bucket.
"""
from __future__ import annotations

import asyncio

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import ValidationIssue
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _FakeClient,
    _StubOrchestrator,
    _build_request,
    _collect_frames,
    _parse,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime


class _BlockerOrchestrator(_StubOrchestrator):
    """Phase 2 finishes cleanly, but Phase 3 leaves an unfixed blocker."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validation_issues = [
            ValidationIssue(severity="blocker", message="requirements: R1 has no implementation evidence"),
            ValidationIssue(severity="style", message="ruff: unused import"),  # ignored
        ]


def _cleanup():
    async def _c():
        import shutil
        async with current_runtime().registry._lock:
            for e in list(current_runtime().registry._entries.values()):
                shutil.rmtree(e.temp_dir, ignore_errors=True)
            current_runtime().registry._entries.clear()
    asyncio.run(_c())


def _done_event(runner):
    frames = asyncio.run(_collect_frames(runner))
    done = [_parse(f) for f in frames if _parse(f).get("event") == "done"]
    assert done, "no done event emitted"
    return done[-1]


def test_unfixed_phase3_blocker_marks_run_incomplete(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _BlockerOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request(), runtime=current_runtime()))
        assert done["incomplete"] is True
        reason = (done.get("incompleteReason") or "").lower()
        assert "unresolved implementation or verification" in reason
        assert "not verified complete" in reason
        assert "likely stop it from running" not in reason
    finally:
        _cleanup()


@pytest.mark.parametrize("required_check", [False, True])
def test_no_blockers_stays_complete(monkeypatch, required_check):
    from besser.spec_driven_agent.validation.issues import _check_did_not_run, required_check_unverified

    class SkippedCheck(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            message = (required_check_unverified("frontend build [.]", "disabled") if required_check
                       else _check_did_not_run("ruff", "unavailable"))
            self._validation_issues = [ValidationIssue("warning", message)]

    monkeypatch.setattr(runner_module, "LLMOrchestrator", SkippedCheck)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request(), runtime=current_runtime()))
        # A check that could not RUN is not confirmed, not a defect: it does not
        # set the verdict. On a finished run blockerCount keeps the released
        # editors' meaning, "could not be verified", so it counts there.
        assert done["incomplete"] is False
        assert done["notConfirmedCount"] == int(required_check)
        assert done["blockerCount"] == int(required_check)
    finally:
        _cleanup()


@pytest.mark.parametrize("bound_target", [None, "generate_web_app", "generate_fastapi_backend"])
def test_failed_diagram_is_visible_and_bound_missing_inputs_fail_before_client(monkeypatch, bound_target):
    from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
    from besser.utilities.web_modeling_editor.backend.services.spec_driven import model_assembly

    request = _build_request(target_generator_override=bound_target)
    request.project.diagrams["GUINoCodeDiagram"] = [DiagramInput(
        id="failed-gui", title="Screen specification", model={"type": "GUINoCodeDiagram"},
    )]
    def reject_gui(*args):
        raise ValueError("secret-api-key=NEVER-PUBLISH-THIS")
    monkeypatch.setattr(model_assembly, "process_gui_diagram", reject_gui)
    clients, observed = [], []
    def client(**kwargs):
        clients.append(True)
        return _FakeClient()
    class CaptureAssembly(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            observed.extend(kwargs["assembly_issues"])
            # Even an implementation ignoring the new field cannot cause the
            # runner to announce complete after dropping project input.
            self._validation_issues = []
    monkeypatch.setattr(runner_module, "create_llm_client", client)
    monkeypatch.setattr(runner_module, "LLMOrchestrator", CaptureAssembly)
    try:
        events = [_parse(frame) for frame in asyncio.run(_collect_frames(SmartGenerationRunner(request, runtime=current_runtime())))]
        assert "NEVER-PUBLISH-THIS" not in repr(events)
        if bound_target == "generate_web_app":
            assert clients == [] and observed == [], "missing required models must stop before paid generation"
            assert not any(event["event"] == "done" for event in events)
            rejection = next(event for event in events if event.get("code") == "BAD_REQUEST")
            assert "generate_web_app" in rejection["message"] and "failed-gui" in rejection["message"]
        else:
            assert clients == [True]
            assert observed == [{"diagram_id": "failed-gui", "diagram_type": "GUINoCodeDiagram",
                                 "diagnostic": "processor_failed: ValueError"}]
            assert any(event.get("code") == "INCOMPLETE" and "failed-gui" in event["message"] for event in events)
            assert any(event["event"] == "phase_update" and "assembly_issues" in event["details"] for event in events)
            done = next(event for event in events if event["event"] == "done")
            assert done["incomplete"] is True and done["blockerCount"] == 1
            assert "First: Not run: model assembly (GUINoCodeDiagram [failed-gui]" in done["incompleteReason"]
            assert "validation unverified" not in done["incompleteReason"]
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# A cap breach detected by the RUNNER must also mark the run incomplete.
#
# The runner measures elapsed/spend across the whole run (Phase 1 + 2 + 3 +
# packaging) and emits TIMEOUT / COST_CAP from that. `incomplete`, computed a
# few lines later, keyed only on the orchestrator's Phase 2 state -- so a run
# whose Phase 2 finished cleanly but whose WALL CLOCK blew the cap emitted
# "Runtime cap reached ... Output may be incomplete" and then reported
# `incomplete: False, incompleteReason: None`.
# ---------------------------------------------------------------------------
class _SlowButCleanOrchestrator(_StubOrchestrator):
    """Phase 2 exits cleanly; the run as a whole overruns its time cap.

    The cap is zeroed AFTER ``super().__init__`` on purpose: the runner reads
    it back off the orchestrator (``getattr(orchestrator, "max_runtime_seconds",
    ...)``), and the base stub sets it from the request, which would otherwise
    overwrite a class attribute. With a cap of 0, any real elapsed time is a
    breach -- no clock monkeypatching needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Negative, not 0: the stub run can complete inside a single monotonic
        # tick, making elapsed exactly 0.0, and `0.0 > 0` is False -- which made
        # this test flaky (1 in 2 runs). Any elapsed beats a negative cap.
        self.max_runtime_seconds = -1


def test_runtime_cap_breach_marks_run_incomplete(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _SlowButCleanOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request(), runtime=current_runtime()))
        assert done["incomplete"] is True, (
            "a run stopped by the time cap must not report itself complete"
        )
        reason = (done.get("incompleteReason") or "").lower()
        assert "cap" in reason or "time" in reason, reason
    finally:
        _cleanup()


def test_a_run_inside_its_caps_stays_complete(monkeypatch):
    """Guard: the cap check must not mark ordinary runs incomplete."""
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request(), runtime=current_runtime()))
        assert done["incomplete"] is False
        assert done.get("incompleteReason") in (None, "")
    finally:
        _cleanup()


_NOT_CONFIRMED = (
    "requirement unverified: R1 — The order screen accepts a CSV attachment — the cited evidence "
    "'frontend/src/pages/OrdersPage.tsx: accept=«.csv»' does not verify it",
    "runtime unverified: create unverified: backend: POST /orders/ - autogenerated input was refused "
    "(422: reference too short); attempted input: {}",
)


@pytest.mark.parametrize("with_missing", [False, True])
def test_findings_that_are_not_confirmed_never_block_or_count(monkeypatch, with_missing):
    """The judge said implemented but its citation was not located, and the create probe's guessed
    POST was refused: neither observed a failure. Before: both were blockers in blockerCount."""
    from besser.spec_driven_agent.validation.issues import _classify_issue

    missing = "requirement: R2 — An order can be cancelled is not implemented"

    class NotConfirmed(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._validation_issues = [_classify_issue(m) for m in _NOT_CONFIRMED + ((missing,) if with_missing else ())]

    monkeypatch.setattr(runner_module, "LLMOrchestrator", NotConfirmed)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request(), runtime=current_runtime()))
        assert done["incomplete"] is False
        assert done["notConfirmedCount"] == 2
        # The legacy count of a finished run: blockers plus findings not confirmed.
        assert done["blockerCount"] == 2 + int(with_missing)
        run = done["recipe"]["run"]
        assert run["counts"]["not_confirmed"] == 2
        assert run["counts"]["blockers"] == int(with_missing)
        if not with_missing:
            assert run["outcome"] == "completed"
            assert run["stop_message"] == ("The app was generated; 2 finding(s) could not be confirmed "
                                           "and are listed for review.")
    finally:
        _cleanup()


@pytest.mark.parametrize("findings, defect", [((), False), (("runtime",), False), ((), True)])
def test_a_class_of_checks_that_never_ran_is_not_verified(monkeypatch, findings, defect):
    """No backend was ever started: not "Application ready", and not "incomplete" either."""
    from besser.spec_driven_agent.validation.issues import _classify_issue, runtime_check_unverified

    messages = [runtime_check_unverified("backend startup", "timed out") for _ in findings]
    if defect:
        messages.append("Syntax error in backend/main_api.py: line 3")

    class NeverStarted(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._validation_issues = [_classify_issue(m) for m in messages]

        def verification_gaps(self):
            return [("runtime", "its startup timed out")]

    monkeypatch.setattr(runner_module, "LLMOrchestrator", NeverStarted)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request(), runtime=current_runtime()))
        run = done["recipe"]["run"]
        assert done["notConfirmedCount"] == len(findings)
        assert run["not_verified"] == [{"check": "runtime", "reason": "its startup timed out"}]
        if defect:
            # A known defect keeps its own headline; the state is not reported twice.
            assert done["incomplete"] is True and done["unverified"] is False
            assert done["unverifiedReason"] is None and done["blockerCount"] == 1
            assert run["outcome"] == "completed_with_blockers"
        else:
            assert done["incomplete"] is False and done["unverified"] is True
            assert done["unverifiedReason"] == ("The app was generated, but it was never started, so "
                                                "nothing shows it runs (its startup timed out).")
            # Released editors show "· N could not be verified" from this: never 0 here.
            assert done["blockerCount"] == 1
            assert run["outcome"] == "completed_unverified"
            assert run["stop_message"] == done["unverifiedReason"]
    finally:
        _cleanup()


def test_a_dependency_install_left_undone_is_not_an_unresolved_blocker(monkeypatch):
    from besser.spec_driven_agent.validation.issues import _classify_issue, required_dependency_setup

    class SetupOpen(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._validation_issues = [_classify_issue(required_dependency_setup("tsc", "frontend"))]

        def verification_gaps(self):
            return [("build", "its dependencies were not installed")]

    monkeypatch.setattr(runner_module, "LLMOrchestrator", SetupOpen)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request(), runtime=current_runtime()))
        run = done["recipe"]["run"]
        assert done["incomplete"] is False and done["unverified"] is True
        assert run["counts"]["blockers"] == 0 and run["counts"]["not_confirmed"] == 1
        assert run["outcome"] == "completed_unverified"
    finally:
        _cleanup()
