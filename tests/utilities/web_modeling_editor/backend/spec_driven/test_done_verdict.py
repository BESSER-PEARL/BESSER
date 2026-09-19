"""F2: the run's success/incomplete verdict must respect Phase-3 blockers.

Before this, `incomplete` keyed ONLY on Phase 2 emitting `end_turn`, so an app
that parsed but had an unfixed blocker-class issue shipped as a green success.
Now unresolved implementation or verification issues mark the run incomplete,
without assuming every blocker means the application cannot start.
"""
from __future__ import annotations

import asyncio

import pytest

from besser.generators.llm.orchestrator import ValidationIssue
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _FakeClient,
    _StubOrchestrator,
    _build_request,
    _collect_frames,
    _parse,
)


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
        async with SMART_RUN_REGISTRY._lock:
            for e in list(SMART_RUN_REGISTRY._entries.values()):
                shutil.rmtree(e.temp_dir, ignore_errors=True)
            SMART_RUN_REGISTRY._entries.clear()
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
        done = _done_event(SmartGenerationRunner(_build_request()))
        assert done["incomplete"] is True
        reason = (done.get("incompleteReason") or "").lower()
        assert "unresolved implementation or verification" in reason
        assert "not verified complete" in reason
        assert "likely stop it from running" not in reason
    finally:
        _cleanup()


@pytest.mark.parametrize("required_check", [False, True])
def test_no_blockers_stays_complete(monkeypatch, required_check):
    from besser.generators.llm.validation.issues import _check_did_not_run, required_check_unverified

    class SkippedCheck(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            message = (required_check_unverified("frontend build [.]", "disabled") if required_check
                       else _check_did_not_run("ruff", "unavailable"))
            self._validation_issues = [ValidationIssue("warning", message)]

    monkeypatch.setattr(runner_module, "LLMOrchestrator", SkippedCheck)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request()))
        assert done["incomplete"] is required_check
        assert done["blockerCount"] == int(required_check)
        if required_check:
            assert "not verified complete" in done["incompleteReason"]
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
        events = [_parse(frame) for frame in asyncio.run(_collect_frames(SmartGenerationRunner(request)))]
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
            assert "validation unverified: model assembly" in done["incompleteReason"]
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
#
# Observed live on 2026-09-15, run 932f1367: TIMEOUT at 1342s against a 1200s
# cap, 37 turns, followed by a done event claiming the run was complete.
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
        done = _done_event(SmartGenerationRunner(_build_request()))
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
        done = _done_event(SmartGenerationRunner(_build_request()))
        assert done["incomplete"] is False
        assert done.get("incompleteReason") in (None, "")
    finally:
        _cleanup()
