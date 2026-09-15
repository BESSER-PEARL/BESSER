"""F2: the run's success/incomplete verdict must respect Phase-3 blockers.

Before this, `incomplete` keyed ONLY on Phase 2 emitting `end_turn`, so an app
that parsed but had an unfixed blocker-class issue (syntax/import/dependency —
the "won't compile / won't boot" class) shipped as a green success. Now an
unfixed blocker-severity ValidationIssue marks the run incomplete with a reason.
"""
from __future__ import annotations

import asyncio

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
            ValidationIssue(severity="blocker", message="syntax [main_api.py]: invalid syntax"),
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
        assert "blocker" in reason
    finally:
        _cleanup()


def test_no_blockers_stays_complete(monkeypatch):
    # Regression guard: the plain stub (no _validation_issues) stays complete —
    # the new blocker check must not make clean runs report incomplete.
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(SmartGenerationRunner(_build_request()))
        assert done["incomplete"] is False
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
