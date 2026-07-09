"""F2: the run's success/incomplete verdict must respect Phase-3 blockers.

Before this, `incomplete` keyed ONLY on Phase 2 emitting `end_turn`, so an app
that parsed but had an unfixed blocker-class issue (syntax/import/dependency —
the "won't compile / won't boot" class) shipped as a green success. Now an
unfixed blocker-severity ValidationIssue marks the run incomplete with a reason.
"""
from __future__ import annotations

import asyncio

from besser.generators.llm.orchestrator import ValidationIssue
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_modify_seed import (
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
