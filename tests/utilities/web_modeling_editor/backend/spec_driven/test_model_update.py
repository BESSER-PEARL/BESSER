"""Runner-side tests for mid-run model-switch visibility + blockerCount.

Covers two contracts added for honest run reporting:

* The orchestrator's ``__model_switch__`` on_progress sentinel (emitted
  when the provider's outage fallback swaps the model mid-run) is
  translated into a ``model_update`` SSE event carrying the new model,
  the previous model (seeded from the start event), and a machine
  ``reason`` the frontend maps to copy.

* ``DoneEvent.blockerCount`` distinguishes "the loop COMPLETED but left
  blocker-severity issues" (count > 0 — the run did *not* stop early)
  from "the loop was genuinely cut short" (count == 0 — clients keep
  their cut-short framing).

The orchestrator is stubbed (same harness as ``test_modify_seed.py``)
so the runner's queue bridge is exercised without a real LLM.
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


def _cleanup():
    async def _c():
        import shutil

        async with SMART_RUN_REGISTRY._lock:
            for e in list(SMART_RUN_REGISTRY._entries.values()):
                shutil.rmtree(e.temp_dir, ignore_errors=True)
            SMART_RUN_REGISTRY._entries.clear()

    asyncio.run(_c())


def _events(runner: SmartGenerationRunner) -> list[dict]:
    frames = asyncio.run(_collect_frames(runner))
    return [_parse(f) for f in frames]


# ---------------------------------------------------------------------
# __model_switch__ sentinel → model_update SSE event
# ---------------------------------------------------------------------


class _SwitchingOrchestrator(_StubOrchestrator):
    """Reports a mid-run model switch through on_progress, like the
    orchestrator does after ``OpenAIProvider._activate_fallback``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._on_progress = kwargs.get("on_progress")

    def run(self, instructions: str) -> str:
        if self._on_progress:
            self._on_progress(0, "__model_switch__", "qwen3-coder:30b")
        return self._finish("run")


def test_model_switch_sentinel_becomes_model_update_event(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _SwitchingOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        events = _events(SmartGenerationRunner(_build_request()))
        updates = [e for e in events if e.get("event") == "model_update"]
        assert len(updates) == 1
        update = updates[0]
        assert update["model"] == "qwen3-coder:30b"
        # previousModel is seeded from the start event's model.
        start = next(e for e in events if e.get("event") == "start")
        assert update["previousModel"] == start["llmModel"]
        assert update["reason"] == "primary_unavailable"
        # The switch must not derail the run — done still arrives.
        assert any(e.get("event") == "done" for e in events)
    finally:
        _cleanup()


def test_no_switch_means_no_model_update_event(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        events = _events(SmartGenerationRunner(_build_request()))
        assert not any(e.get("event") == "model_update" for e in events)
    finally:
        _cleanup()


# ---------------------------------------------------------------------
# DoneEvent.blockerCount
# ---------------------------------------------------------------------


class _CompletedWithBlockersOrchestrator(_StubOrchestrator):
    """Phase 2 finished cleanly; Phase 3 left two unfixed blockers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validation_issues = [
            ValidationIssue(severity="blocker", message="syntax [app.py]: invalid syntax"),
            ValidationIssue(severity="blocker", message="import [main.py]: no module named x"),
            ValidationIssue(severity="style", message="ruff: unused import"),  # ignored
        ]


class _CutShortOrchestrator(_StubOrchestrator):
    """Phase 2 was cut short (turn cap); blockers must NOT be counted —
    the client's cut-short framing wins over blocker framing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._phase2_exited_cleanly = False
        self._phase2_stop_reason = "max_turns"
        self._validation_issues = [
            ValidationIssue(severity="blocker", message="syntax [app.py]: invalid syntax"),
        ]


def _done_event(events: list[dict]) -> dict:
    done = [e for e in events if e.get("event") == "done"]
    assert done, "no done event emitted"
    return done[-1]


def test_completed_with_blockers_sets_blocker_count(monkeypatch):
    monkeypatch.setattr(
        runner_module, "LLMOrchestrator", _CompletedWithBlockersOrchestrator
    )
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(_events(SmartGenerationRunner(_build_request())))
        assert done["incomplete"] is True
        assert done["blockerCount"] == 2
    finally:
        _cleanup()


def test_cut_short_run_reports_zero_blocker_count(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _CutShortOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(_events(SmartGenerationRunner(_build_request())))
        assert done["incomplete"] is True
        assert done["blockerCount"] == 0
    finally:
        _cleanup()


def test_clean_run_reports_zero_blocker_count(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    try:
        done = _done_event(_events(SmartGenerationRunner(_build_request())))
        assert done["incomplete"] is False
        assert done["blockerCount"] == 0
    finally:
        _cleanup()
