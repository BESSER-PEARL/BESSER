"""What the stream carries for the run card's trace view (#649).

* Each ``phase`` event carries the run's cost, turns and elapsed time when the
  phase started, so the card can show what each phase cost and how long it took.
* Each Phase 3 validation round becomes a ``findings`` event; the tool calls
  between two of them are that fix attempt's repairs.
* A fix attempt's tool calls stay under the ``validate`` row. A run whose
  Phase 2 was skipped used to open a "customize" row in the middle of validation
  at the first fix-loop call, so the repairs were listed as customization.
"""

from __future__ import annotations

import asyncio
import time

from besser.spec_driven_agent.pipeline.orchestrator import ValidationIssue
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _FakeClient,
    _StubOrchestrator,
    _build_request,
    _collect_frames,
    _parse,
)


class _RepairingOrchestrator(_StubOrchestrator):
    """Skips Phase 2, finds two blockers, repairs one per attempt."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._on_progress = kwargs["on_progress"]
        self._on_findings = kwargs["on_findings"]

    def run(self, instructions: str) -> str:
        progress, findings = self._on_progress, self._on_findings
        progress(0, "generate_fastapi_backend", "generating")
        progress(0, "__customize_skipped__", "the scaffold covers the request")
        self.client.usage.estimated_cost = 0.25
        self.total_turns = 0
        progress(0, "__validating__", "")
        # Spend that lands after Phase 3 started; the pause lets the consumer
        # serialize the validate event first, so a late stamp would pick it up.
        self.client.usage.estimated_cost = 0.9
        time.sleep(0.2)
        progress(0, "validation", "2 blockers / 3 total")
        findings(0, [
            ValidationIssue("blocker", "import [backend/main.py]: " + "x" * 600),
            ValidationIssue("blocker", "Syntax error in backend/models.py line 3: invalid syntax"),
            ValidationIssue("warning", "ruff: unused import"),
        ])
        self.total_turns = 1
        progress(1, "read_file", "executing", "path=backend/main.py")
        progress(1, "modify_file", "executing", "path=backend/main.py")
        findings(1, [ValidationIssue("blocker", "Syntax error in backend/models.py line 3: invalid syntax")])
        self.total_turns = 2
        progress(2, "modify_file", "executing", "path=backend/models.py")
        findings(2, [])
        return self._finish("run")


def _events(monkeypatch) -> list[dict]:
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _RepairingOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    try:
        return [_parse(frame) for frame in asyncio.run(_collect_frames(runner))]
    finally:
        async def _clear():
            import shutil

            async with current_runtime().registry._lock:
                for entry in list(current_runtime().registry._entries.values()):
                    shutil.rmtree(entry.temp_dir, ignore_errors=True)
                current_runtime().registry._entries.clear()

        asyncio.run(_clear())


def test_each_validation_round_is_a_findings_event(monkeypatch):
    findings = [e for e in _events(monkeypatch) if e.get("event") == "findings"]
    assert [f["attempt"] for f in findings] == [0, 1, 2]
    first = findings[0]
    assert first["blockerCount"] == 2 and first["issueCount"] == 3
    assert len(first["blockers"]) == 2
    # A long message is clipped, and says so.
    assert len(first["blockers"][0]) < 400 and first["blockers"][0].endswith("[truncated]")
    assert findings[1]["blockers"] == ["Syntax error in backend/models.py line 3: invalid syntax"]
    assert findings[2]["blockerCount"] == 0 and findings[2]["restored"] is False


def test_findings_list_is_capped_but_counts_are_true(monkeypatch):
    class _Many(_StubOrchestrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._on_findings = kwargs["on_findings"]

        def run(self, instructions: str) -> str:
            self._on_findings(0, [ValidationIssue("blocker", f"tsc: error {n}") for n in range(45)])
            return self._finish("run")

    monkeypatch.setattr(runner_module, "LLMOrchestrator", _Many)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
    runner = SmartGenerationRunner(_build_request(), runtime=current_runtime())
    events = [_parse(frame) for frame in asyncio.run(_collect_frames(runner))]
    [event] = [e for e in events if e.get("event") == "findings"]
    assert event["blockerCount"] == 45
    assert len(event["blockers"]) == runner_module._MAX_FINDINGS_LISTED


def test_repairs_stay_under_the_validate_row(monkeypatch):
    events = _events(monkeypatch)
    phases = [e["phase"] for e in events if e.get("event") == "phase"]
    assert phases[-1] == "validate"
    # The skipped customize row is the only one: no second one opened by the fix loop.
    assert phases.count("customize") == 1
    last_phase = max(i for i, e in enumerate(events) if e.get("event") == "phase")
    tool_calls = [i for i, e in enumerate(events) if e.get("event") == "tool_call"]
    assert tool_calls and min(tool_calls) > last_phase


def test_validate_row_opens_before_the_first_check_and_carries_its_result(monkeypatch):
    """The first check can take minutes; announced only after it, that time read as customization."""
    events = _events(monkeypatch)
    opened = next(i for i, e in enumerate(events) if e.get("event") == "phase" and e["phase"] == "validate")
    assert events[opened]["message"] == "Validating"
    result = next(i for i, e in enumerate(events) if e.get("event") == "phase_update" and e["phase"] == "validate")
    assert result > opened and events[result]["message"] == "2 blockers / 3 total"
    first_findings = next(i for i, e in enumerate(events) if e.get("event") == "findings")
    assert first_findings > opened


def test_phase_events_carry_the_run_totals_at_their_start(monkeypatch):
    phases = [e for e in _events(monkeypatch) if e.get("event") == "phase"]
    for phase in phases:
        assert isinstance(phase["elapsedSeconds"], (int, float)) and phase["elapsedSeconds"] >= 0
        assert isinstance(phase["turns"], int)
        assert isinstance(phase["usd"], (int, float))
    select, validate = phases[0], phases[-1]
    assert select["phase"] == "select" and select["usd"] == 0 and select["turns"] == 0
    # The totals when the phase started, not when the frame was sent (run on Linux CI: 0.01).
    assert validate["phase"] == "validate" and validate["usd"] == 0.25
    elapsed = [p["elapsedSeconds"] for p in phases]
    assert elapsed == sorted(elapsed)
