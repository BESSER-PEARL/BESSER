"""Tests for the Phase 1 skip event and PhaseUpdateEvent — #1 + #5.

When the orchestrator's Phase 1 is skipped (no class diagram / no
quantum circuit, or no matching deterministic generator), the smart-gen
card used to silently jump from `select` to `gap`, making users wonder
if a step was missing. The fix:

* ``orchestrator._run_phase1`` now calls ``on_progress(0, "__skipped__", <reason>)``
  for both skip branches.
* The runner's on_progress handler turns that into a real
  ``PhaseEvent(phase="generate", message="skipped — ...")``.

Separately, the gap analyser used to emit only a coarse "Analysing gaps"
phase marker — users couldn't see what the planner had actually decided.
The fix adds ``PhaseUpdateEvent(phase="gap", details=<task list>)`` emitted
after the planning LLM call returns, surfaced behind the chevron on the
smart-gen card.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from besser.generators.llm import gap_analyzer
from besser.utilities.web_modeling_editor.backend.services.smart_generation.sse_events import (
    PhaseEvent,
    PhaseUpdateEvent,
    format_sse,
)


# ======================================================================
# PhaseUpdateEvent schema
# ======================================================================


class TestPhaseUpdateEvent:
    def test_minimum_fields(self):
        ev = PhaseUpdateEvent(phase="gap", details="task list...")
        assert ev.event == "phase_update"
        assert ev.phase == "gap"
        assert ev.details == "task list..."
        assert ev.message is None

    def test_with_optional_message(self):
        ev = PhaseUpdateEvent(phase="gap", details="x", message="Analysing gaps")
        assert ev.message == "Analysing gaps"

    def test_sse_serialization_has_event_header(self):
        frame = format_sse(PhaseUpdateEvent(phase="gap", details="x")).decode()
        assert frame.startswith("event: phase_update\n")
        assert "\ndata: " in frame
        body_line = next(
            line for line in frame.split("\n") if line.startswith("data: ")
        )
        payload = json.loads(body_line[len("data: "):])
        assert payload["event"] == "phase_update"
        assert payload["phase"] == "gap"
        assert payload["details"] == "x"

    def test_rejects_unknown_phase(self):
        """The PhaseName Literal must reject phases the frontend can't render."""
        with pytest.raises(Exception):
            PhaseUpdateEvent(phase="bogus_phase", details="x")  # type: ignore[arg-type]


# ======================================================================
# gap_analyzer._emit_phase_details
# ======================================================================


class TestGapAnalyzerPhaseDetails:
    def test_emits_task_list_as_markdown_bullets(self):
        calls: list[tuple[str, str]] = []

        def capture(phase: str, details: str) -> None:
            calls.append((phase, details))

        gap_analyzer._emit_phase_details(
            capture,
            ["Add JWT middleware to main_api.py", "Write pytest fixture", "Delete schemas.py"],
        )

        assert len(calls) == 1
        phase, details = calls[0]
        assert phase == "gap"
        # The card renders this as markdown — bullet list is what we want.
        assert "Identified 3 tasks" in details
        assert "- Add JWT middleware to main_api.py" in details
        assert "- Write pytest fixture" in details
        assert "- Delete schemas.py" in details

    def test_singular_task_uses_singular_label(self):
        calls: list[tuple[str, str]] = []
        gap_analyzer._emit_phase_details(
            lambda p, d: calls.append((p, d)),
            ["Just one thing"],
        )
        assert len(calls) == 1
        _, details = calls[0]
        assert "Identified 1 task" in details
        assert "Identified 1 tasks" not in details  # singular only
        assert "- Just one thing" in details

    def test_no_callback_is_safe(self):
        # ``None`` callback should be a no-op, never raise.
        gap_analyzer._emit_phase_details(None, ["task"])

    def test_empty_task_list_does_not_fire(self):
        calls: list[tuple[str, str]] = []
        gap_analyzer._emit_phase_details(
            lambda p, d: calls.append((p, d)),
            [],
        )
        assert calls == []

    def test_fallback_label_emits_no_generator_blurb(self):
        calls: list[tuple[str, str]] = []
        gap_analyzer._emit_phase_details(
            lambda p, d: calls.append((p, d)),
            ["build everything from scratch"],
            fallback_label=True,
        )
        assert len(calls) == 1
        phase, details = calls[0]
        assert phase == "gap"
        assert "no deterministic generator" in details.lower()
        # Fallback uses a prose summary, not a bullet list.
        assert "scaffold" in details.lower()

    def test_callback_exceptions_are_swallowed(self):
        def boom(_phase: str, _details: str) -> None:
            raise RuntimeError("downstream error")

        # Should not raise — gap analysis must never break the run because
        # the SSE emitter had a bad day.
        gap_analyzer._emit_phase_details(boom, ["task"])
