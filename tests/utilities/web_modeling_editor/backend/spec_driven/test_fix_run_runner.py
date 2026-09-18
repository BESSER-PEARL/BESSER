"""Runner-side tests for the FIX/MODIFY behaviour.

Covers three runner responsibilities:

  * a seeded modify run emits a truthful "Loading your app" phase instead
    of "Selecting generator" (nothing is being selected on a modify run);
  * an orchestrator that could not confirm the reported failure is fixed
    surfaces its honest, target-specific message as the run's
    ``incompleteReason`` (not a clean success);
  * the run trace is copied into the host-mounted incident/telemetry dir
    keyed by run id, so it survives the temp-workspace sweep.

The orchestrator is stubbed (same pattern as ``test_modify_seed`` /
``test_done_verdict``) so the runner's queue bridge and workspace handling
are exercised without a real LLM.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from besser.generators.llm.orchestrator import ValidationIssue
from besser.generators.llm.tracing import TRACE_FILENAME
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SMART_RUN_REGISTRY,
    SmartGenerationRunner,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _FakeClient,
    _StubOrchestrator,
    _build_request,
    _collect_frames,
    _make_base_dir,
    _parse,
)


def _cleanup_registry():
    async def _c():
        import shutil

        async with SMART_RUN_REGISTRY._lock:
            for entry in list(SMART_RUN_REGISTRY._entries.values()):
                shutil.rmtree(entry.temp_dir, ignore_errors=True)
            SMART_RUN_REGISTRY._entries.clear()

    asyncio.run(_c())


@pytest.fixture(autouse=True)
def _clean():
    _cleanup_registry()
    yield
    _cleanup_registry()


def _frames(runner):
    return [_parse(f) for f in asyncio.run(_collect_frames(runner))]


def _select_messages(parsed):
    return [
        p["message"]
        for p in parsed
        if p.get("event") == "phase" and p.get("phase") == "select"
    ]


# ----------------------------------------------------------------------
# Truthful phase copy on a modify run
# ----------------------------------------------------------------------


def test_seeded_modify_run_emits_loading_your_app(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

    base_dir = _make_base_dir(tmp_path)
    base_id = "a" * 32
    asyncio.run(SMART_RUN_REGISTRY.put(
        base_id,
        SmartRunEntry(
            file_path=os.path.join(base_dir, "app", "main.py"),
            file_name="main.py",
            is_zip=False,
            temp_dir=base_dir,
            created_at=time.time(),
        ),
    ))

    request = _build_request(
        mode="modify", base_run_id=base_id,
        instructions="It 500s on save, please fix it",
    )
    runner = SmartGenerationRunner(request, base_run_id=base_id, mode="modify")
    parsed = _frames(runner)

    assert _select_messages(parsed) == ["Loading your app"]
    assert parsed[-1]["event"] == "done"


def test_from_scratch_run_emits_selecting_generator(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _StubOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

    runner = SmartGenerationRunner(_build_request())
    parsed = _frames(runner)

    assert _select_messages(parsed) == ["Selecting generator"]


# ----------------------------------------------------------------------
# Honest incomplete message for an unresolved reported failure
# ----------------------------------------------------------------------


class _UnresolvedFixOrchestrator(_StubOrchestrator):
    """Loop finished cleanly, but the reported failure was not confirmed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validation_issues = [
            ValidationIssue(
                "blocker",
                "reported-failure: acceptance: entity Watchlist — "
                "no frontend POST for it",
            )
        ]
        self._fix_target_message = (
            "I changed the app, but could not confirm the reported failure "
            "is fixed (POST /createWatchlist returning HTTP 400). Outstanding "
            "issue: acceptance: entity Watchlist — no frontend POST for it"
        )


def test_unresolved_fix_target_surfaces_honest_incomplete(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _UnresolvedFixOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

    runner = SmartGenerationRunner(_build_request(instructions="fix the 400"))
    parsed = _frames(runner)

    done = [p for p in parsed if p.get("event") == "done"][-1]
    assert done["incomplete"] is True
    assert "could not confirm the reported failure" in (done["incompleteReason"] or "")

    incomplete_events = [
        p for p in parsed
        if p.get("event") == "error" and p.get("code") == "INCOMPLETE"
    ]
    assert any(
        "could not confirm the reported failure" in p["message"]
        for p in incomplete_events
    )
    # The honest message must replace the generic compile/boot wording.
    assert not any(
        "syntax / import / dependency" in p["message"] for p in incomplete_events
    )


# ----------------------------------------------------------------------
# Trace persistence to the host-mounted incident dir
# ----------------------------------------------------------------------


class _TraceWritingOrchestrator(_StubOrchestrator):
    """Writes a run trace into its workspace, like the real TraceWriter."""

    def run(self, instructions: str) -> str:
        out = self._finish("run")
        with open(os.path.join(out, TRACE_FILENAME), "w", encoding="utf-8") as fh:
            fh.write('{"event":"run_start"}\n{"event":"run_end"}\n')
        return out


def test_run_trace_persisted_to_incident_dir(tmp_path, monkeypatch):
    incident_dir = tmp_path / "incidents"
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(incident_dir))
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _TraceWritingOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

    runner = SmartGenerationRunner(_build_request())
    parsed = _frames(runner)
    assert parsed[-1]["event"] == "done"

    persisted = incident_dir / "traces" / f"{runner.run_id}.besser_trace.jsonl"
    assert persisted.is_file(), "run trace was not persisted to the incident dir"
    assert "run_end" in persisted.read_text(encoding="utf-8")


def test_trace_persistence_is_noop_without_configured_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("BESSER_INCIDENT_LOG_DIR", raising=False)
    monkeypatch.delenv("BESSER_TELEMETRY_DIR", raising=False)
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _TraceWritingOrchestrator)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

    # Must not raise even though nothing is configured.
    runner = SmartGenerationRunner(_build_request())
    parsed = _frames(runner)
    assert parsed[-1]["event"] == "done"
