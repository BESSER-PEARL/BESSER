"""Tests for mid-run model-switch visibility.

The provider's outage fallback (``OpenAIProvider._activate_fallback``)
swaps the client's model sticky-for-the-run *inside* a chat call. The
orchestrator surfaces that by comparing ``self.client.model`` after each
LLM call against the last value it saw (``_notify_model_switch``) and
emitting the ``__model_switch__`` sentinel through ``on_progress`` —
which the SSE runner translates into a ``model_update`` event so the run
card can show the model actually serving the run.
"""

from __future__ import annotations

import pytest

from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator


class _MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockClient:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")
        self.chat_calls = 0

    def chat(self, system, messages, tools):
        self.chat_calls += 1
        return {"stop_reason": "end_turn", "content": [_MockBlock("text", text="Done")]}


class _FallbackSwitchingClient(_MockClient):
    """Swaps its model during the first chat() call, the way
    ``_activate_fallback`` does when the primary endpoint is down."""

    def chat(self, system, messages, tools):
        if self.chat_calls == 0:
            self.model = "fallback-model"
        return super().chat(system, messages, tools)


def _make_orchestrator(tmp_path, client):
    class _SM:
        name = "DummySM"

    return LLMOrchestrator(
        llm_client=client,
        state_machines=[_SM()],
        output_dir=str(tmp_path),
    )


# ----------------------------------------------------------------------
# _notify_model_switch unit behaviour
# ----------------------------------------------------------------------


def test_notify_emits_sentinel_once_on_change(tmp_path):
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    progress: list[tuple] = []
    orch.on_progress = lambda *a: progress.append(a)

    # No change — no emission.
    orch._notify_model_switch()
    assert progress == []

    client.model = "fallback-model"
    orch._notify_model_switch()
    assert progress == [(0, "__model_switch__", "fallback-model")]

    # Sticky — calling again without a further change stays silent.
    orch._notify_model_switch()
    assert len(progress) == 1


def test_notify_is_safe_without_on_progress(tmp_path):
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    assert orch.on_progress is None
    client.model = "fallback-model"
    orch._notify_model_switch()  # must not raise
    assert orch._last_seen_model == "fallback-model"


def test_notify_swallows_on_progress_errors(tmp_path):
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)

    def boom(*_a):
        raise RuntimeError("downstream had a bad day")

    orch.on_progress = boom
    client.model = "fallback-model"
    orch._notify_model_switch()  # must not raise
    assert orch._last_seen_model == "fallback-model"


# ----------------------------------------------------------------------
# Phase-2 loop integration
# ----------------------------------------------------------------------


def test_phase2_loop_reports_switch_after_llm_call(tmp_path, monkeypatch):
    """A model swap that happens inside chat() (fallback activation) is
    reported right after the call returns."""
    client = _FallbackSwitchingClient()
    orch = _make_orchestrator(tmp_path, client)
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: None,
    )
    progress: list[tuple] = []
    orch.on_progress = lambda *a: progress.append(a)

    orch._run_phase2("build a library api", extra_issues=[])

    switches = [a for a in progress if a[1] == "__model_switch__"]
    assert switches == [(0, "__model_switch__", "fallback-model")]
    assert orch._phase2_exited_cleanly is True


def test_phase2_loop_silent_when_model_stable(tmp_path, monkeypatch):
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: None,
    )
    progress: list[tuple] = []
    orch.on_progress = lambda *a: progress.append(a)

    orch._run_phase2("build a library api", extra_issues=[])

    assert not any(a[1] == "__model_switch__" for a in progress)
