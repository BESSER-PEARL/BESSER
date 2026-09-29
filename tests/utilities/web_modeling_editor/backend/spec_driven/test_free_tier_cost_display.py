"""A free-tier run is capped on org credits, but the user is never shown a cost.

The keyless tier's paid models (``moonshotai/Kimi-K3``) are priced at list so
the per-run cap protects the org's Command Code credits. That estimate is not
the user's cost: the run card renders ``cost.usd <= 0`` as "No cost - this run
used the free tier", and anything above as "~$X estimated cost".
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_done_verdict import (
    _cleanup,
)
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _StubOrchestrator,
    _build_request,
    _collect_frames,
    _parse,
)

# Tokens of free run d3a33f95; ~$4.30 at the Kimi-K3 list rate, over a $1 cap.
_REVIEWED_RUN = SimpleNamespace(
    input_tokens=157_953, output_tokens=30_810,
    cache_creation_input_tokens=0, cache_read_input_tokens=1_121_920,
)


class _Orchestrator(_StubOrchestrator):
    """The base stub overwrites ``usage.estimated_cost``; keep the real tracker."""

    def _finish(self, method: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "NEW.md"), "w", encoding="utf-8") as fh:
            fh.write("# added by stub\n")
        return self.output_dir


class _Client:
    def __init__(self, billed_to_user: bool):
        self.usage = UsageTracker("moonshotai/Kimi-K3", billed_to_user=billed_to_user)
        self.usage.record(_REVIEWED_RUN)


def _events(monkeypatch, billed_to_user: bool) -> list[dict]:
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _Orchestrator)
    monkeypatch.setattr(
        runner_module, "create_llm_client", lambda **_: _Client(billed_to_user),
    )
    try:
        frames = asyncio.run(_collect_frames(SmartGenerationRunner(_build_request())))
        return [_parse(f) for f in frames]
    finally:
        _cleanup()


def test_a_free_tier_run_shows_no_cost_even_when_capped(monkeypatch):
    events = _events(monkeypatch, billed_to_user=False)

    costs = [e["usd"] for e in events if e.get("event") == "cost"]
    assert costs and all(usd == 0.0 for usd in costs)
    cap = [e for e in events if e.get("code") == "COST_CAP"]
    assert cap, "the org-credit cap must still stop the run"
    assert "$" not in cap[0]["message"]
    assert "free tier" in cap[0]["message"].lower()


def test_a_byok_run_still_shows_its_cost(monkeypatch):
    events = _events(monkeypatch, billed_to_user=True)

    costs = [e["usd"] for e in events if e.get("event") == "cost"]
    assert costs and costs[-1] > 1.0
    cap = [e for e in events if e.get("code") == "COST_CAP"]
    assert cap and "$" in cap[0]["message"]
