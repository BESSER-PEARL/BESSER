"""The planning calls' fallbacks ask the run's stop and spend gate first.

Review (2026-11, PR 788): ``extract_requirements`` and the gap analyser's
``_call_planner`` sent their primary-model fallback (and the planner its repair
and plain retries) without the ``can_call`` gate the judge uses, so a planning
model that failed after the cost cap was reached, or after a cancel, still
bought a full-price call.
"""

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.gap_analyzer import analyze_gaps_via_llm
from besser.spec_driven_agent.providers.llm_client import UsageTracker

CALL_COST_USD = 0.30


def _model():
    room = Class(name="Room")
    room.attributes = {Property(name="number", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="Hotel", types={room})


class _CostlyClient:
    """Every call costs $0.30; the planning model raises, the primary answers with nothing usable."""

    model = "mock-model"
    planning_model = "cheap-model"

    def __init__(self):
        self._client = object()
        self.usage = UsageTracker("mock-model")
        self.calls = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append(model_override)
        self.usage._extra_cost_usd += CALL_COST_USD
        if model_override:
            raise RuntimeError("planning model not reachable")
        return {"stop_reason": "end_turn", "content": []}


def test_requirement_extraction_sends_no_fallback_past_the_cap(tmp_path):
    client = _CostlyClient()
    orch = LLMOrchestrator(llm_client=client, domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, max_cost_usd=1.0)
    client.usage._extra_cost_usd = 0.95

    orch._extract_requirements("Room numbers are unique.")

    assert client.calls == ["cheap-model"], "the primary fallback was sent after the cap"
    assert orch._requirements is None


def test_requirement_extraction_sends_no_fallback_after_a_cancel(tmp_path):
    client = _CostlyClient()
    orch = LLMOrchestrator(llm_client=client, domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._should_continue = lambda: not client.calls  # cancelled during the first call

    orch._extract_requirements("Room numbers are unique.")

    assert client.calls == ["cheap-model"]


@pytest.mark.parametrize("allowed_calls, expected", [
    (1, ["cheap-model"]),
    # Planning model, primary fallback, repair retry, plain fallback.
    (99, ["cheap-model", None, "cheap-model", None]),
])
def test_the_gap_planner_asks_the_gate_before_every_call(allowed_calls, expected):
    client = _CostlyClient()

    tasks = analyze_gaps_via_llm(
        instructions="Hotel bookings.", generator_used="generate_fastapi_backend", domain_model=None,
        inventory="backend/main_api.py 1200", llm_client=client,
        can_call=lambda: len(client.calls) < allowed_calls,
    )

    assert tasks is None
    assert client.calls == expected


def test_the_gap_planner_sends_nothing_when_the_gate_is_closed():
    client = _CostlyClient()

    analyze_gaps_via_llm(
        instructions="Hotel bookings.", generator_used="generate_fastapi_backend", domain_model=None,
        inventory="backend/main_api.py 1200", llm_client=client, can_call=lambda: False,
    )

    assert client.calls == []
