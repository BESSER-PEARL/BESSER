"""A cost cap exactly reached is spent in Phase 2 too.

``_budget_stop_reason`` and ``_verification_call_allowed`` stop at
``cost >= cap``; the Phase 2 pre-call check compared ``cost > cap``, so a run
whose spend landed exactly on the cap sent one more billable turn.
"""

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

CAP_USD = 1.0


class _Client:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")
        self.calls = 0

    def chat(self, system, messages, tools):
        self.calls += 1
        return {"stop_reason": "end_turn", "content": []}


def test_a_cap_exactly_reached_sends_no_phase2_turn(tmp_path):
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    client = _Client()
    orch = LLMOrchestrator(llm_client=client, domain_model=DomainModel(name="M", types={user}),
                           output_dir=str(tmp_path), max_cost_usd=CAP_USD,
                           enable_tracing=False, enable_checkpointing=False)
    client.usage._extra_cost_usd = CAP_USD
    assert client.usage.estimated_cost == CAP_USD

    orch.run("Build an app")

    assert client.calls == 0
    assert orch._phase2_stop_reason == "cost_cap"
