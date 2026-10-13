"""Every selectable model is billed at its published rate, so the cap works.

Without a row, gpt-6-sol / gpt-6-luna fell to the gpt-4o fallback ($2.50/$10,
25x over for luna) and claude-opus-5-5 to the Claude-3-era opus tier ($15/$75),
so ``max_cost_usd`` fired at a fraction of the real budget.
"""

import pytest

from besser.spec_driven_agent.providers.llm_client import UsageTracker, _get_pricing

# (model id, input $/1M, output $/1M) from the vendors' pricing pages.
SELECTABLE_MODEL_PRICES = [
    ("gpt-6-astra", 10.0, 50.0),
    ("gpt-6-sol", 2.0, 10.0),
    ("gpt-6-luna", 0.1, 0.5),
    ("gpt-5.6-sol", 4.0, 20.0),
    ("gpt-5.6-terra", 2.0, 12.0),
    ("gpt-5.6-luna", 0.2, 1.2),
    ("gpt-5.5", 5.0, 30.0),
    ("gpt-5.4", 2.5, 15.0),
    ("gpt-5.4-mini", 0.75, 4.5),
    ("gpt-4o", 2.5, 10.0),
    ("claude-fable-5-1", 10.0, 50.0),
    ("claude-fable-5", 10.0, 50.0),
    ("claude-opus-5-5", 4.0, 20.0),
    ("claude-opus-5", 5.0, 25.0),
    ("claude-opus-4-8", 5.0, 25.0),
    ("claude-opus-4-7", 5.0, 25.0),
    ("claude-opus-4-6", 5.0, 25.0),
    ("claude-sonnet-5", 2.0, 10.0),
    ("claude-sonnet-4-6", 3.0, 15.0),
    ("claude-haiku-4-5", 1.0, 5.0),
    ("claude-haiku-4-5-20251001", 1.0, 5.0),
]


@pytest.mark.parametrize(("model", "input_rate", "output_rate"), SELECTABLE_MODEL_PRICES)
def test_every_selectable_model_is_billed_at_its_published_rate(model, input_rate, output_rate):
    pricing = _get_pricing(model)
    assert (pricing["input"], pricing["output"]) == (input_rate, output_rate)


@pytest.mark.parametrize(("model", "cache_read"), [
    ("gpt-6-sol", 0.2), ("gpt-6-luna", 0.01), ("claude-opus-5-5", 0.2),
    ("claude-fable-5-1", 0.25),
])
def test_cache_reads_are_billed_at_the_discounted_rate(model, cache_read):
    assert _get_pricing(model)["cache_read"] == cache_read


def test_a_bedrock_style_opus_5_5_id_gets_the_same_rate():
    assert _get_pricing("us.anthropic.claude-opus-5-5")["input"] == 4.0


def test_the_cost_cap_sees_the_real_spend_on_gpt_6_luna():
    tracker = UsageTracker("gpt-6-luna")
    tracker.input_tokens = 1_000_000
    tracker.output_tokens = 1_000_000
    assert tracker.estimated_cost == pytest.approx(0.6)
