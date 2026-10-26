"""PIA (LIST's AI gateway) bills its own rates, not the vendors' list prices.

Before these rows, PIA's recommended models priced from the wrong source:
``claude-sonnet-5-5`` landed on the coarse Sonnet tier ($3 / $15),
``gpt-6.1-sol`` on the ``gpt-4o`` fallback ($2.50 / $10), and
``claude-haiku-4-5-20251001`` on Anthropic's direct rate ($1 / $5).
"""

from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.providers.llm_client import create_llm_client

PIA_URL = "https://gateway.pia.private.list.lu/v1"
ONE_MILLION_EACH = SimpleNamespace(
    input_tokens=1_000_000, output_tokens=1_000_000,
    cache_creation_input_tokens=0, cache_read_input_tokens=0,
)


# PIA's published rates per 1M tokens (input / output), 2026-10-26.
@pytest.mark.parametrize("model, rates", [
    ("claude-haiku-4-5-20251001", (1.1, 5.5)),
    ("gpt-6-luna", (0.1, 0.5)),
    ("claude-sonnet-5-5", (2.2, 11.0)),
    ("gpt-6.1-sol", (2.0, 10.0)),
    ("claude-opus-5-5", (4.4, 22.0)),
    ("gpt-6-astra", (10.0, 50.0)),
])
def test_pia_models_use_the_gateway_rates(model, rates):
    client = create_llm_client(provider="openai", api_key="sk-test", model=model, base_url=PIA_URL)
    client.usage.record(ONE_MILLION_EACH)
    assert client.usage.estimated_cost == pytest.approx(sum(rates))


def test_the_same_id_off_pia_keeps_its_own_price():
    client = create_llm_client(provider="anthropic", api_key="sk-ant-test", model="claude-haiku-4-5-20251001")
    client.usage.record(ONE_MILLION_EACH)
    assert client.usage.estimated_cost == pytest.approx(1.0 + 5.0)
