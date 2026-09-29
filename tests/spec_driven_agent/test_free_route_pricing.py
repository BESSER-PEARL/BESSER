"""Pricing on the keyless tier after the 2026-09-29 lineup change.

The free route now serves ``moonshotai/Kimi-K3`` (default),
``Qwen/Qwen3.8-Max-0902`` and ``meta/muse-spark-1.3``, which draw on PAID org
credits through the server's Command Code token. So those runs stay priced
at list rates and capped at the per-run cap; only the user is not billed
(``billed_to_user`` False, and the runner shows them $0). Genuinely free ids
(``-free`` / ``:free``) and the self-hosted V100 fallback stay $0.

Live evidence: free run d3a33f95 on ``moonshotai/Kimi-K2.7-Code`` recorded
157,953 in / 30,810 out / 1,121,920 cache-read tokens. At the vendored
Kimi-K3 row ($3 / $15, no cache discount) that is ~$4.30 of org credits.
"""

from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.providers.llm_client import (
    _MODEL_PRICING,
    _ZERO_PRICING,
    create_llm_client,
)

FREE_URL = "https://api.commandcode.ai/provider/v1"
LOCAL_URL = "https://ollama.example.invalid/v1"
KIMI_K3 = "moonshotai/Kimi-K3"
LAGUNA = "poolside/laguna-s-2.1-free"
UNPRICED_ALTS = ["Qwen/Qwen3.8-Max-0902", "meta/muse-spark-1.3"]
LOCAL = "qwen3-coder:30b"
CAP_USD = 5.0

REVIEWED_RUN = SimpleNamespace(
    input_tokens=157_953, output_tokens=30_810,
    cache_creation_input_tokens=0, cache_read_input_tokens=1_121_920,
)


@pytest.fixture
def lineup(monkeypatch):
    """The experimental server's free-tier env, tokens replaced by dummies."""
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", FREE_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", KIMI_K3)
    monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", "dummy-free-token")
    monkeypatch.setenv("BESSER_FREE_LLM_ALT_MODELS", ",".join([LAGUNA, *UNPRICED_ALTS]))
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_BASE_URL", LOCAL_URL)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_MODEL", LOCAL)
    monkeypatch.setenv("BESSER_FREE_LLM_FALLBACK_TOKEN", "dummy-local-token")
    monkeypatch.delenv("BESSER_LLM_PLANNING_MODEL", raising=False)


def _run_like_the_reviewed_one(client, repeats=2):
    for _ in range(repeats):
        client.usage.record(REVIEWED_RUN)
    return client.usage.estimated_cost


def test_a_free_tier_kimi_k3_run_is_priced_and_cost_capped(lineup):
    client = create_llm_client(provider="free")
    assert client.model == KIMI_K3
    assert client.usage.pricing["input"] == pytest.approx(3.0)
    assert client.usage.pricing["output"] == pytest.approx(15.0)

    assert _run_like_the_reviewed_one(client) > CAP_USD


def test_a_free_tier_run_is_not_billed_to_the_user(lineup):
    client = create_llm_client(provider="free")
    assert client.usage.billed_to_user is False
    assert client.usage.summary()["billed_to_user"] is False


@pytest.mark.parametrize("alt", UNPRICED_ALTS)
def test_unpriced_paid_alts_get_the_conservative_fallback_rate(lineup, alt):
    client = create_llm_client(provider="free", model=alt)
    assert client.model == alt
    assert client.usage.pricing == _MODEL_PRICING["gpt-4o"]


def test_the_free_suffixed_alt_is_zero(lineup):
    client = create_llm_client(provider="free", model=LAGUNA)
    assert client.usage.pricing == _ZERO_PRICING
    assert _run_like_the_reviewed_one(client) == 0.0


def test_the_self_hosted_fallback_is_zero_when_chosen(lineup):
    client = create_llm_client(provider="free", model=LOCAL)
    assert client.model == LOCAL
    assert _run_like_the_reviewed_one(client) == 0.0


def test_spend_stops_growing_once_the_chain_reaches_the_v100(lineup):
    client = create_llm_client(provider="free")
    client.usage.record(REVIEWED_RUN)
    spent = client.usage.estimated_cost
    assert spent > 0
    for _ in range(3):  # laguna, qwen3.8, muse ... then the V100
        client._activate_fallback(RuntimeError("429 used all requests for today"))
    client._activate_fallback(RuntimeError("503"))
    assert client.model == LOCAL
    before = client.usage.estimated_cost
    client.usage.record(REVIEWED_RUN)
    assert client.usage.estimated_cost == pytest.approx(before)


def test_a_free_route_planning_call_is_not_sent_to_gpt_4o_mini(lineup):
    """The keyless endpoint does not serve gpt-4o-mini (absent from its
    /provider/v1/models catalog), so the override is a failing call on a
    server-owned token that can also trip the run's outage fallback."""
    client = create_llm_client(provider="free")
    assert client.planning_model is None


def test_a_byok_nebius_kimi_k3_run_is_billed_to_the_user(lineup):
    client = create_llm_client(provider="nebius", api_key="dummy", model=KIMI_K3)

    cost = _run_like_the_reviewed_one(client, repeats=1)

    assert client.usage.billed_to_user is True
    assert client.usage.pricing["input"] == pytest.approx(3.0)
    assert client.usage.pricing["output"] == pytest.approx(15.0)
    assert cost == pytest.approx(4.30, abs=0.01)
