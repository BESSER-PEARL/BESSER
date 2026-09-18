"""Deciding "is this model free" wrong disables the cost cap.

`_get_pricing` asks `_is_free_local_model` first, and a True answer returns
`_ZERO_PRICING`. Every call then costs $0, so `max_cost_usd` can never trip and
a run continues until the turn or time cap -- while the provider bills normally
and the run card shows $0.00 throughout.

The old rule matched FAMILY NAMES (`qwen`, `deepseek`, `llama`, ...), which was
correct while open-weight implied self-hosted. It stopped being correct when
the gateway began reselling those same families: `Qwen/Qwen3.8-Max` and
`deepseek/deepseek-v4-pro` are billed cloud models whose ids contain those
markers, so both priced at $0.

The rule now keys on how a model is IDENTIFIED rather than its family.
"""
import pytest

from besser.generators.llm.llm_client import (
    _MODEL_PRICING,
    _ZERO_PRICING,
    _get_pricing,
    _is_free_local_model,
)


@pytest.mark.parametrize("model_id, reason", [
    ("qwen3-coder:30b", "self-hosted Ollama name:tag, no vendor namespace"),
    ("llama3", "tagless self-hosted id, still covered by the family list"),
    ("meituan/LongCat-2.0:free", "explicit :free suffix"),
    ("poolside/laguna-s-2.1-free", "explicit -free suffix"),
    ("inclusionai/ling-3.0-flash-sante:free", "explicit :free suffix"),
])
def test_genuinely_free_models_are_free(model_id, reason):
    assert _is_free_local_model(model_id.lower()) is True, reason
    assert _get_pricing(model_id) == _ZERO_PRICING


@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen3.8-Max",
    "Qwen/Qwen3.8-Flash",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.3",
    "MiniMaxAI/MiniMax-M3",
    "google/gemini-3.8-flash",
])
def test_gateway_models_are_billed_not_free(model_id):
    """The regression this file exists for.

    These are vendor-namespaced gateway models. Somebody is billed for them, so
    pricing them at $0 would switch the cost cap off.
    """
    assert _is_free_local_model(model_id.lower()) is False
    pricing = _get_pricing(model_id)
    assert pricing != _ZERO_PRICING, (
        f"{model_id} priced at $0 -- max_cost_usd cannot trip for it"
    )
    assert pricing["input"] > 0 and pricing["output"] > 0


@pytest.mark.parametrize("model_id, expected_key", [
    ("gpt-5.6-luna", "gpt-5.6-luna"),
    ("gpt-5.6-sol", "gpt-5.6-sol"),
    ("gpt-4o", "gpt-4o"),
])
def test_known_paid_models_keep_their_real_rates(model_id, expected_key):
    assert _get_pricing(model_id) == _MODEL_PRICING[expected_key]


def test_the_self_hosted_fallback_stays_free():
    """The one model this heuristic exists for must not regress into billing."""
    from besser.generators.llm.llm_client import free_fallback_model
    import os

    os.environ.setdefault("BESSER_FREE_LLM_FALLBACK_MODEL", "qwen3-coder:30b")
    fallback = free_fallback_model()
    if fallback:
        assert _get_pricing(fallback) == _ZERO_PRICING
