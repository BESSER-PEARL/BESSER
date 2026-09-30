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

from besser.spec_driven_agent.providers.llm_client import (
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


@pytest.mark.parametrize("model_id, input_rate, output_rate", [
    # Anthropic. The coarse ``_MODEL_PRICING`` tiers cannot express these:
    # one ``sonnet`` row cannot be both 4-6 and 5, and its ``opus`` row still
    # carries Claude-3-era $15/$75 -- 3x over, which trips ``max_cost_usd`` at
    # a third of the intended spend and aborts a healthy run.
    ("claude-sonnet-5", 2.0, 10.0),
    ("claude-sonnet-4-6", 3.0, 15.0),
    ("claude-opus-5", 5.0, 25.0),
    ("claude-opus-4-8", 5.0, 25.0),
    ("claude-haiku-4-5", 1.0, 5.0),
    # OpenAI.
    ("gpt-5.6-terra", 2.0, 12.0),
    ("gpt-5.6-luna", 0.2, 1.2),
    ("gpt-4o", 2.5, 10.0),
    # Resolved through a vendor prefix: the bare id is not published, and the
    # id alone is ambiguous -- ``azure/mistral-large-latest`` is $8/$24
    # against $0.50/$1.50 direct, so the vendor order decides, not chance.
    ("mistral-large-latest", 0.5, 1.5),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", 0.1, 0.3),
])
def test_known_paid_models_bill_at_published_rates(model_id, input_rate, output_rate):
    """Pinned as literals, not as ``_MODEL_PRICING[key]``.

    Asserting equality with the fallback table only proved the two agreed --
    it passed happily while that table billed every Opus 3x over. These are
    the published per-1M rates; a mismatch means either the vendored table
    went stale or a lookup regressed.
    """
    pricing = _get_pricing(model_id)
    assert pricing["input"] == pytest.approx(input_rate)
    assert pricing["output"] == pytest.approx(output_rate)


def test_an_unpublished_paid_model_still_falls_back_rather_than_going_free():
    """A missing price must never read as $0 -- that is the cost cap gone."""
    pricing = _get_pricing("some-unlisted-vendor-model-2099")
    assert pricing != _ZERO_PRICING
    assert pricing == _MODEL_PRICING["gpt-4o"]


def test_the_vendored_table_ships_and_is_readable():
    """Packaged via ``setup.cfg`` package_data; ``packages=find:`` does not
    see ``data/`` as a package, so a key naming it would be inert and every
    install would silently fall back to the coarse tiers."""
    from besser.spec_driven_agent.providers.llm_client import _vendored_prices
    table = _vendored_prices()
    assert table, "vendored price table is missing or empty"
    assert table["claude-sonnet-5"]["input_cost_per_token"] == pytest.approx(2e-06)


def test_the_self_hosted_fallback_stays_free():
    """The one model this heuristic exists for must not regress into billing."""
    from besser.spec_driven_agent.providers.llm_client import free_fallback_model
    import os

    os.environ.setdefault("BESSER_FREE_LLM_FALLBACK_MODEL", "qwen3-coder:30b")
    fallback = free_fallback_model()
    if fallback:
        assert _get_pricing(fallback) == _ZERO_PRICING
