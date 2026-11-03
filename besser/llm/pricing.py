"""Per-model token prices, for cost tracking and the per-run cost cap.

:func:`get_pricing` answers USD per 1M tokens (``input``, ``output``,
``cache_write``, ``cache_read``) from, in order: the rate a gateway bills
(model catalog ``route_prices``), $0 for self-hosted and free models, a rate
the deployment's catalog sets, the vendored public list prices
(``data/model_prices.json``) and the hand-typed :data:`MODEL_PRICING`.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from besser.llm.catalog import load_model_catalog

logger = logging.getLogger(__name__)

# Pricing per million tokens (refreshed early 2026).
# Sources: OpenAI public pricing page + OpenRouter / pricepertoken.com
# corroboration for gpt-5.5. Anthropic rates are stable from 2025.
# ``cache_read`` is OpenAI's prompt-caching discount (~10% of input);
# ``cache_write`` is 0 for OpenAI since they don't charge a write
# premium like Anthropic does.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "haiku":  {"input": 1.0,  "output": 5.0,  "cache_write": 1.25,  "cache_read": 0.1},
    "sonnet": {"input": 3.0,  "output": 15.0, "cache_write": 3.75,  "cache_read": 0.3},
    "opus":   {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.5},
    # Exact Anthropic ids the vendored table does not carry yet; without a row
    # they land on the coarse tier above (Opus 5.5 at $15/$75, ~4x over).
    # Cache write is the standard 1.25x of input.
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.2},
    # OpenAI — early-2026 public rates
    "gpt-4o-mini": {"input": 0.15, "output": 0.6,  "cache_write": 0, "cache_read": 0.075},
    "gpt-4o":      {"input": 2.5,  "output": 10.0, "cache_write": 0, "cache_read": 1.25},
    "gpt-5.6-sol":   {"input": 5.0, "output": 30.0, "cache_write": 0, "cache_read": 0.5},
    # Official model pages (standard requests <=272K input).
    "gpt-5.6-terra": {"input": 2.0, "output": 12.0, "cache_write": 0, "cache_read": 0.2},
    "gpt-5.6-luna":  {"input": 0.2, "output": 1.2,  "cache_write": 0, "cache_read": 0.02},
    # Official pricing page (2026-09-28). Not yet in the vendored table.
    "gpt-6-sol":   {"input": 2.0, "output": 10.0, "cache_write": 0, "cache_read": 0.2},
    "gpt-6-luna":  {"input": 0.1, "output": 0.5,  "cache_write": 0, "cache_read": 0.01},
    "gpt-5.5":     {"input": 5.0,  "output": 30.0, "cache_write": 0, "cache_read": 0.5},
    "gpt-5":       {"input": 1.25, "output": 10.0, "cache_write": 0, "cache_read": 0.125},
    "o3-mini":     {"input": 1.1,  "output": 4.4,  "cache_write": 0, "cache_read": 0},
    "o3":          {"input": 10.0, "output": 40.0, "cache_write": 0, "cache_read": 0},
    # Mistral — public rates as of mid-2026. ``mistral-large-latest``
    # currently bills at $2 / $6 per 1M (input / output). Mistral has no
    # separate cache-write premium; prompt-caching reads, when used, are
    # billed at the input rate (no published discount), so cache_read is
    # set equal to input rather than a fictional cheaper tier.
    # VERIFY: the ``-latest`` alias may route to Mistral Large 3
    # ($0.50 / $1.50). Re-check https://mistral.ai/pricing/ if exactness
    # matters; $2 / $6 is the conservative (higher) of the two.
    "mistral-large": {"input": 2.0, "output": 6.0, "cache_write": 0, "cache_read": 2.0},
    # Nebius Token Factory — open-weight models served on Nebius hardware.
    # Read off the endpoint properties in the Nebius console:
    # $0.10 / 1M in, $0.30 / 1M out (~70 tok/s). Nebius publishes these rates
    # only in the console, not in fetchable docs. Nebius publishes no
    # prompt-cache discount, so cache_read == input rather than a fictional
    # cheaper tier (same reasoning as Mistral above).
    "qwen3-30b-a3b": {"input": 0.1, "output": 0.3, "cache_write": 0, "cache_read": 0.1},
}

# Self-hosted / open-weight models served through an OpenAI-compatible
# endpoint (Ollama, vLLM, LM Studio, ...) cost the platform nothing, so they
# must be priced at $0 — otherwise the ``gpt-4o`` fallback below invents a
# ~$2.50/$10-per-1M phantom cost that trips ``max_cost_usd`` and aborts the
# run mid-generation with ``stop_reason="cost_cap"``. Matched case-insensitively
# as substrings. Kept conservative (clearly open families only) so a genuinely
# unknown *paid* cloud model still gets the protective fallback rate.
ZERO_PRICING: dict[str, float] = {
    "input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0,
}
_FREE_LOCAL_MODEL_MARKERS = (
    "qwen", "llama", "codellama", "deepseek", "mixtral",
    "gemma", "starcoder", "granite", "phi-", "phi3", "phi4",
    "devstral", "mistral-small", "codestral",
    # Sponsored-tier zero-credit models (the ':free' suffix is the
    # aggregator's own marker); zero pricing + no ghost planning model.
    ":free", "longcat", "laguna",
)


def is_free_local_model(model_lower: str) -> bool:
    """True for models that cost the platform $0 — self-hosted, or free tiers.

    Getting this wrong is expensive in both directions: a false positive prices
    a paid model at $0 and SILENTLY DISABLES the cost cap, while a false
    negative bills a self-hosted run that is actually free.

    The decision therefore keys on how the model is IDENTIFIED, not on its
    family: the gateway resells open-weight families, so ``Qwen/Qwen3.8-Max``
    and ``deepseek/deepseek-v4-pro`` are billed cloud models that a family
    allow-list priced at $0.

    1. an explicit free marker (``:free`` / ``-free``) -- always free;
    2. an Ollama ``name:tag`` id with no vendor namespace -- self-hosted;
    3. a vendor-namespaced id (``vendor/model``) -- a gateway model, billed;
    4. otherwise fall back to the family list, which still covers tagless
       self-hosted ids.
    """
    if model_lower.endswith(":free") or model_lower.endswith("-free"):
        return True
    has_namespace = "/" in model_lower
    if ":" in model_lower and not has_namespace:
        return True
    if has_namespace:
        # Served by the gateway, so somebody is being billed for it. Fall
        # through to the paid pricing path rather than assuming $0.
        return False
    if model_lower.endswith("-latest"):
        # A vendor API alias, not a checkout. Mistral's paid ids are tagless
        # and un-namespaced (``mistral-small-latest``, ``codestral-latest``,
        # ...) and would otherwise match the family list below and price at
        # $0, silently disabling the cost cap. Self-hosted Ollama says
        # ``mistral-small:latest`` with a colon, which rule 2 already caught
        # above, so nothing free is lost by treating the hyphen as billed.
        return False
    return any(marker in model_lower for marker in _FREE_LOCAL_MODEL_MARKERS)


_VENDORED_PRICES_PATH = Path(__file__).with_name("data") / "model_prices.json"

# Order is the tie-break, not a preference: one bare id can appear under
# several vendors at wildly different rates (``mistral-large-latest`` is
# $0.50/$1.50 direct and $8/$24 on Azure). Only vendors we actually call are
# listed, so a resold id can never be priced at some third party's rate.
_PRICE_LOOKUP_PREFIXES = ("", "anthropic/", "openai/", "mistral/", "nebius/")

def _route_pricing(model_id: str, route: "str | None") -> "dict[str, float] | None":
    """The rate a gateway bills for ``model_id``, from the model catalog.

    Gateways bill their own rates, which the vendored table does not know:
    pricing Command Code's Kimi-K3 from the Nebius row billed cached input at
    the full input rate, so long runs hit the per-run cap about 3x too early.
    """
    return load_model_catalog().route_price(model_id, route)


@lru_cache(maxsize=1)
def _vendored_price_index() -> dict[str, dict[str, float]]:
    """``vendored_prices`` re-keyed lower-case for case-insensitive lookup.

    Model ids reach us in whatever case the caller typed, and the published
    keys are mixed case themselves (``nebius/Qwen/Qwen3-30B-A3B-Instruct-2507``).
    An exact-case ``dict.get`` would silently miss and fall through to the
    coarse tier table.
    """
    return {key.lower(): value for key, value in vendored_prices().items()}


@lru_cache(maxsize=1)
def vendored_prices() -> dict[str, dict[str, float]]:
    """Published per-model rates, keyed by exact model id.

    Vendored from litellm's ``model_prices_and_context_window.json`` (MIT),
    reduced to the providers we call and the four fields we read. The data
    file rather than the package: an ``import litellm`` would break the
    offline on-prem install. Refresh with::

        curl -s https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json

    then keep rows whose ``litellm_provider`` is anthropic/openai/mistral/
    nebius/bedrock and the ``*_cost_per_token`` fields.

    The modeling-agent prices its calls from a byte-identical copy
    (``src/llm/tracking/data/model_prices.json`` there): copy the refreshed
    file over it in the same change.

    Returns ``{}`` if the file is missing (a wheel built without it), which
    falls the caller through to ``MODEL_PRICING`` rather than to $0 — an
    absent price must never read as free, or the cost cap stops existing.
    """
    try:
        with open(_VENDORED_PRICES_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        logger.warning("Vendored model price table unreadable at %s; "
                       "falling back to the built-in tier table",
                       _VENDORED_PRICES_PATH)
        return {}


def _published_pricing(model_id: str) -> dict[str, float] | None:
    """Exact-id rate for ``model_id``, or ``None`` when it is not published."""
    table = _vendored_price_index()
    if not table:
        return None
    # Per-TOKEN costs scaled to per-million land on values like
    # 0.19999999999999998. Rounding keeps the published figure exact, so a
    # rate reads as 0.2 rather than as float noise in every cost report.
    def per_million(value: float) -> float:
        return round(value * 1_000_000.0, 6)

    for prefix in _PRICE_LOOKUP_PREFIXES:
        row = table.get(f"{prefix}{model_id}".lower())
        if row and row.get("input_cost_per_token") is not None:
            read = row.get("cache_read_input_token_cost")
            return {
                "input": per_million(row["input_cost_per_token"]),
                "output": per_million(row["output_cost_per_token"]),
                "cache_write": per_million(
                    row.get("cache_creation_input_token_cost", 0.0)),
                # No published cache discount means reads bill at the input
                # rate, not free — same reasoning as the Mistral/Nebius rows.
                "cache_read": per_million(
                    read if read is not None else row["input_cost_per_token"]),
            }
    return None


def get_pricing(model_id: str, billed: bool = False, route: "str | None" = None) -> dict[str, float]:
    """Get the per-1M-token rates for a model ID.

    Published per-model rates come first, from the vendored table. The
    hand-typed ``MODEL_PRICING`` below it is a fallback for ids nobody
    publishes, and its Anthropic rows are coarse *tiers*: they cannot tell
    ``claude-sonnet-4-6`` ($3/$15) from ``claude-sonnet-5`` ($2/$10), and the
    ``opus`` row still carries Claude-3-era $15/$75 — 3x over every current
    Opus, which trips ``max_cost_usd`` at a third of the intended spend.
    Prices are external facts that drift silently, so prefer the table that
    is refreshed over the one that is remembered.

    Order matters in the OpenAI loop — longer / more-specific keys
    must come first so e.g. ``gpt-5.5`` doesn't get mis-matched to
    ``gpt-5`` (which is a substring) and silently billed at the
    cheaper tier. Falls back to ``gpt-4o`` so an unknown *paid* model
    gets a reasonable middle-tier rate; self-hosted/open models are
    detected first and priced at $0.

    ``billed`` is set when the route itself is a paid vendor API (Anthropic,
    official OpenAI, Mistral, Nebius): the id-shape free heuristic is skipped
    there, since ``codestral-2508`` on Mistral's API is not a local checkout.
    """
    routed = _route_pricing(model_id, route)
    if routed is not None:
        return routed
    model_lower = model_id.lower()
    # Self-hosted / free local models — never bill, never trip the cost cap.
    if not billed and is_free_local_model(model_lower):
        return ZERO_PRICING
    # A rate the deployment's catalog sets beats the vendored list price.
    listed = load_model_catalog().listed_price(model_id)
    if listed is not None:
        return listed
    published = _published_pricing(model_id)
    if published is not None:
        return published
    # Exact Anthropic ids first: each is also inside a coarse tier below.
    for key in ("claude-opus-5-5",):
        if key in model_lower:
            return MODEL_PRICING[key]
    # Anthropic tiers — unambiguous identifiers so order doesn't matter.
    for tier in ("haiku", "sonnet", "opus"):
        if tier in model_lower:
            return MODEL_PRICING[tier]
    # Mistral tiers — match before the OpenAI fallback so a Mistral
    # model never silently bills at the ``gpt-4o`` default rate.
    for key in ("mistral-large",):
        if key in model_lower:
            return MODEL_PRICING[key]
    # Nebius Token Factory tiers. These ids are vendor-namespaced
    # (``Qwen/Qwen3-30B-A3B-Instruct-2507``), so ``is_free_local_model``
    # correctly classifies them as billed rather than $0 — but without a row
    # here they would then land on the ``gpt-4o`` fallback and be billed ~8x
    # over, tripping ``max_cost_usd`` long before the user's real budget.
    for key in ("qwen3-30b-a3b",):
        if key in model_lower:
            return MODEL_PRICING[key]
    # OpenAI: variant-specific GPT-5.6 / GPT-5.5 keys must precede the
    # generic ``gpt-5`` substring or their spend would be under-counted.
    for key in (
        "gpt-4o-mini", "gpt-4o",
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
        "gpt-6-sol", "gpt-6-luna",
        "gpt-5.5", "gpt-5", "o3-mini", "o3",
    ):
        if key in model_lower:
            return MODEL_PRICING[key]
    return MODEL_PRICING["gpt-4o"]  # default
