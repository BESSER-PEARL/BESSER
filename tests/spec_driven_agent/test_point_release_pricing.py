"""Fable 5.2 and Mythos 5.2 bill at their own published rate, not at the gpt-4o fallback.

The vendored price table carries the 5.1 ids only, so the 5.2 ids logged "No price for
model claude-fable-5-2: using the gpt-4o fallback rate" and billed $2.50/$10 per 1M tokens.
5.2 is cheaper than 5.1 ($8/$40 against $10/$50, cache reads $0.20).
"""

import logging

import pytest

from besser.llm.pricing import _warn_unpriced, get_pricing

RATE_5_2 = {"input": 8.0, "output": 40.0, "cache_write": 10.0, "cache_read": 0.2}


@pytest.mark.parametrize("model", [
    "claude-fable-5-2",
    "claude-mythos-5-2",
    "us.anthropic.claude-fable-5-2-v1:0",
    "claude-mythos-5-2-20260801",
])
def test_point_release_bills_at_the_5_2_rate(model, caplog):
    _warn_unpriced.cache_clear()
    with caplog.at_level(logging.WARNING, logger="besser.llm.pricing"):
        price = get_pricing(model, billed=True)
    assert {key: price[key] for key in RATE_5_2} == RATE_5_2
    assert "No price for model" not in caplog.text


def test_the_5_1_release_keeps_its_own_rate():
    assert get_pricing("claude-fable-5-1", billed=True)["input"] == 10.0


def test_a_dated_base_release_is_not_billed_as_the_point_release():
    assert get_pricing("claude-fable-5-20260101", billed=True) != get_pricing("claude-fable-5-2", billed=True)


@pytest.mark.parametrize("model", [
    "us.anthropic.claude-sonnet-5-5-v1:0",
    "claude-sonnet-5-5-20260901",
])
def test_a_prefixed_or_dated_sonnet_5_5_id_bills_at_the_sonnet_5_5_rate(model):
    price = get_pricing(model, billed=True)
    assert (price["input"], price["output"], price["cache_read"]) == (2.0, 10.0, 0.1)
