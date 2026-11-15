"""Fable 5.2 and Mythos 5.2 are priced like their 5.1 releases, not at the gpt-4o fallback.

The vendored price table carries the 5.1 ids only, so the 5.2 ids logged "No price for
model claude-fable-5-2: using the gpt-4o fallback rate" and billed $2.50/$10 per 1M tokens
against a cost cap meant for $10/$50.
"""

import logging

import pytest

from besser.llm.pricing import _warn_unpriced, get_pricing


@pytest.mark.parametrize(("model", "release_5_1"), [
    ("claude-fable-5-2", "claude-fable-5-1"),
    ("claude-mythos-5-2", "claude-mythos-5-1"),
    ("us.anthropic.claude-fable-5-2-v1:0", "claude-fable-5-1"),
    ("claude-mythos-5-2-20260801", "claude-mythos-5-1"),
])
def test_point_release_bills_at_the_5_1_rate(model, release_5_1, caplog):
    _warn_unpriced.cache_clear()
    with caplog.at_level(logging.WARNING, logger="besser.llm.pricing"):
        assert get_pricing(model, billed=True) == get_pricing(release_5_1, billed=True)
    assert "No price for model" not in caplog.text


def test_a_dated_base_release_is_not_billed_as_the_point_release():
    assert get_pricing("claude-fable-5-20260101", billed=True) != get_pricing("claude-fable-5-2", billed=True)
