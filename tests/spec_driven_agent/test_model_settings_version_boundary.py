"""A version marker matches only up to a boundary, so a dated id keeps its family's row.

``claude-fable-5-2`` is a prefix of the dated Fable 5 id ``claude-fable-5-20260101``; a plain
substring match gave that id the point-release row, which turns forced tool choice off.
"""

import pytest

from besser.spec_driven_agent.providers.model_settings import (
    forced_tool_choice_supported,
    thinking_can_be_disabled,
)

# (model id, forced tool_choice accepted)
CASES = [
    # Base releases, plain, dated and Bedrock-style: forced tool_choice works.
    ("claude-fable-5", True),
    ("claude-fable-5-20260101", True),
    ("claude-fable-5-10260101", True),
    ("us.anthropic.claude-fable-5-20260101-v1:0", True),
    ("claude-mythos-5", True),
    ("claude-mythos-5-20260101", True),
    ("claude-mythos-5-10260101", True),
    ("us.anthropic.claude-mythos-5-20260101-v1:0", True),
    # Point releases, in every shape: forced tool_choice is a 400.
    ("claude-fable-5-1", False),
    ("claude-fable-5-2", False),
    ("claude-fable-5-2-20260801", False),
    ("us.anthropic.claude-fable-5-2-v1:0", False),
    ("us.anthropic.claude-fable-5-1-20260601-v1:0", False),
    ("CLAUDE-FABLE-5-2", False),
    ("claude-mythos-5-1", False),
    ("claude-mythos-5-2", False),
    ("claude-mythos-5-2-20260801", False),
    ("us.anthropic.claude-mythos-5-2-v1:0", False),
    ("claude-opus-5-5", False),
    ("claude-sonnet-5-5-20260901", False),
]


@pytest.mark.parametrize(("model", "forced"), CASES)
def test_forced_tool_choice_follows_the_release_row(model, forced):
    assert forced_tool_choice_supported(model) is forced


@pytest.mark.parametrize(("model", "_forced"), CASES)
def test_every_fable_and_mythos_id_keeps_thinking_on(model, _forced):
    if "fable" in model.lower() or "mythos" in model.lower():
        assert thinking_can_be_disabled(model) is False
