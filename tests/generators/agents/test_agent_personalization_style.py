"""Tests for ``style_text_batch`` agent-reply restyling.

The OpenAI call is mocked, so these assert the style validation and that each
supported style selects a distinct, on-topic system prompt — not the model
output itself.
"""
from unittest.mock import patch

import pytest

from besser.generators.agents.agent_personalization import style_text_batch


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_friendly_style_is_supported(mock_chat):
    mock_chat.return_value = "1. happy to help"
    result = style_text_batch(["help"], "friendly", openai_api_key="k")

    assert result == ["happy to help"]
    system_prompt = mock_chat.call_args[0][0].lower()
    assert "friendly" in system_prompt or "warm" in system_prompt


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_technical_style_is_supported(mock_chat):
    mock_chat.return_value = "1. invoke the endpoint"
    result = style_text_batch(["call the api"], "technical", openai_api_key="k")

    assert result == ["invoke the endpoint"]
    system_prompt = mock_chat.call_args[0][0].lower()
    assert "technical" in system_prompt or "precise" in system_prompt


@pytest.mark.parametrize("style", ["formal", "informal", "friendly", "technical"])
@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_supported_styles_pass_validation(mock_chat, style):
    mock_chat.return_value = "1. rewritten"
    # Should not raise, and should call the model exactly once.
    result = style_text_batch(["original"], style, openai_api_key="k")

    assert result == ["rewritten"]
    assert mock_chat.call_count == 1


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_styles_select_distinct_prompts(mock_chat):
    mock_chat.return_value = "1. x"
    prompts = {}
    for style in ("formal", "informal", "friendly", "technical"):
        style_text_batch(["x"], style, openai_api_key="k")
        prompts[style] = mock_chat.call_args[0][0]

    # Every supported style must map to a different system prompt.
    assert len(set(prompts.values())) == 4


def test_unsupported_style_raises_value_error():
    with pytest.raises(ValueError, match="style must be one of"):
        style_text_batch(["hello"], "bogus", openai_api_key="k")


def test_case_insensitive_style():
    with patch(
        "besser.generators.agents.agent_personalization.call_openai_chat",
        return_value="1. ok",
    ) as mock_chat:
        result = style_text_batch(["hi"], "FRIENDLY", openai_api_key="k")

    assert result == ["ok"]
    assert mock_chat.call_count == 1
