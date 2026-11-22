"""Tests for ``style_text_batch`` agent-reply restyling.

The OpenAI call is mocked, so these assert the style validation and that each
supported style selects a distinct, on-topic system prompt — not the model
output itself.
"""
from unittest.mock import patch

import pytest

from besser.generators.agents.agent_personalization import (
    explanation_text_batch, flatten_agent_config_structure, replace_reply_batch, style_text_batch,
)


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




@pytest.mark.parametrize("config", [
    {"agentStyle": "formal", "explanationStyle": "narrative"},
    {"presentation": {"agentStyle": "formal", "explanationStyle": "narrative"}},
    {"presentation": {"conversationalTone": "formal", "explanationStyle": "narrative"}},
])
def test_normalize_presentation_names_without_mutating_input(config):
    import copy
    original = copy.deepcopy(config)
    assert flatten_agent_config_structure(config) == {
        "conversationalTone": "formal", "explanationStyle": "narrative",
    }
    assert config == original


def test_new_tone_takes_precedence_over_legacy_name():
    assert flatten_agent_config_structure({"agentStyle": "formal", "conversationalTone": "informal"}) == {
        "conversationalTone": "informal",
    }


@pytest.mark.parametrize("style", ["direct", "step-by-step", "narrative"])
@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_explanation_styles_preserve_facts_and_presentation_constraints(mock_chat, style):
    mock_chat.return_value = "1. Rewritten[[NL]]explanation\n2. Open 10:00 to 18:00."
    result = explanation_text_batch(
        ["An explanation\nwith a break", "", "Open 10:00 to 18:00."], style,
        openai_api_key="k", config={"conversationalTone": "informal", "languageComplexity": "simple"},
    )
    assert result == ["Rewritten\nexplanation", "", "Open 10:00 to 18:00."]
    prompt, inputs = mock_chat.call_args.args
    assert "factual correctness" in prompt and "core message" in prompt
    assert "opening hours" in prompt and "practical answers" in prompt
    assert '"conversationalTone": "informal"' in prompt
    assert '"languageComplexity": "simple"' in prompt
    assert "[[NL]]" in inputs


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_original_explanation_style_needs_no_model(mock_chat):
    assert replace_reply_batch(["Original"], {"explanationStyle": "original"}) == ["Original"]
    mock_chat.assert_not_called()


@pytest.mark.parametrize("response", ["1. One", "2. One\n1. Two", "1. One\n2. ", "One\nTwo"])
@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_invalid_explanation_response_fails_without_mutating_sources(mock_chat, response):
    mock_chat.return_value = response
    messages = ["One", "Two"]
    with pytest.raises(RuntimeError, match="Explanation style adaptation"):
        explanation_text_batch(messages, "narrative", openai_api_key="k")
    assert messages == ["One", "Two"]


@pytest.mark.parametrize("style", ["bogus", None, [], {}])
def test_invalid_explanation_style_is_rejected(style):
    with pytest.raises(ValueError, match="explanationStyle"):
        replace_reply_batch(["Original"], {"explanationStyle": style})


def test_explanation_style_requires_credentials():
    with pytest.raises(RuntimeError, match="API key is required"):
        replace_reply_batch(["Original"], {"explanationStyle": "direct"})


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_tone_and_explanation_are_both_applied_through_grouped_payload(mock_chat):
    mock_chat.side_effect = ["1. Friendly explanation", "1. Narrative explanation"]
    result = replace_reply_batch(["An explanation"], {"presentation": {
        "conversationalTone": "informal", "explanationStyle": "narrative",
    }}, openai_api_key="k")
    assert result == ["Narrative explanation"]
    assert "informal" in mock_chat.call_args_list[0].args[0]
    assert "narrative" in mock_chat.call_args_list[1].args[0]


@pytest.mark.parametrize("grouped", [False, True])
def test_language_complexity_is_the_final_transformation(grouped):
    from contextlib import ExitStack

    module = "besser.generators.agents.agent_personalization"
    stages = [
        "translate_text_batch", "style_text_batch", "sentence_length_batch",
        "replace_content_profile_batch", "explanation_text_batch", "complexity_text_batch",
    ]
    config = {
        "agentLanguage": "french", "conversationalTone": "informal",
        "sentenceLength": "concise", "userProfileModel": {"User": {}},
        "explanationStyle": "narrative", "languageComplexity": "simple",
    }
    if grouped:
        profile = config.pop("userProfileModel")
        config = {"presentation": config, "userProfileModel": profile}
    calls = []

    def transform(stage):
        def run(messages, *args, **kwargs):
            calls.append((stage, list(messages)))
            return [*messages, stage]
        return run

    with ExitStack() as stack:
        for stage in stages:
            stack.enter_context(patch(f"{module}.{stage}", side_effect=transform(stage)))
        result = replace_reply_batch(["Original"], config, openai_api_key="k")

    assert [stage for stage, _ in calls] == stages
    assert calls[-1][1] == ["Original", *stages[:-1]]
    assert result == ["Original", *stages]
