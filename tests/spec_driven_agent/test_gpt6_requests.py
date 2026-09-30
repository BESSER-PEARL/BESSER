"""GPT-6 request shape, per a probe of api.openai.com (2026-09-28).

On /v1/chat/completions, gpt-6-sol and gpt-6-luna 400 on ``max_tokens`` and on
function tools with the default or "low" reasoning, and work with
``max_completion_tokens`` + ``reasoning_effort="none"``. gpt-6-astra rejects
"none" and rejects tools with reasoning on.
"""

from unittest.mock import MagicMock

import pytest

from besser.spec_driven_agent.providers.llm_client import OpenAIProvider

TOOL = {"name": "submit", "description": "Submit the answer.",
        "input_schema": {"type": "object", "properties": {"items": {"type": "string"}}}}
USER = [{"role": "user", "content": "Extract the items."}]


def _openai_request(model, *, tools=True, base_url=None, stream=False) -> dict:
    provider = OpenAIProvider(api_key="k", model=model, base_url=base_url)
    provider._client = MagicMock()
    offered = [TOOL] if tools else []
    if stream:
        provider._client.chat.completions.create.return_value = iter(())
        list(provider.chat_stream(system="s", messages=USER, tools=offered))
    else:
        response = MagicMock()
        response.usage = None
        response.model = model
        choice = MagicMock()
        choice.message.content = "hi"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response.choices = [choice]
        provider._client.chat.completions.create.return_value = response
        provider.chat(system="s", messages=USER, tools=offered)
    return provider._client.chat.completions.create.call_args.kwargs


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-6-astra"])
@pytest.mark.parametrize("stream", [False, True])
def test_gpt_6_uses_max_completion_tokens(model, stream):
    request = _openai_request(model, stream=stream)
    assert "max_completion_tokens" in request
    assert "max_tokens" not in request


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize("stream", [False, True])
def test_gpt_6_tools_turn_reasoning_off_on_openai(model, stream):
    request = _openai_request(model, stream=stream)
    assert request["reasoning_effort"] == "none"
    assert "temperature" not in request


def test_gpt_6_without_tools_leaves_reasoning_alone():
    assert "reasoning_effort" not in _openai_request("gpt-6-luna", tools=False)


def test_gpt_6_astra_is_never_sent_reasoning_none():
    """astra 400s on "none"; its tools 400 maps to the pick-another message."""
    assert "reasoning_effort" not in _openai_request("gpt-6-astra")


def test_a_gateway_still_suppresses_reasoning_none_for_gpt_6():
    request = _openai_request("gpt-6-luna", base_url="https://api.commandcode.ai/v1")
    assert "reasoning_effort" not in request
