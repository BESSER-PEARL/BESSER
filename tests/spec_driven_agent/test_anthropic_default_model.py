"""The Spec-Driven Anthropic default is the current Sonnet generation.

Moved from ``claude-sonnet-4-6`` ($3 / $15) to ``claude-sonnet-5`` ($2 / $10)
in lockstep with the modeling-agent's smart-generation default, so the two
agree. Sonnet 5 400s on temperature / top_p / top_k and on budget_tokens, and
sends about half of forced-tool arrays as JSON strings.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic.types as at

from besser.spec_driven_agent.providers.llm_client import (
    DEFAULT_MODELS,
    ClaudeLLMClient,
    _get_pricing,
)

TOOL = {
    "name": "submit",
    "description": "Submit the answer.",
    "input_schema": {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        "required": ["items"],
    },
}


def test_the_anthropic_default_is_sonnet_5():
    assert ClaudeLLMClient.DEFAULT_MODEL == "claude-sonnet-5"
    assert DEFAULT_MODELS["anthropic"] == "claude-sonnet-5"
    assert ClaudeLLMClient(api_key="sk-ant-test").model == "claude-sonnet-5"


def test_the_default_bills_at_sonnet_5_rates():
    pricing = _get_pricing(ClaudeLLMClient.DEFAULT_MODEL)
    assert (pricing["input"], pricing["output"]) == (2.0, 10.0)


def test_a_default_forced_call_sends_no_sampling_or_budget_and_coerces_arrays():
    client = ClaudeLLMClient(api_key="sk-ant-test")
    client._client = MagicMock()
    block = at.ToolUseBlock(type="tool_use", id="toolu_1", name="submit",
                            input={"items": '["a", "b"]'})
    client._client.messages.create.return_value = SimpleNamespace(
        stop_reason="tool_use", content=[block],
        usage=at.Usage(input_tokens=1, output_tokens=1))

    result = client.chat(system="s", messages=[{"role": "user", "content": "x"}],
                         tools=[TOOL], force_tool="submit")

    (request,) = [c.kwargs for c in client._client.messages.create.call_args_list]
    assert request["model"] == "claude-sonnet-5"
    assert not {"temperature", "top_p", "top_k"} & request.keys()
    assert "budget_tokens" not in (request.get("thinking") or {})
    assert result["content"][0].input == {"items": ["a", "b"]}
