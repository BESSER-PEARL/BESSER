"""A Claude call with no tools must not send an empty ``tools`` list.

The KG cleanup endpoints use the provider clients for plain text completions.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic.types as at

from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient


def _client():
    client = ClaudeLLMClient(api_key="sk-ant-test", model="claude-sonnet-5")
    client._client = MagicMock()
    client._client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=[at.TextBlock(type="text", text="hello")],
        usage=at.Usage(input_tokens=1, output_tokens=1),
    )
    return client


def test_no_tools_means_no_tools_parameter():
    client = _client()

    result = client.chat(system="s", messages=[{"role": "user", "content": "hi"}], tools=[])

    (request,) = [c.kwargs for c in client._client.messages.create.call_args_list]
    assert "tools" not in request
    assert result["content"][0].text == "hello"


def test_tools_are_still_sent_when_given():
    client = _client()
    tool = {"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}

    client.chat(system="s", messages=[{"role": "user", "content": "hi"}], tools=[tool])

    (request,) = [c.kwargs for c in client._client.messages.create.call_args_list]
    assert request["tools"][0]["name"] == "t"
