"""Tests for OpenAI provider: tool format translation, response conversion,
factory function, and pricing."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from besser.generators.llm.llm_client import (
    LLMProvider,
    OpenAIProvider,
    UsageTracker,
    _OpenAIUsageAdapter,
    _TextBlock,
    _ToolUseBlock,
    _anthropic_tools_to_openai,
    _get_pricing,
    _openai_messages_to_api,
    _openai_response_to_common,
    create_llm_client,
)


# ======================================================================
# Tool format translation
# ======================================================================

class TestAnthropicToolsToOpenAI:

    def test_single_tool_conversion(self):
        anthropic_tools = [
            {
                "name": "read_file",
                "description": "Read a file from the workspace.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            },
        ]
        result = _anthropic_tools_to_openai(anthropic_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        func = result[0]["function"]
        assert func["name"] == "read_file"
        assert func["description"] == "Read a file from the workspace."
        assert func["parameters"]["type"] == "object"
        assert "path" in func["parameters"]["properties"]

    def test_multiple_tools_conversion(self):
        tools = [
            {"name": "tool_a", "description": "A", "input_schema": {"type": "object", "properties": {}}},
            {"name": "tool_b", "description": "B", "input_schema": {"type": "object", "properties": {}}},
            {"name": "tool_c", "description": "C", "input_schema": {"type": "object", "properties": {}}},
        ]
        result = _anthropic_tools_to_openai(tools)
        assert len(result) == 3
        names = [r["function"]["name"] for r in result]
        assert names == ["tool_a", "tool_b", "tool_c"]
        assert all(r["type"] == "function" for r in result)

    def test_empty_tools(self):
        assert _anthropic_tools_to_openai([]) == []

    def test_missing_description(self):
        tools = [{"name": "my_tool", "input_schema": {"type": "object", "properties": {}}}]
        result = _anthropic_tools_to_openai(tools)
        assert result[0]["function"]["description"] == ""

    def test_missing_input_schema(self):
        tools = [{"name": "my_tool", "description": "A tool"}]
        result = _anthropic_tools_to_openai(tools)
        assert "parameters" not in result[0]["function"]


# ======================================================================
# Message format translation
# ======================================================================

class TestOpenAIMessagesToAPI:

    def test_system_prompt_as_first_message(self):
        result = _openai_messages_to_api("You are a helper.", [])
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helper."

    def test_plain_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _openai_messages_to_api("sys", messages)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_tool_result_conversion(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": '{"status": "ok"}',
                    },
                ],
            },
        ]
        result = _openai_messages_to_api("sys", messages)
        assert len(result) == 2
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_123"
        assert result[1]["content"] == '{"status": "ok"}'

    def test_assistant_message_with_tool_use_blocks(self):
        """Assistant messages with Anthropic content blocks are converted."""
        text_block = SimpleNamespace(type="text", text="Let me help.")
        tool_block = SimpleNamespace(
            type="tool_use", id="call_abc", name="read_file",
            input={"path": "main.py"},
        )
        messages = [{"role": "assistant", "content": [text_block, tool_block]}]
        result = _openai_messages_to_api("sys", messages)
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me help."
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"]) == {"path": "main.py"}

    def test_assistant_message_with_dict_blocks(self):
        """Dict-based content blocks also work."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "OK"},
                    {"type": "tool_use", "id": "tc1", "name": "write_file", "input": {"path": "a.py", "content": "x"}},
                ],
            },
        ]
        result = _openai_messages_to_api("sys", messages)
        assert result[1]["content"] == "OK"
        assert result[1]["tool_calls"][0]["function"]["name"] == "write_file"


# ======================================================================
# Response format translation
# ======================================================================

class TestOpenAIResponseToCommon:

    def _make_response(self, content=None, tool_calls=None, finish_reason="stop"):
        """Create a mock OpenAI ChatCompletion response."""
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])

    def test_text_only_response(self):
        resp = self._make_response(content="Hello world")
        result = _openai_response_to_common(resp)
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0].type == "text"
        assert result["content"][0].text == "Hello world"

    def test_tool_call_response(self):
        tc = SimpleNamespace(
            id="call_xyz",
            function=SimpleNamespace(name="read_file", arguments='{"path": "main.py"}'),
        )
        resp = self._make_response(content=None, tool_calls=[tc], finish_reason="tool_calls")
        result = _openai_response_to_common(resp)
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block.type == "tool_use"
        assert block.id == "call_xyz"
        assert block.name == "read_file"
        assert block.input == {"path": "main.py"}

    def test_mixed_text_and_tool_calls(self):
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="write_file", arguments='{"path":"a.py","content":"x"}'),
        )
        resp = self._make_response(content="Working on it.", tool_calls=[tc], finish_reason="tool_calls")
        result = _openai_response_to_common(resp)
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 2
        assert result["content"][0].type == "text"
        assert result["content"][1].type == "tool_use"

    def test_malformed_tool_arguments(self):
        tc = SimpleNamespace(
            id="call_bad",
            function=SimpleNamespace(name="some_tool", arguments="not-json"),
        )
        resp = self._make_response(tool_calls=[tc], finish_reason="tool_calls")
        result = _openai_response_to_common(resp)
        assert result["content"][0].input == {}

    def test_empty_response(self):
        resp = self._make_response(content=None, tool_calls=None, finish_reason="stop")
        result = _openai_response_to_common(resp)
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == []


# ======================================================================
# Content block types
# ======================================================================

class TestContentBlocks:

    def test_text_block(self):
        b = _TextBlock("hello")
        assert b.type == "text"
        assert b.text == "hello"

    def test_tool_use_block(self):
        b = _ToolUseBlock("id_1", "my_tool", {"key": "val"})
        assert b.type == "tool_use"
        assert b.id == "id_1"
        assert b.name == "my_tool"
        assert b.input == {"key": "val"}


# ======================================================================
# Usage adapter
# ======================================================================

class TestOpenAIUsageAdapter:

    def test_adapter_maps_fields(self):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        adapted = _OpenAIUsageAdapter(usage)
        assert adapted.input_tokens == 100
        assert adapted.output_tokens == 50
        assert adapted.cache_creation_input_tokens == 0
        assert adapted.cache_read_input_tokens == 0

    def test_adapter_handles_none(self):
        usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None)
        adapted = _OpenAIUsageAdapter(usage)
        assert adapted.input_tokens == 0
        assert adapted.output_tokens == 0

    def test_usage_tracker_records_via_adapter(self):
        tracker = UsageTracker("gpt-4o")
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500)
        tracker.record(_OpenAIUsageAdapter(usage))
        assert tracker.input_tokens == 1000
        assert tracker.output_tokens == 500
        assert tracker.api_calls == 1


# ======================================================================
# Pricing
# ======================================================================

class TestOpenAIPricing:

    def test_gpt4o_pricing(self):
        p = _get_pricing("gpt-4o")
        assert p["input"] == 2.5
        assert p["output"] == 10.0
        assert p["cache_write"] == 0
        assert p["cache_read"] == 0

    def test_gpt4o_mini_pricing(self):
        p = _get_pricing("gpt-4o-mini")
        assert p["input"] == 0.15
        assert p["output"] == 0.6

    def test_gpt5_pricing(self):
        p = _get_pricing("gpt-5")
        assert p["input"] == 5.0
        assert p["output"] == 20.0

    def test_o3_pricing(self):
        p = _get_pricing("o3")
        assert p["input"] == 10.0
        assert p["output"] == 40.0

    def test_o3_mini_pricing(self):
        p = _get_pricing("o3-mini")
        assert p["input"] == 1.1
        assert p["output"] == 4.4

    def test_gpt4o_mini_not_matched_as_gpt4o(self):
        """gpt-4o-mini should get its own pricing, not gpt-4o's."""
        p_mini = _get_pricing("gpt-4o-mini")
        p_4o = _get_pricing("gpt-4o")
        assert p_mini["input"] != p_4o["input"]

    def test_cost_estimation(self):
        tracker = UsageTracker("gpt-4o")
        tracker.input_tokens = 1_000_000
        tracker.output_tokens = 100_000
        # 1M * 2.5/1M + 100K * 10.0/1M = 2.5 + 1.0 = 3.5
        assert abs(tracker.estimated_cost - 3.5) < 0.001

    def test_claude_pricing_still_works(self):
        """Anthropic pricing unaffected by OpenAI additions."""
        p = _get_pricing("claude-sonnet-4-20250514")
        assert p["input"] == 3.0
        assert p["output"] == 15.0

    def test_unknown_model_defaults_to_sonnet(self):
        p = _get_pricing("some-unknown-model")
        assert p["input"] == 3.0  # sonnet default


# ======================================================================
# Factory function
# ======================================================================

class TestCreateLLMClient:

    @patch("besser.generators.llm.llm_client._resolve_api_key", return_value="sk-ant-test")
    @patch("besser.generators.llm.llm_client._resolve_base_url", return_value=None)
    @patch("besser.generators.llm.llm_client.ClaudeLLMClient")
    def test_anthropic_provider(self, MockClaude, mock_base, mock_key):
        MockClaude.return_value = MagicMock(spec=LLMProvider)
        client = create_llm_client(provider="anthropic", api_key="sk-ant-test")
        MockClaude.assert_called_once()

    @patch("besser.generators.llm.llm_client._resolve_openai_api_key", return_value="sk-test")
    @patch("besser.generators.llm.llm_client.OpenAIProvider")
    def test_openai_provider(self, MockOpenAI, mock_key):
        MockOpenAI.return_value = MagicMock(spec=LLMProvider)
        client = create_llm_client(provider="openai", api_key="sk-test")
        MockOpenAI.assert_called_once()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(provider="mistral", api_key="x")

    @patch("besser.generators.llm.llm_client._resolve_api_key", return_value="sk-ant-test")
    @patch("besser.generators.llm.llm_client._resolve_base_url", return_value=None)
    @patch("besser.generators.llm.llm_client.ClaudeLLMClient")
    def test_default_is_anthropic(self, MockClaude, mock_base, mock_key):
        MockClaude.return_value = MagicMock(spec=LLMProvider)
        client = create_llm_client(api_key="sk-ant-test")
        MockClaude.assert_called_once()


# ======================================================================
# OpenAI provider integration (mocked SDK)
# ======================================================================

def _make_openai_provider():
    """Create an OpenAIProvider with a mocked OpenAI client."""
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        # Force re-evaluation of the lazy import by creating a new instance
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        provider._client = mock_client
        return provider


class TestOpenAIProviderChat:

    def test_chat_text_response(self):
        provider = _make_openai_provider()

        # Mock a text-only response
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        message = SimpleNamespace(content="Hello!", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)
        provider._client.chat.completions.create.return_value = mock_resp

        result = provider.chat(system="Be helpful.", messages=[], tools=[])
        assert result["stop_reason"] == "end_turn"
        assert result["content"][0].text == "Hello!"
        assert provider.usage.input_tokens == 100
        assert provider.usage.output_tokens == 50

    def test_chat_tool_use_response(self):
        provider = _make_openai_provider()

        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="read_file", arguments='{"path": "main.py"}'),
        )
        usage = SimpleNamespace(prompt_tokens=200, completion_tokens=30)
        message = SimpleNamespace(content=None, tool_calls=[tc])
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)
        provider._client.chat.completions.create.return_value = mock_resp

        result = provider.chat(system="sys", messages=[], tools=[
            {"name": "read_file", "description": "Read", "input_schema": {"type": "object", "properties": {}}},
        ])
        assert result["stop_reason"] == "tool_use"
        assert result["content"][0].name == "read_file"
        assert result["content"][0].input == {"path": "main.py"}

    def test_chat_passes_tools_in_openai_format(self):
        provider = _make_openai_provider()

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)
        provider._client.chat.completions.create.return_value = mock_resp

        tools = [
            {"name": "my_tool", "description": "A tool", "input_schema": {"type": "object", "properties": {}}},
        ]
        provider.chat(system="sys", messages=[], tools=tools)

        call_kwargs = provider._client.chat.completions.create.call_args
        passed_tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert passed_tools[0]["type"] == "function"
        assert passed_tools[0]["function"]["name"] == "my_tool"

    def test_chat_no_tools(self):
        provider = _make_openai_provider()

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)
        provider._client.chat.completions.create.return_value = mock_resp

        provider.chat(system="sys", messages=[], tools=[])

        call_kwargs = provider._client.chat.completions.create.call_args
        # tools should not be in kwargs when empty
        assert "tools" not in (call_kwargs.kwargs or {})


class TestOpenAIProviderStream:

    def test_stream_text(self):
        provider = _make_openai_provider()

        # Simulate streaming chunks
        chunk1 = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Hello", tool_calls=None),
                finish_reason=None,
            )],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=" world", tool_calls=None),
                finish_reason=None,
            )],
            usage=None,
        )
        chunk3 = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )

        provider._client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        events = list(provider.chat_stream(system="sys", messages=[], tools=[]))
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 2
        assert text_events[0]["text"] == "Hello"
        assert text_events[1]["text"] == " world"

        done_event = [e for e in events if e["type"] == "message_done"][0]
        assert done_event["stop_reason"] == "end_turn"
        assert done_event["content"][0].text == "Hello world"


# ======================================================================
# OpenAI import error
# ======================================================================

class TestOpenAIImportError:

    def test_import_error_message(self):
        """If the openai package is not installed, a clear error is raised."""
        import sys
        # Temporarily make 'openai' unimportable
        sentinel = object()
        original = sys.modules.get("openai", sentinel)
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="pip install openai"):
                OpenAIProvider(api_key="sk-test")
        finally:
            if original is sentinel:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = original


# ======================================================================
# LLMProvider ABC
# ======================================================================

class TestLLMProviderABC:

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_claude_is_llm_provider(self):
        """ClaudeLLMClient implements LLMProvider."""
        from besser.generators.llm.llm_client import ClaudeLLMClient
        assert issubclass(ClaudeLLMClient, LLMProvider)

    def test_openai_is_llm_provider(self):
        """OpenAIProvider implements LLMProvider."""
        assert issubclass(OpenAIProvider, LLMProvider)
