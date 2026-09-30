"""Tests for OpenAI provider: tool format translation, response conversion,
factory function, and pricing."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from besser.spec_driven_agent.providers.llm_client import (
    LLMProvider,
    OpenAIProvider,
    UsageTracker,
    _OpenAIUsageAdapter,
    _TextBlock,
    _ToolUseBlock,
    _anthropic_tools_to_openai,
    _get_pricing,
    _needs_reasoning_none_for_tools,
    _openai_messages_to_api,
    _openai_response_to_common,
    create_llm_client,
    free_tier_available,
    free_tier_model,
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

    @pytest.mark.parametrize("finish_reason", ["tool_calls", "stop", None, "length", "content_filter"])
    def test_tool_call_response(self, finish_reason):
        tc = SimpleNamespace(
            id="call_xyz",
            function=SimpleNamespace(name="read_file", arguments='{"path": "main.py"}'),
        )
        resp = self._make_response(content=None, tool_calls=[tc], finish_reason=finish_reason)
        result = _openai_response_to_common(resp)
        assert result["stop_reason"] == (finish_reason if finish_reason in {"length", "content_filter"} else "tool_use")
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

    @pytest.mark.parametrize("finish_reason", ["tool_calls", "stop"])
    def test_malformed_tool_arguments(self, finish_reason):
        tc = SimpleNamespace(
            id="call_bad",
            function=SimpleNamespace(name="some_tool", arguments="not-json"),
        )
        resp = self._make_response(tool_calls=[tc], finish_reason=finish_reason)
        result = _openai_response_to_common(resp)
        assert result["content"][0].input == {}
        assert result["stop_reason"] == ("tool_use" if finish_reason == "tool_calls" else "end_turn")

    def test_empty_response(self):
        resp = self._make_response(content=None, tool_calls=None, finish_reason="stop")
        result = _openai_response_to_common(resp)
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == []


# ======================================================================
# Tool-name case folding
# ======================================================================

class TestToolNameCaseFolding:
    """A returned tool name is matched against the names we offered, folding
    case. The executor's handler table is exact-match, so ``Read_File`` would
    otherwise come back "Unknown tool" and cost a turn. Folding renames a
    NATIVE tool call only - it never promotes prose to one."""

    TOOLS = [
        {"name": "read_file", "description": "Read",
         "input_schema": {"type": "object", "properties": {}, "required": ["path"]}},
        {"name": "write_file", "description": "Write",
         "input_schema": {"type": "object", "properties": {}}},
    ]

    def _response(self, name, arguments='{"path": "main.py"}', finish_reason="stop"):
        tc = SimpleNamespace(
            id="call_1", function=SimpleNamespace(name=name, arguments=arguments),
        )
        message = SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        )

    @pytest.mark.parametrize("returned", ["read_file", "Read_File", "READ_FILE", "rEaD_fIlE"])
    def test_case_variant_resolves_to_the_offered_name(self, returned):
        result = _openai_response_to_common(self._response(returned), self.TOOLS)
        assert result["content"][0].name == "read_file"
        assert result["stop_reason"] == "tool_use"

    def test_unknown_name_is_left_alone(self):
        """An unrecognised name must reach the executor unchanged so it is
        still reported as an unknown tool - folding is not a guess."""
        result = _openai_response_to_common(self._response("summarise_repo"), self.TOOLS)
        assert result["content"][0].name == "summarise_repo"

    def test_no_tools_offered_leaves_the_name_alone(self):
        result = _openai_response_to_common(self._response("Read_File"))
        assert result["content"][0].name == "Read_File"

    def test_prose_is_never_promoted_to_a_tool_call(self):
        """Folding must not add tool_use blocks. A text-only turn that names a
        tool in prose stays end_turn with no tool call (README: 'Never infer a
        tool call from prose')."""
        message = SimpleNamespace(content="I will now read_file main.py.", tool_calls=None)
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
        )
        result = _openai_response_to_common(resp, self.TOOLS)
        assert result["stop_reason"] == "end_turn"
        assert [b.type for b in result["content"]] == ["text"]


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
        # OpenAI's cached-input discount (early-2026 rates).
        assert p["cache_read"] == 1.25

    def test_gpt4o_mini_pricing(self):
        p = _get_pricing("gpt-4o-mini")
        assert p["input"] == 0.15
        assert p["output"] == 0.6

    def test_gpt5_pricing(self):
        # Early-2026 rates; gpt-5.5 (the pricier tier) is matched first
        # by _get_pricing so plain gpt-5 gets its own cheaper rate.
        p = _get_pricing("gpt-5")
        assert p["input"] == 1.25
        assert p["output"] == 10.0
        p55 = _get_pricing("gpt-5.5")
        assert p55["input"] == 5.0
        assert p55["output"] == 30.0

    @pytest.mark.parametrize(
        ("model", "input_rate", "output_rate"),
        [
            # sol: sources DISAGREE -- the hand-typed table said 5.0/30.0,
            # the vendored table says 4.0/20.0, OpenRouter (resale) 2.0/10.0.
            # Pinned to the vendored figure because that is the table we
            # refresh; unlike terra/luna below it was never checked against an
            # official page. Re-verify if sol ever becomes a model we run.
            ("gpt-5.6-sol", 4.0, 20.0),
            ("gpt-5.6-terra", 2.0, 12.0),
            ("gpt-5.6-luna", 0.2, 1.2),
        ],
    )
    def test_gpt56_variant_pricing(self, model, input_rate, output_rate):
        pricing = _get_pricing(model)
        assert pricing["input"] == input_rate
        assert pricing["output"] == output_rate

    def test_o3_pricing(self):
        """Was pinned at 10.0/40.0 -- o3's launch price, cut ~80% since.

        The vendored table and OpenRouter independently agree on 2.0/8.0, so
        the old expectation was 5x over: it asserted that a stale hand-typed
        number stayed stale, and would have kept ``max_cost_usd`` firing at a
        fifth of the intended spend.
        """
        p = _get_pricing("o3")
        assert p["input"] == 2.0
        assert p["output"] == 8.0

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

    def test_unknown_model_defaults_to_gpt4o(self):
        # _get_pricing falls back to the gpt-4o middle tier for unknown
        # *paid* model ids (see the function's docstring). A tagless name
        # that matches no open family must still get the protective rate.
        p = _get_pricing("some-unknown-model")
        assert p["input"] == 2.5

    @pytest.mark.parametrize(
        "model",
        [
            "qwen3-coder:30b",   # the free-tier model (Ollama tag syntax)
            "qwen2.5-coder",
            "llama3.1:8b",
            "deepseek-coder-v2",
            "mixtral:8x7b",
            "gemma2",
            "anything:with-a-colon",  # tag syntax => self-hosted, $0
        ],
    )
    def test_local_open_models_are_free(self, model):
        # Self-hosted / open-weight models must price at $0 so they never
        # trip max_cost_usd and abort a free run mid-generation.
        p = _get_pricing(model)
        assert p == {
            "input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0,
        }

    def test_paid_model_with_no_colon_still_protected(self):
        # A genuinely-unknown *paid* cloud id (no colon, no open family)
        # keeps the protective gpt-4o fallback — the $0 rule must not
        # over-broaden and remove the cost cap for real spend.
        assert _get_pricing("gpt-6-preview")["input"] == 2.5


# ======================================================================
# Factory function
# ======================================================================

class TestCreateLLMClient:

    @patch("besser.spec_driven_agent.providers.llm_client._resolve_api_key", return_value="sk-ant-test")
    @patch("besser.spec_driven_agent.providers.llm_client._resolve_base_url", return_value=None)
    @patch("besser.spec_driven_agent.providers.llm_client.ClaudeLLMClient")
    def test_anthropic_provider(self, MockClaude, mock_base, mock_key):
        MockClaude.return_value = MagicMock(spec=LLMProvider)
        client = create_llm_client(provider="anthropic", api_key="sk-ant-test")
        MockClaude.assert_called_once()

    @patch("besser.spec_driven_agent.providers.llm_client._resolve_openai_api_key", return_value="sk-test")
    @patch("besser.spec_driven_agent.providers.llm_client.OpenAIProvider")
    def test_openai_provider(self, MockOpenAI, mock_key):
        MockOpenAI.return_value = MagicMock(spec=LLMProvider)
        client = create_llm_client(provider="openai", api_key="sk-test")
        MockOpenAI.assert_called_once()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(provider="cohere", api_key="x")

    def test_free_provider_unconfigured_raises(self, monkeypatch):
        # No BESSER_FREE_LLM_* env => free tier fails fast with a clear
        # message rather than silently hitting a paid provider.
        monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("BESSER_FREE_LLM_MODEL", raising=False)
        with pytest.raises(ValueError, match="free tier is not available"):
            create_llm_client(provider="free")

    def test_free_provider_reads_env_and_injects_header(self, monkeypatch):
        # The free provider ignores any client-supplied key/model/base_url and
        # builds an OpenAIProvider from SERVER env, with the token as a bearer
        # header (never a request field).
        monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://free.example/v1")
        monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", "secret-token")
        monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "qwen3-coder:30b")

        captured = {}

        def _fake_openai_provider(**kwargs):
            captured.update(kwargs)
            return MagicMock(spec=LLMProvider)

        monkeypatch.setattr(
            "besser.spec_driven_agent.providers.llm_client.OpenAIProvider", _fake_openai_provider,
        )
        # Client passes junk model/base_url — all must be ignored for free.
        create_llm_client(
            provider="free", api_key="sk-should-be-ignored",
            model="gpt-4o", base_url="http://evil.example",
        )
        assert captured["base_url"] == "https://free.example/v1"
        assert captured["model"] == "qwen3-coder:30b"
        assert captured["default_headers"] == {"Authorization": "Bearer secret-token"}
        assert captured["api_key"] == "free"  # placeholder, endpoint ignores it

    def test_free_provider_without_token_sends_no_auth_header(self, monkeypatch):
        monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://free.example/v1")
        monkeypatch.delenv("BESSER_FREE_LLM_TOKEN", raising=False)
        monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "qwen3-coder:30b")
        captured = {}
        monkeypatch.setattr(
            "besser.spec_driven_agent.providers.llm_client.OpenAIProvider",
            lambda **kw: captured.update(kw) or MagicMock(spec=LLMProvider),
        )
        create_llm_client(provider="free")
        assert captured["default_headers"] is None


class TestFreeTierConfig:

    def test_available_requires_base_url_and_model(self, monkeypatch):
        monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://free.example/v1")
        monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "qwen3-coder:30b")
        assert free_tier_available() is True
        assert free_tier_model() == "qwen3-coder:30b"

    def test_unavailable_when_base_url_missing(self, monkeypatch):
        monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "qwen3-coder:30b")
        assert free_tier_available() is False

    def test_unavailable_when_model_missing(self, monkeypatch):
        monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "https://free.example/v1")
        monkeypatch.delenv("BESSER_FREE_LLM_MODEL", raising=False)
        assert free_tier_available() is False

    @patch("besser.spec_driven_agent.providers.llm_client._resolve_api_key", return_value="sk-ant-test")
    @patch("besser.spec_driven_agent.providers.llm_client._resolve_base_url", return_value=None)
    @patch("besser.spec_driven_agent.providers.llm_client.ClaudeLLMClient")
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

    @pytest.mark.parametrize("finish_reason", ["tool_calls", "stop", None, "length", "content_filter"])
    def test_complete_streamed_calls_survive_normal_stop_labels(self, finish_reason):
        provider = _make_openai_provider()
        chunks = [SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(index=0, id="call_x",
                function=SimpleNamespace(name="read_file", arguments='{"path":'))]), finish_reason=None)]),
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(index=0, id=None,
                    function=SimpleNamespace(name=None, arguments='"app.py"}'))]), finish_reason=finish_reason)])]
        provider._client.chat.completions.create.return_value = iter(chunks)
        result = list(provider.chat_stream(system="sys", messages=[], tools=[]))[-1]
        assert result["stop_reason"] == (finish_reason if finish_reason in {"length", "content_filter"} else "tool_use")
        assert result["content"][0].input == {"path": "app.py"}
        assert result["provider_finish_reason"] == finish_reason

    def test_streamed_tool_name_is_case_folded_against_the_offered_tools(self):
        """Same folding as the non-streaming path: a name echoed with the
        wrong casing must reach the executor's exact-match handler table
        under the name we offered."""
        provider = _make_openai_provider()
        chunks = [SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                index=0, id="call_x",
                function=SimpleNamespace(name="WRITE_FILE", arguments='{"path":"a.py"}'))]),
            finish_reason="stop")])]
        provider._client.chat.completions.create.return_value = iter(chunks)
        tools = [{"name": "write_file", "description": "Write",
                  "input_schema": {"type": "object", "properties": {}}}]
        result = list(provider.chat_stream(system="sys", messages=[], tools=tools))[-1]
        assert result["content"][0].name == "write_file"
        assert result["stop_reason"] == "tool_use"

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

    def test_stream_requests_usage_chunk(self):
        """OpenAI streaming must opt into the trailing usage chunk via
        ``stream_options={"include_usage": True}`` — without it, every
        chunk's ``usage`` is None and the UsageTracker stays at zero,
        which makes the SSE cost emitter report a frozen $0.00 cost
        across all customise-loop turns. Regression test for that bug.
        """
        provider = _make_openai_provider()

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="hi", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )
        provider._client.chat.completions.create.return_value = iter([chunk])

        list(provider.chat_stream(system="sys", messages=[], tools=[]))

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("stream_options") == {"include_usage": True}

    def test_stream_records_usage_from_trailing_chunk(self):
        """When OpenAI emits a trailing usage-only chunk (the API does
        this when ``include_usage`` is set), the provider must call
        ``UsageTracker.record(...)`` so cost reflects the streaming
        call. Without recording, ``estimated_cost`` stays at $0.00
        and the SSE cost emitter looks frozen.
        """
        provider = _make_openai_provider()

        text_chunk = SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="hello", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )
        # OpenAI's trailing usage-only chunk has empty choices and
        # populated usage.
        usage_chunk = SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=500,
                prompt_tokens_details=None,
            ),
        )
        provider._client.chat.completions.create.return_value = iter(
            [text_chunk, usage_chunk]
        )

        list(provider.chat_stream(system="sys", messages=[], tools=[]))

        assert provider.usage.api_calls == 1
        assert provider.usage.input_tokens == 1000
        assert provider.usage.output_tokens == 500
        assert provider.usage.estimated_cost > 0

    def test_streaming_cost_grows_across_turns(self):
        """End-to-end regression for the frozen-cost bug: simulate a
        multi-turn customise loop where each turn streams a response
        with a trailing usage chunk. ``estimated_cost`` must grow
        monotonically — not freeze at the first turn's value.
        """
        provider = _make_openai_provider()

        def _streamed_turn(in_tokens: int, out_tokens: int):
            text_chunk = SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )],
                usage=None,
            )
            usage_chunk = SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=in_tokens,
                    completion_tokens=out_tokens,
                    prompt_tokens_details=None,
                ),
            )
            return iter([text_chunk, usage_chunk])

        # 5 turns, each adding 1000 in + 500 out tokens
        provider._client.chat.completions.create.side_effect = [
            _streamed_turn(1000, 500) for _ in range(5)
        ]

        costs = []
        for _ in range(5):
            list(provider.chat_stream(system="sys", messages=[], tools=[]))
            costs.append(provider.usage.estimated_cost)

        # Strictly monotonically increasing across turns
        for prev, curr in zip(costs, costs[1:]):
            assert curr > prev, (
                f"Cost froze across turns (got {costs}); the customise "
                "loop is invisible to the cost tracker again."
            )
        assert provider.usage.input_tokens == 5000
        assert provider.usage.output_tokens == 2500
        assert provider.usage.api_calls == 5


# ======================================================================
# reasoning_effort='none' for tool-using gpt-5.6 reasoning models
# ======================================================================

def _provider_with_model(model, base_url=None):
    """OpenAIProvider on ``model`` with a mocked SDK client."""
    mock_openai_module = MagicMock()
    mock_client = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client
    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        provider = OpenAIProvider(api_key="sk-test", model=model,
                                  base_url=base_url)
        provider._client = mock_client
        return provider


_ONE_TOOL = [
    {"name": "my_tool", "description": "A tool",
     "input_schema": {"type": "object", "properties": {}}},
]


def _mock_chat_text(provider):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    message = SimpleNamespace(content="ok", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    provider._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[choice], usage=usage,
    )


def _mock_stream_text(provider):
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
    )
    provider._client.chat.completions.create.return_value = iter([chunk])


class TestReasoningEffortHelper:
    """The ``gpt-5.6`` reasoning trio cannot combine function tools + reasoning
    on /chat/completions; the client sends ``reasoning_effort='none'`` for them
    so tools work. Every other tier must NOT get the param (they 400 on it).
    Verified empirically against the live API (scratchpad probe_tools)."""

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra",
                                       "gpt-5.6-luna", "GPT-5.6-TERRA"])
    def test_gpt56_needs_flag(self, model):
        assert _needs_reasoning_none_for_tools(model) is True

    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
                                       "gpt-5", "gpt-5-mini", "gpt-4.1",
                                       "gpt-4o", "gpt-4o-mini",
                                       "claude-sonnet-4-6"])
    def test_others_do_not(self, model):
        assert _needs_reasoning_none_for_tools(model) is False

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra",
                                       "gpt-5.6-luna"])
    def test_custom_endpoint_never_gets_the_flag(self, model):
        """A gateway re-serving gpt-5.6 validates the param against its own
        enum, and ``none`` is not in it. The keyless free tier answers
        400 ``Invalid option: expected one of "low"|"medium"|"high"|"xhigh"
        |"max"``, which aborted the Phase 2 customization loop and shipped a
        deterministic-only bundle with "output may be incomplete".

        Tools need no flag there — verified against gpt-5.6-luna on
        api.commandcode.ai: ``none`` -> 400; omitted / ``low`` / ``medium``
        each returned a proper tool call.
        """
        assert _needs_reasoning_none_for_tools(
            model, "https://api.commandcode.ai/v1") is False

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
        "HTTPS://API.OPENAI.COM/v1",
    ])
    def test_openai_own_endpoint_still_gets_the_flag(self, base_url):
        """A base_url naming OpenAI itself is not a gateway.

        The sponsored tier (server-paid demo key) ALWAYS passes a base_url,
        because its endpoint lives in server env. Treating "a base_url was
        set" as "this is a gateway" suppressed the flag and made OpenAI
        reject every function tool for gpt-5.6-*.
        """
        assert _needs_reasoning_none_for_tools("gpt-5.6-terra", base_url) is True

    def test_a_lookalike_host_is_still_a_gateway(self):
        """Suffix-matching ``api.openai.com`` would trust any attacker domain."""
        assert _needs_reasoning_none_for_tools(
            "gpt-5.6-terra", "https://api.openai.com.evil.test/v1") is False


class TestReasoningEffortInRequest:

    def test_gpt56_chat_sends_reasoning_none_with_tools(self):
        provider = _provider_with_model("gpt-5.6-terra")
        _mock_chat_text(provider)
        provider.chat(system="sys", messages=[], tools=_ONE_TOOL)
        kw = provider._client.chat.completions.create.call_args.kwargs
        assert kw.get("reasoning_effort") == "none"
        assert "tools" in kw

    def test_gpt56_on_free_gateway_omits_reasoning_effort(self):
        """Regression: the free tier 400s on reasoning_effort='none'.

        Sending it aborted the tool-driven Phase 2 loop with
        "OpenAI API streaming failed: Error code: 400 - Invalid option:
        expected one of \"low\"|\"medium\"|\"high\"|\"xhigh\"|\"max\"",
        so the user received a deterministic-only bundle flagged incomplete.
        """
        provider = _provider_with_model(
            "gpt-5.6-luna", base_url="https://api.commandcode.ai/v1")
        _mock_chat_text(provider)
        provider.chat(system="sys", messages=[], tools=_ONE_TOOL)
        kw = provider._client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in kw
        assert "tools" in kw

    def test_gpt56_on_sponsored_openai_endpoint_sends_reasoning_none(self):
        """The demo tier runs gpt-5.6-terra on OpenAI proper via an explicit
        base_url. Without the flag OpenAI refuses the tools and the Phase 2
        customization loop dies mid-demo."""
        provider = _provider_with_model(
            "gpt-5.6-terra", base_url="https://api.openai.com/v1")
        _mock_chat_text(provider)
        provider.chat(system="sys", messages=[], tools=_ONE_TOOL)
        kw = provider._client.chat.completions.create.call_args.kwargs
        assert kw.get("reasoning_effort") == "none"
        assert "tools" in kw

    def test_gpt56_stream_sends_reasoning_none_with_tools(self):
        provider = _provider_with_model("gpt-5.6-sol")
        _mock_stream_text(provider)
        list(provider.chat_stream(system="sys", messages=[], tools=_ONE_TOOL))
        kw = provider._client.chat.completions.create.call_args.kwargs
        assert kw.get("reasoning_effort") == "none"

    def test_gpt56_no_reasoning_effort_without_tools(self):
        """No tools → no need to disable reasoning; don't send the param."""
        provider = _provider_with_model("gpt-5.6-terra")
        _mock_chat_text(provider)
        provider.chat(system="sys", messages=[], tools=[])
        kw = provider._client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in kw

    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-4o"])
    def test_non_gpt56_never_sends_reasoning_effort(self, model):
        """gpt-5/gpt-4o REJECT reasoning_effort — must never be sent for them."""
        provider = _provider_with_model(model)
        _mock_chat_text(provider)
        provider.chat(system="sys", messages=[], tools=_ONE_TOOL)
        kw = provider._client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in kw


class TestUsageTrackerAccumulation:
    """Tracker-level invariant: repeated ``record()`` calls must
    accumulate. This documents the expected behaviour the streaming
    bug violated indirectly (by never calling ``record()`` at all).
    """

    def test_record_accumulates_across_calls(self):
        tracker = UsageTracker("gpt-4o")
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        costs = []
        for _ in range(5):
            tracker.record(usage)
            costs.append(tracker.estimated_cost)

        for prev, curr in zip(costs, costs[1:]):
            assert curr == pytest.approx(2 * prev) or curr > prev
        assert tracker.api_calls == 5
        assert tracker.input_tokens == 5000
        assert tracker.output_tokens == 2500


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
        from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient
        assert issubclass(ClaudeLLMClient, LLMProvider)

    def test_openai_is_llm_provider(self):
        """OpenAIProvider implements LLMProvider."""
        assert issubclass(OpenAIProvider, LLMProvider)
