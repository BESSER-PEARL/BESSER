"""Claude Fable 5 / 5.1 and Opus 5.5 request shape, and refusals.

Vendor-documented, not exercised live (the org key has no credits): Fable 5 /
5.1 and Opus 5.5 400 on ``thinking: disabled``; Fable 5.1 and Opus 5.5 400 on
a forced ``tool_choice`` (tool / any), so a forced call must become
``tool_choice: auto`` plus an instruction, checked for the tool call.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic.types as at
import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
from besser.spec_driven_agent.errors import UpstreamLLMError
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers import llm_client
from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient


def _strict_tool(tool):
    return llm_client._strict_tool(tool)


TOOL = {
    "name": "submit",
    "description": "Submit the answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "count": {"type": "integer", "minimum": 0},
        },
        "required": ["items"],
    },
}
OTHER_TOOL = {"name": "read_file", "description": "Read.",
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
USER = [{"role": "user", "content": "Extract the items."}]


def _text(text="plain answer"):
    return SimpleNamespace(stop_reason="end_turn", content=[at.TextBlock(type="text", text=text)],
                           usage=at.Usage(input_tokens=1, output_tokens=1))


def _tool_use(name="submit", payload=None):
    block = at.ToolUseBlock(type="tool_use", id="toolu_1", name=name,
                            input=payload if payload is not None else {"items": ["a"]})
    return SimpleNamespace(stop_reason="tool_use", content=[block],
                           usage=at.Usage(input_tokens=1, output_tokens=1))


def _claude(model, *responses):
    client = ClaudeLLMClient(api_key="sk-ant-test", model=model)
    client._client = MagicMock()
    client._client.messages.create.side_effect = list(responses) or [_text()]
    return client


def _requests(client) -> list[dict]:
    return [call.kwargs for call in client._client.messages.create.call_args_list]


NEW_ANTHROPIC = ["claude-fable-5-1", "claude-fable-5", "claude-opus-5-5",
                 "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]


@pytest.mark.parametrize("model", NEW_ANTHROPIC)
@pytest.mark.parametrize("force", [None, "submit"])
def test_no_sampling_parameter_reaches_an_anthropic_model(model, force):
    """Guard: sampling is a 400 on Fable 5.x / Opus 5.x / Sonnet 5."""
    client = _claude(model, _tool_use())
    client.chat(system="s", messages=USER, tools=[TOOL], force_tool=force)
    for request in _requests(client):
        assert not {"temperature", "top_p", "top_k"} & request.keys()


@pytest.mark.parametrize("model", [
    "claude-fable-5-1", "claude-fable-5", "claude-opus-5-5",
    "us.anthropic.claude-fable-5-1", "anthropic.claude-opus-5-5",
])
def test_a_forced_call_never_disables_thinking_where_that_is_a_400(model):
    client = _claude(model, _tool_use())
    client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")
    for request in _requests(client):
        assert "thinking" not in request


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5",
                                   "claude-sonnet-4-6", "claude-opus-4-8"])
def test_models_that_accept_it_keep_the_forced_tool_and_thinking_off(model):
    """PIA (Bedrock) takes a forced tool only with thinking off: unchanged."""
    client = _claude(model, _tool_use())
    client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")
    (request,) = _requests(client)
    assert request["tool_choice"] == {"type": "tool", "name": "submit"}
    assert request["thinking"] == {"type": "disabled"}


def test_fable_5_keeps_its_forced_tool_choice():
    """Fable 5 accepts a forced tool; only disabling thinking is a 400."""
    client = _claude("claude-fable-5", _tool_use())
    client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")
    (request,) = _requests(client)
    assert request["tool_choice"] == {"type": "tool", "name": "submit"}


# ----------------------------------------------------------------------
# Forced-tool fallback on Fable 5.1 / Opus 5.5
# ----------------------------------------------------------------------

@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5-5"])
def test_forced_tool_becomes_auto_plus_an_instruction_and_parses_the_call(model):
    client = _claude(model, _tool_use(payload={"items": ["x", "y"]}))

    result = client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")

    (request,) = _requests(client)
    assert request["tool_choice"] == {"type": "auto"}
    last_block = request["messages"][-1]["content"][-1]
    assert "`submit`" in last_block["text"]
    (sent_tool,) = request["tools"]
    assert sent_tool["strict"] is True
    assert sent_tool["input_schema"]["additionalProperties"] is False
    tool_blocks = [b for b in result["content"] if b.type == "tool_use"]
    assert tool_blocks[0].name == "submit"
    assert tool_blocks[0].input == {"items": ["x", "y"]}


def test_the_caller_messages_are_not_mutated():
    client = _claude("claude-fable-5-1", _tool_use())
    messages = [{"role": "user", "content": "Extract the items."}]
    client.chat(system="s", messages=messages, tools=[TOOL], force_tool="submit")
    assert messages == [{"role": "user", "content": "Extract the items."}]


def test_a_reply_without_the_tool_is_asked_again_once():
    client = _claude("claude-fable-5-1", _text(), _tool_use())

    result = client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")

    first, second = _requests(client)
    assert "did not call it" in second["messages"][-1]["content"][-1]["text"]
    assert result["stop_reason"] == "tool_use"


def test_a_second_miss_returns_the_text_for_the_caller_fallback():
    client = _claude("claude-opus-5-5", _text("{\"items\": []}"), _text("{\"items\": []}"))

    result = client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")

    assert len(_requests(client)) == 2
    assert result["stop_reason"] == "end_turn"


def test_a_refusal_is_not_re_asked_and_carries_its_category():
    refusal = SimpleNamespace(stop_reason="refusal", content=[],
                              stop_details=SimpleNamespace(category="cyber"),
                              usage=at.Usage(input_tokens=1, output_tokens=0))
    client = _claude("claude-fable-5-1", refusal)

    result = client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")

    assert len(_requests(client)) == 1
    assert result["stop_reason"] == "refusal"
    assert result["refusal_category"] == "cyber"


def test_a_full_toolset_is_sent_unchanged_to_keep_the_cache_prefix():
    client = _claude("claude-fable-5-1", _tool_use(name="read_file", payload={"path": "a"}))
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]},
    ]

    client.chat(system="s", messages=history, tools=[TOOL, OTHER_TOOL], force_tool="read_file")

    (request,) = _requests(client)
    assert all("strict" not in tool for tool in request["tools"])
    last = request["messages"][-1]["content"]
    assert last[0]["type"] == "tool_result"
    assert "`read_file`" in last[-1]["text"]


def test_a_rejected_strict_schema_is_retried_without_strict():
    client = _claude("claude-opus-5-5")
    client._client.messages.create.side_effect = [
        Exception("400 invalid_request_error: tools.0.strict: schema not supported"),
        _tool_use(),
    ]

    result = client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")

    first, second = _requests(client)
    assert first["tools"][0]["strict"] is True
    assert "strict" not in second["tools"][0]
    assert result["stop_reason"] == "tool_use"


def test_an_unrelated_error_still_raises():
    client = _claude("claude-opus-5-5")
    client._client.messages.create.side_effect = [Exception("400 bad request: max_tokens")]
    with pytest.raises(UpstreamLLMError):
        client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit")


def test_a_planning_override_that_accepts_forcing_is_still_forced():
    """The rule keys on the model being called: Fable 5.1's Haiku planner."""
    client = _claude("claude-fable-5-1", _tool_use())
    client.chat(system="s", messages=USER, tools=[TOOL], force_tool="submit",
                model_override="claude-haiku-4-5")
    (request,) = _requests(client)
    assert request["tool_choice"] == {"type": "tool", "name": "submit"}


class TestStrictSchema:

    def test_objects_are_closed_and_unsupported_constraints_dropped(self):
        schema = _strict_tool(TOOL)["input_schema"]
        assert schema["additionalProperties"] is False
        assert "maxItems" not in schema["properties"]["items"]
        assert "minimum" not in schema["properties"]["count"]
        assert TOOL["input_schema"]["properties"]["count"] == {"type": "integer", "minimum": 0}

    def test_nested_objects_are_closed(self):
        tool = {"name": "t", "input_schema": {"type": "object", "properties": {
            "rows": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}}}}}}}
        rows = _strict_tool(tool)["input_schema"]["properties"]["rows"]
        assert rows["items"]["additionalProperties"] is False

    def test_a_property_named_like_a_keyword_survives(self):
        tool = {"name": "t", "input_schema": {"type": "object", "properties": {
            "minimum": {"type": "number"}}}}
        assert "minimum" in _strict_tool(tool)["input_schema"]["properties"]

    def test_an_open_object_is_left_non_strict(self):
        tool = {"name": "t", "input_schema": {"type": "object", "properties": {},
                                              "additionalProperties": {"type": "string"}}}
        assert _strict_tool(tool) is None


# ----------------------------------------------------------------------
# Refusal in Phase 2
# ----------------------------------------------------------------------

class _ScriptedClient:
    model = "claude-fable-5-1"
    max_tokens = 16_384

    def __init__(self, response):
        self.usage = SimpleNamespace(estimated_cost=0.0,
                                     summary=lambda: {"api_calls": 1, "cost_usd": 0.0})
        self._response = response

    def chat(self, system=None, messages=None, tools=None, **kw):
        return self._response


def test_a_refusal_ends_phase_2_with_a_clear_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "analyze_gaps_via_llm", lambda **kw: None)
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=PrimitiveDataType("str"), is_id=True)}
    orch = LLMOrchestrator(
        llm_client=_ScriptedClient({"stop_reason": "refusal", "content": [],
                                    "refusal_category": "cyber"}),
        domain_model=DomainModel(name="Library", types={book}),
        output_dir=str(tmp_path), enable_tracing=False,
        enable_checkpointing=False, enable_toolchain_validation=False,
    )

    orch._run_phase2("build it", extra_issues=[])

    assert orch._phase2_stop_reason == "api_error"
    assert "declined" in orch._phase2_api_error
    assert "cyber" in orch._phase2_api_error
