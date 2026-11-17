"""An OpenAI reply with an unusable tool call must leave a history the next request can carry.

Chat Completions rejects a request whose assistant ``tool_calls`` entry has no id, an empty
function name or arguments that are not JSON, or is not answered by exactly one ``tool``
message. Three reply shapes broke that: unparseable arguments under ``finish_reason="stop"``
(mapped to ``end_turn``, so the call was replayed with no answer), a call with no name
(replayed as ``function.name: ""``) and a call with no id (answered as ``tool_call_id: None``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import (
    OpenAIProvider,
    UsageTracker,
    _openai_messages_to_api,
    _openai_response_to_common,
)
from tests.spec_driven_agent.message_validity import openai_violations

SYSTEM = "sys"
TOOLS = [{"name": "read_file", "description": "Read",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]

# (id, name, arguments, finish_reason) of the one tool call in the reply.
CASES = {
    "unparseable arguments, finish stop": ("call_a", "read_file", '{"path": "app.py', "stop"),
    "no name": ("call_b", None, '{"path": "app.py"}', "tool_calls"),
    "no id": (None, "read_file", '{"path": "app.py"}', "tool_calls"),
}


def _chat_reply(call_id, name, arguments, finish_reason) -> dict:
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    message = SimpleNamespace(content=None, tool_calls=[call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])
    return _openai_response_to_common(response, TOOLS)


def _stream_reply(call_id, name, arguments, finish_reason) -> dict:
    with patch.dict("sys.modules", {"openai": MagicMock()}):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
    provider._client = MagicMock()
    delta = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        index=0, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))])
    chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])
    provider._client.chat.completions.create.return_value = iter([chunk])
    done = list(provider.chat_stream(system=SYSTEM, messages=[], tools=TOOLS))[-1]
    return {key: value for key, value in done.items() if key != "type"}


class _Client:
    """Replies once with ``first``, then ends the turn; records every request."""

    model = "mock"

    def __init__(self, first: dict):
        self.usage = UsageTracker("mock")
        self.requests: list[list[dict]] = []
        self._first = first

    def chat(self, system, messages, tools):
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            return self._first
        return {"stop_reason": "end_turn", "content": [SimpleNamespace(type="text", text="done")]}


@pytest.mark.parametrize("make_reply", [_chat_reply, _stream_reply], ids=["chat", "stream"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_request_after_an_unusable_tool_call_is_valid(tmp_path, make_reply, case):
    item = Class(name="Item")
    item.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    client = _Client(make_reply(*CASES[case]))
    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=DomainModel(name="Shop", types={item}),
        output_dir=str(tmp_path), use_streaming=False,
        enable_tracing=False, enable_checkpointing=False, enable_requirements_ledger=False,
    )
    orchestrator._resume_messages = [{"role": "user", "content": "Build it"}]
    # An open checklist item makes an end_turn reply replay into the next request.
    orchestrator.executor.open_tasks = lambda: [{"id": 1, "text": "Add the items page"}]
    orchestrator._run_phase2("Build it")

    assert len(client.requests) >= 2
    assert openai_violations(_openai_messages_to_api(SYSTEM, client.requests[1])) == []
    answer = client.requests[1][-1]["content"]
    assert isinstance(answer, list) and len(answer) == 1
    assert answer[0]["type"] == "tool_result"
    assert "error" in str(answer[0]["content"]).lower()


@pytest.mark.parametrize("make_reply", [_chat_reply, _stream_reply], ids=["chat", "stream"])
def test_synthesised_ids_are_unique(make_reply):
    first = make_reply(None, "read_file", '{"path": "a.py"}', "tool_calls")["content"][0]
    second = make_reply(None, "read_file", '{"path": "a.py"}', "tool_calls")["content"][0]
    assert first.id and second.id and first.id != second.id
