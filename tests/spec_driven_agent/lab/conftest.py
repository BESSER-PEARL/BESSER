"""A scripted LLM client for the lab agent's tests.

``scripted_client(responses, stream=False, cost_per_call=0.0)`` (a fixture) builds a
client that answers each call with the next scripted response; a scripted
exception is raised instead. ``tool_use`` and ``end_turn`` build responses; import
them with ``from tests.spec_driven_agent.lab.conftest import end_turn, tool_use``.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

_ids = itertools.count(1)


class Block:
    """A content block shaped like the provider SDKs' (attributes, not keys)."""

    def __init__(self, block_type: str, **fields: Any):
        self.type = block_type
        for name, value in fields.items():
            setattr(self, name, value)


def tool_use(name: str, input: dict, *more: tuple[str, dict], text: str = "") -> dict:
    """A reply calling ``name`` with ``input``, then each ``(name, input)`` in ``more``, in one turn."""
    content = [Block("text", text=text)] if text else []
    for call_name, call_input in ((name, input), *more):
        content.append(Block("tool_use", id=f"toolu_{next(_ids)}", name=call_name, input=call_input))
    return {"stop_reason": "tool_use", "content": content}


def end_turn(text: str = "Done.") -> dict:
    """A final reply with ``text``."""
    return {"stop_reason": "end_turn", "content": [Block("text", text=text)]}


class _Usage:
    def __init__(self):
        self.estimated_cost = 0.0


class ScriptedClient:
    """Answers ``chat`` with the scripted responses, in order, and records every call."""

    model = "scripted-model"
    max_tokens = 4096

    def __init__(self, responses: list, *, cost_per_call: float = 0.0):
        self.responses = list(responses)
        self.cost_per_call = cost_per_call
        self.usage = _Usage()
        #: (system, a copy of the messages, the tool names) of each call
        self.calls: list[tuple[str, list[dict], list[str]]] = []

    def _next(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        self.calls.append((system, list(messages), [tool["name"] for tool in tools]))
        if not self.responses:
            raise AssertionError(f"the scripted client has no response left for call {len(self.calls)}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        self.usage.estimated_cost += self.cost_per_call
        return response

    def chat(self, system: str, messages: list[dict], tools: list[dict], *, force_tool: str | None = None,
             model_override: str | None = None) -> dict:
        return self._next(system, messages, tools)


class StreamingScriptedClient(ScriptedClient):
    """Also streams: each text block of a response arrives as one ``text_delta``."""

    def __init__(self, responses: list, *, cost_per_call: float = 0.0):
        super().__init__(responses, cost_per_call=cost_per_call)
        self.streamed_calls = 0

    def chat_stream(self, system: str, messages: list[dict], tools: list[dict]):
        response = self._next(system, messages, tools)
        self.streamed_calls += 1
        for block in response["content"]:
            if block.type == "text":
                yield {"type": "text_delta", "text": block.text}
        yield {"type": "message_done", **response}


@pytest.fixture
def scripted_client():
    """Factory: ``scripted_client(responses, stream=False, cost_per_call=0.0)``."""
    def make(responses: list, *, stream: bool = False, cost_per_call: float = 0.0) -> ScriptedClient:
        cls = StreamingScriptedClient if stream else ScriptedClient
        return cls(responses, cost_per_call=cost_per_call)
    return make
