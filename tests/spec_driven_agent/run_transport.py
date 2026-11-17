"""Scripted SDK transports for whole-run request tests.

The REAL provider clients (``ClaudeLLMClient``, ``OpenAIProvider``) run with
their SDK client replaced by a fake that records every request kwargs,
deep-copied at call time, and answers from a provider-neutral script. Nothing
here touches the network.

A script is a list of :class:`Step` (or callables ``ScriptedModel -> Step``) for
the Phase 2 loop. Planning calls are recognised by the tool they offer
(``choose_generator``, ``submit_tasks``) and answered from fixed payloads.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import anthropic.types as at

from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient, OpenAIProvider


class TransportCrash(RuntimeError):
    """The scripted transport failing a request (a dropped connection)."""


CRASH = "crash"


@dataclass
class Usage:
    input: int
    output: int
    cache_write: int = 0
    cache_read: int = 0


@dataclass
class Step:
    """One model reply, provider-neutral."""

    calls: list[tuple[str, dict]] = field(default_factory=list)
    text: str = ""
    thinking: str = ""
    usage: Usage | None = None

    @property
    def stop_reason(self) -> str:
        return "tool_use" if self.calls else "end_turn"


ROUTING = {"target_kind": "backend", "target_description": "REST API",
           "has_user_interface": False, "unsupported_stack": "",
           "generator": "generate_fastapi_backend"}


@dataclass
class Request:
    kind: str          # "routing" | "gap" | "phase2" | "other"
    kwargs: dict


class ScriptedModel:
    """Answers each request from the script; records nothing itself."""

    def __init__(self, workspace: str, phase2: list, tasks: list[str],
                 usage: Callable[[int], Usage | None] | None = None):
        self.workspace = workspace
        self.phase2 = list(phase2)
        self.tasks = tasks
        self.usage = usage or (lambda n: Usage(1_000 + n, 100 + n, 50, 400 + n))
        self.served = 0
        self.orchestrator = None  # set by the test, for steps that look up task ids
        self.scripted_usage: list[Usage] = []

    def classify(self, tool_names: list[str]) -> str:
        if "choose_generator" in tool_names:
            return "routing"
        if "submit_tasks" in tool_names:
            return "gap"
        if "task_list" in tool_names:
            return "phase2"
        return "other"

    def reply(self, kind: str) -> Step | str:
        if kind == "routing":
            step = Step(calls=[("choose_generator", dict(ROUTING))])
        elif kind == "gap":
            step = Step(calls=[("submit_tasks", {"tasks": [
                {"task": task, "kind": "change"} for task in self.tasks]})])
        elif kind == "phase2":
            assert self.phase2, "the run asked for more Phase 2 turns than scripted"
            entry = self.phase2.pop(0)
            if entry == CRASH:
                return CRASH
            step = entry(self) if callable(entry) else entry
        else:
            step = Step(text="[]")
        if step.usage is None:
            step.usage = self.usage(self.served)
        self.served += 1
        if step.usage is not False:
            self.scripted_usage.append(step.usage)
        return step


def _call_id(prefix: str, turn: int, index: int) -> str:
    return f"{prefix}_{turn:02d}_{index}"


class _Recorder:
    def __init__(self, model: ScriptedModel):
        self.model = model
        self.requests: list[Request] = []
        # Optional ``() -> float``: the client's spend when each request is sent.
        self.cost_probe: Callable[[], float] | None = None
        self.costs_at_send: list[float | None] = []

    def _record(self, kwargs: dict, tool_names: list[str]) -> str:
        kind = self.model.classify(tool_names)
        self.requests.append(Request(kind, copy.deepcopy(kwargs)))
        self.costs_at_send.append(self.cost_probe() if self.cost_probe else None)
        return kind

    def of(self, kind: str) -> list[dict]:
        return [r.kwargs for r in self.requests if r.kind == kind]


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------

class FakeAnthropic(_Recorder):
    """Stands in for ``anthropic.Anthropic`` behind ``ClaudeLLMClient``."""

    def __init__(self, model: ScriptedModel, with_thinking: bool):
        super().__init__(model)
        self.with_thinking = with_thinking
        self.issued_thinking: dict[str, str] = {}  # signature -> thinking text
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    def _response(self, kwargs: dict):
        kind = self._record(kwargs, [t["name"] for t in kwargs.get("tools") or []])
        step = self.model.reply(kind)
        if step == CRASH:
            raise TransportCrash("connection reset by peer")
        n = len(self.requests)
        content: list[Any] = []
        if step.thinking and self.with_thinking:
            # The signature names the request this block answered (1-based).
            content.append(at.ThinkingBlock(type="thinking", thinking=step.thinking,
                                            signature=f"sig-{n}"))
            self.issued_thinking[f"sig-{n}"] = step.thinking
        if step.text:
            content.append(at.TextBlock(type="text", text=step.text))
        for index, (name, payload) in enumerate(step.calls):
            content.append(at.ToolUseBlock(type="tool_use", id=_call_id("toolu", n, index),
                                           name=name, input=copy.deepcopy(payload)))
        usage = None
        if step.usage:
            usage = at.Usage(input_tokens=step.usage.input, output_tokens=step.usage.output,
                             cache_creation_input_tokens=step.usage.cache_write,
                             cache_read_input_tokens=step.usage.cache_read)
        return SimpleNamespace(stop_reason=step.stop_reason, content=content,
                               usage=usage, stop_details=None)

    def _create(self, **kwargs):
        return self._response(kwargs)

    def _stream(self, **kwargs):
        response = self._response(kwargs)

        class _Stream:
            text_stream = [b.text for b in response.content if b.type == "text"]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return response

        return _Stream()


def claude_run_client(model_id: str, scripted: ScriptedModel, *, with_thinking: bool):
    client = ClaudeLLMClient(api_key="sk-ant-test", model=model_id, max_retries=0)
    fake = FakeAnthropic(scripted, with_thinking)
    client._client = fake
    return client, fake


# ----------------------------------------------------------------------
# OpenAI chat/completions
# ----------------------------------------------------------------------

class FakeOpenAI(_Recorder):
    """Stands in for ``openai.OpenAI`` behind ``OpenAIProvider``."""

    def __init__(self, model: ScriptedModel, model_id: str):
        super().__init__(model)
        self.model_id = model_id
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _usage(self, usage: Usage | None):
        if not usage:
            return None
        return SimpleNamespace(prompt_tokens=usage.input + usage.cache_read,
                               completion_tokens=usage.output,
                               prompt_tokens_details=SimpleNamespace(cached_tokens=usage.cache_read))

    def _create(self, **kwargs):
        names = [t["function"]["name"] for t in kwargs.get("tools") or []]
        kind = self._record(kwargs, names)
        step = self.model.reply(kind)
        if step == CRASH:
            raise TransportCrash("connection reset by peer")
        n = len(self.requests)
        calls = [SimpleNamespace(id=_call_id("call", n, i), type="function",
                                 function=SimpleNamespace(name=name, arguments=json.dumps(payload)))
                 for i, (name, payload) in enumerate(step.calls)]
        finish = "tool_calls" if calls else "stop"
        if not kwargs.get("stream"):
            message = SimpleNamespace(content=step.text or None, tool_calls=calls or None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                                   usage=self._usage(step.usage), model=self.model_id)
        chunks = []
        if step.text:
            chunks.append(_chunk(self.model_id, SimpleNamespace(content=step.text, tool_calls=None)))
        for index, call in enumerate(calls):
            delta = SimpleNamespace(index=index, id=call.id, function=SimpleNamespace(
                name=call.function.name, arguments=call.function.arguments))
            chunks.append(_chunk(self.model_id, SimpleNamespace(content=None, tool_calls=[delta])))
        chunks.append(_chunk(self.model_id, SimpleNamespace(content=None, tool_calls=None), finish))
        chunks.append(SimpleNamespace(model=self.model_id, usage=self._usage(step.usage), choices=[]))
        return iter(chunks)


def _chunk(model_id: str, delta, finish=None):
    return SimpleNamespace(model=model_id, usage=None,
                           choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def openai_run_client(model_id: str, scripted: ScriptedModel):
    client = OpenAIProvider(api_key="sk-test", model=model_id, max_retries=0)
    fake = FakeOpenAI(scripted, model_id)
    client._client = fake
    return client, fake


# ----------------------------------------------------------------------
# Reading the recorded requests
# ----------------------------------------------------------------------

def plain(value: Any) -> Any:
    """A request value as plain JSON-like data (SDK objects dumped)."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return plain(value.model_dump(exclude_none=True))
    if hasattr(value, "__slots__") and not isinstance(value, (str, bytes)):
        return {slot: plain(getattr(value, slot)) for slot in value.__slots__}
    if isinstance(value, SimpleNamespace):
        return plain(vars(value))
    return value


def without_cache_control(value: Any) -> Any:
    """``value`` with every ``cache_control`` key removed (not part of the prefix)."""
    if isinstance(value, dict):
        return {k: without_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [without_cache_control(v) for v in value]
    return value


def blocks(message: dict) -> list[dict]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


def count_cache_markers(value: Any) -> int:
    if isinstance(value, dict):
        return int("cache_control" in value) + sum(count_cache_markers(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_cache_markers(v) for v in value)
    return 0
