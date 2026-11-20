"""``chat_stream(model_override=...)`` calls and bills that model, as ``chat`` does.

The lab's plan step runs on its own model (``BESSER_LAB_PLAN_LLM_MODEL``) through
the run's one client, so the run's one ``UsageTracker`` must price those streamed
calls at the plan model's rate, or the cost cap counts them at the build model's.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic.types as at
import pytest

from besser.spec_driven_agent.providers.llm_client import (
    ClaudeLLMClient,
    OpenAIProvider,
    UsageTracker,
    _OpenAIUsageAdapter,
)
from besser.spec_driven_agent.providers.model_settings import max_output_tokens

MESSAGES = [{"role": "user", "content": "go"}]
#: Above every model's own output ceiling, so an override visibly lowers it.
RAISED_MAX_TOKENS = 1_000_000


def _claude(usage):
    client = ClaudeLLMClient(api_key="sk-ant-test", model="claude-sonnet-5-5", max_tokens=RAISED_MAX_TOKENS)
    client._client = MagicMock()
    stream = client._client.messages.stream.return_value.__enter__.return_value
    stream.text_stream = iter(("done",))
    stream.get_final_message.return_value = SimpleNamespace(
        stop_reason="end_turn", content=[at.TextBlock(type="text", text="done")], usage=usage)
    return client


def _openai(usage):
    with patch.dict("sys.modules", {"openai": MagicMock()}):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o", max_tokens=RAISED_MAX_TOKENS)
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = iter([SimpleNamespace(
        model="served-name", usage=usage,
        choices=[SimpleNamespace(delta=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")])])
    return provider


CLAUDE_USAGE = at.Usage(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=300,
                        cache_read_input_tokens=4000)
OPENAI_USAGE = SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                               prompt_tokens_details=SimpleNamespace(cached_tokens=400))

CASES = [
    pytest.param(_claude, CLAUDE_USAGE, "claude-opus-5-5", lambda u: u,
                 lambda c: c._client.messages.stream.call_args.kwargs, id="claude"),
    pytest.param(_openai, OPENAI_USAGE, "gpt-4o-mini", _OpenAIUsageAdapter,
                 lambda c: c._client.chat.completions.create.call_args.kwargs, id="openai"),
]


def _priced_alone(model: str, client, usage) -> float:
    """What ``usage`` costs on a tracker of ``model`` billed like ``client``'s."""
    tracker = UsageTracker(model, billed=client.usage.billed, route=client.usage.route)
    tracker.record(usage)
    return tracker.estimated_cost


@pytest.mark.parametrize("make, usage, override, adapt, request_of", CASES)
def test_an_override_is_the_model_called_and_billed(make, usage, override, adapt, request_of):
    client = make(usage)
    events = list(client.chat_stream(system="s", messages=MESSAGES, tools=[], model_override=override))

    request = request_of(client)
    assert request["model"] == override
    sent_ceiling = request.get("max_tokens") or request.get("max_completion_tokens")
    assert sent_ceiling == min(RAISED_MAX_TOKENS, max_output_tokens(override))
    assert client.usage.estimated_cost == pytest.approx(_priced_alone(override, client, adapt(usage)))
    assert client.usage.model == client.model  # the tracker stays on the default model
    assert events[-1]["type"] == "message_done"


@pytest.mark.parametrize("make, usage, override, adapt, request_of", CASES)
def test_without_an_override_nothing_changes(make, usage, override, adapt, request_of):
    client = make(usage)
    list(client.chat_stream(system="s", messages=MESSAGES, tools=[]))

    request = request_of(client)
    assert request["model"] == client.model
    assert (request.get("max_tokens") or request.get("max_completion_tokens")) == RAISED_MAX_TOKENS
    assert client.usage.estimated_cost == pytest.approx(_priced_alone(client.model, client, adapt(usage)))


def test_an_openai_override_does_not_set_the_served_model():
    # As in chat(): the served name of a call on another model is not the run's model.
    provider = _openai(OPENAI_USAGE)
    list(provider.chat_stream(system="s", messages=MESSAGES, tools=[], model_override="gpt-4o-mini"))
    assert provider.usage.served_model is None

    provider = _openai(OPENAI_USAGE)
    list(provider.chat_stream(system="s", messages=MESSAGES, tools=[]))
    assert provider.usage.served_model == "served-name"
