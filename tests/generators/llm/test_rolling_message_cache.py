"""Re-sent history must be a cache read, not fresh input.

Only the system prompt and the last tool definition carried a cache
breakpoint, so the message array -- the part that grows every turn -- was
re-billed in full on every call. Measured on run claude-sonnet-5-q0yzuo43:
4,097,784 uncached input tokens, a 25.2% hit rate, about half the run's cost.

A paid A/B on claude-sonnet-5 (four growing turns, flag off then on) settled
the mechanism: cumulative ``input_tokens`` went 890/2,602/5,136/8,492 off
against 2/4/6/8 on, and ``cache_read`` climbed 4,845 -> 24,510. These tests
pin the request shape that produces it, offline.
"""
import pytest

from besser.generators.llm.llm_client import _with_message_cache


def _cache_controls(messages):
    """Every block in *messages* carrying a cache breakpoint."""
    found = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            found += [b for b in content
                      if isinstance(b, dict) and b.get("cache_control")]
    return found


def test_the_breakpoint_lands_on_the_last_message(monkeypatch):
    monkeypatch.setattr(
        "besser.generators.llm.llm_client._ROLLING_MESSAGE_CACHE", True)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]

    marked = _with_message_cache(messages)

    assert len(_cache_controls(marked)) == 1, (
        "exactly one message breakpoint -- system and last tool hold the "
        "other two, and the API caps a request at four"
    )
    assert marked[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_the_prior_prefix_is_left_byte_identical(monkeypatch):
    """A breakpoint that rewrote earlier turns would invalidate the prefix it
    exists to reuse -- caching is a prefix match."""
    monkeypatch.setattr(
        "besser.generators.llm.llm_client._ROLLING_MESSAGE_CACHE", True)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": "second"},
    ]
    before = [dict(m) for m in messages[:-1]]

    marked = _with_message_cache(messages)

    assert marked[:-1] == before
    assert messages[-1]["content"] == "second", "input was mutated in place"


def test_tool_result_turns_are_annotated(monkeypatch):
    """The last message before a call is the user tool_result, which is the
    turn whose prefix we most want cached."""
    monkeypatch.setattr(
        "besser.generators.llm.llm_client._ROLLING_MESSAGE_CACHE", True)
    messages = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "done"},
    ]}]

    marked = _with_message_cache(messages)

    assert marked[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert marked[0]["content"][-1]["type"] == "tool_result"


def test_disabling_it_restores_the_previous_request_shape(monkeypatch):
    monkeypatch.setattr(
        "besser.generators.llm.llm_client._ROLLING_MESSAGE_CACHE", False)
    messages = [{"role": "user", "content": "first"}]

    assert _with_message_cache(messages) is messages


@pytest.mark.parametrize("messages", [[], [object()]])
def test_unannotatable_input_is_returned_untouched(monkeypatch, messages):
    """An SDK object on an assistant tool_use turn cannot be safely annotated;
    returning it unchanged must never raise."""
    monkeypatch.setattr(
        "besser.generators.llm.llm_client._ROLLING_MESSAGE_CACHE", True)

    assert _with_message_cache(messages) is messages


def test_it_is_on_by_default():
    """The default is the whole point: a saving nobody enables is no saving."""
    import importlib
    import besser.generators.llm.llm_client as client

    importlib.reload(client)
    assert client._ROLLING_MESSAGE_CACHE is True
