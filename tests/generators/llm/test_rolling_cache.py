"""Tests for the rolling prompt-cache breakpoint on the conversation tail.

Only the system prompt + last tool carried a cache breakpoint, so the growing
message history was re-billed as fresh input every turn. `_with_message_cache`
adds a breakpoint on the last message so the prior prefix is a cache read. These
tests pin the payload SHAPE (the deterministic part); actual cache hits are
provider behaviour verified live.
"""

import copy

import pytest

from besser.generators.llm import llm_client
from besser.generators.llm.llm_client import _with_message_cache


@pytest.fixture
def rolling_on(monkeypatch):
    monkeypatch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)


def test_disabled_by_default_is_noop():
    # The module default is OFF (must be explicitly enabled after a paid check).
    msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}]
    assert _with_message_cache(msgs) is msgs


def test_marks_last_block_of_last_message(rolling_on):
    msgs = [
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "result body"},
        ]},
    ]
    out = _with_message_cache(msgs)
    last_block = out[-1]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}
    # Earlier messages untouched (same object).
    assert out[0] is msgs[0]


def test_wraps_string_content(rolling_on):
    msgs = [{"role": "user", "content": "just build it"}]
    out = _with_message_cache(msgs)
    block = out[-1]["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "just build it"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_object_last_block_is_left_alone(rolling_on):
    class _Obj:
        type = "tool_use"
        id = "t1"

    msgs = [{"role": "assistant", "content": [_Obj()]}]
    # Can't safely annotate an SDK object -> no-op rather than risk corruption.
    assert _with_message_cache(msgs) is msgs


def test_is_pure_does_not_mutate_input(rolling_on):
    msgs = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "body"}]},
    ]
    before = copy.deepcopy(msgs)
    _with_message_cache(msgs)
    assert msgs == before  # original untouched


def test_at_most_one_message_breakpoint(rolling_on):
    # Only the LAST message gets a breakpoint (system + last tool + tail = 3,
    # under Anthropic's 4-breakpoint cap).
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "c"}]},
    ]
    out = _with_message_cache(msgs)
    marked = sum(
        1
        for m in out
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and "cache_control" in b
    )
    assert marked == 1
