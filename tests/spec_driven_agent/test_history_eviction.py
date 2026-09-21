"""Tests for agent-loop history eviction (stale file-body stubbing).

The Phase-2 loop re-sends the whole growing message array each turn; the biggest
growth term is stale file bodies (write_file inputs + read_file results). The
eviction stubs those in OLDER messages while (a) preserving tool_use/tool_result
pairing, (b) never touching the most recent messages, and (c) never mutating the
caller's blocks. These tests pin all three invariants across dict- and
object-shaped blocks.
"""

import copy
import json

import pytest

from besser.spec_driven_agent import orchestrator as orch_mod
from besser.spec_driven_agent.compaction import COMPACT_TOKEN_THRESHOLD
from besser.spec_driven_agent.history_eviction import (
    evict_stale_file_bodies,
    DEFAULT_PRESERVE_RECENT,
    without_rejected_edit_drafts,
)
from besser.spec_driven_agent.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.llm_client import UsageTracker
from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)


class _Obj:
    """Minimal stand-in for an SDK content-block object (attribute access)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _write_use(bid, path, body, *, as_object=False):
    if as_object:
        return _Obj(type="tool_use", id=bid, name="write_file",
                    input={"path": path, "content": body})
    return {"type": "tool_use", "id": bid, "name": "write_file",
            "input": {"path": path, "content": body}}


def _read_use(bid, path):
    return {"type": "tool_use", "id": bid, "name": "read_file",
            "input": {"path": path}}


def _result(bid, content, is_error=None):
    b = {"type": "tool_result", "tool_use_id": bid, "content": content}
    if is_error is not None:
        b["is_error"] = is_error
    return b


def _big(n=2000):
    return "x = 1\n" * n


def _padding(n):
    """n filler message pairs so the interesting ones fall OUTSIDE preserve_recent."""
    out = []
    for i in range(n):
        out.append({"role": "assistant", "content": [{"type": "text", "text": f"step {i}"}]})
        out.append({"role": "user", "content": [{"type": "text", "text": "ok"}]})
    return out


def test_short_history_is_untouched():
    msgs = [{"role": "user", "content": "hi"}] * (DEFAULT_PRESERVE_RECENT - 1)
    out, n = evict_stale_file_bodies(msgs)
    assert n == 0
    assert out is msgs


def test_recovery_omits_only_rejected_drafts_without_changing_history_or_results():
    messages = [
        {"role": "user", "content": "Keep the complete user specification."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "bad", "name": "modify_file",
             "input": {"path": "app.py", "old_text": "WRONG ANCHOR", "new_text": "BAD DRAFT"}},
            _write_use("good", "other.py", "CORRECT SOURCE")]},
        {"role": "user", "content": [
            _result("bad", json.dumps({"error": "syntax error", "would_write": "BAD DRAFT",
                "current_source": "ACTUAL SOURCE", "edit_recovery": {"next_tool": "read_file"}})),
            _result("good", '{"status":"written"}')]}]
    before = copy.deepcopy(messages)
    projected = without_rejected_edit_drafts(messages)
    rendered = json.dumps(projected)
    assert "BAD DRAFT" not in rendered and "WRONG ANCHOR" not in rendered
    assert "ACTUAL SOURCE" in rendered and "CORRECT SOURCE" in rendered
    assert "syntax error" in rendered and "NOT applied" in rendered
    assert projected[1]["content"][0]["id"] == "bad"
    assert projected[2]["content"][0]["tool_use_id"] == "bad"
    assert projected[0] == messages[0] and messages == before


@pytest.mark.parametrize("tool,key", [("write_file", "content"), ("modify_file", "new_text"),
                                     ("replace_file_lines", "new_text")])
def test_write_file_body_is_stubbed_in_old_messages(tool, key):
    body = _big()
    use = _write_use("t1", "app/main.py", body)
    use["name"] = tool
    use["input"][key] = use["input"].pop("content")
    msgs = [
        {"role": "assistant", "content": [use]},
        {"role": "user", "content": [_result("t1", '{"status":"written"}')]},
    ] + _padding(DEFAULT_PRESERVE_RECENT)  # push the write outside the window

    out, n = evict_stale_file_bodies(msgs, min_body_chars=100)

    assert n == 1
    new_use = out[0]["content"][0]
    assert new_use["type"] == "tool_use"
    assert new_use["id"] == "t1"          # id preserved -> pairing intact
    assert new_use["input"]["path"] == "app/main.py"  # path preserved
    assert "elided" in new_use["input"][key]
    assert "does not mean it was applied" in new_use["input"][key]
    assert body not in json.dumps(out)    # the body is gone from the context


def test_read_file_result_is_stubbed():
    body = _big()
    msgs = [
        {"role": "assistant", "content": [_read_use("r1", "app/models.py")]},
        {"role": "user", "content": [_result("r1", body)]},
    ] + _padding(DEFAULT_PRESERVE_RECENT)

    out, n = evict_stale_file_bodies(msgs, min_body_chars=100)

    assert n == 1
    res = out[1]["content"][0]
    assert res["type"] == "tool_result"
    assert res["tool_use_id"] == "r1"
    assert "elided" in res["content"]
    assert body not in json.dumps(out)


def test_recent_messages_are_preserved_verbatim():
    body = _big()
    # The write is INSIDE the preserve window -> must NOT be stubbed.
    msgs = _padding(2) + [
        {"role": "assistant", "content": [_write_use("t9", "app/x.py", body)]},
        {"role": "user", "content": [_result("t9", '{"status":"written"}')]},
    ]
    out, n = evict_stale_file_bodies(msgs, preserve_recent=DEFAULT_PRESERVE_RECENT)
    assert n == 0
    assert out is msgs


def test_non_file_tools_are_not_touched():
    payload = json.dumps({"findings": ["x"] * 500})
    msgs = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "v1", "name": "validate_app", "input": {}}]},
        {"role": "user", "content": [_result("v1", payload)]},
    ] + _padding(DEFAULT_PRESERVE_RECENT)
    out, n = evict_stale_file_bodies(msgs, min_body_chars=100)
    assert n == 0  # a validate result is not a file body — keep it


def test_small_bodies_are_left_alone():
    msgs = [
        {"role": "assistant", "content": [_write_use("t1", "a.py", "print(1)")]},
        {"role": "user", "content": [_result("t1", "ok")]},
    ] + _padding(DEFAULT_PRESERVE_RECENT)
    out, n = evict_stale_file_bodies(msgs, min_body_chars=600)
    assert n == 0


def test_object_shaped_tool_use_blocks_are_handled():
    body = _big()
    msgs = [
        {"role": "assistant", "content": [_write_use("t1", "app/main.py", body, as_object=True)]},
        {"role": "user", "content": [_result("t1", '{"status":"written"}')]},
    ] + _padding(DEFAULT_PRESERVE_RECENT)

    out, n = evict_stale_file_bodies(msgs, min_body_chars=100)
    assert n == 1
    new_use = out[0]["content"][0]
    assert isinstance(new_use, dict)          # object -> dict (both clients accept)
    assert new_use["id"] == "t1" and new_use["name"] == "write_file"
    assert "elided" in new_use["input"]["content"]


def test_purity_input_is_not_mutated():
    body = _big()
    obj_block = _write_use("t1", "app/main.py", body, as_object=True)
    dict_result = _result("r1", body)
    msgs = [
        {"role": "assistant", "content": [obj_block]},
        {"role": "assistant", "content": [_read_use("r1", "app/models.py")]},
        {"role": "user", "content": [dict_result]},
    ] + _padding(DEFAULT_PRESERVE_RECENT)
    before = copy.deepcopy([
        {"path": obj_block.input["path"], "content": obj_block.input["content"]},
        dict(dict_result),
    ])

    evict_stale_file_bodies(msgs, min_body_chars=100)

    # Original object + dict blocks are unchanged.
    assert obj_block.input["content"] == before[0]["content"]
    assert dict_result["content"] == before[1]["content"]


class _MockClient:
    model = "mock"

    def __init__(self):
        self.usage = UsageTracker("mock")

    def chat(self, **kw):
        return {"stop_reason": "end_turn", "content": []}


def _simple_model():
    cls = Class(name="Item")
    cls.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    return DomainModel(name="Test", types={cls})


def test_checkpoint_eviction_fires_only_when_enabled(tmp_path, monkeypatch):
    """_maybe_compact runs eviction at the checkpoint when the flag is ON."""
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
    )
    body = _big(4000)  # a large write_file body
    # A history that exceeds the compaction threshold, with the big write early
    # (outside the preserve window) so it's eligible for eviction.
    messages = [
        {"role": "assistant", "content": [_write_use("t1", "app/main.py", body)]},
        {"role": "user", "content": [_result("t1", '{"status":"written"}')]},
    ]
    filler = "y" * (COMPACT_TOKEN_THRESHOLD * 4)
    messages.append({"role": "user", "content": [_result("t2", filler)]})
    messages += _padding(DEFAULT_PRESERVE_RECENT)

    # Flag OFF (default): eviction does not run.
    monkeypatch.setattr(orch_mod, "_HISTORY_EVICTION_ENABLED", False)
    out_off = orch._maybe_compact([dict(m) for m in messages])
    assert getattr(orch, "_eviction_count", 0) == 0

    # Flag ON: eviction stubs the stale write body at the checkpoint.
    monkeypatch.setattr(orch_mod, "_HISTORY_EVICTION_ENABLED", True)
    orch._maybe_compact([dict(m) for m in messages])
    assert getattr(orch, "_eviction_count", 0) == 1


def test_pairing_count_is_preserved():
    body = _big()
    msgs = []
    for i in range(4):
        msgs.append({"role": "assistant", "content": [_write_use(f"t{i}", f"f{i}.py", body)]})
        msgs.append({"role": "user", "content": [_result(f"t{i}", '{"status":"written"}')]})
    msgs += _padding(DEFAULT_PRESERVE_RECENT)

    out, n = evict_stale_file_bodies(msgs, min_body_chars=100)

    # Same number of messages and blocks, just smaller content.
    assert len(out) == len(msgs)
    def block_count(ms):
        return sum(len(m["content"]) for m in ms if isinstance(m.get("content"), list))
    assert block_count(out) == block_count(msgs)
    assert n >= 1
