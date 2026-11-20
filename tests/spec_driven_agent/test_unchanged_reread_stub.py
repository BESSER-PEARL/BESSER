"""An identical re-read of an unchanged file is not paid for again.

A recorded run re-read one unchanged 3,167-token file 16 times during an edit
loop (~50k tokens in context at once). The same path, offset and limit on the
same content, while the earlier result is still in the conversation, now gets
a short stub; asking again right after the stub returns the text.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import (
    COMPACT_TOKEN_THRESHOLD,
    LLMOrchestrator,
    ValidationIssue,
)

SOURCE = "".join(f"line_{i} = {i}\n" for i in range(1, 41))


@pytest.fixture
def executor(tmp_path):
    (tmp_path / "app.py").write_text(SOURCE, encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


def _read(executor, **args):
    return executor.execute_typed("read_file", {"path": "app.py", **args}).payload


def test_an_identical_unchanged_re_read_gets_a_stub(executor):
    first = _read(executor)
    again = _read(executor)

    assert "content" not in again and again["unchanged"] is True
    assert again["read_id"] == first["read_id"]
    assert (again["start_line"], again["end_line"]) == (1, 41)
    assert f"read_id {first['read_id']}" in again["note"]
    assert len(json.dumps(again)) < len(json.dumps(first)) / 2


def test_the_stubbed_read_id_still_drives_a_range_edit(executor):
    first = _read(executor)
    stub = _read(executor)

    result = executor.execute_typed("replace_file_lines", {
        "path": "app.py", "read_id": stub["read_id"], "start_line": 2, "end_line": 2,
        "new_text": "line_2 = 22",
    }).payload

    assert result["status"] == "modified", (first["read_id"], result)


def test_a_different_range_is_read_in_full(executor):
    _read(executor)
    assert "content" in _read(executor, offset=10, limit=5)
    assert "content" in _read(executor, limit=5)


def test_asking_again_after_a_stub_returns_the_text(executor):
    _read(executor)
    assert "content" not in _read(executor)

    insisted = _read(executor)

    assert insisted["content"].startswith("1| line_1 = 1")
    # And the cycle starts over: the next identical read is a stub again.
    assert "content" not in _read(executor)


def test_an_edit_to_the_file_means_a_full_read(executor):
    _read(executor)
    executor.execute_typed("modify_file", {"path": "app.py", "old_text": "line_3 = 3", "new_text": "line_3 = 33"})

    assert "line_3 = 33" in _read(executor)["content"]


def test_a_change_on_disk_outside_the_tools_means_a_full_read(executor):
    _read(executor)
    with open(os.path.join(executor.workspace, "app.py"), "a", encoding="utf-8") as handle:
        handle.write("extra = 1\n")

    assert "extra = 1" in _read(executor)["content"]


def test_after_a_reset_the_read_is_full(executor):
    _read(executor)
    executor.forget_shown_results()

    assert "content" in _read(executor)


def test_a_stub_is_not_a_novel_inspection(executor):
    """The orchestrator credits only new read output as progress."""
    full = executor.execute("read_file", {"path": "app.py"})
    stub = executor.execute("read_file", {"path": "app.py"})
    block = SimpleNamespace(id="r1", name="read_file", input={"path": "app.py"})
    seen: set = set()

    assert LLMOrchestrator._record_novel_inspections([block], [{"tool_use_id": "r1", "content": full}], seen)
    assert not LLMOrchestrator._record_novel_inspections([block], [{"tool_use_id": "r1", "content": stub}], seen)
    assert not LLMOrchestrator._record_novel_inspections(
        [block], [{"tool_use_id": "r1", "content": stub}], set())


# --------------------------------------------------- the orchestrator resets


class _Usage:
    estimated_cost = 0.0


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ReadingClient:
    """Reads app.py on its first turn, then stops; keeps what it was sent."""
    model = "mock-model"

    def __init__(self):
        self.usage = _Usage()
        self.sent = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.sent.append(list(messages))
        if len(self.sent) == 1:
            return {"stop_reason": "tool_use", "content": [
                _Block("tool_use", id="r1", name="read_file", input={"path": "app.py"})]}
        return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


class _StateMachine:
    name = "DummySM"


def _orchestrator(tmp_path, client):
    (tmp_path / "app.py").write_text(SOURCE, encoding="utf-8")
    return LLMOrchestrator(llm_client=client, state_machines=[_StateMachine()], output_dir=str(tmp_path),
                           max_cost_usd=10.0, enable_tracing=False, enable_checkpointing=False)


def test_a_phase3_fix_round_starts_without_stubs(tmp_path):
    client = _ReadingClient()
    orch = _orchestrator(tmp_path, client)
    orch.executor.execute("read_file", {"path": "app.py"})  # read in an earlier conversation

    orch._invoke_phase3_fix_loop([ValidationIssue("blocker", "syntax: app.py line 1")], is_first_attempt=True)

    result = next(block["content"] for message in client.sent[1] if isinstance(message["content"], list)
                  for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_result")
    assert '"content"' in result and "unchanged" not in result


def test_compaction_forgets_what_was_shown(tmp_path):
    orch = _orchestrator(tmp_path, _ReadingClient())
    orch.executor.execute("read_file", {"path": "app.py"})
    big = "x" * (COMPACT_TOKEN_THRESHOLD * 5)
    messages = []
    for i in range(10):
        messages += [{"role": "user", "content": f"turn {i}: {big[:20000]}"},
                     {"role": "assistant", "content": [{"type": "text", "text": f"ok {i}"}]}]
    messages.append({"role": "user", "content": big})

    orch._maybe_compact(messages)

    assert orch._compaction_count == 1
    assert "content" in orch.executor.execute_typed("read_file", {"path": "app.py"}).payload


def test_a_frozen_path_is_read_in_full(executor):
    """Its read_id cannot drive an edit, so a stub would point at nothing."""
    _read(executor)
    executor.freeze_path("app.py", "closed by the orchestrator")

    assert "content" in _read(executor)
