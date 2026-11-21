"""Tool blocks that write the SAME path must not run concurrently.

``_execute_tool_blocks`` fans a turn's tool calls out over a thread pool. Every
write tool does read -> transform -> write, so two ``modify_file`` calls on one
file in a single turn each read the pre-turn content and the second write
silently discards the first edit. The tool description used to actively invite
this ("different sections ... in the SAME turn ... in parallel").

Independent paths must still run in parallel, and the returned results must stay
in the model's block order so tool_use/tool_result pairing holds.
"""

import json
import os
import time

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.scripts.trace_metrics import load_runs


@pytest.fixture
def simple_model():
    string_type = PrimitiveDataType("str")
    int_type = PrimitiveDataType("int")
    user = Class(name="User")
    user.attributes = {
        Property(name="id", type=int_type, is_id=True),
        Property(name="name", type=string_type),
    }
    return DomainModel(name="TestModel", types={user})


class _MockClient:
    model = "mock"

    def __init__(self):
        self.usage = UsageTracker("mock")

    def chat(self, **kwargs):
        return {"stop_reason": "end_turn", "content": []}


class _Block:
    """Stand-in for an SDK tool_use block."""

    type = "tool_use"

    def __init__(self, bid, name, tool_input):
        self.id = bid
        self.name = name
        self.input = tool_input


def _orch(simple_model, tmp_path):
    return LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=simple_model,
        output_dir=str(tmp_path),
    )


def _seed(tmp_path, rel, content):
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full) or str(tmp_path), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def _read(tmp_path, rel):
    with open(os.path.join(str(tmp_path), rel), encoding="utf-8") as f:
        return f.read()


def test_two_edits_to_one_file_in_one_turn_both_land(simple_model, tmp_path):
    """The regression: without serialization the second write drops the first."""
    _seed(tmp_path, "app.py", "alpha = 1\nbeta = 2\n")
    orch = _orch(simple_model, tmp_path)

    blocks = [
        _Block("t1", "modify_file", {
            "path": "app.py", "old_text": "alpha = 1", "new_text": "alpha = 111",
        }),
        _Block("t2", "modify_file", {
            "path": "app.py", "old_text": "beta = 2", "new_text": "beta = 222",
        }),
    ]
    results = orch._execute_tool_blocks(blocks, turn=0)

    body = _read(tmp_path, "app.py")
    assert "alpha = 111" in body, body
    assert "beta = 222" in body, body

    # Both calls reported success, in the model's block order.
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    for r in results:
        assert json.loads(r["content"]).get("status") == "modified"


def test_many_edits_to_one_file_compose_in_order(simple_model, tmp_path):
    _seed(tmp_path, "app.py", "a\nb\nc\nd\n")
    orch = _orch(simple_model, tmp_path)
    blocks = [
        _Block("t%d" % i, "modify_file", {
            "path": "app.py", "old_text": ch, "new_text": ch.upper(),
        })
        for i, ch in enumerate("abcd")
    ]
    orch._execute_tool_blocks(blocks, turn=0)
    assert _read(tmp_path, "app.py") == "A\nB\nC\nD\n"


def test_serial_key_shares_a_group_per_write_path(simple_model, tmp_path):
    orch = _orch(simple_model, tmp_path)
    a1 = _Block("1", "modify_file", {"path": "app.py", "old_text": "x", "new_text": "y"})
    a2 = _Block("2", "write_file", {"path": "app.py", "content": "z"})
    a3 = _Block("3", "modify_file", {"path": "./app.py", "old_text": "x", "new_text": "y"})
    b1 = _Block("4", "modify_file", {"path": "other.py", "old_text": "x", "new_text": "y"})
    assert orch._serial_key(a1) == orch._serial_key(a2)      # same path, any write tool
    assert orch._serial_key(a1) != orch._serial_key(b1)      # different files
    assert orch._serial_key(a1) == orch._serial_key(a3)      # lexical aliases normalize
    assert orch._serial_key(a1) != orch._serial_key(
        _Block("5", "read_file", {"path": "app.py"})
    )                                                        # reads never serialize


def test_reads_and_other_tools_get_unique_keys(simple_model, tmp_path):
    orch = _orch(simple_model, tmp_path)
    r1 = _Block("1", "read_file", {"path": "app.py"})
    r2 = _Block("2", "read_file", {"path": "app.py"})
    assert orch._serial_key(r1) != orch._serial_key(r2)


def test_writes_to_different_paths_stay_in_separate_groups(simple_model, tmp_path):
    _seed(tmp_path, "one.py", "x = 1\n")
    _seed(tmp_path, "two.py", "y = 1\n")
    orch = _orch(simple_model, tmp_path)
    blocks = [
        _Block("t1", "modify_file", {
            "path": "one.py", "old_text": "x = 1", "new_text": "x = 2",
        }),
        _Block("t2", "modify_file", {
            "path": "two.py", "old_text": "y = 1", "new_text": "y = 2",
        }),
    ]
    keys = {orch._serial_key(b) for b in blocks}
    assert len(keys) == 2   # parallelism preserved for independent files

    results = orch._execute_tool_blocks(blocks, turn=0)
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert "x = 2" in _read(tmp_path, "one.py")
    assert "y = 2" in _read(tmp_path, "two.py")


def test_a_failing_call_does_not_lose_its_sibling_result(simple_model, tmp_path):
    """One bad edit in a same-path group must still yield a result per block."""
    _seed(tmp_path, "app.py", "alpha = 1\n")
    orch = _orch(simple_model, tmp_path)
    blocks = [
        _Block("t1", "modify_file", {
            "path": "app.py", "old_text": "alpha = 1", "new_text": "alpha = 111",
        }),
        _Block("t2", "modify_file", {
            "path": "app.py", "old_text": "nope_not_there", "new_text": "q",
        }),
    ]
    results = orch._execute_tool_blocks(blocks, turn=0)
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert json.loads(results[0]["content"]).get("status") == "modified"
    assert "error" in json.loads(results[1]["content"])
    assert "alpha = 111" in _read(tmp_path, "app.py")


def test_each_traced_call_records_its_place_in_the_models_batch(simple_model, tmp_path, monkeypatch):
    """Parallel calls are traced as each finishes; batch_index keeps the order the model emitted them."""
    def read_file(self, args):
        if args.get("path") == "a.py":
            time.sleep(0.3)
        return {"content": args.get("path")}

    monkeypatch.setitem(ToolExecutor._handlers, "read_file", read_file)
    orch = _orch(simple_model, tmp_path)
    blocks = [_Block(f"t{i}", "read_file", {"path": path}) for i, path in enumerate(("a.py", "b.py", "c.py"))]

    orch._execute_tool_blocks(blocks, turn=0)

    trace = str(tmp_path / ".besser_trace.jsonl")
    with open(trace, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    traced = [record["payload"] for record in records if record["event"] == "tool_call"]
    assert traced[-1]["input"]["path"] == "a.py", "the slow first call should be traced last"
    assert [event["input"]["path"] for event in sorted(traced, key=lambda e: e["batch_index"])] == [
        "a.py", "b.py", "c.py"]
    [run] = load_runs(trace)
    assert [call.input["path"] for call in run.calls] == ["a.py", "b.py", "c.py"]


def test_single_block_path_is_unchanged(simple_model, tmp_path):
    _seed(tmp_path, "app.py", "a = 1\n")
    orch = _orch(simple_model, tmp_path)
    results = orch._execute_tool_blocks(
        [_Block("t1", "modify_file", {
            "path": "app.py", "old_text": "a = 1", "new_text": "a = 2",
        })],
        turn=0,
    )
    assert len(results) == 1
    assert json.loads(results[0]["content"]).get("status") == "modified"


# -- shell calls and writes in one batch --------------------------------------
# A shell call ran in the thread pool beside the batch's writes, so a build
# emitted after an edit could read a half-edited tree, and the write-time
# import probe (which puts a candidate module on disk while it imports it)
# could overlap a command that imports the same module.

def _recording_orch(simple_model, tmp_path, events, barrier=None):
    import threading
    import time

    orch = _orch(simple_model, tmp_path)
    lock = threading.Lock()

    def note(what):
        with lock:
            events.append(what)

    def fake(block, turn):
        note(f"{block.id}:start")
        if block.name == "read_file" and barrier is not None:
            barrier.wait()
        time.sleep(0.3 if block.name in orch._WRITE_TOOLS else 0.05)
        note(f"{block.id}:end")
        return {"type": "tool_result", "tool_use_id": block.id, "content": "{}"}

    orch._execute_single_tool = fake
    return orch


def test_a_command_waits_for_the_edits_emitted_before_it(simple_model, tmp_path):
    events = []
    orch = _recording_orch(simple_model, tmp_path, events)
    blocks = [_Block("edit", "modify_file", {"path": "a.py", "old_text": "x", "new_text": "y"}),
              _Block("cmd", "run_command", {"command": "npm run build"})]

    results = orch._execute_tool_blocks(blocks, turn=0)

    assert events.index("edit:end") < events.index("cmd:start"), events
    assert [r["tool_use_id"] for r in results] == ["edit", "cmd"]


def test_an_edit_emitted_after_a_command_waits_for_it(simple_model, tmp_path):
    events = []
    orch = _recording_orch(simple_model, tmp_path, events)
    blocks = [_Block("cmd", "run_command", {"command": "pytest"}),
              _Block("edit", "write_file", {"path": "a.py", "content": "x = 1\n"}),
              _Block("cmd2", "run_command", {"command": "pytest"})]

    results = orch._execute_tool_blocks(blocks, turn=0)

    assert events.index("cmd:end") < events.index("edit:start") < events.index("edit:end") \
        < events.index("cmd2:start"), events
    assert [r["tool_use_id"] for r in results] == ["cmd", "edit", "cmd2"]


def test_reads_in_a_batch_with_a_command_still_run_in_parallel(simple_model, tmp_path):
    import threading

    events = []
    # Each read waits for the other: serialized reads would time the barrier out.
    barrier = threading.Barrier(2, timeout=5)
    orch = _recording_orch(simple_model, tmp_path, events, barrier)
    blocks = [_Block("r1", "read_file", {"path": "a.py"}),
              _Block("r2", "read_file", {"path": "b.py"}),
              _Block("edit", "modify_file", {"path": "a.py", "old_text": "x", "new_text": "y"}),
              _Block("cmd", "run_command", {"command": "pytest"})]

    orch._execute_tool_blocks(blocks, turn=0)

    assert not barrier.broken
    assert events.index("edit:end") < events.index("cmd:start"), events
