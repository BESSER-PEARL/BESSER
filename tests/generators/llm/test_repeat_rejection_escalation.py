"""When the executor rejects the same edit again, the orchestrator changes
what the model can DO, in steps, instead of repeating what it is told.

Two live runs on 2026-09-18 (0c537a4e: 13 identical misses; 57160293: 16
identical no-op calls) alternated read_file / modify_file for ~30 turns each
while every guard either never fired or was ignored. Aider's answer is
``max_reflections = 3`` and then the human decides; headless, the runtime
has to decide: force the task list, then close the file, then stop the phase
and deliver what exists.
"""
from __future__ import annotations

import json
import os

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator


@pytest.fixture
def simple_model():
    user = Class(name="User")
    user.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="name", type=PrimitiveDataType("str")),
    }
    return DomainModel(name="TestModel", types={user})


class MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class StuckClient:
    """Sends the same no-op modify_file forever; honours nothing it is told."""
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def __init__(self):
        self.turns = []          # (turn, force_tool)
        self.texts_seen = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        n = len(self.turns) + 1
        self.turns.append((n, force_tool))
        for m in messages:
            c = m.get("content")
            if m.get("role") == "user" and isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        self.texts_seen.append(b["text"])
        if force_tool == "task_list":
            return {"stop_reason": "tool_use", "content": [
                MockBlock("tool_use", name="task_list", id=f"t{n}", input={"action": "list"}),
            ]}
        return {"stop_reason": "tool_use", "content": [
            MockBlock("tool_use", name="modify_file", id=f"m{n}",
                      input={"path": "app.py", "old_text": "x = 1\n", "new_text": "x = 1\n"}),
        ]}


def _run(simple_model, tmp_path, max_turns=20):
    with open(os.path.join(tmp_path, "app.py"), "w", encoding="utf-8") as f:
        f.write("x = 1\n")
    client = StuckClient()
    orch = LLMOrchestrator(llm_client=client, domain_model=simple_model,
                           output_dir=str(tmp_path), max_turns=max_turns)
    orch.run("Build an app")
    return orch, client


def test_the_second_repeat_forces_the_task_list_on_the_next_turn(simple_model, tmp_path):
    orch, client = _run(simple_model, tmp_path)
    forced = [n for n, f in client.turns if f == "task_list"]
    assert forced, client.turns
    assert forced[0] == 4, "rejections at turns 1,2,3 (2nd repeat at 3) -> forced at 4"
    assert any("task_list" in t and "app.py" in t for t in client.texts_seen)


def test_the_fourth_repeat_closes_the_file_and_the_sixth_ends_the_phase(simple_model, tmp_path):
    orch, client = _run(simple_model, tmp_path)
    assert orch._phase2_stop_reason == "stuck_edit_loop"
    assert len(client.turns) < 12, f"the loop ran {len(client.turns)} turns"
    assert any("closed" in t and "app.py" in t for t in client.texts_seen)
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_client_without_force_tool_still_gets_the_message_and_the_stop(simple_model, tmp_path):
    class PlainClient(StuckClient):
        def chat(self, system, messages, tools):        # no force_tool parameter
            return super().chat(system, messages, tools)

    with open(os.path.join(tmp_path, "app.py"), "w", encoding="utf-8") as f:
        f.write("x = 1\n")
    client = PlainClient()
    orch = LLMOrchestrator(llm_client=client, domain_model=simple_model,
                           output_dir=str(tmp_path), max_turns=20)
    orch.run("Build an app")
    assert orch._phase2_stop_reason == "stuck_edit_loop"
    assert any("task_list" in t for t in client.texts_seen)


def test_untruncated_edit_inputs_are_kept_beside_the_trace(simple_model, tmp_path):
    """Item 1's other half: the trace/checkpoint/recipe are bounded at 500
    chars, so the full bytes of every edit go to a sidecar the fixtures can
    be cut from."""
    long_block = "x = 1\n" + ("# padding line to push this input over the log budget\n" * 12)
    with open(os.path.join(tmp_path, "app.py"), "w", encoding="utf-8") as f:
        f.write(long_block)

    class OneEdit(StuckClient):
        def chat(self, system, messages, tools, force_tool=None, model_override=None):
            n = len(self.turns) + 1
            self.turns.append((n, force_tool))
            if n > 1:
                return {"stop_reason": "end_turn", "content": [MockBlock("text", text="done")]}
            return {"stop_reason": "tool_use", "content": [
                MockBlock("tool_use", name="modify_file", id="m1",
                          input={"path": "app.py", "old_text": long_block,
                                 "new_text": long_block + "y = 2\n"}),
            ]}

    LLMOrchestrator(llm_client=OneEdit(), domain_model=simple_model,
                    output_dir=str(tmp_path), max_turns=5).run("Build an app")
    sidecar = tmp_path / ".besser_tool_inputs.jsonl"
    assert sidecar.exists(), sorted(os.listdir(tmp_path))
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["tool"] == "modify_file"
    assert rows[0]["input"]["old_text"] == long_block, "must be the untruncated bytes"
    assert len(long_block) > 500
