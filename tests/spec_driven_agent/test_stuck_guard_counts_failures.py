"""The tool-name streak warning fires on N consecutive FAILURES of one call,
never on successes.

A recorded run: ``_is_stuck`` counted tool names only, so it
wrapped "'modify_file' called 4 times in a row. Move on." onto the results
of two SUCCESSFUL edits (t19, t20) and then onto every error result after,
contradicting the ``advice`` field that told the model to read and retry.
"""
from __future__ import annotations

import json
import os

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


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


def _tool_result_texts(messages):
    out = []
    for m in messages:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                c = b.get("content")
                out.append(c if isinstance(c, str) else json.dumps(c))
    return out


def _run(simple_model, tmp_path, make_call, turns=5):
    counter = {"n": 0}
    captured = {}

    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, system, messages, tools):
            counter["n"] += 1
            if counter["n"] == turns:
                captured["msgs"] = [dict(m) for m in messages]
                return {"stop_reason": "end_turn", "content": [MockBlock("text", text="done")]}
            return {"stop_reason": "tool_use", "content": [make_call(counter["n"])]}

    LLMOrchestrator(llm_client=Client(), domain_model=simple_model,
                    output_dir=str(tmp_path), max_turns=turns + 2).run("Build an app")
    assert "msgs" in captured
    return _tool_result_texts(captured["msgs"])


def test_four_successful_writes_to_one_path_are_not_called_a_loop(simple_model, tmp_path):
    results = _run(simple_model, tmp_path, lambda n: MockBlock(
        "tool_use", name="write_file", id=f"w{n}",
        input={"path": "notes.txt", "content": f"version {n}\n"}))
    assert not any("in a row" in r for r in results), results


def test_four_failed_edits_to_one_path_are(simple_model, tmp_path):
    with open(os.path.join(tmp_path, "app.py"), "w", encoding="utf-8") as f:
        f.write("x = 1\n")
    results = _run(simple_model, tmp_path, lambda n: MockBlock(
        "tool_use", name="modify_file" if n <= 4 else "read_file", id=f"m{n}",
        input={"path": "app.py", "old_text": f"nomatch{n}", "new_text": "y"}), turns=6)
    assert any("in a row" in r for r in results), results
    warned = [json.loads(r) for r in results if "in a row" in r]
    assert all("error" in r for r in warned), "loop wrapping must retain the actual error"
    assert "in a row" not in results[-1], "successful reads must not inherit stale failures"
