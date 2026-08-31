"""Tests for the task_list tool and the end_turn checklist gate.

"Done" means the checklist is closed, not "the model said done": Phase 2
seeds the gap-analysis tasks into the executor's checklist, the LLM
manages them through the ``task_list`` tool, and the end_turn gate sends
the model back to open items (bounded by ``_MAX_TASK_NUDGES``).
"""

import json

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
import besser.generators.llm.orchestrator as orchestrator_module
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tool_executor import ToolExecutor


class _Block:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Usage:
    def __init__(self):
        self.estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 1, "cost_usd": 0.0}


class _ScriptedClient:
    def __init__(self, responses):
        self.model = "test-model"
        self.usage = _Usage()
        self.max_tokens = 4096
        self._responses = list(responses)
        self.chat_calls = 0

    def chat(self, system=None, messages=None, tools=None, **kwargs):
        self.chat_calls += 1
        if self._responses:
            return self._responses.pop(0)
        return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


def _end_turn():
    return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


def _tool_use(name, tool_id, tool_input):
    return {
        "stop_reason": "tool_use",
        "content": [_Block("tool_use", name=name, id=tool_id, input=tool_input)],
    }


def _make_domain() -> DomainModel:
    string_type = PrimitiveDataType("str")
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=string_type, is_id=True)}
    return DomainModel(name="Library", types={book})


def _make_orch(tmp_path, client) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=client,
        domain_model=_make_domain(),
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
    )


# ----------------------------------------------------------------------
# Executor checklist CRUD
# ----------------------------------------------------------------------


def test_task_list_tool_crud(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_tasks(["Add auth", "Style the pages", "  "])  # blank dropped

    listing = json.loads(executor.execute("task_list", {"action": "list"}))
    assert listing["open"] == 2
    assert listing["tasks"][0] == {"id": 1, "text": "Add auth", "status": "open"}

    done = json.loads(executor.execute("task_list", {"action": "done", "id": 1}))
    assert done["open_remaining"] == 1
    assert done["open_items"] == [{"id": 2, "text": "Style the pages"}]

    added = json.loads(executor.execute("task_list", {"action": "add", "text": "Wire delete"}))
    assert added == {"status": "added", "id": 3, "open": 2}

    bad = json.loads(executor.execute("task_list", {"action": "done", "id": 99}))
    assert "No task with id 99" in bad["error"]

    assert [t["id"] for t in executor.open_tasks()] == [2, 3]


# ----------------------------------------------------------------------
# end_turn gate
# ----------------------------------------------------------------------


def test_end_turn_with_open_tasks_is_nudged_then_released(tmp_path, monkeypatch):
    """A model that end_turns past open items gets 2 nudges, then the run
    completes anyway (bounded — no budget loop)."""
    monkeypatch.setattr(
        orchestrator_module, "analyze_gaps_via_llm",
        lambda **kwargs: ["Add auth", "Style the pages"],
    )
    client = _ScriptedClient([_end_turn(), _end_turn(), _end_turn()])
    orch = _make_orch(tmp_path, client)

    orch._run_phase2("build it", extra_issues=[])

    assert client.chat_calls == 3  # initial end_turn + 2 nudged retries
    assert orch._end_turn_task_nudges == 2
    assert orch._phase2_stop_reason == "completed"
    assert orch._phase2_exited_cleanly is True


def test_end_turn_with_closed_checklist_finishes_immediately(tmp_path, monkeypatch):
    """Marking every item done releases the gate with zero nudges."""
    monkeypatch.setattr(
        orchestrator_module, "analyze_gaps_via_llm",
        lambda **kwargs: ["Add auth", "Style the pages"],
    )
    client = _ScriptedClient([
        _tool_use("task_list", "t1", {"action": "done", "id": 1}),
        _tool_use("task_list", "t2", {"action": "done", "id": 2}),
        _end_turn(),
    ])
    orch = _make_orch(tmp_path, client)

    orch._run_phase2("build it", extra_issues=[])

    assert client.chat_calls == 3
    assert orch._end_turn_task_nudges == 0
    assert orch._phase2_stop_reason == "completed"


def test_no_gap_tasks_means_no_gate(tmp_path, monkeypatch):
    """When gap analysis fails (None), the gate is inert."""
    monkeypatch.setattr(
        orchestrator_module, "analyze_gaps_via_llm", lambda **kwargs: None,
    )
    client = _ScriptedClient([_end_turn()])
    orch = _make_orch(tmp_path, client)

    orch._run_phase2("build it", extra_issues=[])

    assert client.chat_calls == 1
    assert orch._end_turn_task_nudges == 0
    assert orch._phase2_stop_reason == "completed"
