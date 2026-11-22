"""One rule for a task id wherever task_list takes one: ``ids``, ``id``, evidence, drop, blocked.

``ids`` accepted ``"1"`` but evidence was matched with ``!=``, so a model that
sent string ids had valid evidence ignored and the task recorded BLOCKED after
three attempts with a false reason. ``drop`` refused ``"1"`` yet dropped task 1
for ``true``.
"""

import json

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


@pytest.fixture
def executor(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_tasks(["Add a health endpoint", "Add a version endpoint"])
    executor._write_file({"path": "app.py", "content": "def health():\n    return {'ok': True}\n"})
    return executor


def test_string_ids_in_evidence_count(executor):
    result = executor._task_list({"action": "done", "ids": ["1"],
                                  "evidence": [{"id": "1", "path": "app.py", "quote": "def health():"}]})
    assert result["done_ids"] == [1], result
    assert "error" not in result


def test_drop_accepts_a_string_id(executor):
    result = executor._task_list({"action": "drop", "id": "1", "reason": "not requested"})
    assert result == {"status": "dropped", "id": 1, "open": 1}


@pytest.mark.parametrize("bad", [True, 1.5, "one", None, [1], float("inf"), float("nan")])
def test_drop_rejects_what_is_not_an_id(executor, bad):
    result = executor._task_list({"action": "drop", "id": bad, "reason": "not requested"})
    assert "error" in result
    assert len(executor.open_tasks()) == 2


@pytest.mark.parametrize("bad", [True, 1.5, float("inf"), float("nan")])
def test_done_rejects_what_is_not_an_id(executor, bad):
    result = executor._task_list({"action": "done", "ids": [bad],
                                  "evidence": [{"id": 1, "path": "app.py", "quote": "def health():"}]})
    assert "error" in result and result.get("done_ids", []) == []
    assert len(executor.open_tasks()) == 2


def test_an_integral_float_is_an_id(executor):
    result = executor._task_list({"action": "done", "ids": [1.0],
                                  "evidence": [{"id": 1.0, "path": "app.py", "quote": "def health():"}]})
    assert result["done_ids"] == [1]


def test_an_overflowing_id_is_a_clean_error(executor):
    """``1e400`` in JSON parses to ``inf``; ``int(inf)`` raises OverflowError."""
    result = json.loads(executor.execute("task_list", {"action": "done", "ids": [float("inf")]}))
    assert "Not integers" in result["error"]


def test_blocked_takes_the_same_ids(executor):
    assert executor._task_list({"action": "blocked", "id": "2", "reason": "no spec"})["blocked_ids"] == [2]
    assert "error" in executor._task_list({"action": "blocked", "id": True, "reason": "no spec"})
