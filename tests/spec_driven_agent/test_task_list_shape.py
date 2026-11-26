"""task_list takes one action-free shape, and every result lists the open items.

Its schema had six actions and ten properties with parallel aliases (id / ids /
done_ids, text / texts / add_texts) and a ``mixed`` action duplicating the other
four: the second-largest tool schema on hosted runs, and recorded runs sent
shapes it did not take. ``add``, ``drop`` and ``blocked`` answered with only an
open count, so a model listed the checklist again to find the next id.
"""
import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import VALIDATION_TOOLS

SCHEMA = next(tool for tool in VALIDATION_TOOLS if tool["name"] == "task_list")


@pytest.fixture
def executor(tmp_path):
    ex = ToolExecutor(workspace=str(tmp_path))
    ex.set_tasks(["Add an order page", "Validate the customer email", "Export orders as CSV"])
    return ex


def _call(executor, **args):
    return executor.execute_typed("task_list", args).payload


def _open_ids(result):
    return [item["id"] for item in result["open_items"]]


def test_the_schema_has_no_action_and_no_aliases():
    assert set(SCHEMA["input_schema"]["properties"]) == {"done", "add", "drop", "blocked"}
    assert "required" not in SCHEMA["input_schema"]
    assert "action=" not in SCHEMA["description"]


def test_one_call_combines_every_verb(executor):
    result = _call(executor, done=[1], add=["Add a customer filter"],
                   drop=[{"id": 2, "reason": "not requested"}], blocked=[{"id": 3, "reason": "no export library"}])

    assert result["status"] == "ok", result
    assert _open_ids(result) == [4]


@pytest.mark.parametrize("args", [
    {"add": ["Add a customer filter"]},
    {"drop": [{"id": 2, "reason": "not requested"}]},
    {"blocked": [{"id": 3, "reason": "no export library"}]},
    {"done": 1},
])
def test_every_result_lists_the_open_items(executor, args):
    result = _call(executor, **args)

    assert "error" not in result, result
    assert all(set(item) == {"id", "label"} for item in result["open_items"])


def test_a_call_with_no_fields_lists_the_checklist(executor):
    result = _call(executor)

    assert [row["id"] for row in result["tasks"]] == [1, 2, 3]
    assert _open_ids(result) == [1, 2, 3]


def test_the_earlier_action_shape_still_works_and_is_not_reported(executor):
    result = _call(executor, action="done", ids=[1])

    assert result["done_ids"] == [1], result
    assert "ignored_arguments" not in result


@pytest.mark.parametrize("args, closed", [
    ({"action": "done", "ids": "[1, 2]"}, [1, 2]),
    ({"action": "mixed", "done_ids": "[3]", "add_texts": '["Add a customer filter"]'}, [3]),
])
def test_the_earlier_shape_decodes_stringified_arrays(executor, args, closed):
    """Its arrays are no longer in the schema, so provider decoding skips them."""
    result = _call(executor, **args)

    assert "error" not in result, result
    assert all(executor._tasks[i - 1]["done"] for i in closed)


def test_empty_arrays_count_as_absent_fields(executor):
    """Models fill every optional array with []; that must not refuse the field they used."""
    result = _call(executor, done=[1], add=[], drop=[], blocked=[])

    assert "error" not in result and result["status"] != "partial", result
    assert executor._tasks[0]["done"]


def test_all_empty_arrays_list_the_checklist(executor):
    result = _call(executor, done=[], add=[], drop=[], blocked=[])

    assert "error" not in result, result
    assert [row["id"] for row in result["tasks"]] == [1, 2, 3]
