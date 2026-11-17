"""A passing validate_app names the open checklist items.

Live evidence (r11-p51 turn 9, r11-p81 turn 20, r10-p51 turn 9, r9-p21 turn 10, r10-p61 turn 12):
the turn after a passing validate_app was spent on ``task_list(action='list')``
alone, only to get the ids it then marked done. The verified result now carries
them, in the shape ``task_list(done)`` already uses.
"""

import json

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

_PASSING = {"verified": True, "blocker_count": 0, "issues": [], "next_step": "These checks pass."}


def _validate(executor: ToolExecutor) -> dict:
    return json.loads(executor.execute("validate_app", {}))


def _executor(tmp_path, report: dict) -> ToolExecutor:
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.app_validator = lambda: report
    executor.set_tasks(["Add the React frontend", "Implement the risk score " + "in detail " * 12])
    return executor


def test_passing_result_lists_open_items_with_ids(tmp_path):
    executor = _executor(tmp_path, dict(_PASSING))
    executor._tasks[0]["done"] = True

    result = _validate(executor)

    assert [item["id"] for item in result["open_items"]] == [2]
    assert result["open_items"][0]["label"].startswith("Implement the risk score")
    assert result["open_items"][0]["label"].endswith("...")  # a label, not the full text


def test_failing_result_does_not_list_open_items(tmp_path):
    executor = _executor(tmp_path, {"verified": False, "blocker_count": 1, "issues": ["x"]})

    assert "open_items" not in _validate(executor)


def test_all_closed_says_so_and_the_validator_cache_is_not_mutated(tmp_path):
    # r10-p61 turn 12: a task_list 'list' alone, after a passing validate_app, found nothing open.
    report = dict(_PASSING)
    executor = _executor(tmp_path, report)
    assert len(_validate(executor)["open_items"]) == 2
    assert "open_items" not in report  # the orchestrator caches and re-serves this dict

    for task in executor._tasks:
        task["done"] = True
    assert _validate(executor)["open_items"] == []


def test_no_checklist_adds_nothing(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.app_validator = lambda: dict(_PASSING)

    assert "open_items" not in _validate(executor)
