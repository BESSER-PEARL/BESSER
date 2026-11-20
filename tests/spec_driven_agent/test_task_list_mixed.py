"""action='mixed' lets one task_list call carry several verbs at once.

Measured on live Qwen run ys4gfj4v: 22 task_list calls against 9 actual
edits. A turn that wants to close two items, add one newly-discovered item,
and drop one the user never asked for needed up to four separate calls
(``done``, ``add``, ``drop``, plus often a ``list`` to see state) because
``action`` was a single enum. ``action='mixed'`` fixes that WITHOUT
resending the whole checklist (the OpenCode shape) and WITHOUT loosening any
gate: every verb here dispatches to the exact single-action handler
(``_do_done`` / ``_do_add`` / ``_do_drop`` / ``_do_blocked``), so
deterministic verifiers and the non-empty-reason requirement of drop and
blocked still apply per item, unchanged.
"""

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


@pytest.fixture
def executor(tmp_path):
    return ToolExecutor(workspace=str(tmp_path))


def _mixed(executor, **kwargs):
    return executor._task_list(dict(action="mixed", **kwargs))


# ======================================================================
# A mixed call that fully succeeds
# ======================================================================


def test_a_mixed_call_that_fully_succeeds(executor):
    executor.set_tasks(["one", "two", "three"])

    result = _mixed(executor, done_ids=[1], add_texts=["discovered item"])

    assert "error" not in result
    assert result["status"] == "ok"
    assert result["results"]["done"]["done_ids"] == [1]
    assert result["results"]["add"]["status"] == "added"
    assert result["results"]["add"]["ids"] == [4]
    assert executor._result_status(result) == "ok"
    assert [t["id"] for t in executor.open_tasks()] == [2, 3, 4]


def test_a_mixed_call_can_drop_and_block_together(executor):
    executor.set_tasks(["one", "two", "three"])

    result = _mixed(
        executor,
        drop=[{"id": 1, "reason": "not requested"}],
        blocked=[{"id": 2, "reason": "external dependency unavailable"}],
    )

    assert "error" not in result
    assert result["status"] == "ok"
    assert result["results"]["drop"]["dropped_ids"] == [1]
    assert result["results"]["blocked"]["blocked_ids"] == [2]
    assert [t["id"] for t in executor.open_tasks()] == [3]


# ======================================================================
# Half succeeds, half refused -- must read as neither total success nor
# total failure.
# ======================================================================


def test_done_refused_by_a_failing_check_but_add_lands(executor):
    executor.set_tasks([{"text": "one", "verify": lambda: False}, "two"])

    result = _mixed(executor, done_ids=[1], add_texts=["discovered item"])

    # The harness reads this as progress: real work (the add) happened, so
    # the loop guards and tracing must not treat it as a wasted call.
    assert "error" not in result
    assert executor._result_status(result) == "ok"
    # But it is not a full success either -- the model must be able to see
    # exactly which verb was refused and why.
    assert result["status"] == "partial"
    assert result["results"]["done"]["done_ids"] == []
    assert "error" in result["results"]["done"]
    assert result["results"]["add"]["status"] == "added"
    assert 1 in [t["id"] for t in executor.open_tasks()]


def test_a_mixed_call_that_fully_fails_is_reported_as_an_error(executor):
    executor.set_tasks(["one"])

    result = _mixed(executor, done_ids=[99], drop=[{"id": 999, "reason": "x"}])

    assert result["status"] == "refused"
    assert "error" in result
    assert executor._result_status(result) == "error"


def test_mixed_requires_at_least_one_verb(executor):
    executor.set_tasks(["one"])

    result = _mixed(executor)

    assert "error" in result
    assert executor.open_tasks(), "nothing should have changed"


# ======================================================================
# drop / blocked still require a non-empty reason, even batched inside a
# mixed call.
# ======================================================================


def test_drop_inside_a_mixed_call_still_requires_a_reason(executor):
    executor.set_tasks(["one"])

    result = _mixed(executor, drop=[{"id": 1}])  # no reason

    assert "error" in result["results"]["drop"]
    assert executor.open_tasks(), "the item must not have been dropped"


def test_blocked_inside_a_mixed_call_still_requires_a_reason(executor):
    executor.set_tasks(["one"])

    result = _mixed(executor, blocked=[{"id": 1}])  # no reason

    assert "error" in result["results"]["blocked"]
    assert executor.blocked_tasks() == []


def test_a_partially_bad_drop_list_still_reports_what_succeeded(executor):
    executor.set_tasks(["one", "two"])

    result = _mixed(executor, drop=[
        {"id": 1, "reason": "not requested"},
        {"id": 999, "reason": "unknown id"},
    ])

    assert result["results"]["drop"]["dropped_ids"] == [1]
    assert result["results"]["drop"]["errors"]
    assert "error" not in result  # real progress happened


# ======================================================================
# Backward compatibility: every existing single-verb call shape must behave
# exactly as before the refactor -- checkpoints and prompts are baked for
# these shapes.
# ======================================================================


def test_single_action_shapes_are_unaffected_by_the_refactor(executor):
    executor.set_tasks(["one", "two", "three"])

    listing = executor._task_list({"action": "list"})
    assert listing["tasks"][0] == {"id": 1, "text": "one", "status": "open"}

    done_single = executor._task_list({"action": "done", "id": 1})
    assert done_single["done_ids"] == [1]
    assert done_single["id"] == 1
    assert done_single["open_remaining"] == 2

    done_batch = executor._task_list({"action": "done", "ids": [2]})
    assert done_batch["done_ids"] == [2]
    assert done_batch["id"] == 2  # a single-item ids=[...] batch still reports `id`

    added_single = executor._task_list({"action": "add", "text": "One more"})
    assert added_single == {"status": "added", "id": 4, "open": 2}

    added_batch = executor._task_list({"action": "add", "texts": ["Batch one", "Batch two"]})
    assert added_batch == {"status": "added", "ids": [5, 6], "open": 4}

    dropped = executor._task_list({"action": "drop", "id": 3, "reason": "not requested"})
    assert dropped == {"status": "dropped", "id": 3, "open": 3}

    blocked = executor._task_list({"action": "blocked", "id": 5, "reason": "external dep down"})
    assert blocked == {"status": "blocked", "blocked_ids": [5], "reason": "external dep down", "open": 2}

    unknown_action = executor._task_list({"action": "nonsense"})
    assert "Unknown action" in unknown_action["error"]
    assert "mixed" in unknown_action["error"]


def test_the_tool_schema_advertises_mixed(executor):
    from besser.spec_driven_agent.agent.tools import VALIDATION_TOOLS
    spec = next(t for t in VALIDATION_TOOLS if t["name"] == "task_list")

    assert "mixed" in spec["input_schema"]["properties"]["action"]["enum"]
    for key in ("done_ids", "add_texts", "drop", "blocked"):
        assert key in spec["input_schema"]["properties"], key
    assert "done_ids" in spec["description"]
    assert "add_texts" in spec["description"]
