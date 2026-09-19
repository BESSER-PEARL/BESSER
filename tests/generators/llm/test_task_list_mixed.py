"""action='mixed' lets one task_list call carry several verbs at once.

Measured on live Qwen run ys4gfj4v: 22 task_list calls against 9 actual
edits. A turn that wants to close two items, add one newly-discovered item,
and drop one the user never asked for needed up to four separate calls
(``done``, ``add``, ``drop``, plus often a ``list`` to see state) because
``action`` was a single enum. ``action='mixed'`` fixes that WITHOUT
resending the whole checklist (the OpenCode shape) and WITHOUT loosening any
gate: every verb here dispatches to the exact single-action handler
(``_do_done`` / ``_do_add`` / ``_do_drop`` / ``_do_blocked``), so
implementation evidence, deterministic verifiers, the
block-after-N-attempts guard, and drop's non-empty-reason requirement all
still apply per item, unchanged.
"""

import pytest

from besser.generators.llm.tool_executor import (
    _MAX_TASK_VERIFY_ATTEMPTS,
    ToolExecutor,
)


@pytest.fixture
def executor(tmp_path):
    return ToolExecutor(workspace=str(tmp_path))


def _mixed(executor, **kwargs):
    return executor._task_list(dict(action="mixed", **kwargs))


def _write_evidence(executor, task_id, path="impl.py"):
    """Real modification receipts, not acceptance proof (mirrors the helper
    in test_task_list_batching.py, duplicated here to keep this file
    self-contained)."""
    content = f"task_{task_id}_done = True\n"
    assert executor._write_file({"path": path, "content": content})["status"] == "written"
    return [{"id": task_id, "path": path, "quote": content.strip()}]


# ======================================================================
# A mixed call that fully succeeds
# ======================================================================


def test_a_mixed_call_that_fully_succeeds(executor):
    executor.set_tasks(["one", "two", "three"])
    evidence = _write_evidence(executor, 1)

    result = _mixed(executor, done_ids=[1], evidence=evidence, add_texts=["discovered item"])

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


def test_done_refused_for_missing_evidence_but_add_lands(executor):
    executor.set_tasks(["one", "two"])

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
# The block-after-N-attempts guard must not be bypassable by routing a
# refusal through action='mixed' instead of action='done'.
# ======================================================================


def test_a_refused_done_inside_a_mixed_call_still_counts_an_attempt(executor):
    executor.set_tasks([{"text": "build the frontend", "verify": lambda: False}])

    for attempt in range(1, _MAX_TASK_VERIFY_ATTEMPTS):
        result = _mixed(executor, done_ids=[1], add_texts=[f"note {attempt}"])
        assert executor._tasks[0]["attempts"] == attempt
        assert result["results"]["done"]["refused"][0]["attempts"] == attempt
        assert executor._tasks[0]["blocked"] is False

    final = _mixed(executor, done_ids=[1])

    assert executor._tasks[0]["blocked"] is True
    assert final["results"]["done"]["blocked"][0]["id"] == 1


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
# The existing=true evidence path (an already-existing implementation cited,
# not written this run) must gate exactly as it does for action='done'.
# ======================================================================


def test_mixed_done_honors_the_existing_true_evidence_path(executor, tmp_path):
    source = (
        "# Payment is implemented\n"
        "def register_payment(bill, db):\n"
        "    if bill.paid:\n"
        "        return False\n"
        "    bill.paid = True\n"
        "    db.commit()\n"
        "    return True\n"
    )
    path = tmp_path / "payment.py"
    path.write_text(source, encoding="utf-8")
    executor.set_tasks(["Implement payment"])
    evidence = [{"id": 1, "path": "payment.py", "quote": "bill.paid = True"}]

    # Unread file: refused, same reason action='done' gives.
    unread = _mixed(executor, done_ids=[1], existing=True, evidence=evidence)
    assert unread["results"]["done"]["done_ids"] == []
    assert "read payment.py" in unread["results"]["done"]["refused"][0]["reason"]

    executor._read_file({"path": "payment.py"})
    accepted = _mixed(executor, done_ids=[1], existing=True, evidence=evidence)

    assert accepted["results"]["done"]["done_ids"] == [1]
    assert accepted["results"]["done"]["unverified_ids"] == [1]
    receipt = executor.unverified_tasks()[0]["implementation_evidence"][0]
    assert receipt["origin"] == "existing"


# ======================================================================
# Backward compatibility: every existing single-verb call shape must behave
# exactly as before the refactor -- checkpoints and prompts are baked for
# these shapes.
# ======================================================================


def test_single_action_shapes_are_unaffected_by_the_refactor(executor):
    executor.set_tasks(["one", "two", "three"])

    listing = executor._task_list({"action": "list"})
    assert listing["tasks"][0] == {"id": 1, "text": "one", "status": "open", "verification": "unverified"}

    evidence1 = _write_evidence(executor, 1)
    done_single = executor._task_list({"action": "done", "id": 1, "evidence": evidence1})
    assert done_single["done_ids"] == [1]
    assert done_single["id"] == 1
    assert done_single["open_remaining"] == 2

    evidence2 = _write_evidence(executor, 2, path="impl2.py")
    done_batch = executor._task_list({"action": "done", "ids": [2], "evidence": evidence2})
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
    from besser.generators.llm.tools import VALIDATION_TOOLS
    spec = next(t for t in VALIDATION_TOOLS if t["name"] == "task_list")

    assert "mixed" in spec["input_schema"]["properties"]["action"]["enum"]
    for key in ("done_ids", "add_texts", "drop", "blocked"):
        assert key in spec["input_schema"]["properties"], key
    assert "done_ids" in spec["description"]
    assert "add_texts" in spec["description"]
