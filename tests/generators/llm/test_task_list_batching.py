"""The checklist must not cost one turn per item, and must not livelock.

Measured across a 10-run live batch on 2026-09-11 (agent modelling + qwen
generation, one app per run):

    202 of 502 turns -- 40% -- were task_list calls.
    Worst run: 71 of 86 turns (83%), of which ~62 were CONSECUTIVE.

Two separate causes, both fixed here:

1. ``action='done'`` took a single ``id``, so completing N items cost N turns.
   Every turn pays a full prompt prefill, which is the binding cost on the
   self-hosted box, so bookkeeping was outrunning the actual work (one run:
   30 task_list calls against 12 write_file calls).

2. A verifier that never passes was retried forever. The refusal told the model
   "do the work first, then mark it done"; it obliged, the check failed again,
   and nothing broke the cycle except the 80-turn cap. That run ended
   INCOMPLETE having used 317s of its 2400s budget -- 13%.

The verifier itself is deliberately kept: it exists because a model once marked
"build the frontend" done without writing a single file. A blocked item is
therefore NOT marked done -- it is recorded, excluded from the open set so the
run can finish, and reported.
"""

import pytest

from besser.generators.llm.tool_executor import (
    _MAX_TASK_VERIFY_ATTEMPTS,
    ToolExecutor,
)


@pytest.fixture
def executor(tmp_path):
    return ToolExecutor(workspace=str(tmp_path))


def _done(executor, **kwargs):
    return executor._task_list(dict(action="done", **kwargs))


# ----------------------------------------------------------- batching


def test_many_items_are_marked_done_in_one_call(executor):
    executor.set_tasks(["one", "two", "three", "four"])
    result = _done(executor, ids=[1, 2, 3, 4])
    assert result["done_ids"] == [1, 2, 3, 4]
    assert result["open_remaining"] == 0


def test_a_single_id_still_works(executor):
    """The old shape is what existing prompts and checkpoints use."""
    executor.set_tasks(["one", "two"])
    result = _done(executor, id=1)
    assert result["done_ids"] == [1]
    assert result["id"] == 1
    assert result["open_remaining"] == 1


def test_batching_is_what_the_tool_advertises(executor):
    from besser.generators.llm.tools import VALIDATION_TOOLS
    spec = next(t for t in VALIDATION_TOOLS if t["name"] == "task_list")
    assert "ids" in spec["input_schema"]["properties"]
    assert spec["input_schema"]["properties"]["ids"]["type"] == "array"
    # The model only batches if the description tells it to.
    assert "ids=[1,2,3]" in spec["description"]


def test_unknown_and_duplicate_ids_do_not_derail_the_batch(executor):
    executor.set_tasks(["one", "two"])
    result = _done(executor, ids=[1, 1, 99])
    assert result["done_ids"] == [1]
    assert result["unknown_ids"] == [99]


@pytest.mark.parametrize("args", [
    {}, {"ids": []}, {"ids": ["not-an-id"]}, {"id": None}, {"ids": [True]},
])
def test_an_unusable_request_is_refused_clearly(executor, args):
    executor.set_tasks(["one"])
    result = _done(executor, **args)
    assert "error" in result
    assert executor.open_tasks(), "nothing should have been marked done"


# ------------------------------------------------- bounded verification


def test_a_failing_check_still_refuses_at_first(executor):
    """The cheat guard must survive: refusing is the whole point of verify."""
    executor.set_tasks([{"text": "build the frontend", "verify": lambda: False}])
    result = _done(executor, id=1)
    assert result["done_ids"] == []
    assert result["refused"][0]["id"] == 1
    assert executor.open_tasks(), "an unverified item must stay open"


def test_retries_are_bounded_and_the_item_is_then_blocked(executor):
    executor.set_tasks([{"text": "build the frontend", "verify": lambda: False}])
    for attempt in range(1, _MAX_TASK_VERIFY_ATTEMPTS):
        result = _done(executor, id=1)
        assert result["refused"][0]["attempts"] == attempt
        assert "blocked" not in result

    final = _done(executor, id=1)
    assert final["blocked"][0]["id"] == 1
    # The model must be told to stop, not invited to try again.
    advice = final["advice"].lower()
    assert "do not call task_list" in advice
    assert "move on" in advice


def test_a_blocked_item_is_not_reported_as_done(executor):
    """Marking it done would reinstate exactly the cheat verify guards against."""
    executor.set_tasks([{"text": "build the frontend", "verify": lambda: False}])
    for _ in range(_MAX_TASK_VERIFY_ATTEMPTS):
        _done(executor, id=1)
    task = executor._tasks[0]
    assert task["blocked"] is True
    assert task["done"] is False


def test_a_blocked_item_lets_the_run_finish(executor):
    """Left open, it kept the run going until the turn cap for work that can
    never be signed off -- the 62-consecutive-turn livelock."""
    executor.set_tasks([
        {"text": "build the frontend", "verify": lambda: False},
        "write the readme",
    ])
    _done(executor, id=2)
    for _ in range(_MAX_TASK_VERIFY_ATTEMPTS):
        _done(executor, id=1)
    assert executor.open_tasks() == []
    assert [t["text"] for t in executor.blocked_tasks()] == ["build the frontend"]


def test_the_loop_cannot_outlast_the_attempt_budget(executor):
    """Simulates the live failure: the model keeps trying the same item."""
    executor.set_tasks([{"text": "impossible", "verify": lambda: False}])
    calls_that_invited_a_retry = 0
    for _ in range(40):                      # the live run managed 62
        result = _done(executor, id=1)
        if result.get("refused"):
            calls_that_invited_a_retry += 1
    assert calls_that_invited_a_retry == _MAX_TASK_VERIFY_ATTEMPTS - 1
    assert executor.open_tasks() == []


def test_a_check_that_starts_failing_and_then_passes_is_accepted(executor):
    """The normal case: the model does the work between attempts."""
    state = {"written": False}
    executor.set_tasks([{"text": "write it", "verify": lambda: state["written"]}])
    assert _done(executor, id=1)["refused"][0]["id"] == 1
    state["written"] = True
    assert _done(executor, id=1)["done_ids"] == [1]
    assert executor.blocked_tasks() == []


def test_a_broken_verifier_never_wedges_the_run(executor):
    def explode():
        raise RuntimeError("verifier bug")
    executor.set_tasks([{"text": "x", "verify": explode}])
    assert _done(executor, id=1)["done_ids"] == [1]


# ------------------------------------------------------- crash recovery


def test_the_attempt_count_survives_a_resume(executor, tmp_path):
    """A resume that reset the counter would restart the livelock."""
    executor.set_tasks([{"text": "build the frontend", "verify": lambda: False}])
    _done(executor, id=1)
    snapshot = executor.task_snapshot()
    assert snapshot[0]["attempts"] == 1

    resumed = ToolExecutor(workspace=str(tmp_path))
    resumed.restore_tasks(
        snapshot, [{"text": "build the frontend", "verify": lambda: False}])
    assert resumed._tasks[0]["attempts"] == 1
    # Only the remaining attempts are left, not a fresh budget.
    for _ in range(_MAX_TASK_VERIFY_ATTEMPTS - 1):
        _done(resumed, id=1)
    assert resumed.blocked_tasks()


def test_a_blocked_item_stays_blocked_across_a_resume(executor, tmp_path):
    executor.set_tasks([{"text": "x", "verify": lambda: False}])
    for _ in range(_MAX_TASK_VERIFY_ATTEMPTS):
        _done(executor, id=1)
    resumed = ToolExecutor(workspace=str(tmp_path))
    resumed.restore_tasks(executor.task_snapshot())
    assert resumed.blocked_tasks()
    assert resumed.open_tasks() == []


# --------------------------------------------- the harness must see a failure


def test_a_fully_refused_call_is_reported_as_an_error(executor):
    """``_result_status`` only treats a payload as an error when it carries an
    "error" key. Batching initially dropped that key, which silently turned
    every refusal into a reported success for the loop guards and tracing."""
    executor.set_tasks([{"text": "build the frontend", "verify": lambda: False}])
    payload = _done(executor, id=1)
    assert "NOT done" in payload["error"]
    assert executor._result_status(payload) == "error"


def test_an_unknown_id_is_reported_as_an_error(executor):
    executor.set_tasks(["one"])
    payload = _done(executor, id=99)
    assert "No task with id 99" in payload["error"]
    assert executor._result_status(payload) == "error"


def test_blocking_is_reported_as_an_error_exactly_once(executor):
    """The blocking call itself failed; afterwards the item is settled, so
    repeating it must NOT keep producing errors (that was the livelock)."""
    executor.set_tasks([{"text": "x", "verify": lambda: False}])
    statuses = [executor._result_status(_done(executor, id=1)) for _ in range(6)]
    assert statuses[:_MAX_TASK_VERIFY_ATTEMPTS] == ["error"] * _MAX_TASK_VERIFY_ATTEMPTS
    assert set(statuses[_MAX_TASK_VERIFY_ATTEMPTS:]) == {"ok"}


def test_a_partial_batch_counts_as_progress(executor):
    """Some work was accepted, so the turn is not a failure."""
    executor.set_tasks(["real", {"text": "unverifiable", "verify": lambda: False}])
    payload = _done(executor, ids=[1, 2])
    assert payload["done_ids"] == [1]
    assert payload["refused"][0]["id"] == 2
    assert "error" not in payload
    assert executor._result_status(payload) == "ok"
