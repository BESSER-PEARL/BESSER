"""The checklist must not cost one turn per item, and must not livelock.

Measured across a 10-run live batch: 202 of 502 turns (40%) were
task_list calls; the worst run spent 71 of 86 turns there, ~62 CONSECUTIVE.
Two causes:

1. ``action='done'`` took a single ``id``, so completing N items cost N turns --
   and every turn pays a full prompt prefill, the binding cost on the
   self-hosted box.
2. Closing an item was refused until the model quoted code it had written, and
   a verifier that never passes was retried until the turn cap.

An item without a verifier now closes on the model's word. The verifier itself
is deliberately kept: it exists because a model once marked "build the
frontend" done without writing a single file. A refused item stays open, not
done; the harness re-runs its check after every edit, and repeated refusals
are bounded by the loop guard and Phase 2's no-progress handoff.
"""

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


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
    assert result["verified_ids"] == []
    assert result["unverified_ids"] == [1, 2, 3, 4]


def test_a_single_id_still_works(executor):
    """The old shape is what existing prompts and checkpoints use."""
    executor.set_tasks(["one", "two"])
    result = _done(executor, id=1)
    assert result["done_ids"] == [1]
    assert result["id"] == 1
    assert result["open_remaining"] == 1


def test_batching_is_what_the_tool_advertises(executor):
    from besser.spec_driven_agent.agent.tools import VALIDATION_TOOLS
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


def test_a_failing_check_is_refused_every_time_and_never_closed_or_blocked(executor):
    """Refusal is the cheat guard; blocking on a count of refusals is gone."""
    executor.set_tasks([{"text": "build the frontend", "verify": lambda: False}])
    for _ in range(5):
        result = _done(executor, id=1)
        assert result["refused"] == [{"id": 1, "reason": "its check still fails"}]
        assert "make the change" in result["advice"]
    task = executor._tasks[0]
    assert (task["done"], task["blocked"]) == (False, False)
    assert executor.open_tasks() == [task]


def test_a_check_that_starts_failing_and_then_passes_is_accepted(executor):
    """The normal case: the model does the work between attempts."""
    state = {"written": False}
    executor.set_tasks([{"text": "write it", "verify": lambda: state["written"]}])
    assert _done(executor, id=1)["refused"][0]["id"] == 1
    state["written"] = True
    assert _done(executor, id=1)["verified_ids"] == [1]
    assert executor.open_tasks() == []


def test_a_broken_verifier_is_a_refusal_not_a_success(executor):
    def explode():
        raise RuntimeError("verifier bug")
    executor.set_tasks([{"text": "x", "verify": explode}])
    result = _done(executor, id=1)
    assert result["done_ids"] == []
    assert "verification could not run" in result["refused"][0]["reason"]
    assert executor.open_tasks()


# ------------------------------------------------------- crash recovery


def test_a_blocked_item_stays_blocked_across_a_resume(executor, tmp_path):
    executor.set_tasks(["x"])
    executor._task_list({"action": "blocked", "id": 1, "reason": "no adapter"})
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


def test_every_refused_call_is_reported_as_an_error(executor):
    executor.set_tasks([{"text": "x", "verify": lambda: False}])
    statuses = [executor._result_status(_done(executor, id=1)) for _ in range(4)]
    assert statuses == ["error"] * 4


def test_a_partial_batch_counts_as_progress(executor):
    """Some work was accepted, so the turn is not a failure."""
    executor.set_tasks(["real", {"text": "unverifiable", "verify": lambda: False}])
    payload = _done(executor, ids=[1, 2])
    assert payload["done_ids"] == [1]
    assert payload["refused"][0]["id"] == 2
    assert "error" not in payload
    assert executor._result_status(payload) == "ok"


def test_an_unchecked_item_closes_on_the_models_word(executor, tmp_path):
    """No quote of the code is asked for: the harness checks the app itself."""
    executor.set_tasks(["Implement payment"])
    result = _done(executor, id=1)
    assert result["done_ids"] == [1]
    assert result["verified_ids"] == [] and result["unverified_ids"] == [1]
    assert executor._task_list({"action": "list"})["tasks"][0]["status"] == "implemented"
    restored = ToolExecutor(workspace=str(tmp_path))
    restored.restore_tasks(executor.task_snapshot())
    assert restored.task_snapshot() == executor.task_snapshot()


def test_explicit_blocked_reason_survives_listing_and_resume(executor, tmp_path):
    executor.set_tasks(["Implement payment"])
    assert "reason" in executor._task_list({"action": "blocked", "id": 1})["error"]
    result = executor._task_list({"action": "blocked", "ids": [1], "reason": "Payment adapter is unavailable"})
    assert result["blocked_ids"] == [1]
    restored = ToolExecutor(workspace=str(tmp_path))
    restored.restore_tasks(executor.task_snapshot())
    listing = restored._task_list({"action": "list"})["tasks"][0]
    assert listing["status"] == "blocked"
    assert "verification" not in listing  # implied by status
    assert restored.task_snapshot()[0]["verification"] == "unverified"
    assert listing["reason"] == "Payment adapter is unavailable"
    # The model may still close it once the work is done.
    repaired = _done(restored, id=1)
    assert repaired["done_ids"] == [1]
    assert repaired["unverified_ids"] == [1]
    assert restored.blocked_tasks() == []


# ======================================================================
# The other half: action='add'
# ======================================================================
#
# An earlier fix batched ``done`` and left ``add`` taking a single
# ``text``. In a later recorded run, turns 11-23 were THIRTEEN
# consecutive ``add`` calls building a 32-item checklist, one item per round
# trip, before a single line of code was written. The tool's own description
# says "Batch them: one task per call wastes a turn each" while offering no
# way to batch an add.


def test_many_items_are_added_in_one_call(executor):
    """The 4efe04ff shape: 13 turns of bookkeeping become 1."""
    res = executor._task_list({"action": "add", "texts": [
        "Implement booking_registerArrival in routers/booking_methods.py",
        "Implement booking_registerDeparture in routers/booking_methods.py",
        "Implement bill_registerPayment in routers/bill_methods.py",
    ]})
    assert res.get("status") == "added", res
    assert res.get("ids") == [1, 2, 3], res
    assert len(executor.open_tasks()) == 3


def test_a_single_text_still_works(executor):
    """The existing call shape must not break."""
    res = executor._task_list({"action": "add", "text": "One item"})
    assert res.get("status") == "added"
    assert res.get("id") == 1
    assert len(executor.open_tasks()) == 1


def test_a_duplicate_inside_a_batch_reuses_its_id(executor):
    """Dedupe already applies per item; a batch must not bypass it."""
    executor._task_list({"action": "add", "text": "Wire the Bill screen"})
    res = executor._task_list({"action": "add", "texts": [
        "Wire the Bill screen.",          # same item, trailing period
        "Wire the Booking screen",
    ]})
    assert res.get("ids") == [1, 2], res
    assert len(executor.open_tasks()) == 2, "the duplicate must not add a second copy"


def test_blank_entries_in_a_batch_are_ignored_not_added(executor):
    res = executor._task_list({"action": "add", "texts": ["Real item", "", "   "]})
    assert res.get("ids") == [1], res
    assert len(executor.open_tasks()) == 1


def test_an_empty_batch_is_refused_clearly(executor):
    for args in ({"action": "add", "texts": []}, {"action": "add", "texts": ["", " "]}):
        res = executor._task_list(args)
        assert "error" in res, (args, res)


def test_the_tool_advertises_batched_adds(executor):
    """A capability the description does not mention is a capability unused."""
    from besser.spec_driven_agent.agent.tools import VALIDATION_TOOLS
    spec = next(t for t in VALIDATION_TOOLS if t["name"] == "task_list")
    assert "texts" in spec["input_schema"]["properties"], spec["input_schema"]["properties"].keys()
    assert "texts" in spec["description"]
