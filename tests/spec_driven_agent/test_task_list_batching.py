"""The checklist must not cost one turn per item, and must not livelock.

Measured across a 10-run live batch on 2026-09-11: 202 of 502 turns (40%) were
task_list calls; the worst run spent 71 of 86 turns there, ~62 CONSECUTIVE.
Two causes, both fixed here:

1. ``action='done'`` took a single ``id``, so completing N items cost N turns --
   and every turn pays a full prompt prefill, the binding cost on the
   self-hosted box.
2. A verifier that never passes was retried forever; only the 80-turn cap broke
   the cycle, ending that run INCOMPLETE on 317s of its 2400s budget.

The verifier itself is deliberately kept: it exists because a model once marked
"build the frontend" done without writing a single file. A blocked item is
therefore NOT marked done -- it is recorded, excluded from the open set so the
run can finish, and reported.
"""

import pytest

from besser.spec_driven_agent.tool_executor import (
    _MAX_TASK_VERIFY_ATTEMPTS,
    ToolExecutor,
)


@pytest.fixture
def executor(tmp_path):
    return ToolExecutor(workspace=str(tmp_path))


def _done(executor, **kwargs):
    return executor._task_list(dict(action="done", **kwargs))


def _record_changes(executor, ids):
    """Real modification receipts for tests of batching, not acceptance proof."""
    evidence = []
    for task_id in ids:
        path, content = f"task_{task_id}.py", f"completed_item_{task_id} = True\n"
        assert executor._write_file({"path": path, "content": content})["status"] == "written"
        evidence.append({"id": task_id, "path": path, "quote": content})
    return evidence


# ----------------------------------------------------------- batching


def test_many_items_are_marked_done_in_one_call(executor):
    executor.set_tasks(["one", "two", "three", "four"])
    result = _done(executor, ids=[1, 2, 3, 4], evidence=_record_changes(executor, [1, 2, 3, 4]))
    assert result["done_ids"] == [1, 2, 3, 4]
    assert result["open_remaining"] == 0
    assert result["verified_ids"] == []
    assert result["unverified_ids"] == [1, 2, 3, 4]


def test_a_single_id_still_works(executor):
    """The old shape is what existing prompts and checkpoints use."""
    executor.set_tasks(["one", "two"])
    result = _done(executor, id=1, evidence=_record_changes(executor, [1]))
    assert result["done_ids"] == [1]
    assert result["id"] == 1
    assert result["open_remaining"] == 1


def test_batching_is_what_the_tool_advertises(executor):
    from besser.spec_driven_agent.tools import VALIDATION_TOOLS
    spec = next(t for t in VALIDATION_TOOLS if t["name"] == "task_list")
    assert "ids" in spec["input_schema"]["properties"]
    assert spec["input_schema"]["properties"]["ids"]["type"] == "array"
    # The model only batches if the description tells it to.
    assert "ids=[1,2,3]" in spec["description"]


def test_unknown_and_duplicate_ids_do_not_derail_the_batch(executor):
    executor.set_tasks(["one", "two"])
    result = _done(executor, ids=[1, 1, 99], evidence=_record_changes(executor, [1]))
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
    _done(executor, id=2, evidence=_record_changes(executor, [2]))
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
    # Repairs performed after automatic blocking can still satisfy the check.
    state["written"] = False
    executor.set_tasks([{"text": "write it", "verify": lambda: state["written"]}])
    for _ in range(_MAX_TASK_VERIFY_ATTEMPTS):
        _done(executor, id=1)
    assert executor.blocked_tasks()
    state["written"] = True
    assert _done(executor, id=1)["verified_ids"] == [1]
    assert executor.blocked_tasks() == []
    assert "blocked_reason" not in executor.task_snapshot()[0]


def test_a_broken_verifier_never_wedges_the_run(executor):
    def explode():
        raise RuntimeError("verifier bug")
    executor.set_tasks([{"text": "x", "verify": explode}])
    for _ in range(_MAX_TASK_VERIFY_ATTEMPTS):
        result = _done(executor, id=1)
        assert result["done_ids"] == []
        assert "error" in result
    assert "verification could not run" in result["blocked"][0]["reason"]
    assert executor.blocked_tasks()[0]["done"] is False
    repeated = _done(executor, id=1)
    assert repeated["status"] == "blocked"
    assert repeated["done_ids"] == []


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
    payload = _done(executor, ids=[1, 2], evidence=_record_changes(executor, [1]))
    assert payload["done_ids"] == [1]
    assert payload["refused"][0]["id"] == 2
    assert "error" not in payload
    assert executor._result_status(payload) == "ok"


def test_unverified_completion_needs_current_write_evidence(executor, tmp_path):
    executor.set_tasks(["Implement payment"])
    (tmp_path / "payment.py").write_text("paid = True\n", encoding="utf-8")
    evidence = [{"id": 1, "path": "payment.py", "quote": "paid = True"}]
    # Existing source and read-only activity do not prove this task was worked on.
    executor._read_file({"path": "payment.py"})
    assert _done(executor, id=1, evidence=evidence)["done_ids"] == []
    executor._modify_file({"path": "payment.py", "old_text": "paid = True", "new_text": "paid = False"})
    evidence[0]["quote"] = "paid = False"
    result = _done(executor, id=1, evidence=evidence)
    assert result["done_ids"] == [1]
    assert result["verified_ids"] == []
    assert executor._task_list({"action": "list"})["tasks"][0]["status"] == "implemented"
    restored = ToolExecutor(workspace=str(tmp_path))
    restored.restore_tasks(executor.task_snapshot())
    assert restored.unverified_tasks()[0]["implementation_evidence"] == executor.unverified_tasks()[0]["implementation_evidence"]


def test_existing_completion_requires_read_executable_evidence(executor, tmp_path):
    source = (
        "# Payment is implemented\n"
        "def register_payment(bill, db):\n"
        "    if bill.paid:\n"
        "        return False\n"
        "    bill.paid = True\n"
        "    db.commit()\n"
        "    return True\n\n"
        "def placeholder():\n"
        "    raise NotImplementedError\n"
        "\nclass Room:\n"
        "    roomNumber = mapped_column(String(100), unique=True)\n"
    )
    path = tmp_path / "payment.py"
    path.write_text(source, encoding="utf-8")
    evidence = [{"id": 1, "path": "payment.py", "quote": "bill.paid = True"}]
    executor.set_tasks(["Implement payment"])
    unread = _done(executor, id=1, existing=True, evidence=evidence)
    assert unread["done_ids"] == []
    assert "read payment.py" in unread["refused"][0]["reason"]
    executor._read_file({"path": "payment.py"})

    for quote in ("# Payment is implemented", "raise NotImplementedError", "def placeholder():", "bill.total = 0"):
        executor.set_tasks(["Implement payment"])
        refused = _done(executor, id=1, existing=True, evidence=[
            {"id": 1, "path": "payment.py", "quote": quote},
        ])
        assert refused["done_ids"] == [], (quote, refused)
        assert executor.open_tasks()

    executor.set_tasks(["Implement payment"])
    accepted = _done(executor, id=1, existing=True, evidence=evidence)
    assert accepted["done_ids"] == [1]
    assert accepted["verified_ids"] == []
    assert accepted["unverified_ids"] == [1], "source evidence is not behavioral acceptance"
    receipt = executor.unverified_tasks()[0]["implementation_evidence"][0]
    assert receipt["origin"] == "existing"
    assert receipt["quote"] == evidence[0]["quote"]
    assert not executor._successful_writes
    assert path.read_text(encoding="utf-8") == source
    # Planning prose is not automatically an action. Database declarations
    # are existing implementation evidence, never behavioral acceptance.
    for task in ("Enforce unique room numbers",
                 {"text": "Enforce unique room numbers", "kind": "uniqueness"}):
        executor.set_tasks([task])
        accepted = _done(executor, id=1, existing=True, evidence=[{
            "id": 1, "path": "payment.py",
            "quote": "roomNumber = mapped_column(String(100), unique=True)",
        }])
        assert accepted["done_ids"] == [1], accepted
        assert accepted["verified_ids"] == []
        assert accepted["unverified_ids"] == [1]
        restored = ToolExecutor(workspace=str(tmp_path))
        restored.restore_tasks(executor.task_snapshot())
        assert restored.task_snapshot() == executor.task_snapshot()
    executor.set_tasks([{"text": "Implement payment", "kind": "action"}])
    refused = _done(executor, id=1, existing=True, evidence=[{
        "id": 1, "path": "payment.py", "quote": "roomNumber = mapped_column(String(100), unique=True)",
    }])
    assert refused["done_ids"] == []
    assert "does not perform the required behaviour" in refused["refused"][0]["reason"]


def test_explicit_blocked_reason_survives_listing_and_resume(executor, tmp_path):
    executor.set_tasks(["Implement payment"])
    assert "reason" in executor._task_list({"action": "blocked", "id": 1})["error"]
    result = executor._task_list({"action": "blocked", "ids": [1], "reason": "Payment adapter is unavailable"})
    assert result["blocked_ids"] == [1]
    assert _done(executor, id=1)["done_ids"] == []
    restored = ToolExecutor(workspace=str(tmp_path))
    restored.restore_tasks(executor.task_snapshot())
    listing = restored._task_list({"action": "list"})["tasks"][0]
    assert listing["status"] == "blocked"
    assert listing["verification"] == "unverified"
    assert listing["reason"] == "Payment adapter is unavailable"
    # An explicit block also closes only on fresh implementation evidence.
    repaired = _done(restored, id=1, evidence=_record_changes(restored, [1]))
    assert repaired["done_ids"] == [1]
    assert repaired["unverified_ids"] == [1]
    assert restored.blocked_tasks() == []


# ======================================================================
# The other half: action='add'
# ======================================================================
#
# The 2026-09-11 fix batched ``done`` and left ``add`` taking a single
# ``text``. Live run 4efe04ff, 2026-09-18: turns 11-23 were THIRTEEN
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
    from besser.spec_driven_agent.tools import VALIDATION_TOOLS
    spec = next(t for t in VALIDATION_TOOLS if t["name"] == "task_list")
    assert "texts" in spec["input_schema"]["properties"], spec["input_schema"]["properties"].keys()
    assert "texts" in spec["description"]
