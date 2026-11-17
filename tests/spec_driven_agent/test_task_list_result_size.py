"""Every tool result stays in the conversation and is re-sent on each later turn.

Before this, one ``done`` reprinted every open task's full text (4.6k chars for
10 recorded-size tasks) and a refusal held its task's text three times. Results
that only need to name an item now carry a short label; the full text and the
planning notes are on ``list``, for the items whose work is still owed.
"""

import json

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor, _TASK_LABEL_CHARS, _task_label

LONG = ("Implement POST /bill/{bill_id}/methods/registerPayment/ in web_app/backend/routers/"
        "bill_methods.py, function execute_bill_registerPayment, according to the specification")
NOTE = "The scaffold returns HTTP 501 here; keep the route signature."


def _executor(tmp_path, count=3, passes=lambda i: False):
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_tasks([{"text": LONG.replace("bill", f"bill{i}"), "planning_notes": [NOTE],
                         "verify": (lambda i=i: passes(i))} for i in range(count)])
    return executor


def _call(executor, **args):
    return json.loads(executor.execute("task_list", args))


def test_a_label_is_cut_at_a_word_boundary():
    label = _task_label(LONG)
    assert len(label) <= _TASK_LABEL_CHARS + 3
    assert label.endswith("...")
    assert LONG.startswith(label[:-3]) and LONG[len(label) - 3] == " "
    assert _task_label("Add auth") == "Add auth"


def test_done_lists_what_is_open_by_label_only(tmp_path):
    executor = _executor(tmp_path, passes=lambda i: i == 0)
    result = _call(executor, action="done", ids=[1])
    assert result["done_ids"] == [1]
    assert result["open_items"] == [{"id": 2, "label": _task_label(LONG.replace("bill", "bill1"))},
                                    {"id": 3, "label": _task_label(LONG.replace("bill", "bill2"))}]
    assert NOTE not in json.dumps(result)


def test_a_refusal_does_not_repeat_the_task_text(tmp_path):
    executor = _executor(tmp_path)
    for _ in range(3):  # refused, refused, then blocked
        raw = executor.execute("task_list", {"action": "done", "ids": [1]})
        assert "execute_bill0_registerPayment" not in raw
    assert executor.blocked_tasks()[0]["id"] == 1
    already = executor.execute("task_list", {"action": "done", "ids": [1]})
    assert "execute_bill0_registerPayment" not in already


def test_list_keeps_full_text_for_owed_work_only(tmp_path):
    executor = _executor(tmp_path, passes=lambda i: i == 0)
    _call(executor, action="done", ids=[1])
    _call(executor, action="blocked", ids=[2], reason="adapter missing")
    done, blocked, open_ = _call(executor, action="list")["tasks"]
    assert done == {"id": 1, "label": _task_label(LONG.replace("bill", "bill0")), "status": "done"}
    assert blocked == {"id": 2, "text": LONG.replace("bill", "bill1"), "status": "blocked",
                       "reason": "adapter missing"}
    assert open_ == {"id": 3, "text": LONG.replace("bill", "bill2"), "status": "open",
                     "planning_notes": [NOTE]}


def test_an_auto_closed_item_is_reported_by_label(tmp_path):
    executor = _executor(tmp_path, count=1, passes=lambda i: True)
    assert executor.autoclose_verified_tasks() == [{"id": 1, "label": _task_label(LONG.replace("bill", "bill0"))}]
