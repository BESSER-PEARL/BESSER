"""A modify_file call whose old_text equals its new_text is a rejection like
any other: it must count, be fingerprinted, and stay refused once stopped.

Live run 57160293 (2026-09-18, Nebius Qwen3-30B-A3B-Instruct): after one
successful edit on routers/booking_methods.py the model sent that edit's
new_text as BOTH old_text and new_text, sixteen times over thirty turns
(read_file between every pair), and the executor answered "old_text and
new_text are identical; no edit was applied" every time. That branch returned
before the miss counter, so the 3-miss refusal, the resend stop and the
orchestrator's streak reminder never fired; the run spent $0.24 of $0.29 on
it. Run 0c537a4e showed the other half: the 3-miss refusal popped its own
counter, so one anchor came back for four full cycles (refusals at attempts
4, 8 and 12).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

FIXTURE_0C53 = Path(__file__).parent / "fixtures" / "run_0c537a4e"
ROUTER = "web_app/backend/routers/booking_methods.py"
PHANTOM = (
    'def computeAmountOwed(booking_id: int, database: Session) -> float:\n'
    '    """Helper function to compute the total amount owed for a booking."""\n'
    '    # This function is called by the produceBill method\n'
    '    pass'
)

SRC = "import os\n\n\ndef total(a, b):\n    return a + b\n\n\ndef main():\n    print(total(1, 2))\n"
BLOCK = "def total(a, b):\n    return a + b\n"


@pytest.fixture
def ex(tmp_path):
    (tmp_path / "app.py").write_text(SRC, encoding="utf-8")
    e = ToolExecutor(workspace=str(tmp_path))
    e.execute("read_file", {"path": "app.py"})
    return e


def _modify(e, old, new, path="app.py"):
    return json.loads(e.execute("modify_file", {"path": path, "old_text": old, "new_text": new}))


def test_a_noop_call_counts_as_a_miss_and_says_not_to_resend(ex):
    res = _modify(ex, BLOCK, BLOCK)
    assert "error" in res
    assert "identical" in res["error"]
    assert "resend" in res["error"].lower()
    assert ex.consecutive_modify_misses("app.py") == 1


def test_a_noop_whose_text_is_already_in_the_file_says_where(ex):
    res = _modify(ex, BLOCK, BLOCK)
    assert "already in the file" in res["error"]
    assert "line 4" in res["error"]


def test_the_second_identical_noop_is_named_a_repeat(ex):
    first = _modify(ex, BLOCK, BLOCK)
    second = _modify(ex, BLOCK, BLOCK)
    assert second["error"] != first["error"]
    assert "already" in second["error"] and "cannot succeed" in second["error"]


def test_identical_noops_reach_the_hard_stop_and_it_does_not_reset(ex):
    for _ in range(3):
        assert "error" in _modify(ex, BLOCK, BLOCK)
    for attempt in range(4, 8):
        res = _modify(ex, BLOCK, BLOCK)
        assert "refused" in res["error"], f"attempt {attempt}: {res['error']}"


def test_the_stop_is_per_fingerprint_so_a_real_edit_still_lands(ex):
    for _ in range(4):
        _modify(ex, BLOCK, BLOCK)
    res = _modify(ex, "    print(total(1, 2))\n", "    print(total(2, 3))\n")
    assert res.get("status") == "modified", res


def test_repeats_are_reported_for_the_orchestrator(ex):
    assert ex.repeat_rejections("app.py") == 0
    _modify(ex, BLOCK, BLOCK)
    assert ex.last_repeat is None
    _modify(ex, BLOCK, BLOCK)
    assert ex.last_repeat == ("app.py", 2)
    assert ex.repeat_rejections("app.py") == 2
    _modify(ex, "    print(total(1, 2))\n", "    print(total(2, 3))\n")
    assert ex.last_repeat is None, "a different call is not a repeat"


def test_a_successful_edit_clears_the_repeat_memory(ex):
    _modify(ex, BLOCK, BLOCK)
    _modify(ex, BLOCK, BLOCK)
    assert _modify(ex, "import os\n", "import os\nimport sys\n").get("status") == "modified"
    assert ex.repeat_rejections("app.py") == 0
    assert "cannot succeed" not in _modify(ex, BLOCK, BLOCK)["error"], "a fresh miss, not a repeat"


def test_a_frozen_path_refuses_every_edit_but_reads_still_work(ex):
    ex.freeze_path("app.py", "closed by the orchestrator")
    res = _modify(ex, "import os\n", "import sys\n")
    assert "closed" in res["error"]
    assert "closed by the orchestrator" in res["error"]
    written = json.loads(ex.execute("write_file", {"path": "app.py", "content": "x = 1\n"}))
    assert "closed" in written["error"]
    assert (Path(ex.workspace) / "app.py").read_text(encoding="utf-8") == SRC
    assert "content" in json.loads(ex.execute("read_file", {"path": "app.py"}))


def test_the_0c537a4e_anchor_stays_refused_after_the_first_stop(tmp_path):
    """Four cycles live: refusals at 4, 8, 12 with fresh 3-strike windows in
    between. Now the fifth attempt is refused like the fourth."""
    shutil.copytree(FIXTURE_0C53, tmp_path, dirs_exist_ok=True)
    e = ToolExecutor(workspace=str(tmp_path))
    e.execute("read_file", {"path": ROUTER})
    for _ in range(3):
        assert "error" in _modify(e, PHANTOM, PHANTOM.replace("pass", "return 0.0"), ROUTER)
    for attempt in range(4, 9):
        res = _modify(e, PHANTOM, PHANTOM.replace("pass", "return 0.0"), ROUTER)
        assert "refused" in res["error"], f"attempt {attempt}: {res['error']}"
