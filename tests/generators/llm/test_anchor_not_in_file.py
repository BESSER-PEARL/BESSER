"""A modify_file anchor that exists nowhere must be named as such, and a
byte-identical resend must be stopped on its second attempt.

Live run 0c537a4e (2026-09-18, Nebius Qwen/Qwen3-30B-A3B-Instruct-2507):
13 modify_file misses on routers/booking_methods.py, turns 21-50, every one
sending the same 206-char old_text - a ``def computeAmountOwed`` stub that
was never in the file. The file holds a CALL to that helper (the model wrote
it five turns earlier) and no definition. Each reply was the generic
"make sure old_text matches exactly including whitespace/indentation";
none carried ``did_you_mean`` (best window ratio 0.43 against the 0.6
threshold); the 3-miss stop fired at attempts 4, 8 and 12 and reset itself
each time; the model re-read the file 16 times in between.

The fixture is that run's file, verbatim; the anchor is the trace's.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from besser.generators.llm.tool_executor import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "run_0c537a4e"
ROUTER = "web_app/backend/routers/booking_methods.py"

# Byte-identical across all 13 attempts (md5 b91f3104).
REAL_OLD_TEXT = (
    'def computeAmountOwed(booking_id: int, database: Session) -> float:\n'
    '    """Helper function to compute the total amount owed for a booking."""\n'
    '    # This function is called by the produceBill method\n'
    '    pass'
)
NEW_TEXT = REAL_OLD_TEXT.replace("    pass", "    return 0.0")

# A real line with one token changed: the near-miss shape did_you_mean exists for.
NEAR_MISS = '        total_amount_owed = computeAmountOwed(db_booking, database)\n'


@pytest.fixture
def executor(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    ex = ToolExecutor(workspace=str(tmp_path))
    ex.execute("read_file", {"path": ROUTER})
    return ex


def _modify(ex, old, new=NEW_TEXT):
    return json.loads(ex.execute("modify_file", {"path": ROUTER, "old_text": old, "new_text": new}))


def test_an_anchor_nothing_resembles_is_not_blamed_on_whitespace(executor):
    res = _modify(executor, REAL_OLD_TEXT)
    err = res["error"]
    assert "did_you_mean" not in res
    assert "whitespace" not in err.lower(), err
    assert "resembles" in err
    assert "def computeAmountOwed" in err, "name the definition that does not exist"


def test_it_says_how_to_add_code_without_suggesting_a_rewrite(executor):
    """The recovery is an edit anchored on a real line. It must not say
    write_file: that wording turned targeted edits into whole-file rewrites
    of scaffold code (2026-09-17, see _build_modify_loop_reminder)."""
    err = _modify(executor, REAL_OLD_TEXT)["error"]
    assert "old_text = " in err and "new_text = " in err
    assert "write_file" not in err


def test_a_byte_identical_resend_is_stopped_on_the_second_attempt(executor):
    first = _modify(executor, REAL_OLD_TEXT)
    second = _modify(executor, REAL_OLD_TEXT)
    assert second["error"] != first["error"]
    assert "already" in second["error"] and "cannot succeed" in second["error"]


def test_a_different_anchor_is_not_called_a_resend(executor):
    _modify(executor, REAL_OLD_TEXT)
    res = _modify(executor, "def somethingElse():\n    pass")
    assert "already" not in res["error"]


def test_a_near_miss_keeps_did_you_mean_and_the_exact_match_advice(executor):
    res = _modify(executor, NEAR_MISS, NEAR_MISS.replace("db_booking,", "db_booking.id,"))
    assert "did_you_mean" in res and "computeAmountOwed(db_booking.id, database)" in res["did_you_mean"]
    assert "resembles" not in res["error"]


def test_identical_resends_still_count_toward_the_hard_stop(executor):
    for _ in range(3):
        assert "error" in _modify(executor, REAL_OLD_TEXT)
    assert "refused" in _modify(executor, REAL_OLD_TEXT)["error"]


def test_a_successful_edit_clears_the_resend_memory(executor):
    _modify(executor, REAL_OLD_TEXT)
    real = '        # Compute the total amount owed\n'
    assert _modify(executor, real, '        # Compute what is owed\n').get("status") == "modified"
    # The same missing anchor after real progress is a fresh miss, not a resend.
    assert "already" not in _modify(executor, REAL_OLD_TEXT)["error"]
