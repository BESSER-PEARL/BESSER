"""``replace_file_lines`` must be bounded by the same guards as ``modify_file``.

The quotation-free range editor was added to break the Qwen "re-quote a
failing old_text forever" loop. Measured against the executor before this
file existed, it introduced a worse loop of its own: eight identical
range edits that each broke the syntax guard left ``last_repeat`` at
``None`` and ``_edit_recovery`` unset, so nothing ever escalated. The same
change deleted ``_REPEAT_FREEZE_AT``, and a *failed* range edit reset
``last_repeat`` — so alternating a bad ``modify_file`` with a bad
``replace_file_lines`` could never reach ``_REPEAT_STOP_AT`` either. Net
effect: no terminal at all for edit flailing, only the turn and cost caps.
"""

import os
import tempfile

import pytest

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


SOURCE = "# keep\nasync def action():\n    return False\n\ndef unrelated():\n    return False\n"
# Unbalanced bracket on the first replacement line: always refused by the
# syntax guard, so the file never changes and the read_id stays valid.
BROKEN = "  @bad(\nasync def action():\n    return True\n"


@pytest.fixture
def executor():
    workspace = tempfile.mkdtemp()
    with open(os.path.join(workspace, "app.py"), "w", encoding="utf-8") as handle:
        handle.write(SOURCE)
    return ToolExecutor(workspace=workspace)


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def _failing_range_edit(executor, new_text=BROKEN):
    read = call(executor, "read_file", path="app.py", offset=1, limit=2)
    return call(executor, "replace_file_lines", path="app.py", read_id=read["read_id"],
                start_line=2, end_line=3, new_text=new_text)


def test_identical_failing_range_edits_escalate_to_a_terminal(executor):
    """Repeating one refused range edit must reach the stop threshold."""
    for _ in range(LLMOrchestrator._REPEAT_STOP_AT):
        result = _failing_range_edit(executor)
        assert result["rejection_kind"] == "syntax_error"

    path, seen = executor.last_repeat
    assert path == "app.py"
    assert seen >= LLMOrchestrator._REPEAT_STOP_AT, (
        f"{seen} repeats recorded; _escalate_repeat_rejection can never end the phase"
    )
    # The file is untouched throughout: nothing was silently applied.
    assert executor.consecutive_modify_misses("app.py") >= LLMOrchestrator._PER_FILE_MODIFY_THRESHOLD


def test_a_failed_range_edit_keeps_a_live_modify_escalation(executor):
    for _ in range(3):
        call(executor, "modify_file", path="app.py", old_text="NOT_IN_THE_FILE", new_text="x")
    assert executor.last_repeat == ("app.py", 3)

    _failing_range_edit(executor)

    assert executor.last_repeat is not None, (
        "a refused range edit erased the modify_file escalation, so alternating "
        "the two tools never reaches _REPEAT_STOP_AT"
    )


def test_a_successful_range_edit_clears_the_escalation(executor):
    for _ in range(3):
        _failing_range_edit(executor)
    assert executor.last_repeat is not None

    read = call(executor, "read_file", path="app.py", offset=1, limit=2)
    result = call(executor, "replace_file_lines", path="app.py", read_id=read["read_id"],
                  start_line=2, end_line=3,
                  new_text="async def action():\n    return True\n")

    assert result["status"] == "modified", result
    assert executor.last_repeat is None
    assert executor.consecutive_modify_misses("app.py") == 0
    assert executor.repeat_rejections("app.py") == 0


def test_a_refused_range_edit_arms_the_strategy_hint(executor):
    """Recovery advice must arm from range failures, not only ``modify_file``.

    The tier it arms changed on 2026-09-20: two refusals used to buy a
    forced ``read_file`` on the way to another range edit, and now buy the
    whole-file rewrite. The property this test exists for is the same one -
    ``replace_file_lines`` refusals feed the same counter ``modify_file``
    refusals do - and the count is still two, not one.
    """
    first = _failing_range_edit(executor)
    assert "edit_recovery" not in first, "one refusal is not an escalation"

    result = _failing_range_edit(executor)

    assert result["edit_recovery"]["next_tool"] == "write_file"
    assert "read_file on the WHOLE file" in result["edit_recovery"]["instruction"]


def test_range_edits_count_toward_the_per_file_streak_guard(simple_library_book_model, tmp_path):
    """A range-edit flail must trip the same per-file guard as a modify flail.

    Before the fix ``replace_file_lines`` was recorded with ``path=None``,
    which *broke* the streak — and the recovery ladder steers the model
    straight into that tool, so reaching recovery disarmed the guard.
    """
    class Client:
        model = "mock-model"

        def chat(self, **kwargs):  # never called by this test
            raise AssertionError("no LLM call expected")

    orchestrator = LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False,
    )
    n = orchestrator._PER_FILE_MODIFY_THRESHOLD
    orchestrator._recent_modify_targets = [("replace_file_lines", "app.py")] * n
    orchestrator.executor._failed_modifies["app.py"] = n

    assert orchestrator._consecutive_modify_on_same_file() == "app.py"


def test_the_offset_off_by_one_is_named_not_just_refused(executor):
    """read_file's offset is a 0-based skip; the printed numbers are 1-based.

    Run 7aybctis read offset=50, asked for start_line=50 against a 51-108
    view, was refused with the range restated, and sent the identical call
    again on the next turn.
    """
    read = call(executor, "read_file", path="app.py", offset=1, limit=3)
    assert read["start_line"] == 2

    result = call(executor, "replace_file_lines", path="app.py", read_id=read["read_id"],
                  start_line=1, end_line=3, new_text="x = 1\n")

    assert result["rejection_kind"] == "unread_range"
    assert "0-based skip" in result["error"], result["error"]
    assert result["displayed_start_line"] == 2


def test_an_ordinary_out_of_range_selection_gets_no_off_by_one_hint(executor):
    read = call(executor, "read_file", path="app.py", offset=1, limit=2)

    result = call(executor, "replace_file_lines", path="app.py", read_id=read["read_id"],
                  start_line=2, end_line=99, new_text="x = 1\n")

    assert result["rejection_kind"] == "unread_range"
    assert "0-based skip" not in result["error"]


def test_the_ladder_stops_steering_into_range_edits_when_they_keep_failing(executor):
    """modify_file lands 86-92%; replace_file_lines 15-43%. The rescue tool is
    now the weaker one, and run se7k3zbx spent 17 of 20 range edits failing
    while the ladder steered back into it 29 times.

    Renamed from ``..._goes_back_to_text_...`` on 2026-09-20: the
    destination is no longer text quotation. The "smallest unique old_text"
    reversal it used to assert is genuinely obsolete - that tier sits below
    the two-refusal rewrite in ``_add_edit_recovery`` and can no longer be
    reached, because ``_range_edit_failures`` and ``_edit_recovery`` are
    incremented and cleared together, so three range refusals always imply
    at least two edit refusals. The obligation that survives is the one in
    the name: after repeated range failures the ladder must not point back
    at ``replace_file_lines``.
    """
    for _ in range(executor._RANGE_EDIT_GIVE_UP):
        result = _failing_range_edit(executor)

    assert result["edit_recovery"]["next_tool"] != "replace_file_lines"
    assert result["edit_recovery"]["next_tool"] == "write_file"
    assert "read_file on the WHOLE file" in result["edit_recovery"]["instruction"]
    assert "do not summarise, elide, or drop code" in result["edit_recovery"]["instruction"]


def test_a_read_stops_advertising_range_edits_once_they_are_exhausted(executor):
    for _ in range(executor._RANGE_EDIT_GIVE_UP):
        _failing_range_edit(executor)

    read = call(executor, "read_file", path="app.py", offset=1, limit=2)

    assert read.get("edit_recovery", {}).get("next_tool") != "replace_file_lines"


def test_one_success_re_enables_the_range_editor(executor):
    for _ in range(executor._RANGE_EDIT_GIVE_UP):
        _failing_range_edit(executor)
    read = call(executor, "read_file", path="app.py", offset=1, limit=2)
    call(executor, "replace_file_lines", path="app.py", read_id=read["read_id"],
         start_line=2, end_line=3, new_text="async def action():\n    return True\n")

    assert executor._range_edit_failures.get("app.py") is None


def test_the_range_editor_is_a_first_miss_aid_only(executor):
    """The forward ladder is what fixed the 0%-edit runs; keep its first rung.

    Renamed from ``test_two_failures_still_steer_toward_the_range_editor``
    on 2026-09-20. The range editor was not dropped - it is still what the
    FIRST miss buys, and that rung is asserted here rather than deleted -
    but it stopped being the destination for a file that keeps refusing
    edits. Two refusals now escalate past it to a whole-file rewrite, so a
    ``read_file`` on that path must no longer advertise another range edit:
    that advertisement is what run se7k3zbx followed 29 times.
    """
    first_miss = call(executor, "modify_file", path="app.py",
                      old_text="    async def action():\n        return False\n",
                      new_text="# replaced\n")
    assert first_miss["edit_recovery"]["next_tool"] == "replace_file_lines"

    for _ in range(2):
        _failing_range_edit(executor)
    read = call(executor, "read_file", path="app.py", offset=1, limit=2)

    assert read["edit_recovery"]["next_tool"] == "write_file"
