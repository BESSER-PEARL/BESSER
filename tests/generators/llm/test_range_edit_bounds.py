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

from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tool_executor import ToolExecutor


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
    """Recovery advice must arm from range failures, not only ``modify_file``."""
    for _ in range(2):
        result = _failing_range_edit(executor)
    assert result["edit_recovery"]["next_tool"] == "read_file"


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
