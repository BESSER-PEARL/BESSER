"""Checklist evidence must survive being copied out of a numbered read.

``read_file`` prefixes every line with ``NNN| `` so a quoted region lands on
``modify_file``'s line-number tier. The checklist's evidence verifier had no
such tier, so a model that copied the region it had just read was told its
quote "is not present" in a file that contained it.

Live run 7aybctis (Qwen, 2026-09-19): task 9 was genuinely implemented in
``sql_alchemy.py``, the model cited it three times straight from the numbered
read, and the task was recorded BLOCKED. Four tasks failed this way, burning
eleven turns — once edits started landing this became the run's top cost.
"""

import os
import tempfile

import pytest

from besser.spec_driven_agent.tool_executor import ToolExecutor


SOURCE = (
    "from sqlalchemy.orm import relationship\n"
    "\n"
    "class Booking(Base):\n"
    '    __tablename__ = "booking"\n'
    '    reserved = relationship("ReservedRoom", back_populates="booking")\n'
)


@pytest.fixture
def executor():
    workspace = tempfile.mkdtemp()
    with open(os.path.join(workspace, "models.py"), "w", encoding="utf-8") as handle:
        handle.write(SOURCE)
    executor = ToolExecutor(workspace=workspace)
    executor.execute_typed("task_list", {"action": "add", "text": "Wire the Booking relationship"})
    return executor


def _read_back(executor):
    """The exact numbered text the model sees, prefixes and all."""
    payload = executor.execute_typed("read_file", {"path": "models.py"}).payload
    return payload["content"]


def _mark_done(executor, quote, existing=True):
    return executor.execute_typed("task_list", {
        "action": "done", "id": 1, "existing": existing,
        "evidence": [{"id": 1, "path": "models.py", "quote": quote}],
    }).payload


def test_evidence_copied_from_a_numbered_read_is_accepted(executor):
    numbered = _read_back(executor)
    quoted = "\n".join(numbered.splitlines()[2:5])
    assert "| " in quoted, quoted  # the prefix really is in what the model sees

    result = _mark_done(executor, quoted)

    assert "error" not in result, result
    assert result.get("status") != "error"


def test_evidence_that_is_genuinely_absent_is_still_refused(executor):
    _read_back(executor)
    result = _mark_done(executor, " 12| class Invoice(Base):\n 13|     pass\n")

    error = str(result.get("error", "")).lower()
    assert "not done" in error, result
    assert "not found" in error or "not present" in error, result


def test_a_partially_numbered_quote_is_not_silently_rewritten(executor):
    """Only a uniform prefix on every non-blank line is a copied numbering."""
    _read_back(executor)
    result = _mark_done(executor, ' 5|     reserved = relationship("ReservedRoom")\nnot_a_real_line\n')

    assert "error" in result, result
