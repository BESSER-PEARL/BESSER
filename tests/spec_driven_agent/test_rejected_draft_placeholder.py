"""The placeholder that stands in for a refused draft is never written as code.

The conversation shows a refused edit's body as a placeholder so the model does
not copy the rejected candidate again. A model copied the placeholder itself
into new edits instead, and the write tools accepted it wherever no parse
check ran: in a stylesheet, ``replace_file_lines`` and ``modify_file`` both
landed it on disk; only ``.py``/``.tsx`` files refused it, by accident of their
syntax check.
"""

import pytest

from besser.spec_driven_agent.agent.history_eviction import (
    REJECTED_DRAFT_PLACEHOLDER,
    without_rejected_edit_drafts,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

STYLES = ".card {\n  color: teal;\n}\n.list {\n  margin: 0;\n}\n"


@pytest.fixture
def executor(tmp_path):
    (tmp_path / "styles.css").write_text(STYLES, encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


def _range_edit(executor, text):
    read = executor.execute_typed("read_file", {"path": "styles.css"}).payload
    return executor.execute_typed("replace_file_lines", {
        "path": "styles.css", "read_id": read["read_id"], "start_line": 2, "end_line": 2,
        "new_text": text}).payload


@pytest.mark.parametrize("tool", ["write_file", "modify_file", "replace_file_lines"])
def test_the_placeholder_is_refused_in_any_file_type(executor, tmp_path, tool):
    executor.execute_typed("read_file", {"path": "styles.css"})
    if tool == "write_file":
        result = executor.execute_typed(tool, {"path": "styles.css",
                                               "content": STYLES + REJECTED_DRAFT_PLACEHOLDER}).payload
    elif tool == "modify_file":
        result = executor.execute_typed(tool, {"path": "styles.css", "old_text": "  color: teal;",
                                               "new_text": REJECTED_DRAFT_PLACEHOLDER}).payload
    else:
        result = _range_edit(executor, REJECTED_DRAFT_PLACEHOLDER)

    assert result.get("rejection_kind") == "placeholder", result
    assert (tmp_path / "styles.css").read_text(encoding="utf-8") == STYLES


def test_the_placeholder_is_what_the_conversation_shows():
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "write_file",
                                           "input": {"path": "a.css", "content": "draft"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": '{"error": "refused"}'}]},
    ]
    shown = without_rejected_edit_drafts(messages)[0]["content"][0]["input"]["content"]
    assert shown == REJECTED_DRAFT_PLACEHOLDER
