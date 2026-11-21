"""An argument a tool does not declare is reported in its result, not dropped silently.

The executor passed undeclared keys through and every handler ignored them, so
intent the model stated was lost without a word: ``old_text`` on a range edit
(where it would have shown the range the model meant), or a ``path`` scope on
``search_in_files``, which then searched the whole workspace.
"""

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import with_scope_widening, FILE_TOOLS


def _executor(tmp_path):
    (tmp_path / "app.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


def test_old_text_on_a_range_edit_is_reported(tmp_path):
    executor = _executor(tmp_path)
    read = executor.execute_typed("read_file", {"path": "app.py"}).payload

    result = executor.execute_typed("replace_file_lines", {
        "path": "app.py", "read_id": read["read_id"], "start_line": 1, "end_line": 1,
        "new_text": "a = 10\n", "old_text": "a = 1\n"}).payload

    assert result.get("status") == "modified", result
    note = result["ignored_arguments"]
    assert "old_text" in note and "read_id" in note and "start_line" in note


def test_a_search_scope_the_tool_does_not_take_is_reported(tmp_path):
    result = _executor(tmp_path).execute_typed(
        "search_in_files", {"pattern": "a =", "path": "backend"}).payload

    assert "path" in result["ignored_arguments"], result
    assert "file_glob" in result["ignored_arguments"]


def test_declared_arguments_leave_no_note(tmp_path):
    executor = _executor(tmp_path)

    assert "ignored_arguments" not in executor.execute_typed(
        "search_in_files", {"pattern": "a =", "file_glob": "*.py"}).payload
    assert "ignored_arguments" not in executor.execute_typed(
        "read_file", {"path": "app.py", "offset": 0, "limit": 1}).payload


def test_the_scope_argument_of_a_limited_fix_attempt_is_not_reported(tmp_path):
    executor = _executor(tmp_path)
    assert "needed_for" in str(with_scope_widening(FILE_TOOLS))
    executor.execute_typed("read_file", {"path": "app.py"})

    result = executor.execute_typed("write_file", {
        "path": "app.py", "content": "a = 1\nb = 3\n", "needed_for": "blocker 1"}).payload

    assert "ignored_arguments" not in result, result
