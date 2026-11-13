"""Tool arguments are checked against each tool's own ``input_schema`` before
dispatch: unambiguous mistakes are coerced, the rest are refused with a message
naming the argument, the expected type and the type received."""
import json
import os

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


@pytest.fixture
def executor(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\nx = 1\nc = 3\nd = 4\n", encoding="utf-8")
    ex = ToolExecutor(str(tmp_path), per_write_diagnostics=False, allow_shell=True)
    yield ex
    ex.close()


def _run(executor, tool, args):
    return json.loads(executor.execute(tool, args))


@pytest.mark.parametrize("tool, args, message", [
    ("write_file", {"path": "p.json", "content": {"name": "x"}}, "content must be a string, got an object"),
    ("modify_file", {"path": "a.py", "old_text": None, "new_text": "x"}, "old_text must be a string, got null"),
    ("modify_file", {"path": "a.py", "old_text": ["x = 1"], "new_text": "y"}, "old_text must be a string, got an array"),
    ("read_file", {"path": "a.py", "limit": True}, "limit must be an integer, got a boolean"),
    ("read_file", {"path": "a.py", "offset": 1.5}, "offset must be an integer, got a number"),
    ("run_command", {"command": "echo hi", "timeout": float("inf")}, "timeout must be an integer, got a number"),
    ("run_command", {"command": "echo hi", "timeout": float("nan")}, "timeout must be an integer, got a number"),
    ("modify_file", {"path": "a.py", "old_text": "c = 3", "new_text": "c", "replace_all": "yes"},
     "replace_all must be a boolean, got a string"),
    ("task_list", {"action": "finish"}, "action must be one of"),
])
def test_wrong_types_are_refused_by_name(executor, tool, args, message):
    result = _run(executor, tool, args)
    assert message in result["error"]
    assert "failed:" not in result["error"]


def test_an_explicit_null_for_an_optional_argument_means_absent(executor):
    assert _run(executor, "search_in_files", {"pattern": "c = 3", "file_glob": None})["total"] == 1
    assert "error" not in _run(executor, "read_file", {"path": "a.py", "offset": None, "limit": None})


def test_install_dependencies_null_working_dir_is_the_default(executor):
    result = _run(executor, "install_dependencies", {"working_dir": None})
    assert "join() argument" not in json.dumps(result)


def test_boolean_strings_are_coerced(executor):
    _run(executor, "read_file", {"path": "a.py"})
    result = _run(executor, "modify_file",
                  {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2", "replace_all": "true"})
    assert "error" not in result
    assert open(f"{executor.workspace}/a.py").read().count("x = 2") == 2


def test_integer_strings_and_integral_floats_are_coerced(executor):
    assert _run(executor, "read_file", {"path": "a.py", "limit": "2"})["end_line"] == 2
    assert _run(executor, "read_file", {"path": "a.py", "offset": 2.0, "limit": 1})["end_line"] == 3
    read = _run(executor, "read_file", {"path": "a.py"})
    result = _run(executor, "replace_file_lines", {
        "path": "a.py", "read_id": read["read_id"], "start_line": "3", "end_line": "3", "new_text": "c = 30",
    })
    assert "error" not in result
    assert "c = 30" in open(f"{executor.workspace}/a.py").read()


def test_a_lone_value_for_an_array_is_one_item(executor):
    executor.set_tasks(["first", "second"])
    result = _run(executor, "task_list", {"action": "drop", "id": "1", "reason": "not asked"})
    assert result["status"] == "dropped"
    assert _run(executor, "task_list", {"action": "add", "texts": "third"})["status"] == "added"


def test_invalid_json_arguments_still_report_the_parse_error(executor):
    from besser.spec_driven_agent.agent.tools import INVALID_ARGUMENTS_KEY
    result = _run(executor, "write_file", {INVALID_ARGUMENTS_KEY: "JSONDecodeError: x"})
    assert "not valid JSON" in result["error"]


def test_a_failed_tool_never_shows_the_server_path(executor):
    os.makedirs(os.path.join(executor.workspace, "dir1"))
    error = _run(executor, "write_file", {"path": "dir1", "content": "x"})["error"]
    assert executor.workspace not in error
    # The link-safe open refuses anything that is not a regular file.
    assert error == "write_file failed: not a regular file: dir1"


def test_an_os_error_outside_the_workspace_names_only_the_file(executor, monkeypatch):
    def failing(self, args):
        raise PermissionError(13, "Permission denied", "/usr/lib/secret/tool.cfg")
    monkeypatch.setitem(ToolExecutor._handlers, "list_files", failing)
    error = _run(executor, "list_files", {})["error"]
    assert error == "list_files failed: Permission denied: tool.cfg"


def test_other_errors_drop_the_workspace_prefix(executor, monkeypatch):
    def failing(self, args):
        raise RuntimeError(f"cannot build {self.workspace}{os.sep}src{os.sep}app.ts")
    monkeypatch.setitem(ToolExecutor._handlers, "list_files", failing)
    assert _run(executor, "list_files", {})["error"] == "list_files failed: cannot build src/app.ts"
