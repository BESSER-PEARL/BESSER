"""Small gaps in the file/shell tool guards, each found by the 2026-09-28 review."""
import sys

import pytest

from besser.spec_driven_agent.agent import tool_executor as tool_executor_module
from besser.spec_driven_agent.agent.tool_executor import (
    ToolExecutor,
    _looks_like_command_not_found,
)
from tests.spec_driven_agent.test_breaks_module_import import GOOD_ORM


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def test_the_modify_streak_counter_accepts_a_dot_slash_spelling(tmp_path):
    """The orchestrator asks with the path as the model spelled it; the
    executor keys on the normalised one, so ``./app.py`` always read 0."""
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))
    for _ in range(3):
        call(executor, "modify_file", path="./app.py", old_text="nope", new_text="y")

    assert executor.consecutive_modify_misses("./app.py") == 3


def test_write_file_cannot_make_the_orm_unimportable(tmp_path):
    (tmp_path / "sql_alchemy.py").write_text(GOOD_ORM, encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))
    call(executor, "read_file", path="sql_alchemy.py")

    result = call(executor, "write_file", path="sql_alchemy.py",
                  content=GOOD_ORM + "    other = relationship(Undefined)\n")

    assert result.get("rejection_kind") == "breaks_import", result
    assert (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8") == GOOD_ORM


def test_a_partial_read_does_not_unlock_a_full_rewrite(tmp_path):
    source = "".join(f"line_{n} = {n}\n" for n in range(40))
    (tmp_path / "big.py").write_text(source, encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))
    call(executor, "read_file", path="big.py", offset=0, limit=5)

    refused = call(executor, "write_file", path="big.py", content="line_0 = 0\n")

    assert "error" in refused, refused
    assert (tmp_path / "big.py").read_text(encoding="utf-8") == source


def test_windows_that_together_cover_the_file_unlock_a_rewrite(tmp_path):
    source = "".join(f"line_{n} = {n}\n" for n in range(40))
    (tmp_path / "big.py").write_text(source, encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))
    call(executor, "read_file", path="big.py", offset=0, limit=25)
    call(executor, "read_file", path="big.py", offset=25)

    result = call(executor, "write_file", path="big.py", content="line_0 = 0\n")

    assert result.get("status") == "written", result


def test_dash_not_found_is_a_missing_runtime():
    """The worker's /bin/sh is dash; measured in the smartgen_worker image."""
    assert _looks_like_command_not_found("/bin/sh: 1: ps: not found\n")


def test_a_not_found_file_is_still_not_a_missing_runtime():
    assert not _looks_like_command_not_found("ERROR: requirements.txt: not found in context")


def test_a_timeout_keeps_the_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_executor_module, "COMMAND_TIMEOUT", 3)
    executor = ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    command = f'"{sys.executable}" -c "print(\'PARTIAL-OUTPUT\', flush=True); import time; time.sleep(30)"'

    result = call(executor, "run_command", command=command)

    assert "timed out" in result["error"], result
    assert "PARTIAL-OUTPUT" in result.get("stdout", ""), result


@pytest.mark.parametrize("tool,args,missing", [
    ("write_file", {"path": "a.py"}, "content"),
    ("read_file", {}, "path"),
    ("modify_file", {"path": "a.py", "old_text": "x"}, "new_text"),
])
def test_a_missing_argument_is_a_clear_tool_error(tmp_path, tool, args, missing):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))

    error = executor.execute_typed(tool, args).payload["error"]

    assert "KeyError" not in error and "missing required argument" in error, error
    assert repr(missing) in error or missing in error
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_unparseable_arguments_are_refused_not_run_with_nothing(tmp_path):
    """Providers used to hand ``{}`` on a JSON parse failure, and list_files
    / task_list etc. then ran as if the model had asked for that."""
    from besser.spec_driven_agent.agent.tools import INVALID_ARGUMENTS_KEY

    executor = ToolExecutor(workspace=str(tmp_path))
    marker = {INVALID_ARGUMENTS_KEY: "Expecting ',' delimiter: line 1 column 30"}

    result = executor.execute_typed("list_files", marker)

    assert result.status == "error"
    assert "not valid JSON" in result.payload["error"]
    assert "line 1 column 30" in result.payload["error"]


def test_non_object_arguments_are_refused(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))

    result = executor.execute_typed("list_files", ["app.py"])

    assert result.status == "error" and "JSON object" in result.payload["error"]
