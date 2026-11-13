"""A workspace-wide finding is not re-attached to every unrelated write.

Repro: after renaming ``title`` in BookCreate, every later ``.py`` write,
however unrelated, returned the same book.py mismatch with "Fix them now
while this file is still in context" (467 chars for a ~70-char result, up to
~3k chars per write with 10 mismatches) until it was fixed.
"""

import json

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

SCHEMAS = "from pydantic import BaseModel\n\nclass BookCreate(BaseModel):\n    title: str\n"
ROUTER = "def create_book(book_data: BookCreate):\n    return book_data.title\n"


def _call(executor, tool, **arguments):
    return json.loads(executor.execute(tool, arguments))


def _mismatches(result):
    return [item for item in result.get("diagnostics", []) if item["code"] == "schema-consumer-mismatch"]


@pytest.fixture
def broken(tmp_path):
    """A workspace whose router reads a field the schema just lost."""
    executor = ToolExecutor(workspace=str(tmp_path))
    _call(executor, "write_file", path="backend/pydantic_classes.py", content=SCHEMAS)
    _call(executor, "write_file", path="backend/routers/book.py", content=ROUTER)
    renamed = _call(executor, "modify_file", path="backend/pydantic_classes.py",
                    old_text="    title: str", new_text="    name: str")
    return executor, renamed


def test_the_schema_edit_reports_the_consumer_and_names_its_file(broken):
    _, renamed = broken

    [finding] = _mismatches(renamed)
    assert "does not define `title`" in finding["message"]
    assert finding["in_other_files"] == ["backend/routers/book.py"]
    assert "in_other_files" in renamed["diagnostic_message"]


def test_an_unrelated_write_does_not_carry_it_again(broken):
    executor, _ = broken

    for i in range(3):
        result = _call(executor, "write_file", path=f"backend/util{i}.py", content="def helper():\n    return 1\n")
        assert "diagnostics" not in result, result


def test_writing_a_file_the_finding_names_still_shows_it(broken):
    executor, _ = broken

    router = _call(executor, "write_file", path="backend/routers/book.py",
                   content=ROUTER + "\n# still reads title\n")
    schema = _call(executor, "modify_file", path="backend/pydantic_classes.py",
                   old_text="    name: str", new_text="    name: str\n    isbn: str = ''")

    [finding] = _mismatches(router)
    assert "in_other_files" not in finding
    assert len(_mismatches(schema)) == 1


def test_a_finding_that_is_fixed_and_reintroduced_is_shown_again(broken):
    executor, _ = broken
    _call(executor, "modify_file", path="backend/pydantic_classes.py",
          old_text="    name: str", new_text="    title: str")
    assert "diagnostics" not in _call(executor, "write_file", path="backend/a.py", content="x = 1\n")

    # Broken again to the very same bytes outside the file tools (run_command).
    schema = f"{executor.workspace}/backend/pydantic_classes.py"
    with open(schema, "w", encoding="utf-8") as handle:
        handle.write(SCHEMAS.replace("title", "name"))

    assert len(_mismatches(_call(executor, "write_file", path="backend/b.py", content="x = 2\n"))) == 1


def test_a_change_to_a_named_file_shows_the_finding_again(broken):
    executor, _ = broken
    assert "diagnostics" not in _call(executor, "write_file", path="backend/a.py", content="x = 1\n")

    # Change the router on disk without the executor (another tool, a revert):
    # the finding is now about different content and is shown once more.
    router = f"{executor.workspace}/backend/routers/book.py"
    with open(router, "a", encoding="utf-8") as handle:
        handle.write("# touched\n")

    again = _call(executor, "write_file", path="backend/b.py", content="x = 2\n")
    assert len(_mismatches(again)) == 1
    assert _call(executor, "write_file", path="backend/c.py", content="x = 3\n").get("diagnostics") is None
