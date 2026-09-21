"""An edit that leaves the ORM parseable but unimportable is refused.

We already refuse an edit that makes a file unparseable. One level up is the
edit that parses fine and still kills the application: every generated router
does ``from sql_alchemy import *``, so a relationship naming a property that
does not exist, or a name used before definition, takes the whole app down
while every static check stays green.

The Phase 3 repair loop reaches for that file constantly and keeps losing:

    p_qopu92     all 5 of its edits went to sql_alchemy.py while ten of its
                 blockers named booking_methods.py; 7 -> 150 hard blockers
    iteration 10 2 -> 160
    iteration 2  10 -> 45, shipped a backend that would not start
    mbzbzhq9     traded flat, shipped dead

The phase rollback caught these, but a whole repair attempt was wasted each
time and one shipped anyway. Refusing the edit costs one turn instead.
"""

import os
import tempfile

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


GOOD_ORM = (
    "from sqlalchemy import Column, Integer, String\n"
    "from sqlalchemy.orm import declarative_base, relationship\n"
    "\n"
    "Base = declarative_base()\n"
    "\n"
    "class Booking(Base):\n"
    "    __tablename__ = 'booking'\n"
    "    id = Column(Integer, primary_key=True)\n"
    "    label = Column(String)\n"
)


@pytest.fixture
def executor():
    workspace = tempfile.mkdtemp()
    with open(os.path.join(workspace, "sql_alchemy.py"), "w", encoding="utf-8") as handle:
        handle.write(GOOD_ORM)
    with open(os.path.join(workspace, "notes.py"), "w", encoding="utf-8") as handle:
        handle.write("VALUE = 1\n")
    return ToolExecutor(workspace=workspace)


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def on_disk(executor, name="sql_alchemy.py"):
    return open(os.path.join(executor.workspace, name), encoding="utf-8").read()


def test_an_edit_that_kills_the_orm_import_is_refused(executor):
    """Parses fine; NameError on import. The exact p_qopu92 shape."""
    result = call(executor, "modify_file", path="sql_alchemy.py",
                  old_text="    label = Column(String)",
                  new_text="    label = Column(String)\n    other = relationship(Undefined)")

    assert result.get("rejection_kind") == "breaks_import", result
    assert "no longer" in result["error"] and "importable" in result["error"]
    assert on_disk(executor) == GOOD_ORM, "the refused edit was written anyway"


def test_a_healthy_edit_to_the_orm_still_lands(executor):
    result = call(executor, "modify_file", path="sql_alchemy.py",
                  old_text="    label = Column(String)",
                  new_text="    label = Column(String)\n    note = Column(String)")

    assert result.get("status") == "modified", result
    assert "note = Column(String)" in on_disk(executor)


def test_an_already_broken_orm_can_still_be_repaired(executor):
    """Only judged when the module imported BEFORE the edit."""
    broken = GOOD_ORM + "\nbad = Missing()\n"
    with open(os.path.join(executor.workspace, "sql_alchemy.py"), "w", encoding="utf-8") as h:
        h.write(broken)

    result = call(executor, "modify_file", path="sql_alchemy.py",
                  old_text="bad = Missing()", new_text="bad = 1")

    assert result.get("status") == "modified", result
    assert "bad = 1" in on_disk(executor)


def test_an_ordinary_module_is_not_import_checked(executor):
    """Only the structural modules every router star-imports."""
    result = call(executor, "modify_file", path="notes.py",
                  old_text="VALUE = 1", new_text="VALUE = Undefined")

    assert result.get("status") == "modified", result
    assert "VALUE = Undefined" in on_disk(executor, "notes.py")


def test_the_range_editor_is_guarded_too(executor):
    read = call(executor, "read_file", path="sql_alchemy.py")
    line = read["content"].splitlines().index("    9|     label = Column(String)".replace("    9| ", "    9| ")) + 1 \
        if False else 9

    result = call(executor, "replace_file_lines", path="sql_alchemy.py",
                  read_id=read["read_id"], start_line=line, end_line=line,
                  new_text="    label = relationship(Undefined)")

    assert result.get("rejection_kind") == "breaks_import", result
    assert on_disk(executor) == GOOD_ORM
