"""A missed ``modify_file`` hands back a located line range, not a guess.

Pooled over six live Qwen runs ``modify_file`` lands 86-92% while
``replace_file_lines`` lands 15-43%, so the rescue tool is the weaker one -
and two of its four documented failure modes are line-number *selection*
errors (run 7aybctis picked read_file's 0-based offset; run fcdh0s9k pasted
``" 101|       </nav>"`` into Booking.tsx). Both disappear when the executor
supplies the range instead of making the model read again and guess.

``locate_anchored_span`` is a LOCATOR: it never applies anything. The model
still authors ``new_text``, and every guard on ``replace_file_lines`` - syntax,
module import, elision, line-number prefixes - still runs on the result.
"""

import os
import tempfile

import pytest

from besser.generators.llm.tool_executor import ToolExecutor


# The w7zoeszt shape: 16 of its refusals quoted the decorator indented four
# spaces deeper than the file holds it. Every ladder tier declines a quote the
# file is LESS indented than; the anchors still bracket it.
ROUTER = (
    'from fastapi import APIRouter\n'
    '\n'
    'router = APIRouter()\n'
    '\n'
    '@router.post("/bill/{bill_id}/methods/registerPayment/")\n'
    'async def execute_bill_registerPayment(\n'
    '    bill_id: int,\n'
    '    params: dict = None,\n'
    '):\n'
    '    """Execute the registerPayment method on a Bill instance."""\n'
    '    raise NotImplementedError\n'
)

OVER_INDENTED_QUOTE = (
    '    @router.post("/bill/{bill_id}/methods/registerPayment/")\n'
    'async def execute_bill_registerPayment(\n'
    '    bill_id: int,\n'
    '    params: dict = None,\n'
    '):\n'
    '    """Execute the registerPayment method on a Bill instance."""\n'
)

ORM = (
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
    for name, source in (("bill_methods.py", ROUTER), ("sql_alchemy.py", ORM)):
        with open(os.path.join(workspace, name), "w", encoding="utf-8") as handle:
            handle.write(source)
    return ToolExecutor(workspace=workspace)


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def on_disk(executor, name="bill_methods.py"):
    return open(os.path.join(executor.workspace, name), encoding="utf-8").read()


def miss(executor, path="bill_methods.py", old_text=OVER_INDENTED_QUOTE,
         new_text="# replaced\n"):
    return call(executor, "modify_file", path=path, old_text=old_text, new_text=new_text)


def test_the_first_miss_returns_a_pre_filled_range(executor):
    result = miss(executor)

    assert "error" in result, result
    assert on_disk(executor) == ROUTER, "a locator must never write"
    located = result["located_range"]
    assert (located["start_line"], located["end_line"]) == (5, 10), located
    assert located["read_id"]
    # 1-based numbers, in read_file's own format, straight from the file.
    assert located["lines"].startswith('   5| @router.post(')
    assert located["lines"].endswith('registerPayment method on a Bill instance."""')
    # One miss, not two: this is what removes the wasted turns.
    assert executor.consecutive_modify_misses("bill_methods.py") == 1
    recovery = result["edit_recovery"]
    assert recovery["next_tool"] == "replace_file_lines"
    assert (recovery["start_line"], recovery["end_line"]) == (5, 10)
    assert recovery["read_id"] == located["read_id"]
    # Two competing windows would just reintroduce the guess.
    assert "did_you_mean" not in result


def test_the_pre_filled_range_lands_in_one_call(executor):
    located = miss(executor)["located_range"]
    body = (
        '@router.post("/bill/{bill_id}/methods/registerPayment/")\n'
        'async def execute_bill_registerPayment(\n'
        '    bill_id: int,\n'
        '    params: dict = None,\n'
        '):\n'
        '    """Register a payment."""\n'
    )
    result = call(executor, "replace_file_lines", path="bill_methods.py",
                  read_id=located["read_id"], start_line=located["start_line"],
                  end_line=located["end_line"], new_text=body)

    assert result["status"] == "modified", result
    assert on_disk(executor) == ROUTER.replace(
        '    """Execute the registerPayment method on a Bill instance."""\n',
        '    """Register a payment."""\n')


def test_the_range_authorizes_only_the_located_span(executor):
    located = miss(executor)["located_range"]
    result = call(executor, "replace_file_lines", path="bill_methods.py",
                  read_id=located["read_id"], start_line=1, end_line=11,
                  new_text="# everything\n")

    assert result["rejection_kind"] == "unread_range", result
    assert on_disk(executor) == ROUTER


def test_the_syntax_guard_still_refuses_the_located_edit(executor):
    located = miss(executor)["located_range"]
    result = call(executor, "replace_file_lines", path="bill_methods.py",
                  read_id=located["read_id"], start_line=located["start_line"],
                  end_line=located["end_line"], new_text="async def broken(\n")

    assert result["rejection_kind"] == "syntax_error", result
    assert on_disk(executor) == ROUTER


def test_the_elision_guard_still_refuses_the_located_edit(executor):
    located = miss(executor)["located_range"]
    result = call(executor, "replace_file_lines", path="bill_methods.py",
                  read_id=located["read_id"], start_line=located["start_line"],
                  end_line=located["end_line"],
                  new_text='@router.post("/x/")\nasync def f():\n    # rest of the handler unchanged...\n    pass\n')

    assert result["rejection_kind"] == "elision", result
    assert on_disk(executor) == ROUTER


def test_line_number_prefixes_are_still_stripped_from_the_located_edit(executor):
    """The fcdh0s9k failure: the model pastes back the numbered display it was
    shown. We now show it MORE numbered output, so this must keep holding."""
    located = miss(executor)["located_range"]
    result = call(executor, "replace_file_lines", path="bill_methods.py",
                  read_id=located["read_id"], start_line=located["start_line"],
                  end_line=located["end_line"],
                  new_text='   5| @router.post("/x/")\n   6| async def f():\n'
                           '   7|     return None\n   8|     # pad\n'
                           '   9|     # pad\n  10|     # pad\n')

    assert result["status"] == "modified", result
    assert "| " not in on_disk(executor)
    assert "5|" not in on_disk(executor)
    assert result["note"]


def test_the_module_import_guard_still_refuses_the_located_edit(executor):
    """sql_alchemy.py is star-imported by every router, so an edit that parses
    but kills the import is refused on this path too."""
    quote = (
        "class Booking(Base):\n"
        "    __tablename__ = 'booking'\n"
        "    id = Column(Integer, primary_key=True, autoincrement=True)\n"
        "    label = Column(String)\n"
    )
    located = miss(executor, path="sql_alchemy.py", old_text=quote)["located_range"]
    assert (located["start_line"], located["end_line"]) == (6, 9)

    result = call(executor, "replace_file_lines", path="sql_alchemy.py",
                  read_id=located["read_id"], start_line=located["start_line"],
                  end_line=located["end_line"],
                  new_text="class Booking(Base):\n"
                           "    __tablename__ = 'booking'\n"
                           "    id = Column(Integer, primary_key=True)\n"
                           "    other = relationship(Undefined)\n")

    assert result["rejection_kind"] == "breaks_import", result
    assert on_disk(executor, "sql_alchemy.py") == ORM


def test_an_ambiguous_quote_gets_no_range(executor):
    """Two candidate regions must return the old hint, never a guessed span."""
    path = os.path.join(executor.workspace, "dup.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("def a():\n    x = 1\n    return x\n\ndef a():\n    x = 1\n    return x\n")

    result = miss(executor, path="dup.py", old_text="def a():\n    y = 1\n    return x\n")

    assert "located_range" not in result, result
    assert "did_you_mean" in result


def test_no_range_once_range_edits_are_exhausted_for_the_path(executor):
    """Run se7k3zbx failed 17 of 20 range edits on one file; past the give-up
    count the ladder steers back to quoting and must not offer a span."""
    read = call(executor, "read_file", path="bill_methods.py")
    for _ in range(executor._RANGE_EDIT_GIVE_UP):
        refused = call(executor, "replace_file_lines", path="bill_methods.py",
                       read_id=read["read_id"], start_line=6, end_line=6,
                       new_text="async def execute_bill_registerPayment(\n")
        assert "error" in refused

    result = miss(executor)

    assert "located_range" not in result, result
    assert result["edit_recovery"]["next_tool"] == "modify_file"
