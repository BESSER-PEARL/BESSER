"""The missing import that killed three runs, fixed without a model.

Live Qwen runs kept shipping the same shape: an edit uses a framework name
the file never imported, the module stops importing, every router that
star-imports it dies, and Phase 3 rolls the whole repair back.

    trilraak  booking_guest = Table(...)   sqlalchemy.Table never imported
    trilraak  dt_date / datetime           datetime never imported
    pcovsppe  datetime                     same

trilraak shipped a backend that could not start. pcovsppe and se7k3zbx lost
otherwise-good repairs to the rollback this caused.

The repair is deliberately narrow: exactly one allowlisted module must
export the name, checked by importing it. A wrong import is worse than a
missing one.
"""

import pytest

from besser.spec_driven_agent.import_repair import repair_missing_imports


TRILRAAK = (
    "from sqlalchemy import Column, Integer, String, ForeignKey\n"
    "from sqlalchemy.ext.declarative import declarative_base\n"
    "\n"
    "Base = declarative_base()\n"
    "\n"
    "booking_guest = Table('booking_guest', Base.metadata,\n"
    "    Column('booking_id', Integer, ForeignKey('booking.id')),\n"
    ")\n"
)


def test_the_trilraak_break_is_repaired(monkeypatch):
    repaired, notes = repair_missing_imports("sql_alchemy.py", TRILRAAK)

    assert "Table" in repaired.splitlines()[0], repaired.splitlines()[0]
    assert notes and "Table" in notes[0]
    # The rest of the file is untouched.
    assert "booking_guest = Table('booking_guest', Base.metadata," in repaired
    assert len(repaired.splitlines()) == len(TRILRAAK.splitlines())


def test_a_name_with_no_existing_import_gets_a_new_line():
    source = "def when() -> None:\n    return datetime.now()\n"

    repaired, notes = repair_missing_imports("x.py", source)

    assert "from datetime import datetime" in repaired
    assert "return datetime.now()" in repaired
    assert notes


def test_nothing_is_invented_for_an_unknown_name():
    source = "def go():\n    return ReservedRoom()\n"

    repaired, notes = repair_missing_imports("x.py", source)

    assert repaired == source
    assert notes == []


@pytest.mark.parametrize("name", ["Enum", "Query", "UUID"])
def test_an_ambiguous_name_is_left_alone(name):
    """Enum is enum.Enum OR sqlalchemy.Enum; picking one would be a real bug."""
    source = f"def go():\n    return {name}\n"

    repaired, notes = repair_missing_imports("x.py", source)

    assert repaired == source and notes == []


def test_an_unambiguous_typing_name_is_imported():
    source = "def go(x: Optional) -> None:\n    return None\n"

    repaired, notes = repair_missing_imports("x.py", source)

    assert "from typing import Optional" in repaired and notes


def test_a_file_that_does_not_parse_is_untouched():
    source = "def broken(:\n    pass\n"

    assert repair_missing_imports("x.py", source) == (source, [])


def test_a_clean_file_is_untouched():
    source = "from datetime import datetime\n\nx = datetime.now()\n"

    assert repair_missing_imports("x.py", source) == (source, [])


def test_a_non_python_file_is_untouched():
    source = "const Table = 1;\n"

    assert repair_missing_imports("app.tsx", source) == (source, [])


def test_a_star_import_line_is_not_extended():
    """Appending to 'from x import *' would be a syntax error."""
    source = "from sqlalchemy import *\n\nt = Table('x')\n"

    repaired, _ = repair_missing_imports("x.py", source)

    assert "from sqlalchemy import *" in repaired
    assert "import *, Table" not in repaired


@pytest.mark.parametrize("name", ["id", "type", "list", "date"])
def test_generic_names_are_never_auto_imported(name):
    source = f"def go():\n    return {name}\n"

    repaired, notes = repair_missing_imports("x.py", source)

    assert repaired == source and notes == []


def test_the_repair_reaches_the_file_through_a_real_edit(tmp_path):
    """End to end: the executor writes the fix back to disk."""
    from besser.spec_driven_agent.tool_executor import ToolExecutor

    target = tmp_path / "sql_alchemy.py"
    # A real declarative base. With `Base = None` this edit is genuinely
    # unimportable, and the structural-import guard is right to refuse it.
    target.write_text(
        "from sqlalchemy import Column, Integer\n"
        "from sqlalchemy.orm import declarative_base\n"
        "\nBase = declarative_base()\n", encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))

    payload = executor.execute_typed("modify_file", {
        "path": "sql_alchemy.py",
        "old_text": "Base = declarative_base()",
        "new_text": "Base = declarative_base()\n"
                    "t = Table('x', Base.metadata, Column('id', Integer, primary_key=True))",
    }).payload

    assert payload.get("status") == "modified", payload
    on_disk = target.read_text(encoding="utf-8")
    assert "Table" in on_disk.splitlines()[0], on_disk
    assert payload.get("auto_imports"), "the model was not told about the repair"
    # And the file still parses.
    import ast
    ast.parse(on_disk)
