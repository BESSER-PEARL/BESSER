"""A replacement must never carry read_file's line numbers to disk.

``read_file`` prefixes every line with ``NNN| ``. A model that selected a
range BY those numbers very often pastes them back into ``new_text``.
``modify_file``'s ladder has always stripped that from both sides;
``replace_file_lines`` shipped without the guard.

Run fcdh0s9k, on one file, two turns apart:

    turn 29  modify_file         107/107 numbered lines  -> error (refused)
    turn 31  replace_file_lines  293/293 numbered lines  -> ok    (written)

Lines 101-393 of the delivered ``Booking.tsx`` were then literally
``" 101|       </nav>"`` - not valid TSX. The backend workflow probe scored
that run 10/10 and nothing caught the frontend, because tsc was disabled.
"""

import os
import tempfile

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


SOURCE = "const a = 1;\nconst b = 2;\nconst c = 3;\n"


@pytest.fixture
def executor():
    workspace = tempfile.mkdtemp()
    with open(os.path.join(workspace, "Page.tsx"), "w", encoding="utf-8") as handle:
        handle.write(SOURCE)
    return ToolExecutor(workspace=workspace)


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def test_numbered_new_text_is_stripped_not_written(executor):
    read = call(executor, "read_file", path="Page.tsx")

    result = call(executor, "replace_file_lines", path="Page.tsx",
                  read_id=read["read_id"], start_line=2, end_line=2,
                  new_text="   2| const b = 22;")

    assert result["status"] == "modified", result
    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert "2| " not in on_disk, on_disk
    assert "const b = 22;" in on_disk
    assert "line numbers" in result.get("note", "")


def test_the_live_multiline_shape_is_stripped(executor):
    """293 numbered lines in one call is what actually corrupted the file."""
    read = call(executor, "read_file", path="Page.tsx")

    result = call(executor, "replace_file_lines", path="Page.tsx",
                  read_id=read["read_id"], start_line=1, end_line=3,
                  new_text="   1| const a = 1;\n   2| const b = 2;\n   3| const z = 9;")

    assert result["status"] == "modified", result
    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert on_disk == "const a = 1;\nconst b = 2;\nconst z = 9;\n", repr(on_disk)


def test_code_that_merely_looks_numbered_is_left_alone(executor):
    """Only a UNIFORM prefix on every non-blank line is a copied numbering."""
    read = call(executor, "read_file", path="Page.tsx")

    call(executor, "replace_file_lines", path="Page.tsx", read_id=read["read_id"],
         start_line=2, end_line=2, new_text='const b = "12| not a prefix";')

    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert '"12| not a prefix"' in on_disk, on_disk


def test_an_ordinary_replacement_reports_no_note(executor):
    read = call(executor, "read_file", path="Page.tsx")

    result = call(executor, "replace_file_lines", path="Page.tsx",
                  read_id=read["read_id"], start_line=2, end_line=2,
                  new_text="const b = 22;")

    assert result["status"] == "modified"
    assert "note" not in result


def test_modify_file_still_refuses_or_strips_the_same_shape(executor):
    """The tool that already handled this must not regress."""
    result = call(executor, "modify_file", path="Page.tsx",
                  old_text="   2| const b = 2;", new_text="   2| const b = 22;")

    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert "2| " not in on_disk, on_disk
    if result.get("status") == "modified":
        assert "const b = 22;" in on_disk


def test_mixed_numbering_is_stripped_too(executor):
    """Run kinie9zr turn 21: 48 of 49 lines carried ``NNN| `` and the odd one
    out vetoed the all-or-nothing rule, so the numbers were WRITTEN. 11 such
    mixed inputs across 23 runs, this one the only accepted write."""
    read = call(executor, "read_file", path="Page.tsx")

    result = call(executor, "replace_file_lines", path="Page.tsx",
                  read_id=read["read_id"], start_line=1, end_line=3,
                  new_text="const a = 1;\n   2| const b = 2;\n   3| const z = 9;")

    assert result["status"] == "modified", result
    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert on_disk == "const a = 1;\nconst b = 2;\nconst z = 9;\n", repr(on_disk)
    assert "line numbers" in result.get("note", "")


def test_mixed_numbering_from_two_separate_reads_is_stripped(executor):
    """The same run pasted two non-contiguous numbered regions into one
    new_text, so the numbers jump. Rising is the rule, not consecutive."""
    read = call(executor, "read_file", path="Page.tsx")

    result = call(executor, "replace_file_lines", path="Page.tsx",
                  read_id=read["read_id"], start_line=1, end_line=3,
                  new_text=" 141| const a = 1;\nconst b = 2;\n 260| const z = 9;")

    assert result["status"] == "modified", result
    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert on_disk == "const a = 1;\nconst b = 2;\nconst z = 9;\n", repr(on_disk)


def test_a_dict_literal_with_numeric_keys_is_not_treated_as_numbering(executor):
    """``1: "a"`` matches the general prefix pattern, so the mixed rule accepts
    only read_file's own ``NNN| `` form. Calibrated at zero matches over 2.27M
    line windows of besser/ and 23 generated apps."""
    read = call(executor, "read_file", path="Page.tsx")

    call(executor, "replace_file_lines", path="Page.tsx", read_id=read["read_id"],
         start_line=1, end_line=3,
         new_text='const m = {\n1: "a",\n2: "b",\n3: "c",\n};')

    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert '1: "a",' in on_disk and '3: "c",' in on_disk, on_disk


def test_numbers_that_do_not_rise_are_not_a_copied_numbering(executor):
    read = call(executor, "read_file", path="Page.tsx")

    call(executor, "replace_file_lines", path="Page.tsx", read_id=read["read_id"],
         start_line=1, end_line=3,
         new_text="9| const a = 1;\nconst b = 2;\n2| const z = 9;")

    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert "9| const a = 1;" in on_disk, on_disk


def test_one_numbered_line_among_unnumbered_ones_is_left_alone(executor):
    """Below the majority: a lone match is far likelier to be real code."""
    read = call(executor, "read_file", path="Page.tsx")

    call(executor, "replace_file_lines", path="Page.tsx", read_id=read["read_id"],
         start_line=1, end_line=3,
         new_text="const a = 1;\nconst b = 2;\n3| const z = 9;")

    on_disk = open(os.path.join(executor.workspace, "Page.tsx"), encoding="utf-8").read()
    assert "3| const z = 9;" in on_disk, on_disk
