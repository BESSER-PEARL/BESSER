"""``_changed_region`` on a large file answers quickly and shows the same hunks.

It diffed every line with ``SequenceMatcher(autojunk=False)``: one changed line in a
20,000-line file took about 5 s, inside a tool call, on every write and refused edit.
"""

import time

import pytest

from besser.spec_driven_agent.agent.tool_executor import _changed_region

# Well under the ~5 s it took, well over the milliseconds it should take on a loaded machine.
_FAST_SECONDS = 1.0


def _source(count: int) -> list[str]:
    return [f"    value_{i} = compute({i})" if i % 3 else "" for i in range(count)]


def _timed(old: list[str], new: list[str]) -> tuple[str, float]:
    started = time.perf_counter()
    region = _changed_region("\n".join(old), "\n".join(new))
    return region, time.perf_counter() - started


@pytest.mark.parametrize("edits", [(10_000,), (100, 19_900), (0,), (19_999,)])
def test_edits_in_a_20000_line_file_are_shown_quickly(edits):
    old = _source(20_000)
    new = list(old)
    for index in edits:
        new[index] = f"    changed_{index} = 1"

    region, seconds = _timed(old, new)

    assert seconds < _FAST_SECONDS
    for index in edits:
        assert f"{index + 1:>4}|     changed_{index} = 1" in region
    assert "value_5000 " not in region


def test_small_files_show_the_same_hunks_as_before():
    old = ["import os", "", "def a():", "    return 1", "", "def b():", "    return 2", ""]
    new = ["import os", "import sys", "", "def a():", "    return 1", "", "def b():", "    return 3", ""]
    assert _changed_region("\n".join(old), "\n".join(new)) == "\n".join([
        "   1| import os",
        "   2| import sys",
        "   3| ",
        "   4| def a():",
        "   ...|",
        "   6| ",
        "   7| def b():",
        "   8|     return 3",
        "   9| ",
    ])


def test_an_insertion_and_a_deletion_keep_their_context():
    old = [f"line {i}" for i in range(50)]
    inserted = old[:20] + ["new line"] + old[20:]
    assert _changed_region("\n".join(old), "\n".join(inserted)) == "\n".join(
        f"{n:>4}| {text}" for n, text in [(19, "line 18"), (20, "line 19"), (21, "new line"),
                                          (22, "line 20"), (23, "line 21")])
    deleted = old[:20] + old[21:]
    assert _changed_region("\n".join(old), "\n".join(deleted)) == "\n".join(
        f"{n:>4}| {text}" for n, text in [(19, "line 18"), (20, "line 19"), (21, "line 21"), (22, "line 22")])


def _reference(old: list[str], new: list[str], context: int = 2) -> str:
    """The full-diff output: what any file small enough to diff whole must show."""
    from difflib import SequenceMatcher

    rows: list[str] = []
    for group in SequenceMatcher(None, old, new, autojunk=False).get_grouped_opcodes(context):
        if rows:
            rows.append("   ...|")
        rows += [f"{i + 1:>4}| {new[i]}" for i in range(group[0][3], group[-1][4])]
    return "\n".join(rows)


@pytest.mark.parametrize("count", [2_500, 3_000, 5_000])
def test_an_import_and_an_edit_far_below_are_diffed_whole(count):
    """Repair-style: an import at the top, a five-line edit 500 lines from the end. Diffing
    the whole file takes at most ~0.25 s here, so both hunks must be shown in full."""
    old = [f"    v_{i} = f({i % 7})" if i % 4 else "" for i in range(count)]
    new = list(old)
    new.insert(1, "import sys")
    new[count - 500:count - 495] = [f"    body_{j} = 2" for j in range(5)]
    assert _changed_region("\n".join(old), "\n".join(new)) == _reference(old, new)


def test_a_region_too_large_to_diff_shows_each_end_whole_and_says_the_rest_was_not_compared():
    old = _source(20_000)
    new = list(old)
    new[1:1] = ["import sys", "import json", "import re"]
    new[19_000:19_005] = [f"    edited_{j} = 1" for j in range(5)]

    region = _changed_region("\n".join(old), "\n".join(new))

    shown = [line for line in region.split("\n") if "| " in line and not line.lstrip().startswith("...|")]
    for text in ["import sys", "import json", "import re"] + [f"    edited_{j} = 1" for j in range(5)]:
        assert any(line.endswith(f"| {text}") for line in shown), text
    gaps = [line for line in region.split("\n") if line.lstrip().startswith("...|")]
    assert len(gaps) == 1 and "not compared" in gaps[0]
