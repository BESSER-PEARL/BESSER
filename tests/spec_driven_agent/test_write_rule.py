"""One write rule: an existing file shown whole this run, a rewrite that keeps its code.

A rewrite of an existing file is refused for what it does to the file (it drops
declarations, most of its lines, or abbreviates code), never for how many edit
calls came before it. Reproduced before these tests: a modify run refused a
full-read rewrite that kept every line of a 42-line module, a generated
241-line router refused a rewrite that kept all of its functions, and two
throwaway ``modify_file`` calls (an empty old_text, a no-op) unlocked a rewrite
that dropped half of a generated module's functions.
"""

import os

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


def call(executor, name, **args):
    return executor.execute_typed(name, args).payload


def _functions(prefix, count):
    return "".join(f"def {prefix}{i}():\n    return {i}\n\n" for i in range(count))


def _seed(workspace, rel, source):
    target = os.path.join(workspace, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source)


def _modify_run(tmp_path, rel, source):
    _seed(str(tmp_path), rel, source)
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.enable_modify_guard()
    return executor


def _generated(tmp_path, rel, source):
    _seed(str(tmp_path), rel, source)
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.update_generator_files([rel], [])
    return executor


def _junk_edits(executor, rel):
    """Two calls that change nothing: an empty quote and a no-op."""
    assert "error" in call(executor, "modify_file", path=rel, old_text="  ", new_text="x")
    assert "error" in call(executor, "modify_file", path=rel, old_text="return 1", new_text="return 1")


def test_a_full_read_rewrite_that_keeps_the_code_lands_in_a_modify_run(tmp_path):
    source = _functions("f", 14)
    executor = _modify_run(tmp_path, "backend/util.py", source)
    call(executor, "read_file", path="backend/util.py")

    written = call(executor, "write_file", path="backend/util.py",
                   content=source + "def g():\n    return 0\n")

    assert written.get("status") == "written", written


def test_a_generated_backend_rewrite_that_keeps_its_functions_lands(tmp_path):
    rel = "backend/routers/book.py"
    source = _functions("r", 80)
    executor = _generated(tmp_path, rel, source)
    call(executor, "read_file", path=rel)

    written = call(executor, "write_file", path=rel, content=source + "def extra():\n    return 1\n")

    assert written.get("status") == "written", written


def test_a_generated_backend_rewrite_that_drops_functions_is_refused_by_name(tmp_path):
    rel = "backend/routers/book.py"
    source = _functions("r", 80)
    executor = _generated(tmp_path, rel, source)
    call(executor, "read_file", path=rel)

    lossy = source.replace("def r7():\n    return 7\n\n", "")
    refused = call(executor, "write_file", path=rel, content=lossy)

    assert "r7" in refused.get("error", ""), refused
    assert "unlock" not in refused["error"]
    with open(tmp_path / rel, encoding="utf-8") as handle:
        assert handle.read() == source


def test_junk_edits_do_not_unlock_a_lossy_rewrite(tmp_path):
    rel = "backend/routers/book.py"
    source = _functions("r", 80)
    executor = _generated(tmp_path, rel, source)
    call(executor, "read_file", path=rel)
    _junk_edits(executor, rel)

    refused = call(executor, "write_file", path=rel, content=_functions("r", 40))

    assert "error" in refused, refused
    with open(tmp_path / rel, encoding="utf-8") as handle:
        assert handle.read() == source


def test_junk_edits_do_not_unlock_a_lossy_rewrite_in_a_modify_run(tmp_path):
    source = _functions("f", 14)
    executor = _modify_run(tmp_path, "backend/util.py", source)
    call(executor, "read_file", path="backend/util.py")
    _junk_edits(executor, "backend/util.py")

    lossy = source.replace("def f5():\n    return 5\n\n", "")
    refused = call(executor, "write_file", path="backend/util.py", content=lossy)

    assert "f5" in refused.get("error", ""), refused


def test_a_requested_removal_refused_as_a_rewrite_names_the_range_edit_that_lands(tmp_path):
    """The ladder's rewrite step ends here for a removal; modify_file has failed twice."""
    rel = "backend/util.py"
    source = _functions("f", 14)
    executor = _modify_run(tmp_path, rel, source)
    read_id = call(executor, "read_file", path=rel)["read_id"]
    for quote in ("def f5():\n    return 55\n", "def f5():\n  return 5\n"):
        call(executor, "modify_file", path=rel, old_text=quote, new_text="")

    refused = call(executor, "write_file", path=rel, content=source.replace("def f5():\n    return 5\n\n", ""))

    assert refused.get("rejection_kind") == "rewrite_loss", refused
    assert "replace_file_lines" in refused["error"] and read_id in refused["error"], refused
    start = source.splitlines().index("def f5():") + 1
    removed = call(executor, "replace_file_lines", path=rel, read_id=read_id,
                   start_line=start, end_line=start + 2, new_text="")
    assert removed.get("status") == "modified", removed
    with open(tmp_path / rel, encoding="utf-8") as handle:
        assert "def f5" not in handle.read()


def test_a_whole_range_edit_that_drops_functions_is_refused_like_a_rewrite(tmp_path):
    """The refusal's read_id covers the whole file; the range edit must not apply the same loss."""
    rel = "backend/routers/book.py"
    source = _functions("r", 80)
    executor = _generated(tmp_path, rel, source)
    read = call(executor, "read_file", path=rel)
    assert call(executor, "write_file", path=rel, content=_functions("r", 10))["rejection_kind"] == "rewrite_loss"

    refused = call(executor, "replace_file_lines", path=rel, read_id=read["read_id"],
                   start_line=1, end_line=read["end_line"], new_text=_functions("r", 10))

    assert refused.get("rejection_kind") == "rewrite_loss", refused
    with open(tmp_path / rel, encoding="utf-8") as handle:
        assert handle.read() == source


def test_a_most_of_the_file_range_edit_that_drops_functions_is_refused(tmp_path):
    rel = "backend/routers/book.py"
    source = _functions("r", 80)
    executor = _generated(tmp_path, rel, source)
    read = call(executor, "read_file", path=rel)

    refused = call(executor, "replace_file_lines", path=rel, read_id=read["read_id"],
                   start_line=4, end_line=read["end_line"], new_text=_functions("r", 10))

    assert refused.get("rejection_kind") == "rewrite_loss", refused


def test_a_whole_file_modify_that_drops_functions_is_refused_like_a_rewrite(tmp_path):
    rel = "backend/routers/book.py"
    source = _functions("r", 80)
    executor = _generated(tmp_path, rel, source)
    call(executor, "read_file", path=rel)

    refused = call(executor, "modify_file", path=rel, old_text=source, new_text=_functions("r", 10))

    assert refused.get("rejection_kind") == "rewrite_loss", refused
    with open(tmp_path / rel, encoding="utf-8") as handle:
        assert handle.read() == source


def test_a_whole_range_edit_that_keeps_the_code_lands(tmp_path):
    rel = "backend/routers/book.py"
    source = _functions("r", 80)
    executor = _generated(tmp_path, rel, source)
    read = call(executor, "read_file", path=rel)

    edited = call(executor, "replace_file_lines", path=rel, read_id=read["read_id"],
                  start_line=1, end_line=read["end_line"], new_text=source + "def extra():\n    return 1\n")

    assert edited.get("status") == "modified", edited


_LONG_TOTAL = (
    "def total(items):\n    result = 0\n    for item in items:\n        price = item.price\n"
    "        qty = item.qty\n        line = price * qty\n        if line < 0:\n            line = 0\n"
    "        result = result + line\n    if result < 0:\n        result = 0\n    return result\n")
_CALLS = "".join(f"app.include_router(r{i}.router)\n" for i in range(8))
_STEPS = "".join(f"- step {i}: run the thing number {i}\n" for i in range(24))
_TABLE = ("    <table>\n      <tbody>\n        {rows.map(row => (\n        <tr>\n"
          + "".join(f"          <td>{{row.c{i}}}</td>\n" for i in range(12))
          + "        </tr>\n        ))}\n      </tbody>\n    </table>\n")
# Edits that make a modify-run file shorter over 57-77% of its non-blank lines:
# each landed before range edits were held to the rewrite rule, and must still land.
_SHRINKING_EDITS = {
    "simplify a function": (
        "app/calc.py",
        "import math\n\n" + _LONG_TOTAL + "\n\ndef label(x):\n    return str(x)\n\n\n"
        "def rounded(x):\n    return math.floor(x)\n",
        _LONG_TOTAL,
        "def total(items):\n    return max(0, sum(max(0, i.price * i.qty) for i in items))\n"),
    "fold registrations into a loop": (
        "backend/main.py",
        "from fastapi import FastAPI\nfrom routers import " + ", ".join(f"r{i}" for i in range(8))
        + "\n\napp = FastAPI()\n\n" + _CALLS + "\n\n@app.get('/health')\ndef health():\n    return {'ok': True}\n",
        _CALLS,
        "for _m in (" + ", ".join(f"r{i}" for i in range(8)) + "):\n    app.include_router(_m.router)\n"),
    "shorten a document section": (
        "README.md",
        "# Project\n\nIntro line.\n\n## Setup\n\n" + _STEPS + "\n## Usage\n\nRun it.\nMore text.\nEven more.\n",
        _STEPS,
        "Run `make setup`.\n"),
    "replace markup with a component": (
        "frontend/src/Page.tsx",
        "import React from 'react';\nimport { Grid } from './Grid';\n\nexport function Page({ rows }) {\n"
        "  return (\n" + _TABLE + "  );\n}\n\nexport default Page;\n",
        _TABLE,
        "    <Grid rows={rows} />\n"),
}


@pytest.mark.parametrize("case", sorted(_SHRINKING_EDITS))
def test_a_shrinking_edit_over_part_of_a_file_lands_in_a_modify_run(tmp_path, case):
    rel, source, old, new = _SHRINKING_EDITS[case]
    executor = _modify_run(tmp_path, rel, source)
    call(executor, "read_file", path=rel)
    start =source[:source.index(old)].count("\n") + 1

    by_quote = call(executor, "modify_file", path=rel, old_text=old, new_text=new)
    assert by_quote.get("status") == "modified", by_quote

    _seed(str(tmp_path), rel, source)
    read = call(executor, "read_file", path=rel)
    by_range = call(executor, "replace_file_lines", path=rel, read_id=read["read_id"],
                    start_line=start, end_line=start + old.count("\n") - 1, new_text=new)
    assert by_range.get("status") == "modified", by_range
    with open(tmp_path / rel, encoding="utf-8") as handle:
        assert handle.read() == source.replace(old, new)


def test_replace_all_is_measured_over_every_occurrence(tmp_path):
    """Each occurrence is a small span; together they rewrite the whole file to half its lines."""
    rel = "docs/checklist.md"
    block = "- [ ] open item\n  owner: none\n"
    executor = _modify_run(tmp_path, rel, block * 15)
    call(executor, "read_file", path=rel)

    refused = call(executor, "modify_file", path=rel, old_text=block, new_text="- [ ] open item\n",
                   replace_all=True)

    assert refused.get("rejection_kind") == "rewrite_loss", refused
    with open(tmp_path / rel, encoding="utf-8") as handle:
        assert handle.read() == block * 15


def test_removing_a_large_part_of_a_small_file_on_purpose_still_lands(tmp_path):
    """An empty replacement is a removal, however much of the file it spans."""
    rel = "backend/util.py"
    body = "".join(f"    total += {i}\n" for i in range(12))
    source = f"def keep():\n    return 0\n\n\ndef big():\n    total = 0\n{body}    return total\n"
    executor = _modify_run(tmp_path, rel, source)
    read_id = call(executor, "read_file", path=rel)["read_id"]

    by_range = call(executor, "replace_file_lines", path=rel, read_id=read_id, start_line=5,
                    end_line=source.count("\n"), new_text="")
    assert by_range.get("status") == "modified", by_range

    _seed(str(tmp_path), rel, source)
    call(executor, "read_file", path=rel)
    by_quote = call(executor, "modify_file", path=rel, old_text=source[source.index("def big"):], new_text="")
    assert by_quote.get("status") == "modified", by_quote


def test_a_file_created_this_run_is_rewritten_freely_in_a_modify_run(tmp_path):
    executor = _modify_run(tmp_path, "backend/seed.py", _functions("s", 3))
    created = call(executor, "write_file", path="backend/new_service.py", content=_functions("n", 20))
    assert created.get("status") == "written", created

    rewritten = call(executor, "write_file", path="backend/new_service.py", content=_functions("m", 4))

    assert rewritten.get("status") == "written", rewritten


def test_a_rewrite_from_a_partial_view_is_still_refused(tmp_path):
    """The file must have been shown whole: a landed edit does not count as a read."""
    source = _functions("f", 14)
    executor = _modify_run(tmp_path, "backend/util.py", source)
    edited = call(executor, "modify_file", path="backend/util.py",
                  old_text="def f3():\n    return 3\n", new_text="def f3():\n    return 30\n")
    assert edited.get("status") == "modified", edited

    refused = call(executor, "write_file", path="backend/util.py",
                   content=source.replace("return 3\n", "return 30\n") + "def g():\n    return 0\n")

    assert "have read only part" in refused.get("error", ""), refused
