"""Tests for the flexible modify_file apply ladder (``edit_apply``).

The ladder is ported from Aider (editblock_coder.py); these cases mirror
aider's own ``tests/basic/test_editblock.py`` whitespace suite, plus the two
tiers we deliberately did NOT port (similarity matching and ``...`` elision),
which must therefore FAIL to match rather than guess.
"""

import json
import os

from besser.generators.llm.edit_apply import (
    find_similar_lines,
    replace_most_similar_chunk,
)
from besser.generators.llm.tool_executor import ToolExecutor


# -- the ladder -----------------------------------------------------------

def test_exact_match_replaces_first_occurrence_only():
    whole = "line1\nline2\nline1\nline3\n"
    assert replace_most_similar_chunk(whole, "line1\n", "new\n") == (
        "new\nline2\nline1\nline3\n"
    )


def test_uniform_missing_leading_whitespace():
    """The model reproduced the block unindented - the commonest real miss."""
    whole = "    line1\n    line2\n    line3\n"
    assert replace_most_similar_chunk(whole, "line1\nline2\n", "new1\nnew2\n") == (
        "    new1\n    new2\n    line3\n"
    )


def test_varied_leading_whitespace_is_preserved_relatively():
    whole = "\n    line1\n    line2\n        line3\n    line4\n"
    out = replace_most_similar_chunk(whole, "line2\n    line3\n", "new2\n    new3\n")
    assert out == "\n    line1\n    new2\n        new3\n    line4\n"


def test_partially_missing_whitespace():
    whole = "    line1\n    line2\n    line3\n"
    out = replace_most_similar_chunk(whole, " line1\n line2\n", " new1\n     new2\n")
    assert out == "    new1\n        new2\n    line3\n"


def test_spurious_leading_blank_line_is_dropped():
    """Aider issue #25 - models prepend a blank line to the block."""
    whole = "    line1\n    line2\n    line3\n"
    out = replace_most_similar_chunk(whole, "\n  line1\n  line2\n", "  new1\n  new2\n")
    assert out == "    new1\n    new2\n    line3\n"


def test_blank_lines_are_not_reindented():
    """A blank line inside the block must not gain trailing whitespace."""
    whole = "    a = 1\n\n    b = 2\n"
    out = replace_most_similar_chunk(whole, "a = 1\n\nb = 2\n", "a = 9\n\nb = 8\n")
    assert out == "    a = 9\n\n    b = 8\n"


def test_inconsistent_indent_is_refused():
    """Two different extra-indent prefixes -> ambiguous, so no match."""
    whole = "    line1\n        line2\n"
    assert replace_most_similar_chunk(whole, "line1\nline2\n", "x\ny\n") is None


def test_genuinely_absent_text_returns_none():
    assert replace_most_similar_chunk("a\nb\n", "zzz\n", "q\n") is None


def test_similar_but_not_equal_is_not_guessed():
    """We deliberately omitted aider's disabled SequenceMatcher tier: an
    80%-similar block must NOT be silently edited."""
    whole = "def compute_total(items):\n    return sum(items)\n"
    part = "def compute_total(item):\n    return sum(item)\n"
    assert replace_most_similar_chunk(whole, part, "x\n") is None


def test_python_ellipsis_body_is_not_treated_as_elision():
    """Three dots on their own line are valid Python (Protocol/stub bodies).
    We did not port aider's elision tier, so this must not match spuriously."""
    whole = "class P(Protocol):\n    def f(self) -> int:\n        ...\n"
    part = "class Q(Protocol):\n    ...\n"
    assert replace_most_similar_chunk(whole, part, "class R:\n    pass\n") is None


# -- the "did you mean" hint ---------------------------------------------

def test_find_similar_lines_returns_the_close_region():
    content = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    hint = find_similar_lines("def b():\n    return 22\n", content)
    assert "def b():" in hint


def test_find_similar_lines_pads_when_not_anchored():
    content = "\n".join("line%d" % i for i in range(30))
    hint = find_similar_lines("line10\nline11x", content)
    assert "line10" in hint
    assert len(hint.splitlines()) <= 12  # padded window, not the whole file


def test_find_similar_lines_returns_empty_below_threshold():
    assert find_similar_lines("zzz\nqqq\n", "aaa\nbbb\nccc\n") == ""


def test_find_similar_lines_handles_empty_inputs():
    assert find_similar_lines("", "a\n") == ""
    assert find_similar_lines("a\n", "") == ""


# -- executor integration ------------------------------------------------

def _seed(tmp_path, rel, content):
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full) or str(tmp_path), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def _call(executor, tool, args):
    out = executor.execute(tool, args)
    return json.loads(out) if isinstance(out, str) else out


def test_modify_file_uses_the_flexible_ladder(tmp_path):
    _seed(tmp_path, "app.py", "def f():\n    x = 1\n    return x\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "app.py",
        "old_text": "x = 1\nreturn x\n",   # unindented by the model
        "new_text": "x = 2\nreturn x\n",
    })
    assert res.get("status") == "modified", res
    assert res.get("matched_by") == "flexible"
    with open(os.path.join(str(tmp_path), "app.py"), encoding="utf-8") as f:
        assert "    x = 2" in f.read()


def test_modify_file_miss_carries_did_you_mean_and_escalates(tmp_path):
    _seed(tmp_path, "app.py", "def compute():\n    return 42\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    args = {
        "path": "app.py",
        "old_text": "def compute():\n    return 43\n",
        "new_text": "def compute():\n    return 44\n",
    }

    first = _call(ex, "modify_file", args)
    assert "error" in first
    assert "did_you_mean" in first and "return 42" in first["did_you_mean"]
    assert "advice" not in first

    second = _call(ex, "modify_file", args)
    assert "advice" in second and "read_file" in second["advice"]


def test_modify_file_flags_edit_already_applied(tmp_path):
    _seed(tmp_path, "app.py", "value = 2\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "app.py",
        "old_text": "value = 1\n",   # not present
        "new_text": "value = 2\n",   # already present
    })
    assert "error" in res
    assert "ALREADY present" in res.get("note", "")


def test_modify_file_refuses_short_ambiguous_anchor(tmp_path):
    _seed(tmp_path, "app.py", "a = 1\nb = 2\na = 1\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "app.py", "old_text": "a = 1", "new_text": "a = 9",
    })
    assert "error" in res and "ambiguous" in res["error"]
    with open(os.path.join(str(tmp_path), "app.py"), encoding="utf-8") as f:
        assert f.read() == "a = 1\nb = 2\na = 1\n"   # nothing written


def test_modify_file_refuses_any_ambiguous_anchor_without_replace_all(tmp_path):
    body = (
        "def handler():\n    record_the_start_of_request()\n    return 1\n"
        "def other():\n    record_the_start_of_request()\n    return 2\n"
    )
    _seed(tmp_path, "app.py", body)
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "app.py",
        "old_text": "    record_the_start_of_request()\n",
        "new_text": "    record_begin()\n",
    })
    assert "error" in res and "replace_all=true" in res["error"]
    with open(os.path.join(str(tmp_path), "app.py"), encoding="utf-8") as f:
        assert f.read() == body


def test_modify_file_replace_all_is_explicit_and_counted(tmp_path):
    _seed(tmp_path, "app.py", "a = 1\nb = 2\na = 1\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "app.py",
        "old_text": "a = 1",
        "new_text": "a = 9",
        "replace_all": True,
    })
    assert res.get("status") == "modified", res
    assert res["replacements"] == 2
    with open(os.path.join(str(tmp_path), "app.py"), encoding="utf-8") as f:
        assert f.read() == "a = 9\nb = 2\na = 9\n"


def test_modify_file_rejects_empty_old_text(tmp_path):
    _seed(tmp_path, "app.py", "a = 1\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "app.py", "old_text": "   ", "new_text": "b = 2\n",
    })
    assert "error" in res and "write_file" in res["error"]


def test_modify_file_rejects_noop(tmp_path):
    _seed(tmp_path, "app.py", "a = 1\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "app.py", "old_text": "a = 1", "new_text": "a = 1",
    })
    assert "error" in res and "identical" in res["error"]


def test_typed_execution_result_distinguishes_error_and_success(tmp_path):
    _seed(tmp_path, "app.py", "a = 1\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    ok = ex.execute_typed("read_file", {"path": "app.py"})
    error = ex.execute_typed("read_file", {"path": "missing.py"})

    assert ok.status == "ok" and ok.succeeded is True
    assert error.status == "error" and error.succeeded is False
    assert json.loads(ok.to_json())["content"] == "a = 1\n"
