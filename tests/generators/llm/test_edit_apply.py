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
    assert json.loads(ok.to_json())["content"] == "   1| a = 1\n   2| "


# ----------------------------------------------------------------------
# Blank-line runs. Generated scaffolds carry runs of 3-7 blank lines (Jinja
# whitespace); models collapse them to one when quoting. Measured on a fresh
# FastAPI scaffold, 2026-09-17: a quote spanning a run missed the ladder in
# 23/25 windows of routers/bill.py and 13/13 of main_api.py - the single
# largest reason modify_file "could not find" text the model had just read.
# ----------------------------------------------------------------------

def test_blank_line_run_quoted_as_one_blank_line_still_matches():
    whole = "a = 1\n\n\n\nb = 2\n\n\n\n\n\n\nc = 3\n"
    part = "a = 1\n\nb = 2\n"
    res = replace_most_similar_chunk(whole, part, "a = 1\n\nB = 2\n")
    assert res is not None
    assert res.startswith("a = 1\n\nB = 2\n")
    assert "b = 2" not in res
    assert res.endswith("c = 3\n")          # everything after the span untouched


def test_blank_line_run_quoted_as_two_blank_lines_still_matches():
    whole = "def f():\n    pass\n\n\n\n\n@decorator\ndef g():\n    pass\n"
    part = "    pass\n\n\n@decorator\n"
    res = replace_most_similar_chunk(whole, part, "    return 1\n\n@decorator\n")
    assert res == "def f():\n    return 1\n\n@decorator\ndef g():\n    pass\n"


def test_omitting_the_blank_line_entirely_is_still_refused():
    """Only the COUNT within a run is forgiven; every line the model quotes
    must exist, so a quote with no blank line where the file has a run is
    a genuine miss, not a match."""
    whole = "a = 1\n\n\nb = 2\n"
    assert replace_most_similar_chunk(whole, "a = 1\nb = 2\n", "x\n") is None


def test_collapsed_blank_runs_never_match_different_code():
    whole = "a = 1\n\n\n\nb = 2\n"
    assert replace_most_similar_chunk(whole, "a = 1\n\nb = 3\n", "x\n") is None


def test_modify_file_accepts_a_quote_with_collapsed_blank_lines(tmp_path):
    """End to end through the tool: the shape of a generated 501 stub."""
    stub = (
        "    try:\n"
        "        sys.stdout = captured_output\n"
        "\n\n\n"
        "        # Booking.cancel: no body in the model - be honest: 501\n"
        "        raise HTTPException(status_code=501)\n"
    )
    (tmp_path / "booking_methods.py").write_text(stub, encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))
    result = executor._modify_file({
        "path": "booking_methods.py",
        "old_text": (
            "        sys.stdout = captured_output\n"
            "\n"
            "        # Booking.cancel: no body in the model - be honest: 501\n"
            "        raise HTTPException(status_code=501)\n"
        ),
        "new_text": (
            "        sys.stdout = captured_output\n"
            "        booking.status = CANCELLED\n"
            "        return {\"status\": \"cancelled\"}\n"
        ),
    })
    assert result.get("status") == "modified", result
    text = (tmp_path / "booking_methods.py").read_text(encoding="utf-8")
    assert "booking.status = CANCELLED" in text
    assert "status_code=501" not in text


# ----------------------------------------------------------------------
# read_file numbers its output ("   7| code"). The model pastes those lines
# back verbatim; tier 5 strips the prefix. And a template copied verbatim out
# of a Windows checkout reaches the workspace with CRLF.
# ----------------------------------------------------------------------

def test_a_quote_copied_with_read_file_line_numbers_still_applies(tmp_path):
    _seed(tmp_path, "app.py", "def f():\n    a = 1\n    b = 2\n    return a + b\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    numbered = _call(ex, "read_file", {"path": "app.py"})["content"].split("\n")
    assert numbered[1] == "   2|     a = 1", numbered

    res = _call(ex, "modify_file", {
        "path": "app.py",
        "old_text": "\n".join(numbered[1:3]),
        "new_text": "    a = 10\n    b = 20",
    })

    assert res.get("status") == "modified", res
    with open(os.path.join(str(tmp_path), "app.py"), encoding="utf-8") as f:
        assert f.read() == "def f():\n    a = 10\n    b = 20\n    return a + b\n"


def test_a_crlf_file_is_read_and_written_back_as_lf(tmp_path):
    path = os.path.join(str(tmp_path), "InputComponents.tsx")
    with open(path, "wb") as f:
        f.write(b"export function Input() {\r\n  return null;\r\n}\r\n")
    ex = ToolExecutor(workspace=str(tmp_path))

    assert "\r" not in _call(ex, "read_file", {"path": "InputComponents.tsx"})["content"]
    res = _call(ex, "modify_file", {
        "path": "InputComponents.tsx",
        "old_text": "export function Input() {\n  return null;",
        "new_text": "export function Input() {\n  return <input />;",
    })

    assert res.get("status") == "modified", res
    with open(path, "rb") as f:
        assert b"\r" not in f.read()
