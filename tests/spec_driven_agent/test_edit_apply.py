"""Tests for the flexible modify_file apply ladder (``edit_apply``).

The ladder is ported from Aider (editblock_coder.py); these cases mirror
aider's own ``tests/basic/test_editblock.py`` whitespace suite, plus the two
tiers we deliberately did NOT port (similarity matching and ``...`` elision),
which must therefore FAIL to match rather than guess.
"""

import ast
import json
import os

import pytest

from besser.spec_driven_agent.agent.edit_apply import (
    AmbiguousEdit,
    find_similar_lines,
    locate_anchored_span,
    replace_most_similar_chunk,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


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


def test_modify_file_needs_a_success_receipt_not_incidental_multiline_text(tmp_path):
    _seed(tmp_path, "app.py", "def get_value():\n    return 2\n")
    ex = ToolExecutor(workspace=str(tmp_path))
    args = {
        "path": "app.py",
        "old_text": "def get_value():\n    return 1\n",   # not present
        "new_text": "def get_value():\n    return 2\n",   # substantial block
    }
    res = _call(ex, "modify_file", args)
    assert "error" in res
    assert "old_text not found" in res["error"]
    assert res.get("status") != "already_applied"
    # Once this exact request really changes the file, its receipt is trusted.
    _seed(tmp_path, "app.py", args["old_text"])
    assert _call(ex, "modify_file", args)["status"] == "modified"
    assert _call(ex, "modify_file", args)["status"] == "already_applied"
    # Any intervening change invalidates the receipt, even if the replacement
    # chunk remains present. An unrecorded match must not report success.
    _seed(tmp_path, "app.py", args["new_text"] + "other = 3\n")
    assert _call(ex, "modify_file", args).get("status") != "already_applied"


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


# ----------------------------------------------------------------------
# New tiers vs. sst/opencode's edit.ts (MIT): trailing-whitespace tolerance
# (LineTrimmedReplacer)
# (WhitespaceNormalizedReplacer), and a symmetric blank-boundary trim
# (TrimmedBoundaryReplacer, extending aider issue #25 to both ends).
# ----------------------------------------------------------------------

def test_trailing_whitespace_only_difference_still_matches():
    """Tier 2 only lstrip()s, so a quoted line differing from the file ONLY
    in trailing whitespace (Jinja-generated scaffolds carry it) matched no
    tier at all and the edit was refused outright."""
    whole = "def foo():\n    x = 1  \n    return x\n"
    out = replace_most_similar_chunk(whole, "    x = 1\n", "    x = 2\n")
    assert out == "def foo():\n    x = 2\n    return x\n"


def test_spurious_blank_lines_at_both_ends_are_dropped():
    """Aider issue #25 (tier 3) only drops a spurious LEADING blank line; a
    model can just as easily pad the TRAILING end, or both."""
    whole = "    line1\n    line2\n    line3\n"
    out = replace_most_similar_chunk(whole, "\n  line1\n  line2\n\n", "  new1\n  new2\n")
    assert out == "    new1\n    new2\n    line3\n"


def test_spurious_trailing_blank_line_alone_is_dropped():
    whole = "    line1\n    line2\n    line3\n"
    out = replace_most_similar_chunk(whole, "  line1\n  line2\n\n", "  new1\n  new2\n")
    assert out == "    new1\n    new2\n    line3\n"


def test_new_tiers_do_not_match_genuinely_different_text():
    """Whitespace normalization must not paper over a real content mismatch:
    'item' vs 'items' survives the full-trim comparison."""
    whole = "def compute_total(items):\n    return sum(items)\n"
    part = "def   compute_total(item):\n    return   sum(item)\n"
    assert replace_most_similar_chunk(whole, part, "x\n") is None


def test_trim_tier_ambiguous_match_is_refused():
    """Two windows equal only after a full trim - and equal to each other
    only there - must still raise, not silently pick one. require_unique is
    what the real modify_file tool always passes (tool_executor.py); without
    it, any tier legitimately takes the first occurrence (see
    test_exact_match_replaces_first_occurrence_only)."""
    whole = "    x = 1  \n    y = 2\n    x = 1\t\n"
    with pytest.raises(AmbiguousEdit):
        replace_most_similar_chunk(whole, "    x = 1\n", "    x = 9\n", require_unique=True)


# -- locate_anchored_span (locator only, never an applier) ----------------

def test_anchored_span_brackets_a_quote_the_ladder_refuses():
    """The live failure mode of run w7zoeszt (16 refusals): the quote is
    indented DEEPER than the file. _uniform_indent_prefix rejects a negative
    delta, so every ladder tier declines - the anchors still bracket it."""
    whole = '@router.post("/x/")\nasync def handler(\n    a: int,\n):\n    """Doc."""\n'
    part = '    @router.post("/x/")\nasync def handler(\n    a: int,\n):\n    """Doc."""\n'
    assert replace_most_similar_chunk(whole, part, "x\n", require_unique=True) is None
    assert locate_anchored_span(whole, part) == (1, 5)


def test_anchored_span_tolerates_lines_the_quote_skipped():
    """A quote that omits interior lines can never match a tier; its first and
    last lines still name the region (run mbzbzhq9 t117)."""
    whole = "".join(f"v{n} = {n}\n" for n in range(10))
    part = "".join(f"v{n} = {n}\n" for n in [0, 1, 2, 3, 4, 5, 6, 9])
    assert replace_most_similar_chunk(whole, part, "x\n", require_unique=True) is None
    assert locate_anchored_span(whole, part) == (1, 10)


def test_anchored_span_refuses_two_candidates():
    whole = "start\nmid\nend\nfiller\nstart\nmid\nend\n"
    assert locate_anchored_span(whole, "start\nmid\nend\n") is None


def test_anchored_span_refuses_when_the_block_size_is_out_of_range():
    """Anchors alone are not enough: a candidate more than 25% off the quote's
    length is a different region that happens to share its edges."""
    whole = "open\n" + "".join(f"body{n}\n" for n in range(20)) + "close\n"
    assert locate_anchored_span(whole, "open\nbody0\nbody1\nclose\n") is None
    # ...and within 25% it is accepted.
    tight = "open\nx\ny\nz\nclose\n"
    assert locate_anchored_span(tight, "open\na\nb\nclose\n") == (1, 5)


def test_anchored_span_ignores_boundary_blank_lines_and_line_numbers():
    whole = "def f():\n    return 1\n"
    assert locate_anchored_span(whole, "\n\ndef f():\n    return 1\n\n") == (1, 2)
    assert locate_anchored_span(whole, "   1| def f():\n   2|     return 1\n") == (1, 2)


def test_anchored_span_declines_a_quote_with_no_anchor_in_the_file():
    whole = "def f():\n    return 1\n"
    assert locate_anchored_span(whole, "def g():\n    return 2\n") is None
    assert locate_anchored_span(whole, "\n\n") is None


def test_anchored_span_is_a_locator_and_never_applies():
    """The apply ladder must not gain an anchor tier: the module docstring
    records why similarity-driven application stays out of this file."""
    whole = "head\nORIGINAL BODY\ntail\n"
    part = "head\nSOMETHING ELSE ENTIRELY\ntail\n"
    assert locate_anchored_span(whole, part) == (1, 3)
    assert replace_most_similar_chunk(whole, part, "x\n", require_unique=True) is None


# -- tier 8: the quote's FIRST line alone is over-indented ----------------
# Calibrated 2026-09-20 over 411 refused old_text values from 197 completed
# runs: of the 20 Qwen misses with a window matching modulo whitespace, 18 are
# this shape and none is a uniform shift, so tier 2 cannot reach them.

_STUB_ROUTER = (
    '@router.post("/bill/{bill_id}/methods/registerPayment/", tags=["Bill Methods"])\n'
    "async def execute_bill_registerPayment(\n"
    "    bill_id: int,\n"
    "):\n"
    '    """Execute the registerPayment method on a Bill instance.\n'
    '    """\n'
    "    raise HTTPException(status_code=501)\n"
)
# The quote as Qwen sends it: the decorator reconstructed at indent 4 above a
# body at 0, and new_text repeating the same mistake.
_OVERINDENTED_QUOTE = (
    '    @router.post("/bill/{bill_id}/methods/registerPayment/", tags=["Bill Methods"])\n'
    "async def execute_bill_registerPayment(\n"
    "    bill_id: int,\n"
    "):\n"
    '    """Execute the registerPayment method on a Bill instance.\n'
    '    """\n'
)
_OVERINDENTED_REPLACEMENT = (
    '    @router.post("/bill/{bill_id}/methods/registerPayment/", tags=["Bill Methods"])\n'
    "async def execute_bill_registerPayment(\n"
    "    bill_id: int,\n"
    "    database: Session = Depends(get_db),\n"
    "):\n"
    '    """Execute the registerPayment method on a Bill instance.\n'
    '    """\n'
)


def test_first_line_only_overindent_is_forgiven():
    out = replace_most_similar_chunk(
        _STUB_ROUTER, _OVERINDENTED_QUOTE, _OVERINDENTED_REPLACEMENT, require_unique=True,
    )
    assert out == (
        '@router.post("/bill/{bill_id}/methods/registerPayment/", tags=["Bill Methods"])\n'
        "async def execute_bill_registerPayment(\n"
        "    bill_id: int,\n"
        "    database: Session = Depends(get_db),\n"
        "):\n"
        '    """Execute the registerPayment method on a Bill instance.\n'
        '    """\n'
        "    raise HTTPException(status_code=501)\n"
    )


def test_first_line_overindent_comes_off_the_replacement_too():
    """The model repeats the mistake in new_text. Writing it verbatim puts a
    decorator at indent 4 above a module-level def - 17 of the 18 live Python
    rescues would not parse."""
    out = replace_most_similar_chunk(
        _STUB_ROUTER, _OVERINDENTED_QUOTE, _OVERINDENTED_REPLACEMENT, require_unique=True,
    )
    assert out.startswith("@router.post(")
    ast.parse("from x import *\n" + out)
    with pytest.raises(IndentationError):
        ast.parse("from x import *\n" + _STUB_ROUTER.replace(
            _OVERINDENTED_QUOTE.lstrip(), _OVERINDENTED_REPLACEMENT, 1))


def test_first_line_tier_corrects_an_under_indented_first_line():
    """Superseded 2026-09-20: the tier used to refuse this direction on the
    principle that it "never adds indent it invented". Two of the three real
    ladder gaps left in the Qwen corpus are exactly this shape (App.jsx t51,
    bill_methods.py t16), and tier 2/6 cannot take them - the shift is not
    uniform, so their single-prefix rule rejects the window. The file's own
    leading whitespace is restored on the replacement, never invented; the
    contradicting-body case stays refused (see test_edit_apply_weak_model)."""
    whole = "    @deco\n    def f():\n        pass\n"
    assert replace_most_similar_chunk(
        whole, "@deco\n    def f():\n", "@deco2\n    def f():\n", require_unique=True,
    ) == "    @deco2\n    def f():\n        pass\n"


def test_first_line_tier_refuses_when_a_second_line_also_shifts():
    """One of the 2 irregular live cases shifts the first TWO lines. Refused:
    the tier is first-line-only by construction."""
    whole = "#--- Relationships\nGuest.x = relationship()\nBooking.y = relationship()\n"
    part = "    #--- Relationships\n    Guest.x = relationship()\nBooking.y = relationship()\n"
    assert replace_most_similar_chunk(whole, part, "z = 1\n", require_unique=True) is None


def test_first_line_tier_refuses_a_replacement_without_the_same_prefix():
    """An unindented replacement first line could be the model's intent or a
    second mistake. We cannot tell, so we do not guess."""
    whole = "@deco\ndef f():\n    pass\n"
    assert replace_most_similar_chunk(
        whole, "    @deco\ndef f():\n", "@deco2\ndef f():\n", require_unique=True,
    ) is None


def test_first_line_tier_ambiguity_is_refused():
    whole = "    x = 1\n    y = 2\nx = 1\n    y = 2\n"
    with pytest.raises(AmbiguousEdit):
        replace_most_similar_chunk(
            whole, "        x = 1\n    y = 2\n", "        x = 9\n    y = 8\n",
            require_unique=True,
        )


def test_first_line_tier_does_not_match_a_different_tail():
    whole = "@deco\ndef f():\n    return 1\n"
    assert replace_most_similar_chunk(
        whole, "    @deco\ndef g():\n", "    @deco\ndef h():\n", require_unique=True,
    ) is None


def test_modify_file_applies_the_overindented_first_line(tmp_path):
    """The whole point: run w7zoeszt resent these six edits from turn 19 to
    turn 38 and never landed one."""
    _seed(tmp_path, "bill_methods.py", _STUB_ROUTER)
    ex = ToolExecutor(workspace=str(tmp_path))
    res = _call(ex, "modify_file", {
        "path": "bill_methods.py",
        "old_text": _OVERINDENTED_QUOTE,
        "new_text": _OVERINDENTED_REPLACEMENT,
    })
    assert res.get("status") == "modified", res
    with open(os.path.join(str(tmp_path), "bill_methods.py"), encoding="utf-8") as f:
        written = f.read()
    # at column 0, not the indent 4 the model quoted it at
    assert "\n@router.post(" in written and "    @router.post(" not in written
    assert "    database: Session = Depends(get_db),\n" in written
