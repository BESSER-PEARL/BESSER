"""A truncated log record must never be readable as code.

Cost of the old marker, 2026-09-18. ``_sanitize_for_log`` cut every string
over 500 chars to ``v[:500] + "..."`` and was applied to the trace, the
checkpoint's ``tool_calls_log`` and the recipe — the only records of what the
model actually sent. Two separate investigations then read that trailing
``...`` as the MODEL's text:

  * run 57160293 — 16 identical failed edits were diagnosed as an elision;
    every one of them is exactly 503 chars, i.e. 500 + the marker.
  * the run behind ``test_elided_edit.py`` — "``...`` in 37/38 old_text" is
    the same artifact; each of those is 503 chars too, and no generated file
    in any captured workspace contains a bare ``...`` line.

Both diagnoses were wrong, and both cost a day's reasoning. The marker must
be unmistakable, and the untruncated value must survive somewhere.
"""
from __future__ import annotations

from besser.generators.llm.edit_apply import find_elision
from besser.generators.llm.orchestrator import _sanitize_for_log


def _long_python(n: int = 600) -> str:
    """Realistic over-budget tool input: plain code, no ellipsis in it."""
    body = "        db_booking = database.query(Booking).first()\n"
    return ("    try:\n" + body * 40)[:n]


def test_a_truncated_value_is_not_mistaken_for_an_elision():
    """The exact failure: find_elision must not fire on OUR marker."""
    src = _long_python()
    assert find_elision(src) is None, "fixture must be clean before truncation"
    out = _sanitize_for_log({"old_text": src})["old_text"]
    assert out != src, "fixture must actually be over the budget"
    assert find_elision(out) is None, (
        "the truncation marker reads as a model-written elision: " + out[-60:]
    )


def test_the_marker_says_it_is_a_truncation():
    out = _sanitize_for_log({"old_text": _long_python()})["old_text"]
    assert not out.endswith("..."), out[-60:]
    assert "truncated" in out.lower()


def test_the_marker_reports_how_much_was_cut():
    src = _long_python(600)
    out = _sanitize_for_log({"old_text": src})["old_text"]
    assert str(len(src) - 500) in out, out[-60:]


def test_the_marker_carries_a_fingerprint_so_repeats_are_comparable():
    """Two different 600-char inputs must not produce the same record.

    Run 57160293 was read as 16 byte-identical resends. That happened to be
    true, but the record could not have shown otherwise: everything over the
    budget rendered as the same 503 chars.
    """
    a = _sanitize_for_log({"old_text": _long_python(600)})["old_text"]
    b = _sanitize_for_log({"old_text": _long_python(600).replace("first", "one__")})["old_text"]
    assert a != b


def test_a_short_value_is_untouched():
    for value in ("", "def f():\n    pass\n", "x" * 500):
        assert _sanitize_for_log({"k": value})["k"] == value


def test_non_string_values_and_non_dicts_pass_through():
    assert _sanitize_for_log({"n": 3, "b": True, "none": None}) == {
        "n": 3, "b": True, "none": None,
    }
    assert _sanitize_for_log(["a", "b"]) == ["a", "b"]
