"""Apply-ladder tiers calibrated on refused Qwen3-30B-A3B edits.

Every case here is a real refused ``old_text`` taken from the run corpus under
``verification/spec-iterations`` (362 runs, 10,545 edit calls), replayed against
the file state reconstructed for that turn.

The negatives matter as much as the positives: a weak model's quote must still
be refused when the *replacement* contradicts it, because an edit that lands at
the wrong indent is worse than one that misses.
"""

import pytest

from besser.generators.llm.edit_apply import (
    AmbiguousEdit,
    describe_escape_mismatch,
    replace_most_similar_chunk,
)


# -- tier 8b: the quote's first line is UNDER-indented --------------------
#
# Tier 8 already forgave an over-indented first line. The corpus shows the
# mirror shape just as often: the model drops a space or a whole level from
# the opening line and copies the rest verbatim.

def test_first_line_under_indented_by_one_space():
    """Live: run ...-0zafgaos frontend/src/App.jsx turn 51.

    ``<Routes>`` quoted at 7 spaces against a file holding it at 8; every
    other line byte-exact. new_text repeats the same 7-space opening, so the
    file's own extra space is restored on the replacement.
    """
    whole = (
        "      return (\n"
        "        <Routes>\n"
        '          <Route path="/persons" element={<PersonList />} />\n'
        "        </Routes>\n"
        "      );\n"
    )
    part = (
        "       <Routes>\n"
        '          <Route path="/persons" element={<PersonList />} />\n'
        "        </Routes>\n"
    )
    replace = (
        "       <Routes>\n"
        '          <Route path="/" element={<Navigate to="/persons" />} />\n'
        '          <Route path="/persons" element={<PersonList />} />\n'
        "        </Routes>\n"
    )
    assert replace_most_similar_chunk(whole, part, replace) == (
        "      return (\n"
        "        <Routes>\n"
        '          <Route path="/" element={<Navigate to="/persons" />} />\n'
        '          <Route path="/persons" element={<PersonList />} />\n'
        "        </Routes>\n"
        "      );\n"
    )


def test_first_line_under_indented_by_a_full_level():
    whole = (
        "class C:\n"
        "    def run(self):\n"
        "        value = compute()\n"
        "        return value\n"
    )
    part = (
        "    def run(self):\n"[4:]  # quoted at column 0, body at 8
        + "        value = compute()\n"
        "        return value\n"
    )
    replace = (
        "def run(self):\n"
        "        value = compute() * 2\n"
        "        return value\n"
    )
    assert replace_most_similar_chunk(whole, part, replace) == (
        "class C:\n"
        "    def run(self):\n"
        "        value = compute() * 2\n"
        "        return value\n"
    )


def test_under_indented_first_line_refused_when_body_contradicts_it():
    """Live: run ...-1f6aecc4 web_app/backend/routers/bill_methods.py turn 16.

    The quote's first line sits at 4 while the rest of the quote matched the
    file exactly at 8 - but the whole *replacement* is authored at 4. Shifting
    only its first line would splice indent-4 code into an indent-8 block, so
    the tier must decline rather than produce that.
    """
    whole = (
        "    try:\n"
        "        # Bill.registerPayment: no body in the model\n"
        "        raise HTTPException(\n"
        "            status_code=501,\n"
        "        )\n"
    )
    part = (
        "    # Bill.registerPayment: no body in the model\n"
        "        raise HTTPException(\n"
        "            status_code=501,\n"
        "        )\n"
    )
    replace = (
        "    # Bill.registerPayment: execute\n"
        "    try:\n"
        "        settle(bill)\n"
    )
    assert replace_most_similar_chunk(whole, part, replace) is None


def test_under_indented_first_line_refused_when_replacement_did_not_repeat_it():
    """The model mis-indented the quote but authored the replacement at the
    file's own indent. Shifting it again would double the correction."""
    whole = "class C:\n    def run(self):\n        return 1\n"
    part = "def run(self):\n        return 1\n"
    replace = "    def run(self):\n        return 2\n"
    assert replace_most_similar_chunk(whole, part, replace) is None


def test_first_line_indent_tier_still_requires_a_unique_window():
    whole = (
        "class A:\n"
        "    def run(self):\n"
        "        return 1\n"
        "class B:\n"
        "    def run(self):\n"
        "        return 1\n"
    )
    part = "def run(self):\n        return 1\n"
    replace = "def run(self):\n        return 2\n"
    with pytest.raises(AmbiguousEdit):
        replace_most_similar_chunk(whole, part, replace, (), True)


# -- the doubled-backslash quote: diagnosed, never applied ----------------
#
# 11 refused quotes in the corpus spell a regex with a doubled backslash where
# the file has a single one. It looks like a one-line tier, and it is a trap:
# BESSER's pydantic generator emits the same regex twice, raw in the check and
# re-escaped in the message right below it, so a quote spanning both carries
# BOTH conventions. Replayed against the three live cases, a whole-quote
# un-double rescues none of them - and an un-doubled new_text would have
# written a DIFFERENT regex into a file that still parses. So the ladder names
# the mistake instead of guessing at it.

_GENERATED_VALIDATOR = (
    "    @field_validator('email')\n"
    "    def check_email(cls, v):\n"
    "        if not (re.fullmatch(r'^[^\\s@]+@[^\\s@]+\\.[A-Za-z]{2,}$', v) is not None):\n"
    '            raise ValueError("email must match \'^[^\\\\s@]+@[^\\\\s@]+\\\\.[A-Za-z]{2,}$\'")\n'
    "        return v\n"
)
# the check line carries single backslashes, the message line doubled ones
assert "r'^[^\\s@]" in _GENERATED_VALIDATOR
assert "'^[^\\\\s@]" in _GENERATED_VALIDATOR


def test_double_escaped_quote_is_never_applied():
    """Live: backend/pydantic_classes.py turns 30 / 56 / 83."""
    part = (
        "        if not (re.fullmatch(r'^[^\\\\s@]+@[^\\\\s@]+\\\\.[A-Za-z]{2,}$', v) is not None):\n"
        "            raise ValueError('bad email')\n"
    )
    assert replace_most_similar_chunk(_GENERATED_VALIDATOR, part, "x = 1\n") is None


def test_double_escaped_quote_is_diagnosed():
    part = "        if not (re.fullmatch(r'^[^\\\\s@]+@[^\\\\s@]+\\\\.[A-Za-z]{2,}$', v) is not None):\n"
    hint = describe_escape_mismatch(_GENERATED_VALIDATOR, part)
    assert hint is not None
    assert "escapes its backslashes twice" in hint
    assert "line 3" in hint


def test_escape_diagnostic_is_silent_on_a_correctly_quoted_line():
    """The message line really does carry doubled backslashes in the file;
    quoting it that way is correct, not a mistake."""
    part = '            raise ValueError("email must match \'^[^\\\\s@]+@[^\\\\s@]+\\\\.[A-Za-z]{2,}$\'")\n'
    assert describe_escape_mismatch(_GENERATED_VALIDATOR, part) is None


def test_escape_diagnostic_is_silent_when_nothing_is_doubled():
    assert describe_escape_mismatch(_GENERATED_VALIDATOR, "        return v\n") is None
