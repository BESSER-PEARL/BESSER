"""Regression tests for BESSER-PEARL/BESSER#202.

The OCL parser's raw ANTLR diagnostics ("mismatched input 'then' expecting
<EOF>", "missing ')' at '<EOF>'", etc.) were uninformative and sometimes
leaked internal parser state (``pop from empty list``). The error listener
in ``besser/BUML/notations/ocl/error_handling.py`` now rewrites these into
OCL-level hints.
"""

import pytest

from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    _parse_only,
)


@pytest.mark.parametrize("expr, expected_fragment", [
    # Missing 'if' — the motivating example from issue #202.
    ("context Department inv:self.size>5 then self.boss.salary>2000 else true",
     "forget the 'if'"),
    # 'then' branch ends without 'else'.
    ("context Department inv: if self.size > 5 then true",
     "before the 'else' branch"),
    # Missing closing 'endif'.
    ("context Department inv: if self.size > 5 then true else false",
     "Missing 'endif'"),
    # Stray 'endif' without a matching 'if'.
    ("context Department inv: self.size > 5 endif",
     "'endif' without a matching 'if'"),
    # Unbalanced parentheses.
    ("context Department inv: (self.size > 5",
     "Missing closing ')'"),
    ("context Department inv: self.size > 5)",
     "no matching opening '('"),
    # Incomplete trailing expression.
    ("context Department inv: self.size +",
     "operand on the right is missing"),
    # Malformed header.
    ("Department inv: self.size > 5",
     "must start with 'context"),
])
def test_friendly_ocl_error(expr, expected_fragment):
    """Each broken expression must surface the targeted hint."""
    with pytest.raises(BOCLSyntaxError) as exc_info:
        _parse_only(expr)
    assert expected_fragment in str(exc_info.value)


@pytest.mark.parametrize("expr", [
    "context Department inv: self.size > 5",
    "context Department inv: if self.size > 5 then true else false endif",
    "context Department inv: self.employees->size() > 5",
])
def test_valid_ocl_still_parses(expr):
    """The error-translation layer must not break valid OCL."""
    _parse_only(expr)


def test_error_message_contains_location():
    """Every translated error must still include line + column so the user
    can find the offending token even when the hint is terse."""
    with pytest.raises(BOCLSyntaxError) as exc_info:
        _parse_only("context Department inv:self.size>5 then self.boss.salary>2000 else true")
    message = str(exc_info.value)
    assert "line 1" in message
    assert "column" in message


def test_single_error_has_no_multi_line_preamble():
    """Single-error messages should not read as 'Multiple OCL syntax errors:'
    — that preamble is reserved for the >1 errors case."""
    try:
        _parse_only("Department inv: self.size > 5")
    except BOCLSyntaxError as exc:
        assert "Multiple OCL syntax errors" not in str(exc)
    else:
        pytest.fail("expected BOCLSyntaxError")
