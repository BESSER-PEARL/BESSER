"""Tests for besser.BUML.notations.ocl.api.parse_ocl."""

import pytest

from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.notations.ocl import parse_ocl, BOCLSyntaxError


def test_parse_returns_ocl_constraint(model):
    result = parse_ocl("context Employee inv: self.age > 16", model)
    assert isinstance(result, OCLConstraint)
    assert result.context.name == "Employee"
    assert result.language == "OCL"


def test_parse_resolves_context_from_text(model):
    result = parse_ocl("context Department inv: self.name <> ''", model)
    assert result.context.name == "Department"


def test_parse_explicit_context_class_overrides_text(model):
    employee = next(t for t in model.types if getattr(t, "name", None) == "Employee")
    # No header in the text — but explicit context_class supplied.
    result = parse_ocl(
        "context Employee inv: self.age > 16",
        model,
        context_class=employee,
    )
    assert result.context is employee


def test_parse_raises_for_unknown_context(model):
    with pytest.raises(ValueError, match="Nonexistent"):
        parse_ocl("context Nonexistent inv: 1 = 1", model)


def test_parse_raises_for_missing_header(model):
    # No `context X inv:` prefix and no explicit context_class → ValueError.
    with pytest.raises(ValueError, match="header"):
        parse_ocl("self.age > 16", model)


def test_parse_raises_bocl_syntax_error_on_lex_or_parse_failure(model):
    # Unbalanced parenthesis → ANTLR error.
    with pytest.raises(BOCLSyntaxError):
        parse_ocl("context Employee inv: (self.age > 16", model)


def test_parse_raises_bocl_syntax_error_on_unresolved_property(model):
    # Property doesn't exist on the context class — must surface as
    # BOCLSyntaxError, not a bare Exception.
    with pytest.raises(BOCLSyntaxError, match="not found"):
        parse_ocl("context Employee inv: self.nonexistent > 16", model)


def test_parse_iterator_constraint(model):
    result = parse_ocl(
        "context Department inv: self.employee->forAll(e | e.age > 16)",
        model,
    )
    assert isinstance(result, OCLConstraint)
    assert result.context.name == "Department"
