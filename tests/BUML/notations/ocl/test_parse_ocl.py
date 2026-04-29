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


# Tests for OCLConstraint.ast / .expression separation (Pre-work B)

def test_ocl_constraint_ast_is_the_parsed_tree(model):
    """``OCLConstraint.ast`` exposes the parsed AST."""
    from besser.BUML.metamodel.ocl.ocl import OCLExpression
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    assert isinstance(constraint.ast, OCLExpression)


def test_ocl_constraint_expression_is_source_text(model):
    """``OCLConstraint.expression`` returns pretty-printed OCL source text."""
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    assert isinstance(constraint.expression, str)
    assert "self.age" in constraint.expression


def test_ocl_constraint_rejects_non_ast_expression(model):
    """Constructing OCLConstraint with a string raises TypeError."""
    employee = next(t for t in model.types if getattr(t, "name", None) == "Employee")
    with pytest.raises(TypeError, match="OCLExpression"):
        OCLConstraint(name="bad", context=employee, expression="self.age > 16")


def test_ocl_constraint_ast_setter_refreshes_expression(model):
    """Reassigning .ast updates .expression to a fresh pretty-print."""
    c1 = parse_ocl("context Employee inv: self.age > 16", model)
    c2 = parse_ocl("context Employee inv: self.age > 99", model)
    c1.ast = c2.ast
    assert "99" in c1.expression


def test_ocl_constraint_ast_setter_rejects_non_ast(model):
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    with pytest.raises(TypeError, match="OCLExpression"):
        constraint.ast = "not an ast"


# Tests for SourceLocation on OCLExpression (Pre-work C)

def test_parsed_ast_root_has_source_location(model):
    """The root AST node carries line/col/source_text from the parser."""
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    root = constraint.ast
    assert root.line == 1
    assert root.col is not None
    assert root.source_text is not None
    assert "self" in root.source_text


def test_parsed_ast_inner_nodes_have_distinct_locations(model):
    """Inner AST nodes get their own line/col, not inherited from the root."""
    from besser.BUML.metamodel.ocl import walk
    from besser.BUML.metamodel.ocl.ocl import (
        OCLExpression, VariableExp, IntegerLiteralExpression,
    )
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    nodes = [n for n in walk(constraint.ast) if isinstance(n, OCLExpression)]
    var = next(n for n in nodes if isinstance(n, VariableExp))
    intlit = next(n for n in nodes if isinstance(n, IntegerLiteralExpression))
    assert var.col != intlit.col, (
        f"VariableExp col {var.col} should differ from IntegerLiteralExpression col {intlit.col}"
    )


def test_source_location_setters_validate_type(model):
    """Setters reject wrong types but accept None."""
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    with pytest.raises(TypeError, match="line must be int"):
        constraint.ast.line = "1"
    with pytest.raises(TypeError, match="col must be int"):
        constraint.ast.col = "0"
    with pytest.raises(TypeError, match="source_text must be str"):
        constraint.ast.source_text = 42
    # None is allowed (clears the field)
    constraint.ast.line = None
    constraint.ast.col = None
    constraint.ast.source_text = None
    assert constraint.ast.line is None


def test_copy_location_from_helper(model):
    """copy_location_from copies missing fields without overwriting set ones."""
    from besser.BUML.metamodel.ocl.ocl import IntegerLiteralExpression
    src = parse_ocl("context Employee inv: self.age > 16", model).ast
    dst = IntegerLiteralExpression(name="lit", value=99)
    assert dst.line is None
    dst.copy_location_from(src)
    assert dst.line == src.line
    assert dst.col == src.col
    assert dst.source_text == src.source_text
    # Existing values not overwritten
    dst.line = 999
    dst.copy_location_from(src)
    assert dst.line == 999
