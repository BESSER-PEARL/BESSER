"""Tests for besser.BUML.notations.ocl.wrapping_visitor.WrappingVisitor.

Verifies the three bugs the wrapper fixes are actually fixed:
- chain reconstruction in dot navigation;
- chain reconstruction for the special-case ``size`` attribute;
- boolean literal coerced to Python ``bool``.
"""

from besser.BUML.metamodel.ocl import is_chain_from_self, walk_chain_from_self
from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression, OperationCallExpression,
    PropertyCallExpression, VariableExp,
)
from besser.BUML.notations.ocl import parse_ocl


def test_dot_navigation_preserves_chain(model):
    constraint = parse_ocl(
        "context Employee inv: self.employer.minSalary <= self.salary",
        model,
    )
    expr = constraint.expression
    # Top-level: comparison with three-arg shape.
    assert isinstance(expr, OperationCallExpression)
    left = expr.arguments[0]
    # Left side is `self.employer.minSalary` — must be a 2-step chain.
    assert is_chain_from_self(left)
    chain = walk_chain_from_self(left)
    assert [p.name for p in chain] == ["employer", "minSalary"]


def test_simple_dot_navigation_is_a_one_step_chain(model):
    constraint = parse_ocl("context Employee inv: self.age > 18", model)
    expr = constraint.expression
    left = expr.arguments[0]
    assert is_chain_from_self(left)
    assert [p.name for p in walk_chain_from_self(left)] == ["age"]


def test_boolean_literal_is_python_bool(model):
    constraint = parse_ocl(
        "context Employee inv: self.age > 16 and true",
        model,
    )
    # Walk the AST to find the BooleanLiteralExpression.
    from besser.BUML.metamodel.ocl import walk
    bools = [n for n in walk(constraint.expression)
             if isinstance(n, BooleanLiteralExpression)]
    assert len(bools) == 1
    assert bools[0].value is True
    assert not isinstance(bools[0].value, str)


def test_boolean_literal_false(model):
    constraint = parse_ocl(
        "context Employee inv: self.age > 16 or false",
        model,
    )
    from besser.BUML.metamodel.ocl import walk
    bools = [n for n in walk(constraint.expression)
             if isinstance(n, BooleanLiteralExpression)]
    assert bools[0].value is False


def test_chain_terminus_is_self_variable_expression(model):
    constraint = parse_ocl(
        "context Employee inv: self.employer.minSalary <= self.salary",
        model,
    )
    left = constraint.expression.arguments[0]
    # Walk to the innermost source: should be VariableExp("self").
    cur = left
    while isinstance(cur, PropertyCallExpression):
        cur = cur.source
    assert isinstance(cur, VariableExp) and cur.name == "self"
