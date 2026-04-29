"""Tests for besser.BUML.metamodel.ocl.substitute."""

from besser.BUML.metamodel.ocl import substitute, ScopeStack
from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression, IfExp, InfixOperator, IntegerLiteralExpression,
    IteratorExp, LoopExp, OperationCallExpression, PropertyCallExpression,
    VariableExp,
)
from besser.BUML.metamodel.structural import IntegerType, Property


def _int(v):
    return IntegerLiteralExpression("NP", v)


def _comp(left, op, right):
    return OperationCallExpression(
        name="Operation", operation=op,
        arguments=[left, InfixOperator(op), right],
    )


def _and(left, right):
    return OperationCallExpression(
        name="AND_BINARY", operation="and", arguments=[left, right],
    )


# ---------------------------------------------------------------------------
# Free-variable substitution
# ---------------------------------------------------------------------------

def test_replaces_free_variable_in_leaf():
    expr = VariableExp("e", None)
    repl = VariableExp("self", None)
    out = substitute(expr, "e", repl)
    assert isinstance(out, VariableExp) and out.name == "self"


def test_unrelated_variable_is_unchanged():
    expr = VariableExp("x", None)
    repl = VariableExp("self", None)
    out = substitute(expr, "e", repl)
    assert isinstance(out, VariableExp) and out.name == "x"
    # Distinct identity — substitute returns a clone for non-matches.
    assert out is not expr


def test_replaces_variable_inside_comparison():
    e = VariableExp("e", None)
    expr = _comp(e, ">", _int(16))
    out = substitute(expr, "e", VariableExp("self", None))
    new_left = out.arguments[0]
    assert isinstance(new_left, VariableExp) and new_left.name == "self"


def test_substitute_does_not_mutate_input():
    e = VariableExp("e", None)
    expr = _comp(e, ">", _int(16))
    _ = substitute(expr, "e", VariableExp("self", None))
    # Original AST untouched.
    assert expr.arguments[0] is e
    assert e.name == "e"


def test_substitute_with_property_chain_replacement():
    # X[self.r1/v]: replace `e` with `self.r1`.
    p = Property("r1", IntegerType)
    chain = PropertyCallExpression("r1", p)
    chain.source = VariableExp("self", None)

    body = _comp(VariableExp("e", None), ">", _int(0))
    out = substitute(body, "e", chain)

    new_left = out.arguments[0]
    assert isinstance(new_left, PropertyCallExpression)
    assert new_left.property is p  # Property reference shared
    assert isinstance(new_left.source, VariableExp)
    assert new_left.source.name == "self"


def test_replacement_is_cloned_per_occurrence():
    # Two occurrences of `e` must each get a fresh clone — no aliasing.
    e1 = VariableExp("e", None)
    e2 = VariableExp("e", None)
    expr = _and(e1, e2)
    repl = VariableExp("self", None)
    out = substitute(expr, "e", repl)
    assert out.arguments[0] is not out.arguments[1]


# ---------------------------------------------------------------------------
# Capture avoidance
# ---------------------------------------------------------------------------

def test_inner_loop_shadows_outer_substitution():
    # forAll(e | e > 0)  — substituting `e` should NOT touch the body
    # because the inner loop binds `e`.
    inner_body = _comp(VariableExp("e", None), ">", _int(0))
    loop = LoopExp("forAll", None)
    loop.source = VariableExp("self", None)
    loop.addIterator(IteratorExp("e", "NotMentioned"))
    loop.add_body(inner_body)

    out = substitute(loop, "e", VariableExp("REPLACED", None))

    # Source is a free `self`, not affected by `e` substitution either way.
    assert out.source.name == "self"
    # Body's `e` is bound — must NOT have been replaced.
    new_body_left = out.body[0].arguments[0]
    assert isinstance(new_body_left, VariableExp)
    assert new_body_left.name == "e"  # untouched


def test_substitution_reaches_loop_source():
    # `e->forAll(x | x > 0)` — substituting `e` should affect the source.
    body = _comp(VariableExp("x", None), ">", _int(0))
    loop = LoopExp("forAll", None)
    loop.source = VariableExp("e", None)
    loop.addIterator(IteratorExp("x", "NotMentioned"))
    loop.add_body(body)

    out = substitute(loop, "e", VariableExp("self", None))
    assert isinstance(out.source, VariableExp) and out.source.name == "self"
    # x stays x in the body
    assert out.body[0].arguments[0].name == "x"


def test_nested_loop_with_distinct_iterator_names_substitutes_correctly():
    # forAll(e | forAll(d | e > d))
    # Substituting `e` reaches `e > d` because only `d` is bound by inner.
    inner_body = _comp(VariableExp("e", None), ">", VariableExp("d", None))
    inner = LoopExp("forAll", None)
    inner.source = VariableExp("ignored", None)
    inner.addIterator(IteratorExp("d", "NotMentioned"))
    inner.add_body(inner_body)

    outer = LoopExp("forAll", None)
    outer.source = VariableExp("ignored2", None)
    outer.addIterator(IteratorExp("e", "NotMentioned"))
    outer.add_body(inner)

    # Reach inside both layers because of the explicit scope stack:
    # we want substitute(outer.body[0], "e", ...) to substitute through
    # the inner loop because `e` is not shadowed there.
    out = substitute(inner, "e", VariableExp("REPLACED", None))
    new_e = out.body[0].arguments[0]
    assert new_e.name == "REPLACED"


def test_nested_loop_with_same_iterator_name_does_not_substitute():
    # forAll(e | e > 0) — substituting `e` in the loop is a no-op for the body.
    inner_body = _comp(VariableExp("e", None), ">", _int(0))
    loop = LoopExp("forAll", None)
    loop.source = VariableExp("ignored", None)
    loop.addIterator(IteratorExp("e", "NotMentioned"))
    loop.add_body(inner_body)

    out = substitute(loop, "e", VariableExp("REPLACED", None))
    # Body's `e` is bound by this loop's iterator; not substituted.
    assert out.body[0].arguments[0].name == "e"


# ---------------------------------------------------------------------------
# IfExp and other forms
# ---------------------------------------------------------------------------

def test_substitute_inside_if():
    cond = VariableExp("e", None)
    then_e = VariableExp("e", None)
    else_e = _int(0)
    if_node = IfExp("if", "IfExpression")
    if_node.ifCondition = cond
    if_node.thenExpression = then_e
    if_node.elseCondition = else_e

    out = substitute(if_node, "e", BooleanLiteralExpression("NP", True))
    assert isinstance(out.ifCondition, BooleanLiteralExpression)
    assert out.ifCondition.value is True
    assert isinstance(out.thenExpression, BooleanLiteralExpression)
    # else branch unchanged
    assert isinstance(out.elseCondition, IntegerLiteralExpression)


def test_substitute_through_property_chain():
    # e.r1 — substitute e -> self
    p = Property("r1", IntegerType)
    pce = PropertyCallExpression("r1", p)
    pce.source = VariableExp("e", None)
    out = substitute(pce, "e", VariableExp("self", None))
    assert out.property is p
    assert isinstance(out.source, VariableExp) and out.source.name == "self"


# ---------------------------------------------------------------------------
# ScopeStack unit
# ---------------------------------------------------------------------------

def test_scope_stack_push_pop_is_bound():
    s = ScopeStack()
    assert not s.is_bound("e")
    s.push(["e"])
    assert s.is_bound("e")
    s.push(["x"])
    assert s.is_bound("e") and s.is_bound("x")
    s.pop()
    assert s.is_bound("e") and not s.is_bound("x")
    s.pop()
    assert not s.is_bound("e")
