"""Tests for besser.BUML.metamodel.ocl.walk."""

from besser.BUML.metamodel.ocl import walk
from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, LoopExp, IteratorExp, IfExp,
    PropertyCallExpression, VariableExp,
    IntegerLiteralExpression, BooleanLiteralExpression, InfixOperator,
)
from besser.BUML.metamodel.structural import Property, IntegerType


def _int(v):
    return IntegerLiteralExpression("NP", v)


def _comp(left, op, right):
    return OperationCallExpression(
        name="Operation", operation=op,
        arguments=[left, InfixOperator(op), right],
    )


def test_walk_none_yields_nothing():
    assert list(walk(None)) == []


def test_walk_leaf():
    n = _int(7)
    assert list(walk(n)) == [n]


def test_walk_comparison_post_order():
    left = _int(3)
    right = _int(5)
    cmp = _comp(left, "<", right)
    nodes = list(walk(cmp))
    assert nodes == [left, right, cmp]


def test_walk_skips_infix_operator():
    left = _int(1)
    right = _int(2)
    cmp = _comp(left, "=", right)
    for n in walk(cmp):
        assert not isinstance(n, InfixOperator)


def test_walk_boolean_binary():
    left = _comp(_int(1), "<", _int(2))
    right = _comp(_int(3), ">", _int(4))
    and_node = OperationCallExpression(
        name="AND_BINARY", operation="and", arguments=[left, right]
    )
    nodes = list(walk(and_node))
    # post-order: leaves first, then comparisons, then AND
    assert nodes[-1] is and_node
    assert left in nodes and right in nodes


def test_walk_unary_not():
    operand = _comp(_int(1), "<", _int(2))
    not_node = OperationCallExpression(
        name="unary_not", operation="not", arguments=[operand]
    )
    nodes = list(walk(not_node))
    assert nodes[-1] is not_node
    assert operand in nodes


def test_walk_property_call_chain():
    p1 = Property("r1", IntegerType)
    p2 = Property("r2", IntegerType)
    self_var = VariableExp("self", None)
    pce1 = PropertyCallExpression("r1", p1)
    pce1.source = self_var
    pce2 = PropertyCallExpression("r2", p2)
    pce2.source = pce1
    nodes = list(walk(pce2))
    assert nodes == [self_var, pce1, pce2]


def test_walk_loop_visits_source_and_body_not_iterator():
    src = VariableExp("self", None)
    body = _comp(_int(1), ">", _int(0))
    loop = LoopExp("forAll", None)
    loop.source = src
    loop.addIterator(IteratorExp("e", "NotMentioned"))
    loop.add_body(body)
    nodes = list(walk(loop))
    # IteratorExp is a variable declaration, not visited as a child
    assert all(not isinstance(n, IteratorExp) for n in nodes)
    assert src in nodes and body in nodes
    assert nodes[-1] is loop


def test_walk_if_visits_all_three_branches():
    cond = BooleanLiteralExpression("NP", True)
    then_e = _int(1)
    else_e = _int(2)
    if_node = IfExp("if", "IfExpression")
    if_node.ifCondition = cond
    if_node.thenExpression = then_e
    if_node.elseCondition = else_e
    nodes = list(walk(if_node))
    assert cond in nodes and then_e in nodes and else_e in nodes
    assert nodes[-1] is if_node
