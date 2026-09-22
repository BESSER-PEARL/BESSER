"""Tests for besser.BUML.metamodel.ocl.clone."""

from besser.BUML.metamodel.ocl import clone
from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, LoopExp, IteratorExp, IfExp,
    PropertyCallExpression, VariableExp, TypeExp,
    IntegerLiteralExpression, RealLiteralExpression,
    BooleanLiteralExpression, StringLiteralExpression,
    InfixOperator,
)
from besser.BUML.metamodel.structural import Property, IntegerType


def test_clone_none_returns_none():
    assert clone(None) is None


def test_clone_integer_literal_is_fresh():
    n = IntegerLiteralExpression("NP", 42)
    c = clone(n)
    assert c is not n
    assert c.value == 42


def test_clone_real_string_boolean_literals_round_trip():
    for n in [
        RealLiteralExpression("NP", 1.5),
        StringLiteralExpression("str", "hello"),
        BooleanLiteralExpression("NP", True),
        BooleanLiteralExpression("NP", "true"),  # tolerate legacy lexeme
    ]:
        c = clone(n)
        assert c is not n
        assert c.value == n.value


def test_clone_variable_exp():
    v = VariableExp("e", None)
    c = clone(v)
    assert c is not v
    assert c.name == "e"


def test_clone_type_exp():
    t = TypeExp("Employee", "Employee")
    c = clone(t)
    assert c is not t
    assert c.name == "Employee"


def test_clone_property_call_shares_property_reference():
    p = Property("age", IntegerType)
    pce = PropertyCallExpression("age", p)
    pce.source = VariableExp("self", None)
    c = clone(pce)
    assert c is not pce
    assert c.property is p  # Property is shared, not duplicated
    assert c.source is not pce.source
    assert c.source.name == "self"


def test_clone_operation_preserves_infix_operator_identity():
    infix = InfixOperator(">")
    op = OperationCallExpression(
        name="Operation", operation=">",
        arguments=[
            IntegerLiteralExpression("NP", 1),
            infix,
            IntegerLiteralExpression("NP", 2),
        ],
    )
    c = clone(op)
    assert c is not op
    assert c.arguments[0] is not op.arguments[0]
    assert c.arguments[1] is infix  # InfixOperator shared (treated as immutable)
    assert c.arguments[2] is not op.arguments[2]
    assert c.operation == ">"


def test_clone_boolean_binary_two_arg_shape():
    left = IntegerLiteralExpression("NP", 1)
    right = IntegerLiteralExpression("NP", 2)
    op = OperationCallExpression(name="AND_BINARY", operation="and", arguments=[left, right])
    c = clone(op)
    assert c is not op
    assert c.operation == "and"
    assert len(c.arguments) == 2
    assert c.arguments[0] is not left


def test_clone_loop_deep_copies_body_and_iterators():
    src = VariableExp("self", None)
    body = IntegerLiteralExpression("NP", 1)
    loop = LoopExp("forAll", None)
    loop.source = src
    loop.addIterator(IteratorExp("e", "NotMentioned"))
    loop.add_body(body)
    c = clone(loop)
    assert c is not loop
    assert c.source is not loop.source
    assert c.iterator[0] is not loop.iterator[0]
    assert c.iterator[0].name == "e"
    assert c.body[0] is not body


def test_clone_if_expression():
    cond = BooleanLiteralExpression("NP", True)
    if_node = IfExp("if", "IfExpression")
    if_node.ifCondition = cond
    if_node.thenExpression = IntegerLiteralExpression("NP", 1)
    if_node.elseCondition = IntegerLiteralExpression("NP", 2)
    c = clone(if_node)
    assert c is not if_node
    assert c.ifCondition is not cond
    assert c.thenExpression.value == 1
    assert c.elseCondition.value == 2


def test_clone_does_not_mutate_input():
    inner = IntegerLiteralExpression("NP", 5)
    outer = OperationCallExpression(
        name="Operation", operation=">",
        arguments=[inner, InfixOperator(">"), IntegerLiteralExpression("NP", 3)],
    )
    c = clone(outer)
    # mutate clone
    c.arguments[0].value = 999
    # original untouched
    assert inner.value == 5
