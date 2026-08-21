"""Small constructors and dispatch helpers reused by multiple rule modules."""

from besser.BUML.metamodel.ocl import clone
from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression, InfixOperator, IntegerLiteralExpression,
    OperationCallExpression,
)


def make_bool(value: bool) -> BooleanLiteralExpression:
    return BooleanLiteralExpression("NP", value)


def make_int(value: int) -> IntegerLiteralExpression:
    return IntegerLiteralExpression("NP", value)


def make_not(operand) -> OperationCallExpression:
    return OperationCallExpression(
        name="unary_not", operation="not", arguments=[clone(operand)],
    )


def make_and(left, right) -> OperationCallExpression:
    return OperationCallExpression(
        name="AND_BINARY", operation="and",
        arguments=[clone(left), clone(right)],
    )


def make_or(left, right) -> OperationCallExpression:
    return OperationCallExpression(
        name="OR_BINARY", operation="or",
        arguments=[clone(left), clone(right)],
    )


def make_comparison(left, op: str, right) -> OperationCallExpression:
    return OperationCallExpression(
        name="Operation", operation=op,
        arguments=[clone(left), InfixOperator(op), clone(right)],
    )


def is_zero_literal(node) -> bool:
    return isinstance(node, IntegerLiteralExpression) and node.value == 0
