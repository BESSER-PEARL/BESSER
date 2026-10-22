"""Small constructors and dispatch helpers reused by multiple rule modules."""

from besser.BUML.metamodel.ocl import clone, is_comparison, is_size, walk
from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression, IfExp, InfixOperator, IntegerLiteralExpression,
    LoopExp, OperationCallExpression, PropertyCallExpression, VariableExp,
)
from besser.BUML.metamodel.structural import Property


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


_BOOLEAN_OPS = {
    "and", "or", "not", "xor", "implies",
    "=", "<>", "<", ">", "<=", ">=",
    "IsEmpty", "NotEmpty", "OCLISTYPEOF", "OCLISKINDOF",
}
_BOOLEAN_LOOPS = {"forAll", "exists", "one", "isUnique"}


def is_boolean_expr(node) -> bool:
    """True when `node` is statically known to evaluate to a boolean."""
    if isinstance(node, BooleanLiteralExpression):
        return True
    if isinstance(node, OperationCallExpression):
        return node.operation in _BOOLEAN_OPS
    if isinstance(node, LoopExp):
        return node.name in _BOOLEAN_LOOPS
    if isinstance(node, IfExp):
        return (is_boolean_expr(node.thenExpression)
                and is_boolean_expr(node.elseCondition))
    return False


def expressions_equal(left, right) -> bool:
    """True when two expressions pretty-print to the same OCL text."""
    from besser.BUML.notations.ocl.pretty_printer import pretty_print
    return pretty_print(left) == pretty_print(right)


def references_self(node, context_class) -> bool:
    """True when `node` mentions the constraint's ``self``.

    A bare attribute of the context class (or of one of its parents) is an
    implicit ``self`` reference. Chapter 3 refuses the ``allInstances``
    elimination in either case.
    """
    owners = set()
    if context_class is not None:
        owners.add(context_class)
        parents = getattr(context_class, "all_parents", None)
        if parents is not None:
            owners.update(parents())
    for current in walk(node):
        if isinstance(current, VariableExp) and current.name == "self":
            return True
        if isinstance(current, Property) and getattr(current, "owner", None) in owners:
            return True
    return False


def comparison_operands(node):
    """Return ``(left, right)`` of a comparison, or ``None``."""
    if not is_comparison(node):
        return None
    return node.arguments[0], node.arguments[2]


def same_comparison_operands(left, right) -> bool:
    """True when two comparisons test the same pair of expressions."""
    pair_left = comparison_operands(left)
    pair_right = comparison_operands(right)
    if pair_left is None or pair_right is None:
        return False
    return (expressions_equal(pair_left[0], pair_right[0])
            and expressions_equal(pair_left[1], pair_right[1]))


def size_compared_with_zero(node, operators):
    """Return the ``size()`` call when `node` compares it with 0.

    ``operators`` is the set of comparison names to accept (``=``, ``<=``, …).
    The zero may sit on either side.
    """
    if not is_comparison(node) or node.operation not in operators:
        return None
    left, right = node.arguments[0], node.arguments[2]
    if is_size(left) and is_zero_literal(right):
        return left
    if is_size(right) and is_zero_literal(left):
        return right
    return None


def make_property_nav(source, prop: Property) -> PropertyCallExpression:
    """Navigate `prop` from `source`."""
    call = PropertyCallExpression(prop.name, prop)
    call.source = clone(source)
    return call


def make_forall(source, iterators, body) -> LoopExp:
    """Build ``source->forAll(iterators | body)``."""
    loop = LoopExp("forAll", None)
    loop.source = clone(source)
    for iterator in iterators:
        loop.addIterator(clone(iterator))
    loop.add_body(clone(body))
    return loop
