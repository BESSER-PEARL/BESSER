"""Tests for besser.BUML.metamodel.ocl.predicates."""

from besser.BUML.metamodel.ocl import (
    is_op, is_and, is_or, is_xor, is_implies, is_not,
    is_size, is_isempty, is_allinstances,
    is_comparison, is_atomic_type_test,
    is_loop, is_loop_with_n_iterators,
    is_self, is_bool_const,
)
from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, LoopExp, IteratorExp,
    BooleanLiteralExpression, VariableExp, TypeExp,
    IntegerLiteralExpression, InfixOperator,
)


def _bin(op_name, op, args):
    return OperationCallExpression(name=op_name, operation=op, arguments=args)


def test_is_op_basic():
    assert is_op(_bin("AND_BINARY", "and", []), "and")
    assert not is_op(_bin("AND_BINARY", "and", []), "or")
    assert not is_op(IntegerLiteralExpression("NP", 1), "and")


def test_boolean_op_predicates():
    assert is_and(_bin("AND_BINARY", "and", []))
    assert is_or(_bin("OR_BINARY", "or", []))
    assert is_xor(_bin("XOR_BINARY", "xor", []))
    assert is_implies(_bin("IMPLIES_BINARY", "implies", []))
    assert is_not(_bin("unary_not", "not", []))
    assert not is_and(_bin("OR_BINARY", "or", []))


def test_collection_op_predicates():
    assert is_size(OperationCallExpression(name="Size", operation="Size", arguments=[]))
    assert is_isempty(OperationCallExpression(name="IsEmpty", operation="IsEmpty", arguments=[]))
    assert is_allinstances(
        OperationCallExpression(name="ALLInstances", operation="ALLInstances", arguments=[])
    )


def test_is_comparison_requires_three_arg_shape():
    three = OperationCallExpression(
        name="Operation", operation="<",
        arguments=[
            IntegerLiteralExpression("NP", 1), InfixOperator("<"),
            IntegerLiteralExpression("NP", 2),
        ],
    )
    assert is_comparison(three)
    # Two-arg shape with same operation string is not a comparison.
    two = OperationCallExpression(name="Op", operation="<",
                                  arguments=[IntegerLiteralExpression("NP", 1),
                                             IntegerLiteralExpression("NP", 2)])
    assert not is_comparison(two)


def test_is_atomic_type_test():
    for op in ("OCLISTYPEOF", "OCLISKINDOF", "OCLASTYPE"):
        node = OperationCallExpression(name=op, operation=op,
                                       arguments=[TypeExp("T", "T")])
        assert is_atomic_type_test(node)
    assert not is_atomic_type_test(_bin("AND_BINARY", "and", []))


def test_is_loop():
    loop = LoopExp("forAll", None)
    assert is_loop(loop)
    assert is_loop(loop, "forAll")
    assert not is_loop(loop, "exists")
    assert not is_loop(IntegerLiteralExpression("NP", 1))


def test_is_loop_with_n_iterators():
    loop = LoopExp("forAll", None)
    assert is_loop_with_n_iterators(loop, 0)
    loop.addIterator(IteratorExp("a", "NotMentioned"))
    assert is_loop_with_n_iterators(loop, 1)
    loop.addIterator(IteratorExp("b", "NotMentioned"))
    assert is_loop_with_n_iterators(loop, 2)
    assert not is_loop_with_n_iterators(loop, 1)


def test_is_self():
    assert is_self(VariableExp("self", None))
    assert not is_self(VariableExp("e", None))
    assert not is_self(IntegerLiteralExpression("NP", 1))


def test_is_bool_const_with_python_bool():
    n_true = BooleanLiteralExpression("NP", True)
    n_false = BooleanLiteralExpression("NP", False)
    assert is_bool_const(n_true)
    assert is_bool_const(n_true, True)
    assert not is_bool_const(n_true, False)
    assert is_bool_const(n_false, False)


def test_is_bool_const_tolerates_lexeme_string():
    n_true = BooleanLiteralExpression("NP", "true")
    n_false = BooleanLiteralExpression("NP", "false")
    assert is_bool_const(n_true, True)
    assert is_bool_const(n_false, False)
    assert not is_bool_const(n_true, False)
