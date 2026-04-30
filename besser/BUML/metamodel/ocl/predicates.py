"""Predicate helpers for dispatching on OCL AST nodes.

The B-OCL metamodel encodes most operators as ``OperationCallExpression``
nodes with an ``operation`` string discriminator (``"and"``, ``"or"``,
``"forAll"``, ``"="``, etc.) rather than as dedicated subclasses. Direct
``isinstance`` checks are therefore not enough; consumers need to compose
``isinstance`` with an ``operation`` string equality.

These predicates centralize that pattern so every consumer (rewrite rules,
validators, code generators, analysis passes) tests for the same operator
shapes the same way.
"""

from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, LoopExp, BooleanLiteralExpression, VariableExp,
)


_COMP_OPS = {"<", ">", "<=", ">=", "=", "<>"}
_TYPE_TEST_OPS = {"OCLISTYPEOF", "OCLISKINDOF", "OCLASTYPE"}


def is_op(node, op):
    """True iff `node` is an ``OperationCallExpression`` with ``operation == op``."""
    return isinstance(node, OperationCallExpression) and node.operation == op


def is_and(node):
    return is_op(node, "and")


def is_or(node):
    return is_op(node, "or")


def is_xor(node):
    return is_op(node, "xor")


def is_implies(node):
    return is_op(node, "implies")


def is_not(node):
    return is_op(node, "not")


def is_size(node):
    return is_op(node, "Size")


def is_isempty(node):
    return is_op(node, "IsEmpty")


def is_allinstances(node):
    return is_op(node, "ALLInstances")


def is_comparison(node):
    """True iff `node` is a binary comparison operator (``<``, ``>``, ``<=``, ``>=``, ``=``, ``<>``).

    Comparisons use the 3-argument shape ``[L, InfixOperator(op), R]``.
    """
    return (isinstance(node, OperationCallExpression)
            and node.operation in _COMP_OPS
            and len(node.arguments) == 3)


def is_atomic_type_test(node):
    """True iff `node` is ``oclIsTypeOf`` / ``oclIsKindOf`` / ``oclAsType``.

    Rules that distribute negation must leave these alone — they are atomic
    predicates from a normalization standpoint.
    """
    return (isinstance(node, OperationCallExpression)
            and node.operation in _TYPE_TEST_OPS)


def is_loop(node, name=None):
    """True iff `node` is a ``LoopExp``. With ``name``, requires ``node.name == name``.

    Accepts ``"forAll"``, ``"exists"``, ``"select"``, ``"reject"``, ``"collect"``,
    ``"iterate"``.
    """
    if not isinstance(node, LoopExp):
        return False
    if name is None:
        return True
    return node.name == name


def is_loop_with_n_iterators(node, k):
    """True iff `node` is a loop with exactly ``k`` iterator variables.

    Most rewrite rules in the literature assume single-iterator loops; multi-
    iterator forms (``forAll(a, b | …)``) should be detected and skipped.
    """
    return isinstance(node, LoopExp) and len(node.iterator) == k


def is_self(node):
    """True iff `node` is the ``self`` variable expression."""
    return isinstance(node, VariableExp) and node.name == "self"


def is_bool_const(node, value=None):
    """True iff `node` is a ``BooleanLiteralExpression``.

    With ``value``, requires the boolean to equal ``value``. Tolerates the
    legacy lexeme-as-string representation (``"true"`` / ``"false"``) so
    callers don't need to know whether the AST has been through ``WrappingVisitor``.
    """
    if not isinstance(node, BooleanLiteralExpression):
        return False
    if value is None:
        return True
    return _coerce_bool(node.value) == value


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)
