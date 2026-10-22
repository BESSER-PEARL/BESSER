"""Paper §2.1 Table 1 — boolean simplifications.

Constant collapse, double-negation elimination, DeMorgan's laws, and
negated-comparison flattening. These rules are the workhorse of the
normalizer: every other rule produces output that these rules then clean up.
"""

from besser.BUML.metamodel.ocl import (
    clone, is_and, is_or, is_not, is_comparison, is_bool_const,
)
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    is_boolean_expr, make_bool, make_int, make_or, make_and, make_comparison,
    make_not, same_comparison_operands, size_compared_with_zero,
)


_NEGATED_COMP = {
    "<": ">=",
    ">": "<=",
    "<=": ">",
    ">=": "<",
    "=": "<>",
    "<>": "=",
}


# ---------------------------------------------------------------------------
# Constant collapse
# ---------------------------------------------------------------------------
class AndTrue:
    name = "AndTrue"

    def applies(self, node, ctx):
        return (is_and(node) and len(node.arguments) == 2
                and (is_bool_const(node.arguments[0], True)
                     or is_bool_const(node.arguments[1], True)))

    def rewrite(self, node, ctx):
        if is_bool_const(node.arguments[0], True):
            return clone(node.arguments[1])
        return clone(node.arguments[0])


class AndFalse:
    name = "AndFalse"

    def applies(self, node, ctx):
        return (is_and(node) and len(node.arguments) == 2
                and (is_bool_const(node.arguments[0], False)
                     or is_bool_const(node.arguments[1], False)))

    def rewrite(self, node, ctx):
        return make_bool(False)


class OrTrue:
    name = "OrTrue"

    def applies(self, node, ctx):
        return (is_or(node) and len(node.arguments) == 2
                and (is_bool_const(node.arguments[0], True)
                     or is_bool_const(node.arguments[1], True)))

    def rewrite(self, node, ctx):
        return make_bool(True)


class OrFalse:
    name = "OrFalse"

    def applies(self, node, ctx):
        return (is_or(node) and len(node.arguments) == 2
                and (is_bool_const(node.arguments[0], False)
                     or is_bool_const(node.arguments[1], False)))

    def rewrite(self, node, ctx):
        if is_bool_const(node.arguments[0], False):
            return clone(node.arguments[1])
        return clone(node.arguments[0])


class NotTrue:
    name = "NotTrue"

    def applies(self, node, ctx):
        return is_not(node) and is_bool_const(node.arguments[0], True)

    def rewrite(self, node, ctx):
        return make_bool(False)


class NotFalse:
    name = "NotFalse"

    def applies(self, node, ctx):
        return is_not(node) and is_bool_const(node.arguments[0], False)

    def rewrite(self, node, ctx):
        return make_bool(True)


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------
class DoubleNeg:
    name = "DoubleNeg"

    def applies(self, node, ctx):
        return is_not(node) and is_not(node.arguments[0])

    def rewrite(self, node, ctx):
        return clone(node.arguments[0].arguments[0])


class NegatedComparison:
    name = "NegatedComparison"

    def applies(self, node, ctx):
        return is_not(node) and is_comparison(node.arguments[0])

    def rewrite(self, node, ctx):
        cmp = node.arguments[0]
        flipped_op = _NEGATED_COMP[cmp.operation]
        return make_comparison(cmp.arguments[0], flipped_op, cmp.arguments[2])


class DeMorganAnd:
    name = "DeMorganAnd"

    def applies(self, node, ctx):
        return is_not(node) and is_and(node.arguments[0])

    def rewrite(self, node, ctx):
        and_node = node.arguments[0]
        return make_or(make_not(and_node.arguments[0]),
                       make_not(and_node.arguments[1]))


class DeMorganOr:
    name = "DeMorganOr"

    def applies(self, node, ctx):
        return is_not(node) and is_or(node.arguments[0])

    def rewrite(self, node, ctx):
        or_node = node.arguments[0]
        return make_and(make_not(or_node.arguments[0]),
                        make_not(or_node.arguments[1]))


class EqualTrue:
    """``X = true`` → ``X`` (either side)."""
    name = "EqualTrue"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != "=":
            return False
        return (is_bool_const(node.arguments[0], True)
                or is_bool_const(node.arguments[2], True))

    def rewrite(self, node, ctx):
        if is_bool_const(node.arguments[0], True):
            return clone(node.arguments[2])
        return clone(node.arguments[0])


class EqualFalse:
    """``X = false`` → ``not X`` (either side)."""
    name = "EqualFalse"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != "=":
            return False
        return (is_bool_const(node.arguments[0], False)
                or is_bool_const(node.arguments[2], False))

    def rewrite(self, node, ctx):
        if is_bool_const(node.arguments[0], False):
            return make_not(node.arguments[2])
        return make_not(node.arguments[0])


class BooleanEquality:
    """``X = Y`` → ``(X and Y) or (not X and not Y)`` when both are boolean."""
    name = "BooleanEquality"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != "=":
            return False
        left, right = node.arguments[0], node.arguments[2]
        if is_bool_const(left) or is_bool_const(right):
            return False
        return is_boolean_expr(left) and is_boolean_expr(right)

    def rewrite(self, node, ctx):
        left, right = node.arguments[0], node.arguments[2]
        return make_or(
            make_and(left, right),
            make_and(make_not(left), make_not(right)),
        )


class _BoundPair:
    """Shared matcher for ``X op Y`` combined with ``X op2 Y``.

    ``connective`` is the string ``"and"`` or ``"or"``. It is not a function:
    a function stored on the class would be bound as a method.
    """

    connective = ""
    pairs = frozenset()

    def applies(self, node, ctx):
        if self.connective == "and" and not is_and(node):
            return False
        if self.connective == "or" and not is_or(node):
            return False
        if self.connective not in ("and", "or"):
            return False
        left, right = node.arguments
        if not same_comparison_operands(left, right):
            return False
        return frozenset((left.operation, right.operation)) in self.pairs


class ContradictoryBounds(_BoundPair):
    """``X > Y and X <= Y`` → ``false`` (and the ``<`` / ``>=`` mirror)."""
    name = "ContradictoryBounds"
    connective = "and"
    pairs = frozenset({frozenset((">", "<=")), frozenset(("<", ">="))})

    def rewrite(self, node, ctx):
        return make_bool(False)


class ExhaustiveBounds(_BoundPair):
    """``X > Y or X <= Y`` → ``true`` (and the ``<`` / ``>=`` mirror)."""
    name = "ExhaustiveBounds"
    connective = "or"
    pairs = frozenset({frozenset((">", "<=")), frozenset(("<", ">="))})

    def rewrite(self, node, ctx):
        return make_bool(True)


class StrictDisjunction(_BoundPair):
    """``X > Y or X < Y`` → ``X <> Y``."""
    name = "StrictDisjunction"
    connective = "or"
    pairs = frozenset({frozenset((">", "<"))})

    def rewrite(self, node, ctx):
        left = node.arguments[0]
        return make_comparison(left.arguments[0], "<>", left.arguments[2])


class NotNaturalZero:
    """``not (X->size() = 0)`` → ``X->size() > 0``.

    ``size`` is a natural number, which is the case the thesis states for
    ``not X = 0``.
    """
    name = "NotNaturalZero"

    def applies(self, node, ctx):
        return is_not(node) and size_compared_with_zero(node.arguments[0], {"="}) is not None

    def rewrite(self, node, ctx):
        size = size_compared_with_zero(node.arguments[0], {"="})
        return make_comparison(size, ">", make_int(0))


BOOLEAN_RULES = [
    # Constants are cheapest and unblock everything else.
    AndTrue(), AndFalse(), OrTrue(), OrFalse(), NotTrue(), NotFalse(),
    EqualTrue(), EqualFalse(),
    # Bounds on the same pair collapse before negation is pushed inward.
    ContradictoryBounds(), ExhaustiveBounds(), StrictDisjunction(),
    # Negation pushed inward. Natural zero is more specific than flipping '='.
    DoubleNeg(),
    NotNaturalZero(),
    NegatedComparison(),
    DeMorganAnd(), DeMorganOr(),
    # Boolean '=' expands into and/or, which distribution then normalizes.
    BooleanEquality(),
]
