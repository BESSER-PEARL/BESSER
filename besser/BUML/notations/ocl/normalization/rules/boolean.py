"""Paper §2.1 Table 1 — boolean simplifications.

Constant collapse, double-negation elimination, DeMorgan's laws, and
negated-comparison flattening. These rules are the workhorse of the
normalizer: every other rule produces output that these rules then clean up.
"""

from besser.BUML.metamodel.ocl import (
    clone, is_and, is_or, is_not, is_comparison, is_bool_const,
)
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    make_bool, make_or, make_and, make_comparison, make_not,
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


BOOLEAN_RULES = [
    # Constants are cheapest and unblock everything else.
    AndTrue(), AndFalse(), OrTrue(), OrFalse(), NotTrue(), NotFalse(),
    # Negation pushed inward.
    DoubleNeg(),
    NegatedComparison(),
    DeMorganAnd(), DeMorganOr(),
]
