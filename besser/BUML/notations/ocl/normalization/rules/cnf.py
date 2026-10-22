"""Chapter 3.3 — sugar elimination and conjunctive normal form.

Eliminates ``implies``, ``xor``, and boolean ``if`` expressions, then
distributes ``or`` over ``and``. The shapes match the thesis:

- ``X implies Y`` → ``not X or Y``
- ``X xor Y`` → ``(X or Y) and (not X or not Y)``
- ``if X then Y else Z`` → ``(not X or Y) and (X or Z)`` when both branches
  are boolean
- ``X or (Y and Z)`` → ``(X or Y) and (X or Z)``

Distribution grows the tree. The driver caps the number of passes.
"""

from besser.BUML.metamodel.ocl import is_and, is_implies, is_or, is_xor
from besser.BUML.metamodel.ocl.ocl import IfExp
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    is_boolean_expr, make_and, make_not, make_or,
)


class ImpliesElim:
    name = "ImpliesElim"

    def applies(self, node, ctx):
        return is_implies(node) and len(node.arguments) == 2

    def rewrite(self, node, ctx):
        # A implies B  →  (not A) or B
        left, right = node.arguments
        return make_or(make_not(left), right)


class XorElim:
    name = "XorElim"

    def applies(self, node, ctx):
        return is_xor(node) and len(node.arguments) == 2

    def rewrite(self, node, ctx):
        # A xor B  →  (A or B) and (not A or not B)
        left, right = node.arguments
        return make_and(
            make_or(left, right),
            make_or(make_not(left), make_not(right)),
        )


class IfBoolElim:
    name = "IfBoolElim"

    def applies(self, node, ctx):
        # Only fold IfExp whose branches are themselves boolean. Folding
        # ``if C then 5 else 10 endif`` would put integers under ``or``.
        if not isinstance(node, IfExp):
            return False
        return (is_boolean_expr(node.thenExpression)
                and is_boolean_expr(node.elseCondition))

    def rewrite(self, node, ctx):
        # if C then T else E  →  (not C or T) and (C or E)
        return make_and(
            make_or(make_not(node.ifCondition), node.thenExpression),
            make_or(node.ifCondition, node.elseCondition),
        )


class OrOverAnd:
    """``X or (Y and Z)`` → ``(X or Y) and (X or Z)`` (either side)."""
    name = "OrOverAnd"

    def applies(self, node, ctx):
        if not is_or(node) or len(node.arguments) != 2:
            return False
        return is_and(node.arguments[0]) or is_and(node.arguments[1])

    def rewrite(self, node, ctx):
        left, right = node.arguments
        if is_and(right):
            disjunct, conjunction = left, right
        else:
            conjunction, disjunct = left, right
        first, second = conjunction.arguments
        return make_and(make_or(disjunct, first), make_or(disjunct, second))


CNF_RULES = [ImpliesElim(), XorElim(), IfBoolElim(), OrOverAnd()]
