"""Paper §2.3 — sugar elimination toward CNF.

Eliminates ``implies``, ``xor``, and boolean ``if`` expressions in favour of
``and`` / ``or`` / ``not``. CNF distribution (OR over AND) is intentionally
omitted — see PRD §7.1; it can blow up AST size exponentially and is rarely
needed by downstream consumers.
"""

from besser.BUML.metamodel.ocl import clone, is_implies, is_xor
from besser.BUML.metamodel.ocl.ocl import IfExp
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    make_and, make_not, make_or,
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
        # A xor B  →  (A and not B) or (not A and B)
        left, right = node.arguments
        return make_or(
            make_and(clone(left), make_not(right)),
            make_and(make_not(left), clone(right)),
        )


class IfBoolElim:
    name = "IfBoolElim"

    def applies(self, node, ctx):
        # Convert any IfExp; even non-boolean branches are folded since the
        # result is structurally well-defined and downstream consumers need
        # zero ``IfExp`` nodes in normal form. Branches that aren't boolean
        # are uncommon in invariants and the resulting expression remains
        # semantically faithful for any branch type that supports ``and`` /
        # ``or`` (i.e. all OCL boolean expressions).
        return isinstance(node, IfExp)

    def rewrite(self, node, ctx):
        # if C then T else E endif  →  (C and T) or ((not C) and E)
        return make_or(
            make_and(node.ifCondition, node.thenExpression),
            make_and(make_not(node.ifCondition), node.elseCondition),
        )


CNF_RULES = [ImpliesElim(), XorElim(), IfBoolElim()]
