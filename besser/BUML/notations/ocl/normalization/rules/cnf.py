"""Paper §2.3 — sugar elimination toward CNF.

Eliminates ``implies``, ``xor``, and boolean ``if`` expressions in favour of
``and`` / ``or`` / ``not``. CNF distribution (OR over AND) is intentionally
omitted — see PRD §7.1; it can blow up AST size exponentially and is rarely
needed by downstream consumers.
"""

from besser.BUML.metamodel.ocl import clone, is_implies, is_xor
from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression, IfExp, LoopExp, OperationCallExpression,
)
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    make_and, make_not, make_or,
)


_BOOLEAN_OPS = {
    "and", "or", "not", "xor", "implies",
    "=", "<>", "<", ">", "<=", ">=",
    "IsEmpty", "OCLISTYPEOF", "OCLISKINDOF",
}
_BOOLEAN_LOOPS = {"forAll", "exists"}


def _is_boolean_expr(node) -> bool:
    """True when `node` is statically known to evaluate to a boolean.

    Used by :class:`IfBoolElim` to skip ``IfExp`` whose branches return
    something other than a boolean — rewriting those would produce
    ``and``/``or`` over non-boolean operands, which is not OCL-valid.
    """
    if isinstance(node, BooleanLiteralExpression):
        return True
    if isinstance(node, OperationCallExpression):
        return node.operation in _BOOLEAN_OPS
    if isinstance(node, LoopExp):
        return node.name in _BOOLEAN_LOOPS
    if isinstance(node, IfExp):
        return (_is_boolean_expr(node.thenExpression)
                and _is_boolean_expr(node.elseCondition))
    return False


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
        # Only fold IfExp whose branches are themselves boolean — rewriting
        # ``if C then 5 else 10 endif`` to ``(C and 5) or ((not C) and 10)``
        # would produce ``and``/``or`` over integers, which isn't OCL-valid.
        # Non-boolean IfExp are left in place for downstream consumers.
        if not isinstance(node, IfExp):
            return False
        return (_is_boolean_expr(node.thenExpression)
                and _is_boolean_expr(node.elseCondition))

    def rewrite(self, node, ctx):
        # if C then T else E endif  →  (C and T) or ((not C) and E)
        return make_or(
            make_and(node.ifCondition, node.thenExpression),
            make_and(make_not(node.ifCondition), node.elseCondition),
        )


CNF_RULES = [ImpliesElim(), XorElim(), IfBoolElim()]
