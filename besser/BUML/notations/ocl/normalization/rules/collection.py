"""Paper §2.1 Table 2 — trivial collection sugar.

``X->isEmpty()`` becomes ``X->size() = 0`` and ``X->notEmpty()`` becomes
``X->size() > 0``. ``X->reject(p)`` becomes ``X->select(not p)``.
"""

from besser.BUML.metamodel.ocl import clone, is_isempty, is_loop, is_op
from besser.BUML.metamodel.ocl.ocl import LoopExp, OperationCallExpression
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    make_comparison, make_int, make_not, size_compared_with_zero,
)


class IsEmptyToSize:
    name = "IsEmptyToSize"

    def applies(self, node, ctx):
        return is_isempty(node)

    def rewrite(self, node, ctx):
        # X->isEmpty()  →  X->size() = 0
        size = OperationCallExpression(
            name="Size", operation="Size", arguments=[],
        )
        size.source = clone(node.source)
        return make_comparison(size, "=", make_int(0))


class NotEmptyToSize:
    name = "NotEmptyToSize"

    def applies(self, node, ctx):
        return is_op(node, "NotEmpty")

    def rewrite(self, node, ctx):
        # X->notEmpty()  →  X->size() > 0
        size = OperationCallExpression(
            name="Size", operation="Size", arguments=[],
        )
        size.source = clone(node.source)
        return make_comparison(size, ">", make_int(0))


class RejectToSelect:
    name = "RejectToSelect"

    def applies(self, node, ctx):
        return is_loop(node, "reject")

    def rewrite(self, node, ctx):
        # X->reject(v | p)  →  X->select(v | not p)
        new_loop = LoopExp("select", None)
        new_loop.source = clone(node.source)
        for it in node.iterator:
            new_loop.addIterator(clone(it))
        if node.body:
            new_loop.add_body(make_not(node.body[0]))
        return new_loop


class SizeAtMostZero:
    """``X->size() <= 0`` → ``X->size() = 0``.

    ``size`` is a natural, so this is the same test as ``isEmpty``. It lets
    ``not X->notEmpty()`` reach ``X->size() = 0``, the form ``isEmpty`` uses.
    """
    name = "SizeAtMostZero"

    def applies(self, node, ctx):
        return size_compared_with_zero(node, {"<="}) is not None

    def rewrite(self, node, ctx):
        size = size_compared_with_zero(node, {"<="})
        return make_comparison(size, "=", make_int(0))


COLLECTION_RULES = [IsEmptyToSize(), NotEmptyToSize(), RejectToSelect(), SizeAtMostZero()]
