"""Paper §2.1 Table 2 — trivial collection sugar.

``X->isEmpty()`` becomes ``X->size() = 0``. ``X->reject(p)`` becomes
``X->select(not p)``.
"""

from besser.BUML.metamodel.ocl import clone, is_isempty, is_loop
from besser.BUML.metamodel.ocl.ocl import LoopExp, OperationCallExpression
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    make_comparison, make_int, make_not,
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


COLLECTION_RULES = [IsEmptyToSize(), RejectToSelect()]
