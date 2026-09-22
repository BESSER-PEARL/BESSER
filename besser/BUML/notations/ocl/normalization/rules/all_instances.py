"""Paper §2.2 — eliminate ``allInstances`` over the constraint context type.

When the receiver of ``allInstances()`` is the same class as the constraint's
context, ``ct.allInstances()->forAll(v | Y)`` is equivalent to ``Y[self/v]``.
"""

from besser.BUML.metamodel.ocl import (
    clone, is_allinstances, is_loop, is_loop_with_n_iterators, substitute,
)
from besser.BUML.metamodel.ocl.ocl import TypeExp, VariableExp


class AllInstancesContextElim:
    name = "AllInstancesContextElim"

    def applies(self, node, ctx):
        if not is_loop(node, "forAll") or not is_loop_with_n_iterators(node, 1):
            return False
        src = node.source
        if not is_allinstances(src):
            return False
        type_exp = src.source
        if not isinstance(type_exp, TypeExp):
            return False
        return type_exp.name == ctx.constraint.context.name

    def rewrite(self, node, ctx):
        var_name = node.iterator[0].name
        body = clone(node.body[0])
        # Replace every free occurrence of `var_name` with `self`.
        return substitute(body, var_name, VariableExp("self", None))


ALL_INSTANCES_RULES = [AllInstancesContextElim()]
