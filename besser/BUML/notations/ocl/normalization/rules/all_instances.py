"""Chapter 3.2 — eliminate ``allInstances`` over the constraint context type.

``ct.allInstances()->forAll(v | Y)`` becomes ``Y[self/v]`` when ``ct`` is the
constraint context and ``Y`` does not already mention ``self``. With several
iterators only the first is replaced, and the ``allInstances`` call stays:

``ct.allInstances()->forAll(v1, v2 | Y)`` →
``ct.allInstances()->forAll(v2 | Y[self/v1])``.
"""

from besser.BUML.metamodel.ocl import (
    clone, is_allinstances, is_loop, substitute,
)
from besser.BUML.metamodel.ocl.ocl import LoopExp, TypeExp, VariableExp
from besser.BUML.notations.ocl.normalization.rules._helpers import references_self


class AllInstancesContextElim:
    name = "AllInstancesContextElim"

    def applies(self, node, ctx):
        if not is_loop(node, "forAll") or not node.iterator or not node.body:
            return False
        if not _is_context_all_instances(node, ctx):
            return False
        # Chapter 3: the rule does not apply once ``self`` is already in use.
        return not references_self(node.body[0], ctx.constraint.context)

    def rewrite(self, node, ctx):
        first, *rest = node.iterator
        body = substitute(clone(node.body[0]), first.name, VariableExp("self", None))
        if not rest:
            return body
        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(node.source)
        for iterator in rest:
            new_loop.addIterator(clone(iterator))
        new_loop.add_body(body)
        return new_loop


def _is_context_all_instances(node, ctx):
    source = node.source
    if not is_allinstances(source):
        return False
    type_exp = source.source
    if not isinstance(type_exp, TypeExp) or ctx.constraint.context is None:
        return False
    return type_exp.name == ctx.constraint.context.name


ALL_INSTANCES_RULES = [AllInstancesContextElim()]
