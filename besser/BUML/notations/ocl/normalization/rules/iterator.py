"""Paper §2.1 Table 3 — iterator simplifications.

- ``exists`` rewritten to ``not(forAll(not …))``.
- ``size(select(X, p)) = 0`` rewritten to ``X->forAll(v | not p)``.
- ``size(select(X, p)) > 0`` rewritten to ``X->exists(v | p)`` (which then
  reduces via :class:`ExistsToForAll`).
- ``select(X, p)->forAll(v | Y)`` fused to ``X->forAll(v | p implies Y)``.
"""

from besser.BUML.metamodel.ocl import (
    clone, is_loop, is_loop_with_n_iterators, is_size, is_comparison,
    substitute,
)
from besser.BUML.metamodel.ocl.ocl import LoopExp
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    is_zero_literal, make_not,
)


class ExistsToForAll:
    name = "ExistsToForAll"

    def applies(self, node, ctx):
        return is_loop(node, "exists") and is_loop_with_n_iterators(node, 1)

    def rewrite(self, node, ctx):
        # X->exists(v | Y)  →  not (X->forAll(v | not Y))
        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(node.source)
        for it in node.iterator:
            new_loop.addIterator(clone(it))
        if node.body:
            new_loop.add_body(make_not(node.body[0]))
        return make_not(new_loop)


def _is_size_of_select(node):
    return (is_size(node) and is_loop(node.source, "select")
            and is_loop_with_n_iterators(node.source, 1))


class SelectSizeZero:
    """``X->select(v | p)->size() = 0``  →  ``X->forAll(v | not p)``."""
    name = "SelectSizeZero"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != "=":
            return False
        return ((_is_size_of_select(node.arguments[0])
                 and is_zero_literal(node.arguments[2]))
                or (_is_size_of_select(node.arguments[2])
                    and is_zero_literal(node.arguments[0])))

    def rewrite(self, node, ctx):
        size = (node.arguments[0]
                if _is_size_of_select(node.arguments[0])
                else node.arguments[2])
        select_loop = size.source
        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(select_loop.source)
        for it in select_loop.iterator:
            new_loop.addIterator(clone(it))
        new_loop.add_body(make_not(select_loop.body[0]))
        return new_loop


class SelectSizeGtZero:
    """``X->select(v | p)->size() > 0``  →  ``X->exists(v | p)`` (then ExistsToForAll)."""
    name = "SelectSizeGtZero"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != ">":
            return False
        return (_is_size_of_select(node.arguments[0])
                and is_zero_literal(node.arguments[2]))

    def rewrite(self, node, ctx):
        size = node.arguments[0]
        select_loop = size.source
        new_loop = LoopExp("exists", None)
        new_loop.source = clone(select_loop.source)
        for it in select_loop.iterator:
            new_loop.addIterator(clone(it))
        new_loop.add_body(clone(select_loop.body[0]))
        return new_loop


class SelectForAllFusion:
    """``select(X, v1 | p)->forAll(v2 | Y)``  →  ``X->forAll(v2 | p[v2/v1] implies Y)``."""
    name = "SelectForAllFusion"

    def applies(self, node, ctx):
        return (is_loop(node, "forAll")
                and is_loop_with_n_iterators(node, 1)
                and is_loop(node.source, "select")
                and is_loop_with_n_iterators(node.source, 1)
                and node.body)

    def rewrite(self, node, ctx):
        from besser.BUML.metamodel.ocl.ocl import OperationCallExpression, VariableExp
        select_loop = node.source
        select_var = select_loop.iterator[0].name
        forall_var = node.iterator[0].name
        # Rename select's variable to forAll's variable in the predicate.
        renamed_pred = substitute(
            select_loop.body[0], select_var, VariableExp(forall_var, None),
        )
        forall_body = clone(node.body[0])

        # implies node — eliminated by the next pass.
        implies = OperationCallExpression(
            name="IMPLIES_BINARY", operation="implies",
            arguments=[renamed_pred, forall_body],
        )

        new_loop = LoopExp("forAll", None)
        new_loop.source = clone(select_loop.source)
        for it in node.iterator:
            new_loop.addIterator(clone(it))
        new_loop.add_body(implies)
        return new_loop


ITERATOR_RULES = [
    # The size-of-select fast paths must fire before ExistsToForAll so they
    # produce a tighter rewrite when their pattern matches the comparison.
    SelectSizeZero(), SelectSizeGtZero(),
    SelectForAllFusion(),
    ExistsToForAll(),
]
