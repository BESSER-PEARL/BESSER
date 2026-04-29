"""Paper §3.1.3 rules 1–2 — multiplicity-aware static collapses.

When every step of a navigation chain rooted at ``self`` has a minimum
multiplicity ``≥ 1``, the chain is statically guaranteed non-empty and
``->size() > 0`` (or ``->size() = 0``) reduces to ``true`` (or ``false``).
When every step has a maximum multiplicity ``≤ 1``, the chain has at most
one element and ``->forAll(v | X)`` reduces to ``X[chain/v]``.
"""

from besser.BUML.metamodel.ocl import (
    clone, is_chain_from_self, chain_min_multiplicity,
    chain_max_multiplicity, is_size, is_comparison,
    is_loop, is_loop_with_n_iterators, substitute,
)
from besser.BUML.notations.ocl.normalization.rules._helpers import (
    is_zero_literal, make_bool,
)


def _size_of_chain(node):
    return (is_size(node) and is_chain_from_self(node.source))


class NotEmptyChainTrue:
    """``size(chain) > 0`` collapses to ``true`` when every step has min ≥ 1."""
    name = "NotEmptyChainTrue"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != ">":
            return False
        if not _size_of_chain(node.arguments[0]):
            return False
        if not is_zero_literal(node.arguments[2]):
            return False
        chain_min = chain_min_multiplicity(node.arguments[0].source)
        return chain_min is not None and chain_min >= 1

    def rewrite(self, node, ctx):
        return make_bool(True)


class IsEmptyChainFalse:
    """``size(chain) = 0`` collapses to ``false`` when every step has min ≥ 1."""
    name = "IsEmptyChainFalse"

    def applies(self, node, ctx):
        if not is_comparison(node) or node.operation != "=":
            return False
        # Match either side ordering.
        if _size_of_chain(node.arguments[0]) and is_zero_literal(node.arguments[2]):
            chain_src = node.arguments[0].source
        elif _size_of_chain(node.arguments[2]) and is_zero_literal(node.arguments[0]):
            chain_src = node.arguments[2].source
        else:
            return False
        chain_min = chain_min_multiplicity(chain_src)
        return chain_min is not None and chain_min >= 1

    def rewrite(self, node, ctx):
        return make_bool(False)


class ForAllChainSingle:
    """``chain->forAll(v | X)`` collapses to ``X[chain/v]`` when every step has max ≤ 1."""
    name = "ForAllChainSingle"

    def applies(self, node, ctx):
        if not is_loop(node, "forAll") or not is_loop_with_n_iterators(node, 1):
            return False
        if not is_chain_from_self(node.source):
            return False
        chain_max = chain_max_multiplicity(node.source)
        return chain_max is not None and chain_max <= 1

    def rewrite(self, node, ctx):
        var_name = node.iterator[0].name
        chain = clone(node.source)
        body = clone(node.body[0])
        return substitute(body, var_name, chain)


MULTIPLICITY_RULES = [
    NotEmptyChainTrue(),
    IsEmptyChainFalse(),
    ForAllChainSingle(),
]
