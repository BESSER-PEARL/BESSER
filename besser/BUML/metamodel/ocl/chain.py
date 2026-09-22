"""Property-chain analysis on OCL ASTs.

Helpers for inspecting navigation chains rooted at ``self``, e.g.
``self.r1.r2.r3``. After parsing through the bug-fixing ``WrappingVisitor``,
such chains appear as nested ``PropertyCallExpression`` nodes terminating in
``VariableExp("self")``.

These helpers are useful for any pass that reasons about static reachability
through the structural model: the multiplicity-based normalization rule
(paper §3.1.3), the context-change feature, foreign-key inference in code
generators, dangling-end checks in validators.
"""

from besser.BUML.metamodel.ocl.ocl import PropertyCallExpression, VariableExp


def is_chain_from_self(node):
    """True iff `node` is a chain of ``PropertyCallExpression`` nodes terminating in ``self``.

    A bare ``Property`` (i.e. an AST node from an unwrapped visitor) returns
    ``False`` — the helper only recognises chains in their fully reconstructed
    form.
    """
    if not isinstance(node, PropertyCallExpression):
        return False
    cur = node.source
    while isinstance(cur, PropertyCallExpression):
        cur = cur.source
    return isinstance(cur, VariableExp) and cur.name == "self"


def walk_chain_from_self(node):
    """Return the chain's ``Property`` list from outermost to innermost step.

    For ``self.r1.r2.r3`` returns ``[r1, r2, r3]``. Returns ``None`` if `node`
    is not a chain rooted at ``self``.
    """
    if not is_chain_from_self(node):
        return None
    chain = []
    cur = node
    while isinstance(cur, PropertyCallExpression):
        chain.append(cur.property)
        cur = cur.source
    chain.reverse()
    return chain


def chain_min_multiplicity(node):
    """Smallest minimum multiplicity along the chain, or ``None`` if unknown.

    If every step has a defined multiplicity and the returned value is ``>= 1``,
    the chain is statically guaranteed to be non-empty: the
    ``self.r1...rn->notEmpty() == true`` simplification (paper §3.1.3 rule 1)
    is sound.
    """
    chain = walk_chain_from_self(node)
    if chain is None:
        return None
    mins = []
    for prop in chain:
        if prop.multiplicity is None:
            return None
        mins.append(prop.multiplicity.min)
    return min(mins) if mins else None


def chain_max_multiplicity(node):
    """Largest maximum multiplicity along the chain, or ``None`` if unknown.

    If every step has a defined multiplicity and the returned value is ``<= 1``,
    the chain has at most one element and the ``forAll`` collapse simplification
    (paper §3.1.3 rule 2) is sound.
    """
    chain = walk_chain_from_self(node)
    if chain is None:
        return None
    maxs = []
    for prop in chain:
        if prop.multiplicity is None:
            return None
        maxs.append(prop.multiplicity.max)
    return max(maxs) if maxs else None
