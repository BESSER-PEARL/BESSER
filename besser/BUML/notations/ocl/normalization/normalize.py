"""Bottom-up fixed-point rewrite driver.

:func:`normalize` clones the input AST, then iterates: each pass walks the AST
in post-order, asks every rule (in fixed order) whether it applies at each
node, and rewrites at the first matching rule. The pass terminates when a
full traversal produces no rewrite. Termination is guaranteed by the
lexicographic measure ``⟨#sugar-operators, AST size, #negations⟩`` — every
rule strictly decreases that measure.

The driver returns a fresh :class:`OCLConstraint`. Inputs are not mutated.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from besser.BUML.metamodel.ocl import clone
from besser.BUML.metamodel.ocl.ocl import (
    IfExp, InfixOperator, LoopExp, OCLConstraint,
    OperationCallExpression, PropertyCallExpression,
)
from besser.BUML.metamodel.structural import DomainModel


@dataclass
class Context:
    """Shared state passed to every rule.

    Attributes:
        model: The :class:`DomainModel` providing class and property bindings.
        constraint: The original :class:`OCLConstraint` (read-only) — gives
            rules access to the context class via ``constraint.context``.
        log: Trace of rule applications as ``(rule_name, before, after)``
            triples. ``before`` and ``after`` are short string repr's, suitable
            for debugging or assertion in tests.
    """
    model: DomainModel
    constraint: OCLConstraint
    log: List[Tuple[str, str, str]] = field(default_factory=list)


def normalize(constraint: OCLConstraint, model: DomainModel,
              max_iterations: int = 1000) -> OCLConstraint:
    """Return a fresh :class:`OCLConstraint` in B-OCL normal form.

    Args:
        constraint: The constraint to normalize. Not mutated.
        model: The owning :class:`DomainModel` (used for class/property
            lookups in some rules).
        max_iterations: Hard cap on outer fixed-point iterations. Default is
            1000; reaching this implies a rule is non-terminating, which is a
            bug.

    Returns:
        A new :class:`OCLConstraint` with the same name, context, and language
        as the input, and a normalized expression.
    """
    # Defer rule import to avoid a circular import on package load.
    from besser.BUML.notations.ocl.normalization.rules import build_default_rules

    rules = build_default_rules()
    expr = clone(constraint.ast)
    ctx = Context(model=model, constraint=constraint)

    for _ in range(max_iterations):
        new_expr, changed = _rewrite_pass(expr, rules, ctx)
        if not changed:
            return OCLConstraint(
                name=constraint.name,
                context=constraint.context,
                expression=new_expr,
                language=constraint.language or "OCL",
            )
        expr = new_expr

    raise RuntimeError(
        f"normalize: rewrite did not reach a fixed point in "
        f"{max_iterations} iterations — possible non-terminating rule. "
        f"Trace: {ctx.log[-10:]}"
    )


def _rewrite_pass(node, rules, ctx):
    """One bottom-up traversal. Returns ``(node, changed_anywhere)``."""
    if node is None:
        return None, False

    changed = False

    # Recurse into children first (post-order). We mutate `node` in place
    # because the top-level driver cloned the input — every node we touch
    # here is freshly owned by this normalization run.
    if isinstance(node, OperationCallExpression):
        if node.source is not None:
            new_src, ch = _rewrite_pass(node.source, rules, ctx)
            if ch:
                node.source = new_src
                changed = True
        for i, arg in enumerate(node.arguments):
            if isinstance(arg, InfixOperator):
                continue
            new_arg, ch = _rewrite_pass(arg, rules, ctx)
            if ch:
                node.arguments[i] = new_arg
                changed = True
    elif isinstance(node, LoopExp):
        if node.source is not None:
            new_src, ch = _rewrite_pass(node.source, rules, ctx)
            if ch:
                node.source = new_src
                changed = True
        for i, body_expr in enumerate(node.body):
            new_body, ch = _rewrite_pass(body_expr, rules, ctx)
            if ch:
                node.body[i] = new_body
                changed = True
    elif isinstance(node, IfExp):
        new_c, ch = _rewrite_pass(node.ifCondition, rules, ctx)
        if ch:
            node.ifCondition = new_c
            changed = True
        new_t, ch = _rewrite_pass(node.thenExpression, rules, ctx)
        if ch:
            node.thenExpression = new_t
            changed = True
        new_e, ch = _rewrite_pass(node.elseCondition, rules, ctx)
        if ch:
            node.elseCondition = new_e
            changed = True
    elif isinstance(node, PropertyCallExpression):
        if node.source is not None:
            new_src, ch = _rewrite_pass(node.source, rules, ctx)
            if ch:
                node.source = new_src
                changed = True

    # Try each rule on this (now-children-normalized) node.
    for rule in rules:
        if rule.applies(node, ctx):
            new_node = rule.rewrite(node, ctx)
            # Preserve source location from the original node for diagnostics.
            # copy_location_from only fills fields that are None on the new node,
            # so synthesized children that already carry their own positions are
            # left intact.
            if hasattr(new_node, "copy_location_from"):
                new_node.copy_location_from(node)
            ctx.log.append((rule.name, _short(node), _short(new_node)))
            return new_node, True

    return node, changed


def _short(node) -> str:
    """Compact repr for the trace log."""
    try:
        from besser.BUML.notations.ocl.pretty_printer import pretty_print
        return pretty_print(node)
    except Exception:
        return repr(node)
