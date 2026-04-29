"""Structural cloning of OCL ASTs.

``clone(node)`` returns a fresh AST tree that mirrors `node`. References to
domain-model objects (``Property``, ``Class``, primitive types) are shared
with the original, not duplicated — these belong to the structural model and
must remain identity-equal across clones.

Use this whenever a transformation must not mutate its input. ``copy.deepcopy``
is unsuitable because it would clone the entire ``DomainModel`` graph reachable
through ``Property.owner``.
"""

from besser.BUML.metamodel.ocl.ocl import (
    OCLExpression, OperationCallExpression, LoopExp, IteratorExp, IfExp,
    PropertyCallExpression, VariableExp, TypeExp, InfixOperator,
    IntegerLiteralExpression, RealLiteralExpression,
    BooleanLiteralExpression, StringLiteralExpression, DateLiteralExpression,
)


def clone(node):
    """Return a fresh copy of `node` with shared structural-model references.

    Source-location fields (``line``, ``col``, ``source_text``) on
    :class:`OCLExpression` nodes are preserved on the clone so diagnostics
    issued downstream still point at the original OCL source.
    """
    new = _clone_inner(node)
    if isinstance(node, OCLExpression) and isinstance(new, OCLExpression):
        new.copy_location_from(node)
    return new


def _clone_inner(node):
    if node is None:
        return None

    # IteratorExp checked before LoopExp because IteratorExp <: LoopExp.
    if isinstance(node, IteratorExp):
        return IteratorExp(node.name, node.type)

    if isinstance(node, LoopExp):
        new = LoopExp(node.name, node.type)
        if node.source is not None:
            new.source = clone(node.source)
        for it in node.iterator:
            new.addIterator(clone(it))
        for body_expr in node.body:
            new.add_body(clone(body_expr))
        return new

    if isinstance(node, IfExp):
        new = IfExp(node.name, node.type)
        new.ifCondition = clone(node.ifCondition)
        new.thenExpression = clone(node.thenExpression)
        new.elseCondition = clone(node.elseCondition)
        return new

    if isinstance(node, OperationCallExpression):
        new_args = []
        for arg in node.arguments:
            if isinstance(arg, InfixOperator):
                # InfixOperator carries a single string operator; treat as immutable.
                new_args.append(arg)
            else:
                new_args.append(clone(arg))
        new = OperationCallExpression(node.name, node.operation, new_args)
        if node.source is not None:
            new.source = clone(node.source)
        if node.referredOperation is not None:
            new.referredOperation = clone(node.referredOperation)
        return new

    if isinstance(node, PropertyCallExpression):
        # Share the underlying Property reference.
        new = PropertyCallExpression(node.name, node.property)
        if node.source is not None:
            new.source = clone(node.source)
        return new

    if isinstance(node, VariableExp):
        return VariableExp(node.name, None)

    if isinstance(node, TypeExp):
        return TypeExp(node.name, node.name)

    if isinstance(node, IntegerLiteralExpression):
        return IntegerLiteralExpression(node.name, node.value)
    if isinstance(node, RealLiteralExpression):
        return RealLiteralExpression(node.name, node.value)
    if isinstance(node, BooleanLiteralExpression):
        return BooleanLiteralExpression(node.name, node.value)
    if isinstance(node, StringLiteralExpression):
        return StringLiteralExpression(node.name, node.value)
    if isinstance(node, DateLiteralExpression):
        return DateLiteralExpression(node.name, node.value)

    # Bare structural-model objects (e.g. Property leaked through unwrapped
    # visitor) are shared, not cloned.
    return node
