"""Post-order traversal of OCL ASTs.

Generic walker that yields every expression node in a B-OCL AST. Intended for
any pass that needs to inspect an AST: validators, code generators, the
normalization rule engine, the future context-change pass, static analysis, etc.

Children-visiting policy:
- ``OperationCallExpression``: visit ``source`` (if not None), then arguments
  left-to-right. ``InfixOperator`` instances inside ``arguments`` are skipped
  (they are markers, not sub-expressions).
- ``LoopExp`` / ``IteratorExp``: visit ``source`` (if not None), then each
  expression in ``body``. ``iterator`` (variable declarations) is not visited.
- ``IfExp``: visit ``ifCondition``, ``thenExpression``, ``elseCondition``.
- ``PropertyCallExpression``: visit ``source`` only. ``property`` is a
  metamodel ``Property`` reference and is not recursed into.
- All literal types, ``VariableExp``, ``TypeExp``, and bare ``Property``: leaf.
"""

from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, LoopExp, IfExp, PropertyCallExpression,
    InfixOperator,
)


def walk(node):
    """Yield every node in `node`'s OCL subtree in post-order.

    Children are visited before their parent. ``None`` is a no-op.
    """
    if node is None:
        return
    if isinstance(node, OperationCallExpression):
        if node.source is not None:
            yield from walk(node.source)
        for arg in node.arguments:
            if isinstance(arg, InfixOperator):
                continue
            yield from walk(arg)
        yield node
    elif isinstance(node, LoopExp):
        if node.source is not None:
            yield from walk(node.source)
        for body_expr in node.body:
            yield from walk(body_expr)
        yield node
    elif isinstance(node, IfExp):
        yield from walk(node.ifCondition)
        yield from walk(node.thenExpression)
        yield from walk(node.elseCondition)
        yield node
    elif isinstance(node, PropertyCallExpression):
        if node.source is not None:
            yield from walk(node.source)
        yield node
    else:
        yield node
