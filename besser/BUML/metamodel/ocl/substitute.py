"""Capture-avoiding variable substitution on OCL ASTs.

:func:`substitute` replaces every *free* occurrence of a variable in an OCL
expression with a clone of a replacement expression. A variable is *bound*
inside the body of a ``LoopExp`` whose iterator declarations include a
matching name; bound occurrences are not substituted.

Used by:

- the ``allInstances`` normalization rule (substitute the iterator variable
  with ``self``);
- the multiplicity-1 ``forAll`` collapse (substitute the iterator with the
  source navigation chain);
- any future pass that performs let-inlining, alpha-renaming, or partial
  evaluation.

The base visitor's flat-dict iterator scope (in ``BOCLVisitorImpl``) has a
known shadowing bug for nested loops with same-named iterators. This module
uses an explicit stack so substitution is correct under any nesting.
"""

from besser.BUML.metamodel.ocl.clone import clone
from besser.BUML.metamodel.ocl.ocl import (
    IfExp, InfixOperator, LoopExp, OperationCallExpression,
    PropertyCallExpression, VariableExp,
)


class ScopeStack:
    """Stack of variable-binding frames for capture-avoiding substitution.

    Each frame is a set of bound names. Pushing a frame on entry to a loop
    body and popping on exit ensures inner shadowing does not leak.
    """

    def __init__(self):
        self._frames = []

    def push(self, names):
        self._frames.append(set(names))

    def pop(self):
        return self._frames.pop()

    def is_bound(self, name):
        return any(name in frame for frame in self._frames)


def substitute(expr, var_name, replacement, scope=None):
    """Return a fresh AST with every free ``VariableExp(var_name)`` replaced by ``clone(replacement)``.

    ``expr`` is not mutated. If a nested ``LoopExp`` declares an iterator with
    the same name as ``var_name``, occurrences inside that loop's body are
    treated as bound and left untouched.

    Args:
        expr: The expression to substitute into. May be any OCL AST node.
        var_name: The variable name to replace.
        replacement: The expression to substitute in place of each free
            occurrence. Cloned at each substitution site so the result is a
            tree (no aliasing).
        scope: Optional initial scope stack. Defaults to a fresh empty stack.

    Returns:
        A fresh AST tree.
    """
    if scope is None:
        scope = ScopeStack()
    return _sub(expr, var_name, replacement, scope)


def _sub(node, var, repl, scope):
    if node is None:
        return None

    if isinstance(node, VariableExp):
        if node.name == var and not scope.is_bound(var):
            return clone(repl)
        return clone(node)

    if isinstance(node, LoopExp):
        new = LoopExp(node.name, node.type)
        if node.source is not None:
            # Source is in the enclosing scope; iterator shadowing applies
            # only to the body.
            new.source = _sub(node.source, var, repl, scope)
        for it in node.iterator:
            new.addIterator(clone(it))

        bound_names = [it.name for it in node.iterator]
        scope.push(bound_names)
        try:
            for body_expr in node.body:
                new.add_body(_sub(body_expr, var, repl, scope))
        finally:
            scope.pop()
        return new

    if isinstance(node, IfExp):
        new = IfExp(node.name, node.type)
        new.ifCondition = _sub(node.ifCondition, var, repl, scope)
        new.thenExpression = _sub(node.thenExpression, var, repl, scope)
        new.elseCondition = _sub(node.elseCondition, var, repl, scope)
        return new

    if isinstance(node, OperationCallExpression):
        new_args = []
        for arg in node.arguments:
            if isinstance(arg, InfixOperator):
                new_args.append(arg)
            else:
                new_args.append(_sub(arg, var, repl, scope))
        new = OperationCallExpression(node.name, node.operation, new_args)
        if node.source is not None:
            new.source = _sub(node.source, var, repl, scope)
        if node.referredOperation is not None:
            new.referredOperation = clone(node.referredOperation)
        return new

    if isinstance(node, PropertyCallExpression):
        new = PropertyCallExpression(node.name, node.property)
        if node.source is not None:
            new.source = _sub(node.source, var, repl, scope)
        return new

    # Leaf — clone (which is identity for shared metamodel objects).
    return clone(node)
