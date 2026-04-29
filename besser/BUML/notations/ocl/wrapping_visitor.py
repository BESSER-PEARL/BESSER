"""Bug-fixing subclass of the generated B-OCL visitor.

The default ``BOCLVisitorImpl`` (in ``visitor.py``) has several issues that
break any consumer that needs the AST in its proper shape:

1. **Chain flattening in ``visitDotNavigation``**: the receiver expression
   visit result is discarded, so ``self.r1.r2`` parses to a bare leaf
   ``Property`` with no chain in the AST.
2. **Same flattening in ``visitDotSizeNavigation``**: the special-case
   ``size`` attribute access has the same bug.
3. **Wrong lookup context for chained navigation**: every step of a property
   chain is resolved against ``self.context_class`` rather than against the
   receiver's actual type, so ``self.employer.minSalary`` would look up
   ``minSalary`` on ``Employee`` (the constraint context) instead of on
   ``Department`` (the type of ``self.employer``).
4. **Wrong iterator-variable typing in ``visitArrowIterator``**: when no
   explicit type annotation is supplied (``forAll(e | …)``), the base visitor
   sets ``iterators_context[e]`` to the constraint's context class, regardless
   of what collection is being iterated. The element type of the source
   collection is what's needed.
5. **Boolean-lexeme-as-string in ``visitBooleanLiteral``**: the literal value
   is stored as the lexeme ``"true"`` / ``"false"`` (a Python ``str``)
   instead of a Python ``bool``.

This subclass overrides the affected methods to fix all of the above. Until
the fixes are upstreamed into ``BOCLVisitorImpl`` itself, all consumers
should parse via ``WrappingVisitor`` (or via :func:`parse_ocl`, which uses it).
"""

from besser.BUML.metamodel.ocl.ocl import (
    BooleanLiteralExpression, IteratorExp, LoopExp, OperationCallExpression,
    PropertyCallExpression, TypeExp, VariableExp,
)
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError
from besser.BUML.notations.ocl.visitor import BOCLVisitorImpl


class WrappingVisitor(BOCLVisitorImpl):
    """``BOCLVisitorImpl`` subclass that produces a chain-preserving AST."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _receiver_type(self, receiver):
        """Return the static type of `receiver`, or None if it cannot be inferred."""
        if isinstance(receiver, VariableExp):
            if receiver.name == "self":
                return self.context_class
            return self.iterators_context.get(receiver.name)
        if isinstance(receiver, PropertyCallExpression):
            return receiver.property.type
        return None

    def _infer_collection_element_type(self, source):
        """Best-effort element type for `source` when treated as a collection.

        ``self.employee`` (an association end) → ``Employee``.
        ``Employee.allInstances()`` → ``Employee``.
        ``X->select(...)`` / ``X->reject(...)`` / ``X->collect(...)`` →
        same as the select/reject/collect's source.
        """
        if isinstance(source, PropertyCallExpression):
            return source.property.type
        if isinstance(source, OperationCallExpression):
            if source.operation == "ALLInstances":
                type_exp = source.source
                if isinstance(type_exp, TypeExp):
                    return self._get_class_by_name(type_exp.name)
            return None
        if isinstance(source, LoopExp):
            # select / reject / collect preserve element type.
            return self._infer_collection_element_type(source.source)
        return None

    def _wrap_navigation(self, receiver, name, ctx):
        """Resolve `name` against the receiver's type, falling back to the
        constraint's scopes, then return a ``PropertyCallExpression``."""
        prop = None
        target_type = self._receiver_type(receiver)
        if target_type is not None:
            prop = self._resolve_property(name, target_type)
        if prop is None:
            prop = self._resolve_property(name)
        if prop is None:
            prop = self._resolve_property_in_iterators(name)
        if prop is None:
            ctx_name = self.context_class.name if self.context_class else "?"
            raise BOCLSyntaxError([
                f"Property '{name}' not found in context '{ctx_name}'"
            ])
        pce = PropertyCallExpression(prop.name, prop)
        pce.source = receiver
        return pce

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def visitDotNavigation(self, ctx):
        receiver = self.visit(ctx.expression())
        return self._wrap_navigation(receiver, ctx.ID().getText(), ctx)

    def visitDotSizeNavigation(self, ctx):
        receiver = self.visit(ctx.expression())
        return self._wrap_navigation(receiver, "size", ctx)

    def visitBooleanLiteral(self, ctx):
        text = ctx.BOOLEAN_LITERAL().getText()
        return BooleanLiteralExpression("NP", text == "true")

    def visitArrowIterator(self, ctx: BOCLParser.ArrowIteratorContext):
        source = self.visit(ctx.expression(0))
        op_name = ctx.iteratorOp().getText()
        loop = LoopExp(op_name, None)
        loop.source = source

        var_decl = ctx.iteratorVarDecl()
        ids = var_decl.ID()
        registered = []  # iterator names we added to the scope, to clean up
        i = 0
        while i < len(ids):
            var_name = ids[i].getText()
            iter_type = None

            # Explicit ``e: T`` annotation — second ID is the type name.
            if i + 1 < len(ids):
                type_name = ids[i + 1].getText()
                explicit = self._get_class_by_name(type_name)
                if explicit is not None:
                    iter_type = explicit
                    self.iterators_context[var_name] = explicit
                    loop.addIterator(IteratorExp(var_name, explicit))
                    registered.append(var_name)
                    i += 2
                    continue

            # No annotation — infer from the iterated source.
            inferred = self._infer_collection_element_type(source)
            iter_type = inferred if inferred is not None else self.context_class
            stored_type = inferred if inferred is not None else "NotMentioned"
            self.iterators_context[var_name] = iter_type
            loop.addIterator(IteratorExp(var_name, stored_type))
            registered.append(var_name)
            i += 1

        body = self.visit(ctx.expression(1))
        loop.add_body(body)

        for var_name in registered:
            self.iterators_context.pop(var_name, None)
        return loop
