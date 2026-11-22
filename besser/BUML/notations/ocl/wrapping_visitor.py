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
4. **Incomplete iterator-variable typing in ``visitArrowIterator``**: the
   base visitor types an untyped variable (``forAll(e | …)``) from association
   ends, ``allInstances()``, ``select`` and ``reject``; this subclass also
   follows ``asSet`` and the ordered-collection operations, and records the
   inferred type on the ``IteratorExp``.

This subclass overrides the affected methods to fix all of the above. Until
the fixes are upstreamed into ``BOCLVisitorImpl`` itself, all consumers
should parse via ``WrappingVisitor`` (or via :func:`parse_ocl`, which uses it).
"""

from besser.BUML.metamodel.ocl.ocl import (
    IteratorExp, LoopExp, OperationCallExpression,
    OrderedSetType, PropertyCallExpression, SequenceType, SetType, TypeExp,
    VariableExp,
)
from besser.BUML.metamodel.structural import Class, Property
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError
from besser.BUML.notations.ocl.visitor import BOCLVisitorImpl


# Operations whose result is a Sequence or an OrderedSet.
_ORDERED_OPERATIONS = {"SUBSEQUENCE", "SUBORDEREDSET", "APPEND", "PREPEND"}


class WrappingVisitor(BOCLVisitorImpl):
    """``BOCLVisitorImpl`` subclass that produces a chain-preserving AST."""

    def __init__(self, dm, om, context_class):
        super().__init__(dm, om, context_class)
        # Element types of the enclosing short-form iterators (``src->op(body)``),
        # innermost last: their implicit iterator is the body's implicit source.
        self._implicit_iterator_types = []

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
        if isinstance(receiver, Property):  # implicit-source navigation
            return receiver.type
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
        if isinstance(source, (VariableExp, Property)):
            return self._receiver_type(source)
        if isinstance(source, OperationCallExpression):
            if source.operation == "ALLInstances":
                type_exp = source.source
                if isinstance(type_exp, TypeExp):
                    return self._get_class_by_name(type_exp.name)
            if source.operation == "ASSET" or source.operation in _ORDERED_OPERATIONS:
                return self._infer_collection_element_type(source.source)
            return None
        if isinstance(source, LoopExp):
            # select / reject / collect / closure preserve element type.
            return self._infer_collection_element_type(source.source)
        return None

    def _check_closure(self, loop):
        """Validate a ``closure`` loop and set its result type.

        OCL 2.4 §8.3.7 (IteratorExp closure): one iterator; the body yields
        the source element type or a collection of it; the result is an
        OrderedSet for a Sequence/OrderedSet source and a Set otherwise.
        """
        if len(loop.iterator) > 1:
            raise BOCLSyntaxError(["'closure' takes exactly one iterator variable"])
        element = self._infer_collection_element_type(loop.source)
        body = loop.body[0]
        if (isinstance(body, VariableExp) and body.name != "self"
                and body.name not in self.iterators_context):
            # ``visitIdExpr`` found no property of that name.
            raise BOCLSyntaxError([
                f"'{body.name}' in the 'closure' body is not a property of "
                f"'{element.name if element is not None else '?'}'"
            ])
        body_type = None
        if not (isinstance(body, LoopExp) and body.name == "collect"):
            body_type = self._infer_collection_element_type(body)
        if isinstance(element, Class) and body_type is not None and not (
                body_type is element
                or isinstance(body_type, Class) and element in body_type.all_parents()):
            raise BOCLSyntaxError([
                f"The 'closure' body must yield {element.name} or a collection "
                f"of {element.name}, not {body_type.name}"
            ])
        ordered = (
            isinstance(loop.source, OperationCallExpression)
            and loop.source.operation in _ORDERED_OPERATIONS
        ) or isinstance(getattr(loop.source, "type", None), (OrderedSetType, SequenceType))
        kind, result_type = ("OrderedSet", OrderedSetType) if ordered else ("Set", SetType)
        loop.type = result_type(f"{kind}({element.name})" if element is not None else kind)

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
            owner = target_type if isinstance(target_type, Class) else self.context_class
            raise BOCLSyntaxError(
                [f"Property '{name}' not found in context '{ctx_name}'"],
                missing_element=f"{owner.name}.{name}" if owner is not None else name,
            )
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

    def visitDotDateNavigation(self, ctx):
        receiver = self.visit(ctx.expression())
        return self._wrap_navigation(receiver, ctx.DATE().getText(), ctx)

    def visitDotClosureNavigation(self, ctx):
        receiver = self.visit(ctx.expression())
        return self._wrap_navigation(receiver, "closure", ctx)

    # ``visitAllInstancesExp`` (``X.allInstances()``), ``visitArrowasSet``,
    # ``visitIntersection`` and ``visitIsunique`` live on ``BOCLVisitorImpl``:
    # the B-OCL evaluator (``bocl.OCLWrapper``) builds its AST with the base
    # visitor, so they must be there for evaluation to see them too.

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
        if op_name == "closure":
            self._check_closure(loop)

        for var_name in registered:
            self.iterators_context.pop(var_name, None)
        return loop

    def visitIdExpr(self, ctx):
        # OCL 2.4 §7.8: an implicit source resolves against the innermost
        # implicit iterator before ``self``.
        name = ctx.ID().getText()
        if name not in self.iterators_context:
            for element in reversed(self._implicit_iterator_types):
                if isinstance(element, Class):
                    prop = self._resolve_property(name, element)
                    if prop is not None:
                        return prop
        return super().visitIdExpr(ctx)

    def visitArrowIteratorShort(self, ctx: BOCLParser.ArrowIteratorShortContext):
        source = self.visit(ctx.expression(0))
        loop = LoopExp(ctx.iteratorOp().getText(), None)
        loop.source = source

        self._implicit_iterator_types.append(self._infer_collection_element_type(source))
        loop.add_body(self.visit(ctx.expression(1)))
        self._implicit_iterator_types.pop()
        if loop.name == "closure":
            self._check_closure(loop)
        return loop
