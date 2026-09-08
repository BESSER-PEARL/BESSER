from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, LoopExp, IfExp, IteratorExp,
    IntegerLiteralExpression, RealLiteralExpression,
    BooleanLiteralExpression, StringLiteralExpression,
    DateLiteralExpression, InfixOperator, PropertyCallExpression,
    TypeExp, VariableExp,
)
from besser.BUML.metamodel.structural.structural import Property
from .BOCLVisitor import BOCLVisitor
from .BOCLParser import BOCLParser


class BOCLVisitorImpl(BOCLVisitor):
    """Visitor that builds an OCL AST from the ANTLR4 parse tree.

    Replaces the old BOCLListener + RootHandler + FactoryInstance chain
    with a single, clean visitor that returns AST nodes directly.
    """

    def __init__(self, dm, om, context_class):
        self.dm = dm
        self.om = om
        self.context_class = context_class
        self.context_name = None
        self.iterators_context = {}

    # ------------------------------------------------------------------
    # Property / attribute resolution
    # ------------------------------------------------------------------
    def _resolve_property(self, name, context_class=None):
        if context_class is None:
            context_class = self.context_class
        if context_class is None:
            return None
        for attr in context_class.all_attributes():
            if attr.name == name:
                return attr
        for end in context_class.all_association_ends():
            if end.name == name:
                return end
        return None

    def _resolve_property_in_iterators(self, name):
        for var_name, cls in self.iterators_context.items():
            prop = self._resolve_property(name, cls)
            if prop is not None:
                return prop
        return None

    def _receiver_type(self, receiver):
        """Return the static type of `receiver`, or None if it can't be inferred."""
        if isinstance(receiver, VariableExp):
            if receiver.name == "self":
                return self.context_class
            return self.iterators_context.get(receiver.name)
        if isinstance(receiver, Property):
            return receiver.type
        if isinstance(receiver, PropertyCallExpression):
            return receiver.property.type
        return None

    def _get_class_by_name(self, name):
        for t in self.dm.types:
            if t.name == name:
                return t
        return None

    # ------------------------------------------------------------------
    # Top-level rules
    # ------------------------------------------------------------------
    def visitOclFile(self, ctx: BOCLParser.OclFileContext):
        return self.visit(ctx.getChild(0))

    def visitContextDeclaration(self, ctx: BOCLParser.ContextDeclarationContext):
        self.context_name = ctx.ID().getText()
        results = []
        for c in ctx.constraint():
            results.append(self.visit(c))
        if len(results) == 1:
            return results[0]
        return results

    def visitConstraint(self, ctx: BOCLParser.ConstraintContext):
        return self.visit(ctx.expression())

    def visitPreCondition(self, ctx: BOCLParser.PreConditionContext):
        self.context_name = ctx.ID(0).getText()
        return self.visit(ctx.expression())

    def visitPostCondition(self, ctx: BOCLParser.PostConditionContext):
        self.context_name = ctx.ID(0).getText()
        return self.visit(ctx.expression())

    def visitInitConstraints(self, ctx: BOCLParser.InitConstraintsContext):
        self.context_name = ctx.ID(0).getText()
        return self.visit(ctx.expression())

    # ------------------------------------------------------------------
    # Expression alternatives
    # ------------------------------------------------------------------

    # --- Dot postfix ---
    def visitDotNavigation(self, ctx: BOCLParser.DotNavigationContext):
        receiver = self.visit(ctx.expression())
        name = ctx.ID().getText()
        # Resolve against the receiver's actual type so chained navigation
        # like `self.dept.maxsalary` looks up `maxsalary` on Dept (the type
        # of `self.dept`), not on the constraint's context class.
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
            raise Exception(f"Property '{name}' not found in context '{ctx_name}'")
        # Top-level access (`self.x`, `iter.x`) keeps the legacy bare-Property
        # AST so existing evaluator paths still work. Chained navigation
        # (`self.x.y`) wraps in a PropertyCallExpression so the evaluator can
        # walk the chain by traversing association ends.
        if isinstance(receiver, VariableExp):
            return prop
        pce = PropertyCallExpression(prop.name, prop)
        pce.source = receiver
        return pce

    def visitDotSize(self, ctx: BOCLParser.DotSizeContext):
        source = self.visit(ctx.expression())
        oce = OperationCallExpression(name="Size", operation="Size", arguments=[])
        oce.source = source
        return oce

    def visitDotSizeNavigation(self, ctx: BOCLParser.DotSizeNavigationContext):
        """Resolve ``self.size`` as a property when no parens follow.

        ``size`` is a reserved OCL keyword for the ``->size()`` / ``.size()``
        collection op, so the lexer always tokenizes it as ``SIZE`` and the
        plain ``dotNavigation`` rule (which expects ``ID``) never matches.
        Without this fallback, any domain model with an attribute literally
        named ``size`` fails to parse even though the metamodel allows it
        (see BESSER-PEARL/BESSER#198).
        """
        receiver = self.visit(ctx.expression())
        prop = None
        target_type = self._receiver_type(receiver)
        if target_type is not None:
            prop = self._resolve_property("size", target_type)
        if prop is None:
            prop = self._resolve_property("size")
        if prop is None:
            prop = self._resolve_property_in_iterators("size")
        if prop is None:
            raise Exception(
                f"Property 'size' not found in context "
                f"'{self.context_class.name if self.context_class else '?'}'"
            )
        if isinstance(receiver, VariableExp):
            return prop
        pce = PropertyCallExpression(prop.name, prop)
        pce.source = receiver
        return pce

    def visitDotMethodCall(self, ctx: BOCLParser.DotMethodCallContext):
        source = self.visit(ctx.expression())
        method_name = ctx.ID().getText()
        args = []
        if ctx.argList():
            args = [self.visit(e) for e in ctx.argList().expression()]

        # Handle Date method chains: Date::today().addDays(N)
        if isinstance(source, DateLiteralExpression) and method_name == "addDays":
            days = ""
            if args and isinstance(args[0], IntegerLiteralExpression):
                days = str(args[0].value)
            return DateLiteralExpression("date", f"{source.value}.addDays({days})")

        oce = OperationCallExpression(name=method_name, operation=method_name, arguments=args)
        oce.source = source
        return oce

    def visitDotOclIsTypeOf(self, ctx: BOCLParser.DotOclIsTypeOfContext):
        source = self.visit(ctx.expression())
        type_name = ctx.typeRef().getText()
        type_exp = TypeExp(type_name, type_name)
        oce = OperationCallExpression(name="OCLISTYPEOF", operation="OCLISTYPEOF", arguments=[type_exp])
        oce.source = source
        return oce

    def visitDotOclAsType(self, ctx: BOCLParser.DotOclAsTypeContext):
        source = self.visit(ctx.expression())
        type_name = ctx.typeRef().getText()
        type_exp = TypeExp(type_name, type_name)
        oce = OperationCallExpression(name="OCLASTYPE", operation="OCLASTYPE", arguments=[type_exp])
        oce.source = source
        return oce

    def visitDotOclIsKindOf(self, ctx: BOCLParser.DotOclIsKindOfContext):
        source = self.visit(ctx.expression())
        type_name = ctx.typeRef().getText()
        type_exp = TypeExp(type_name, type_name)
        oce = OperationCallExpression(name="OCLISKINDOF", operation="OCLISKINDOF", arguments=[type_exp])
        oce.source = source
        return oce

    # --- Arrow postfix ---
    def visitArrowIterator(self, ctx: BOCLParser.ArrowIteratorContext):
        source = self.visit(ctx.expression(0))
        op_name = ctx.iteratorOp().getText()
        loop = LoopExp(op_name, None)
        loop.source = source

        # Parse iterator variable declarations
        var_decl = ctx.iteratorVarDecl()
        ids = var_decl.ID()
        i = 0
        while i < len(ids):
            var_name = ids[i].getText()
            var_type = None
            # Check if next token is a type (COLON ID pattern)
            if i + 1 < len(ids):
                # Check if there's a colon between them in the original text
                type_name = ids[i + 1].getText()
                var_type = self._get_class_by_name(type_name)
                if var_type is not None:
                    self.iterators_context[var_name] = var_type
                    it = IteratorExp(var_name, var_type)
                    loop.addIterator(it)
                    i += 2
                    continue
            # No type annotation
            it = IteratorExp(var_name, "NotMentioned")
            loop.addIterator(it)
            self.iterators_context[var_name] = self.context_class
            i += 1

        # Visit the body expression
        body = self.visit(ctx.expression(1))
        loop.add_body(body)

        # Clean up iterator context
        for it in loop.get_iterator:
            if it.name in self.iterators_context:
                del self.iterators_context[it.name]

        return loop

    def visitArrowIteratorShort(self, ctx: BOCLParser.ArrowIteratorShortContext):
        source = self.visit(ctx.expression(0))
        op_name = ctx.iteratorOp().getText()
        loop = LoopExp(op_name, None)
        loop.source = source

        body = self.visit(ctx.expression(1))
        loop.add_body(body)
        return loop

    def visitArrowSize(self, ctx: BOCLParser.ArrowSizeContext):
        source = self.visit(ctx.expression())
        oce = OperationCallExpression(name="Size", operation="Size", arguments=[])
        oce.source = source
        return oce

    def visitArrowIsEmpty(self, ctx: BOCLParser.ArrowIsEmptyContext):
        source = self.visit(ctx.expression())
        oce = OperationCallExpression(name="IsEmpty", operation="IsEmpty", arguments=[])
        oce.source = source
        return oce

    def visitArrowSum(self, ctx: BOCLParser.ArrowSumContext):
        source = self.visit(ctx.expression())
        oce = OperationCallExpression(name="Sum", operation="Sum", arguments=[])
        oce.source = source
        return oce

    def visitArrowIncludes(self, ctx: BOCLParser.ArrowIncludesContext):
        source = self.visit(ctx.expression(0))
        arg = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="INCLUDES", operation="INCLUDES", arguments=[arg])
        oce.source = source
        return oce

    def visitArrowExcludes(self, ctx: BOCLParser.ArrowExcludesContext):
        source = self.visit(ctx.expression(0))
        arg = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="EXCLUDES", operation="EXCLUDES", arguments=[arg])
        oce.source = source
        return oce

    def visitArrowUnion(self, ctx: BOCLParser.ArrowUnionContext):
        source = self.visit(ctx.expression(0))
        arg = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="UNION", operation="UNION", arguments=[arg])
        oce.source = source
        return oce

    def visitArrowFirst(self, ctx: BOCLParser.ArrowFirstContext):
        source = self.visit(ctx.expression())
        oce = OperationCallExpression(name="First", operation="First", arguments=[])
        oce.source = source
        return oce

    def visitArrowLast(self, ctx: BOCLParser.ArrowLastContext):
        source = self.visit(ctx.expression())
        oce = OperationCallExpression(name="Last", operation="Last", arguments=[])
        oce.source = source
        return oce

    def visitArrowPrepend(self, ctx: BOCLParser.ArrowPrependContext):
        source = self.visit(ctx.expression(0))
        arg = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="PREPEND", operation="PREPEND", arguments=[arg])
        oce.source = source
        return oce

    def visitArrowAppend(self, ctx: BOCLParser.ArrowAppendContext):
        source = self.visit(ctx.expression(0))
        arg = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="APPEND", operation="APPEND", arguments=[arg])
        oce.source = source
        return oce

    def visitArrowSymDiff(self, ctx: BOCLParser.ArrowSymDiffContext):
        source = self.visit(ctx.expression(0))
        arg = self.visit(ctx.expression(1))
        oce = OperationCallExpression(
            name="SYMMETRICDIFFERENCE", operation="SYMMETRICDIFFERENCE", arguments=[arg]
        )
        oce.source = source
        return oce

    def visitArrowSubSequence(self, ctx: BOCLParser.ArrowSubSequenceContext):
        source = self.visit(ctx.expression(0))
        arg1 = self.visit(ctx.expression(1))
        arg2 = self.visit(ctx.expression(2))
        oce = OperationCallExpression(
            name="SUBSEQUENCE", operation="SUBSEQUENCE", arguments=[arg1, arg2]
        )
        oce.source = source
        return oce

    def visitArrowSubOrderedSet(self, ctx: BOCLParser.ArrowSubOrderedSetContext):
        source = self.visit(ctx.expression(0))
        arg1 = self.visit(ctx.expression(1))
        arg2 = self.visit(ctx.expression(2))
        oce = OperationCallExpression(
            name="SUBORDEREDSET", operation="SUBORDEREDSET", arguments=[arg1, arg2]
        )
        oce.source = source
        return oce

    # --- Unary ---
    def visitUnaryExpr(self, ctx: BOCLParser.UnaryExprContext):
        operand = self.visit(ctx.expression())
        if ctx.NOT():
            op = "not"
        else:
            op = "-"
        oce = OperationCallExpression(name="unary_" + op, operation=op, arguments=[operand])
        return oce

    # --- Binary arithmetic ---
    def visitMulDivExpr(self, ctx: BOCLParser.MulDivExprContext):
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.getChild(1).getText()
        infix = InfixOperator(op)
        oce = OperationCallExpression(name="Operation", operation=op, arguments=[left, infix, right])
        return oce

    def visitAddSubExpr(self, ctx: BOCLParser.AddSubExprContext):
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.getChild(1).getText()
        infix = InfixOperator(op)
        oce = OperationCallExpression(name="Operation", operation=op, arguments=[left, infix, right])
        return oce

    # --- Comparison ---
    def visitComparisonExpr(self, ctx: BOCLParser.ComparisonExprContext):
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.compOp().getText()
        infix = InfixOperator(op)
        oce = OperationCallExpression(name="Operation", operation=op, arguments=[left, infix, right])
        return oce

    # --- Logical ---
    def visitAndExpr(self, ctx: BOCLParser.AndExprContext):
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="AND_BINARY", operation="and", arguments=[left, right])
        return oce

    def visitOrExpr(self, ctx: BOCLParser.OrExprContext):
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="OR_BINARY", operation="or", arguments=[left, right])
        return oce

    def visitXorExpr(self, ctx: BOCLParser.XorExprContext):
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="XOR_BINARY", operation="xor", arguments=[left, right])
        return oce

    def visitImpliesExpr(self, ctx: BOCLParser.ImpliesExprContext):
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        oce = OperationCallExpression(name="IMPLIES_BINARY", operation="implies", arguments=[left, right])
        return oce

    # --- If-then-else ---
    def visitIfThenElseExpr(self, ctx: BOCLParser.IfThenElseExprContext):
        condition = self.visit(ctx.expression(0))
        then_expr = self.visit(ctx.expression(1))
        else_expr = self.visit(ctx.expression(2))
        if_exp = IfExp("if", "IfExpression")
        if_exp.ifCondition = condition
        if_exp.thenExpression = then_expr
        if_exp.elseCondition = else_expr
        return if_exp

    # --- Primary pass-through ---
    def visitPrimaryExpr(self, ctx: BOCLParser.PrimaryExprContext):
        return self.visit(ctx.primaryExpression())

    # ------------------------------------------------------------------
    # Primary expression alternatives
    # ------------------------------------------------------------------
    def visitSelfExpr(self, ctx: BOCLParser.SelfExprContext):
        return VariableExp("self", None)

    def visitStringLiteral(self, ctx: BOCLParser.StringLiteralContext):
        raw = ctx.STRING_LITERAL().getText()
        # Remove surrounding single quotes
        value = raw[1:-1]
        return StringLiteralExpression("str", value)

    def visitNumberLiteral(self, ctx: BOCLParser.NumberLiteralContext):
        text = ctx.NUMBER().getText()
        if '.' in text:
            return RealLiteralExpression("NP", float(text))
        return IntegerLiteralExpression("NP", int(text))

    def visitBooleanLiteral(self, ctx: BOCLParser.BooleanLiteralContext):
        text = ctx.BOOLEAN_LITERAL().getText()
        return BooleanLiteralExpression("NP", text)

    def visitNullLiteral(self, ctx: BOCLParser.NullLiteralContext):
        return None

    def visitAllInstancesExpr(self, ctx: BOCLParser.AllInstancesExprContext):
        class_name = ctx.ID().getText()
        type_exp = TypeExp(class_name, class_name)
        oce = OperationCallExpression(name="ALLInstances", operation="ALLInstances", arguments=[])
        oce.referredOperation = OperationCallExpression(
            name="ALLInstances", operation="ALLInstances", arguments=[]
        )
        oce.source = type_exp
        return oce

    def visitDateFuncExpr(self, ctx: BOCLParser.DateFuncExprContext):
        func_name = ctx.ID().getText()
        date_str = f"Date::{func_name}()"
        return DateLiteralExpression("date", date_str)

    def visitFunctionCallExpr(self, ctx: BOCLParser.FunctionCallExprContext):
        name = ctx.ID().getText()
        args = []
        if ctx.argList():
            args = [self.visit(e) for e in ctx.argList().expression()]
        oce = OperationCallExpression(name=name, operation=name, arguments=args)
        return oce

    def visitParenExpr(self, ctx: BOCLParser.ParenExprContext):
        return self.visit(ctx.expression())

    def visitIdExpr(self, ctx: BOCLParser.IdExprContext):
        name = ctx.ID().getText()
        # Check if it's a known iterator variable
        if name in self.iterators_context:
            return VariableExp(name, None)
        # Try to resolve as a property
        prop = self._resolve_property(name)
        if prop is not None:
            return prop
        prop = self._resolve_property_in_iterators(name)
        if prop is not None:
            return prop
        # Return as variable expression
        return VariableExp(name, None)
