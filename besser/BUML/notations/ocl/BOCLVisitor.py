# Generated from BOCL.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .BOCLParser import BOCLParser
else:
    from BOCLParser import BOCLParser

# This class defines a complete generic visitor for a parse tree produced by BOCLParser.

class BOCLVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by BOCLParser#oclFile.
    def visitOclFile(self, ctx:BOCLParser.OclFileContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#preCondition.
    def visitPreCondition(self, ctx:BOCLParser.PreConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#postCondition.
    def visitPostCondition(self, ctx:BOCLParser.PostConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#initConstraints.
    def visitInitConstraints(self, ctx:BOCLParser.InitConstraintsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#contextDeclaration.
    def visitContextDeclaration(self, ctx:BOCLParser.ContextDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#constraint.
    def visitConstraint(self, ctx:BOCLParser.ConstraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#paramList.
    def visitParamList(self, ctx:BOCLParser.ParamListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#param.
    def visitParam(self, ctx:BOCLParser.ParamContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#typeRef.
    def visitTypeRef(self, ctx:BOCLParser.TypeRefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#primitiveType.
    def visitPrimitiveType(self, ctx:BOCLParser.PrimitiveTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#collectionType.
    def visitCollectionType(self, ctx:BOCLParser.CollectionTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#iteratorVarDecl.
    def visitIteratorVarDecl(self, ctx:BOCLParser.IteratorVarDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#iteratorOp.
    def visitIteratorOp(self, ctx:BOCLParser.IteratorOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#compOp.
    def visitCompOp(self, ctx:BOCLParser.CompOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#argList.
    def visitArgList(self, ctx:BOCLParser.ArgListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#dotMethodCall.
    def visitDotMethodCall(self, ctx:BOCLParser.DotMethodCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowIterator.
    def visitArrowIterator(self, ctx:BOCLParser.ArrowIteratorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowAppend.
    def visitArrowAppend(self, ctx:BOCLParser.ArrowAppendContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowSubOrderedSet.
    def visitArrowSubOrderedSet(self, ctx:BOCLParser.ArrowSubOrderedSetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#ifThenElseExpr.
    def visitIfThenElseExpr(self, ctx:BOCLParser.IfThenElseExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#dotSize.
    def visitDotSize(self, ctx:BOCLParser.DotSizeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#dotOclIsTypeOf.
    def visitDotOclIsTypeOf(self, ctx:BOCLParser.DotOclIsTypeOfContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#xorExpr.
    def visitXorExpr(self, ctx:BOCLParser.XorExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowSubSequence.
    def visitArrowSubSequence(self, ctx:BOCLParser.ArrowSubSequenceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowIncludes.
    def visitArrowIncludes(self, ctx:BOCLParser.ArrowIncludesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowSymDiff.
    def visitArrowSymDiff(self, ctx:BOCLParser.ArrowSymDiffContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#unaryExpr.
    def visitUnaryExpr(self, ctx:BOCLParser.UnaryExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#dotNavigation.
    def visitDotNavigation(self, ctx:BOCLParser.DotNavigationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#primaryExpr.
    def visitPrimaryExpr(self, ctx:BOCLParser.PrimaryExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowIteratorShort.
    def visitArrowIteratorShort(self, ctx:BOCLParser.ArrowIteratorShortContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowUnion.
    def visitArrowUnion(self, ctx:BOCLParser.ArrowUnionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#addSubExpr.
    def visitAddSubExpr(self, ctx:BOCLParser.AddSubExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowFirst.
    def visitArrowFirst(self, ctx:BOCLParser.ArrowFirstContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#dotOclIsKindOf.
    def visitDotOclIsKindOf(self, ctx:BOCLParser.DotOclIsKindOfContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#orExpr.
    def visitOrExpr(self, ctx:BOCLParser.OrExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#comparisonExpr.
    def visitComparisonExpr(self, ctx:BOCLParser.ComparisonExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowLast.
    def visitArrowLast(self, ctx:BOCLParser.ArrowLastContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#dotOclAsType.
    def visitDotOclAsType(self, ctx:BOCLParser.DotOclAsTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowIsEmpty.
    def visitArrowIsEmpty(self, ctx:BOCLParser.ArrowIsEmptyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowSize.
    def visitArrowSize(self, ctx:BOCLParser.ArrowSizeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#impliesExpr.
    def visitImpliesExpr(self, ctx:BOCLParser.ImpliesExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowSum.
    def visitArrowSum(self, ctx:BOCLParser.ArrowSumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowExcludes.
    def visitArrowExcludes(self, ctx:BOCLParser.ArrowExcludesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#arrowPrepend.
    def visitArrowPrepend(self, ctx:BOCLParser.ArrowPrependContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#mulDivExpr.
    def visitMulDivExpr(self, ctx:BOCLParser.MulDivExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#andExpr.
    def visitAndExpr(self, ctx:BOCLParser.AndExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#selfExpr.
    def visitSelfExpr(self, ctx:BOCLParser.SelfExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#stringLiteral.
    def visitStringLiteral(self, ctx:BOCLParser.StringLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#numberLiteral.
    def visitNumberLiteral(self, ctx:BOCLParser.NumberLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#booleanLiteral.
    def visitBooleanLiteral(self, ctx:BOCLParser.BooleanLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#nullLiteral.
    def visitNullLiteral(self, ctx:BOCLParser.NullLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#allInstancesExpr.
    def visitAllInstancesExpr(self, ctx:BOCLParser.AllInstancesExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#dateFuncExpr.
    def visitDateFuncExpr(self, ctx:BOCLParser.DateFuncExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#functionCallExpr.
    def visitFunctionCallExpr(self, ctx:BOCLParser.FunctionCallExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#parenExpr.
    def visitParenExpr(self, ctx:BOCLParser.ParenExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BOCLParser#idExpr.
    def visitIdExpr(self, ctx:BOCLParser.IdExprContext):
        return self.visitChildren(ctx)



del BOCLParser