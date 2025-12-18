# Generated from D:/Projects/BESSER/besser/BUML/notations/action_language/BESSERActionLanguage.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .BESSERActionLanguageParser import BESSERActionLanguageParser
else:
    from BESSERActionLanguageParser import BESSERActionLanguageParser

# This class defines a complete generic visitor for a parse tree produced by BESSERActionLanguageParser.

class BESSERActionLanguageVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by BESSERActionLanguageParser#function_definition.
    def visitFunction_definition(self, ctx:BESSERActionLanguageParser.Function_definitionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#parameter.
    def visitParameter(self, ctx:BESSERActionLanguageParser.ParameterContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#statements.
    def visitStatements(self, ctx:BESSERActionLanguageParser.StatementsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#cond_loop.
    def visitCond_loop(self, ctx:BESSERActionLanguageParser.Cond_loopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#while.
    def visitWhile(self, ctx:BESSERActionLanguageParser.WhileContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#do_while.
    def visitDo_while(self, ctx:BESSERActionLanguageParser.Do_whileContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#for.
    def visitFor(self, ctx:BESSERActionLanguageParser.ForContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#iterator.
    def visitIterator(self, ctx:BESSERActionLanguageParser.IteratorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#conditional_branch.
    def visitConditional_branch(self, ctx:BESSERActionLanguageParser.Conditional_branchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#block.
    def visitBlock(self, ctx:BESSERActionLanguageParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#condition.
    def visitCondition(self, ctx:BESSERActionLanguageParser.ConditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#return.
    def visitReturn(self, ctx:BESSERActionLanguageParser.ReturnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#type.
    def visitType(self, ctx:BESSERActionLanguageParser.TypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#single_type.
    def visitSingle_type(self, ctx:BESSERActionLanguageParser.Single_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#sequence_type.
    def visitSequence_type(self, ctx:BESSERActionLanguageParser.Sequence_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#function_type.
    def visitFunction_type(self, ctx:BESSERActionLanguageParser.Function_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#any_type.
    def visitAny_type(self, ctx:BESSERActionLanguageParser.Any_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#classifier_type.
    def visitClassifier_type(self, ctx:BESSERActionLanguageParser.Classifier_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#real_type.
    def visitReal_type(self, ctx:BESSERActionLanguageParser.Real_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#string_type.
    def visitString_type(self, ctx:BESSERActionLanguageParser.String_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#int_type.
    def visitInt_type(self, ctx:BESSERActionLanguageParser.Int_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#bool_type.
    def visitBool_type(self, ctx:BESSERActionLanguageParser.Bool_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#expression.
    def visitExpression(self, ctx:BESSERActionLanguageParser.ExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#assign_target.
    def visitAssign_target(self, ctx:BESSERActionLanguageParser.Assign_targetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#explicit_declaration.
    def visitExplicit_declaration(self, ctx:BESSERActionLanguageParser.Explicit_declarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#assignment.
    def visitAssignment(self, ctx:BESSERActionLanguageParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#ternary.
    def visitTernary(self, ctx:BESSERActionLanguageParser.TernaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#boolean.
    def visitBoolean(self, ctx:BESSERActionLanguageParser.BooleanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#or.
    def visitOr(self, ctx:BESSERActionLanguageParser.OrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#and.
    def visitAnd(self, ctx:BESSERActionLanguageParser.AndContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#equality.
    def visitEquality(self, ctx:BESSERActionLanguageParser.EqualityContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#comparison.
    def visitComparison(self, ctx:BESSERActionLanguageParser.ComparisonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#instanceof.
    def visitInstanceof(self, ctx:BESSERActionLanguageParser.InstanceofContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#arithmetic.
    def visitArithmetic(self, ctx:BESSERActionLanguageParser.ArithmeticContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#plus_minus.
    def visitPlus_minus(self, ctx:BESSERActionLanguageParser.Plus_minusContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#mult_div.
    def visitMult_div(self, ctx:BESSERActionLanguageParser.Mult_divContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#remain.
    def visitRemain(self, ctx:BESSERActionLanguageParser.RemainContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#primary.
    def visitPrimary(self, ctx:BESSERActionLanguageParser.PrimaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#not.
    def visitNot(self, ctx:BESSERActionLanguageParser.NotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#minus.
    def visitMinus(self, ctx:BESSERActionLanguageParser.MinusContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#cast.
    def visitCast(self, ctx:BESSERActionLanguageParser.CastContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#null_coalessing.
    def visitNull_coalessing(self, ctx:BESSERActionLanguageParser.Null_coalessingContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#ArrayAccess.
    def visitArrayAccess(self, ctx:BESSERActionLanguageParser.ArrayAccessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#FunctionCall.
    def visitFunctionCall(self, ctx:BESSERActionLanguageParser.FunctionCallContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#FieldAccess.
    def visitFieldAccess(self, ctx:BESSERActionLanguageParser.FieldAccessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#Atom.
    def visitAtom(self, ctx:BESSERActionLanguageParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#field_access.
    def visitField_access(self, ctx:BESSERActionLanguageParser.Field_accessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#array_access.
    def visitArray_access(self, ctx:BESSERActionLanguageParser.Array_accessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#function_call.
    def visitFunction_call(self, ctx:BESSERActionLanguageParser.Function_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#atomic.
    def visitAtomic(self, ctx:BESSERActionLanguageParser.AtomicContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#procedure_call.
    def visitProcedure_call(self, ctx:BESSERActionLanguageParser.Procedure_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#this.
    def visitThis(self, ctx:BESSERActionLanguageParser.ThisContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#new.
    def visitNew(self, ctx:BESSERActionLanguageParser.NewContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#literal.
    def visitLiteral(self, ctx:BESSERActionLanguageParser.LiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#single_literal.
    def visitSingle_literal(self, ctx:BESSERActionLanguageParser.Single_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#int_literal.
    def visitInt_literal(self, ctx:BESSERActionLanguageParser.Int_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#string_literal.
    def visitString_literal(self, ctx:BESSERActionLanguageParser.String_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#bool_literal.
    def visitBool_literal(self, ctx:BESSERActionLanguageParser.Bool_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#real_literal.
    def visitReal_literal(self, ctx:BESSERActionLanguageParser.Real_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#null_literal.
    def visitNull_literal(self, ctx:BESSERActionLanguageParser.Null_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#enum_literal.
    def visitEnum_literal(self, ctx:BESSERActionLanguageParser.Enum_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#sequence_literal.
    def visitSequence_literal(self, ctx:BESSERActionLanguageParser.Sequence_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#range_literal.
    def visitRange_literal(self, ctx:BESSERActionLanguageParser.Range_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#symbol.
    def visitSymbol(self, ctx:BESSERActionLanguageParser.SymbolContext):
        return self.visitChildren(ctx)



del BESSERActionLanguageParser