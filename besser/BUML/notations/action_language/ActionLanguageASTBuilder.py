# Generated from D:/Projects/BESSER/besser/BUML/notations/action_language/BESSERActionLanguage.g4 by ANTLR 4.13.2
from antlr4 import *
from win32comext.mapi.mapi import FORCE_SAVE

from ...metamodel.action_language.action_language import FunctionDefinition, Parameter, While, DoWhile, For, Iterator, \
    Statements, Block, Condition, AnyType, SequenceType, RealType, StringType, IntType, BoolType, Assignment, Ternary, \
    Or, And, Equal, LessEq, InstanceOf, Plus, Remain, Mult, Not, UnaryMinus, Cast, NullCoalessing, ArrayAccess, \
    FunctionCall, FieldAccess, This, New, IntLiteral, StringLiteral, BoolLiteral, RealLiteral, NullLiteral, EnumLiteral, \
    SequenceLiteral, RangeLiteral, NameDecl, Reference, ImplicitDecl, Unequal, Greater, GreaterEq, Less, Minus, Div
from ...metamodel.structural import DomainModel

if "." in __name__:
    from .BESSERActionLanguageParser import BESSERActionLanguageParser
else:
    from BESSERActionLanguageParser import BESSERActionLanguageParser

# This class defines a complete generic visitor for a parse tree produced by BESSERActionLanguageParser.

class BESSERActionLanguageVisitor(ParseTreeVisitor):

    def __init__(self, domain_model: DomainModel):
        self.__symbols:dict[str, NameDecl] = {}
        self.__model = domain_model

    # Visit a parse tree produced by BESSERActionLanguageParser#function_definition.
    def visitFunction_definition(self, ctx:BESSERActionLanguageParser.Function_definitionContext):
        parameters:list[Parameter] = []
        for param in ctx.params:
            parameters.append(self.visit(param))

        return FunctionDefinition(
            ctx.name.text,
            parameters,
            self.visit(ctx.return_type),
            self.visit(ctx.body)
        )


    # Visit a parse tree produced by BESSERActionLanguageParser#parameter.
    def visitParameter(self, ctx:BESSERActionLanguageParser.ParameterContext):
        return Parameter(self.visit(ctx.name), self.visit(ctx.declared_type))


    # Visit a parse tree produced by BESSERActionLanguageParser#statements.
    def visitStatements(self, ctx:BESSERActionLanguageParser.StatementsContext):
        return self.visit(ctx.getChild(0))


    # Visit a parse tree produced by BESSERActionLanguageParser#cond_loop.
    def visitCond_loop(self, ctx:BESSERActionLanguageParser.Cond_loopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#while.
    def visitWhile(self, ctx:BESSERActionLanguageParser.WhileContext):
        return While(self.visit(ctx.cond), self.visit(ctx.body))


    # Visit a parse tree produced by BESSERActionLanguageParser#do_while.
    def visitDo_while(self, ctx:BESSERActionLanguageParser.Do_whileContext):
        return DoWhile(self.visit(ctx.cond), self.visit(ctx.body))


    # Visit a parse tree produced by BESSERActionLanguageParser#for.
    def visitFor(self, ctx:BESSERActionLanguageParser.ForContext):
        iterators:set[Iterator] = set()
        for it in ctx.iterators:
            iterators.add(self.visit(it))
        return For(iterators, self.visit(ctx.body))


    # Visit a parse tree produced by BESSERActionLanguageParser#iterator.
    def visitIterator(self, ctx:BESSERActionLanguageParser.IteratorContext):
        return Iterator(self.visit(ctx.var_name), self.visit(ctx.sequence))


    # Visit a parse tree produced by BESSERActionLanguageParser#conditional_branch.
    def visitConditional_branch(self, ctx:BESSERActionLanguageParser.Conditional_branchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#block.
    def visitBlock(self, ctx:BESSERActionLanguageParser.BlockContext):
        children: list[Statements] = list()
        for elem in ctx.stmts:
            children.append(self.visit(elem))
        return Block(children)


    # Visit a parse tree produced by BESSERActionLanguageParser#condition.
    def visitCondition(self, ctx:BESSERActionLanguageParser.ConditionContext):
        elze = None
        if ctx.elze is not None:
            elze = self.visit(ctx.elze)
        return Condition(self.visit(ctx.cond), self.visit(ctx.then), elze)


    # Visit a parse tree produced by BESSERActionLanguageParser#type.
    def visitType(self, ctx:BESSERActionLanguageParser.TypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#single_type.
    def visitSingle_type(self, ctx:BESSERActionLanguageParser.Single_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#any_type.
    def visitAny_type(self, ctx:BESSERActionLanguageParser.Any_typeContext):
        return AnyType()


    # Visit a parse tree produced by BESSERActionLanguageParser#classifier_type.
    def visitClassifier_type(self, ctx:BESSERActionLanguageParser.Classifier_typeContext):
        # TODO: Find Class or Enum in model
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#sequence_type.
    def visitSequence_type(self, ctx:BESSERActionLanguageParser.Sequence_typeContext):
        return SequenceType(self.visit(ctx.the_type))


    # Visit a parse tree produced by BESSERActionLanguageParser#real_type.
    def visitReal_type(self, ctx:BESSERActionLanguageParser.Real_typeContext):
        return RealType()


    # Visit a parse tree produced by BESSERActionLanguageParser#string_type.
    def visitString_type(self, ctx:BESSERActionLanguageParser.String_typeContext):
        return StringType()


    # Visit a parse tree produced by BESSERActionLanguageParser#int_type.
    def visitInt_type(self, ctx:BESSERActionLanguageParser.Int_typeContext):
        return IntType()


    # Visit a parse tree produced by BESSERActionLanguageParser#bool_type.
    def visitBool_type(self, ctx:BESSERActionLanguageParser.Bool_typeContext):
        return BoolType()


    # Visit a parse tree produced by BESSERActionLanguageParser#expression.
    def visitExpression(self, ctx:BESSERActionLanguageParser.ExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#assign_target.
    def visitAssign_target(self, ctx:BESSERActionLanguageParser.Assign_targetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#assignment.
    def visitAssignment(self, ctx:BESSERActionLanguageParser.AssignmentContext):
        if ctx.target is None or ctx.assignee is None:
            return  self.visitChildren(ctx)
        return Assignment(self.visit(ctx.target), self.visit(ctx.assignee))


    # Visit a parse tree produced by BESSERActionLanguageParser#ternary.
    def visitTernary(self, ctx:BESSERActionLanguageParser.TernaryContext):
        if ctx.expr is None:
            return self.visitChildren(ctx)
        return Ternary(self.visit(ctx.expr), self.visit(ctx.then), self.visit(ctx.elze))


    # Visit a parse tree produced by BESSERActionLanguageParser#boolean.
    def visitBoolean(self, ctx:BESSERActionLanguageParser.BooleanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#or.
    def visitOr(self, ctx:BESSERActionLanguageParser.OrContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        return Or(self.visit(ctx.left), self.visit(ctx.right))


    # Visit a parse tree produced by BESSERActionLanguageParser#and.
    def visitAnd(self, ctx:BESSERActionLanguageParser.AndContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        return And(self.visit(ctx.left), self.visit(ctx.right))


    # Visit a parse tree produced by BESSERActionLanguageParser#equality.
    def visitEquality(self, ctx:BESSERActionLanguageParser.EqualityContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        match ctx.op:
            case "==":
                return Equal(self.visit(ctx.left), self.visit(ctx.right))
            case "!=":
                return Unequal(self.visit(ctx.left), self.visit(ctx.right))


    # Visit a parse tree produced by BESSERActionLanguageParser#comparison.
    def visitComparison(self, ctx:BESSERActionLanguageParser.ComparisonContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        match ctx.op:
            case "<":
                return Less(self.visit(ctx.left), self.visit(ctx.right))
            case "<=":
                return LessEq(self.visit(ctx.left), self.visit(ctx.right))
            case ">":
                return Greater(self.visit(ctx.left), self.visit(ctx.right))
            case ">=":
                return GreaterEq(self.visit(ctx.left), self.visit(ctx.right))


    # Visit a parse tree produced by BESSERActionLanguageParser#instanceof.
    def visitInstanceof(self, ctx:BESSERActionLanguageParser.InstanceofContext):
        if ctx.instance is None or ctx.the_type is None:
            return self.visitChildren(ctx)
        return InstanceOf(self.visit(ctx.instance), self.visit(ctx.the_type))


    # Visit a parse tree produced by BESSERActionLanguageParser#arithmetic.
    def visitArithmetic(self, ctx:BESSERActionLanguageParser.ArithmeticContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#plus_minus.
    def visitPlus_minus(self, ctx:BESSERActionLanguageParser.Plus_minusContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        match ctx.op:
            case '+':
                return Plus(self.visit(ctx.left), self.visit(ctx.right))
            case '-':
                return Minus(self.visit(ctx.left), self.visit(ctx.right))


    # Visit a parse tree produced by BESSERActionLanguageParser#mult_div.
    def visitMult_div(self, ctx:BESSERActionLanguageParser.Mult_divContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        match ctx.op:
            case '*':
                return Mult(self.visit(ctx.left), self.visit(ctx.right))
            case '/':
                return Div(self.visit(ctx.left), self.visit(ctx.right))


    # Visit a parse tree produced by BESSERActionLanguageParser#remain.
    def visitRemain(self, ctx:BESSERActionLanguageParser.RemainContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        return Remain(self.visit(ctx.left), self.visit(ctx.right))


    # Visit a parse tree produced by BESSERActionLanguageParser#primary.
    def visitPrimary(self, ctx:BESSERActionLanguageParser.PrimaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#not.
    def visitNot(self, ctx:BESSERActionLanguageParser.NotContext):
        return Not(self.visit(ctx.expr))


    # Visit a parse tree produced by BESSERActionLanguageParser#minus.
    def visitMinus(self, ctx:BESSERActionLanguageParser.MinusContext):
        return UnaryMinus(self.visit(ctx.expr))


    # Visit a parse tree produced by BESSERActionLanguageParser#cast.
    def visitCast(self, ctx:BESSERActionLanguageParser.CastContext):
        return Cast(self.visit(ctx.expr), self.visit(ctx.as_))


    # Visit a parse tree produced by BESSERActionLanguageParser#null_coalessing.
    def visitNull_coalessing(self, ctx:BESSERActionLanguageParser.Null_coalessingContext):
        return NullCoalessing(self.visit(ctx.nullable), self.visit(ctx.elze))


    # Visit a parse tree produced by BESSERActionLanguageParser#ArrayAccess.
    def visitArrayAccess(self, ctx:BESSERActionLanguageParser.ArrayAccessContext):
        return ArrayAccess(self.visit(ctx.receiver), self.visit(ctx.index))


    # Visit a parse tree produced by BESSERActionLanguageParser#FunctionCall.
    def visitFunctionCall(self, ctx:BESSERActionLanguageParser.FunctionCallContext):
        args = list()
        for arg in ctx.args:
            args.append(self.visit(arg))
        # TODO : Add method retrieval
        method = ctx.name.text
        return FunctionCall(self.visit(ctx.receiver), None, args)


    # Visit a parse tree produced by BESSERActionLanguageParser#FieldAccess.
    def visitFieldAccess(self, ctx:BESSERActionLanguageParser.FieldAccessContext):
        # TODO : add property retrieval
        field = ctx.field.text
        return FieldAccess(self.visit(ctx.receiver), None)


    # Visit a parse tree produced by BESSERActionLanguageParser#Atom.
    def visitAtom(self, ctx:BESSERActionLanguageParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#field_access.
    def visitField_access(self, ctx:BESSERActionLanguageParser.Field_accessContext):
        # TODO : add property retrieval
        field = ctx.field.text
        return FieldAccess(self.visit(ctx.receiver), None)


    # Visit a parse tree produced by BESSERActionLanguageParser#array_access.
    def visitArray_access(self, ctx:BESSERActionLanguageParser.Array_accessContext):
        return ArrayAccess(self.visit(ctx.receiver), self.visit(ctx.index))


    # Visit a parse tree produced by BESSERActionLanguageParser#function_call.
    def visitFunction_call(self, ctx:BESSERActionLanguageParser.Function_callContext):
        args = list()
        for arg in ctx.args:
            args.append(self.visit(arg))
        # TODO : Add method retrieval
        method = ctx.name.text
        return FunctionCall(self.visit(ctx.receiver), None, args)


    # Visit a parse tree produced by BESSERActionLanguageParser#atomic.
    def visitAtomic(self, ctx:BESSERActionLanguageParser.AtomicContext):
        if ctx.expr is not None:
            return self.visit(ctx.expr)
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#this.
    def visitThis(self, ctx:BESSERActionLanguageParser.ThisContext):
        return This()


    # Visit a parse tree produced by BESSERActionLanguageParser#new.
    def visitNew(self, ctx:BESSERActionLanguageParser.NewContext):
        args = list()
        for arg in ctx.args:
            args.append(self.visit(arg))
        return New(self.visit(ctx.clazz), args)


    # Visit a parse tree produced by BESSERActionLanguageParser#literal.
    def visitLiteral(self, ctx:BESSERActionLanguageParser.LiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#single_literal.
    def visitSingle_literal(self, ctx:BESSERActionLanguageParser.Single_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#int_literal.
    def visitInt_literal(self, ctx:BESSERActionLanguageParser.Int_literalContext):
        return IntLiteral(int(ctx.value.text))


    # Visit a parse tree produced by BESSERActionLanguageParser#string_literal.
    def visitString_literal(self, ctx:BESSERActionLanguageParser.String_literalContext):
        return StringLiteral(ctx.value.text)


    # Visit a parse tree produced by BESSERActionLanguageParser#bool_literal.
    def visitBool_literal(self, ctx:BESSERActionLanguageParser.Bool_literalContext):
        return BoolLiteral(ctx.value.text == "true")


    # Visit a parse tree produced by BESSERActionLanguageParser#real_literal.
    def visitReal_literal(self, ctx:BESSERActionLanguageParser.Real_literalContext):
        return RealLiteral(float(ctx.value.text))


    # Visit a parse tree produced by BESSERActionLanguageParser#null_literal.
    def visitNull_literal(self, ctx:BESSERActionLanguageParser.Null_literalContext):
        return NullLiteral


    # Visit a parse tree produced by BESSERActionLanguageParser#enum_literal.
    def visitEnum_literal(self, ctx:BESSERActionLanguageParser.Enum_literalContext):
        # TODO : check literal exist in enum
        lit = ctx.name.text
        return EnumLiteral(self.visit(ctx.enum), ctx.name.text)


    # Visit a parse tree produced by BESSERActionLanguageParser#sequence_literal.
    def visitSequence_literal(self, ctx:BESSERActionLanguageParser.Sequence_literalContext):
        values = list()
        for val in ctx.values:
            values.append(self.visit(val))
        return SequenceLiteral(self.visit(ctx.the_type), values)


    # Visit a parse tree produced by BESSERActionLanguageParser#range_literal.
    def visitRange_literal(self, ctx:BESSERActionLanguageParser.Range_literalContext):
        return RangeLiteral(self.visit(ctx.first), self.visit(ctx.last))


    # Visit a parse tree produced by BESSERActionLanguageParser#symbol.
    def visitSymbol(self, ctx:BESSERActionLanguageParser.SymbolContext):
        if ctx.name.text in self.__symbols:
            decl = self.__symbols[ctx.name.text]
            return Reference(decl)
        else:
            decl = ImplicitDecl(ctx.name.text, None, None)
            self.__symbols[ctx.name.text] = decl
            return decl



del BESSERActionLanguageParser