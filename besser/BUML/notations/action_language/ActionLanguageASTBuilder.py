# Generated from D:/Projects/BESSER/besser/BUML/notations/action_language/BESSERActionLanguage.g4 by ANTLR 4.13.2
from antlr4 import *
from win32comext.mapi.mapi import FORCE_SAVE

from ...metamodel.action_language.action_language import FunctionDefinition, Parameter, While, DoWhile, For, Iterator, \
    Statements, Block, Condition, AnyType, SequenceType, RealType, StringType, IntType, BoolType, Assignment, Ternary, \
    Or, And, Equal, LessEq, InstanceOf, Plus, Remain, Mult, Not, UnaryMinus, Cast, NullCoalessing, ArrayAccess, \
    FieldAccess, This, New, IntLiteral, StringLiteral, BoolLiteral, RealLiteral, NullLiteral, EnumLiteral, \
    SequenceLiteral, RangeLiteral, NameDecl, Reference, ImplicitDecl, Unequal, Greater, GreaterEq, Less, Minus, Div, \
    ObjectType, EnumType, MethodCall, StandardLibCall, Concatenation, Multiplicity, Return, ExplicitDecl, FunctionType, \
    ProcedureCall
from ...metamodel.structural import DomainModel, Class, Enumeration

if "." in __name__:
    from .BESSERActionLanguageParser import BESSERActionLanguageParser
else:
    from BESSERActionLanguageParser import BESSERActionLanguageParser

base_classes = {
    'int':          IntType,
    'float':        RealType,
    'str':          StringType,
    'bool':         BoolType,
    'time':         AnyType,
    'date':         AnyType,
    'datetime':     AnyType,
    'timedelta':    AnyType,
    'any':          AnyType
}

def functions_for_sequence_type(sequence_type: SequenceType):
    elem = sequence_type.elementsType
    elem_to_bool = FunctionType([elem],BoolType())
    elem_to_any = FunctionType([elem], AnyType())
    reduce = FunctionType([elem, elem], elem)
    sequence_functions = {
        'size':         FunctionType([],                IntType()),
        'is_empty':     FunctionType([],                BoolType()),
        'add':          FunctionType([elem], None),
        'remove':       FunctionType([elem], None),
        'contains':     FunctionType([elem],            BoolType()),
        'filter':       FunctionType([elem_to_bool],    sequence_type),
        'forall':       FunctionType([elem_to_bool],    BoolType()),
        'exists':       FunctionType([elem_to_bool],    BoolType()),
        'one':          FunctionType([elem_to_bool],    BoolType()),
        'is_unique':    FunctionType([elem_to_any],     BoolType()),
        'map':          FunctionType([elem_to_any],     BoolType()),
        'reduce':       FunctionType([reduce],          AnyType())
    }
    return sequence_functions


# This class defines a complete generic visitor for a parse tree produced by BESSERActionLanguageParser.

class BESSERActionLanguageVisitor(ParseTreeVisitor):

    def __init__(self, domain_model: DomainModel, context_class: Class):
        self.__symbols:dict[str, NameDecl] = {}
        self.__model = domain_model
        self.__method_class = context_class
        self.__current_type = ObjectType(context_class)
        self.__is_root = True


    # Visit a parse tree produced by BESSERActionLanguageParser#function_definition.
    def visitFunction_definition(self, ctx:BESSERActionLanguageParser.Function_definitionContext):
        is_root = self.__is_root
        self.__is_root = False
        parameters:list[Parameter] = []
        for param in ctx.params:
            parameters.append(self.visit(param))

        definition = FunctionDefinition(
            ctx.name.text,
            parameters,
            self.visit(ctx.return_type),
            self.visit(ctx.body)
        )

        self.__current_type = None

        if not is_root:
            self.__symbols[ctx.name.text] = definition

        return definition


    # Visit a parse tree produced by BESSERActionLanguageParser#parameter.
    def visitParameter(self, ctx:BESSERActionLanguageParser.ParameterContext):
        the_type = self.visit(ctx.declared_type)
        default = None
        if ctx.expr is not None:
            default = self.visit(ctx.expr)
        param =  Parameter(ctx.name.text, the_type, default)
        self.__symbols[ctx.name.text] = param
        self.__current_type = None
        return param


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
        var_name = self.visit(ctx.var_name)
        sequence = self.visit(ctx.sequence)
        if isinstance(self.__current_type, SequenceType):
            var_name.declared_type = self.__current_type.elementsType
            var_name.multiplicity = Multiplicity(True, False)
        self.__current_type = None
        return Iterator(var_name, sequence)


    # Visit a parse tree produced by BESSERActionLanguageParser#conditional_branch.
    def visitConditional_branch(self, ctx:BESSERActionLanguageParser.Conditional_branchContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#block.
    def visitBlock(self, ctx:BESSERActionLanguageParser.BlockContext):
        children: list[Statements] = list()
        for elem in ctx.stmts:
            children.append(self.visit(elem))
        self.__current_type = None
        return Block(children)


    # Visit a parse tree produced by BESSERActionLanguageParser#condition.
    def visitCondition(self, ctx:BESSERActionLanguageParser.ConditionContext):
        elze = None
        if ctx.elze is not None:
            elze = self.visit(ctx.elze)
        self.__current_type = None
        return Condition(self.visit(ctx.cond), self.visit(ctx.then), elze)


    # Visit a parse tree produced by BESSERActionLanguageParser#return.
    def visitReturn(self, ctx: BESSERActionLanguageParser.ReturnContext):
        expr = self.visit(ctx.expr)
        self.__current_type = None
        return Return(expr)


    # Visit a parse tree produced by BESSERActionLanguageParser#type.
    def visitType(self, ctx:BESSERActionLanguageParser.TypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#single_type.
    def visitSingle_type(self, ctx:BESSERActionLanguageParser.Single_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#sequence_type.
    def visitSequence_type(self, ctx: BESSERActionLanguageParser.Sequence_typeContext):
        return SequenceType(self.visit(ctx.the_type))

    # Visit a parse tree produced by BESSERActionLanguageParser#function_type.
    def visitFunction_type(self, ctx: BESSERActionLanguageParser.Function_typeContext):
        params_types = []
        for param in ctx.params_type:
            params_types.append(self.visit(param))
        return FunctionType(params_types, self.visit(ctx.return_type))


    # Visit a parse tree produced by BESSERActionLanguageParser#any_type.
    def visitAny_type(self, ctx:BESSERActionLanguageParser.Any_typeContext):
        return AnyType()


    # Visit a parse tree produced by BESSERActionLanguageParser#classifier_type.
    def visitClassifier_type(self, ctx:BESSERActionLanguageParser.Classifier_typeContext):
        model_type = self.__model.get_type_by_name(ctx.name.text)
        if isinstance(model_type, Class):
            return ObjectType(model_type)
        elif isinstance(model_type, Enumeration):
            return EnumType(model_type)
        return None


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


    # Visit a parse tree produced by BESSERActionLanguageParser#explicit_declaration.
    def visitExplicit_declaration(self, ctx: BESSERActionLanguageParser.Explicit_declarationContext):
        the_type = self.visit(ctx.declared_type)
        mult = Multiplicity(True, isinstance(the_type, SequenceType))
        decl = ExplicitDecl(ctx.name.text, the_type, mult)
        self.__symbols[ctx.name.text] = decl
        self.__current_type = None
        return decl


    # Visit a parse tree produced by BESSERActionLanguageParser#assignment.
    def visitAssignment(self, ctx:BESSERActionLanguageParser.AssignmentContext):
        if ctx.target is None or ctx.assignee is None:
            return  self.visitChildren(ctx)
        target = self.visit(ctx.target)
        assignee = self.visit(ctx.assignee)
        if isinstance(target, NameDecl):
            if target.declared_type is None:
                target.declared_type = self.__current_type
                target.multiplicity = Multiplicity(True, isinstance(self.__current_type, SequenceType))
        return Assignment(target, assignee)


    # Visit a parse tree produced by BESSERActionLanguageParser#ternary.
    def visitTernary(self, ctx:BESSERActionLanguageParser.TernaryContext):
        if ctx.expr is None:
            return self.visitChildren(ctx)
        expr = self.visit(ctx.expr)
        elze = self.visit(ctx.elze)
        then = self.visit(ctx.then)
        return Ternary(expr, then, elze)


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
        left = self.visit(ctx.left)
        right = self.visit(ctx.right)
        self.__current_type = BoolType()
        match ctx.op.text:
            case "==":
                return Equal(left, right)
            case "!=":
                return Unequal(left, right)


    # Visit a parse tree produced by BESSERActionLanguageParser#comparison.
    def visitComparison(self, ctx:BESSERActionLanguageParser.ComparisonContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        left = self.visit(ctx.left)
        right = self.visit(ctx.right)
        self.__current_type = BoolType()
        match ctx.op.text:
            case "<":
                return Less(left, right)
            case "<=":
                return LessEq(left, right)
            case ">":
                return Greater(left, right)
            case ">=":
                return GreaterEq(left, right)


    # Visit a parse tree produced by BESSERActionLanguageParser#instanceof.
    def visitInstanceof(self, ctx:BESSERActionLanguageParser.InstanceofContext):
        if ctx.instance is None or ctx.the_type is None:
            return self.visitChildren(ctx)
        self.__current_type = BoolType()
        return InstanceOf(self.visit(ctx.instance), self.visit(ctx.the_type))


    # Visit a parse tree produced by BESSERActionLanguageParser#arithmetic.
    def visitArithmetic(self, ctx:BESSERActionLanguageParser.ArithmeticContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#plus_minus.
    def visitPlus_minus(self, ctx:BESSERActionLanguageParser.Plus_minusContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        left = self.visit(ctx.left)
        left_type = self.__current_type
        right = self.visit(ctx.right)
        right_type = self.__current_type
        match ctx.op.text:
            case '+':
                if isinstance(left_type, StringType) or isinstance(right_type, StringType):
                    self.__current_type = StringType
                    return Concatenation(left, right)
                elif isinstance(left_type, RealType):
                    self.__current_type = RealType
                return Plus(left, right)

            case '-':
                if isinstance(left_type, StringType) or isinstance(right_type, StringType):
                    return None
                elif isinstance(left_type, RealType):
                    self.__current_type = RealType
                return Minus(left, right)


    # Visit a parse tree produced by BESSERActionLanguageParser#mult_div.
    def visitMult_div(self, ctx:BESSERActionLanguageParser.Mult_divContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        left = self.visit(ctx.left)
        left_type = self.__current_type
        right = self.visit(ctx.right)
        if isinstance(left_type, RealType):
            self.__current_type = RealType
        match ctx.op.text:
            case '*':
                return Mult(left, right)
            case '/':
                return Div(left, right)


    # Visit a parse tree produced by BESSERActionLanguageParser#remain.
    def visitRemain(self, ctx:BESSERActionLanguageParser.RemainContext):
        if ctx.left is None or ctx.right is None:
            return self.visitChildren(ctx)
        left = self.visit(ctx.left)
        right = self.visit(ctx.right)
        self.__current_type = IntType()
        return Remain(left, right)


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
        expr = self.visit(ctx.expr)
        as_type = self.visit(ctx.as_)
        self.__current_type = as_type
        return Cast(expr, as_type)


    # Visit a parse tree produced by BESSERActionLanguageParser#null_coalessing.
    def visitNull_coalessing(self, ctx:BESSERActionLanguageParser.Null_coalessingContext):
        elze = self.visit(ctx.elze)
        nullable = self.visit(ctx.nullable)
        return NullCoalessing(nullable, elze)


    # Visit a parse tree produced by BESSERActionLanguageParser#ArrayAccess.
    def visitArrayAccess(self, ctx:BESSERActionLanguageParser.ArrayAccessContext):
        index = self.visit(ctx.index)
        receiver = self.visit(ctx.receiver)
        if isinstance(self.__current_type, SequenceType):
            self.__current_type = self.__current_type.elementsType
        return ArrayAccess(receiver, index)


    # Visit a parse tree produced by BESSERActionLanguageParser#FunctionCall.
    def visitFunctionCall(self, ctx:BESSERActionLanguageParser.FunctionCallContext):
        args = list()
        for arg in ctx.args:
            args.append(self.visit(arg))

        receiver = self.visit(ctx.receiver)
        if isinstance(self.__current_type, SequenceType):
            fns = functions_for_sequence_type(self.__current_type)
            fn_type = fns.get(ctx.name.text)
            return StandardLibCall(receiver, ctx.name.text, fn_type, args)
        elif isinstance(self.__current_type, Class):
            method = ctx.name.text
            methods = {m for m in self.__current_type.methods if m.name == method}
            if len(methods) > 0:
                method_obj = next(iter(methods))
                return_type = method_obj.type
                the_type = None
                if isinstance(return_type, Class):
                    the_type = ObjectType(return_type)
                elif isinstance(return_type, Enumeration):
                    the_type = EnumType(return_type)
                elif return_type.name in base_classes:
                    the_type = base_classes[return_type.name]
                self.__current_type = the_type
                return MethodCall(receiver, method_obj, args)
            return MethodCall(receiver, None, args)


    # Visit a parse tree produced by BESSERActionLanguageParser#FieldAccess.
    def visitFieldAccess(self, ctx:BESSERActionLanguageParser.FieldAccessContext):
        receiver = self.visit(ctx.receiver)
        prop = None
        if isinstance(self.__current_type, ObjectType):
            clazz = self.__current_type.clazz
            field = ctx.field.text
            for attr in clazz.all_attributes():
                if attr.name == field:
                    prop = attr
            for end in clazz.all_association_ends():
                if end.name == field:
                    prop = end
        if prop is not None:
            prop_type = prop.type
            the_type = None
            if isinstance(prop_type, Class):
                the_type = ObjectType(prop_type)
            elif isinstance(prop_type, Enumeration):
                the_type = EnumType(prop_type)
            elif prop_type.name in base_classes:
                the_type = base_classes[prop_type.name]

            if prop.multiplicity.max > 1:
                the_type = SequenceType(the_type)
            self.__current_type = the_type

        return FieldAccess(receiver, prop)


    # Visit a parse tree produced by BESSERActionLanguageParser#Atom.
    def visitAtom(self, ctx:BESSERActionLanguageParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#field_access.
    def visitField_access(self, ctx:BESSERActionLanguageParser.Field_accessContext):
        receiver = self.visit(ctx.receiver)
        prop = None
        if isinstance(self.__current_type, ObjectType):
            clazz = self.__current_type.clazz
            field = ctx.field.text
            for attr in clazz.all_attributes():
                if attr.name == field:
                    prop = attr
            for end in clazz.all_association_ends():
                if end.name == field:
                    prop = end
        if prop is not None:
            prop_type = prop.type
            the_type = None
            if isinstance(prop_type, Class):
                the_type = ObjectType(prop_type)
            elif isinstance(prop_type, Enumeration):
                the_type = EnumType(prop_type)
            elif prop_type.name in base_classes:
                the_type = base_classes[prop_type.name]

            if prop.multiplicity.max > 1:
                the_type = SequenceType(the_type)
            self.__current_type = the_type

        return FieldAccess(receiver, prop)


    # Visit a parse tree produced by BESSERActionLanguageParser#array_access.
    def visitArray_access(self, ctx:BESSERActionLanguageParser.Array_accessContext):
        index = self.visit(ctx.index)
        receiver = self.visit(ctx.receiver)
        if isinstance(self.__current_type, SequenceType):
            self.__current_type = self.__current_type.elementsType
        return ArrayAccess(receiver, index)


    # Visit a parse tree produced by BESSERActionLanguageParser#function_call.
    def visitFunction_call(self, ctx:BESSERActionLanguageParser.Function_callContext):
        args = list()
        for arg in ctx.args:
            args.append(self.visit(arg))

        receiver = self.visit(ctx.receiver)
        if isinstance(self.__current_type, SequenceType):
            fns = functions_for_sequence_type(self.__current_type)
            fn_type = fns.get(ctx.name.text)
            return StandardLibCall(receiver, ctx.name.text, fn_type, args)
        elif isinstance(self.__current_type, Class):
            method = ctx.name.text
            methods = {m for m in self.__current_type.methods if m.name == method}
            if len(methods) > 0:
                method_obj = next(iter(methods))
                return_type = method_obj.type
                the_type = None
                if isinstance(return_type, Class):
                    the_type = ObjectType(return_type)
                elif isinstance(return_type, Enumeration):
                    the_type = EnumType(return_type)
                elif return_type.name in base_classes:
                    the_type = base_classes[return_type.name]
                self.__current_type = the_type
                return MethodCall(receiver, method_obj, args)
            return MethodCall(receiver, None, args)


    # Visit a parse tree produced by BESSERActionLanguageParser#atomic.
    def visitAtomic(self, ctx:BESSERActionLanguageParser.AtomicContext):
        if ctx.expr is not None:
            return self.visit(ctx.expr)
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#procedure_call.
    def visitProcedure_call(self, ctx: BESSERActionLanguageParser.Procedure_callContext):
        args = list()
        for arg in ctx.args:
            args.append(self.visit(arg))

        decl = self.__symbols.get(ctx.name.text)
        if isinstance(decl, FunctionDefinition):
            self.__current_type = decl.return_type
            return ProcedureCall(decl, args)


    # Visit a parse tree produced by BESSERActionLanguageParser#this.
    def visitThis(self, ctx:BESSERActionLanguageParser.ThisContext):
        self.__current_type = ObjectType(self.__method_class)
        return This()


    # Visit a parse tree produced by BESSERActionLanguageParser#new.
    def visitNew(self, ctx:BESSERActionLanguageParser.NewContext):
        args = list()
        for arg in ctx.args:
            args.append(self.visit(arg))
        obj_type = self.visit(ctx.clazz)
        self.__current_type = obj_type
        return New(obj_type, args)


    # Visit a parse tree produced by BESSERActionLanguageParser#literal.
    def visitLiteral(self, ctx:BESSERActionLanguageParser.LiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#single_literal.
    def visitSingle_literal(self, ctx:BESSERActionLanguageParser.Single_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by BESSERActionLanguageParser#int_literal.
    def visitInt_literal(self, ctx:BESSERActionLanguageParser.Int_literalContext):
        self.__current_type = IntType()
        return IntLiteral(int(ctx.value.text))


    # Visit a parse tree produced by BESSERActionLanguageParser#string_literal.
    def visitString_literal(self, ctx:BESSERActionLanguageParser.String_literalContext):
        self.__current_type = StringType()
        return StringLiteral(ctx.value.text)


    # Visit a parse tree produced by BESSERActionLanguageParser#bool_literal.
    def visitBool_literal(self, ctx:BESSERActionLanguageParser.Bool_literalContext):
        self.__current_type = BoolType()
        return BoolLiteral(ctx.value.text == "true")


    # Visit a parse tree produced by BESSERActionLanguageParser#real_literal.
    def visitReal_literal(self, ctx:BESSERActionLanguageParser.Real_literalContext):
        self.__current_type = RealType()
        return RealLiteral(float(ctx.value.text))


    # Visit a parse tree produced by BESSERActionLanguageParser#null_literal.
    def visitNull_literal(self, ctx:BESSERActionLanguageParser.Null_literalContext):
        self.__current_type = None
        return NullLiteral()


    # Visit a parse tree produced by BESSERActionLanguageParser#enum_literal.
    def visitEnum_literal(self, ctx:BESSERActionLanguageParser.Enum_literalContext):
        enum: EnumType = self.visit(ctx.enum)
        valid = False
        for literal in enum.enum.literals:
            if literal.name == ctx.name.text:
                valid = True
        # TODO : raise typechecking error if not valid
        self.__current_type = enum
        return EnumLiteral(enum, ctx.name.text)


    # Visit a parse tree produced by BESSERActionLanguageParser#sequence_literal.
    def visitSequence_literal(self, ctx:BESSERActionLanguageParser.Sequence_literalContext):
        values = list()
        for val in ctx.values:
            values.append(self.visit(val))
        value_type:SequenceType = self.visit(ctx.the_type)
        self.__current_type = value_type
        return SequenceLiteral(value_type.elementsType, values)


    # Visit a parse tree produced by BESSERActionLanguageParser#range_literal.
    def visitRange_literal(self, ctx:BESSERActionLanguageParser.Range_literalContext):
        self.__current_type = SequenceType(IntType())
        return RangeLiteral(self.visit(ctx.first), self.visit(ctx.last))


    # Visit a parse tree produced by BESSERActionLanguageParser#symbol.
    def visitSymbol(self, ctx:BESSERActionLanguageParser.SymbolContext):
        if ctx.name.text in self.__symbols:
            decl = self.__symbols[ctx.name.text]
            self.__current_type = decl.declared_type
            return Reference(decl)
        else:
            decl = ImplicitDecl(ctx.name.text, None, None)
            self.__symbols[ctx.name.text] = decl
            return decl



del BESSERActionLanguageParser