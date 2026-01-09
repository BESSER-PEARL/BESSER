from besser.BUML.metamodel.action_language.action_language import Condition, Block, ConditionalBranch, DoWhile, While, \
    CondLoop, Iterator, RangeLiteral, SequenceLiteral, EnumLiteral, NullLiteral, RealLiteral, BoolLiteral, \
    StringLiteral, IntLiteral, Literal, Reference, InstanceOf, Concatenation, Minus, Plus, Mult, Remain, Div, \
    BinaryArithmetic, UnaryMinus, Arithmetic, Ternary, NullCoalessing, Cast, ArrayAccess, FieldAccess, This, \
    ProcedureCall, StandardLibCall, MethodCall, New, Call, Not, GreaterEq, Unequal, Less, Greater, Equal, Or, And, \
    LessEq, BinaryBoolean, Boolean, Expression, Return, For, ImplicitDecl, ExplicitDecl, NameDecl, Statement, \
    Assignment, BoolType, EnumType, IntType, StringType, RealType, Type, FunctionType, SequenceType, ObjectType, \
    Multiplicity, AnyType, OptionalType, FunctionDefinition, Parameter, AssignTarget, NoType, Nothing, TypeUnion
from besser.BUML.metamodel.action_language.visitors import BALVisitor
from besser.BUML.metamodel.structural import Class, Enumeration
from besser.BUML.notations.action_language.helpers import UnknownClassifier, base_classes, get_typename
from besser.generators.action_language.PythonGenerator import BALPythonGenerator


class TypeCheckFeedback:
    def __init__(self, message: str, code_excerpt: str = None):
        self.__message = message
        self.__code_excerpt = code_excerpt

    def message(self):
        if self.__code_excerpt is not None:
            return f"{self.__message} in :\n{self.__code_excerpt}"
        return self.__message

class Warning(TypeCheckFeedback):
    pass

class Error(TypeCheckFeedback):
    pass


class TypeCheckingContext:
    def __init__(self):
        self.__errors: list[Error] = list()
        self.__warnings: list[Warning] = list()

    def add_error(self, error: Error) -> None:
        self.__errors.append(error)

    def errors(self) -> list[Error]:
        return self.__errors

    def add_warning(self, warning: Warning) -> None:
        self.__warnings.append(warning)

    def warnings(self) -> list[Warning]:
        return self.__warnings



class BALTypeChecker(BALVisitor[TypeCheckingContext, Type]):

    def __init__(self):
        self.__to_text_visitor = BALPythonGenerator()

    def check(self, node: FunctionDefinition):
        context = TypeCheckingContext()
        node.accept(self, context)
        return context

    def visit_AssignTarget(self, node: AssignTarget, context: TypeCheckingContext) -> Type:
        pass

    def visit_Parameter(self, node: Parameter, context: TypeCheckingContext) -> Type:
        if node.default is not None:
            default_type = node.default.accept(self, context)
            if not default_type <= node.declared_type:
                context.add_error(Error(f"Parameter {node.name} expect values of type {get_typename(node.declared_type)}, default returns {get_typename(default_type)}"))
        return node.declared_type

    def visit_FunctionDefinition(self, node: FunctionDefinition, context: TypeCheckingContext) -> Type:
        params_types = []
        for param in node.parameters:
            params_types.append(param.accept(self, context))

        effective_return_type = node.body.accept(self, context).supertype()
        if effective_return_type == NoType():
            effective_return_type = Nothing()

        if not effective_return_type <= node.return_type:
            context.add_error(Error(f"A return statement in function {node.name} is returning an invalid type, returns {get_typename(effective_return_type)} rather than {get_typename(node.return_type)}"))

        return NoType()

    def visit_AnyType(self, node: AnyType, context: TypeCheckingContext) -> Type:
        return node

    def visit_Multiplicity(self, node: Multiplicity, context: TypeCheckingContext) -> Type:
        pass

    def visit_ObjectType(self, node: ObjectType, context: TypeCheckingContext) -> Type:
        return node

    def visit_SequenceType(self, node: SequenceType, context: TypeCheckingContext) -> Type:
        return node

    def visit_OptionalType(self, node: OptionalType, context: TypeCheckingContext) -> Type:
        return node

    def visit_FunctionType(self, node: FunctionType, context: TypeCheckingContext) -> Type:
        return node

    def visit_Type(self, node: Type, context: TypeCheckingContext) -> Type:
        return node

    def visit_RealType(self, node: RealType, context: TypeCheckingContext) -> Type:
        return node

    def visit_StringType(self, node: StringType, context: TypeCheckingContext) -> Type:
        return node

    def visit_IntType(self, node: IntType, context: TypeCheckingContext) -> Type:
        return node

    def visit_EnumType(self, node: EnumType, context: TypeCheckingContext) -> Type:
        return node

    def visit_BoolType(self, node: BoolType, context: TypeCheckingContext) -> Type:
        return node

    def visit_Nothing(self, node: Nothing, context: TypeCheckingContext) -> Type:
        return node

    def visit_Assignment(self, node: Assignment, context: TypeCheckingContext) -> Type:
        target = node.target.accept(self, context)
        assignee = node.assignee.accept(self, context)
        if not assignee <= target:
            # repr = node.accept(self.__to_text_visitor, None)
            context.add_error(Error(f"A value of type {get_typename(assignee)} cannot be assigned to a variable of type {get_typename(target)}"))

        if isinstance(assignee, TypeUnion): # if supertype is union then type is T | Nothing
            assign_target = node.target
            if isinstance(assign_target, NameDecl):
                assign_target.multiplicity.nullable = True

        return NoType()

    def visit_Statement(self, node: Statement, context: TypeCheckingContext) -> Type:
        pass

    def visit_NameDecl(self, node: NameDecl, context: TypeCheckingContext) -> Type:
        pass

    def visit_ExplicitDecl(self, node: ExplicitDecl, context: TypeCheckingContext) -> Type:
        return node.declared_type

    def visit_ImplicitDecl(self, node: ImplicitDecl, context: TypeCheckingContext) -> Type:
        return node.declared_type

    def visit_For(self, node: For, context: TypeCheckingContext) -> Type:
        for iterator in node.iterators:
            iterator.accept(self, context)

        return node.body.accept(self, context)

    def visit_Return(self, node: Return, context: TypeCheckingContext) -> Type:
        expr = node.expr.accept(self, context)
        if expr == NoType():
            return Nothing()
        return expr

    def visit_Expression(self, node: Expression, context: TypeCheckingContext) -> Type:
        pass

    def visit_Boolean(self, node: Boolean, context: TypeCheckingContext) -> Type:
        pass

    def visit_BinaryBoolean(self, node: BinaryBoolean, context: TypeCheckingContext) -> Type:
        pass

    def visit_LessEq(self, node: LessEq, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a less equal operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a less equal operation is not a number"))

        return BoolType()

    def visit_And(self, node: And, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != BoolType():
            context.add_error(Error("left-hand side of a and operation is not a boolean"))
        if right != BoolType():
            context.add_error(Error("right-hand side of a and operation is not a boolean"))

        return BoolType()

    def visit_Or(self, node: Or, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != BoolType():
            context.add_error(Error("left-hand side of a or operation is not a boolean"))
        if right != BoolType():
            context.add_error(Error("right-hand side of a or operation is not a boolean"))

        return BoolType()

    def visit_Equal(self, node: Equal, context: TypeCheckingContext) -> Type:
        return BoolType()

    def visit_Greater(self, node: Greater, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a greater operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a greater operation is not a number"))

        return BoolType()

    def visit_Less(self, node: Less, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a less operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a less operation is not a number"))

        return BoolType()

    def visit_Unequal(self, node: Unequal, context: TypeCheckingContext) -> Type:
        return BoolType()

    def visit_GreaterEq(self, node: GreaterEq, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a greater equal operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a greater equal operation is not a number"))

        return BoolType()

    def visit_Not(self, node: Not, context: TypeCheckingContext) -> Type:
        expr = node.expr.accept(self, context)
        if expr != BoolType():
            context.add_error(Error("Sub-expression of a not operator is not a boolean"))
        return BoolType()

    def visit_Call(self, node: Call, context: TypeCheckingContext) -> Type:
        pass

    def visit_New(self, node: New, context: TypeCheckingContext) -> Type:
        args = []
        for arg in node.arguments:
            args.append(arg.accept(self, context)[0])
        return [f"{node.clazz.clazz.name}({', '.join(args)})"]

    def visit_MethodCall(self, node: MethodCall, context: TypeCheckingContext) -> Type:
        receiver = node.receiver.accept(self, context)
        receiver_typename = get_typename(receiver)
        if node.method is None:
            context.add_error(Error(f"Method does not exist for object of type {receiver_typename}"))
            return NoType()

        args_types = []
        for arg in node.arguments:
            args_types.append(arg.accept(self, context))

        struct_to_action_type = lambda x: ObjectType(x.type) if isinstance(x.type, Class) else base_classes[x.type.name]

        parameters = list(node.method.parameters)
        param_types = list(map(struct_to_action_type, parameters))
        shortest, longest = (args_types, param_types) \
            if len(args_types) <= len(param_types) \
            else (param_types, args_types)

        for i in range(0, len(shortest)):
            if not args_types[i] <= param_types[i]:  # if ~ P2 <= P1
                context.add_error(Error(
                    f"Argument for parameter {parameters[i].name} is of type {get_typename(args_types[i])}, {get_typename(param_types[i])} expected"))

        for i in range(len(shortest), len(longest)):
            if not isinstance(longest[i], OptionalType):
                context.add_error(Error(
                    f"Parameter {parameters[i].name} of method {node.method.name} not supplied"))

        return struct_to_action_type(node.method.type)

    def visit_StandardLibCall(self, node: StandardLibCall, context: TypeCheckingContext) -> Type:
        receiver = node.receiver.accept(self, context)
        receiver_typename = get_typename(receiver)
        if node.function_type is None or node.function_type.params_type is None:
            context.add_error(Error(f"No method named {node.function} for instances of type {receiver_typename}"))

        args_types = []
        for arg in node.arguments:
            args_types.append(arg.accept(self, context))

        param_types = node.function_type.params_type
        shortest, longest = (args_types, param_types) \
            if len(args_types) <= len(param_types) \
            else (param_types, args_types)

        for i in range(0, len(shortest)):
            if not args_types[i] <= param_types[i]:  # if ~ P2 <= P1
                context.add_error(Error(
                    f"Argument for parameter number {i+1} is of type {get_typename(args_types[i])}, {get_typename(param_types[i])} expected"))

        for i in range(len(shortest), len(longest)):
            if not isinstance(longest[i], OptionalType):
                context.add_error(Error(
                    f"Parameter number {i+1} of method {node.function} not supplied"))

        return node.function_type.return_type

    def visit_ProcedureCall(self, node: ProcedureCall, context: TypeCheckingContext) -> Type:
        if node.function is None:
            context.add_error(Error("Function call on a non function object"))
            return NoType()

        if node.function.parameters is None:
            context.add_warning(Warning("Unknown function called, this may result in errors in generated code"))


        args_types = []
        for arg in node.arguments:
            args_types.append(arg.accept(self, context))

        param_types = list(map(lambda e: e.declared_type , node.function.parameters))
        shortest, longest = (args_types, param_types) \
            if len(args_types) <= len(param_types) \
            else (param_types, args_types)

        for i in range(0, len(shortest)):
            if not args_types[i] <= param_types[i]:  # if ~ P2 <= P1
                context.add_error(Error(f"Argument for parameter {node.function.parameters[i].name} of type {get_typename(args_types[i])}, {get_typename(param_types[i])} expected"))

        for i in range(len(shortest), len(longest)):
            if not isinstance(longest[i], OptionalType):
                context.add_error(Error(f"Parameter {node.function.parameters[i].name} of function {node.function.name} not supplied"))

        return node.function.return_type

    def visit_This(self, node: This, context: TypeCheckingContext) -> Type:
        return ObjectType(node.clazz)

    def visit_FieldAccess(self, node: FieldAccess, context: TypeCheckingContext) -> Type:
        receiver = node.receiver.accept(self, context)
        if not isinstance(receiver, ObjectType):
            context.add_error(Error("Field access on a non-object type"))
            return NoType()
        if node.field is None:
            context.add_error(Error(f"Invalid field access on object of type \"{receiver.clazz.name}\""))
            return NoType()

        if node.field.type.name in base_classes:
            return base_classes[node.field.type.name]

        if isinstance(node.field.type, Class):
            return ObjectType(node.field.type)

        if isinstance(node.field.type, Enumeration):
            return EnumType(node.field.type)

        return NoType()

    def visit_ArrayAccess(self, node: ArrayAccess, context: TypeCheckingContext) -> Type:
        receiver = node.receiver.accept(self, context)
        index = node.index.accept(self, context)
        if index != IntType():
            context.add_error(Error("Invalid array access, index is not an integer"))
        if not isinstance(receiver, SequenceType):
            context.add_error(Error("Invalid array access, the receiver is not a sequence"))
            return NoType()
        return receiver.elementsType

    def visit_Cast(self, node: Cast, context: TypeCheckingContext) -> Type:
        return node.as_type

    def visit_NullCoalessing(self, node: NullCoalessing, context: TypeCheckingContext) -> Type:
        elze = node.elze.accept(self, context)
        nullable = node.nullable.accept(self, context)
        nullable = nullable.supertype()
        if isinstance(nullable, TypeUnion):
            if nullable.a == Nothing():
                nullable = nullable.b
            else:
                nullable = nullable.a
        if nullable == Nothing():
            return elze

        return nullable | elze

    def visit_Ternary(self, node: Ternary, context: TypeCheckingContext) -> Type:
        expr = node.expr.accept(self, context)
        if expr != BoolType():
            context.add_error(Error("Condition of a ternary expression is not a boolean"))
        then = node.then.accept(self, context)
        elze = node.elze.accept(self, context)
        return then | elze

    def visit_Arithmetic(self, node: Arithmetic, context: TypeCheckingContext) -> Type:
        pass

    def visit_UnaryMinus(self, node: UnaryMinus, context: TypeCheckingContext) -> Type:
        expr = node.expr.accept(self, context)
        if expr != IntType() and expr != RealType():
            context.add_error(Error("Sub-expression of a unary minus is not a number"))

        return expr

    def visit_BinaryArithmetic(self, node: BinaryArithmetic, context: TypeCheckingContext) -> Type:
        pass

    def visit_Div(self, node: Div, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a division operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a division operation is not a number"))

        return RealType() if left == RealType() or right == RealType() else IntType()

    def visit_Remain(self, node: Remain, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a reminder operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a reminder operation is not a number"))

        return IntType()

    def visit_Mult(self, node: Mult, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a multiply operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a multiply operation is not a number"))

        return RealType() if left == RealType() or right == RealType() else IntType()

    def visit_Plus(self, node: Plus, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a plus operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a plus operation is not a number"))

        return RealType() if left == RealType() or right == RealType() else IntType()

    def visit_Minus(self, node: Minus, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != IntType() and left != RealType():
            context.add_error(Error("left-hand side of a minus operation is not a number"))
        if right != IntType() and right != RealType():
            context.add_error(Error("right-hand side of a minus operation is not a number"))

        return RealType() if left == RealType() or right == RealType() else IntType()

    def visit_Concatenation(self, node: Concatenation, context: TypeCheckingContext) -> Type:
        left = node.left.accept(self, context)
        right = node.right.accept(self, context)
        if left != StringType() and right != StringType():
            context.add_error(Error("Unexpected type error, concatenation of two non-string type"))
        return StringType()

    def visit_InstanceOf(self, node: InstanceOf, context: TypeCheckingContext) -> Type:
        instance = node.instance.accept(self, context)
        the_type = node.type.accept(self, context)
        return BoolType()

    def visit_Reference(self, node: Reference, context: TypeCheckingContext) -> Type:
        if node.definition is None:
            context.add_error(Error(f"Undefined symbol \"{node.symbol}\""))
            return NoType()
        return node.definition.declared_type

    def visit_Literal(self, node: Literal, context: TypeCheckingContext) -> Type:
        pass

    def visit_IntLiteral(self, node: IntLiteral, context: TypeCheckingContext) -> Type:
        return IntType()

    def visit_StringLiteral(self, node: StringLiteral, context: TypeCheckingContext) -> Type:
        return StringType()

    def visit_BoolLiteral(self, node: BoolLiteral, context: TypeCheckingContext) -> Type:
        return BoolType()

    def visit_RealLiteral(self, node: RealLiteral, context: TypeCheckingContext) -> Type:
        return RealType()

    def visit_NullLiteral(self, node: NullLiteral, context: TypeCheckingContext) -> Type:
        return NoType()

    def visit_EnumLiteral(self, node: EnumLiteral, context: TypeCheckingContext) -> Type:
        enum_type = node.enumeration
        if isinstance(enum_type, UnknownClassifier):
            context.add_error(Error(f"Unknown enumeration named \"{enum_type.name()}\""))
        elif node.name is None:
            context.add_error(Error(f"Unknown literal of enum \"{enum_type.enum.name}\""))
        return enum_type

    def visit_SequenceLiteral(self, node: SequenceLiteral, context: TypeCheckingContext) -> Type:
        value_type = node.value_type
        for val in node.values:
            val_type = val.accept(self, context)
            if val_type != value_type:
                context.add_error(Error(f"Value of type {get_typename(val_type)} in a sequence literal of type {get_typename(value_type)}"))
        return SequenceType(value_type)

    def visit_RangeLiteral(self, node: RangeLiteral, context: TypeCheckingContext) -> Type:
        first_type = node.first.accept(self, context)
        if first_type != IntType():
            context.add_error(Error("Range lower bound is not an integer"))
        last_type = node.last.accept(self, context)
        if last_type != IntType():
            context.add_error(Error("Range upper bound is not an integer"))
        return SequenceType(IntType())

    def visit_Iterator(self, node: Iterator, context: TypeCheckingContext) -> Type:
        sequence_type = node.sequence.accept(self, context)
        if isinstance(sequence_type, SequenceType):
            context.add_error(Error("Iterator over a non-sequence type"))
        return NoType()

    def visit_CondLoop(self, node: CondLoop, context: TypeCheckingContext) -> Type:
        pass

    def visit_While(self, node: While, context: TypeCheckingContext) -> Type:
        cond_type = node.condition.accept(self, context)
        if cond_type != BoolType():
            context.add_error(Error("While condition is not a boolean"))
        return NoType() | node.body.accept(self, context)

    def visit_DoWhile(self, node: DoWhile, context: TypeCheckingContext) -> Type:
        cond_type = node.condition.accept(self, context)
        if cond_type != BoolType():
            context.add_error(Error("Do-While condition is not a boolean"))
        return NoType() | node.body.accept(self, context)

    def visit_ConditionalBranch(self, node: ConditionalBranch, context: TypeCheckingContext) -> Type:
        pass

    def visit_Block(self, node: Block, context: TypeCheckingContext) -> Type:
        type = NoType()
        for statement in node.statements:
            t = statement.accept(self, context)
            if not isinstance(statement, Expression):
                type = t | type

        return type

    def visit_Condition(self, node: Condition, context: TypeCheckingContext) -> Type:
        cond_type = node.condition.accept(self, context)
        if cond_type != BoolType():
            context.add_error(Error("If condition is not a boolean"))
        the_type = node.then.accept(self, context)
        if node.elze is not None:
            the_type = node.elze.accept(self, context) | the_type
        else:
            the_type = NoType() | the_type
        return the_type

