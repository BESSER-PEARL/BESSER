from besser.BUML.metamodel.action_language.action_language import Condition, Block, ConditionalBranch, DoWhile, While, \
    CondLoop, Iterator, RangeLiteral, SequenceLiteral, EnumLiteral, NullLiteral, RealLiteral, BoolLiteral, \
    StringLiteral, IntLiteral, Literal, Reference, InstanceOf, Concatenation, Minus, Plus, Mult, Remain, Div, \
    BinaryArithmetic, UnaryMinus, Arithmetic, Ternary, NullCoalessing, Cast, ArrayAccess, FieldAccess, This, \
    ProcedureCall, StandardLibCall, MethodCall, New, Call, Not, GreaterEq, Unequal, Less, Greater, Equal, Or, And, \
    LessEq, BinaryBoolean, Boolean, Expression, Return, For, ImplicitDecl, ExplicitDecl, NameDecl, Statement, \
    Assignment, BoolType, EnumType, IntType, StringType, RealType, Type, FunctionType, SequenceType, ObjectType, \
    Multiplicity, AnyType, OptionalType, FunctionDefinition, Parameter, AssignTarget, Nothing
from besser.BUML.metamodel.action_language.visitors import BALVisitor
from besser.BUML.metamodel.structural import IntegerType, FloatType, BooleanType, TimeType, DateType, DateTimeType, \
    TimeDeltaType, data_types
from besser.BUML.notations.action_language.ActionLanguageTypeChecker import BALTypeChecker, TypeCheckingContext


def bal_to_rest(method: FunctionDefinition, class_name:str) -> str:
    generator = BALRESTGenerator(class_name)
    return generator.generate(method)

class RESTGenerationContext:
    pass

def indent(lines):
    out = []
    for line in lines:
        out.append('    '+line)
    return out

class BALRESTGenerator(BALVisitor[RESTGenerationContext, list[str]]):

    def __init__(self, class_name:str):
        self.class_name = class_name
        self.root_function = True

    def generate(self, node: FunctionDefinition):
        lines = node.accept(self, None)
        return "\n".join(lines)

    def visit_AssignTarget(self, node: AssignTarget, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_Parameter(self, node: Parameter, context: RESTGenerationContext) -> list[str]:
        if node.declared_type is None:
            return [node.name]
        param_type = node.declared_type.accept(self, context)[0]
        expr = ""
        if node.default is not None:
            expr = node.default.accept(self, context)[0]
            expr = " = " + expr
        return [f"{node.name}: {param_type}{expr}"]

    def visit_FunctionDefinition(self, node: FunctionDefinition, context: RESTGenerationContext) -> list[str]:
        lines = []

        params = []
        for param in node.parameters:
            params.extend(param.accept(self, context))

        if self.root_function:
            lines.append(f"async def {node.name}(self, {', '.join(params)}) -> {node.return_type.accept(self, context)[0]} :")
            self.root_function = False
        else:
            lines.append(f"async def {node.name}({', '.join(params)}) -> {node.return_type.accept(self, context)[0]} :")

        body = node.body.accept(self, context)
        lines.extend(indent(body))

        return lines


    def visit_AnyType(self, node: AnyType, context: RESTGenerationContext) -> list[str]:
        return ["Any"]

    def visit_Multiplicity(self, node: Multiplicity, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_ObjectType(self, node: ObjectType, context: RESTGenerationContext) -> list[str]:
        return [node.clazz.name]

    def visit_SequenceType(self, node: SequenceType, context: RESTGenerationContext) -> list[str]:
        return [f"list[{node.elementsType.accept(self, context)[0]}]"]

    def visit_OptionalType(self, node: OptionalType, context: RESTGenerationContext) -> list[str]:
        return node.type.accept(self, context)

    def visit_FunctionType(self, node: FunctionType, context: RESTGenerationContext) -> list[str]:
        return ["Any"]

    def visit_Type(self, node: Type, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_RealType(self, node: RealType, context: RESTGenerationContext) -> list[str]:
        return ["float"]

    def visit_StringType(self, node: StringType, context: RESTGenerationContext) -> list[str]:
        return ["str"]

    def visit_IntType(self, node: IntType, context: RESTGenerationContext) -> list[str]:
        return ["int"]

    def visit_EnumType(self, node: EnumType, context: RESTGenerationContext) -> list[str]:
        return [node.enum.name]

    def visit_BoolType(self, node: BoolType, context: RESTGenerationContext) -> list[str]:
        return ["bool"]

    def visit_Nothing(self, node: Nothing, context: RESTGenerationContext) -> list[str]:
        return ["None"]

    def visit_Assignment(self, node: Assignment, context: RESTGenerationContext) -> list[str]:

        target = node.target
        assignee = node.assignee.accept(self, context)[0]
        assignee_by_id = f"{assignee}.id"
        list_update = None

        if isinstance(target, ArrayAccess):
            a_access:ArrayAccess = target
            if isinstance(a_access.receiver, FieldAccess):
                a_receiver = a_access.receiver.accept(self, context)[0]
                index = a_access.index.accept(self, context)[0]
                list_update = [f"list_update = {a_receiver}", f"list_update[{index}] = {assignee_by_id}"]
                target = a_access.receiver
                assignee = assignee_by_id = "list_update"


        if isinstance(target, FieldAccess):
            f_access: FieldAccess = target
            receiver = f_access.receiver.accept(self, context)[0]

            out = [f"inst_to_update = {receiver}"]
            if list_update is not None:
                out.extend(list_update)

            # We use the typechecker to know the type of the object on which we access the field
            tc = BALTypeChecker()
            tc_context = TypeCheckingContext()
            type = f_access.receiver.accept(tc, tc_context)

            if isinstance(type, ObjectType):
                class_name = type.clazz.name
                mapping = []
                for attr in type.clazz.all_attributes():
                    if attr.name == f_access.field.name:
                        mapping.append(f"{attr.name} = {assignee}")
                    else:
                        mapping.append(f"{attr.name} = inst_to_update.{attr.name}")

                for end in type.clazz.all_association_ends():
                    if end.name == f_access.field.name:
                        mapping.append(f"{end.name} = {assignee_by_id}")
                    else:
                        mapping.append(f"{end.name} = inst_to_update.{end.name}")

                out.append(f"await update_{class_name.lower()}(inst_to_update.id, {class_name}Create({", ".join(mapping)}), database)")
                return out
            return []
        else:
            target = node.target.accept(self, context)[0]
            return [f"{target} = {assignee}"]

    def visit_Statement(self, node: Statement, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_NameDecl(self, node: NameDecl, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_ExplicitDecl(self, node: ExplicitDecl, context: RESTGenerationContext) -> list[str]:
        return [f"{node.name}: {node.declared_type.accept(self, context)[0]}"]

    def visit_ImplicitDecl(self, node: ImplicitDecl, context: RESTGenerationContext) -> list[str]:
        return [f"{node.name}"]

    def visit_For(self, node: For, context: RESTGenerationContext) -> list[str]:
        lines = []
        body = []
        for iterator in node.iterators:
            expr = iterator.sequence.accept(self, context)[0]
            name = iterator.var_name.name + "_" + str(node.__hash__())
            lines.append(f"{name} = {expr}")
            body.append(f"{iterator.var_name.name} = {next(iter(node.iterators)).var_name.name}_{node.__hash__()}[i]")

        lines.append(f"for i in range(0, len({next(iter(node.iterators)).var_name.name}_{node.__hash__()})):")
        body.extend(node.body.accept(self, context))
        lines.extend(indent(body))

        return lines

    def visit_Return(self, node: Return, context: RESTGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)[0]
        return [f"return {expr}"]

    def visit_Expression(self, node: Expression, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_Boolean(self, node: Boolean, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_BinaryBoolean(self, node: BinaryBoolean, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_LessEq(self, node: LessEq, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} <= {right})"]

    def visit_And(self, node: And, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} and {right})"]

    def visit_Or(self, node: Or, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} or {right})"]

    def visit_Equal(self, node: Equal, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} == {right})"]

    def visit_Greater(self, node: Greater, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} > {right})"]

    def visit_Less(self, node: Less, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} < {right})"]

    def visit_Unequal(self, node: Unequal, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} != {right})"]

    def visit_GreaterEq(self, node: GreaterEq, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} >= {right})"]

    def visit_Not(self, node: Not, context: RESTGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)[0]
        return [f"(not {expr})"]

    def visit_Call(self, node: Call, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_New(self, node: New, context: RESTGenerationContext) -> list[str]:
        param_mapping_str = []
        for param, arg in zip(node.clazz.clazz.all_attributes(), node.arguments):
            expr = arg.accept(self, context)[0]
            param_mapping_str.append(f"{param.name} = {expr}")

        return [f"(await create_{node.clazz.clazz.name.lower()}({node.clazz.clazz.name}Create({', '.join(param_mapping_str)}), database))"]

    def visit_MethodCall(self, node: MethodCall, context: RESTGenerationContext) -> list[str]:
        param_dict_str = []
        for param, arg in zip(node.method.parameters, node.arguments):
            expr = arg.accept(self, context)[0]
            param_dict_str.append(f"'{param.name}': {expr}")

        receiver = node.receiver.accept(self, context)[0]

        return [f"(await execute_{self.class_name.lower()}_{node.method.name}({receiver}.id, {{ {', '.join(param_dict_str)} }}, database))"]

    def visit_StandardLibCall(self, node: StandardLibCall, context: RESTGenerationContext) -> list[str]:
        args = []
        for arg in node.arguments:
            args.append(arg.accept(self, context)[0])
        receiver = node.receiver.accept(self, context)[0]
        return [f"{node.function}({receiver}, {', '.join(args)})"]

    def visit_ProcedureCall(self, node: ProcedureCall, context: RESTGenerationContext) -> list[str]:
        args = []
        for arg in node.arguments:
            args.append(arg.accept(self, context)[0])
        return [f"(await {node.function.name}({', '.join(args)}))"]

    def visit_This(self, node: This, context: RESTGenerationContext) -> list[str]:
        return ['_' + self.class_name.lower() + '_object']

    def visit_FieldAccess(self, node: FieldAccess, context: RESTGenerationContext) -> list[str]:
        receiver = node.receiver.accept(self, context)[0]
        if node.field.type in data_types:
            return f"{receiver}.{node.field.name}"
        else:
            return f"(await get_{self.class_name.lower()}({receiver}.{node.field.name}.id, database))"

    def visit_ArrayAccess(self, node: ArrayAccess, context: RESTGenerationContext) -> list[str]:
        receiver = node.receiver.accept(self, context)[0]
        index = node.index.accept(self, context)[0]
        return [f"{receiver}[{index}]"]

    def visit_Cast(self, node: Cast, context: RESTGenerationContext) -> list[str]:
        return node.instance.accept(self, context)

    def visit_NullCoalessing(self, node: NullCoalessing, context: RESTGenerationContext) -> list[str]:
        nullable = node.nullable.accept(self, context)
        elze = node.elze.accept(self, context)
        return [f"({nullable} if {nullable} is not None else {elze})"]

    def visit_Ternary(self, node: Ternary, context: RESTGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)
        then = node.then.accept(self, context)
        elze = node.elze.accept(self, context)
        return [f"({then} if {expr} else {elze})"]

    def visit_Arithmetic(self, node: Arithmetic, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_UnaryMinus(self, node: UnaryMinus, context: RESTGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)[0]
        return [f"(- {expr})"]

    def visit_BinaryArithmetic(self, node: BinaryArithmetic, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_Div(self, node: Div, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} / {right})"]

    def visit_Remain(self, node: Remain, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} % {right})"]

    def visit_Mult(self, node: Mult, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} * {right})"]

    def visit_Plus(self, node: Plus, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} + {right})"]

    def visit_Minus(self, node: Minus, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} - {right})"]

    def visit_Concatenation(self, node: Concatenation, context: RESTGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"(str({left}) + str({right}))"]

    def visit_InstanceOf(self, node: InstanceOf, context: RESTGenerationContext) -> list[str]:
        instance = node.instance.accept(self, context)
        the_type = node.type.accept(self, context)
        return [f"isinstance({instance}, {the_type})"]

    def visit_Reference(self, node: Reference, context: RESTGenerationContext) -> list[str]:
        return [node.definition.name]

    def visit_Literal(self, node: Literal, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_IntLiteral(self, node: IntLiteral, context: RESTGenerationContext) -> list[str]:
        return [f"{node.value}"]

    def visit_StringLiteral(self, node: StringLiteral, context: RESTGenerationContext) -> list[str]:
        return [f"'{node.value}'"]

    def visit_BoolLiteral(self, node: BoolLiteral, context: RESTGenerationContext) -> list[str]:
        return [f"{'True' if node.value else 'False'}"]

    def visit_RealLiteral(self, node: RealLiteral, context: RESTGenerationContext) -> list[str]:
        return [f"{node.value}"]

    def visit_NullLiteral(self, node: NullLiteral, context: RESTGenerationContext) -> list[str]:
        return [f"None"]

    def visit_EnumLiteral(self, node: EnumLiteral, context: RESTGenerationContext) -> list[str]:
        return [f"{node.enumeration.enum.name}.{node.name}"]

    def visit_SequenceLiteral(self, node: SequenceLiteral, context: RESTGenerationContext) -> list[str]:
        elems = []
        for elem in node.values:
            elems += elem.accept(self, context)
        return [f"[{', '.join(elems)}]"]

    def visit_RangeLiteral(self, node: RangeLiteral, context: RESTGenerationContext) -> list[str]:
        first = node.first.accept(self, context)
        last = node.last.accept(self, context)
        return [f"list(range({first}, {last}))"]

    def visit_Iterator(self, node: Iterator, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_CondLoop(self, node: CondLoop, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_While(self, node: While, context: RESTGenerationContext) -> list[str]:
        cond = node.condition.accept(self, context)[0]
        lines = [f"while {cond}:"]
        lines += indent(node.body.accept(self, context))
        return lines

    def visit_DoWhile(self, node: DoWhile, context: RESTGenerationContext) -> list[str]:
        cond = node.condition.accept(self, context)[0]
        body = node.body.accept(self, context)
        body += [f"if not ({cond}):", "\tbreak"]
        lines = ["while True:"]
        lines += indent(body)
        return lines

    def visit_ConditionalBranch(self, node: ConditionalBranch, context: RESTGenerationContext) -> list[str]:
        pass

    def visit_Block(self, node: Block, context: RESTGenerationContext) -> list[str]:
        statements = []
        for statement in node.statements:
            statements += statement.accept(self, context)

        if len(statements) > 0:
            return statements
        else:
            return ["pass"]

    def visit_Condition(self, node: Condition, context: RESTGenerationContext) -> list[str]:
        cond = node.condition.accept(self, context)[0]
        lines = [f"if {cond}:"]
        then = node.then.accept(self, context)
        lines += indent(then)

        if isinstance(node.elze, Condition):
            elseif = node.elze.accept(self, context)
            elseif[0] = "el" + elseif[0]
            lines += elseif
        elif isinstance(node.elze, Block):
            elze = node.elze.accept(self, context)
            lines.append("else:")
            lines += indent(elze)

        return lines

