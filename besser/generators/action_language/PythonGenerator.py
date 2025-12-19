from besser.BUML.metamodel.action_language.action_language import Condition, Block, ConditionalBranch, DoWhile, While, \
    CondLoop, Iterator, RangeLiteral, SequenceLiteral, EnumLiteral, NullLiteral, RealLiteral, BoolLiteral, \
    StringLiteral, IntLiteral, Literal, Reference, InstanceOf, Concatenation, Minus, Plus, Mult, Remain, Div, \
    BinaryArithmetic, UnaryMinus, Arithmetic, Ternary, NullCoalessing, Cast, ArrayAccess, FieldAccess, This, \
    ProcedureCall, StandardLibCall, MethodCall, New, Call, Not, GreaterEq, Unequal, Less, Greater, Equal, Or, And, \
    LessEq, BinaryBoolean, Boolean, Expression, Return, For, ImplicitDecl, ExplicitDecl, NameDecl, Statements, \
    Assignment, BoolType, EnumType, IntType, StringType, RealType, NaturalType, Type, FunctionType, SequenceType, \
    ObjectType, Multiplicity, AnyType, FunctionDefinition, Parameter, AssignTarget
from besser.BUML.metamodel.action_language.visitors import BALVisitor


class PythonGenerationContext:
    pass

def indent(lines):
    out = []
    for line in lines:
        out.append('\t'+line)
    return out

class BALPythonGenerator(BALVisitor[PythonGenerationContext, list[str]]):

    def generate(self, node: FunctionDefinition):
        lines = node.accept(self, PythonGenerationContext())
        return "\n".join(lines)

    def visit_AssignTarget(self, node: AssignTarget, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_Parameter(self, node: Parameter, context: PythonGenerationContext) -> list[str]:
        param_type = node.declared_type.accept(self, context)[0]
        return [f"{node.name}: {param_type}"]

    def visit_FunctionDefinition(self, node: FunctionDefinition, context: PythonGenerationContext) -> list[str]:
        lines = []

        params = []
        for param in node.parameters:
            params += param.accept(self, context)
        lines.append(f"def {node.name}({', '.join(params)}) -> {node.return_type.accept(self, context)[0]} :")

        body = node.body.accept(self, context)
        lines.extend(indent(body))

        return lines


    def visit_AnyType(self, node: AnyType, context: PythonGenerationContext) -> list[str]:
        return ["Any"]

    def visit_Multiplicity(self, node: Multiplicity, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_ObjectType(self, node: ObjectType, context: PythonGenerationContext) -> list[str]:
        return [node.clazz.name]

    def visit_SequenceType(self, node: SequenceType, context: PythonGenerationContext) -> list[str]:
        return [f"list[{node.elementsType.accept(self, context)[0]}]"]

    def visit_FunctionType(self, node: FunctionType, context: PythonGenerationContext) -> list[str]:
        return ["Any"]

    def visit_Type(self, node: Type, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_NaturalType(self, node: NaturalType, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_RealType(self, node: RealType, context: PythonGenerationContext) -> list[str]:
        return ["float"]

    def visit_StringType(self, node: StringType, context: PythonGenerationContext) -> list[str]:
        return ["str"]

    def visit_IntType(self, node: IntType, context: PythonGenerationContext) -> list[str]:
        return ["int"]

    def visit_EnumType(self, node: EnumType, context: PythonGenerationContext) -> list[str]:
        return [node.enum.name]

    def visit_BoolType(self, node: BoolType, context: PythonGenerationContext) -> list[str]:
        return ["bool"]

    def visit_Assignment(self, node: Assignment, context: PythonGenerationContext) -> list[str]:
        target = node.target.accept(self, context)
        assignee = node.assignee.accept(self, context)
        return [f"{target[0]} = {assignee[0]}"]

    def visit_Statements(self, node: Statements, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_NameDecl(self, node: NameDecl, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_ExplicitDecl(self, node: ExplicitDecl, context: PythonGenerationContext) -> list[str]:
        return [f"{node.name}: {node.declared_type.accept(self, context)[0]}"]

    def visit_ImplicitDecl(self, node: ImplicitDecl, context: PythonGenerationContext) -> list[str]:
        return [f"{node.name}"]

    def visit_For(self, node: For, context: PythonGenerationContext) -> list[str]:
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

    def visit_Return(self, node: Return, context: PythonGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)[0]
        return [f"return {expr}"]

    def visit_Expression(self, node: Expression, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_Boolean(self, node: Boolean, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_BinaryBoolean(self, node: BinaryBoolean, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_LessEq(self, node: LessEq, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} <= {right})"]

    def visit_And(self, node: And, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} and {right})"]

    def visit_Or(self, node: Or, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} or {right})"]

    def visit_Equal(self, node: Equal, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} == {right})"]

    def visit_Greater(self, node: Greater, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} > {right})"]

    def visit_Less(self, node: Less, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} < {right})"]

    def visit_Unequal(self, node: Unequal, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} != {right})"]

    def visit_GreaterEq(self, node: GreaterEq, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} >= {right})"]

    def visit_Not(self, node: Not, context: PythonGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)[0]
        return [f"(not {expr})"]

    def visit_Call(self, node: Call, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_New(self, node: New, context: PythonGenerationContext) -> list[str]:
        args = []
        for arg in node.arguments:
            args.append(arg.accept(self, context)[0])
        return [f"{node.clazz.clazz.name}({', '.join(args)})"]

    def visit_MethodCall(self, node: MethodCall, context: PythonGenerationContext) -> list[str]:
        args = []
        for arg in node.arguments:
            args.append(arg.accept(self, context)[0])
        receiver = node.receiver.accept(self, context)[0]
        return [f"{receiver}.{node.method.name}({', '.join(args)})"]

    def visit_StandardLibCall(self, node: StandardLibCall, context: PythonGenerationContext) -> list[str]:
        args = []
        for arg in node.arguments:
            args.append(arg.accept(self, context)[0])
        receiver = node.receiver.accept(self, context)[0]
        return [f"{node.function}({receiver}, {', '.join(args)})"]

    def visit_ProcedureCall(self, node: ProcedureCall, context: PythonGenerationContext) -> list[str]:
        args = []
        for arg in node.arguments:
            args.append(arg.accept(self, context))
        return [f"{node.function.name}({', '.join(args)})"]

    def visit_This(self, node: This, context: PythonGenerationContext) -> list[str]:
        return ["self"]

    def visit_FieldAccess(self, node: FieldAccess, context: PythonGenerationContext) -> list[str]:
        receiver = node.receiver.accept(self, context)[0]
        return [f"{receiver}.{node.field.name}"]

    def visit_ArrayAccess(self, node: ArrayAccess, context: PythonGenerationContext) -> list[str]:
        receiver = node.receiver.accept(self, context)[0]
        index = node.index.accept(self, context)[0]
        return [f"{receiver}[{index}]"]

    def visit_Cast(self, node: Cast, context: PythonGenerationContext) -> list[str]:
        return node.instance.accept(self, context)

    def visit_NullCoalessing(self, node: NullCoalessing, context: PythonGenerationContext) -> list[str]:
        nullable = node.nullable.accept(self, context)
        elze = node.elze.accept(self, context)
        return [f"({nullable} if {nullable} is not None else {elze})"]

    def visit_Ternary(self, node: Ternary, context: PythonGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)
        then = node.then.accept(self, context)
        elze = node.elze.accept(self, context)
        return [f"({then} if {expr} else {elze})"]

    def visit_Arithmetic(self, node: Arithmetic, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_UnaryMinus(self, node: UnaryMinus, context: PythonGenerationContext) -> list[str]:
        expr = node.expr.accept(self, context)[0]
        return [f"(- {expr})"]

    def visit_BinaryArithmetic(self, node: BinaryArithmetic, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_Div(self, node: Div, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} / {right})"]

    def visit_Remain(self, node: Remain, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} % {right})"]

    def visit_Mult(self, node: Mult, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} * {right})"]

    def visit_Plus(self, node: Plus, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} + {right})"]

    def visit_Minus(self, node: Minus, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"({left} - {right})"]

    def visit_Concatenation(self, node: Concatenation, context: PythonGenerationContext) -> list[str]:
        left = node.left.accept(self, context)[0]
        right = node.right.accept(self, context)[0]
        return [f"(str({left}) + str({right}))"]

    def visit_InstanceOf(self, node: InstanceOf, context: PythonGenerationContext) -> list[str]:
        instance = node.instance.accept(self, context)
        the_type = node.type.accept(self, context)
        return [f"isinstance({instance}, {the_type})"]

    def visit_Reference(self, node: Reference, context: PythonGenerationContext) -> list[str]:
        return [node.definition.name]

    def visit_Literal(self, node: Literal, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_IntLiteral(self, node: IntLiteral, context: PythonGenerationContext) -> list[str]:
        return [f"{node.value}"]

    def visit_StringLiteral(self, node: StringLiteral, context: PythonGenerationContext) -> list[str]:
        return [f"'{node.value}'"]

    def visit_BoolLiteral(self, node: BoolLiteral, context: PythonGenerationContext) -> list[str]:
        return [f"{'True' if node.value else 'False'}"]

    def visit_RealLiteral(self, node: RealLiteral, context: PythonGenerationContext) -> list[str]:
        return [f"{node.value}"]

    def visit_NullLiteral(self, node: NullLiteral, context: PythonGenerationContext) -> list[str]:
        return [f"None"]

    def visit_EnumLiteral(self, node: EnumLiteral, context: PythonGenerationContext) -> list[str]:
        return [f"{node.enumeration.enum.name}.{node.name}"]

    def visit_SequenceLiteral(self, node: SequenceLiteral, context: PythonGenerationContext) -> list[str]:
        elems = []
        for elem in node.values:
            elems += elem.accept(self, context)
        return [f"[{', '.join(elems)}]"]

    def visit_RangeLiteral(self, node: RangeLiteral, context: PythonGenerationContext) -> list[str]:
        first = node.first.accept(self, context)
        last = node.last.accept(self, context)
        return [f"list(range({first}, {last}))"]

    def visit_Iterator(self, node: Iterator, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_CondLoop(self, node: CondLoop, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_While(self, node: While, context: PythonGenerationContext) -> list[str]:
        cond = node.condition.accept(self, context)[0]
        lines = [f"while {cond}:"]
        lines += indent(node.body.accept(self, context))
        return lines

    def visit_DoWhile(self, node: DoWhile, context: PythonGenerationContext) -> list[str]:
        cond = node.condition.accept(self, context)[0]
        body = node.body.accept(self, context)
        body += [f"if not ({cond}):", "\tbreak"]
        lines = ["while True:"]
        lines += indent(body)
        return lines

    def visit_ConditionalBranch(self, node: ConditionalBranch, context: PythonGenerationContext) -> list[str]:
        pass

    def visit_Block(self, node: Block, context: PythonGenerationContext) -> list[str]:
        statements = []
        for statement in node.statements:
            statements += statement.accept(self, context)

        if len(statements) > 0:
            return statements
        else:
            return ["pass"]

    def visit_Condition(self, node: Condition, context: PythonGenerationContext) -> list[str]:
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

