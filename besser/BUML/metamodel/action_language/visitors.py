from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from besser.BUML.metamodel.action_language.action_language import Parameter, FunctionDefinition, AnyType, Multiplicity, \
    ObjectType, SequenceType, Type, RealType, StringType, IntType, EnumType, BoolType, Assignment, Statement, \
    NameDecl, ExplicitDecl, ImplicitDecl, For, Expression, Boolean, BinaryBoolean, LessEq, And, Or, Equal, Greater, \
    Less, Unequal, GreaterEq, Not, Call, New, This, FieldAccess, Cast, NullCoalessing, Ternary, Arithmetic, UnaryMinus, \
    BinaryArithmetic, Div, Remain, Mult, Plus, Minus, Concatenation, InstanceOf, Reference, Literal, IntLiteral, \
    StringLiteral, BoolLiteral, RealLiteral, NullLiteral, EnumLiteral, SequenceLiteral, RangeLiteral, Iterator, \
    CondLoop, While, DoWhile, ConditionalBranch, Block, Condition, ArrayAccess, MethodCall, StandardLibCall, Return, \
    AssignTarget, ProcedureCall, FunctionType, OptionalType, Nothing

ContextType = TypeVar('ContextType')
ReturnType = TypeVar('ReturnType')




class BALVisitor(ABC, Generic[ContextType, ReturnType]):
    @abstractmethod
    def visit_AssignTarget(self, node: AssignTarget, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Parameter(self, node: Parameter, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_FunctionDefinition(self, node: FunctionDefinition, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_AnyType(self, node: AnyType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Multiplicity(self, node: Multiplicity, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_ObjectType(self, node: ObjectType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_SequenceType(self, node: SequenceType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_OptionalType(self, node: OptionalType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_FunctionType(self, node: FunctionType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Type(self, node: Type, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_RealType(self, node: RealType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_StringType(self, node: StringType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_IntType(self, node: IntType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_EnumType(self, node: EnumType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_BoolType(self, node: BoolType, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Nothing(self, node: Nothing, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Assignment(self, node: Assignment, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Statement(self, node: Statement, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_NameDecl(self, node: NameDecl, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_ExplicitDecl(self, node: ExplicitDecl, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_ImplicitDecl(self, node: ImplicitDecl, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_For(self, node: For, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Return(self, node: Return, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Expression(self, node: Expression, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Boolean(self, node: Boolean, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_BinaryBoolean(self, node: BinaryBoolean, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_LessEq(self, node: LessEq, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_And(self, node: And, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Or(self, node: Or, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Equal(self, node: Equal, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Greater(self, node: Greater, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Less(self, node: Less, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Unequal(self, node: Unequal, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_GreaterEq(self, node: GreaterEq, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Not(self, node: Not, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Call(self, node: Call, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_New(self, node: New, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_MethodCall(self, node: MethodCall, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_StandardLibCall(self, node: StandardLibCall, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_ProcedureCall(self, node: ProcedureCall, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_This(self, node: This, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_FieldAccess(self, node: FieldAccess, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_ArrayAccess(self, node: ArrayAccess, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Cast(self, node: Cast, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_NullCoalessing(self, node: NullCoalessing, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Ternary(self, node: Ternary, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Arithmetic(self, node: Arithmetic, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_UnaryMinus(self, node: UnaryMinus, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_BinaryArithmetic(self, node: BinaryArithmetic, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Div(self, node: Div, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Remain(self, node: Remain, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Mult(self, node: Mult, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Plus(self, node: Plus, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Minus(self, node: Minus, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Concatenation(self, node: Concatenation, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_InstanceOf(self, node: InstanceOf, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Reference(self, node: Reference, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Literal(self, node: Literal, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_IntLiteral(self, node: IntLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_StringLiteral(self, node: StringLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_BoolLiteral(self, node: BoolLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_RealLiteral(self, node: RealLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_NullLiteral(self, node: NullLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_EnumLiteral(self, node: EnumLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_SequenceLiteral(self, node: SequenceLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_RangeLiteral(self, node: RangeLiteral, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Iterator(self, node: Iterator, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_CondLoop(self, node: CondLoop, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_While(self, node: While, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_DoWhile(self, node: DoWhile, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_ConditionalBranch(self, node: ConditionalBranch, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Block(self, node: Block, context: ContextType) -> ReturnType:
        pass

    @abstractmethod
    def visit_Condition(self, node: Condition, context: ContextType) -> ReturnType:
        pass
