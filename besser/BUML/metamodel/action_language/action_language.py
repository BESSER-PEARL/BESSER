from abc import ABC
from typing import TypeVar

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from besser.BUML.metamodel.action_language.visitors import BALVisitor
    from besser.BUML.metamodel.structural import Method, Property, Class, Enumeration

ContextType = TypeVar('ContextType')
ReturnType = TypeVar('ReturnType')

############################################
# Definition of Classes
############################################


#==================#
# Abstract Classes #
#==================#


class Statements(ABC):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Statements(self, context)
    

class Expression(Statements):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Expression(self, context)


class AssignTarget(ABC):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_AssignTarget(self, context)


#===============#
# Types Classes #
#===============#


class Type(ABC):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Type(self, context)


class AnyType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_AnyType(self, context)


class ObjectType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_ObjectType(self, context)

    def __init__(self, clazz: 'Class'):
        self.__clazz = clazz

    @property
    def clazz(self) -> 'Class':
        return self.__clazz

    @clazz.setter
    def clazz(self, clazz: 'Class'):
        self.__clazz = clazz


class SequenceType(Type):
    def __init__(self, elementsType: Type):
        self.__elementsType = elementsType

    @property
    def elementsType(self) -> Type:
        return self.__elementsType

    @elementsType.setter
    def elementsType(self, elementsType: Type):
        self.__elementsType = elementsType

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_SequenceType(self, context)


class BoolType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_BoolType(self, context)


class StringType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_StringType(self, context)


class RealType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_RealType(self, context)


class IntType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_IntType(self, context)


class NaturalType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_NaturalType(self, context)


class EnumType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_EnumType(self, context)

    def __init__(self, enum: 'Enumeration'):
        self.__enum = enum

    @property
    def enum(self) -> 'Enumeration':
        return self.__enum

    @enum.setter
    def enum(self, enum: 'Enumeration'):
        self.__enum = enum


class FunctionType(Type):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_EnumType(self, context)

    def __init__(self, params_type: list[Type], return_type: Type):
        self.__params_type = params_type
        self.__return_type = return_type

    @property
    def params_type(self) -> list[Type]:
        return self.__params_type

    @params_type.setter
    def params_type(self, params_type: list[Type]):
        self.__params_type = params_type

    @property
    def return_type(self) -> Type:
        return self.__return_type

    @return_type.setter
    def return_type(self, return_type: Type):
        self.__return_type = return_type


#==========================#
# Name Declaration Classes #
#==========================#


class Multiplicity:
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Multiplicity(self, context)

    def __init__(self, nullable: bool, many: bool):
        self.__nullable = nullable
        self.__many = many

    @property
    def many(self) -> bool:
        return self.__many

    @many.setter
    def many(self, many: bool):
        self.__many = many

    @property
    def nullable(self) -> bool:
        return self.__nullable

    @nullable.setter
    def nullable(self, nullable: bool):
        self.__nullable = nullable


class NameDecl(Statements):
    def __init__(self, name: str, declared_type: Type, multiplicity: Multiplicity):
        self.__name = name
        self.__declared_type = declared_type
        self.__multiplicity = multiplicity

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def declared_type(self) -> Type:
        return self.__declared_type

    @declared_type.setter
    def declared_type(self, declared_type: Type):
        self.__declared_type = declared_type

    @property
    def multiplicity(self) -> Multiplicity:
        return self.__multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity: Multiplicity):
        self.__multiplicity = multiplicity

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_NameDecl(self, context)


class ExplicitDecl(AssignTarget, NameDecl):
    def __init__(self, name: str, declared_type: Type, multiplicity: Multiplicity):
        super().__init__(name, declared_type, multiplicity)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_ExplicitDecl(self, context)


class ImplicitDecl(AssignTarget, NameDecl):
    def __init__(self, name: str, declared_type: Type, multiplicity: Multiplicity):
        super().__init__(name, declared_type, multiplicity)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_ImplicitDecl(self, context)
    

class Parameter(NameDecl):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Parameter(self, context)

    def __init__(self, name: str, declared_type: "Type" = None, default: "Expression" = None):
        mult = Multiplicity(True, isinstance(declared_type, SequenceType))
        super().__init__(name, declared_type, mult)
        self.__default = default

    @property
    def default(self):
        return self.__default

    @default.setter
    def default(self, value):
        self.__default = value


#=====================#
# Function Definition #
#      Root Class     #
#=====================#


class FunctionDefinition(NameDecl):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_FunctionDefinition(self, context)

    def __init__(self, name: str, parameters: list["Parameter"] = None, return_type: "Type" = None,
                 body: "Block" = None):
        self.__parameters = parameters if parameters is not None else set()
        self.__return_type = return_type
        self.__body = body
        param_types = [param.declared_type for param in self.__parameters]
        super().__init__(name, FunctionType(param_types, return_type), Multiplicity(False, False))



    @property
    def return_type(self):
        return self.__return_type

    @return_type.setter
    def return_type(self, value):
        self.__return_type = value

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, value):
        self.__parameters = value if value is not None else set()

    @property
    def body(self):
        return self.__body

    @body.setter
    def body(self, value):
        self.__body = value


#====================#
# Statements Classes #
#====================#


class ConditionalBranch(ABC):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_ConditionalBranch(self, context)
    

class Block(ConditionalBranch):
    def __init__(self, statements:list[Statements]):
        self.__statements = statements

    @property
    def statements(self) -> any:
        return self.__statements

    @statements.setter
    def statements(self, statements: list[Statements]):
        self.__statements = statements

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Block(self, context)


class Condition(Statements, ConditionalBranch):
    def __init__(self, condition: Expression, then: Block, elze: ConditionalBranch):
        self.__condition = condition
        self.__then = then
        self.__elze = elze

    @property
    def condition(self) -> Expression:
        return self.__condition

    @condition.setter
    def condition(self, condition: Expression):
        self.__condition = condition

    @property
    def then(self) -> Block:
        return self.__then

    @then.setter
    def then(self, then: Block):
        self.__then = then

    @property
    def elze(self) -> ConditionalBranch:
        return self.__elze

    @elze.setter
    def elze(self, elze: ConditionalBranch):
        self.__elze = elze

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Condition(self, context)
    

class CondLoop(Statements):
    def __init__(self, condition: Expression, body: Block):
        self.__condition = condition
        self.__body = body

    @property
    def condition(self) -> any:
        return self.__condition

    @condition.setter
    def condition(self, condition: Expression):
        self.__condition = condition

    @property
    def body(self) -> any:
        return self.__body

    @body.setter
    def body(self, body: Block):
        self.__body = body

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_CondLoop(self, context)


class DoWhile(CondLoop):
    def __init__(self, condition: Expression, body: Block):
        super().__init__(condition, body)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_DoWhile(self, context)


class While(CondLoop):
    def __init__(self, condition: Expression, body: Block):
        super().__init__(condition, body)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_While(self, context)


class Iterator:
    def __init__(self, var_name: NameDecl, sequence: Expression):
        self.__var_name = var_name
        self.__sequence = sequence

    @property
    def var_name(self) -> NameDecl:
        return self.__var_name

    @var_name.setter
    def var_name(self, var_name: NameDecl):
        self.__var_name = var_name

    @property
    def sequence(self) -> Expression:
        return self.__sequence

    @sequence.setter
    def sequence(self, sequence: Expression):
        self.__sequence = sequence

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Iterator(self, context)


class For(Statements):
    def __init__(self, iterators:set[Iterator], body:Block):
        self.__iterators = iterators
        self.__body = body

    @property
    def iterators(self) -> set[Iterator]:
        return self.__iterators

    @iterators.setter
    def iterators(self, iterators: set[Iterator]):
        self.__iterators = iterators

    @property
    def body(self) -> Block:
        return self.__body

    @body.setter
    def body(self, body: Block):
        self.__body = body

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_For(self, context)


class Return(Statements):
    def __init__(self, expr: Expression):
        self.__expr = expr

    @property
    def expr(self) -> Expression:
        return self.__expr

    @expr.setter
    def expr(self, expr: Expression):
        self.__expr = expr

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Return(self, context)


#=====================#
# Expressions Classes #
#=====================#


class Assignment:
    def __init__(self, target: AssignTarget, assignee: Expression):
        self.__target = target
        self.__assignee = assignee

    @property
    def target(self) -> any:
        return self.__target

    @target.setter
    def target(self, target: any):
        self.__target = target

    @property
    def assignee(self) -> any:
        return self.__assignee

    @assignee.setter
    def assignee(self, assignee: any):
        self.__assignee = assignee

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Assignment(self, context)


class Ternary(Expression):
    def __init__(self, expr: Expression, then: Expression, elze: Expression):
        self.__expr = expr
        self.__then = then
        self.__elze = elze

    @property
    def expr(self) -> Expression:
        return self.__expr

    @expr.setter
    def expr(self, expr: Expression):
        self.__expr = expr

    @property
    def then(self):
        return self.__then

    @then.setter
    def then(self, value):
        self.__then = value

    @property
    def elze(self) -> Expression:
        return self.__elze

    @elze.setter
    def elze(self, elze: Expression):
        self.__elze = elze

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Ternary(self, context)


# Boolean Expressions #
class Boolean(Expression):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Boolean(self, context)


class Not(Boolean):
    def __init__(self, expr: Expression):
        self.__expr = expr

    @property
    def expr(self):
        return self.__expr

    @expr.setter
    def expr(self, value):
        self.__expr = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Not(self, context)


class BinaryBoolean(Boolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        self.__left = left
        self.__right = right

    @property
    def left(self) -> Expression:
        return self.__left

    @left.setter
    def left(self, left: Expression):
        self.__left = left

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, value):
        self.__right = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_BinaryBoolean(self, context)


class Or(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Or(self, context)


class And(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_And(self, context)


class Unequal(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Unequal(self, context)


class Equal(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Equal(self, context)


class GreaterEq(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_GreaterEq(self, context)


class LessEq(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_LessEq(self, context)


class Greater(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Greater(self, context)


class Less(BinaryBoolean):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Less(self, context)


class InstanceOf(Expression):
    def __init__(self, instance: Expression, the_type: Type):
        self.__instance = instance
        self.__type = the_type

    @property
    def instance(self) -> Expression:
        return self.__instance

    @instance.setter
    def instance(self, instance: Expression):
        self.__instance = instance

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        self.__type = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_InstanceOf(self, context)


# Arithmetic Expressions #
class Concatenation(Expression):
    def __init__(self, left: Expression, right: "Expression" = None):
        self.__left = left
        self.__right = right

    @property
    def left(self) -> Expression:
        return self.__left

    @left.setter
    def left(self, left: Expression):
        self.__left = left

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, value):
        self.__right = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Concatenation(self, context)


class Arithmetic(Expression):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Arithmetic(self, context)


class UnaryMinus(Arithmetic):
    def __init__(self, expr: Expression):
        self.__expr = expr

    @property
    def expr(self):
        return self.__expr

    @expr.setter
    def expr(self, value):
        self.__expr = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_UnaryMinus(self, context)


class BinaryArithmetic(Arithmetic):
    def __init__(self, left: Expression, right: "Expression" = None):
        self.__left = left
        self.__right = right

    @property
    def left(self) -> Expression:
        return self.__left

    @left.setter
    def left(self, left: Expression):
        self.__left = left

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, value):
        self.__right = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_BinaryArithmetic(self, context)


class Plus(BinaryArithmetic):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Plus(self, context)


class Minus(BinaryArithmetic):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Minus(self, context)


class Mult(BinaryArithmetic):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Mult(self, context)


class Div(BinaryArithmetic):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Div(self, context)


class Remain(BinaryArithmetic):
    def __init__(self, left: Expression, right: "Expression" = None):
        super().__init__(left, right)

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Remain(self, context)


# Type Expressions #
class Cast(Expression):
    def __init__(self, instance: Expression, as_type: Type):
        self.__instance = instance
        self.__as_type = as_type

    @property
    def instance(self) -> Expression:
        return self.__instance

    @instance.setter
    def instance(self, instance: Expression):
        self.__instance = instance

    @property
    def as_type(self):
        return self.__as_type

    @as_type.setter
    def as_type(self, value):
        self.__as_type = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Cast(self, context)


class NullCoalessing(Expression):
    def __init__(self, nullable: Expression, elze: "Expression" = None):
        self.__nullable = nullable
        self.__elze = elze

    @property
    def nullable(self) -> Expression:
        return self.__nullable

    @nullable.setter
    def nullable(self, nullable: Expression):
        self.__nullable = nullable

    @property
    def elze(self):
        return self.__elze

    @elze.setter
    def elze(self, value):
        self.__elze = value

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_NullCoalessing(self, context)


# Access Expressions #
class FieldAccess(AssignTarget, Expression):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_FieldAccess(self, context)

    def __init__(self, receiver: Expression, field: 'Property'):
        self.__field = field
        self.__receiver = receiver

    @property
    def field(self) -> 'Property':
        return self.__field

    @field.setter
    def field(self, field: 'Property'):
        self.__field = field

    @property
    def receiver(self):
        return self.__receiver

    @receiver.setter
    def receiver(self, value):
        self.__receiver = value

class ArrayAccess(AssignTarget, Expression):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_ArrayAccess(self, context)

    def __init__(self, receiver: "Expression", index: int):
        self.__index = index
        self.__receiver = receiver

    @property
    def index(self) -> int:
        return self.__index

    @index.setter
    def index(self, index: int):
        self.__index = index

    @property
    def receiver(self):
        return self.__receiver

    @receiver.setter
    def receiver(self, value):
        self.__receiver = value


# Call Expressions #
class Call(Expression):
    def __init__(self, arguments: list[Expression]):
        self.__arguments = arguments

    @property
    def arguments(self) -> list[Expression]:
        return self.__arguments

    @arguments.setter
    def arguments(self, arguments: list[Expression]):
        self.__arguments = arguments

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Call(self, context)


class MethodCall(Call):
    def __init__(self, receiver: Expression, method: 'Method', arguments: list[Expression]):
        super().__init__(arguments)
        self.__receiver = receiver
        self.__method = method

    @property
    def receiver(self) -> Expression:
        return self.__receiver

    @receiver.setter
    def receiver(self, receiver: Expression):
        self.__receiver = receiver

    @property
    def method(self) -> 'Method':
        return self.__method

    @method.setter
    def method(self, method: 'Method'):
        self.__method = method

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_MethodCall(self, context)


class New(Call):
    def __init__(self, clazz: ObjectType, arguments: list[Expression]):
        super().__init__(arguments)
        self.__clazz = clazz

    @property
    def clazz(self) -> ObjectType:
        return self.__clazz

    @clazz.setter
    def clazz(self, clazz: ObjectType):
        self.__clazz = clazz

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_New(self, context)

class StandardLibCall(Call):
    def __init__(self, receiver: Expression, function: str, fn_type: FunctionType, arguments: list[Expression]):
        super().__init__(arguments)
        self.__receiver = receiver
        self.__function = function
        self.__function_type = fn_type

    @property
    def receiver(self) -> Expression:
        return self.__receiver

    @receiver.setter
    def receiver(self, receiver: Expression):
        self.__receiver = receiver

    @property
    def function(self) -> str:
        return self.__function

    @function.setter
    def function(self, function: str):
        self.__function = function

    @property
    def function_type(self) -> str:
        return self.__function_type

    @function_type.setter
    def function_type(self, function_type: str):
        self.__function_type = function_type

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_StandardLibCall(self, context)


class ProcedureCall(Call):
    def __init__(self, function: FunctionDefinition, arguments: list[Expression]):
        super().__init__(arguments)
        self.__function = function

    @property
    def function(self) -> FunctionDefinition:
        return self.__function

    @function.setter
    def function(self, function: FunctionDefinition):
        self.__function = function

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_ProcedureCall(self, context)


# References Expressions #
class This(Expression):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_This(self, context)


class Reference(AssignTarget, Expression):
    def __init__(self, definition: NameDecl):
        self.__definition = definition

    @property
    def definition(self) -> NameDecl:
        return self.__definition

    @definition.setter
    def definition(self, definition: NameDecl):
        self.__definition = definition

    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Reference(self, context)


#==================#
# Literals Classes #
#==================#


class Literal(Expression):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_Literal(self, context)


class StringLiteral(Literal):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_StringLiteral(self, context)

    def __init__(self, value: str):
        self.__value = value

    @property
    def value(self) -> str:
        return self.__value

    @value.setter
    def value(self, value: str):
        self.__value = value


class BoolLiteral(Literal):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_BoolLiteral(self, context)

    def __init__(self, value: bool):
        self.__value = value

    @property
    def value(self) -> bool:
        return self.__value

    @value.setter
    def value(self, value: bool):
        self.__value = value


class RealLiteral(Literal):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_RealLiteral(self, context)

    def __init__(self, value: float):
        self.__value = value

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value: float):
        self.__value = value


class IntLiteral(Literal):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_IntLiteral(self, context)

    def __init__(self, value: int):
        self.__value = value

    @property
    def value(self) -> int:
        return self.__value

    @value.setter
    def value(self, value: int):
        self.__value = value


class NullLiteral(Literal):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_NullLiteral(self, context)


class EnumLiteral(Literal):
    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_EnumLiteral(self, context)

    def __init__(self, enumeration: EnumType, name: str):
        self.__name = name
        self.__enumeration = enumeration

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def enumeration(self):
        return self.__enumeration

    @enumeration.setter
    def enumeration(self, value):
        self.__enumeration = value


class SequenceLiteral(Literal):
    def __init__(self, value_type: Type, values: list[Expression]):
        self.__value_type = value_type
        self.__values = values


    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_SequenceLiteral(self, context)

    @property
    def value_type(self) -> Type:
        return self.__value_type

    @value_type.setter
    def value_type(self, value_type: Type):
        self.__value_type = value_type

    @property
    def values(self) -> list[Expression]:
        return self.__values

    @values.setter
    def values(self, values: list[Expression]):
        self.__values = values


class RangeLiteral(Literal):
    def __init__(self, first: Expression, last: Expression):
        self.__first = first
        self.__last = last

    @property
    def first(self) -> Expression:
        return self.__first

    @first.setter
    def first(self, first: Expression):
        self.__first = first

    @property
    def last(self) -> Expression:
        return self.__last

    @last.setter
    def last(self, last: Expression):
        self.__last = last


    def accept(self, bal_visitor: 'BALVisitor[ContextType, ReturnType]', context: ContextType) -> ReturnType:
        return bal_visitor.visit_RangeLiteral(self, context)