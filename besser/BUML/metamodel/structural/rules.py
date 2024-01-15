from typing import Any
from besser.BUML.metamodel.structural import Class, NamedElement, TypedElement, Type, PrimitiveDataType, Property, Constraint


class OCLExpression(TypedElement):
    def __init__(self, name: str, type: Type):
        super().__init__(name, type)

    def __str__(self) -> str:
        pass


# A literal value part of an OCL expression
class LiteralExpression(OCLExpression):
    def __init__(self, name: str, type: Type, value: Any):
        super().__init__(name, type)
        self.value: Any = value

    @property
    def value(self) -> Any:
        return self.__value

    @value.setter
    def value(self, value: Any):
        self.__value = value

    def __str__(self) -> str:
        return str(self.value)

class IntegerLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: int):
        super().__init__(name, type=PrimitiveDataType(name="int"), value=value)

    def __repr__(self):
        return f'IntegerLiteralExpression({self.value})'


# Expression that returns the value of a given Property
class PropertyCallExpression(OCLExpression):
    def __init__(self, name: str, property: Property):
        super().__init__(name, Type(property.type))
        self.property: Property = property

    def __repr__(self):
        return f'PropertyCallExpression({self.property.name})'

    def __str__(self) -> str:
        return str(self.property.name)

    @property
    def property(self) -> Property:
        return self.__property

    @property.setter
    def property(self, property: Property):
        self.__property = property


# Expression that returns the value of a given Operation on a given set of ordered arguments
# The operation could also be a reference to any operation in an existing class but for now we stick to simple
# arithmetic comparison operations from the standard OCL library
class OperationCallExpression(OCLExpression):
    def __init__(self, name: str, operation: str, arguments: list[OCLExpression]):
        super().__init__(name, Type(PrimitiveDataType("bool")))  # Type for now is always boolean, it should be the return type of the operation
        self.operation: str = operation
        self.arguments: list[OCLExpression] = arguments

    def __repr__(self):
        return f'OperationCallExpression({self.operation},{self.arguments})'

    @property
    def operation(self) -> str:
        return self.__operation

    @operation.setter
    def operation(self, operation: str):
        self.__operation = operation

    @property
    def arguments(self) -> list[OCLExpression]:
        return self.__arguments

    @arguments.setter
    def arguments(self, arguments: list[OCLExpression]):
        self.__arguments = arguments

    def __str__(self) -> str:
        return f'{self.arguments[0]} {self.operation} {self.arguments[1]}'

# A class to represents OCL constriants, i.e. constraints written with the OCL language
class OCLConstraint(Constraint):
    def __init__(self, name: str, context: Class, expression: OCLExpression, language: str = "OCL"):
        super().__init__(name, context, expression, language)


class IfExp(OCLExpression):
    def __init__(self, name: str, type: Type,ifcond = None, elseExp = None, thenExp = None,):
        # self.ifOwner = null
        super().__init__(name, type)

        self.ifCondition = ifcond
        self.else_expression = elseExp
        self.then_expression = thenExp

    @property
    def get_if_Condition (self):
        return self.ifCondition

    @property
    def get_else_condition(self):
        return self.else_expression
    @property
    def get_then_expression(self):
        return self.then_expression

    @property
    def set_if_Condition (self,if_cond):
        self.ifCondition = if_cond

    @property
    def set_else_condition(self,else_cond):
        self.else_expression = else_cond
    @property
    def set_then_expression(self, then_expression):
        self.then_expression = then_expression
class VariableExp(OCLExpression):
    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

        self.variable = Variable()
    def set_refferred_variable (self,val):
        if "." in val:
            val = val.split(".")[1]
        self.variable.set_value(val)
    def getVal(self):
        return self.variable.get_value()
class Variable(OCLExpression):
    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

    def set_value(self, val):
        self.representatedParameter = Parameter(val)
    def get_value(self):
        return self.representatedParameter.get_value()

class Property:
    def __init__(self):
        self.referringExp = []
        self.val = None

class TypeExp(OCLExpression):
        def __init__(self,name: str, type: Type):
            super().__init__(name, type)

            self.referedType = Classifier()
class Parameter:
    def __init__(self,val):
        self.value =val

    def get_value(self):
        return self.value


class StateExp(OCLExpression):
        def __init__(self,name: str, type: Type):
            super().__init__(name, type)

            self.referedState = State()
class State:
    def __init__(self):
        self.stateExp = []

class RealLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: float):
        super().__init__(name, type=PrimitiveDataType(name="float"), value=value)

    def __repr__(self):
        return f'RealLiteralExpression({self.value})'


class Classifier:
    pass
class CallExp(OCLExpression):
    pass
class FeatureCallExp(CallExp):
    pass
class LiteralExp(OCLExpression):
    pass
class InvalidLiteralExp(LiteralExp):
    pass
class LoopExp(CallExp):
    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

        self.body = None#OCLExpression()
        self.iterator = None#Variable()
    @property
    def set_body(self,body):
        self.body = body
    @property
    def get_body(self):
        return self.body
    @property
    def set_iterator (self,iterator):
        self.iterator = iterator
    @property
    def get_iterator(self):
        return self.iterator

class MessageExp(OCLExpression):
    pass
class NavigationCallExp(OCLExpression):
    pass
class NullLiteralExp(LiteralExp):
    pass

class PrimitiveLiteralExp(LiteralExp):
    pass
class NumericLiteralExp(PrimitiveLiteralExp):
    pass
class IterateExp(LoopExp):
    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

        self.result=Variable()
class IteratorExp(LoopExp):
    pass
class LetExp(OCLExpression):
    def __init__(self, name: str, type: Type):
        super().__init__(name, type)
        self.OCLExpression = None
        self.variable = None
class BooleanLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: bool):
        super().__init__(name, type=PrimitiveDataType(name="Boolean"), value=value)

    def __repr__(self):
        return f'BooleanLiteralExpression({self.value})'
class StringLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: str):
        super().__init__(name, type=PrimitiveDataType(name="String"), value=value)

    def __repr__(self):
        return f'StringLiteralExpression({self.value})'