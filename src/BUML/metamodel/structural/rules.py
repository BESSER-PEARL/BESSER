from typing import Any
from BUML.metamodel.structural import Class, NamedElement, TypedElement, Type, PrimitiveDataType, Property, Constraint


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
    def __init__(self, ifcond = None, elseExp = None, thenExp = None):
        # self.ifOwner = null
        self.ifCondition = ifcond
        self.elseExpression = elseExp
        self.thenExpression = thenExp
class VariableExp(OCLExpression):
    def __init__(self):
        self.variable = Variable()
    def set_refferred_variable (self,val):
        if "." in val:
            val = val.split(".")[1]
        self.variable.set_val(val)
    def getVal(self):
        return self.variable.getVal()
class Variable(OCLExpression):
    def set_val(self,val):
        self.representatedParameter = Parameter(val)
    def getVal(self):
        return self.representatedParameter.getVal()

class Property:
    def __init__(self):
        self.referringExp = []
        self.val = None

class TypeExp(OCLExpression):
        def __init__(self):
            self.refferedType = Classifier()
class Parameter:
    def __init__(self,val):
        self.value =val

    def getVal(self):
        return self.value
class Classifier:
    pass

class StateExp(OCLExpression):
        def __init__(self):
            self.refferedState = State()
class State:
    def __init__(self):
        self.stateExp = []

class RealLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: float):
        super().__init__(name, type=PrimitiveDataType(name="float"), value=value)

    def __repr__(self):
        return f'IntegerLiteralExpression({self.value})'
class CallExp(OCLExpression):
    pass
class FeatureCallExp(CallExp):
    pass
class LiteralExp(OCLExpression):
    pass
class InvalidLiteralExp(LiteralExp):
    pass
class LoopExp(CallExp):
    def __init__(self):
        self.body = OCLExpression()
        self.iterator = Variable()
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
    def __init__(self):
        self.result=Variable()
class IteratorExp(LoopExp):
    pass
class LetExp(OCLExpression):
    def __init__(self):
        self.OCLExpression = None
        self.variable = None
class BooleanLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: bool):
        super().__init__(name, type=PrimitiveDataType(name="Boolean"), value=value)

    def __repr__(self):
        return f'IntegerLiteralExpression({self.value})'
class StringLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: str):
        super().__init__(name, type=PrimitiveDataType(name="String"), value=value)

    def __repr__(self):
        return f'IntegerLiteralExpression({self.value})'