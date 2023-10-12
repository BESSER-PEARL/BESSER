from typing import Any
from metamodel.structural.structural import Class, NamedElement, TypedElement, Type, DataType, Property, Constraint


class OCLExpression(TypedElement):
    def __init__(self, name: str, type: Type):
        super().__init__(name, type)

    def to_string(self) -> str:
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

    def to_string(self) -> str:
        return str(self.value)

class IntegerLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: int):
        super().__init__(name, type=DataType(name="int"), value=value)

    def __repr__(self):
        return f'IntegerLiteralExpression({self.value})'


# Expression that returns the value of a given Property
class PropertyCallExpression(OCLExpression):
    def __init__(self, name: str, property: Property):
        super().__init__(name, Type(property.type))
        self.property: Property = property

    def __repr__(self):
        return f'PropertyCallExpression({self.property.name})'

    def to_string(self) -> str:
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
        super().__init__(name, Type(DataType("bool")))  # Type for now is always boolean, it should be the return type of the operation
        self.operation: str = operation
        self.arguments: list[OCLExpression] = arguments

    def __repr__(self):
        return f'OperationCallExpression({self.operation},{self.arguments})'

    @property
    def operation(self) -> Property:
        return self.__operation

    @operation.setter
    def operation(self, operation: Property):
        self.__operation = operation

    @property
    def arguments(self) -> list[OCLExpression]:
        return self.__arguments

    @arguments.setter
    def arguments(self, arguments: list[OCLExpression]):
        self.__arguments = arguments

    def to_string(self) -> str:
        return str(self.arguments[0].to_string()+ " " + self.operation + " " + self.arguments[1].to_string())

# A class to represents OCL constriants, i.e. constraints written with the OCL language
class OCLConstraint(Constraint):
    def __init__(self, name: str, context: Class, expression: OCLExpression, language: str = "OCL"):
        super().__init__(name, context, expression, language)






