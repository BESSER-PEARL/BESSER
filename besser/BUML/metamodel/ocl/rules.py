from typing import Any
from besser.BUML.metamodel.structural import Class, NamedElement, TypedElement, Type, PrimitiveDataType, Property, Constraint


class OCLExpression(TypedElement):
    def __init__(self, name: str, type: Type):
        super().__init__(name, type)
        self._source = None
        self._referredOperation = None

    @property
    def source(self) ->Any:
        return self._source
    @source.setter
    def source(self, source)->Any:
        self._source = source
    @property
    def referredOperation(self)->Any:
        return self._referredOperation

    @referredOperation.setter
    def referredOperation(self, op):
        self._referredOperation = op



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
    def add(self,item):
        self.arguments.append(item)
    def __str__(self) -> str:
        toRet= ""
        for arg in self.arguments:
            # print(type(arg))
            toRet = toRet+"  \n "+ str(arg)
        return toRet
        # return f'{self.arguments[0]} {self.operation} {self.arguments[1]}'

# A class to represents OCL constriants, i.e. constraints written with the OCL language
class OCLConstraint(Constraint):
    def __init__(self, name: str, context: Class, expression: OCLExpression, language: str = "OCL"):
        super().__init__(name, context, expression, language)


class IfExp(OCLExpression):
    def __init__(self, name: str, type: Type,ifcond = None, elseExp = None, thenExp = None,):
        # self.ifOwner = null
        super().__init__(name, type)

        self._ifCondition = None
        self._else_expression = None
        self._then_expression = None

    @property
    def ifCondition (self) -> OCLExpression:
        return self._ifCondition

    @property
    def elseCondition(self) -> OCLExpression:
        return self._else_expression
    @property
    def thenExpression(self) ->OCLExpression:
        return self._then_expression

    @ifCondition.setter
    def ifCondition (self,if_cond):
        self._ifCondition = if_cond

    @elseCondition.setter
    def elseCondition(self,else_cond):
        self._else_expression = else_cond
    @thenExpression.setter
    def thenExpression(self, then_expression):
        self._then_expression = then_expression
class VariableExp(OCLExpression):
    def __init__(self,name: str, type: Type):
        super().__init__(name, type)
        self.name = name
        self.variable = Variable(name,type)

    def set_refferred_variable (self,val):
        val.replace('self.','')
        self.variable.set_value(val)
    def getVal(self):
        return self.variable.get_value()

    def __str__(self):
        return self.name

class Variable(OCLExpression):
    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

    def set_value(self, val):
        self.representatedParameter = Parameter(val)
    def get_value(self):
        return self.representatedParameter.get_value()

# class Property:
#     def __init__(self):
#         self.referringExp = []
#         self.val = None

class TypeExp(OCLExpression):
        def __init__(self,name: str, type: Type):
            super().__init__(name, type)

            self.referedType = Classifier(name)
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
    def __init__(self, name= None):
        self.name = name
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

        self.name = name
        self.type = type
        self.body = []
        self.iterator = []

    def __str__(self):
        stringToRet = "LoopExpression: " +str(self.name)
        for it in self.iterator:
            stringToRet = stringToRet + "\nIterator: " + str(it)
        for body in self.body:
            stringToRet = stringToRet + "\nbody: " + str(body)
        return stringToRet


    def add_body(self,body):
        self.body.append(body)
    @property
    def get_body(self):
        return self.body
    def addIterator(self,iterator):
        self.iterator.append(iterator)
    # @property
    # def set_iterator (self,iterator):
    #     self.iterator = iterator
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
    def __init__(self,name: str, type: Type):
        super().__init__(name, type)
        self.name = name
        self.type = type
    pass
    def __str__(self):
        return self.name +":" +self.type

class LetExp(OCLExpression):
    def __init__(self, name: str, type: Type):
        super().__init__(name, type)
        self.OCLExpression = None
        self.variable = None
class BooleanLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: bool):
        super().__init__(name, type=PrimitiveDataType(name="bool"), value=value)

    def __repr__(self):
        return f'BooleanLiteralExpression({self.value})'
class StringLiteralExpression(LiteralExpression):
    def __init__(self, name: str, value: str):
        super().__init__(name, type=PrimitiveDataType(name="str"), value=value)

    def __repr__(self):
        return f'StringLiteralExpression({self.value})'

class InfixOperator:
    def __init__(self,operator):
        self.operator = operator
    def get_infix_operator(self):
        return self.operator
    def __str__(self):
        return self.operator

class DataType(Classifier):
    def __init__(self,name):
        super().__init__(name)

class CollectionType(DataType):
    def __init__(self,name):
        super().__init__(name)

class OrderedSetType(CollectionType):
    def __init__(self,name):
        super().__init__(name)

class SequenceType(CollectionType):
    def __init__(self,name):
        super().__init__(name)

class BagType(CollectionType):
    def __init__(self,name):
        super().__init__(name)

class SetType(CollectionType):
    def __init__(self,name):
        super().__init__(name)


class CollectionLiteralExp(LiteralExp):
    def __init__(self, name, type):
        super().__init__(name,type)
        self.kind = type
        self.collectionItems = []
    def __str__(self):
        toRet= str(self.kind) +": "
        for item in self.collectionItems:
            toRet = toRet + str(item)
        return toRet
    def add(self, item):
        self.collectionItems.append(item)

class CollectionLiteralPart(TypedElement):
    def __init__(self, name):
        super().__init__(name,type = "NP")


class CollectionItem(CollectionLiteralPart):
    def __init__(self, name,item):
        super().__init__(name)
        self.value = item
    def set(self,  value):
        self.value = value
    def __str__(self):
        return str(self.value)+","
    def get(self):
        return self.value
class CollectionRange(CollectionLiteralPart):
    def __init__(self, name):
        super().__init__(name)
        self.first = None
        self.last = None


