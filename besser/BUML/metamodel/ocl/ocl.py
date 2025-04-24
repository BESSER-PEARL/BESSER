from typing import Any
from besser.BUML.metamodel.structural import (
    Class, TypedElement, Type, StringType,
    Property, Constraint, BooleanType, FloatType,
    IntegerType, DateType
)


class OCLExpression(TypedElement):
    """The OCLExpression is the Superclass of all OCL elements with.

    Args:
        name (str): the name of the expression
        type: type of the expression.

    Attributes:
        source (OCLExpression): the source of expression
        _referredOperation: points to any refferred operation that the expression has.
    """

    def __init__(self, name: str, type: Type):
        super().__init__(name, type)
        self._source = None
        self._referredOperation = None

    @property
    def source(self) ->Any:
        """Get the source of OCL Expression"""
        return self._source

    @source.setter
    def source(self, source)->Any:
        """Set the source of OCL Expression"""
        self._source = source

    @property
    def referredOperation(self)->Any:
        """Get the referredOperation of OCL Expression"""
        return self._referredOperation

    @referredOperation.setter
    def referredOperation(self, op):
        """Set the referredOperation of OCL Expression"""
        self._referredOperation = op

    def __str__(self) -> str:
        pass

class LiteralExpression(OCLExpression):
    """ A literal value part of an OCL expression

    Args:
        name (str): the name of the expression
        type: type of the expression.
        value: value of the expression

    Attributes:
        value: value of the expression
    """

    def __init__(self, name: str, type: Type, value: Any):
        super().__init__(name, type)
        self.value: Any = value

    @property
    def value(self) -> Any:
        """Get the Value of Expression"""
        return self.__value

    @value.setter
    def value(self, value: Any):
        """Set the Value of Expression"""
        self.__value = value

    def __str__(self) -> str:
        return str(self.value)

class IntegerLiteralExpression(LiteralExpression):
    """ A Integer literal Expression of type LiteralExpression

    Args:
        name (str): the name of the expression
        value: value of the expression
    """

    def __init__(self, name: str, value: int):
        super().__init__(name, type=IntegerType, value=value)

    def __repr__(self):
        return f'IntegerLiteralExpression({self.value})'

class PropertyCallExpression(OCLExpression):
    """ A Property Call Expression of type OCLExpression

    Args:
        name (str): the name of the expression
        property: property of the expression

    Attributes:
        property: Property of the expression
    """

    def __init__(self, name: str, property: Property):
        super().__init__(name, Type(property.type))
        self.property: Property = property

    def __repr__(self):
        return f'PropertyCallExpression({self.property.name})'

    def __str__(self) -> str:
        return str(self.property.name)

    @property
    def property(self) -> Property:
        """Get the Property of Expression"""
        return self.__property

    @property.setter
    def property(self, property: Property):
        """Set the Property of Expression"""
        self.__property = property

class OperationCallExpression(OCLExpression):
    """
    Expression that returns the value of a given Operation on a given set of ordered arguments
    The operation could also be a reference to any operation in an existing class but for now we stick to simple
    arithmetic comparison operations from the standard OCL library.

    An Operation Call Expression of type OCL expression.

    Args:
        name : Name of the function
        operation: Operation of the expression
        arguments: Arguments of the function

    Attributes:
        operation: Operation of the expression
        arguments: Arguments of the function
    """

    def __init__(self, name: str, operation: str, arguments: list[OCLExpression]):
        super().__init__(name, BooleanType)  # Type for now is always boolean, it should be the return type of the operation
        self.operation: str = operation
        self.arguments: list[OCLExpression] = arguments

    def __repr__(self):
        return f'OperationCallExpression({self.operation},{self.arguments})'

    @property
    def operation(self) -> str:
        """Get the operation of Expression"""
        return self.__operation

    @operation.setter
    def operation(self, operation: str):
        """Set the operation of Expression"""
        self.__operation = operation

    @property
    def arguments(self) -> list[OCLExpression]:
        """Get the arguments of Operation"""
        return self.__arguments

    @arguments.setter
    def arguments(self, arguments: list[OCLExpression]):
        """Set the arguments of Operation"""
        self.__arguments = arguments
    def add(self,item):
        """Add in the arguments of Operation"""
        self.arguments.append(item)
    def __str__(self) -> str:
        toRet= ""
        for arg in self.arguments:
            # print(type(arg))
            toRet = toRet+"  \n "+ str(arg)
        return toRet
        # return f'{self.arguments[0]} {self.operation} {self.arguments[1]}'

class OCLConstraint(Constraint):
    """A class to represents OCL constriants, i.e. constraints written with the OCL language

    Args:
        name: name of constraint
        context: class of constraint
        expression: expression of constraint
        language: Language of constraint
    """

    def __init__(self, name: str, context: Class, expression: OCLExpression, language: str = "OCL"):
        super().__init__(name, context, expression, language)


class IfExp(OCLExpression):
    """A class to represents IfExpressions

    Args:
        name: name of constraint
        type: type of constraint
        ifcond: If Condition of constraint
        elseExp: Else expression of constraint
        then Exp: then expression of constraint
    """

    def __init__(self, name: str, type: Type,ifcond = None, elseExp = None, thenExp = None,):
        # self.ifOwner = null
        super().__init__(name, type)

        self._ifCondition = None
        self._else_expression = None
        self._then_expression = None

    @property
    def ifCondition (self) -> OCLExpression:
        """Get the if Condition of expression"""
        return self._ifCondition

    @property
    def elseCondition(self) -> OCLExpression:
        """Get the else Condition of expression"""
        return self._else_expression

    @property
    def thenExpression(self) ->OCLExpression:
        """Get the then expression of OCLexpression"""
        return self._then_expression

    @ifCondition.setter
    def ifCondition (self,if_cond):
        """Set the if Condition of expression"""
        self._ifCondition = if_cond

    @elseCondition.setter
    def elseCondition(self,else_cond):
        """Set the else Condition of expression"""
        self._else_expression = else_cond

    @thenExpression.setter
    def thenExpression(self, then_expression):
        """Set the then expression of OCLexpression"""
        self._then_expression = then_expression

class VariableExp(OCLExpression):
    """A class to define the variable expression of the type OCL expression

    Args:
        name: Name of the expression
        type: Type of the expression

    Attributes:
        name: name of the expression
        variable: variable of the expression
    """

    def __init__(self,name: str, type: Type):
        super().__init__(name, type)
        self.name = name
        self.variable = Variable(name,type)

    def set_refferred_variable (self,val):
        """Set the value of variable in the expression"""
        val.replace('self.','')
        self.variable.set_value(val)

    def getVal(self):
        """Get the value of variable in the expression"""
        return self.variable.get_value()

    def __str__(self):
        return self.name

class Variable(OCLExpression):
    """A class to define the variable in the OCL expression

    Args:
        name: name of the expression
        type: type of the expression
    """

    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

    def set_value(self, val):
        """set the value of variable in the expression"""
        self.representatedParameter = Parameter(val)
    def get_value(self):
        """Get the value of variable in the expression"""
        return self.representatedParameter.get_value()


class TypeExp(OCLExpression):
    """ A class to define the type expression of the type OCL Expressions

    Args:
        name: name of the classifier
        type: type of expressions

    Attributes:
        referedType: classifier of the expression
    """

    def __init__(self,name: str, type: Type):
        super().__init__(name, type)
        self.referedType = Classifier(name)

class Parameter:
    """A class to define parameter

    Args:
        val: Value of the parameter

    Attributes:
        value: Value of the parameter
    """

    def __init__(self,val):
        self.value =val

    def get_value(self):
        """Get the value of parameter in the expression"""
        return self.value


class StateExp(OCLExpression):
    """A class to define the state expressions of type OCL expression

    Args:
        name: name of the expression
        type: type of the expression

    Attributes:
        ReferedState: Referred state of the expression
    """

    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

        self.referedState = State()

class State:
    """A class to define state"""

    def __init__(self):
        self.stateExp = []

class RealLiteralExpression(LiteralExpression):
    """ A Real literal Expression of type LiteralExpression

    Args:
        name (str): the name of the expression
        value: value of the expression
    """

    def __init__(self, name: str, value: float):
        super().__init__(name, type=FloatType, value=value)

    def __repr__(self):
        return f'RealLiteralExpression({self.value})'


class Classifier:
    """ A class to define classifier

    Args:
        name (str): the name of the classifier
    """

    def __init__(self, name= None):
        self.name = name
    pass

class CallExp(OCLExpression):
    """ A class to define call expression"""
    pass

class FeatureCallExp(CallExp):
    """ A class to define feature call expression"""
    pass

class LiteralExp(OCLExpression):
    """ A class to define Literal expression"""
    pass

class InvalidLiteralExp(LiteralExp):
    """ A class to define invalid literal expression"""
    pass

class LoopExp(CallExp):
    """ A class to define loop expression

    Args:
        name: name of the expression
        type: type of the expression

    Attributes:
        name: name of the expression
        type: type of the expression
        body: a list to store expressions in the body of loop expression
        iterator: list to store all the iterators
    """

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
        """Add expression to the body"""
        self.body.append(body)

    @property
    def get_body(self):
        """Get body of the expression"""
        return self.body

    def addIterator(self,iterator):
        """add iterator to the expression"""
        self.iterator.append(iterator)
    # @property
    # def set_iterator (self,iterator):
    #     self.iterator = iterator

    @property
    def get_iterator(self):
        """add iterator of the expression"""
        return self.iterator


class MessageExp(OCLExpression):
    """A class to define message expression"""
    pass

class NavigationCallExp(OCLExpression):
    """A class to define navigation call expression"""
    pass

class NullLiteralExp(LiteralExp):
    """A class to define null literal expression"""
    pass

class PrimitiveLiteralExp(LiteralExp):
    """A class to define primitive literal expression"""
    pass

class NumericLiteralExp(PrimitiveLiteralExp):
    """A class to define numeric literal expression"""
    pass

class IterateExp(LoopExp):
    """A class to define Iterate expression

    Args:
        name: name of expression
        type: type of expression

    Attributes:
        result: variable to store the result
    """

    def __init__(self,name: str, type: Type):
        super().__init__(name, type)

        self.result=Variable(name, type)

class IteratorExp(LoopExp):
    """A class to define Iterator expression

    Args:
        name: name of expression
        type: type of expression

    Attributes:
        name: name of expression
        type: type of expression
    """

    def __init__(self,name: str, type: Type):
        super().__init__(name, type)
        self.name = name
        self.type = type
    pass

    def __str__(self):
        return self.name +":" +self.type

class LetExp(OCLExpression):
    """A class to define Let expression

    Args:
        name: name of expression
        type: type of expression

    Attributes:
        OCLExpression: OCL expression of the LetExp
        variable: variable of the Let expression
    """

    def __init__(self, name: str, type: Type):
        super().__init__(name, type)
        self.OCLExpression = None
        self.variable = None

class BooleanLiteralExpression(LiteralExpression):
    """ A Boolean literal Expression of type LiteralExpression

    Args:
        name (str): the name of the expression
        value: value of the expression
    """

    def __init__(self, name: str, value: bool):
        super().__init__(name, type=BooleanType, value=value)

    def __repr__(self):
        return f'BooleanLiteralExpression({self.value})'

class DateLiteralExpression(LiteralExpression):
    """ A Date literal Expression of type LiteralExpression

    Args:
        name (str): the name of the expression
        value: value of the expression
    """

    def __init__(self, name: str, value: str):
        super().__init__(name, type=DateType, value=value)

    def __repr__(self):
        return f'DateLiteralExpression({self.value})'


class StringLiteralExpression(LiteralExpression):
    """ A String literal Expression of type LiteralExpression

    Args:
        name (str): the name of the expression
        value: value of the expression
    """

    def __init__(self, name: str, value: str):
        super().__init__(name, type=StringType, value=value)

    def __repr__(self):
        return f'StringLiteralExpression({self.value})'

class InfixOperator:
    """ A class to define Infinix Operator Expression

    Args:
        operator: Operator of the expression

    Attributes:
        operator: Operator of the expression
    """

    def __init__(self,operator):
        self.operator = operator

    def get_infix_operator(self):
        """Get the infinix Operator"""
        return self.operator

    def __str__(self):
        return self.operator

class DataType(Classifier):
    """A class to define the data type

    Args:
        name: Name of the expression
    """

    def __init__(self,name):
        super().__init__(name)

class CollectionType(DataType):
    """A class to define the Collection Type

    Args:
        name: Name of the expression
    """

    def __init__(self,name):
        super().__init__(name)

class OrderedSetType(CollectionType):
    """A class to define the Ordered Set Type

    Args:
        name: Name of the expression
    """

    def __init__(self,name):
        super().__init__(name)

class SequenceType(CollectionType):
    """A class to define the Sequence Type

    Args:
        name: Name of the expression
    """

    def __init__(self,name):
        super().__init__(name)

class BagType(CollectionType):
    """A class to define the Bag Type

    Args:
        name: Name of the expression
    """

    def __init__(self,name):
        super().__init__(name)

class SetType(CollectionType):
    """A class to define the Set Type

    Args:
        name: Name of the expression
    """

    def __init__(self,name):
        super().__init__(name)


class CollectionLiteralExp(LiteralExp):
    """A class to define the Collection Literal Expression

    Args:
        name: Name of the expression
        type: type of expression

    Attributes:
        kind: type of expression
        collectionItems: Items in the collection literal expression
    """

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
        """Method to add item in the collection items"""
        self.collectionItems.append(item)

class CollectionLiteralPart(TypedElement):
    """A class to define the Collection Literal Part

    Args:
        name: Name of the expression
    """

    def __init__(self, name):
        super().__init__(name,type = "NP")


class CollectionItem(CollectionLiteralPart):
    """A class to define the Collection Item

    Args:
        name: Name of the expression

    Attributes:
        value: Value of the item
    """

    def __init__(self, name,item):
        super().__init__(name)
        self.value = item

    def set(self,  value):
        """set the value of item"""
        self.value = value

    def __str__(self):
        return str(self.value)+","

    def get(self):
        """Get the value of item"""
        return self.value

class CollectionRange(CollectionLiteralPart):
    """A class to define collection range

    Args:
        name: name of expression

    Attributes:
        first: first item of collection
        last: last item of collection
    """

    def __init__(self, name):
        super().__init__(name)
        self.first = None
        self.last = None