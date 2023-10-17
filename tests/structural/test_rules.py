from BUML.metamodel.structural.rules import Constraint, OCLConstraint, OCLExpression, IntegerLiteralExpression, PropertyCallExpression, OperationCallExpression
from BUML.metamodel.structural.structural import Class, Property, PrimitiveDataType


# Test I can create a class and assign an empty constraint to it
def test_simple_constraint():
    class1: Class = Class(name="name1", attributes=None)
    constraint1: Constraint = Constraint(name="constraint1", context=class1, expression="expression1", language="none")
    assert constraint1.name == "constraint1"
    assert constraint1.expression == "expression1"
    assert constraint1.context == class1


# Test I can create a class and assign an empty OCL constraint to it
def test_simple_ocl_constraint():
    class1: Class = Class(name="name1", attributes=None)
    constraint1: Constraint = OCLConstraint(name="constraint1", context=class1, expression="expression1")
    assert constraint1.name == "constraint1"
    assert constraint1.expression == "expression1"
    assert constraint1.context == class1
    assert constraint1.language == "OCL"


# Test I can create an OCL constraint checking that the value of the integer property a1 of class A is greater than 0
def test_oclexpression_property_arithmetic_comparison_constraint():
    #We first create a class A with a Property a1 of type Integer
    classA: Class = Class(name="A", attributes=None)
    propertyA1: Property = Property(name="a1", owner=classA, property_type=PrimitiveDataType(name="int"))
    classA.add_attribute(propertyA1)

    # We then create an OCL constraint linked to classA
    # To do so, we first create an OCL OperationCallExpression representing the '>' operation
    left: OCLExpression = PropertyCallExpression(name="a1", property=propertyA1)
    right: OCLExpression = IntegerLiteralExpression(name="0", value=0)
    root: OperationCallExpression = OperationCallExpression(name=">", operation=">", arguments=[left,right])

    # and we now create OCL constraint linked to class A with the root node as expression
    constraint1: Constraint = OCLConstraint(name="constraint1", context=classA, expression=root)

    assert constraint1.name == "constraint1"
    assert constraint1.expression == root
    assert constraint1.expression.operation == ">"
    assert constraint1.expression.arguments[0].property == propertyA1
    print(constraint1)




