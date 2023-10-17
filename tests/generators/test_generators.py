from BUML.metamodel.structural.rules import OCLExpression, PropertyCallExpression, IntegerLiteralExpression, \
    OperationCallExpression, OCLConstraint
from BUML.metamodel.structural.structural import DomainModel, Class, Property, PrimitiveDataType, Multiplicity, Constraint
from generators.django.django_generator import DjangoGenerator


def test_django_generator_rule_based_model_with_just_classes():
    model = DomainModel(name="test_model", types = None, associations = None, packages = None, constraints = None)
    class1: Class
    attribute1: Property = Property(name="attribute1", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    attribute2: Property = Property(name="attribute2", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    class1 = Class(name="name1", attributes={attribute1, attribute2})
    model.types = {class1}
    generator = DjangoGenerator(model, "./", rule_based=True)
    generator.generate()
    assert True


def test_django_generator_rule_based_model_with_validators():
    model = DomainModel(name="test_model", types = None, associations = None, packages = None, constraints = None)
    class1: Class
    attribute1: Property = Property(name="attribute1", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    attribute2: Property = Property(name="attribute2", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    class1 = Class(name="A", attributes={attribute1, attribute2})
    attribute3: Property = Property(name="attribute3", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    class2 = Class(name="B", attributes={attribute3})

    left: OCLExpression = PropertyCallExpression(name="a1", property=attribute1)
    right: OCLExpression = IntegerLiteralExpression(name="0", value=0)
    root: OperationCallExpression = OperationCallExpression(name=">", operation=">", arguments=[left, right])

    # and we now create OCL constraint linked to class A with the root node as expression
    constraint1: Constraint = OCLConstraint(name="constraint1", context=class1, expression=root)

    model.types = {class1, class2}
    model.constraints = {constraint1}




    generator = DjangoGenerator(model, "./", rule_based=True)
    generator.generate()
    assert True


def test_django_generator_llm_based():
    model = DomainModel(name="test_model", types=None, associations=None, packages=None, constraints=None)
    class1: Class
    attribute1: Property = Property(name="id", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    attribute2: Property = Property(name="name", owner=None, property_type=PrimitiveDataType("str"),
                                    multiplicity=Multiplicity(1, 1))
    class1 = Class(name="Customer", attributes={attribute1, attribute2})
    model.tyeps = {class1}
    generator = DjangoGenerator(model, "./", rule_based=False, llm_based=True)
    generator.generate()
    assert True