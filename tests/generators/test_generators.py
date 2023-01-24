from core.structural.structural import DomainModel, Class, Property, PrimitiveDataType, Multiplicity
from generators.django.django_generator import DjangoGenerator


def test_django_generator_rule_based():
    model = DomainModel(name="test_model")
    class1: Class
    attribute1: Property = Property(name="attribute1", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    attribute2: Property = Property(name="attribute2", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    class1 = Class(name="name1", attributes={attribute1, attribute2})
    model.elements = {class1}
    generator = DjangoGenerator(model, "./", rule_based=True)
    generator.generate()
    assert True


def test_django_generator_llm_based():
    model = DomainModel(name="test_model")
    class1: Class
    attribute1: Property = Property(name="id", owner=None, property_type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(1, 1))
    attribute2: Property = Property(name="name", owner=None, property_type=PrimitiveDataType("str"),
                                    multiplicity=Multiplicity(1, 1))
    class1 = Class(name="Customer", attributes={attribute1, attribute2})
    model.elements = {class1}
    generator = DjangoGenerator(model, "./", rule_based=False, llm_based=True)
    generator.generate()
    assert True