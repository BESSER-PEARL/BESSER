from textx import metamodel_from_file
from MyUML.core.structural.structural import DomainModel, Type, Class, Property, PrimitiveDataType, Multiplicity, \
    Association, BinaryAssociation, Generalization, GeneralizationSet, AssociationClass
from notations.textx.textx_to_core import textx_to_core


# Testing TextX parsing of a simple domain concept
def test_textx_parsing():
    myuml_mm = metamodel_from_file('../MyUML/notations/textx/myuml.tx')
    hello_world_myuml_model = myuml_mm.model_from_file('./notations/hello_world.myuml')
    print(hello_world_myuml_model.name)
    assert hello_world_myuml_model.name == "MyFirstModel"
    print(hello_world_myuml_model.classes)
    # assert size of classes list is 2
    assert len(hello_world_myuml_model.classes) == 2
    assert hello_world_myuml_model.classes[0].name == "A"


# Testing Core mdoel generation from TextX file
def test_textx_transf():
    myuml_mm = metamodel_from_file('../MyUML/notations/textx/myuml.tx')
    hello_world_myuml_model = myuml_mm.model_from_file('./notations/hello_world.myuml')
    domain: DomainModel = textx_to_core(hello_world_myuml_model)
    assert domain.name == "MyFirstModel"
    assert len(domain.elements) == 2
    # assert an element of the elements set is named A via transforming first the set into a list
    assert list(domain.elements)[0].name == "A" or list(domain.elements)[1].name == "A"

