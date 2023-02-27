from textx import metamodel_from_file

import sys
sys.path.append('../../')

from MyUML.core.structural.structural import DomainModel, Type, Class, Property, PrimitiveDataType, Multiplicity, \
    Association, BinaryAssociation, Generalization, GeneralizationSet, AssociationClass
from MyUML.notations.textx.textx_to_core import textx_to_core


# Testing TextX parsing of a simple domain concept
def test_textx_parsing():
    myuml_mm = metamodel_from_file('../../MyUML/notations/textx/myuml.tx')
    hello_world_myuml_model = myuml_mm.model_from_file('./hello_world.myuml')
    assert len(hello_world_myuml_model.umlElements) == 8
    # assert number of classes
    assert sum(1 if x.__class__.__name__ == 'Class' else 0 for x in hello_world_myuml_model.umlElements) == 3
    # assert number of aggregation relationships
    assert sum(1 if x.__class__.__name__ == 'Aggregation' else 0 for x in hello_world_myuml_model.umlElements) == 2
    # assert multiplicity of the BidirectionalBC relationship
    for rel in (rel for rel in hello_world_myuml_model.umlElements if rel.name=='BidirectionalBC'):
        assert rel.fromCar.min == 1
        assert rel.toCar.min == 1
        assert rel.toCar.max == '*'
    # assert visibility  of the attribute b2 of lass B
    for cl in (cl for cl in hello_world_myuml_model.umlElements if cl.name == 'B'):
        for attr in (attr for attr in cl.classContents if attr.name == 'b2'):
            assert attr.visibility == '#'

# Testing Core mdoel generation from TextX file
def test_textx_transf():
    myuml_mm = metamodel_from_file('../../MyUML/notations/textx/myuml.tx')
    hello_world_myuml_model = myuml_mm.model_from_file('./hello_world.myuml')
    domain: DomainModel = textx_to_core(hello_world_myuml_model)
    # assert number of classes
    assert sum(1 if x.__class__.__name__ == 'Class' else 0 for x in domain.types) == 3

test_textx_parsing()
test_textx_transf()