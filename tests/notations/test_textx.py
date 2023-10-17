from textx import metamodel_from_file

from BUML.metamodel.structural.structural import DomainModel, Type, Class, Property, PrimitiveDataType, Multiplicity, \
    Association, BinaryAssociation, Generalization, GeneralizationSet, AssociationClass
from BUML.notations.textx.textx_to_buml import textx_to_buml, build_buml_mm_from_grammar


# Testing TextX parsing of a simple domain concept
def test_textx_parsing():
    buml_mm = build_buml_mm_from_grammar()
    hello_world_buml_model = buml_mm.model_from_file('tests/notations/hello_world.buml')
    assert len(hello_world_buml_model.elements) == 10
    # assert number of classes
    assert sum(1 if x.__class__.__name__ == 'Class' else 0 for x in hello_world_buml_model.elements) == 4
    # assert number of aggregation relationships
    assert sum(1 if x.__class__.__name__ == 'Aggregation' else 0 for x in hello_world_buml_model.elements) == 1
    # assert multiplicity of the BidirectionalDC relationship
    for rel in (rel for rel in hello_world_buml_model.elements if rel.__class__.__name__ == "Bidirectional"):
        if rel.name == "BidirectionalDC":
            assert rel.fromCar.min == 1
            assert rel.toCar.min == 1
            assert rel.toCar.max == '*'
    # assert visibility  of the attribute b2 of class B
    for cl in (cl for cl in hello_world_buml_model.elements if cl.__class__.__name__ == 'Class'):
        if cl == "B":
            for attr in (attr for attr in cl.classContents if attr.name == 'b2'):
                assert attr.visibility == '#'

# Testing Core model generation from TextX file
def test_textx_transf():
    buml_mm = build_buml_mm_from_grammar()
    hello_world_buml_model = buml_mm.model_from_file('tests/notations/hello_world.buml')
    domain: DomainModel = textx_to_buml(hello_world_buml_model)
    # assert number of classes
    assert len(domain.get_classes()) == 4
    # assert number of aggregation associations
    assert len(domain.associations) == 3
    # assert that the CompositionAC relationship is composite
    for rel in domain.associations:
        if rel.name == "CompositionAC":
            assert list(rel.ends)[0].is_composite == True or list(rel.ends)[1].is_composite == True
    # assert number of generalizations
    assert len(domain.generalizations) == 2
    # assert number of constraints
    assert len(domain.constraints) == 2

test_textx_transf()