from textx import metamodel_from_file
from MyUML.core.structural.structural import DomainModel, Type, Class, Property, PrimitiveDataType, Multiplicity, Association, BinaryAssociation, Generalization, GeneralizationSet, AssociationClass

myuml_mm = metamodel_from_file('../../MyUML/notations/textx/myuml.tx', classes=[DomainModel])

hello_world_myuml_model = myuml_mm.model_from_file('hello_world.myuml')

print(hello_world_myuml_model.name)



#print("Greeting", ", ".join([to_greet.name
#                            for to_greet in hello_world_myuml_model.to_greet]))