from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.metamodel.structural import DomainModel
from besser.generators.django import DjangoGenerator

# PlantUML to B-UML model
dpp_buml: DomainModel = plantuml_to_buml(plantUML_model_path='dpp.plantuml')

# Code Generation
django = DjangoGenerator(model=dpp_buml)
django.generate()