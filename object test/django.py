from besser.generators.django import DjangoGenerator
# Import methods and classes
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.metamodel.structural import DomainModel

# PlantUML to B-UML model
library_buml: DomainModel = plantuml_to_buml(plantUML_model_path='django.plantuml')


generator: DjangoGenerator = DjangoGenerator(model=library_buml)
generator.generate()