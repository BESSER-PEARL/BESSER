# Import methods and classes
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.metamodel.structural import DomainModel

# PlantUML to B-UML model
library_buml: DomainModel = plantuml_to_buml(plantUML_model_path='library.plantuml')

# Print class names
for cls in library_buml.get_classes():
    print(cls.name)