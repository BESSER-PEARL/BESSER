# Import methods and classes
from BUML.notations.plantUML.plantuml_to_buml import plantuml_to_buml
from BUML.metamodel.structural.structural import DomainModel

# PlantUML to B-UML model
library_buml: DomainModel = plantuml_to_buml(model_path='examples/library/library.plantuml')

# Print class names
for cls in library_buml.get_classes():
    print(cls.name)