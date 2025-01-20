
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.mockup_to_gui.config import code_file, structural_model_path

def convert_to_buml(): 
    global code_file, structural_model_path  # Ensure we use the imported variables, not local ones
    buml_model: DomainModel = plantuml_to_buml(plantUML_model_path=code_file, buml_file_path=structural_model_path)


