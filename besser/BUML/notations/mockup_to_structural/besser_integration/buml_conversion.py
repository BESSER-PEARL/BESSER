
import os
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml

def convert_to_buml(output_folder: str):

    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")
    structural_model_path = os.path.join(output_folder, "buml", "model.py")

    plantuml_to_buml(plantUML_model_path=code_file, buml_file_path=structural_model_path)


