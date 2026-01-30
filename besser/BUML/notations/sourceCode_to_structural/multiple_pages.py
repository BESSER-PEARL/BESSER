import os
from besser.BUML.notations.structuralPlantUML.plantuml_to_buml import plantuml_to_buml
from besser.BUML.notations.sourceCode_to_structural.besser_integration.multiple_pages import \
    run_pipeline_plantuml_generation

from besser.BUML.notations.sourceCode_to_structural.besser_integration import \
    refactor_plantuml_code

def source_code_base_to_structural_multiple_pages(api_key: str, source_code_files_path: str,
                                                  output_folder: str, additional_text_file_path: str):
    """
    Main function to execute the workflow for processing source code files,
    generating PlantUML and converting to B-UML, generating GUI models and integration into
    one GUI model, and refining the final GUI model.
    """


    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")
    structural_model_path = os.path.join(output_folder, "buml", "model.py")


    # Step 1: Generate PlantUML code using the GPT API
    print("Step 1: Generating PlantUML code...")
    run_pipeline_plantuml_generation(api_key, source_code_files_path, output_folder, additional_text_file_path)

    # Step 2: Refine the generated PlantUML code
    print("Step 2: Refining the PlantUML code...")
    refactor_plantuml_code(output_folder)

    # Step 3: Convert PlantUML code to B-UML format
    print("Step 3: Converting PlantUML code to B-UML format...")
    plantuml_to_buml(plantUML_model_path=code_file, buml_file_path=structural_model_path)

