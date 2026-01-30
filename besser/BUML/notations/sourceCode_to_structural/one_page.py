
import os
from besser.BUML.notations.sourceCode_to_structural.utilities import read_file_contents
from besser.BUML.notations.sourceCode_to_structural.besser_integration.one_page import \
    run_pipeline_plantuml_generation
from besser.BUML.notations.sourceCode_to_structural.besser_integration import \
    refactor_plantuml_code
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml


def source_code_to_structural_one_page(api_key: str, code_file_path: str, output_folder: str):
    """
    Main function to execute the workflow for processing a code file,
    generating PlantUML and converting to Structural model.
    """
    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")
    structural_model_path = os.path.join(output_folder, "buml", "model.py")

    # Path to the folder containing the source code file
    one_code_folder_path = code_file_path

    # Get the list of source files in the folder
    files = os.listdir(one_code_folder_path)

    # Find the first source file
    source_code_filename = [file for file in files if file.lower().endswith((".py", ".html", ".js", ".css", ".ts"))]
    print(source_code_filename)


    # Use the first file found
    code_file_path = os.path.join(one_code_folder_path, source_code_filename[0])
    print(f"Selected source file: {code_file_path}")


    # Step 1: Read the code file
    print("Step 1: Reading the code file...")
    source_code_content = read_file_contents(code_file_path)


    # Step 2: Generate PlantUML code using the GPT API
    print("Step 2: Generating PlantUML code...")
    run_pipeline_plantuml_generation(api_key, source_code_content, output_folder)

    # Step 3: Refine the generated PlantUML code
    print("Step 3: Refining the PlantUML code...")
    refactor_plantuml_code(output_folder)

    # Step 4: Convert PlantUML code to Structural format
    print("Step 4: Converting PlantUML code to B-UML format...")
    plantuml_to_buml(plantUML_model_path=code_file, buml_file_path=structural_model_path)
