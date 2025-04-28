import os
from besser.BUML.notations.mockup_to_structural.besser_integration.multiple_pages\
      import run_pipeline_plantuml_generation
from besser.BUML.notations.mockup_to_structural.besser_integration \
      import refactor_plantuml_code
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml


def mockup_to_structural_multiple_pages(api_key: str, mockup_images_path: str, output_folder: str,
                                        additional_info_path: str):
    """
    Main function to execute the workflow for processing mockup images,
    generating PlantUML and converting to Structural model.
    """
    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")
    structural_model_path = os.path.join(output_folder, "buml", "model.py")


    # Step 1: Generate PlantUML code using the GPT API
    print("Step 1: Generating PlantUML code...")
    run_pipeline_plantuml_generation(api_key, mockup_images_path, output_folder,
                                     additional_info_path)

    # Step 2: Refine the generated PlantUML code
    print("Step 2: Refining the PlantUML code...")
    refactor_plantuml_code(output_folder)

    # Step 3: Convert PlantUML code to Structural model
    print("Step 3: Converting PlantUML code to Structural model...")
    plantuml_to_buml(plantUML_model_path=code_file, buml_file_path=structural_model_path)
