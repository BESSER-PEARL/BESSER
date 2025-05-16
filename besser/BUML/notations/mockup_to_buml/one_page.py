import os
from besser.BUML.notations.mockup_to_structural.besser_integration.one_page \
   import run_pipeline_plantuml_generation
from besser.BUML.notations.mockup_to_structural.besser_integration \
   import refactor_plantuml_code
from besser.BUML.notations.mockup_to_buml.besser_integration.one_page \
   import run_pipeline_gui_generation
from besser.BUML.notations.mockup_to_buml.besser_integration \
   import refactor_gui_code
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml

def mockup_to_buml_one_page(api_key: str, mockup_image_path: str, output_folder: str):
    """
    Main function to execute the workflow for processing a mockup image,
    generating PlantUML, converting to Structural model, and generating GUI model.
    """

    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")
    structural_model_path = os.path.join(output_folder, "buml", "model.py")
    # Path to the folder containing the mockup image
    one_mockup_folder_path = mockup_image_path

    # Get the list of files in the folder
    files = os.listdir(one_mockup_folder_path)

    # Find the first image file
    mockup_image_filename = [file for file in files
                             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    if mockup_image_filename:
       # Use the first image file found
        mockup_image_path = os.path.join(one_mockup_folder_path, mockup_image_filename[0])
    else:
       print("No image files found in the specified folder.")

    # Step 1: Generate PlantUML code using the GPT API
    print("Step 1: Generating PlantUML code...")
    run_pipeline_plantuml_generation(api_key, mockup_image_path, output_folder)

    # Step 2: Refine the generated PlantUML code
    print("Step 2: Refining the PlantUML code...")
    refactor_plantuml_code(output_folder)

    # Step 3: Convert PlantUML code to structural model
    print("Step 3: Converting PlantUML code to structural moodel...")
    plantuml_to_buml(plantUML_model_path=code_file, buml_file_path=structural_model_path)

    # Step 4: Generate the GUI model
    print("Step 4: Generating the GUI model...")
    run_pipeline_gui_generation(api_key, mockup_image_path, output_folder)

    # Step 5: Refine the generated GUI model
    print("Step 5: Refining the GUI model...")
    refactor_gui_code(output_folder)
