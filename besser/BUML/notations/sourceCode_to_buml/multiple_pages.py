
from besser.BUML.notations.structuralPlantUML.plantuml_to_buml import plantuml_to_buml
from besser.BUML.notations.sourceCode_to_buml.utilities.file_utils import *

# Integration modules for specific functionality
from besser.BUML.notations.sourceCode_to_structural.besser_integration.multiple_pages import \
   run_pipeline_plantuml_generation
from besser.BUML.notations.sourceCode_to_structural.besser_integration import refactor_plantuml_code
from besser.BUML.notations.sourceCode_to_buml.besser_integration \
   import refactor_gui_code
from besser.BUML.notations.sourceCode_to_buml.besser_integration.multiple_pages import (
    run_pipeline_gui_models_generation,
    run_pipeline_gui_model_generation
)
from besser.BUML.notations.sourceCode_to_buml.besser_integration import (
    svg_code_generate,
    sanitize_generated_gui_model,
    run_pipeline_svg_enhancement,
    refactor_all_in_dir
)



def source_code_to_buml_multiple_pages(api_key: str, source_code_files_path: str, navigation_image_path: str,
                                       pages_order_file_path: str, additional_text_file_path: str,
                                       output_folder: str, styling_file_path: str= None):
    """
    Main function to execute the workflow for processing source code files,
    generating PlantUML and converting to B-UML, generating GUI models and integration into
    one GUI model, and refining the final GUI model.
    """

    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")
    structural_model_path = os.path.join(output_folder, "buml", "model.py")
    svg_code_path = os.path.join(output_folder, "hci_enhanced")
    svg_code_output_path = os.path.join(svg_code_path, "enhanced_svg")


    # Step 1: Generate PlantUML code using the GPT API
    print("Step 1: Generating PlantUML code...")
    run_pipeline_plantuml_generation(api_key, source_code_files_path, output_folder, additional_text_file_path)

    # Step 2: Refine the generated PlantUML code
    print("Step 2: Refining the PlantUML code...")
    refactor_plantuml_code(output_folder)

    # Step 3: Convert PlantUML code to B-UML format
    print("Step 3: Converting PlantUML code to B-UML format...")
    plantuml_to_buml(plantUML_model_path=code_file, buml_file_path=structural_model_path)


    # Path to the folder containing the CSS file
    if styling_file_path:
        one_css_file_path = styling_file_path
        styling_files = os.listdir(one_css_file_path)

        # Find the first css file
        css_code_filename = [file for file in styling_files if
                           file.lower().endswith((".py", ".html", ".js", ".css", ".ts"))]
        print(css_code_filename)


        if css_code_filename:
        # Use the first file found
            css_file_path = os.path.join(one_css_file_path, css_code_filename[0])
        else:
            print("No code files found in the specified folder.")

    if styling_file_path:
       # Step 4: Generate the GUI model from the code file
        print("Step 4: Generating the GUI model...")
        run_pipeline_gui_models_generation(api_key, source_code_files_path, css_file_path, output_folder)

    else:
        # Step 4: Generate the GUI model from the code file
        print("Step 4: Generating the GUI model...")
        run_pipeline_gui_models_generation(api_key, source_code_files_path, None, output_folder)


    # Step 6: Generate the final integrated GUI model
    print("Step 6: Generating the final integrated GUI model...")
    run_pipeline_gui_model_generation(api_key, navigation_image_path, pages_order_file_path, output_folder)

    # Step 7: Refine the generated GUI model
    print("Step 7: Refining the GUI model...")
    refactor_gui_code(output_folder)

    # Step 8: Sanitize the GUI model before using it
    print("Step 8: Validating & cleaning GUI model args...")
    sanitize_generated_gui_model(output_folder)

    print("Step 9: Generating the SVG format by runnig the svg code generator...")
    svg_code_generate(output_folder)

    print("Step 10: enhance SVG code accordijg to the HCI and synch...")
    run_pipeline_svg_enhancement(api_key, output_folder)

    # Step 11: Refine the generated SVG code
    print("Step 11: Refining the svg code...")
    refactor_all_in_dir(svg_code_path, svg_code_output_path)

