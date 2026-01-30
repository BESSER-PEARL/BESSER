import os
from besser.BUML.notations.sourceCode_to_buml.utilities import *
from besser.BUML.notations.sourceCode_to_structural.besser_integration.one_page \
    import run_pipeline_plantuml_generation
from besser.BUML.notations.sourceCode_to_buml.besser_integration.one_page \
    import run_pipeline_gui_generation
from besser.BUML.notations.sourceCode_to_structural.besser_integration \
    import refactor_plantuml_code
from besser.BUML.notations.sourceCode_to_buml.besser_integration import refactor_gui_code
from besser.BUML.notations.sourceCode_to_buml.besser_integration import refactor_svg_code
from besser.BUML.notations.sourceCode_to_buml.besser_integration import svg_code_generate
from besser.BUML.notations.sourceCode_to_buml.besser_integration import sanitize_generated_gui_model
from besser.BUML.notations.sourceCode_to_buml.besser_integration import run_pipeline_svg_enhancement
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml

def source_code_to_buml_one_page(api_key: str, code_file_path: str,
                                 styling_file_path: str= None,
                                 output_folder: str = "output"):
    """
    Main function to execute the workflow for processing a code file,
    generating PlantUML and converting to B-UML, generating GUI model.
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


   # Path to the folder containing the CSS file
    if styling_file_path:
        one_css_file_path = styling_file_path
        styling_files = os.listdir(one_css_file_path)

        # Find the first css file
        css_code_filename = [file for file in styling_files if
                             file.lower().endswith((".py", ".html", ".js", ".css", ".ts", ".scss"))]
        print(css_code_filename)


        if css_code_filename:
            # Use the first file found
            css_file_path = os.path.join(one_css_file_path, css_code_filename[0])
        else:
            print("No code files found in the specified folder.")

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


    if styling_file_path:
       # Step 5: Generate the GUI model from the code file
        print("Step 5: Generating the GUI model with css file...")
        run_pipeline_gui_generation(api_key, code_file_path, output_folder, css_file_path)

    else:
        # Step 5: Generate the GUI model from the code file
        print("Step 5: Generating the GUI model...")
        run_pipeline_gui_generation(api_key, code_file_path, output_folder)


    # Step 6: Refine the generated GUI model
    print("Step 6: Refining the GUI model...")
    refactor_gui_code(output_folder)

    # Step 7: Sanitize the GUI model before using it
    print("Step 7: Validating & cleaning GUI model args...")
    sanitize_generated_gui_model(output_folder)


    print("Step 8: Generating the SVG format by runnig the svg code generator...")
    svg_code_generate(output_folder)


    print("Step 9: enhance SVG code accordijg to the HCI...")
    svg_output_file = run_pipeline_svg_enhancement(api_key, output_folder)


    # Step 10: Refine the generated SVG code
    print("Step 10: Refining the svg code...")
    refactor_svg_code(svg_output_file, output_folder)
