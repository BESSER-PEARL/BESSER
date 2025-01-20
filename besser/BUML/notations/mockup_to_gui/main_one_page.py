
from besser.BUML.notations.mockup_to_gui.config import *
from besser.BUML.notations.mockup_to_gui.utilities.image_utils import encode_image
from besser.BUML.notations.mockup_to_gui.utilities.file_utils import *

# Integration modules for specific functionality
from besser.BUML.notations.mockup_to_gui.besser_integration.one_page.plantuml_generation import run_pipeline_plantuml_generation
from besser.BUML.notations.mockup_to_gui.besser_integration.refactor_plantuml import refactor_plantuml_code
from besser.BUML.notations.mockup_to_gui.besser_integration.refactor_gui_model import refactor_gui_code
from besser.BUML.notations.mockup_to_gui.besser_integration.buml_conversion import convert_to_buml
from besser.BUML.notations.mockup_to_gui.besser_integration.one_page.gui_generation import run_pipeline_gui_generation


def mockup_to_gui(mockup_image_path: str):
    """
    Main function to execute the workflow for processing a mockup image,
    generating PlantUML and converting to B-UML, generating GUI model.
    """

    #form this
    # Path to the folder containing the mockup image  
    one_mockup_folder_path = mockup_image_path  

    # Get the list of files in the folder  
    files = os.listdir(one_mockup_folder_path)  

    # Find the first image file 
    mockup_image_filename = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]  

    if mockup_image_filename:  
       # Use the first image file found  
       mockup_image_path = os.path.join(one_mockup_folder_path, mockup_image_filename[0])   
    else:  
       print("No image files found in the specified folder.") 


    # Step 1: Encode the mockup image
    print("Step 1: Encoding the mockup image...")
    base64_image = encode_image(mockup_image_path)

    # Step 2: Generate PlantUML code using the GPT API
    print("Step 2: Generating PlantUML code...")
    run_pipeline_plantuml_generation(mockup_image_path)

    # Step 3: Refine the generated PlantUML code
    print("Step 3: Refining the PlantUML code...")
    refactor_plantuml_code()

    # Step 4: Convert PlantUML code to B-UML format
    print("Step 4: Converting PlantUML code to B-UML format...")
    convert_to_buml()

    # Step 5: Generate the GUI model from the B-UML
    print("Step 5: Generating the GUI model...")
    run_pipeline_gui_generation(mockup_image_path)

    # Step 6: Refine the generated GUI model
    print("Step 6: Refining the GUI model...")
    refactor_gui_code()


if __name__ == "__main__":
    mockup_to_gui()
