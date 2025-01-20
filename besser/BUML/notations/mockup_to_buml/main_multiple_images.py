
from besser.BUML.notations.mockup_to_buml.config import *
from besser.BUML.notations.mockup_to_buml.utilities.image_utils import encode_image
from besser.BUML.notations.mockup_to_buml.utilities.file_utils import *
from besser.BUML.notations.mockup_to_buml.besser_integration.multiple_pages.plantuml_generation import run_pipeline_plantuml_generation
from besser.BUML.notations.mockup_to_buml.besser_integration.refactor_plantuml import refactor_plantuml_code
from besser.BUML.notations.mockup_to_buml.besser_integration.buml_conversion import convert_to_buml


def mockups_to_buml(mockup_images_path: str, additional_text_file_path: str):
    """
    Main function to execute the workflow for processing mockup images,
    generating PlantUML and converting to B-UML, generating GUI models and integration into one GUI model, and refining the final GUI model.
    """

    # Step 1: Encode the mockup images
    print("Step 1: Encoding the mockup images...")
    
    # List all image files in the directory
    image_files = [f for f in os.listdir(mockup_images_path) if os.path.isfile(os.path.join(mockup_images_path, f))]
    
    # Process each image in the directory
    for image_file in image_files:
        image_path = os.path.join(mockup_images_path, image_file)
        base64_image = encode_image(image_path)  # Pass each image file path to the encode_image function
        print(f"Encoded image: {image_file}")

    # Step 2: Generate PlantUML code using the GPT API
    print("Step 2: Generating PlantUML code...")
    run_pipeline_plantuml_generation(mockup_images_path, additional_text_file_path)

    # Step 3: Refine the generated PlantUML code
    print("Step 3: Refining the PlantUML code...")
    refactor_plantuml_code()

    # Step 4: Convert PlantUML code to B-UML format
    print("Step 4: Converting PlantUML code to B-UML format...")
    convert_to_buml()


if __name__ == "__main__":
    mockups_to_buml()




