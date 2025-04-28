import os
from besser.BUML.notations.mockup_to_structural.one_page import mockup_to_structural_one_page
from besser.BUML.notations.mockup_to_structural.multiple_images import mockup_to_structural_multiple_pages


def count_images(folder_path):
    """Counts the number of image files in the given folder."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif')
    return [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

def mockup_to_structural(api_key: str, input_folder: str, output_folder: str = None,
                              additional_info_path: str=None):
    """
    Main function to process mockup images and convert them to Structural model.

    - If there is **one image**, calls the **single image processing** function.
    - If there are **multiple images**, calls the **multiple images processing** function.
    """

    if not os.path.isdir(input_folder):
        print(f"Error: The specified input folder '{input_folder}' does not exist.")
        return

    # Count images
    image_files = count_images(input_folder)
    image_count = len(image_files)

    if image_count == 0:
        print("No valid images found in the folder.")
        return

    print(f"Found {image_count} image(s) in '{input_folder}'.")

    if image_count == 1:
        # Process a single image
        print("Processing a single mockup image...")

        if output_folder:
            mockup_to_structural_one_page(api_key, input_folder, output_folder)
        else:
            # Use the current directory where the script was called
            current_directory = os.getcwd()
            default_output_folder = os.path.join(current_directory, "output")
            mockup_to_structural_one_page(api_key, input_folder, default_output_folder)
    else:
        # Process multiple images
        print("Processing multiple mockup image...")
        if output_folder:
            mockup_to_structural_multiple_pages(api_key, input_folder, output_folder,
                                                additional_info_path)
        else:
            current_directory = os.getcwd()
            default_output_folder = os.path.join(current_directory, "output")
            mockup_to_structural_multiple_pages(api_key, input_folder, default_output_folder,
                                                additional_info_path)

    print("✅ Processing completed successfully!")


