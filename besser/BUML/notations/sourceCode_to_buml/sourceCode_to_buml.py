import os
from besser.BUML.notations.sourceCode_to_buml.one_page import source_code_to_buml_one_page
from besser.BUML.notations.sourceCode_to_buml.multiple_pages import source_code_to_buml_multiple_pages


def count_pages(folder_path):
    """Counts the number of HTML files in the given folder."""
    html_extensions = ('.html', '.htm')
    return [f for f in os.listdir(folder_path) if f.lower().endswith(html_extensions)]


def source_code_to_buml(api_key: str, input_folder: str, navigation_image_path: str=None, styling_file_path: str=None,
                        pages_order_file_path: str=None, additional_info_path: str=None,
                        output_folder: str = None):

    """
    Main function to process source code files and convert them to B-UML model.

    - If there is **one page**, calls the **single page processing** function.
    - If there are **multiple pages**, calls the **multiple pages processing** function.
    """

    if not os.path.isdir(input_folder):
        print(f"Error: The specified input folder '{input_folder}' does not exist.")
        return

    # Count pages (source code files) in the input folder
    code_files = count_pages(input_folder)
    pages_count = len(code_files)


    if pages_count == 0:
        print("No valid pages found in the folder.")
        return

    print(f"Found {pages_count} page(s) in '{input_folder}'.")
    if pages_count == 1:
        # Process a single page
        print("Processing a single source code file...")

        if output_folder:
            source_code_to_buml_one_page(api_key, input_folder, styling_file_path, output_folder)
        else:
            # Use the current directory where the script was called
            current_directory = os.getcwd()
            default_output_folder = os.path.join(current_directory, "output")
            source_code_to_buml_one_page(api_key, input_folder, styling_file_path, default_output_folder)
    else:
        # Process multiple source code files
        print("Processing multiple source code files...")
        if output_folder:
            source_code_to_buml_multiple_pages(api_key, input_folder, navigation_image_path,
                                          pages_order_file_path, additional_info_path,
                                          output_folder, styling_file_path)
        else:
            current_directory = os.getcwd()
            default_output_folder = os.path.join(current_directory, "output")
            source_code_to_buml_multiple_pages(api_key, input_folder, navigation_image_path,
                                          pages_order_file_path, additional_info_path,
                                          default_output_folder, styling_file_path)

    print("✅ Processing completed successfully!")
