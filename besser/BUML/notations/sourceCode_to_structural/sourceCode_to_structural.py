import os
from besser.BUML.notations.sourceCode_to_structural.one_page import source_code_to_structural_one_page
from besser.BUML.notations.sourceCode_to_structural.multiple_pages import source_code_base_to_structural_multiple_pages


def count_pages(folder_path):
    """Counts the number of HTML files in the given folder."""
    html_extensions = ('.html', '.htm')
    return [f for f in os.listdir(folder_path) if f.lower().endswith(html_extensions)]



def source_code_to_structural(api_key: str, input_folder: str, output_folder: str = None,
                              additional_info_path: str=None):
    """
    Main function to process source code and convert it to Structural model.

    - If there is **one source code file**, calls the **single source code processing** function.
    - If there are **multiple source code files**, calls the **multiple source code processing** function.
    """

    if not os.path.isdir(input_folder):
        print(f"Error: The specified input folder '{input_folder}' does not exist.")
        return

    # Count pages (source code files) in the input folder
    code_files = count_pages(input_folder)
    pages_count = len(code_files)

    if pages_count == 0:
        print("No valid source code files found in the folder.")
        return

    print(f"Found {pages_count} source code file(s) in '{input_folder}'.")

    if pages_count == 1:
        # Process a single source code file
        print("Processing a single source code file...")

        if output_folder:
            source_code_to_structural_one_page(api_key, input_folder, output_folder)
        else:
            # Use the current directory where the script was called
            current_directory = os.getcwd()
            default_output_folder = os.path.join(current_directory, "output")
            source_code_to_structural_one_page(api_key, input_folder, default_output_folder)
    else:
        # Process multiple source code files
        print("Processing multiple source code files...")
        if output_folder:
            source_code_base_to_structural_multiple_pages(api_key, input_folder, output_folder,
                                                additional_info_path)
        else:
            current_directory = os.getcwd()
            default_output_folder = os.path.join(current_directory, "output")
            source_code_base_to_structural_multiple_pages(api_key, input_folder, default_output_folder,
                                                additional_info_path)

    print("✅ Processing completed successfully!")
