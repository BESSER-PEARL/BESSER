from besser.BUML.notations.mockup_to_gui.refactoring_model.refactor_model import RefactoredModelGenerator
from besser.BUML.notations.mockup_to_gui.refactoring_model.complete_model import CompeletedCodeGenerator
from besser.BUML.notations.mockup_to_gui.config import gui_output_dir, gui_output_file
import time

def refactor_gui_code():
    # Use folder_path and single_gui_code_path from the imported configuration
    global gui_output_dir, gui_output_file  # Ensure we use the imported variables, not local ones

    # Create an instance of GUIGenerator by providing the required arguments
    gui_generator = RefactoredModelGenerator(
        output_dir=gui_output_dir,
        output_file_name="generated_gui_model.py",
        structure_file_path=None,  # If no structure file is needed, set this to None
        code_file=gui_output_file,
        keyworld='python'
    )

    # Call the generate method to generate the GUI code
    gui_generator.generate()

    # Create an instance of CompeleteGUIGenerator for generating the complete GUI code
    gui_generator = CompeletedCodeGenerator(
        output_dir=gui_output_dir,
        output_file_name="generated_gui_model.py",
        structure_file_path=None,  # If no structure file is needed, set this to None
        code_file=gui_output_file,
        keyworld='python'
    )

    # Call the generate method to generate the complete GUI code
    gui_generator.generate()

# Allow standalone execution
if __name__ == "__main__":
    refactor_gui_code()
