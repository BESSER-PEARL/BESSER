import os
from besser.BUML.notations.mockup_to_buml.refactoring_model import RefactoredModelGenerator
from besser.BUML.notations.mockup_to_buml.refactoring_model import CompeletedCodeGenerator


def refactor_gui_code(output_folder: str):

    gui_output_dir = os.path.join(output_folder, "gui_model")
    gui_output_file = os.path.join(gui_output_dir, "generated_gui_model.py")

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

