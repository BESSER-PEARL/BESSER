from besser.BUML.notations.mockup_to_buml.refactoring_model.refactor_model import RefactoredModelGenerator
from besser.BUML.notations.mockup_to_buml.config import output_dir, code_file

def refactor_plantuml_code():
    # Use code_file and output_dir from the imported configuration
    global output_dir, code_file  # Ensure we use the imported variables, not local ones

    # Create an instance of GUIGenerator by providing the required arguments
    gui_generator = RefactoredModelGenerator(
        output_dir=output_dir,
        output_file_name="generated_plantuml.puml",  # This can also be imported if defined in the config
        structure_file_path=None,  # If no structure file is needed, set this to None
        code_file=code_file,
        keyworld='plantuml'
    )

    # Call the generate method to generate the GUI code
    gui_generator.generate()
    print(f"Generated final plantuml code saved to {output_dir}")

# Allow standalone execution
if __name__ == "__main__":
    refactor_plantuml_code()
