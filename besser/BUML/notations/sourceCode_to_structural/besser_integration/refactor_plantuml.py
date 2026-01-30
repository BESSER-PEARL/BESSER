import os
from besser.BUML.notations.sourceCode_to_structural.refactoring_model import RefactoredModelGenerator


def refactor_plantuml_code(output_folder: str):
    """
    Refines the generated PlantUML code by utilizing the RefactoredModelGenerator.
    """

    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")

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
    print(f"Generated PlantUML code saved to {output_dir}")

