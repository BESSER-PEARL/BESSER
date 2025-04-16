import os
from besser.BUML.notations.mockup_to_structural.refactoring_model \
    import RefactoredModelGenerator


def refactor_plantuml_code(output_folder: str):
    """
    Refactors the generated PlantUML code and saves the final output.

    This function initializes an instance of `RefactoredModelGenerator` to process
    the generated PlantUML file and refine it. The output is saved in the specified
    output folder.

    Args:
        output_folder (str): The directory where the generated PlantUML files are stored.

    Returns:
        None
    """

    output_dir = os.path.join(output_folder, "plantuml")
    code_file = os.path.join(output_dir, "generated_plantuml.puml")

    # Create an instance of GUIGenerator by providing the required arguments
    gui_generator = RefactoredModelGenerator(
        output_dir=output_dir,
        output_file_name="generated_plantuml.puml",
        structure_file_path=None,  # If no structure file is needed, set this to None
        code_file=code_file,
        keyworld='plantuml'
    )

    # Call the generate method to generate the GUI code
    gui_generator.generate()
    print(f"Generated PlantUML code saved to {code_file}")
