import importlib.util
import os
from besser.BUML.notations.sourceCode_to_buml.besser_integration.ifml_to_svg \
      import SVGCodeGenerator


def load_gui_model_from_file(file_path: str):
    """Import the generated gui_model Python file and return the gui_model object."""
    spec = importlib.util.spec_from_file_location("generated_gui_model", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gui_model   # the actual GUIModel instance


def svg_code_generate(output_folder: str):
    """
    Generate SVG code from the GUI model.
    Args:
        output_folder (str): The output folder where the GUI model is located.
    """

    gui_output_dir = os.path.join(output_folder, "gui_model")
    gui_output_file = os.path.join(gui_output_dir, "generated_gui_model.py")

    svg_output_dir=os.path.join(output_folder, "svg")


    # Make sure the path is absolute
    gui_model_path = os.path.abspath(gui_output_file)

    # Load the actual gui_model object
    gui_model = load_gui_model_from_file(gui_model_path)

    # Create an instance of SVGCodeGenerator by providing the required arguments
    svg_generator = SVGCodeGenerator(
        gui_model=gui_model,
        output_dir=svg_output_dir
    )

    # Call the generate method to generate the GUI code
    svg_generator.generate()
    print(f"Generated svg code saved to {svg_output_dir}")

