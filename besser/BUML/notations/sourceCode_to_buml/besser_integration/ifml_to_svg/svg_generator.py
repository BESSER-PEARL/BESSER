import os
from jinja2 import Environment, FileSystemLoader
from besser.generators import GeneratorInterface
from besser.BUML.metamodel.gui.graphical_ui import *


class SVGCodeGenerator(GeneratorInterface):
    """
    SVGCodeGenerator generates SVG files that represent graphical UI screens
    defined in a B-UML GUI model.

    The generator traverses the GUI model structure (modules, screens, and UI
    elements) and renders one SVG file per screen using Jinja2 templates.

    Args:
        gui_model (GUIModel): An instance of the GUI model representing the
            graphical B-UML specification.
        output_dir (str, optional): Directory where the generated SVG files
            will be written. If not provided, a default
            '<current_directory>/output/svg' directory is used.
    """

    @staticmethod
    def is_button(value):
        """Check if the given value is an instance of Button class."""
        return isinstance(value, Button)

    @staticmethod
    def is_list(value):
        """Check if the given value is an instance of DataList class."""
        return isinstance(value, DataList)

    @staticmethod
    def is_model_element(value):
        """Check if the given value is an instance of DataSourceElement class."""
        return isinstance(value, DataSourceElement)

    def __init__(self, gui_model: GUIModel, output_dir: str = None):
        super().__init__(output_dir)

        self.gui_model = gui_model


    def generate(self):
        """
        Generates SVG code based on the provided B-UML model (gui_model) and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named sql_alchemy.py
        """

        # If user didn’t pass an output dir, build a default one
        if not self.output_dir:
            self.output_dir = os.path.join(os.getcwd(), "output", "svg")

        # Now create the directory safely
        os.makedirs(self.output_dir, exist_ok=True)

        ##file_path = self.build_generation_path(file_name="page.svg")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True, lstrip_blocks=True)
        template = env.get_template('svg_code.py.j2')
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element


        if not self.gui_model.modules:
            raise ValueError("GUI model has no modules")

        module = next(iter(self.gui_model.modules))
        screens = module.screens

        for screen in screens:
            file_name = f"{screen.name}.svg"
            print(file_name)
            file_path = os.path.join(self.output_dir, file_name)
            generated_code = template.render(
                app=self.gui_model,
                screen =screen
            )
            # Write to the SVG file
            with open(file_path, mode="w", encoding="utf-8") as f:
                f.write(generated_code)


            print("SVG code generated in the location: " + file_path)









