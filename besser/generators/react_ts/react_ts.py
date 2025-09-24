"""
This module generates Django code using Jinja2 templates based on BUML models.
"""
import os
import subprocess
import sys
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.gui import GUIModel, Module, Button, DataList, DataSourceElement
from besser.BUML.metamodel.structural import DomainModel, PrimitiveDataType, Enumeration
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp

##############################
#   React TypeScript Generator
##############################
class ReactTSGenerator(GeneratorInterface):
    """
    ReactTS is responsible for generating React TypeScript code based on
    input B-UML and GUI models. It implements the GeneratorInterface and facilitates
    the creation of a React TypeScript application structure.

    Args:
        model (DomainModel): The B-UML model representing the application's domain.
        gui_model (GUIModel): The GUI model instance containing necessary configurations.
        output_dir (str, optional): Directory where generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, gui_model: GUIModel, output_dir: str = None):
        super().__init__(model, output_dir)
        self.gui_model = gui_model
        # Jinja environment configuration
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True,
                               lstrip_blocks=True, extensions=['jinja2.ext.do'])

    def generate(self):
        """
        Generates React TS code based on the provided B-UML and GUI models.

        Returns:
            None, but store the generated code as a file named vocabulary.ttl
        """
        self._generate_frontend(self.env)
        self._generate_server(self.env)

    def _generate_frontend(self, env):
        # Helper function to generate a file from a template
        def generate_file(file_name, template_name, context=None):
            file_path = self.build_generation_path(file_name=file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            template = env.get_template(template_name)
            generated_code = template.render(**(context or {}))
            with open(file_path, mode="w", encoding="utf-8") as f:
                f.write(generated_code)
            print("Code generated in the location: " + file_path)

        # List of files to generate
        files_to_generate = [
            ("frontend/tsconfig.json", "frontend/tsconfig.json.j2", None),
            ("frontend/package.json", "frontend/package.json.j2", None),
            ("frontend/vite.config.ts", "frontend/vite.config.ts.j2", None),
            ("frontend/index.html", "frontend/index.html.j2", {"model": self.gui_model}),
            ("frontend/src/types/chroma-js.d.ts", "frontend/src/types/chroma-js.d.ts.j2", None),
            ("frontend/src/main.tsx", "frontend/src/main.tsx.j2", None),
            ("frontend/src/components/LineChart.tsx",
             "frontend/src/components/LineChart.tsx.j2", None),
            ("frontend/src/components/BarChart.tsx",
             "frontend/src/components/BarChart.tsx.j2", None),
            ("frontend/src/components/PieChart.tsx",
             "frontend/src/components/PieChart.tsx.j2", None),
            ("frontend/src/components/RadarChart.tsx",
             "frontend/src/components/RadarChart.tsx.j2", None),
            ("frontend/src/components/RadialBarChart.tsx",
             "frontend/src/components/RadialBarChart.tsx.j2", None),
        ]

        for file_name, template_name, context in files_to_generate:
            generate_file(file_name, template_name, context)

        # Generate App.tsx
        file_path = self.build_generation_path(file_name="frontend/src/App.tsx")
        template = env.get_template('frontend/src/App.tsx.j2')
        with open(file_path, mode="w", encoding="utf-8") as f:
            # Get the module of the GUI model
            module = next(iter(self.gui_model.modules))
            # Get the dashboard screen of that module
            screen = next(iter(module.screens))
            # Access the view elements
            view_elements = screen.view_elements
            generated_code = template.render(view_elements=view_elements,
                                             px_to_percent=px_to_percent_relative)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)

    def _generate_server(self, env):
        file_path = self.build_generation_path(file_name="server/server.js")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        template = env.get_template('server/server.js.j2')
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(data_model=self.model)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)

        file_path = self.build_generation_path(file_name="server/package.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        template = env.get_template('server/package.json.j2')
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render()
            f.write(generated_code)
            print("Code generated in the location: " + file_path)

def px_to_percent_relative(view_element):
    """
    Converts pixel-based position and size of a view element 
    to percentage-based integer values.

    Args:
        view_element (ViewElement): The view element to convert.

    Returns:
        dict: A dictionary containing the percentage-based position and size values (integers).
    """
    container_width = view_element.owner.styling.size.width if view_element.owner else 1
    container_height = view_element.owner.styling.size.height if view_element.owner else 1

    return {
        "x": int((view_element.styling.position.left / container_width) * 100),
        "y": int((view_element.styling.position.top / container_height) * 100),
        "width": int((view_element.styling.size.width / container_width) * 100),
        "height": int((view_element.styling.size.height / container_height) * 100),
    }
