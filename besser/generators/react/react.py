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
#   React Generator
##############################
class ReactGenerator(GeneratorInterface):
    """
    ReactGenerator is responsible for generating React code based on
    input B-UML and GUI models. It implements the GeneratorInterface and facilitates
    the creation of a React application structure.

    Args:
        model (DomainModel): The B-UML model representing the application's domain.
        gui_model (GUIModel): The GUI model instance containing necessary configurations.
        output_dir (str, optional): Directory where generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, gui_model: GUIModel, output_dir: str = None):
        super().__init__(model, output_dir)
        self.gui_model = gui_model
        # Jinja environment configuration with custom delimiters to avoid React/JSX conflicts
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=['jinja2.ext.do'],
            variable_start_string='[[',
            variable_end_string=']]',
        )

    def generate(self):
        """
        Generates React TS code based on the provided B-UML and GUI models.
        Generates all files from the templates directory, preserving structure and file names (removing .j2 extension).
        """
        def generate_file_from_template(template_path, rel_template_path):
            # Remove .j2 extension for output file
            if rel_template_path.endswith('.j2'):
                rel_output_path = rel_template_path[:-3]
            else:
                rel_output_path = rel_template_path
            file_path = self.build_generation_path(file_name=rel_output_path)
            print(f"Generating file: {file_path} from template: {rel_template_path}")
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                template = self.env.get_template(rel_template_path.replace("\\", "/"))
                context = {"model": self.gui_model} if "index.html" in rel_output_path else {}
                generated_code = template.render(**context)
                with open(file_path, mode="w", encoding="utf-8") as f:
                    f.write(generated_code)
                print(f"Code generated in the location: {file_path}")
            except Exception as e:
                print(f"Error generating {file_path} from {rel_template_path}: {e}")
                raise

        # Walk through the templates directory and generate all .j2 files
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        for root, _, files in os.walk(templates_path):
            for file in files:
                if file.endswith('.j2'):
                    abs_template_path = os.path.join(root, file)
                    rel_template_path = os.path.relpath(abs_template_path, templates_path)
                    generate_file_from_template(abs_template_path, rel_template_path)
