import os
import subprocess
import sys
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.gui import GUIModel, Module, Button, DataList, DataSourceElement
from besser.BUML.metamodel.structural import DomainModel, PrimitiveDataType, Enumeration
from besser.generators.backend import BackendGenerator
from besser.generators.react import ReactGenerator
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp

##############################
#   Web Application Generator
##############################
class WebAppGenerator(GeneratorInterface):
    """
    WebAppGenerator is responsible for generating a web application structure based on
    input B-UML and GUI models. It implements the GeneratorInterface and facilitates
    the creation of a web application.

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
        Generates web application code based on the provided B-UML and GUI models.

        Returns:
            None, but store the generated code in the specified output directory.
        """
        self._generate_frontend(self.env)
        self._generate_backend(self.env)
        self._generate_docker_files(self.env)

    def _generate_frontend(self, env):
        # Generate frontend code in 'frontend' subfolder
        frontend_dir = os.path.join(self.output_dir, "frontend") if self.output_dir else "frontend"
        frontend_gen = ReactGenerator(self.model, self.gui_model, output_dir=frontend_dir)
        frontend_gen.generate()

    def _generate_backend(self, env):
        # Generate backend code in 'backend' subfolder
        backend_dir = os.path.join(self.output_dir, "backend") if self.output_dir else "backend"
        backend_gen = BackendGenerator(self.model, output_dir=backend_dir)
        backend_gen.generate()

    def _generate_docker_files(self, env):
        """
        Generates Docker-related files for deployment.
        
        Args:
            env: The Jinja2 environment for template rendering.
        """
        # Generate docker-compose.yml
        docker_compose_template = env.get_template('docker-compose.yml.j2')
        docker_compose_path = os.path.join(self.output_dir, 'docker-compose.yml')
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_template.render())

        # Generate frontend Dockerfile
        frontend_dockerfile_template = env.get_template('frontend.Dockerfile.j2')
        frontend_dockerfile_path = os.path.join(self.output_dir, 'frontend', 'Dockerfile')
        os.makedirs(os.path.dirname(frontend_dockerfile_path), exist_ok=True)
        with open(frontend_dockerfile_path, 'w') as f:
            f.write(frontend_dockerfile_template.render())

        # Generate backend Dockerfile
        backend_dockerfile_template = env.get_template('backend.Dockerfile.j2')
        backend_dockerfile_path = os.path.join(self.output_dir, 'backend', 'Dockerfile')
        os.makedirs(os.path.dirname(backend_dockerfile_path), exist_ok=True)
        with open(backend_dockerfile_path, 'w') as f:
            f.write(backend_dockerfile_template.render())
