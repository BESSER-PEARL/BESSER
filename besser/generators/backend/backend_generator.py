import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.backend.docker_files import generate_docker_files


class BackendGenerator(GeneratorInterface):
    """
    BackendGenerator is a class that implements the GeneratorInterface and is responsible for generating
    a Backend model code with a REST API using FAST API framework, SQLAlchemy and a Pydantic model based on the input B-UML model .

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        http_methods (list, optional): A list of HTTP methods to be used in the REST API. Defaults to All.
        nested_creations (bool, optional): This parameter specifies how entities are linked in the API request. If set to True, the API expects 
                                identifiers and links entities based on these IDs. If set to False, the API handles the creation of 
                                new entities based on the data provided in the request. Defaults to True
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, http_methods: list = None, nested_creations: bool = False, output_dir: str = None, docker_image: bool=False):
        super().__init__(model, output_dir)
        allowed_methods = ["GET", "POST", "PUT", "DELETE"]
        if not http_methods:
            http_methods = allowed_methods
        else:
            http_methods = [method for method in http_methods if method in allowed_methods]
        self.http_methods = http_methods
        self.nested_creations = nested_creations
        self.docker_image = docker_image

    def generate(self):
        """
        Generates Backend model code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output_backend
        folder.

        Returns:
            None, but store the generated code as files main_api.py, sql_alchemy.py and pydantic_classes.py
        """
        if self.output_dir is not None:
            backend_folder_path = self.output_dir
            os.makedirs(backend_folder_path, exist_ok=True)
            print(f"Backend folder created at {backend_folder_path}")
        else:
            backend_folder_path = os.path.join(os.path.abspath(''), "output_backend")
            os.makedirs(backend_folder_path, exist_ok=True)
            print(f"Backend folder created at {backend_folder_path}")

        rest_api = RESTAPIGenerator(model=self.model, http_methods=self.http_methods, nested_creations=self.nested_creations, output_dir=backend_folder_path, backend=True)
        rest_api.generate()

        sql_alchemy = SQLAlchemyGenerator(model=self.model, output_dir=backend_folder_path)
        sql_alchemy.generate()

        pydantic_model = PydanticGenerator(model=self.model, output_dir=backend_folder_path, backend=True, nested_creations=self.nested_creations)
        pydantic_model.generate()
        
        # Generate files if Docker image is required
        if self.docker_image == True:
            generate_docker_files(backend_folder_path)