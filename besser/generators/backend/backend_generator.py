import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.pydantic_classes import Pydantic_Generator


class Backend_Generator(GeneratorInterface):
    """
    Backend_Generator is a class that implements the GeneratorInterface and is responsible for generating
    a Backend model code with a REST API using FAST API framework, SQLAlchemy and a Pydantic model based on the input B-UML model .

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        http_methods (list, optional): A list of HTTP methods to be used in the REST API. Defaults to None.
        by_id (bool, optional): This parameter specifies how entities are linked in the API request. If set to True, the API expects 
                                identifiers and links entities based on these IDs. If set to False, the API handles the creation of 
                                new entities based on the data provided in the request. Defaults to True
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, http_methods: list = None, by_id: bool = True, output_dir: str = None):
        if output_dir is None:
            output_dir = os.getcwd()  # set to current directory if output_dir is None
        super().__init__(model, output_dir)
        allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        if not http_methods:
            http_methods = allowed_methods
        else:
            http_methods = [method for method in http_methods if method in allowed_methods]
        self.http_methods = http_methods
        self.by_id = by_id

    def generate(self):
        """
        Generates Backend model code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output_backend
        folder.

        Returns:
            None, but store the generated code as files main_api.py, sql_alchemy.py and pydantic_classes.py
        """
        backend_folder_path = os.path.join(self.output_dir, "output_backend")
        os.makedirs(backend_folder_path, exist_ok=True)
        print(f"Backend folder created at {backend_folder_path}")

        rest_api = RESTAPIGenerator(model=self.model, http_methods=self.http_methods, by_id=self.by_id, output_dir=backend_folder_path, backend=True)
        rest_api.generate()
        
        sql_alchemy = SQLAlchemyGenerator(model=self.model, output_dir=backend_folder_path)
        sql_alchemy.generate()