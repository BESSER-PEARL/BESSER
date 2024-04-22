import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.pydantic_classes import Pydantic_Generator


class Backend_Generator(GeneratorInterface):
    """
    Backend_Generator is a class that implements the GeneratorInterface and is responsible for generating
    a Backend model code with a REST API using FAST API framework and SQLAlchemy based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, http_methods: list = None, output_dir: str = None):
        if output_dir is None:
            output_dir = os.getcwd()  # set to current directory if output_dir is None
        super().__init__(model, output_dir)
        allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        if not http_methods:
            http_methods = allowed_methods
        else:
            http_methods = [method for method in http_methods if method in allowed_methods]
        self.http_methods = http_methods

    def generate(self):
        """
        Generates Backend model code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output_backend
        folder.

        Returns:
            None, but store the generated code as files classes.py, rest_api.py, sql.py and sql_alchemy.py
        """
        backend_folder_path = os.path.join(self.output_dir, "output_backend")
        os.makedirs(backend_folder_path, exist_ok=True)
        print(f"Backend folder created at {backend_folder_path}")

        sql_alchemy = SQLAlchemyGenerator(model=self.model, output_dir=backend_folder_path)
        sql_alchemy.generate()

        pydantic = Pydantic_Generator(model=self.model, output_dir=backend_folder_path, backend=True)
        pydantic.generate()

        file_path = os.path.join(backend_folder_path, "main_api.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path),
                          trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])
        template = env.get_template('backend_fast_api_template.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(classes=self.model.classes_sorted_by_inheritance(),
                                             http_methods=self.http_methods)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
