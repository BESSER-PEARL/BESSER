import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface

class RESTAPIGenerator(GeneratorInterface):
    """
    Rest_API_Generator is a class that implements the GeneratorInterface and is responsible for generating
    the Rest API domain model code based on the input B-UML model. This version of the generator allows
    specifying which HTTP methods (e.g., GET, POST, PATCH) should be included in the generated code. It can
    generate code for one or more specified methods, enabling more customizable API endpoint generation.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        http_methods (list): A list of strings representing the HTTP methods for which code should be generated.
                         Each element should be one of "GET", "POST", "PUT","PATCH","DELETE". This allows generating
                         only the parts of the API that are needed.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: DomainModel, http_methods: list = None, output_dir: str = None):
        super().__init__(model, output_dir)
        allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        if not http_methods:
            http_methods = allowed_methods
        else:
            http_methods = [method for method in http_methods if method in allowed_methods]
        self.http_methods = http_methods

    def generate(self):
        """
        Generates Rest API model code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named rest_api.py 
        """
        file_path = self.build_generation_path(file_name="rest_api.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path),
                           trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])
        template = env.get_template('fast_api_template.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(classes=self.model.classes_sorted_by_inheritance(), http_methods=self.http_methods)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)