import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface

class RESTAPIGenerator(GeneratorInterface):
    """
    Rest_API_Generator is a class that implements the GeneratorInterface and is responsible for generating
    the Rest API domain model code based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

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
            generated_code = template.render(classes=self.model.classes_sorted_by_inheritance())
            f.write(generated_code)
            print("Code generated in the location: " + file_path)