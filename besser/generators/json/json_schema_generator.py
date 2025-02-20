import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface

class JSONSchemaGenerator(GeneratorInterface):
    """
    JSONSchemaGenerator is a class that implements the GeneratorInterface and is responsible for generating
    JSONSchema code based on B-UML models.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
        
    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)
        # Add enums to TYPES dictionary

    def generate(self):
        """
        Generates JSONSchema code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named json_schema.json
        """
        file_path = self.build_generation_path(file_name="json_schema.json")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('json_schema.json.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                classes=self.model.classes_sorted_by_inheritance(),
                enumerations = self.model.get_enumerations(),
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
