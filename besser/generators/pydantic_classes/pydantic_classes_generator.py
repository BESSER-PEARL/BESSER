import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface

class Pydantic_Generator(GeneratorInterface):
    """
    Pydantic_Generator is a class that implements the GeneratorInterface and is responsible for generating
    the Python domain model code based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: DomainModel, backend: bool = False, output_dir: str = None):
        super().__init__(model, output_dir)
        self.backend = backend

    def generate(self):
        """
        Generates Python domain model code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named pydantic_classes.py 
        """
        file_path = self.build_generation_path(file_name="pydantic_classes.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path),
                          trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])
        template = env.get_template('pydantic_classes_template.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(classes=self.model.classes_sorted_by_inheritance(), backend=self.backend)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)