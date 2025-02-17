import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp

class PythonGenerator(GeneratorInterface):
    """
    PythonGenerator is a class that implements the GeneratorInterface and is responsible
    for generating the Python domain model code based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be 
            saved. Defaults to None.
    """
    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self, *args):
        """
        Generates Python domain model code based on the provided B-UML model and saves it to 
        the specified output directory.
        If the output directory was not specified, the code generated will be stored in the 
        <current directory>/output folder.

        Returns:
            None, but store the generated code as a file named classes.py 
        """
        file_path = self.build_generation_path(file_name="classes.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('python_classes_template.py.j2')
        with open(file_path, mode="w", encoding='utf-8') as f:
            generated_code = template.render(domain=self.model, sort_by_timestamp=sort_by_timestamp)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
