import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface


class RDFGenerator(GeneratorInterface):
    """
    RDFGenerator is a class that implements the GeneratorInterface and produces the vocabulary spec in RDF Turtle format 
    for a structural model defined with B-UML.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    TYPES = {
        "int": "integer",
        "str": "string",
        "float": "float",
        "bool": "boolean",
        "time": "time",
        "date": "date",
        "datetime": "dateTime",
        "timedelta": "duration",
    }

    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self):
        """
        Generates RDF vocabulary on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named vocabulary.ttl
        """
        file_path = self.build_generation_path(file_name="vocabulary.ttl")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True, lstrip_blocks=True)
        template = env.get_template('rdf_template.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(model=self.model, types=self.TYPES)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
