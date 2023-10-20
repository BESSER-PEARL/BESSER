import os
from jinja2 import Environment, FileSystemLoader
from BUML.metamodel.structural.structural import DomainModel
from generators.generator_interface import GeneratorInterface

class Python_Generator(GeneratorInterface):
    def generate(self, buml_model: DomainModel):
        # Start generation of basic python classes
        file_name = "classes.py"
        file_path = os.path.join("generated_output", file_name)
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('python_classes_template.py')
        with open(file_path, "w") as f:
            generated_code = template.render(classes=self.model.get_classes())
            f.write(generated_code)