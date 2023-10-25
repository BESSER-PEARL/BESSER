import os
from jinja2 import Environment, FileSystemLoader
from BUML.metamodel.structural.structural import DomainModel
from generators.generator_interface import GeneratorInterface

class Python_Generator(GeneratorInterface):
    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self):
        # Start generation of basic python classes
        file_name = "classes.py"
        working_path = os.path.abspath('')
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        if self.output_dir != None:
            file_path = os.path.join(self.output_dir, file_name)
        else:
            os.makedirs(os.path.join(working_path, "output"), exist_ok=True)
            file_path = os.path.join(working_path, "output", file_name)
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('python_classes_template.py')
        with open(file_path, mode="w") as f:
            generated_code = template.render(classes=self.model.get_classes())
            f.write(generated_code)
            print("Code generated in the location: " + file_path)