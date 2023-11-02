import os
from jinja2 import Environment, FileSystemLoader
from BUML.metamodel.structural.structural import DomainModel
from generators.generator_interface import GeneratorInterface

class SQLAlchemyGenerator(GeneratorInterface):
    
    TYPES = {
        "int": "Integer",
        "str": "String(100)",
        "float": "Float",
        "bool": "Boolean",
        "time": "Time",
        "date": "Date",
        "datetime": "DateTime",
    }
        
    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self):
        file_path = self.build_generation_path(file_name="sql_alchemy.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('sql_alchemy_template.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(classes=self.model.get_classes(), types=self.TYPES, associations=self.model.associations)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)