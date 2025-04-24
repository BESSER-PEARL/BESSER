import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface


class JavaGenerator(GeneratorInterface):

    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self):
        processed_associations = []
        for class_obj in self.model.classes_sorted_by_inheritance():
            file_path = self.build_generation_path(file_name=class_obj.name+".java")
            templates_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "templates")
            env = Environment(loader=FileSystemLoader(
                templates_path), trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])
            template = env.get_template('java_template.py.j2')
            package_name = ""
            if self.output_dir is not None:
                if 'tmp' or 'AppData' in self.output_dir:
                    package_name = None
                else:
                    package_name = self.output_dir
            else:
                package_name = None
            with open(file_path, mode="w") as f:
                generated_code = template.render(class_obj=class_obj,
                                                 processed_associations=processed_associations,
                                                 package_name=package_name)
                f.write(generated_code)
                print("Code generated in the location: " + file_path)
