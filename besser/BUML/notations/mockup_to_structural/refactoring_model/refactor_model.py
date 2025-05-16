import os
from besser.BUML.metamodel.gui import *
from besser.BUML.metamodel.structural import *
from jinja2 import Environment, FileSystemLoader
from besser.generators import GeneratorInterface


##############################
#    Models Generator
##############################
class RefactoredModelGenerator(GeneratorInterface):
    """
    GUIGenerator is a class that implements the GeneratorInterface and is responsible for refactoring
    the GUI code based on the input B-UML model and Python code.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
        structure_file (str): Path to the structure.py file.
        python_code (str): The Python code to be included.
    """

    def __init__(self, output_dir: str = None, output_file_name: str = None, structure_file_path: str = None, code_file: str = None, keyworld: str=None):
        super().__init__(output_dir)
        # Set default directory to 'output/app_files' if not specified
        self.output_dir = output_dir
        self.output_file_name = output_file_name
        self.structure_file_path = structure_file_path
        self.code_file = code_file
        self.keyworld = keyworld

    def generate(self):
        """
        Generates GUI code based on the provided python code saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named generated_gui_model.py.
        """
        os.makedirs(self.output_dir, exist_ok=True)



        file_path = os.path.join(self.output_dir, self.output_file_name)
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "template")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('refactor_code.py.j2')

        with open(self.code_file, 'r') as f:
           code_contnet= f.read()


        with open(file_path, mode="w") as f:
            generated_code = template.render(
                output_file_name= self.output_file_name,
                structure_file_path=self.structure_file_path,
                code_content=code_contnet,
                keyworld= self.keyworld
                )
            f.write(generated_code)



