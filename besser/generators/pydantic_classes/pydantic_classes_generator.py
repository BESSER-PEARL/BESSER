import os
import re
import unicodedata
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.structural_utils import get_foreign_keys
from besser.generators.pydantic_classes.ocl_utils import build_constraints_map

class PydanticGenerator(GeneratorInterface):
    """
    PydanticGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the Python domain model code based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        backend (bool, optional): A boolean flag indicating whether the generator should generate code for a backend API.
        nested_creations (bool, optional): This parameter determines how entities are linked in the API request. 
                                            If set to True, both nested creations and linking by the ID of the entity 
                                            are enabled. If set to False, only the ID of the linked entity will be used.
                                            The default value is False.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: DomainModel, backend: bool = False, nested_creations: bool = False, output_dir: str = None):
        super().__init__(model, output_dir)
        self.domain_model = model
        self.backend = backend
        self.nested_creations = nested_creations
    
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
        def ascii_identifier(name: str) -> str:
            normalized = unicodedata.normalize("NFKD", name)
            ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
            ascii_name = re.sub(r"\\W", "_", ascii_name)
            if ascii_name and ascii_name[0].isdigit():
                ascii_name = f"_{ascii_name}"
            return ascii_name

        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            extensions=['jinja2.ext.do']
        )
        env.filters["ascii_identifier"] = ascii_identifier
        template = env.get_template('pydantic_classes_template.py.j2')
        
        # Use DomainModel's built-in method to sort classes by inheritance (parents before children)
        sorted_classes = self.domain_model.classes_sorted_by_inheritance()
        
        # Build constraints map for OCL validation
        constraints_map = build_constraints_map(self.domain_model)
        
        class_names = {cls.name for cls in sorted_classes}
        with open(file_path, mode="w", newline='\n', encoding="utf-8") as f:
            generated_code = template.render(
                domain=self.domain_model,
                sorted_classes=sorted_classes,
                backend=self.backend,
                nested_creations=self.nested_creations,
                constraints_map=constraints_map,
                fkeys=get_foreign_keys(self.domain_model),
                class_names=class_names
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
