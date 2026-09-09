import os

from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.action_language.ActionLanguageASTBuilder import parse_bal
from besser.generators import GeneratorInterface
from besser.generators.structural_utils import normalize_method_code
from besser.generators.action_language.RESTGenerator import bal_to_rest
from besser.generators.pydantic_classes import PydanticGenerator

class RESTAPIGenerator(GeneratorInterface):
    """
    Rest_API_Generator is a class that implements the GeneratorInterface and is responsible for generating
    the Rest API domain model code based on the input B-UML model. This version of the generator allows
    specifying which HTTP methods (e.g., GET, POST, PATCH) should be included in the generated code. It can
    generate code for one or more specified methods, enabling more customizable API endpoint generation.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        http_methods (list): A list of strings representing the HTTP methods for which code should be generated.
                         Each element should be one of "GET", "POST", "PUT","PATCH","DELETE". This allows generating
                         only the parts of the API that are needed.
        backend (bool, optional): A boolean flag indicating whether the generator should generate code for a backend API.
        nested_creations (bool, optional): This parameter determines how entities are linked in the API request.
                                            If set to True, both nested creations and linking by the ID of the entity
                                            are enabled. If set to False, only the ID of the linked entity will be used.
                                            The default value is False.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: DomainModel, http_methods: list = None, nested_creations: bool = False, backend: bool = False, port: int = None, output_dir: str = None):
        super().__init__(model, output_dir)
        allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        if not http_methods:
            http_methods = allowed_methods
        else:
            invalid_methods = [method for method in http_methods if method not in allowed_methods]
            if invalid_methods:
                import logging
                logging.warning(f"Invalid HTTP methods ignored: {invalid_methods}. Allowed methods are: {allowed_methods}")
            http_methods = [method for method in http_methods if method in allowed_methods]
        self.http_methods = http_methods
        self.backend = backend
        self.nested_creations = nested_creations
        self.port = port

    def generate_requirements(self):
        """
        Generates requirements.txt file with necessary dependencies
        """
        requirements = [
            "fastapi>=0.103.0",
            "uvicorn>=0.15.0",
            "pydantic>=2.0.0",
            "typing-extensions>=4.6.0",
            "sqlalchemy>=2.0.0",
            "python-multipart>=0.0.6"
        ]

        file_path = self.build_generation_path(file_name="requirements.txt")
        with open(file_path, "w") as f:
            f.write("\n".join(requirements))

    def generate(self):
        """
        Generates Rest API model code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named rest_api.py and uses the Pydantic_Generator to generate
            the Pydantic classes
        """
        # Generate requirements.txt first
        self.generate_requirements()

        # Custom Jinja filter to extract clean method name (remove parameters if included)
        def clean_method_name(name):
            """Extract just the method name without parameters."""
            if '(' in str(name):
                return str(name).split('(')[0].strip()
            return str(name).strip()

        if self.backend:
            # The FastAPI application layer is produced by the modular per-file
            # renderer shared with BackendGenerator (a slim main_api.py +
            # routers/<class>.py + database.py + bal_stdlib.py) — the single
            # source of truth that replaced the retired monolithic main_api.py
            # template, so the association-class / OCL / method-normalization
            # logic can never drift between two generators again. Only the API
            # layer is rendered here: sql_alchemy.py and pydantic_classes.py
            # remain the caller's responsibility (as they always were for
            # backend=True), so this generator never overwrites files it does
            # not own. BackendGenerator is the orchestrator that adds them.
            # Imported lazily to avoid a circular import (BackendGenerator
            # reuses RESTAPIGenerator.generate_requirements()).
            from besser.generators.backend.api_generator import generate_modular_api
            api_output_dir = os.path.dirname(self.build_generation_path(file_name="main_api.py"))
            generate_modular_api(
                model=self.model,
                http_methods=self.http_methods,
                nested_creations=self.nested_creations,
                port=self.port,
                output_dir=api_output_dir,
            )

        else:
            pydantic_model = PydanticGenerator(model=self.model, backend=self.backend, nested_creations=self.nested_creations, output_dir=self.output_dir)
            pydantic_model.generate()

            file_path = self.build_generation_path(file_name="rest_api.py")
            templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
            env = Environment(loader=FileSystemLoader(templates_path),
                          trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])
            env.globals.update(parse_bal=parse_bal, bal_to_rest=bal_to_rest,
                               normalize_code=normalize_method_code)
            template = env.get_template('fast_api_template.py.j2')
            with open(file_path, mode="w", encoding="utf-8") as f:
                generated_code = template.render(
                    classes=self.model.classes_sorted_by_inheritance(),
                    http_methods=self.http_methods,
                    model=self.model
                )
                f.write(generated_code)
            print("Code generated in the location: " + file_path)
