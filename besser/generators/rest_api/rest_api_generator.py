import os
from typing import Dict

from jinja2 import Environment, FileSystemLoader
from besser.generators.pk_types import pk_python_types
from besser.BUML.metamodel.structural import AssociationClass, DomainModel
from besser.BUML.notations.action_language.ActionLanguageASTBuilder import parse_bal
from besser.generators import GeneratorInterface
from besser.generators.structural_utils import get_foreign_keys, normalize_method_code
from besser.generators.action_language.RESTGenerator import bal_to_rest
from besser.generators.pydantic_classes import PydanticGenerator
from besser.utilities.utils import sort_by_timestamp

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

    def get_pk_names(self) -> Dict[str, str]:
        """
        Maps every class name of the model to the name of its primary key attribute.

        The selection mirrors the one of the SQLAlchemy generator: the attribute flagged
        with ``is_id``, otherwise an attribute literally named ``id``, otherwise the
        surrogate ``id`` column that SQLAlchemy adds to the table.

        Returns:
            dict: A dictionary with class names as keys and primary key attribute names as values.
        """
        pk_names: Dict[str, str] = {}
        for cls in self.model.get_classes():
            attributes = sort_by_timestamp(cls.attributes)
            id_attr = next((attr.name for attr in attributes if attr.is_id), None)
            if not id_attr:
                id_attr = next((attr.name for attr in attributes if attr.name == "id"), None)
            pk_names[cls.name] = id_attr or "id"
        return pk_names

    def get_association_classes(self) -> Dict[str, dict]:
        """
        Describes the association classes of the model.

        An association carrying an association class is materialized by the SQLAlchemy
        generator as a mapped class (with one ``<end name>_id`` foreign key column per
        association end plus the attributes of the association class) instead of a plain
        secondary table, so the REST API has to go through that class to read and write
        the links.

        Returns:
            dict: A dictionary with association class names as keys and a description
            (``association``, ``ends`` and ``attributes``) as values.
        """
        assoc_classes: Dict[str, dict] = {}
        for cls in self.model.get_classes():
            if not isinstance(cls, AssociationClass):
                continue
            assoc_classes[cls.name] = {
                "association": cls.association.name,
                "ends": [
                    {"name": end.name, "type_name": end.type.name}
                    for end in sorted(cls.association.ends, key=lambda end: end.name)
                ],
                "attributes": [
                    {
                        "name": attribute.name,
                        "is_enum": attribute.type.__class__.__name__ == "Enumeration",
                    }
                    for attribute in sort_by_timestamp(cls.attributes)
                ],
            }
        return assoc_classes

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
            pk_names = self.get_pk_names()
            assoc_classes = self.get_association_classes()
            assoc_by_association = {
                info["association"]: assoc_class_name
                for assoc_class_name, info in assoc_classes.items()
            }

            def pk_of(class_name: str) -> str:
                """Jinja filter returning the primary key attribute name of a class."""
                return pk_names.get(str(class_name), "id")

            file_path = self.build_generation_path(file_name="main_api.py")
            templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
            env = Environment(loader=FileSystemLoader(templates_path),
                          trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])
            env.filters['clean_method_name'] = clean_method_name
            env.filters['pk'] = pk_of
            env.globals.update(parse_bal=parse_bal, bal_to_rest=bal_to_rest,
                               normalize_code=normalize_method_code)
            template = env.get_template('backend_fast_api_template.py.j2')
            with open(file_path, mode="w", encoding="utf-8") as f:
                generated_code = template.render(
                    pk_types=pk_python_types(self.model),
                    name=self.model.name,
                    model=self.model,
                    classes=self.model.classes_sorted_by_inheritance(),
                    http_methods=self.http_methods,
                    nested_creations=self.nested_creations,
                    port=self.port,
                    fkeys=get_foreign_keys(self.model),
                    pk_names=pk_names,
                    assoc_classes=assoc_classes,
                    assoc_by_association=assoc_by_association
                )
                f.write(generated_code)
            print("Code generated in the location: " + file_path)

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
