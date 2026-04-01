import os
import re
import unicodedata
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.action_language.ActionLanguageASTBuilder import parse_bal
from besser.generators import GeneratorInterface
from besser.generators.structural_utils import (
    get_foreign_keys, get_ids, separate_classes,
    get_concrete_table_inheritance, get_sqlalchemy_types,
)
from besser.generators.action_language.RESTGenerator import bal_to_rest
from besser.generators.pydantic_classes import PydanticGenerator

class RESTAPIGenerator(GeneratorInterface):
    """
    RESTAPIGenerator generates FastAPI REST API code from a B-UML model.

    It supports two output modes:

    - **Standalone** (default): generates a simple ``rest_api.py`` with Pydantic classes
      for quick prototyping.
    - **Layered** (via :meth:`generate_layered`): generates per-entity router files
      inside an ``app/routers/`` directory, used by :class:`BackendGenerator`.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        http_methods (list): HTTP methods to generate ("GET", "POST", "PUT", "PATCH", "DELETE").
        nested_creations (bool, optional): Allow inline entity creation in M:N relationships. Defaults to False.
        output_dir (str, optional): The output directory for standalone mode. Defaults to None.
    """
    def __init__(self, model: DomainModel, http_methods: list = None, nested_creations: bool = False, output_dir: str = None):
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
        self.nested_creations = nested_creations

    def _create_layered_env(self):
        """Create Jinja2 environment for layered output templates."""
        def clean_method_name(name):
            if '(' in str(name):
                return str(name).split('(')[0].strip()
            return str(name).strip()

        def ascii_identifier(name: str) -> str:
            normalized = unicodedata.normalize("NFKD", name)
            ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
            ascii_name = re.sub(r"\W", "_", ascii_name)
            if ascii_name and ascii_name[0].isdigit():
                ascii_name = f"_{ascii_name}"
            return ascii_name

        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            extensions=['jinja2.ext.do'],
        )
        env.filters['clean_method_name'] = clean_method_name
        env.filters['ascii_identifier'] = ascii_identifier
        env.globals.update(parse_bal=parse_bal, bal_to_rest=bal_to_rest)
        return env

    def generate_layered(self, app_dir: str):
        """Generate per-entity FastAPI router files into app_dir/routers/.

        Args:
            app_dir: Path to the app/ directory where routers/ will be created.
        """
        routers_dir = os.path.join(app_dir, "routers")
        os.makedirs(routers_dir, exist_ok=True)

        env = self._create_layered_env()
        classes, asso_classes = separate_classes(self.model)

        ctx = {
            "name": self.model.name,
            "model": self.model,
            "classes": classes,
            "sorted_classes": self.model.classes_sorted_by_inheritance(),
            "asso_classes": asso_classes,
            "http_methods": self.http_methods,
            "nested_creations": self.nested_creations,
            "fkeys": get_foreign_keys(self.model),
            "ids": get_ids(self.model),
            "types": get_sqlalchemy_types(self.model),
            "enumerations": list(self.model.get_enumerations()),
            "associations": list(self.model.associations),
            "concrete_parents": get_concrete_table_inheritance(self.model),
        }

        def _write(path, content):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)

        # Per-entity router files
        for class_obj in classes:
            _write(
                os.path.join(routers_dir, f"{class_obj.name.lower()}.py"),
                env.get_template("router_entity.py.j2").render(class_obj=class_obj, **ctx),
            )

        # __init__.py — Collects all routers into all_routers list
        _write(
            os.path.join(routers_dir, "__init__.py"),
            env.get_template("routers_init.py.j2").render(**ctx),
        )

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
        Generates a standalone REST API: ``rest_api.py`` + ``pydantic_classes.py`` + ``requirements.txt``.

        This is the simple single-file mode for quick prototyping.
        For the layered backend structure, use :meth:`generate_layered` (called by BackendGenerator).
        """
        self.generate_requirements()

        pydantic_model = PydanticGenerator(model=self.model, backend=False, nested_creations=self.nested_creations, output_dir=self.output_dir)
        pydantic_model.generate()

        file_path = self.build_generation_path(file_name="rest_api.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path),
                          trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])
        env.globals.update(parse_bal=parse_bal, bal_to_rest=bal_to_rest)
        template = env.get_template('fast_api_template.py.j2')
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                classes=self.model.classes_sorted_by_inheritance(),
                http_methods=self.http_methods,
                model=self.model
            )
            f.write(generated_code)
        print("Code generated in the location: " + file_path)
