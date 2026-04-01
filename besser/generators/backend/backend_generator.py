import os
import configparser
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.structural_utils import separate_classes
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.rest_api import RESTAPIGenerator


class BackendGenerator(GeneratorInterface):
    """
    BackendGenerator generates a layered FastAPI backend project from a B-UML domain model.

    It orchestrates three sub-generators (SQLAlchemyGenerator, PydanticGenerator,
    RESTAPIGenerator) to produce per-entity files, then adds the app-level glue
    (main.py, config.py, database.py, bal.py).

    The generated output follows a standard FastAPI project structure::

        app/
        ├── __init__.py
        ├── main.py         # FastAPI app, middleware, system endpoints
        ├── config.py        # API metadata, DB URL, port
        ├── database.py      # SQLAlchemy engine, session, get_db
        ├── bal.py           # BESSER Action Language helpers
        ├── models/          # SQLAlchemy ORM models (per-entity)
        ├── schemas/         # Pydantic schemas (per-entity)
        └── routers/         # FastAPI routers (per-entity)

    Args:
        model (DomainModel): The B-UML domain model.
        http_methods (list, optional): HTTP methods to generate (GET, POST, PUT, DELETE). Defaults to all.
        nested_creations (bool, optional): Allow inline entity creation in M:N relationships. Defaults to False.
        output_dir (str, optional): Output directory. Defaults to ``./output_backend``.
        api_title (str, optional): FastAPI title. Defaults to ``"{model.name} API"``.
        api_description (str, optional): FastAPI description.
        api_version (str, optional): API version string. Defaults to ``"1.0.0"``.
        docker_image (bool, optional): Generate Docker files. Defaults to False.
        docker_config_path (str, optional): Path to Docker config file.
    """

    def __init__(
        self,
        model: DomainModel,
        http_methods: list = None,
        nested_creations: bool = False,
        output_dir: str = None,
        api_title: str = None,
        api_description: str = None,
        api_version: str = "1.0.0",
        docker_image: bool = False,
        docker_config_path: str = None,
    ):
        super().__init__(model, output_dir)
        allowed_methods = ["GET", "POST", "PUT", "DELETE"]
        if not http_methods:
            http_methods = allowed_methods
        else:
            http_methods = [method for method in http_methods if method in allowed_methods]
        self.http_methods = http_methods
        self.nested_creations = nested_creations
        self.api_title = api_title or f"{model.name} API"
        self.api_description = api_description or "Auto-generated REST API with full CRUD operations, relationship management, and advanced features"
        self.api_version = api_version
        self.docker_image = docker_image
        self.docker_config_path = docker_config_path
        self.config = self._load_config()

    def _load_config(self):
        if self.docker_config_path and os.path.exists(self.docker_config_path):
            config = configparser.ConfigParser()
            config.read(self.docker_config_path)
            return {
                "docker_username": config.get("DEFAULT", "docker_username"),
                "docker_password": config.get("DEFAULT", "docker_password"),
                "docker_image_name": config.get("DEFAULT", "docker_image_name"),
                "docker_repository": config.get("DEFAULT", "docker_repository"),
                "docker_tag": config.get("DEFAULT", "docker_tag"),
                "docker_port": config.get("DEFAULT", "docker_port"),
            }
        return None

    # ------------------------------------------------------------------
    # Jinja2 environment (for backend-specific templates only)
    # ------------------------------------------------------------------

    def _create_jinja_env(self):
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        return Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write(path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    def generate(self):
        """Generate a layered FastAPI backend project.

        Delegates per-entity file generation to the sub-generators and
        renders the app-level glue files itself.
        """
        root = self.output_dir or os.path.join(os.path.abspath(""), "output_backend")
        app_dir = os.path.join(root, "app")
        os.makedirs(app_dir, exist_ok=True)

        # -- Sub-generators produce per-entity files --
        sql_gen = SQLAlchemyGenerator(model=self.model)
        sql_gen.generate_layered(app_dir)

        pydantic_gen = PydanticGenerator(
            model=self.model,
            backend=True,
            nested_creations=self.nested_creations,
        )
        pydantic_gen.generate_layered(app_dir)

        rest_gen = RESTAPIGenerator(
            model=self.model,
            http_methods=self.http_methods,
            nested_creations=self.nested_creations,
        )
        rest_gen.generate_layered(app_dir)

        # -- App-level glue files (BackendGenerator's own templates) --
        env = self._create_jinja_env()
        classes, asso_classes = separate_classes(self.model)
        port = int(self.config["docker_port"]) if self.config else 8000

        ctx = {
            "name": self.model.name,
            "classes": classes,
            "asso_classes": asso_classes,
            "api_title": self.api_title,
            "api_description": self.api_description,
            "api_version": self.api_version,
            "port": port,
        }

        self._write(
            os.path.join(app_dir, "__init__.py"),
            env.get_template("app_init.py.j2").render(),
        )
        self._write(
            os.path.join(app_dir, "config.py"),
            env.get_template("config.py.j2").render(**ctx),
        )
        self._write(
            os.path.join(app_dir, "database.py"),
            env.get_template("database.py.j2").render(**ctx),
        )
        self._write(
            os.path.join(app_dir, "bal.py"),
            env.get_template("bal.py.j2").render(),
        )
        self._write(
            os.path.join(app_dir, "main.py"),
            env.get_template("main.py.j2").render(**ctx),
        )
        self._write(
            os.path.join(root, "requirements.txt"),
            env.get_template("requirements.txt.j2").render(),
        )
        self._write(
            os.path.join(root, ".env.example"),
            env.get_template("env.example.j2").render(**ctx),
        )

        # -- Docker --
        if self.docker_image:
            if self.config:
                self._build_and_push_docker_image(root)
            else:
                from besser.generators.backend.docker_files import generate_docker_files
                generate_docker_files(root)

        print(f"Backend generated at {root}")

    # ------------------------------------------------------------------
    # Docker
    # ------------------------------------------------------------------

    def _build_and_push_docker_image(self, backend_folder_path):
        import docker as docker_lib

        docker_port = self.config["docker_port"]

        dockerfile_content = f"""
FROM python:3.11-slim
WORKDIR /app

RUN mkdir -p /app/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE {docker_port}
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "{docker_port}"]
"""

        with open(os.path.join(backend_folder_path, "Dockerfile"), "w") as dockerfile:
            dockerfile.write(dockerfile_content)

        image_name = self.config["docker_image_name"]
        repository = self.config["docker_repository"]
        tag = self.config["docker_tag"]
        full_image_name = f"{repository}/{image_name}:{tag}"

        client = docker_lib.from_env()
        image, build_logs = client.images.build(path=backend_folder_path, rm=True, tag=full_image_name)
        for log in build_logs:
            print(log)

        username = self.config["docker_username"]
        password = self.config["docker_password"]
        client.login(username=username, password=password)

        resp = client.api.push(full_image_name, stream=True, decode=True)
        for line in resp:
            print(line)
