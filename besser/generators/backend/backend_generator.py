import os
import configparser
import docker
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.backend.docker_files import generate_docker_files

class BackendGenerator(GeneratorInterface):
    """
    BackendGenerator is a class that implements the GeneratorInterface and is responsible for generating
    a Backend model code with a REST API using FAST API framework, SQLAlchemy and a Pydantic model based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        http_methods (list, optional): A list of HTTP methods to be used in the REST API. Defaults to All.
        nested_creations (bool, optional): This parameter specifies how entities are linked in the API request. If set to True, the API expects 
                                identifiers and links entities based on these IDs. If set to False, the API handles the creation of 
                                new entities based on the data provided in the request. Defaults to True
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
        docker_image (bool, optional): Flag to indicate if Docker image generation is required. Defaults to False.
        docker_config_path (str, optional): The path to the docker configuration file to auto upload the image. Defaults to None.
    """

    def __init__(self, model: DomainModel, http_methods: list = None, nested_creations: bool = False, output_dir: str = None, docker_image: bool = False, docker_config_path: str = None):
        super().__init__(model, output_dir)
        allowed_methods = ["GET", "POST", "PUT", "DELETE"]
        if not http_methods:
            http_methods = allowed_methods
        else:
            http_methods = [method for method in http_methods if method in allowed_methods]
        self.http_methods = http_methods
        self.nested_creations = nested_creations
        self.docker_image = docker_image
        self.docker_config_path = docker_config_path
        self.config = self.load_config()

    def load_config(self):
        """
        Loads the configuration from the specified config file path if provided.

        Returns:
            dict: A dictionary containing the Docker configuration, or None if the config file path is not provided or file not found.
        """
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
        else:
            if not self.docker_config_path:
                print("No configuration docker file.")
            else:
                print(f"Configuration file not found at path: {self.docker_config_path}")
            return None

    def generate(self):
        """
        Generates Backend model code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output_backend folder.

        Returns:
            None, but stores the generated code as files main_api.py, sql_alchemy.py and pydantic_classes.py
        """
        if self.output_dir is not None:
            backend_folder_path = self.output_dir
            os.makedirs(backend_folder_path, exist_ok=True)
            print(f"Backend folder created at {backend_folder_path}")
        else:
            backend_folder_path = os.path.join(os.path.abspath(''), "output_backend")
            os.makedirs(backend_folder_path, exist_ok=True)
            print(f"Backend folder created at {backend_folder_path}")

        docker_port = self.config["docker_port"] if self.config else 8000  # Use default port if config not provided

        rest_api = RESTAPIGenerator(model=self.model, http_methods=self.http_methods, nested_creations=self.nested_creations, output_dir=backend_folder_path, backend=True, port=docker_port)
        rest_api.generate()

        sql_alchemy = SQLAlchemyGenerator(model=self.model, output_dir=backend_folder_path)
        sql_alchemy.generate()

        pydantic_model = PydanticGenerator(model=self.model, output_dir=backend_folder_path, backend=True, nested_creations=self.nested_creations)
        pydantic_model.generate()

        if self.docker_image:
            if self.config:
                self.build_and_push_docker_image(backend_folder_path)
            else:
                generate_docker_files(backend_folder_path)

    def build_and_push_docker_image(self, backend_folder_path):
        """
        Builds and pushes the Docker image based on the provided backend folder path.

        Args:
            backend_folder_path (str): The path to the backend folder containing the generated code.

        Returns:
            None
        """
        docker_port = self.config["docker_port"]

        dockerfile_content = f"""
        FROM python:3.9-slim
        WORKDIR /app

        COPY main_api.py /app
        COPY pydantic_classes.py /app
        COPY sql_alchemy.py /app

        RUN pip install requests==2.31.0
        RUN pip install fastapi==0.110.0
        RUN pip install pydantic==2.6.3
        RUN pip install uvicorn==0.28.0
        RUN pip install SQLAlchemy==2.0.29
        RUN pip install httpx==0.27.0

        EXPOSE {docker_port}
        CMD ["python", "main_api.py"]
        """

        with open(os.path.join(backend_folder_path, 'Dockerfile'), 'w') as dockerfile:
            dockerfile.write(dockerfile_content)

        image_name = self.config["docker_image_name"]
        repository = self.config["docker_repository"]
        tag = self.config["docker_tag"]
        full_image_name = f"{repository}/{image_name}:{tag}"

        # Create docker client
        client = docker.from_env()

        # Create docker image
        image, build_logs = client.images.build(
            path=backend_folder_path,
            rm=True,
            tag=full_image_name
        )

        for log in build_logs:
            print(log)

        # Docker login
        username = self.config["docker_username"]
        password = self.config["docker_password"]
        client.login(username=username, password=password)

        # Push Docker image
        resp = client.api.push(full_image_name, stream=True, decode=True)
        for line in resp:
            print(line)
