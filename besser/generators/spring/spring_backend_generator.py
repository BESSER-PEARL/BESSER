"""Generates a complete, buildable Spring Boot backend from a B-UML domain model."""

import copy
import os
import shutil
import stat
from pathlib import Path

from besser.BUML.metamodel.structural import DomainModel, IntegerType, Property
from besser.generators.generator_interface import GeneratorInterface
from besser.generators.spring._sub_generator import build_environment
from besser.generators.spring.java_types import to_java_class_name, validate_java_package
from besser.generators.spring.spring_controller_generator import SpringControllerGenerator
from besser.generators.spring.spring_entity_generator import SpringEntityGenerator
from besser.generators.spring.spring_http_generator import SpringHttpGenerator
from besser.generators.spring.spring_repository_generator import SpringRepositoryGenerator
from besser.generators.spring.spring_service_generator import SpringServiceGenerator

#: Defaults for the generated project. They live here rather than in the web
#: editor's constants module so that the generator never has to import the web
#: backend; ``backend/constants/constants.py`` re-exports these instead.
DEFAULT_SPRING_BOOT_VERSION: str = "3.4.4"
DEFAULT_JAVA_VERSION: str = "21"
DEFAULT_SPRING_APP_NAME: str = "Application"
DEFAULT_SPRING_PACKAGE_NAME: str = "com.example"
DEFAULT_SPRING_GROUP_ID: str = "com.example"

#: Name of the identifier added to a class hierarchy that declares none.
SURROGATE_ID_NAME: str = "id"


def with_surrogate_ids(model: DomainModel) -> DomainModel:
    """Return a copy of ``model`` in which every class has an identifier.

    A JPA entity needs exactly one ``@Id``. Most models drawn in the editor mark
    no attribute with ``is_id``, so for each class hierarchy without one the
    root class gets an ``id: int`` identifier, which becomes an auto-generated
    ``Integer`` ``@Id``. An existing attribute named ``id`` is promoted instead
    of adding a second one. Classes that already have an identifier, directly
    or inherited, are left as they are. The input model is not modified.
    """
    classes = model.get_classes()
    if all(any(attr.is_id for attr in cls.all_attributes()) for cls in classes):
        return model

    model = copy.deepcopy(model)
    # Parents first, so an identifier added to a root is inherited by its subclasses.
    for cls in model.classes_sorted_by_inheritance():
        if any(attr.is_id for attr in cls.all_attributes()):
            continue
        existing = next((attr for attr in cls.attributes if attr.name == SURROGATE_ID_NAME), None)
        if existing is not None:
            existing.is_id = True
        else:
            cls.add_attribute(Property(name=SURROGATE_ID_NAME, type=IntegerType, is_id=True))
    return model


#: Static (non-templated) files shipped with the generator.
RESOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


class SpringBackendGenerator(GeneratorInterface):
    """Generates a Spring Boot project (Maven, JPA entities, repositories,
    services and REST controllers) from a B-UML domain model.

    Args:
        model (DomainModel): The domain model to generate the backend from.
        output_dir (str): The directory the project is written to.
        spring_boot_version (str): The ``spring-boot-starter-parent`` version.
        java_version (str): The ``java.version`` property of the generated POM.
        app_name (str): The name of the application (and of its main class).
        package_name (str): The root Java package of the generated sources.
        group_id (str): The Maven ``groupId`` of the generated project.
        description (str): The Maven ``description`` of the generated project.
    """

    def __init__(self,
                 model: DomainModel,
                 output_dir: str = None,
                 *,
                 spring_boot_version: str = DEFAULT_SPRING_BOOT_VERSION,
                 java_version: str = DEFAULT_JAVA_VERSION,
                 app_name: str = DEFAULT_SPRING_APP_NAME,
                 package_name: str = DEFAULT_SPRING_PACKAGE_NAME,
                 group_id: str = DEFAULT_SPRING_GROUP_ID,
                 description: str = ""):
        super().__init__(model, output_dir)

        self.package_name: str = validate_java_package(package_name)
        self.spring_boot_version: str = spring_boot_version
        self.java_version: str = java_version
        self.app_name: str = app_name
        self.group_id: str = group_id
        self.description: str = description

    @property
    def project_dir(self) -> Path:
        """Path: The root of the generated project."""
        return Path(self.build_generation_dir())

    @property
    def main_class_name(self) -> str:
        """str: The sanitized name of the generated ``@SpringBootApplication``."""
        return to_java_class_name(self.app_name)

    @property
    def package_dir(self) -> Path:
        """Path: The relative path of the root package inside a source folder."""
        return Path(*self.package_name.split("."))

    def generate(self):
        # Generate from a copy in which every class has an identifier, so the
        # caller's model is never modified.
        original_model = self.model
        self.model = with_surrogate_ids(original_model)
        try:
            self._generate_pom_file()
            self._generate_mvn_files()
            self._generate_main_and_test_files()
            self._generate_properties_file()
            self._generate_entities()
            self._generate_repositories()
            self._generate_services()
            self._generate_controllers()
            self._generate_http()
        finally:
            self.model = original_model

    # ------------------------------------------------------------------
    # Project scaffolding
    # ------------------------------------------------------------------

    def _render(self, template_name: str, relative_path: Path | str, **context):
        env = build_environment()
        content = env.get_template(template_name).render(**context)
        file_path = self.project_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode="w", encoding="utf-8", newline="\n") as file:
            file.write(content)

    def _generate_pom_file(self):
        self._render(
            "pom.xml.j2", "pom.xml",
            name=self.app_name,
            spring_boot_version=self.spring_boot_version,
            group_id=self.group_id,
            description=self.description,
            java_version=self.java_version,
        )

    def _generate_mvn_files(self):
        """Copy the Maven wrapper to the layout Maven expects.

        ``mvnw`` and ``mvnw.cmd`` have to sit at the project root; only
        ``maven-wrapper.properties`` belongs under ``.mvn/wrapper/``. The two
        scripts carry no model-dependent content, so they are plain static
        resources rather than templates.
        """
        project_dir = self.project_dir
        project_dir.mkdir(parents=True, exist_ok=True)

        wrapper_dir = project_dir / ".mvn" / "wrapper"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            os.path.join(RESOURCES_PATH, "maven-wrapper.properties"),
            wrapper_dir / "maven-wrapper.properties",
        )

        # The line endings are forced on copy: a CRLF ``mvnw`` is unusable on
        # POSIX ("bad interpreter"), whatever the checkout settings were.
        self._copy_script("mvnw", project_dir / "mvnw", newline=b"\n", executable=True)
        self._copy_script("mvnw.cmd", project_dir / "mvnw.cmd", newline=b"\r\n")

    @staticmethod
    def _copy_script(resource_name: str, destination: Path, newline: bytes, executable: bool = False):
        content = Path(RESOURCES_PATH, resource_name).read_bytes().replace(b"\r\n", b"\n")
        if newline != b"\n":
            content = content.replace(b"\n", newline)
        destination.write_bytes(content)
        if executable:
            mode = os.stat(destination).st_mode
            os.chmod(destination, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _generate_main_and_test_files(self):
        self._render(
            "main.java.j2",
            Path("src", "main", "java") / self.package_dir / f"{self.main_class_name}.java",
            name=self.main_class_name,
            package=self.package_name,
        )
        self._render(
            "test.java.j2",
            Path("src", "test", "java") / self.package_dir / f"{self.main_class_name}Tests.java",
            name=self.main_class_name,
            package=self.package_name,
        )

    def _generate_properties_file(self):
        self._render(
            "application.properties.j2",
            Path("src", "main", "resources", "application.properties"),
            name=self.app_name,
        )

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def _source_dir(self, sub_package: str) -> Path:
        return self.project_dir / Path("src", "main", "java") / self.package_dir / sub_package

    def _generate_entities(self):
        SpringEntityGenerator(
            self.model,
            output_dir=self._source_dir("entity"),
            package_name=f"{self.package_name}.entity",
        ).generate()

    def _generate_repositories(self):
        SpringRepositoryGenerator(
            self.model,
            f"{self.package_name}.entity",
            output_dir=self._source_dir("repository"),
            package_name=f"{self.package_name}.repository",
        ).generate()

    def _generate_services(self):
        SpringServiceGenerator(
            self.model,
            f"{self.package_name}.entity",
            f"{self.package_name}.repository",
            output_dir=self._source_dir("service"),
            package_name=f"{self.package_name}.service",
        ).generate()

    def _generate_controllers(self):
        SpringControllerGenerator(
            self.model,
            f"{self.package_name}.entity",
            f"{self.package_name}.service",
            output_dir=self._source_dir("controller"),
            package_name=f"{self.package_name}.controller",
        ).generate()

    def _generate_http(self):
        # Scratch request files are test resources, not compilation units: they
        # must stay out of ``src/main/java``.
        SpringHttpGenerator(
            self.model,
            output_dir=self.project_dir / Path("src", "test", "resources", "http"),
        ).generate()
