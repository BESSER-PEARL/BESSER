"""Writes the Spring Data JPA repository interfaces of a generated project."""

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.generators.spring._sub_generator import SpringSubGenerator, build_environment
from besser.generators.spring.java_types import (
    get_id_attribute,
    java_type_for,
    to_java_class_name,
    validate_java_package,
)


class SpringRepositoryGenerator(SpringSubGenerator):
    """Generates the ``repository`` package of a Spring Boot project (internal helper)."""

    def __init__(self, model: DomainModel,
                 entity_package_name: str,
                 output_dir: str = "./generated/repository",
                 package_name: str = "com.example.repository"):
        super().__init__(model, output_dir)

        self.package_name: str = validate_java_package(package_name)
        self.entity_package_name: str = validate_java_package(entity_package_name)

    def generate(self):
        for cls in self.concrete_classes():
            self._generate_repository_file(cls)

    def _generate_repository_file(self, cls: Class):
        class_name: str = to_java_class_name(cls.name)
        env = build_environment(trim_blocks=True, lstrip_blocks=True)
        repository_template = env.get_template("irepository.java.j2")

        imports: set[str] = {
            "org.springframework.data.jpa.repository.JpaRepository",
            "org.springframework.stereotype.Repository",
            f"{self.entity_package_name}.{class_name}",
        }
        methods = self.derived_finder_methods(cls, self.entity_package_name, imports)
        id_attr = get_id_attribute(cls)

        context = {
            "package": self.package_name,
            "imports": sorted(imports),
            "cls": class_name,
            "methods": methods,
            "id_type": java_type_for(id_attr.type, self.model_type_names),
        }

        self.write(f"I{class_name}Repository.java", repository_template.render(**context))
