"""Writes the service interfaces and implementations of a generated project."""

from pathlib import Path
from typing import List

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.generators.spring._sub_generator import SpringSubGenerator, build_environment
from besser.generators.spring.java_types import (
    get_id_attribute,
    java_type_for,
    to_java_class_name,
    to_java_field_name,
    validate_java_package,
)


class SpringServiceGenerator(SpringSubGenerator):
    """Generates the ``service`` package of a Spring Boot project (internal helper)."""

    def __init__(self, model: DomainModel,
                 entity_package_name: str,
                 repository_package_name: str,
                 output_dir: str = "./generated/service",
                 package_name: str = "com.example.service"):
        super().__init__(model, output_dir)

        self.package_name: str = validate_java_package(package_name)
        self.entity_package_name: str = validate_java_package(entity_package_name)
        self.repository_package_name: str = validate_java_package(repository_package_name)

    def generate(self):
        for cls in self.concrete_classes():
            self._generate_service_files(cls)

    def _generate_service_files(self, cls: Class):
        class_name: str = to_java_class_name(cls.name)
        env = build_environment(trim_blocks=True, lstrip_blocks=True)

        imports: set[str] = {
            f"{self.entity_package_name}.{class_name}",
            "java.util.List",
            "java.util.Optional",
        }
        # The very same list the repository renders, so that the implementation
        # below can delegate to it method for method.
        methods: List[dict] = self.derived_finder_methods(cls, self.entity_package_name, imports)
        methods.extend(self._get_crud_methods(cls, class_name))
        methods.sort(key=lambda method: method["name"])

        context = {
            "package": f"{self.package_name}.interfaces",
            "imports": sorted(imports),
            "cls": class_name,
            "instance": to_java_field_name(class_name[0].lower() + class_name[1:]),
            "methods": methods,
        }

        self.write(
            Path("interfaces") / f"I{class_name}Service.java",
            env.get_template("iservice.java.j2").render(**context),
        )

        imports.add("org.springframework.beans.factory.annotation.Autowired")
        imports.add("org.springframework.stereotype.Service")
        imports.add(f"{self.package_name}.interfaces.I{class_name}Service")
        imports.add(f"{self.repository_package_name}.I{class_name}Repository")

        for method in methods:
            parameter: str = method["parameter"]
            tokens: List[str] = parameter.split(", ")
            method["parameter_names"] = ", ".join(token.split(" ")[1] for token in tokens) if parameter else ""

        context["package"] = f"{self.package_name}.impl"
        context["imports"] = sorted(imports)

        self.write(
            Path("impl") / f"{class_name}Service.java",
            env.get_template("service.java.j2").render(**context),
        )

    def _get_crud_methods(self, cls: Class, class_name: str) -> List[dict]:
        id_attr = get_id_attribute(cls)
        instance_name: str = to_java_field_name(class_name[0].lower() + class_name[1:])

        return [
            {
                "return_value": f"List<{class_name}>",
                "name": "findAll",
                "parameter": "",
            },
            {
                "return_value": f"Optional<{class_name}>",
                "name": "findById",
                "parameter": (f"{java_type_for(id_attr.type, self.model_type_names)} "
                              f"{to_java_field_name(id_attr.name)}"),
            },
            {
                "return_value": class_name,
                "name": "save",
                "parameter": f"{class_name} {instance_name}",
            },
            {
                "return_value": "void",
                "name": "delete",
                "parameter": f"{class_name} {instance_name}",
            },
        ]
