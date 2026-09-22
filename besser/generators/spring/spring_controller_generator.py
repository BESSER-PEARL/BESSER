"""Writes the REST controllers of a generated Spring Boot project."""

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.generators.spring._sub_generator import SpringSubGenerator, build_environment
from besser.generators.spring.java_types import (
    get_id_attribute,
    java_type_for,
    java_type_import,
    to_java_accessor_suffix,
    to_java_class_name,
    to_java_field_name,
    validate_java_package,
)


class SpringControllerGenerator(SpringSubGenerator):
    """Generates the ``controller`` package of a Spring Boot project (internal helper)."""

    def __init__(self, model: DomainModel,
                 entity_package_name: str,
                 service_package_name: str,
                 output_dir: str = "./generated/controller",
                 package_name: str = "com.example.controller"):
        super().__init__(model, output_dir)

        self.package_name: str = validate_java_package(package_name)
        self.entity_package_name: str = validate_java_package(entity_package_name)
        self.service_package_name: str = validate_java_package(service_package_name)

    def generate(self):
        for cls in self.concrete_classes():
            self._generate_controller_file(cls)

    def _generate_controller_file(self, cls: Class):
        class_name: str = to_java_class_name(cls.name)
        env = build_environment(trim_blocks=True, lstrip_blocks=True)
        controller_template = env.get_template("controller.java.j2")

        imports: set[str] = {
            "org.springframework.beans.factory.annotation.Autowired",
            "org.springframework.web.bind.annotation.RestController",
            "org.springframework.web.bind.annotation.RequestMapping",
            "org.springframework.web.bind.annotation.PathVariable",
            "org.springframework.web.bind.annotation.RequestBody",
            "org.springframework.web.bind.annotation.GetMapping",
            "org.springframework.web.bind.annotation.PostMapping",
            "org.springframework.web.bind.annotation.PutMapping",
            "org.springframework.web.bind.annotation.DeleteMapping",
            "org.springframework.http.ResponseEntity",
            "java.util.Optional",
            "java.util.List",
            f"{self.service_package_name}.interfaces.I{class_name}Service",
            f"{self.entity_package_name}.{class_name}",
        }

        id_attr = get_id_attribute(cls)
        id_type: str = java_type_for(id_attr.type, self.model_type_names)
        type_import = java_type_import(id_type)
        if type_import:
            imports.add(type_import)

        context = {
            "package": self.package_name,
            "imports": sorted(imports),
            "cls": class_name,
            "instance": to_java_field_name(class_name[0].lower() + class_name[1:]),
            "route": class_name[0].lower() + class_name[1:],
            "id_type": id_type,
            # The identifier is not necessarily called "id": the setter has to
            # follow whatever the model named it.
            "id_accessor": to_java_accessor_suffix(id_attr.name),
        }

        self.write(f"{class_name}Controller.java", controller_template.render(**context))
