"""Writes the ``.http`` request samples of a generated Spring Boot project."""

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.generators.spring._sub_generator import SpringSubGenerator, build_environment
from besser.generators.spring.java_types import to_java_class_name


class SpringHttpGenerator(SpringSubGenerator):
    """Generates the ``.http`` scratch files of a Spring Boot project (internal helper)."""

    def __init__(self, model: DomainModel, output_dir: str = "./generated/http"):
        super().__init__(model, output_dir)

    def generate(self):
        for cls in self.concrete_classes():
            self._generate_http_file(cls)

    def _generate_http_file(self, cls: Class):
        class_name: str = to_java_class_name(cls.name)
        route: str = class_name[0].lower() + class_name[1:]
        env = build_environment(trim_blocks=True, lstrip_blocks=True)
        http_template = env.get_template("http.http.j2")

        self.write(f"{route}.http", http_template.render(route=route))
