"""Internal base class shared by the Spring sub-generators.

The entity, repository, service, controller and HTTP writers are *not* stand-alone
generators: they are never registered in ``SUPPORTED_GENERATORS`` and they only
make sense inside the directory layout that
:class:`~besser.generators.spring.spring_backend_generator.SpringBackendGenerator`
lays out. They therefore deliberately do not implement ``GeneratorInterface`` —
they are plain helpers composed by the public generator, in the same way
``WebAppGenerator`` composes its own writers.
"""

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import (
    Class,
    DateTimeType,
    DateType,
    DomainModel,
    Enumeration,
    TimeType,
)
from besser.generators.spring.java_types import (
    is_many,
    java_type_for,
    java_type_import,
    to_java_accessor_suffix,
    to_java_class_name,
    to_java_field_name,
)

TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

#: B-UML types that get an extra ``findAllBy<Attribute>Between`` range finder.
RANGE_FINDER_TYPES = (DateType.name, DateTimeType.name, TimeType.name)


def build_environment(**options) -> Environment:
    """Environment: A Jinja2 environment bound to the Spring template folder."""
    return Environment(loader=FileSystemLoader(TEMPLATES_PATH), **options)


class SpringSubGenerator:
    """Common plumbing for the Spring sub-generators (internal helper)."""

    def __init__(self, model: DomainModel, output_dir: str | Path):
        self.model: DomainModel = model
        self.output_dir: Path = Path(output_dir)
        self.enumerations: set[Enumeration] = model.get_enumerations()
        self.classes: list[Class] = model.classes_sorted_by_inheritance()
        # Names of every user-defined type, used to tell a class/enum reference
        # apart from a primitive type when mapping to Java.
        self.model_type_names: set[str] = (
            {cls.name for cls in self.classes} | {enum.name for enum in self.enumerations}
        )

    def concrete_classes(self) -> list[Class]:
        """list[Class]: The non-abstract classes, in a deterministic order."""
        return [cls for cls in self.classes if not cls.is_abstract]

    def derived_finder_methods(self, cls: Class, entity_package_name: str,
                               imports: set[str]) -> list[dict]:
        """Build the ``findAllBy<Attribute>`` query methods derived from a class.

        The repository interface, the service interface and the service
        implementation all render the very same list, so that the delegation in
        the implementation always matches the repository signature. ``imports``
        is extended in place with whatever the signatures need.
        """
        class_name = to_java_class_name(cls.name)
        methods: list[dict] = []

        for attr in sorted(cls.attributes, key=lambda a: a.name):
            if is_many(attr.multiplicity) or attr.is_id:
                continue

            parameter_type = java_type_for(attr.type, self.model_type_names)
            if attr.type.name in self.model_type_names:
                imports.add(f"{entity_package_name}.{parameter_type}")
            type_import = java_type_import(parameter_type)
            if type_import:
                imports.add(type_import)

            finder_suffix = to_java_accessor_suffix(attr.name)

            if attr.type.name in RANGE_FINDER_TYPES:
                methods.append({
                    "return_value": f"ArrayList<{class_name}>",
                    "name": f"findAllBy{finder_suffix}Between",
                    "parameter": f"{parameter_type} start, {parameter_type} end",
                })

            methods.append({
                "return_value": f"ArrayList<{class_name}>",
                "name": f"findAllBy{finder_suffix}",
                "parameter": f"{parameter_type} {to_java_field_name(attr.name)}",
            })

        if methods:
            imports.add("java.util.ArrayList")

        return sorted(methods, key=lambda method: method["name"])

    def write(self, relative_path: str | Path, content: str) -> str:
        """str: Write ``content`` under the output directory, creating parents."""
        file_path = self.output_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" keeps the generated Java byte-identical across platforms.
        with open(file_path, mode="w", encoding="utf-8", newline="\n") as file:
            file.write(content)
        return str(file_path)
