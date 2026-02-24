"""
Generator configuration and metadata for the BESSER backend.
"""

from typing import Dict, Any, NamedTuple
from besser.generators.django import DjangoGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.java_classes import JavaGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.backend import BackendGenerator
from besser.generators.json import JSONSchemaGenerator, JSONObjectGenerator
from besser.generators.agents.baf_generator import BAFGenerator
from besser.generators.web_app import WebAppGenerator


class GeneratorInfo(NamedTuple):
    """Information about a generator."""
    generator_class: Any
    output_type: str  # "file" or "zip"
    file_extension: str
    category: str
    requires_class_diagram: bool = True


# Generator configuration mapping
SUPPORTED_GENERATORS: Dict[str, GeneratorInfo] = {
    # Object-oriented generators (class diagram based)
    "python": GeneratorInfo(
        generator_class=PythonGenerator,
        output_type="file",
        file_extension=".py",
        category="object_oriented",
        requires_class_diagram=True
    ),
    "java": GeneratorInfo(
        generator_class=JavaGenerator,
        output_type="zip",
        file_extension=".zip",
        category="object_oriented",
        requires_class_diagram=True
    ),
    "pydantic": GeneratorInfo(
        generator_class=PydanticGenerator,
        output_type="file",
        file_extension=".py",
        category="object_oriented",
        requires_class_diagram=True
    ),
    
    # Web framework generators (class diagram based)
    "django": GeneratorInfo(
        generator_class=DjangoGenerator,
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),
    "backend": GeneratorInfo(
        generator_class=BackendGenerator,
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),
    "web_app": GeneratorInfo(
        generator_class=WebAppGenerator,
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),
    
    # Database generators (class diagram based)
    "sqlalchemy": GeneratorInfo(
        generator_class=SQLAlchemyGenerator,
        output_type="file",
        file_extension=".py",
        category="database",
        requires_class_diagram=True
    ),
    "sql": GeneratorInfo(
        generator_class=SQLGenerator,
        output_type="file",
        file_extension=".sql",
        category="database",
        requires_class_diagram=True
    ),
    
    # Data format generators (class diagram based)
    "jsonschema": GeneratorInfo(
        generator_class=JSONSchemaGenerator,
        output_type="file",  # This can be either file or zip depending on mode
        file_extension=".json",
        category="data_format",
        requires_class_diagram=True
    ),
    "json": GeneratorInfo(
        generator_class=JSONObjectGenerator,
        output_type="file",
        file_extension=".json",
        category="object_model",
        requires_class_diagram=False
    ),
    
    # AI/Agent generators (agent diagram based)
    "agent": GeneratorInfo(
        generator_class=BAFGenerator,
        output_type="zip",
        file_extension=".zip",
        category="ai_agent",
        requires_class_diagram=False
    ),
}


def get_generator_info(generator_type: str) -> GeneratorInfo:
    """Get generator information by type."""
    return SUPPORTED_GENERATORS.get(generator_type)


def get_filename_for_generator(generator_type: str, base_name: str = "output") -> str:
    """Get the appropriate filename for a generator."""
    info = get_generator_info(generator_type)
    if not info:
        return f"{base_name}.txt"
    
    if generator_type == "python":
        return "classes.py"
    elif generator_type == "pydantic":
        return "pydantic_classes.py"
    elif generator_type == "sqlalchemy":
        return "sql_alchemy.py"
    elif generator_type == "sql":
        return "tables.sql"
    elif generator_type == "jsonschema":
        return "json_schema.json"
    elif generator_type == "json":
        return "object_model.json"
    else:
        return f"{generator_type}_output{info.file_extension}"


def is_generator_supported(generator_type: str) -> bool:
    """Check if a generator type is supported."""
    return generator_type in SUPPORTED_GENERATORS
