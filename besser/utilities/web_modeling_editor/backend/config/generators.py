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
from besser.generators.qiskit import QiskitGenerator
from besser.generators.rdf import RDFGenerator
from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.react import ReactGenerator
from besser.generators.flutter import FlutterGenerator
from besser.generators.terraform import TerraformGenerator
try:
    from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator
except ImportError:
    PytorchGenerator = None

try:
    from besser.generators.nn.tf.tf_code_generator import TFGenerator
except ImportError:
    TFGenerator = None


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
    # Note: "Smart Data Models" in the frontend uses this same generator with mode="smart_data".
    # There is no separate "smartdata" generator entry; the mode is passed via config.
    "jsonschema": GeneratorInfo(
        generator_class=JSONSchemaGenerator,
        output_type="file",  # file for regular mode, zip for smart_data mode
        file_extension=".json",
        category="data_format",
        requires_class_diagram=True
    ),
    "jsonobject": GeneratorInfo(
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

    # Quantum generators
    "qiskit": GeneratorInfo(
        generator_class=QiskitGenerator,
        output_type="file",
        file_extension=".py",
        category="quantum",
        requires_class_diagram=False
    ),

    # RDF generator (class diagram based)
    "rdf": GeneratorInfo(
        generator_class=RDFGenerator,
        output_type="file",
        file_extension=".ttl",
        category="data_format",
        requires_class_diagram=True
    ),

    # REST API generator (class diagram based)
    "rest_api": GeneratorInfo(
        generator_class=RESTAPIGenerator,
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),

    # React generator (requires class diagram + GUI model)
    "react": GeneratorInfo(
        generator_class=ReactGenerator,
        output_type="zip",
        file_extension=".zip",
        category="frontend",
        requires_class_diagram=True
    ),

    # Flutter generator (requires class diagram + GUI model)
    "flutter": GeneratorInfo(
        generator_class=FlutterGenerator,
        output_type="zip",
        file_extension=".zip",
        category="frontend",
        requires_class_diagram=True
    ),

    # Terraform generator (deployment model based)
    "terraform": GeneratorInfo(
        generator_class=TerraformGenerator,
        output_type="zip",
        file_extension=".zip",
        category="deployment",
        requires_class_diagram=False
    ),

}

# Neural network generators are conditionally registered since they
# require optional dependencies (torch, tensorflow) that may not be installed.
if PytorchGenerator is not None:
    SUPPORTED_GENERATORS["pytorch"] = GeneratorInfo(
        generator_class=PytorchGenerator,
        output_type="file",
        file_extension=".py",
        category="neural_network",
        requires_class_diagram=False
    )

if TFGenerator is not None:
    SUPPORTED_GENERATORS["tensorflow"] = GeneratorInfo(
        generator_class=TFGenerator,
        output_type="file",
        file_extension=".py",
        category="neural_network",
        requires_class_diagram=False
    )


def get_generator_info(generator_type: str) -> GeneratorInfo | None:
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
    elif generator_type == "jsonobject":
        return "object_model.json"
    elif generator_type == "qiskit":
        return "qiskit_circuit.py"
    elif generator_type == "rdf":
        return "vocabulary.ttl"
    elif generator_type == "rest_api":
        return "rest_api.zip"
    elif generator_type == "react":
        return "react_app.zip"
    elif generator_type == "flutter":
        return "flutter_app.zip"
    elif generator_type == "terraform":
        return "terraform.zip"
    elif generator_type == "pytorch":
        return "pytorch_nn.py"
    elif generator_type == "tensorflow":
        return "tf_nn.py"
    else:
        return f"{generator_type}_output{info.file_extension}"


def is_generator_supported(generator_type: str) -> bool:
    """Check if a generator type is supported."""
    return generator_type in SUPPORTED_GENERATORS
