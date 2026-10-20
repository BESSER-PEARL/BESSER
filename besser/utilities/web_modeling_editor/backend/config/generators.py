"""
Generator configuration and metadata for the BESSER backend.
"""

from typing import Dict, Any, NamedTuple, Optional
from besser.generators.django import DjangoGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.java_classes import JavaGenerator
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.spring.spring_backend_generator import SpringBackendGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.supabase import SupabaseGenerator
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
from besser.generators.testgen import TestCaseGenerator
from besser.generators.bpmn import BPMNGenerator
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    BPMN_PROJECT_DIAGRAM_KEY,
    AGENT_OUTPUT_FILENAME,
    DEPLOYMENT_DIAGRAM_TYPE,
)
from besser.generators.docker_compose import DockerComposeGenerator
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
    # Non-class diagram type this generator consumes (e.g. ``NNDiagram``,
    # ``QuantumCircuitDiagram``, ``AgentDiagram``). ``None`` means the
    # generator works off the project's active class diagram. Used by the
    # project endpoint to dispatch to the right diagram without branching
    # on generator-name string literals.
    required_diagram_type: Optional[str] = None
    # Name of the downloaded file (``Content-Disposition``). ``None`` falls
    # back to ``<generator_type>_output<file_extension>``; see
    # ``get_filename_for_generator``.
    download_filename: Optional[str] = None
    # Single-file generators only: ship the file under the name the generator
    # chose instead of ``download_filename`` (e.g. Supabase's migration-style
    # ``<timestamp>_<slug>.sql``, whose name is meaningful).
    keep_generated_filename: bool = False


# Generator configuration mapping
SUPPORTED_GENERATORS: Dict[str, GeneratorInfo] = {
    # Object-oriented generators (class diagram based)
    "python": GeneratorInfo(
        generator_class=PythonGenerator,
        download_filename="classes.py",
        output_type="file",
        file_extension=".py",
        category="object_oriented",
        requires_class_diagram=True
    ),
    "java": GeneratorInfo(
        generator_class=JavaGenerator,
        download_filename="java_classes.zip",
        output_type="zip",
        file_extension=".zip",
        category="object_oriented",
        requires_class_diagram=True
    ),
    "pydantic": GeneratorInfo(
        generator_class=PydanticGenerator,
        download_filename="pydantic_classes.py",
        output_type="file",
        file_extension=".py",
        category="object_oriented",
        requires_class_diagram=True
    ),

    "test_case": GeneratorInfo(
        generator_class=TestCaseGenerator,
        download_filename="test_hypothesis.py",
        output_type="file",
        file_extension=".py",
        category="object_oriented",
        requires_class_diagram=True
    ),

    # Web framework generators (class diagram based)
    "django": GeneratorInfo(
        generator_class=DjangoGenerator,
        download_filename="django_project.zip",
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),
    "spring": GeneratorInfo(
        generator_class=SpringBackendGenerator,
        download_filename="spring_boot_project.zip",
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),
    "backend": GeneratorInfo(
        generator_class=BackendGenerator,
        download_filename="fastapi_backend.zip",
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),
    "web_app": GeneratorInfo(
        generator_class=WebAppGenerator,
        download_filename="web_app.zip",
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),

    # Database generators (class diagram based)
    "sqlalchemy": GeneratorInfo(
        generator_class=SQLAlchemyGenerator,
        download_filename="sql_alchemy.py",
        output_type="file",
        file_extension=".py",
        category="database",
        requires_class_diagram=True
    ),
    "sql": GeneratorInfo(
        generator_class=SQLGenerator,
        download_filename="tables.sql",
        output_type="file",
        file_extension=".sql",
        category="database",
        requires_class_diagram=True
    ),
    "supabase": GeneratorInfo(
        generator_class=SupabaseGenerator,
        keep_generated_filename=True,
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
        download_filename="json_schema.json",
        output_type="file",  # file for regular mode, zip for smart_data mode
        file_extension=".json",
        category="data_format",
        requires_class_diagram=True
    ),
    "jsonobject": GeneratorInfo(
        generator_class=JSONObjectGenerator,
        download_filename="object_model.json",
        output_type="file",
        file_extension=".json",
        category="object_model",
        requires_class_diagram=False
    ),

    # AI/Agent generators (agent diagram based)
    "agent": GeneratorInfo(
        generator_class=BAFGenerator,
        download_filename=AGENT_OUTPUT_FILENAME,
        output_type="zip",
        file_extension=".zip",
        category="ai_agent",
        requires_class_diagram=False
    ),

    # Quantum generators
    "qiskit": GeneratorInfo(
        generator_class=QiskitGenerator,
        download_filename="qiskit_circuit.py",
        output_type="file",
        file_extension=".py",
        category="quantum",
        requires_class_diagram=False,
        required_diagram_type="QuantumCircuitDiagram",
    ),

    # RDF generator (class diagram based)
    "rdf": GeneratorInfo(
        generator_class=RDFGenerator,
        download_filename="vocabulary.ttl",
        output_type="file",
        file_extension=".ttl",
        category="data_format",
        requires_class_diagram=True
    ),

    # REST API generator (class diagram based)
    "rest_api": GeneratorInfo(
        generator_class=RESTAPIGenerator,
        download_filename="rest_api.zip",
        output_type="zip",
        file_extension=".zip",
        category="web_framework",
        requires_class_diagram=True
    ),

    # React generator (requires class diagram + GUI model)
    "react": GeneratorInfo(
        generator_class=ReactGenerator,
        download_filename="react_app.zip",
        output_type="zip",
        file_extension=".zip",
        category="frontend",
        requires_class_diagram=True
    ),

    # Flutter generator (requires class diagram + GUI model)
    "flutter": GeneratorInfo(
        generator_class=FlutterGenerator,
        download_filename="flutter_app.zip",
        output_type="zip",
        file_extension=".zip",
        category="frontend",
        requires_class_diagram=True
    ),

    # Terraform generator (deployment model based)
    "terraform": GeneratorInfo(
        generator_class=TerraformGenerator,
        download_filename="terraform.zip",
        output_type="zip",
        file_extension=".zip",
        category="deployment",
        requires_class_diagram=False
    ),

    # BPMN generator (vendor-neutral BPMN 2.0 XML; reads BPMNDiagram)
    "bpmn": GeneratorInfo(
        generator_class=BPMNGenerator,
        download_filename="bpmn_diagram.bpmn",
        output_type="file",
        file_extension=".bpmn",
        category="business_process",
        requires_class_diagram=False,
        # Project bucket key ("BPMN"); the model inside has type "BPMNDiagram".
        required_diagram_type=BPMN_PROJECT_DIAGRAM_KEY,
    ),

    # Docker Compose generator (UML Deployment diagram based). A ZIP holding
    # docker-compose.yml plus one baked BAF build context per linked agent.
    "docker_compose": GeneratorInfo(
        generator_class=DockerComposeGenerator,
        download_filename="docker_compose.zip",
        output_type="zip",
        file_extension=".zip",
        category="deployment",
        requires_class_diagram=False,
        required_diagram_type=DEPLOYMENT_DIAGRAM_TYPE,
    ),
}

# Neural network generators are conditionally registered since they
# require optional dependencies (torch, tensorflow) that may not be installed.
if PytorchGenerator is not None:
    SUPPORTED_GENERATORS["pytorch"] = GeneratorInfo(
        generator_class=PytorchGenerator,
        download_filename="pytorch_nn.py",
        output_type="file",
        file_extension=".py",
        category="neural_network",
        requires_class_diagram=False,
        required_diagram_type="NNDiagram",
    )

if TFGenerator is not None:
    SUPPORTED_GENERATORS["tensorflow"] = GeneratorInfo(
        generator_class=TFGenerator,
        download_filename="tf_nn.py",
        output_type="file",
        file_extension=".py",
        category="neural_network",
        requires_class_diagram=False,
        required_diagram_type="NNDiagram",
    )


# Filename prefix for NN generator output. The NN generator strips the
# trailing ".py" from its configured file_name and appends
# ``_{generation_type}.py``, so the artifact for ``pytorch`` /
# ``tensorflow`` is e.g. ``pytorch_nn_subclassing.py`` /
# ``tf_nn_sequential.py``. Centralizing the mapping here keeps the router
# free of hard-coded string switches.
_NN_FILENAME_PREFIX: Dict[str, str] = {
    "pytorch": "pytorch",
    "tensorflow": "tf",
}


def get_nn_filename(generator_type: str, generation_type: str) -> str:
    """Return the on-disk filename produced by an NN generator for the given
    ``generation_type`` variant (``"subclassing"`` or ``"sequential"``)."""
    prefix = _NN_FILENAME_PREFIX[generator_type]
    return f"{prefix}_nn_{generation_type}.py"


def get_generator_info(generator_type: str) -> GeneratorInfo | None:
    """Get generator information by type."""
    return SUPPORTED_GENERATORS.get(generator_type)


def get_filename_for_generator(generator_type: str, base_name: str = "output") -> str:
    """Return the download filename for a generator.

    Uses ``GeneratorInfo.download_filename`` when set, otherwise
    ``<generator_type>_output<file_extension>``; an unknown generator gets
    ``<base_name>.txt``.
    """
    info = get_generator_info(generator_type)
    if not info:
        return f"{base_name}.txt"
    return info.download_filename or f"{generator_type}_output{info.file_extension}"


def is_generator_supported(generator_type: str) -> bool:
    """Check if a generator type is supported."""
    return generator_type in SUPPORTED_GENERATORS
