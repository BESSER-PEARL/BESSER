"""The one list of BESSER's code generators.

The web editor backend (``SUPPORTED_GENERATORS``) and the Spec-Driven Agent
(its ``generate_*`` tools) are both derived from ``GENERATORS``. Registering a
generator is one entry here. Generator classes are imported only on
``load_class()``, so importing this module pulls in no generator or optional
dependency.
"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentTool:
    """How the Spec-Driven Agent offers a generator as an LLM tool."""

    name: str
    # Shown in the run preview's summary ("Generate <label> from ..."); not sent to the LLM.
    label: str
    description: str
    # Models the tool needs loaded: domain, gui, quantum, object, agent, bpmn, nn.
    requires: frozenset[str]
    # JSON Schema ``properties`` of the tool's input.
    parameters: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        """The tool definition sent to the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {"type": "object", "properties": copy.deepcopy(self.parameters)},
        }


@dataclass(frozen=True)
class GeneratorSpec:
    """One code generator and where it is offered."""

    id: str
    class_path: str  # "package.module:ClassName"
    category: str
    output_type: str  # "file" or "zip"
    file_extension: str
    requires_class_diagram: bool = True
    # The non-class diagram the generator consumes (e.g. ``NNDiagram``); None = the class diagram.
    required_diagram_type: str | None = None
    # Download name; None falls back to ``<id>_output<file_extension>``.
    download_filename: str | None = None
    # Single-file generators: download under the name the generator chose.
    keep_generated_filename: bool = False
    # What the generator needs besides one diagram (e.g. a GUI model), so
    # ``/generate-output``, which carries one, cannot run it; None = one is enough.
    needs_beyond_one_diagram: str | None = None
    agent_tool: AgentTool | None = None

    def load_class(self) -> type:
        module_name, _, class_name = self.class_path.partition(":")
        return getattr(importlib.import_module(module_name), class_name)


_DOMAIN = frozenset({"domain"})
_DOMAIN_GUI = frozenset({"domain", "gui"})
_HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]
_DBMS = ["sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"]


# Ordered as the web editor lists them (``/besser_api/`` ``supported_generators``).
GENERATORS: tuple[GeneratorSpec, ...] = (
    # Object-oriented generators (class diagram based)
    GeneratorSpec(
        id="python",
        class_path="besser.generators.python_classes:PythonGenerator",
        category="object_oriented",
        output_type="file",
        file_extension=".py",
        download_filename="classes.py",
        agent_tool=AgentTool(
            name="generate_python_classes",
            label="plain Python classes",
            description=(
                "Generate plain Python classes with __init__, getters/setters, relationships. "
                "Output: classes.py"
            ),
            requires=_DOMAIN,
        ),
    ),
    GeneratorSpec(
        id="java",
        class_path="besser.generators.java_classes:JavaGenerator",
        category="object_oriented",
        output_type="zip",
        file_extension=".zip",
        download_filename="java_classes.zip",
        agent_tool=AgentTool(
            name="generate_java_classes",
            label="Java classes",
            description="Generate Java classes from the domain model. Output: .java files zipped.",
            requires=_DOMAIN,
        ),
    ),
    GeneratorSpec(
        id="pydantic",
        class_path="besser.generators.pydantic_classes:PydanticGenerator",
        category="object_oriented",
        output_type="file",
        file_extension=".py",
        download_filename="pydantic_classes.py",
        agent_tool=AgentTool(
            name="generate_pydantic",
            label="Pydantic models",
            description=(
                "Generate Pydantic BaseModel classes from the domain model. "
                "Produces validated data models with type hints and field validators. "
                "Output: pydantic_classes.py"
            ),
            requires=_DOMAIN,
            parameters={
                "backend": {
                    "type": "boolean",
                    "description": "Generate 'Create' suffixed models for API request bodies",
                    "default": False,
                },
                "nested_creations": {
                    "type": "boolean",
                    "description": "Allow nested entity creation in request bodies",
                    "default": False,
                },
            },
        ),
    ),
    # Not an agent tool: none was ever added.
    GeneratorSpec(
        id="test_case",
        class_path="besser.generators.testgen:TestCaseGenerator",
        category="object_oriented",
        output_type="file",
        file_extension=".py",
        download_filename="test_hypothesis.py",
    ),
    # Web framework generators (class diagram based)
    GeneratorSpec(
        id="django",
        class_path="besser.generators.django:DjangoGenerator",
        category="web_framework",
        output_type="zip",
        file_extension=".zip",
        download_filename="django_project.zip",
        agent_tool=AgentTool(
            name="generate_django",
            label="Django project (models + views + admin)",
            description=(
                "Generate a full Django project with models, views, templates, URL routing, "
                "and admin panel. Output: full Django project directory"
            ),
            requires=_DOMAIN,
            parameters={
                "project_name": {"type": "string", "default": "myproject"},
                "app_name": {"type": "string", "default": "myapp"},
            },
        ),
    ),
    # Not an agent tool: the agent writes a Spring Boot request from scratch
    # ("spring boot" is in planning/generator_keywords.STACKS_WITHOUT_GENERATOR).
    GeneratorSpec(
        id="spring",
        class_path="besser.generators.spring.spring_backend_generator:SpringBackendGenerator",
        category="web_framework",
        output_type="zip",
        file_extension=".zip",
        download_filename="spring_boot_project.zip",
    ),
    GeneratorSpec(
        id="backend",
        class_path="besser.generators.backend:BackendGenerator",
        category="web_framework",
        output_type="zip",
        file_extension=".zip",
        download_filename="fastapi_backend.zip",
        agent_tool=AgentTool(
            name="generate_fastapi_backend",
            label="FastAPI backend (CRUD + ORM + schemas)",
            description=(
                "Generate a complete, MODULAR FastAPI backend with CRUD endpoints, SQLAlchemy ORM, "
                "and Pydantic models. This is the most complete backend generator. "
                "Output: main_api.py (slim app) + database.py + routers/<Class>.py (endpoints per "
                "class) + sql_alchemy.py + pydantic_classes.py + bal_stdlib.py + requirements.txt"
            ),
            requires=_DOMAIN,
            parameters={
                "http_methods": {
                    "type": "array",
                    "items": {"type": "string", "enum": _HTTP_METHODS},
                    "description": "HTTP methods to generate endpoints for",
                    "default": ["GET", "POST", "PUT", "DELETE"],
                },
            },
        ),
    ),
    GeneratorSpec(
        id="web_app",
        class_path="besser.generators.web_app:WebAppGenerator",
        category="web_framework",
        output_type="zip",
        file_extension=".zip",
        download_filename="web_app.zip",
        needs_beyond_one_diagram=(
            "a GUI model (GUINoCodeDiagram) as well as the ClassDiagram; "
            "use /generate-output-from-project with a project that has both"
        ),
        agent_tool=AgentTool(
            name="generate_web_app",
            label="full-stack web app (FastAPI + React + Docker)",
            description=(
                "Generate a complete full-stack web app: React frontend + FastAPI backend "
                "+ Docker deployment. REQUIRES a GUI model. "
                "Output: frontend/ + backend/ + docker-compose.yml"
            ),
            requires=_DOMAIN_GUI,
        ),
    ),
    # Database generators (class diagram based)
    GeneratorSpec(
        id="sqlalchemy",
        class_path="besser.generators.sql_alchemy:SQLAlchemyGenerator",
        category="database",
        output_type="file",
        file_extension=".py",
        download_filename="sql_alchemy.py",
        agent_tool=AgentTool(
            name="generate_sqlalchemy",
            label="SQLAlchemy ORM models",
            description=(
                "Generate SQLAlchemy ORM models from the domain model. "
                "Produces table definitions with relationships, foreign keys, and inheritance. "
                "Output: sql_alchemy.py"
            ),
            requires=_DOMAIN,
            parameters={
                "dbms": {
                    "type": "string",
                    "enum": _DBMS,
                    "description": "Target database system",
                    "default": "sqlite",
                },
            },
        ),
    ),
    GeneratorSpec(
        id="sql",
        class_path="besser.generators.sql:SQLGenerator",
        category="database",
        output_type="file",
        file_extension=".sql",
        download_filename="tables.sql",
        agent_tool=AgentTool(
            name="generate_sql",
            label="SQL CREATE TABLE statements",
            description="Generate raw SQL CREATE TABLE statements. Output: tables.sql",
            requires=_DOMAIN,
        ),
    ),
    GeneratorSpec(
        id="supabase",
        class_path="besser.generators.supabase:SupabaseGenerator",
        category="database",
        output_type="file",
        file_extension=".sql",
        keep_generated_filename=True,
        agent_tool=AgentTool(
            name="generate_supabase",
            label="Supabase schema and client configuration",
            description=(
                "Generate Supabase-flavoured PostgreSQL DDL from the domain model — "
                "UUID primary keys, auth.users mirroring, and Row-Level-Security (RLS) "
                "policies. Output: a .sql file."
            ),
            requires=_DOMAIN,
            parameters={
                "user_root": {
                    "type": "string",
                    "description": "Class mapped to Supabase auth users (default: User)",
                },
            },
        ),
    ),
    # Data format generators. "Smart Data Models" in the frontend is jsonschema
    # with mode="smart_data" (a zip in that mode), not a separate generator.
    GeneratorSpec(
        id="jsonschema",
        class_path="besser.generators.json:JSONSchemaGenerator",
        category="data_format",
        output_type="file",
        file_extension=".json",
        download_filename="json_schema.json",
        agent_tool=AgentTool(
            name="generate_json_schema",
            label="JSON Schema",
            description="Generate JSON Schema from the domain model. Output: json_schema.json",
            requires=_DOMAIN,
            parameters={
                "mode": {
                    "type": "string",
                    "enum": ["regular", "smart_data"],
                    "default": "regular",
                },
            },
        ),
    ),
    GeneratorSpec(
        id="jsonobject",
        class_path="besser.generators.json:JSONObjectGenerator",
        category="object_model",
        output_type="file",
        file_extension=".json",
        requires_class_diagram=False,
        download_filename="object_model.json",
        agent_tool=AgentTool(
            name="generate_json_object",
            label="JSON object seed data",
            description=(
                "Generate a JSON document of the object (instance) model — the concrete "
                "instances and their attribute/relationship values (useful as seed data / "
                "fixtures). REQUIRES an object model. Output: <model>.json"
            ),
            requires=frozenset({"object"}),
        ),
    ),
    # AI agent generator (agent diagram based)
    GeneratorSpec(
        id="agent",
        class_path="besser.generators.agents.baf_generator:BAFGenerator",
        category="ai_agent",
        output_type="zip",
        file_extension=".zip",
        requires_class_diagram=False,
        download_filename="agent_output.zip",
        agent_tool=AgentTool(
            name="generate_baf",
            label="BESSER Agent Framework chatbot",
            description=(
                "Generate a BESSER Agent Framework (BAF) chatbot/agent project from the "
                "agent model. REQUIRES an agent model. Output: a Python BAF agent project."
            ),
            requires=frozenset({"agent"}),
        ),
    ),
    GeneratorSpec(
        id="qiskit",
        class_path="besser.generators.qiskit:QiskitGenerator",
        category="quantum",
        output_type="file",
        file_extension=".py",
        requires_class_diagram=False,
        required_diagram_type="QuantumCircuitDiagram",
        download_filename="qiskit_circuit.py",
        agent_tool=AgentTool(
            name="generate_qiskit",
            label="Qiskit circuit code",
            description=(
                "Generate a Qiskit Python script from the Quantum Circuit model. "
                "REQUIRES a quantum circuit. Output: qiskit_circuit.py"
            ),
            requires=frozenset({"quantum"}),
            parameters={
                "backend_type": {
                    "type": "string",
                    "description": "Qiskit backend (default: aer_simulator)",
                },
                "shots": {
                    "type": "integer",
                    "description": "Number of shots for the simulation (default: 1024)",
                },
            },
        ),
    ),
    GeneratorSpec(
        id="rdf",
        class_path="besser.generators.rdf:RDFGenerator",
        category="data_format",
        output_type="file",
        file_extension=".ttl",
        download_filename="vocabulary.ttl",
        agent_tool=AgentTool(
            name="generate_rdf",
            label="RDF / OWL vocabulary",
            description="Generate RDF/OWL vocabulary (Turtle format). Output: vocabulary.ttl",
            requires=_DOMAIN,
        ),
    ),
    GeneratorSpec(
        id="rest_api",
        class_path="besser.generators.rest_api:RESTAPIGenerator",
        category="web_framework",
        output_type="zip",
        file_extension=".zip",
        download_filename="rest_api.zip",
        agent_tool=AgentTool(
            name="generate_rest_api",
            label="FastAPI REST endpoints",
            description=(
                "Generate FastAPI REST endpoints + Pydantic models. "
                "Lighter than generate_fastapi_backend (no ORM). "
                "Output: rest_api.py + pydantic_classes.py"
            ),
            requires=_DOMAIN,
            parameters={
                "http_methods": {
                    "type": "array",
                    "items": {"type": "string", "enum": _HTTP_METHODS},
                    "default": ["GET", "POST", "PUT", "DELETE"],
                },
            },
        ),
    ),
    # Frontend generators (class diagram + GUI model)
    GeneratorSpec(
        id="react",
        class_path="besser.generators.react:ReactGenerator",
        category="frontend",
        output_type="zip",
        file_extension=".zip",
        download_filename="react_app.zip",
        needs_beyond_one_diagram="a GUI model (GUINoCodeDiagram) as well as the ClassDiagram",
        agent_tool=AgentTool(
            name="generate_react",
            label="React frontend",
            description=(
                "Generate a React TypeScript frontend from the domain model + GUI model. "
                "REQUIRES a GUI model — fails without one. Output: full React app directory"
            ),
            requires=_DOMAIN_GUI,
        ),
    ),
    # Not an agent tool: dropped in #694, the agent writes a mobile target from
    # scratch in the stack the user names.
    GeneratorSpec(
        id="flutter",
        class_path="besser.generators.flutter:FlutterGenerator",
        category="frontend",
        output_type="zip",
        file_extension=".zip",
        download_filename="flutter_app.zip",
        needs_beyond_one_diagram="a GUI model (GUINoCodeDiagram) as well as the ClassDiagram",
    ),
    # Deployment generators. Not agent tools: the agent loads no deployment model.
    GeneratorSpec(
        id="terraform",
        class_path="besser.generators.terraform:TerraformGenerator",
        category="deployment",
        output_type="zip",
        file_extension=".zip",
        requires_class_diagram=False,
        download_filename="terraform.zip",
        needs_beyond_one_diagram="a deployment model (the cloud clusters to provision), which no editor diagram holds",
    ),
    # Vendor-neutral BPMN 2.0 XML from a BPMNDiagram
    GeneratorSpec(
        id="bpmn",
        class_path="besser.generators.bpmn:BPMNGenerator",
        category="business_process",
        output_type="file",
        file_extension=".bpmn",
        requires_class_diagram=False,
        required_diagram_type="BPMNDiagram",
        download_filename="bpmn_diagram.bpmn",
        agent_tool=AgentTool(
            name="generate_bpmn",
            label="BPMN process XML",
            description=(
                "Generate vendor-neutral BPMN 2.0 XML from the BPMN process model. "
                "REQUIRES a BPMN model. Output: bpmn_diagram.bpmn"
            ),
            requires=frozenset({"bpmn"}),
        ),
    ),
    # A ZIP holding docker-compose.yml plus one baked BAF build context per linked agent.
    GeneratorSpec(
        id="docker_compose",
        class_path="besser.generators.docker_compose:DockerComposeGenerator",
        category="deployment",
        output_type="zip",
        file_extension=".zip",
        requires_class_diagram=False,
        required_diagram_type="DeploymentDiagram",
        download_filename="docker_compose.zip",
    ),
    # Formal methods (class diagram based). A ZIP holding the Alloy
    # specification model.als plus the library modules it opens.
    GeneratorSpec(
        id="alloy",
        class_path="besser.generators.alloy:AlloyGenerator",
        category="formal_methods",
        output_type="zip",
        file_extension=".zip",
        download_filename="alloy_specification.zip",
    ),
    # Neural network generators
    GeneratorSpec(
        id="pytorch",
        class_path="besser.generators.nn.pytorch.pytorch_code_generator:PytorchGenerator",
        category="neural_network",
        output_type="file",
        file_extension=".py",
        requires_class_diagram=False,
        required_diagram_type="NNDiagram",
        download_filename="pytorch_nn.py",
        agent_tool=AgentTool(
            name="generate_pytorch",
            label="PyTorch neural network",
            description=(
                "Generate PyTorch neural-network training/eval code from the NN model. "
                "REQUIRES a neural-network model (and PyTorch installed). Output: pytorch_nn.py"
            ),
            requires=frozenset({"nn"}),
            parameters={
                "generation_type": {
                    "type": "string",
                    "enum": ["subclassing", "sequential"],
                    "description": "PyTorch code style (default: subclassing)",
                },
            },
        ),
    ),
    GeneratorSpec(
        id="tensorflow",
        class_path="besser.generators.nn.tf.tf_code_generator:TFGenerator",
        category="neural_network",
        output_type="file",
        file_extension=".py",
        requires_class_diagram=False,
        required_diagram_type="NNDiagram",
        download_filename="tf_nn.py",
        agent_tool=AgentTool(
            name="generate_tensorflow",
            label="TensorFlow neural network",
            description=(
                "Generate TensorFlow/Keras neural-network training/eval code from the NN "
                "model. REQUIRES a neural-network model (and TensorFlow installed). "
                "Output: tf_nn.py"
            ),
            requires=frozenset({"nn"}),
            parameters={
                "generation_type": {
                    "type": "string",
                    "enum": ["subclassing", "sequential"],
                    "description": "TensorFlow code style (default: subclassing)",
                },
            },
        ),
    ),
    # Writes ``forecaster_<model name>.py``; the download keeps that name.
    # Not an agent tool: none was ever added.
    GeneratorSpec(
        id="forecaster",
        class_path="besser.generators.nn.forecaster:TimeSeriesForecasterGenerator",
        category="neural_network",
        output_type="file",
        file_extension=".py",
        requires_class_diagram=False,
        required_diagram_type="NNDiagram",
        keep_generated_filename=True,
    ),
)

# The agent's tool order, which is part of its prompt. Every generator with an
# ``agent_tool`` must be listed; ``agent_tools()`` raises otherwise.
AGENT_TOOL_ORDER: tuple[str, ...] = (
    "pydantic", "sqlalchemy", "backend", "django", "python", "java", "sql", "jsonschema",
    "rest_api", "react", "web_app", "rdf", "qiskit", "supabase", "jsonobject", "agent",
    "bpmn", "pytorch", "tensorflow",
)


def agent_tools() -> tuple[AgentTool, ...]:
    """The Spec-Driven Agent's generator tools, in prompt order."""
    rank = {gid: i for i, gid in enumerate(AGENT_TOOL_ORDER)}
    specs = [spec for spec in GENERATORS if spec.agent_tool is not None]
    unranked = [spec.id for spec in specs if spec.id not in rank]
    if unranked:
        raise ValueError(f"Agent tools missing from AGENT_TOOL_ORDER: {unranked}")
    specs.sort(key=lambda spec: rank[spec.id])
    return tuple(spec.agent_tool for spec in specs)
