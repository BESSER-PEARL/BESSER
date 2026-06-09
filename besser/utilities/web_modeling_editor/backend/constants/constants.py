# API Configuration
API_VERSION = "1.0.0"

# Temp directory prefixes
TEMP_DIR_PREFIX = "besser_"
AGENT_TEMP_DIR_PREFIX = "besser_agent_"
CSV_TEMP_DIR_PREFIX = "besser_csv_"

# Output defaults
OUTPUT_DIR_NAME = "output"
AGENT_MODEL_FILENAME = "agent_model.py"
AGENT_OUTPUT_FILENAME = "agent_output.zip"

# Generator defaults
DEFAULT_SQL_DIALECT = "standard"
DEFAULT_DBMS = "sqlite"
DEFAULT_JSONSCHEMA_MODE = "regular"
DEFAULT_QISKIT_BACKEND = "aer_simulator"
DEFAULT_QISKIT_SHOTS = 1024
DEFAULT_DJANGO_PROJECT_NAME = "myproject"
DEFAULT_DJANGO_APP_NAME = "myapp"
DEFAULT_SUPABASE_USER_ROOT = "User"

# CORS defaults
DEFAULT_CORS_ORIGINS = [
    "https://editor.besser-pearl.org",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
]

VALID_PRIMITIVE_TYPES = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "double": "float",
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
    "time": "time",
    "timedelta": "timedelta",
    "any": "any"
}

VISIBILITY_MAP = {
    "+": "public",
    "-": "private",
    "#": "protected",
    "~": "package"
}

RELATIONSHIP_TYPES = {
    "bidirectional": "ClassBidirectional",
    "unidirectional": "ClassUnidirectional",
    "composition": "ClassComposition",
    "aggregation": "ClassAggregation",
    "inheritance": "ClassInheritance"
}

# ---------------------------------------------------------------------------
# BPMN — WME ↔ B-UML BPMN metamodel constants
# Single source of truth for the BPMN converter pair (json_to_buml /
# buml_to_json). Diagram-type discriminator: "BPMNDiagram" (uniform with
# "ClassDiagram", "StateMachineDiagram", …).
# ---------------------------------------------------------------------------
BPMN_DIAGRAM_TYPE = "BPMNDiagram"
BPMN_RELATIONSHIP_TYPE = "BPMNFlow"

# WME element-type string → metamodel class name.
BPMN_ELEMENT_TYPES = {
    "BPMNTask": "Task",
    "BPMNSubprocess": "SubProcess",
    "BPMNTransaction": "Transaction",
    "BPMNCallActivity": "CallActivity",
    "BPMNStartEvent": "StartEvent",
    "BPMNIntermediateEvent": "IntermediateEvent",
    "BPMNEndEvent": "EndEvent",
    "BPMNGateway": "Gateway",
    "BPMNDataObject": "DataObject",
    "BPMNDataStore": "DataStore",
    "BPMNAnnotation": "TextAnnotation",
    "BPMNGroup": "Group",
    "BPMNPool": "Participant",
    "BPMNSwimlane": "Lane",
}

# WME flowType → metamodel class name.
BPMN_FLOW_TYPES = {
    "sequence": "SequenceFlow",
    "message": "MessageFlow",
    "association": "Association",
    "data association": "DataAssociation",
}

# ---------------------------------------------------------------------------
# UML Component / Deployment diagram wire constants (02-... §3)
# ---------------------------------------------------------------------------

# Element type strings emitted by Apollon / WME.
COMPONENT_ELEMENT_TYPES = {
    "Component": "Component",
    "Subsystem": "Subsystem",
    "ComponentInterface": "ComponentInterface",
}

COMPONENT_RELATIONSHIP_TYPES = {
    "ComponentDependency": "ComponentDependency",
    "ComponentInterfaceProvided": "ComponentInterfaceProvided",
    "ComponentInterfaceRequired": "ComponentInterfaceRequired",
}

DEPLOYMENT_ELEMENT_TYPES = {
    "DeploymentNode": "DeploymentNode",
    "DeploymentArtifact": "DeploymentArtifact",
    "DeploymentComponent": "DeploymentComponent",
    "DeploymentInterface": "DeploymentInterface",
}

DEPLOYMENT_RELATIONSHIP_TYPES = {
    "DeploymentAssociation": "DeploymentAssociation",
    "DeploymentDependency": "DeploymentDependency",
    "DeploymentInterfaceProvided": "DeploymentInterfaceProvided",
    "DeploymentInterfaceRequired": "DeploymentInterfaceRequired",
}

# Stereotype tokens consumed *silently* — Apollon defaults that re-state the
# element-type discrimination and would otherwise round-trip as noise in the
# free-form `stereotypes` list (02-... §3.3 / Appendix B.2).
COMPONENT_DEFAULT_STEREOTYPES = frozenset({"component", "subsystem"})
DEPLOYMENT_DEFAULT_STEREOTYPES = frozenset({"node"})  # handled via NodeKind.GENERIC

# Stereotype tokens that promote a Component subtype. `llm`/`db`/`rag`
# (meeting 2026-06-08 §5) are Tool subclasses — see uml_component/agentic.py.
COMPONENT_SUBTYPE_TOKENS = {
    "skill": "Skill",
    "tool": "Tool",
    "llm": "LLM",
    "db": "Database",
    "rag": "RAG",
}

# Locality tokens (shared Component/Deployment side).
LOCALITY_TOKENS = frozenset({"local", "external", "hybrid"})

# AgentCategory tokens (Component side).
AGENT_CATEGORY_TOKENS = frozenset({
    "solution", "supervision", "consensus", "collaboration",
})

# AgenticEdgeKind tokens (Component side, on ComponentDependency edges).
AGENTIC_EDGE_KIND_TOKENS = frozenset({
    "delegates", "supervises", "revises", "collaborates",
    "has", "uses", "granted", "implements",
})

# NodeKind tokens (Deployment side).
NODE_KIND_TOKENS = frozenset({"node", "device", "executionenvironment"})
