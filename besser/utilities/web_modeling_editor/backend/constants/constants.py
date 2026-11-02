from besser.BUML.metamodel.uml_component import AgentCategory, AgenticEdgeKind
from besser.BUML.metamodel.uml_component import Locality as ComponentLocality
from besser.BUML.metamodel.uml_deployment import NodeKind
from besser.llm import DEFAULT_OPENAI_MODEL
from besser.BUML.notations.kg_to_buml.consistency import DEFAULT_CONSISTENCY_TIMEOUT_SECONDS
from besser.generators.spring.spring_backend_generator import (
    DEFAULT_JAVA_VERSION as _SPRING_DEFAULT_JAVA_VERSION,
    DEFAULT_SPRING_APP_NAME as _SPRING_DEFAULT_APP_NAME,
    DEFAULT_SPRING_BOOT_VERSION as _SPRING_DEFAULT_BOOT_VERSION,
    DEFAULT_SPRING_PACKAGE_NAME as _SPRING_DEFAULT_PACKAGE_NAME,
)

# API Configuration
API_VERSION = "1.0.0"

# Temp directory prefixes
TEMP_DIR_PREFIX = "besser_"
AGENT_TEMP_DIR_PREFIX = "besser_agent_"
CSV_TEMP_DIR_PREFIX = "besser_csv_"
LLM_TEMP_DIR_PREFIX = "besser_llm_"
USER_PROFILE_TEMP_DIR_PREFIX = "besser_user_profile_"
# Every prefix above; services/cleanup.py removes stale temp dirs that match one.
TEMP_DIR_PREFIXES = (
    TEMP_DIR_PREFIX,
    AGENT_TEMP_DIR_PREFIX,
    CSV_TEMP_DIR_PREFIX,
    LLM_TEMP_DIR_PREFIX,
    USER_PROFILE_TEMP_DIR_PREFIX,
)

# Upload size ceilings and accepted extensions, per upload kind (see
# ``services/utils/upload_validation.py``).
MAX_CSV_SIZE = 5 * 1024 * 1024       # 5 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024    # 10 MB
MAX_BUML_SIZE = 2 * 1024 * 1024      # 2 MB
# OWL/RDF ontology (``/kg/import-owl``).
MAX_OWL_SIZE = 5 * 1024 * 1024       # 5 MB
# Raw knowledge graph sent whole to the LLM (``/get-json-model-from-kg``).
MAX_KG_SIZE = 2 * 1024 * 1024        # 2 MB

ALLOWED_SPREADSHEET_EXTENSIONS = frozenset({".csv", ".xlsx"})
ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
ALLOWED_BUML_EXTENSIONS = frozenset({".py"})
ALLOWED_KG_EXTENSIONS = frozenset({".ttl", ".rdf", ".json"})
# Every RDF serialisation the OWL importer reads; the content of each is
# checked by ``validate_file_content``.
ALLOWED_OWL_EXTENSIONS = frozenset({".owl", ".ttl", ".rdf", ".xml", ".nt", ".n3"})
# The subset of the above that rdflib parses as RDF/XML (``guess_format`` ->
# ``"xml"``), i.e. the uploads that go through an XML parser.
RDF_XML_EXTENSIONS = frozenset({".owl", ".rdf", ".xml"})

# ---------------------------------------------------------------------
# Knowledge Graph (KG) endpoints
# ---------------------------------------------------------------------
# Model the Refine-KG LLM endpoints use when the request names none. The
# frontend never sends one, so in production this *is* the model. It follows
# the shared OpenAI default rather than restating it.
DEFAULT_KG_LLM_MODEL = DEFAULT_OPENAI_MODEL

# Seconds one consistency check (``/kg/consistency``, and the KG branch of
# ``/validate-diagram``) may spend before its pyshacl child process is killed
# (a user-authored sh:pattern can backtrack forever). The library's default.
KG_CONSISTENCY_TIMEOUT_SECONDS = DEFAULT_CONSISTENCY_TIMEOUT_SECONDS

# Consistency checks allowed to run at once in this process. Each one is a
# Python child process holding pyshacl + owlrl and up to the timeout above of
# CPU; a request beyond the cap is refused (HTTP 429), not queued.
KG_CONSISTENCY_MAX_CONCURRENT = 4

# Overall server deadline, in seconds, for one KG LLM request
# (``/kg/llm-clean``, ``/kg/classify-orphans``, ``/get-json-model-from-kg``),
# however many model calls it makes. Each call is given what is left of it as its
# own timeout. Published to the editor by ``GET /kg/config``.
KG_LLM_REQUEST_TIMEOUT_SECONDS = 120

# Largest number of orphan nodes one ``/kg/classify-orphans`` call may
# carry. The classifier sends one model call per ``MAX_ORPHAN_BATCH`` (40) nodes,
# sequentially, so this is two calls: what fits in the request deadline above
# with room to spare. The editor splits longer lists into several requests
# (it reads this value from ``GET /kg/config``).
MAX_ORPHAN_CLASSIFY_NODES = 80

# Deploy-time configuration (the BESSER_LLM_* caps and flags, SDMX limits,
# agent simulation) is in ``backend/settings.py``.

# Output defaults
OUTPUT_DIR_NAME = "output"
AGENT_MODEL_FILENAME = "agent_model.py"
AGENT_OUTPUT_FILENAME = "agent_output.zip"

# Generator defaults
# Not "standard": SQLGenerator passes this straight through as
# SQLAlchemyGenerator's `dbms`, so every /generate-output for `sql` without an
# explicit config.dialect raised "Invalid DBMS".
DEFAULT_SQL_DIALECT = "sqlite"
DEFAULT_DBMS = "sqlite"
DEFAULT_JSONSCHEMA_MODE = "regular"
DEFAULT_QISKIT_BACKEND = "aer_simulator"
DEFAULT_QISKIT_SHOTS = 1024
DEFAULT_DJANGO_PROJECT_NAME = "myproject"
DEFAULT_DJANGO_APP_NAME = "myapp"
DEFAULT_SUPABASE_USER_ROOT = "User"
# Owned by the generator (generators must not depend on the web backend), and
# re-exported here so the API layer has a single place to read them from.
DEFAULT_SPRING_BOOT_VERSION = _SPRING_DEFAULT_BOOT_VERSION
DEFAULT_JAVA_VERSION = _SPRING_DEFAULT_JAVA_VERSION
DEFAULT_SPRING_APP_NAME = _SPRING_DEFAULT_APP_NAME
DEFAULT_SPRING_PACKAGE_NAME = _SPRING_DEFAULT_PACKAGE_NAME
# Only the web editor wraps the generated sources in a named project folder.
DEFAULT_SPRING_PROJECT_NAME = "springproject"

# CORS defaults
DEFAULT_CORS_ORIGINS = [
    "https://editor.besser-pearl.org",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
    # Browsers treat 127.0.0.1 as a different origin from localhost.
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
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

# UML Deployment diagram type (the Docker Compose generator's input).
DEPLOYMENT_DIAGRAM_TYPE = "DeploymentDiagram"

# Knowledge Graph diagram type (the KG editor's model, see ``routers/kg_router.py``).
KG_DIAGRAM_TYPE = "KnowledgeGraphDiagram"

# ---------------------------------------------------------------------------
# UML Component / Deployment stereotype tokens
# Lower-cased stereotype tokens the converters map to typed metamodel slots;
# every other token is kept verbatim in the element's free-form `stereotypes`.
# ---------------------------------------------------------------------------

# Stereotype tokens that promote a bare Component to a subclass. `llm`/`db`/`rag`
# are Tool subclasses -- see uml_component/agentic.py.
COMPONENT_SUBTYPE_TOKENS = {
    "skill": "Skill",
    "tool": "Tool",
    "llm": "LLM",
    "db": "Database",
    "rag": "RAG",
}

# Locality tokens (shared Component/Deployment side).
LOCALITY_TOKENS = frozenset(locality.value for locality in ComponentLocality)

# AgentCategory tokens (Component side). NONE has no token: it is the default.
AGENT_CATEGORY_TOKENS = frozenset(
    category.value for category in AgentCategory if category is not AgentCategory.NONE
)

# AgenticEdgeKind tokens (Component side, on ComponentDependency edges).
AGENTIC_EDGE_KIND_TOKENS = frozenset(kind.value for kind in AgenticEdgeKind)

# NodeKind tokens (Deployment side) -> NodeKind. GENERIC is the default kind and
# has no token of its own: a literal «node» stereotype is kept as free text.
NODE_KIND_TOKENS = {
    kind.value.lower(): kind for kind in NodeKind if kind is not NodeKind.GENERIC
}


# ---------------------------------------------------------------------------
# Agent simulator (live agent testing, /besser_api/simulation/*)
# ---------------------------------------------------------------------------
# Header carrying the secret shared with the simulator (AGENT_SIMULATOR_API_TOKEN).
AGENT_SIMULATOR_TOKEN_HEADER = "X-Agent-Simulator-Token"

# The simulator kills sessions after this long by default; the backend's
# session-ownership map forgets them after the same delay.
DEFAULT_AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS = 900

# Fallback config.yaml when the BAF generator does not emit one.
AGENT_SIMULATOR_FALLBACK_WS_PORT = 7700

# Timeouts (seconds) for calls to the simulator service.
AGENT_SIMULATOR_CREATE_TIMEOUT_SECONDS = 30
AGENT_SIMULATOR_FILES_TIMEOUT_SECONDS = 10
AGENT_SIMULATOR_DELETE_TIMEOUT_SECONDS = 5
AGENT_SIMULATOR_WS_OPEN_TIMEOUT_SECONDS = 15

# Frontend -> backend WebSocket: the first frame must be
# {"type": "auth", "githubSession": "..."} within this many seconds.
AGENT_SIMULATOR_WS_AUTH_TIMEOUT_SECONDS = 10

# WebSocket close codes sent to the frontend.
WS_CLOSE_BAD_REQUEST = 4400
WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_NOT_FOUND = 4404
WS_CLOSE_RATE_LIMITED = 4429
