import os
from typing import Optional

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


# ---------------------------------------------------------------------------
# Agent simulator (live agent testing, /besser_api/simulation/*)
# Every value below is read ONCE, at import time, from the environment.
# ---------------------------------------------------------------------------
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in _TRUTHY_ENV_VALUES if raw else default


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


# Base URL of the isolated simulator service (docker compose service name).
DEFAULT_AGENT_SIMULATOR_URL = "http://besser-wme-agent-simulator:8001"
AGENT_SIMULATOR_URL = (os.environ.get("AGENT_SIMULATOR_URL") or DEFAULT_AGENT_SIMULATOR_URL).rstrip("/")

# Shared secret the backend sends on every HTTP request and on the WS
# handshake to the simulator. The token itself is read per request so a
# missing value fails that request with 503 (the simulator fails closed too).
AGENT_SIMULATOR_TOKEN_ENV_VAR = "AGENT_SIMULATOR_API_TOKEN"
AGENT_SIMULATOR_TOKEN_HEADER = "X-Agent-Simulator-Token"

# GitHub sign-in gate for the simulation endpoints (on by default).
AGENT_SIMULATOR_REQUIRE_AUTH = _env_bool("AGENT_SIMULATOR_REQUIRE_AUTH", True)

# Refuse to simulate agents that contain custom Python code bodies (on by
# default; operators may opt out). Has no effect on code generation.
AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE = _env_bool("AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", True)

# Per-actor sliding-window rate limit on the simulation endpoints. The
# limiter is in-process: the backend runs a single uvicorn process
# (see backend.py __main__ and the Dockerfile CMD). Running several workers
# would give each worker its own budget.
AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS = _env_int("AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS", 60)
AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS = _env_int("AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS", 12)
# Upper bound on the number of actors the limiter tracks at once.
AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS = _env_int("AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS", 10000)

# How many simulation sessions one actor may hold at the same time.
AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR = _env_int("AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR", 1)

# Resource limits advertised by GET /simulation/limits (None = not configured).
AGENT_SIMULATOR_MEMORY_MB = _env_int("AGENT_SIMULATOR_MEMORY_MB", None)
AGENT_SIMULATOR_CPU_CORES = _env_float("AGENT_SIMULATOR_CPU_CORES", None)
AGENT_SIMULATOR_DISK_MB = _env_int("AGENT_SIMULATOR_DISK_MB", None)
AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS = _env_int("AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS", None)
AGENT_SIMULATOR_QUOTA_ENABLED = _env_bool("AGENT_SIMULATOR_QUOTA_ENABLED", False)

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
