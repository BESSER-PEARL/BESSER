# API Configuration
API_VERSION = "1.0.0"

# Temp directory prefixes
TEMP_DIR_PREFIX = "besser_"
AGENT_TEMP_DIR_PREFIX = "besser_agent_"
CSV_TEMP_DIR_PREFIX = "besser_csv_"
LLM_TEMP_DIR_PREFIX = "besser_llm_"

# ---------------------------------------------------------------------
# LLM smart-generation caps & feature flags
# ---------------------------------------------------------------------
# All of these can be overridden at deploy time via environment
# variables. Reading env once at import keeps the hot path free of
# ``os.environ`` lookups and lets tests monkeypatch the module-level
# constants if they need to. Sensible server-side hard defaults remain
# here so a forgotten env var never exposes unbounded cost / runtime.


def _env_float(name: str, default: float) -> float:
    import os as _os
    value = _os.environ.get(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_int(name: str, default: int) -> int:
    import os as _os
    value = _os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    import os as _os
    value = _os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Per-run spend / runtime caps. The HARD_CAP values are the absolute
# ceiling a request's fields are clamped to; DEFAULT values are what
# clients get when they don't send an explicit number.
LLM_MAX_COST_USD_HARD_CAP = _env_float("BESSER_LLM_MAX_COST_USD_HARD_CAP", 2.0)
LLM_MAX_RUNTIME_SECONDS_HARD_CAP = _env_int("BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP", 900)
LLM_DEFAULT_MAX_COST_USD = min(
    _env_float("BESSER_LLM_DEFAULT_MAX_COST_USD", 1.0),
    LLM_MAX_COST_USD_HARD_CAP,
)
LLM_DEFAULT_MAX_RUNTIME_SECONDS = min(
    _env_int("BESSER_LLM_DEFAULT_MAX_RUNTIME_SECONDS", 600),
    LLM_MAX_RUNTIME_SECONDS_HARD_CAP,
)

# Generated-output TTL + SSE cadence.
LLM_DOWNLOAD_TTL_SECONDS = _env_int("BESSER_LLM_DOWNLOAD_TTL_SECONDS", 1800)
LLM_COST_EMITTER_INTERVAL_SECONDS = _env_float(
    "BESSER_LLM_COST_EMITTER_INTERVAL_SECONDS", 2.0,
)

# Concurrency cap. Each in-flight smart-generation run holds a worker
# thread, a temp dir, and an SSE connection. 10 is a sensible default
# for a modest VM; bump for beefier hosts, lower for shared infra.
LLM_MAX_CONCURRENT_RUNS = _env_int("BESSER_LLM_MAX_CONCURRENT_RUNS", 10)

# Observability + crash-recovery toggles. Tracing is cheap; disabling
# it is mostly useful for deployments with strict log-footprint caps.
# Checkpointing adds a per-turn file write — also cheap, but the
# toggle lets deployments opt out if disk volume is tight.
LLM_ENABLE_TRACING = _env_bool("BESSER_LLM_ENABLE_TRACING", True)
LLM_ENABLE_CHECKPOINTING = _env_bool("BESSER_LLM_ENABLE_CHECKPOINTING", True)

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
