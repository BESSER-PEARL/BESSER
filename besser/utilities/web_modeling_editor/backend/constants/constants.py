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
# Set to $5 to match the from-scratch ceiling below: the UI reads this hard
# cap from /smart-gen/config, so a lower value here was DISHONEST — it showed
# "$2" while a from-scratch run was already permitted up to $5. One honest
# ceiling everywhere. (The DEFAULT below stays $1 — the cap is the max a user
# can opt into, not what a normal run spends.)
LLM_MAX_COST_USD_HARD_CAP = _env_float("BESSER_LLM_MAX_COST_USD_HARD_CAP", 5.0)
LLM_MAX_RUNTIME_SECONDS_HARD_CAP = _env_int("BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP", 900)
LLM_DEFAULT_MAX_COST_USD = min(
    _env_float("BESSER_LLM_DEFAULT_MAX_COST_USD", 1.0),
    LLM_MAX_COST_USD_HARD_CAP,
)
LLM_DEFAULT_MAX_RUNTIME_SECONDS = min(
    _env_int("BESSER_LLM_DEFAULT_MAX_RUNTIME_SECONDS", 600),
    LLM_MAX_RUNTIME_SECONDS_HARD_CAP,
)

# Adaptive ceiling for FROM-SCRATCH runs — i.e. Phase 1 found no
# deterministic BESSER generator to run (no domain/quantum model at all,
# or the request targets a stack BESSER doesn't scaffold, e.g. Next.js /
# Rust / Kotlin). Those runs have no Phase-1 output to build on top of:
# the LLM has to author the entire application in Phase 2, which
# legitimately costs and takes longer than the common case (a Python
# scaffold that Phase 2 only patches). A benchmark showed from-scratch
# runs blowing the $2.00 default hard cap (up to $2.51) and truncating
# output as a result (hitting the output-token limit mid-write).
#
# ``LLMOrchestrator._apply_adaptive_budget`` raises the run's internal
# ``max_cost_usd`` / ``max_runtime_seconds`` to these values ONLY when
# Phase 1 ran no generator; the scaffolded-Python default above (and its
# hard cap) is untouched, so that common case's cost profile doesn't
# change. Deliberately NOT clamped against ``LLM_MAX_COST_USD_HARD_CAP``
# / ``LLM_MAX_RUNTIME_SECONDS_HARD_CAP`` above — those are the ceiling
# for the common case; this is a separate, larger ceiling for the harder
# one. Env-tunable so ops can raise/lower it per deploy without a code
# change.
LLM_FROM_SCRATCH_MAX_COST_USD_HARD_CAP = _env_float(
    "BESSER_LLM_FROM_SCRATCH_MAX_COST_USD_HARD_CAP", 5.0,
)
LLM_FROM_SCRATCH_MAX_RUNTIME_SECONDS_HARD_CAP = _env_int(
    "BESSER_LLM_FROM_SCRATCH_MAX_RUNTIME_SECONDS_HARD_CAP", 1800,
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

# Phase 3 toolchain validation (tsc / cargo / kotlinc) compiles real
# projects server-side and can add minutes of wall-clock + real billing
# to a run. Off by default in the web deployment — the cheap checks
# (ast.parse, Dockerfile refs, ruff, pip dry-run) always run. Enable
# per deploy when the host has the toolchains installed and the extra
# latency is acceptable.
LLM_ENABLE_TOOLCHAIN_VALIDATION = _env_bool(
    "BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION", False,
)

# Phase 3 auto-fix: when validation finds BLOCKER issues (Python syntax
# errors, broken Dockerfile refs, dependency conflicts — the "app doesn't
# even start" class), give the LLM a bounded fix loop (up to 3 rounds x
# 5 turns, snapshot/rollback if fixes regress) instead of shipping the
# broken artifact as a green success. ON by default for the web deploy:
# without it Phase 3 is validate-and-report only. The fix loop enforces
# the run's cancellation / cost-cap / runtime-cap per turn and uses the
# same guarded tool executor as Phase 2 (no new capability surface).
LLM_ENABLE_AUTO_FIX = _env_bool("BESSER_LLM_ENABLE_AUTO_FIX", True)

# Whether the Phase-2 / Phase-3 agent may use the arbitrary-shell tools
# (run_command / install_dependencies). OFF by default for the hosted deploy:
# they execute LLM/user-authored commands with shell=True in the backend
# process, and the cwd lock + denylist are UX, not a sandbox — on a shared,
# BYOK, ministry-facing box that is user-steerable remote code execution and
# server-secret exfiltration. The agent keeps every static tool (read/write/
# modify/check_syntax). Trusted local/CLI/bench runs can re-enable via the env
# var. Durable answer is per-run container isolation.
LLM_ENABLE_SHELL_TOOLS = _env_bool("BESSER_LLM_ENABLE_SHELL_TOOLS", False)

# Grace period added on top of the request's max_runtime_seconds before
# the runner-level watchdog force-cancels a run (covers Phase 3 +
# packaging time after the Phase 2 loop hits its own runtime check).
LLM_WATCHDOG_GRACE_SECONDS = _env_int("BESSER_LLM_WATCHDOG_GRACE_SECONDS", 120)

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

