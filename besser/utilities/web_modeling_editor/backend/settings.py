"""Environment configuration of the web editor backend.

Every variable the backend reads is declared once on :class:`BackendSettings`
with its type, default and meaning. Two ways to read them:

- :func:`get_settings` returns one instance per process, built on first use.
  It holds the values the backend reads once (startup configuration).
- ``BackendSettings()`` reads the environment again. The fields marked
  "read per call" are read this way at each use, so a change to the
  environment reaches the next request.

A malformed value never raises: each field falls back to its default the way
its documented parsing says, so one bad variable cannot stop the backend.
"""

from __future__ import annotations

import functools
import hmac
import os
from typing import Annotated, Optional

from pydantic import BeforeValidator, Field, SecretStr, model_validator
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, SettingsConfigDict

_TRUTHY = ("1", "true", "yes", "on")


def _positive(parse):
    """A blank, unparsable or non-positive value gives the default."""
    def validate(value):
        if not value:
            raise PydanticUseDefault()
        try:
            parsed = parse(value)
        except (TypeError, ValueError):
            raise PydanticUseDefault() from None
        if not parsed > 0:
            raise PydanticUseDefault()
        return parsed
    return validate


def _number_or_default(parse):
    """A blank or unparsable value gives the default; any number is kept."""
    def validate(value):
        raw = str(value).strip()
        if not raw:
            raise PydanticUseDefault()
        try:
            return parse(raw)
        except ValueError:
            raise PydanticUseDefault() from None
    return validate


def _flag(value) -> bool:
    """Set means on only for 1/true/yes/on; set but blank is off."""
    return str(value).strip().lower() in _TRUTHY


def _flag_or_default(value) -> bool:
    """Like ``_flag``, but a blank value gives the default."""
    raw = str(value).strip().lower()
    if not raw:
        raise PydanticUseDefault()
    return raw in _TRUTHY


def _abspath(value) -> Optional[str]:
    """Absolute path with ``~`` expanded; blank is unset."""
    raw = str(value).strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else None


def _one_or_true(value) -> bool:
    return str(value).strip().lower() in ("1", "true")


_DEFAULT_AGENT_SIMULATOR_URL = "http://besser-wme-agent-simulator:8001"


def _simulator_url(value) -> str:
    """A blank value gives the default; a trailing slash is dropped."""
    return (value or _DEFAULT_AGENT_SIMULATOR_URL).rstrip("/")


_PositiveInt = Annotated[int, BeforeValidator(_positive(int))]
_PositiveFloat = Annotated[float, BeforeValidator(_positive(float))]
_Flag = Annotated[bool, BeforeValidator(_flag)]
_FlagOrDefault = Annotated[bool, BeforeValidator(_flag_or_default)]
_IntOrDefault = Annotated[Optional[int], BeforeValidator(_number_or_default(int))]
_FloatOrDefault = Annotated[Optional[float], BeforeValidator(_number_or_default(float))]
_AbsPath = Annotated[Optional[str], BeforeValidator(_abspath)]


class BackendSettings(BaseSettings):
    """The backend's environment variables (all optional)."""

    # Defaults are final values, not raw text: validating them would parse
    # them again (and loop on a default the parser rejects, such as None).
    model_config = SettingsConfigDict(
        case_sensitive=True, extra="ignore", validate_default=False)

    # -- B-UML export and GitHub deploys ----------------------------------
    buml_emit_ids: _Flag = Field(
        False, validation_alias="BESSER_BUML_EMIT_IDS",
        description="Write element ids into exported B-UML; releases before Element.id cannot load them.")
    deploy_from_repo: _Flag = Field(
        False, validation_alias="BESSER_DEPLOY_FROM_REPO",
        description="'Go live' deploys the project's own repository instead of a regenerated copy.")

    # -- Spec-Driven Agent: per-run caps -----------------------------------
    # The UI reads the hard caps from /spec-driven/config, so they are what a
    # run is actually allowed. The turns cap fits models that make one tool
    # call per turn (about 4x the turns of one that batches).
    llm_max_cost_usd_hard_cap: _PositiveFloat = Field(
        5.0, validation_alias="BESSER_LLM_MAX_COST_USD_HARD_CAP",
        description="Ceiling (USD) a run's max_cost_usd is clamped to.")
    llm_max_runtime_seconds_hard_cap: _PositiveInt = Field(
        2400, validation_alias="BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP",
        description="Ceiling (seconds) a run's max_runtime_seconds is clamped to.")
    llm_max_turns_hard_cap: _PositiveInt = Field(
        150, validation_alias="BESSER_LLM_MAX_TURNS_HARD_CAP",
        description="Ceiling a run's max_turns is clamped to.")
    llm_default_max_cost_usd: _PositiveFloat = Field(
        5.0, validation_alias="BESSER_LLM_DEFAULT_MAX_COST_USD",
        description="max_cost_usd of a request that sends none; at most the hard cap.")
    llm_default_max_runtime_seconds: _PositiveInt = Field(
        2400, validation_alias="BESSER_LLM_DEFAULT_MAX_RUNTIME_SECONDS",
        description="max_runtime_seconds of a request that sends none; at most the hard cap.")
    llm_default_max_turns: _PositiveInt = Field(
        120, validation_alias="BESSER_LLM_DEFAULT_MAX_TURNS",
        description="max_turns of a request that sends none; at most the hard cap.")
    llm_watchdog_grace_seconds: _PositiveInt = Field(
        120, validation_alias="BESSER_LLM_WATCHDOG_GRACE_SECONDS",
        description="Seconds past a run's runtime cap before the watchdog force-cancels it.")

    # -- Spec-Driven Agent: runs, downloads and concurrency ----------------
    llm_download_ttl_seconds: _PositiveInt = Field(
        1800, validation_alias="BESSER_LLM_DOWNLOAD_TTL_SECONDS",
        description="How long a finished run's download stays available.")
    llm_cost_emitter_interval_seconds: _PositiveFloat = Field(
        2.0, validation_alias="BESSER_LLM_COST_EMITTER_INTERVAL_SECONDS",
        description="Seconds between two cost events on a run's stream.")
    llm_cancel_abandoned_runs: _Flag = Field(
        True, validation_alias="BESSER_LLM_CANCEL_ABANDONED_RUNS",
        description="Cancel a run nobody has followed for the disconnect grace period.")
    llm_disconnected_grace_seconds: _PositiveInt = Field(
        300, validation_alias="BESSER_LLM_DISCONNECTED_GRACE_SECONDS",
        description="Seconds a run may have no subscriber before it counts as abandoned.")
    llm_run_workspace_root: _AbsPath = Field(
        None, validation_alias="BESSER_LLM_RUN_WORKSPACE_ROOT",
        description="Persistent parent of run workspaces and checkpoints (unset: the system temp dir).")
    llm_max_concurrent_runs: _PositiveInt = Field(
        10, validation_alias="BESSER_LLM_MAX_CONCURRENT_RUNS",
        description="Runs (and other LLM jobs) allowed at once.")
    llm_run_store_path: _AbsPath = Field(
        None, validation_alias="BESSER_LLM_RUN_STORE_PATH",
        description="SQLite file of the durable run store (unset: the system temp dir). Read per call.")

    # -- Spec-Driven Agent: feature flags ----------------------------------
    llm_enable_tracing: _Flag = Field(
        True, validation_alias="BESSER_LLM_ENABLE_TRACING",
        description="Write the per-run trace.")
    llm_enable_checkpointing: _Flag = Field(
        True, validation_alias="BESSER_LLM_ENABLE_CHECKPOINTING",
        description="Write per-turn checkpoints, which resume needs.")
    llm_per_write_diagnostics: _Flag = Field(
        True, validation_alias="BESSER_LLM_PER_WRITE_DIAGNOSTICS",
        description="Check each file the agent writes as it writes it.")
    llm_enable_toolchain_validation: _Flag = Field(
        False, validation_alias="BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION",
        description="Phase 3 compiles TypeScript, Rust and Kotlin projects (needs the toolchains).")
    llm_enable_auto_fix: _Flag = Field(
        True, validation_alias="BESSER_LLM_ENABLE_AUTO_FIX",
        description="Phase 3 gives the agent a bounded loop to fix blocker issues.")
    # Off: on a shared BYOK host they are user-steerable remote code execution.
    llm_enable_shell_tools: _Flag = Field(
        False, validation_alias="BESSER_LLM_ENABLE_SHELL_TOOLS",
        description="Let the agent run shell commands (run_command, install_dependencies).")
    # Runs LLM-authored Python on the host, with the stripped subprocess
    # environment, no network and a 30 s timeout.
    llm_enable_import_smoke_check: _Flag = Field(
        True, validation_alias="BESSER_LLM_ENABLE_IMPORT_SMOKE_CHECK",
        description="Phase 3 imports the generated ORM module in a subprocess.")
    llm_enable_requirements_ledger: _Flag = Field(
        True, validation_alias="BESSER_LLM_ENABLE_REQUIREMENTS_LEDGER",
        description="Check each requirement of the user's request against the generated code.")
    # Off: a server opening any user-given URL is an SSRF vector. Local and
    # on-premises installs turn it on; catalog gateways do not need it.
    llm_allow_custom_base_url: _Flag = Field(
        False, validation_alias="BESSER_LLM_ALLOW_CUSTOM_BASE_URL",
        description="Accept a run request's own LLM base_url (an SSRF risk on a shared host).")
    demo_token: SecretStr = Field(
        SecretStr(""), validation_alias="BESSER_DEMO_TOKEN",
        description="Secret that unlocks the sponsored tier; unset refuses it. Read per call.")

    # -- Telemetry and incidents -------------------------------------------
    telemetry_enabled: Annotated[bool, BeforeValidator(_one_or_true)] = Field(
        False, validation_alias="BESSER_TELEMETRY_ENABLED",
        description="Study telemetry master switch, on only for 1 or true. Read per call.")
    telemetry_dir: Optional[str] = Field(
        None, validation_alias="BESSER_TELEMETRY_DIR",
        description="Study telemetry directory (unset: /app/telemetry). Read per call.")
    telemetry_admin_token: SecretStr = Field(
        SecretStr(""), validation_alias="BESSER_TELEMETRY_ADMIN_TOKEN",
        description="Token GET /telemetry/report requires; unset answers 404. Read per call.")
    incident_log_dir: Optional[str] = Field(
        None, validation_alias="BESSER_INCIDENT_LOG_DIR",
        description="Incident log directory (unset: /app/incidents); set, it also keeps run traces. "
                    "Read per call.")

    # -- SDMX import ---------------------------------------------------------
    # The largest message seen (Eurostat Comext DS-045409) is 23.6 MB; parsing
    # takes about 15 times a message's size in memory.
    sdmx_timeout_seconds: _PositiveFloat = Field(
        60.0, validation_alias="BESSER_SDMX_TIMEOUT",
        description="Seconds for one whole SDMX import.")
    sdmx_max_mb: _PositiveFloat = Field(
        32.0, validation_alias="BESSER_SDMX_MAX_MB",
        description="Largest single SDMX structure message, in MB.")
    sdmx_max_total_mb: _PositiveFloat = Field(
        64.0, validation_alias="BESSER_SDMX_MAX_TOTAL_MB",
        description="All the messages of one SDMX import together, in MB.")
    sdmx_max_concurrent_imports: _PositiveInt = Field(
        2, validation_alias="BESSER_SDMX_MAX_CONCURRENT_IMPORTS",
        description="SDMX imports running at once; more are refused with 503.")
    sdmx_allow_custom_url: _Flag = Field(
        False, validation_alias="BESSER_SDMX_ALLOW_CUSTOM_URL",
        description="Accept an SDMX API that is not one of the presets.")

    # -- Semantic check (the Alloy Analyzer) ----------------------------------
    # Behind /semantic-consistency-check and /generate-object-diagram.
    alloy_max_concurrent_checks: _PositiveInt = Field(
        1, validation_alias="BESSER_ALLOY_MAX_CONCURRENT_CHECKS",
        description="Semantic checks running at once, each in its own JVM; more are refused with 503.")
    alloy_max_heap_mb: _PositiveInt = Field(
        1024, validation_alias="BESSER_ALLOY_MAX_HEAP_MB",
        description="Maximum heap of each Alloy Analyzer JVM (-Xmx), in MB, so that a large search "
                    "cannot exhaust the host's memory.")

    # -- Agent simulation ----------------------------------------------------
    # The rate limiter is per process: the backend runs one uvicorn worker.
    agent_simulator_url: Annotated[str, BeforeValidator(_simulator_url)] = Field(
        _DEFAULT_AGENT_SIMULATOR_URL, validation_alias="AGENT_SIMULATOR_URL",
        description="Base URL of the agent simulator service.")
    agent_simulator_api_token: SecretStr = Field(
        SecretStr(""), validation_alias="AGENT_SIMULATOR_API_TOKEN",
        description="Secret shared with the simulator; unset, simulation answers 503. Read per call.")
    agent_simulator_require_auth: _FlagOrDefault = Field(
        True, validation_alias="AGENT_SIMULATOR_REQUIRE_AUTH",
        description="Require GitHub sign-in for simulation.")
    agent_simulator_restrict_custom_code: _FlagOrDefault = Field(
        True, validation_alias="AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE",
        description="Refuse to simulate agents with custom Python code actions.")
    agent_simulator_rate_limit_window_seconds: _IntOrDefault = Field(
        60, validation_alias="AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS",
        description="Window of the per-actor simulation rate limit.")
    agent_simulator_rate_limit_max_requests: _IntOrDefault = Field(
        12, validation_alias="AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS",
        description="Requests one actor may make per window.")
    agent_simulator_rate_limit_max_keys: _IntOrDefault = Field(
        10000, validation_alias="AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS",
        description="Actors the rate limiter tracks at once.")
    agent_simulator_max_sessions_per_actor: _IntOrDefault = Field(
        1, validation_alias="AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR",
        description="Simulation sessions one actor may hold at once.")
    agent_simulator_memory_mb: _IntOrDefault = Field(
        None, validation_alias="AGENT_SIMULATOR_MEMORY_MB",
        description="Memory limit reported by /simulation/limits.")
    agent_simulator_cpu_cores: _FloatOrDefault = Field(
        None, validation_alias="AGENT_SIMULATOR_CPU_CORES",
        description="CPU limit reported by /simulation/limits.")
    agent_simulator_disk_mb: _IntOrDefault = Field(
        None, validation_alias="AGENT_SIMULATOR_DISK_MB",
        description="Disk limit reported by /simulation/limits.")
    agent_simulator_session_lifetime_seconds: _IntOrDefault = Field(
        None, validation_alias="AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS",
        description="Session lifetime reported by /simulation/limits (unset: the simulator's 900 s).")
    agent_simulator_quota_enabled: _FlagOrDefault = Field(
        False, validation_alias="AGENT_SIMULATOR_QUOTA_ENABLED",
        description="Quota flag reported by /simulation/limits.")

    # -- GitHub sign-in --------------------------------------------------------
    github_client_id: str = Field(
        "", validation_alias="GITHUB_CLIENT_ID",
        description="GitHub OAuth app client id; unset disables GitHub sign-in.")
    github_client_secret: SecretStr = Field(
        SecretStr(""), validation_alias="GITHUB_CLIENT_SECRET",
        description="GitHub OAuth app client secret.")
    github_redirect_uri: str = Field(
        "http://localhost:9000/besser_api/github/auth/callback", validation_alias="GITHUB_REDIRECT_URI",
        description="OAuth callback URL registered on the GitHub app.")
    deployment_url: str = Field(
        "http://localhost:8080", validation_alias="DEPLOYMENT_URL",
        description="Editor URL the OAuth flow redirects back to.")
    oauth_sessions_path: Optional[str] = Field(
        None, validation_alias="OAUTH_SESSIONS_PATH",
        description="Encrypted store of pending OAuth states (unset: SESSION_STORE_PATH).")
    user_tokens_path: Optional[str] = Field(
        None, validation_alias="USER_TOKENS_PATH",
        description="Encrypted store of signed-in GitHub sessions (unset: SESSION_STORE_PATH).")
    session_store_path: Optional[str] = Field(
        None, validation_alias="SESSION_STORE_PATH",
        description="Session store used when no path is given (unset: <tempdir>/besser_sessions.enc). "
                    "Read per store.")

    # -- Other services ----------------------------------------------------------
    cors_origins: str = Field(
        "", validation_alias="CORS_ORIGINS",
        description="Comma-separated allowed origins; unset keeps the built-in list.")
    wme_node_server_url: str = Field(
        "http://localhost:8080", validation_alias="WME_NODE_SERVER_URL",
        description="WME Node server that renders SVG for /get-svg. Read per call.")
    feedback_email: str = Field(
        "", validation_alias="FEEDBACK_EMAIL",
        description="Comma-separated feedback recipients. Read per call.")
    smtp_host: str = Field(
        "smtp.gmail.com", validation_alias="SMTP_HOST",
        description="SMTP server for feedback mail. Read per call.")
    # Kept as text: an invalid port fails the feedback mail, not the settings.
    smtp_port: str = Field(
        "587", validation_alias="SMTP_PORT",
        description="SMTP port for feedback mail. Read per call.")
    smtp_username: Optional[str] = Field(
        None, validation_alias="SMTP_USERNAME",
        description="SMTP user (unset: the first FEEDBACK_EMAIL recipient). Read per call.")
    smtp_password: SecretStr = Field(
        SecretStr(""), validation_alias="SMTP_PASSWORD",
        description="SMTP password; unset, feedback is only stored locally. Read per call.")

    @model_validator(mode="after")
    def _defaults_within_hard_caps(self) -> "BackendSettings":
        self.llm_default_max_cost_usd = min(self.llm_default_max_cost_usd, self.llm_max_cost_usd_hard_cap)
        self.llm_default_max_runtime_seconds = min(
            self.llm_default_max_runtime_seconds, self.llm_max_runtime_seconds_hard_cap)
        self.llm_default_max_turns = min(self.llm_default_max_turns, self.llm_max_turns_hard_cap)
        return self


@functools.cache
def get_settings() -> BackendSettings:
    """The process-wide settings, read from the environment on first use."""
    return BackendSettings()


def is_demo_token(supplied: str) -> bool:
    """True when ``supplied`` is the demo secret; always False while ``BESSER_DEMO_TOKEN`` is unset."""
    expected = BackendSettings().demo_token.get_secret_value().strip()
    return bool(expected) and hmac.compare_digest(supplied.strip().encode("utf-8"), expected.encode("utf-8"))
