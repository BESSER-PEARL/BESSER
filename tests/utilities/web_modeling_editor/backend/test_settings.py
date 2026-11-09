"""BackendSettings reads every variable exactly as the code it replaced did.

Each row pairs the expression the backend used before the settings class
(``old``, kept verbatim, reading ``os.environ``) with how the backend reads the
same value now (``new``). Every row is checked unset and with a list of edge
values, so a change to a variable's name, default or parsing fails here.
"""

import os
import sys
import tempfile

import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven import telemetry
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings, get_settings


# -- The helpers of constants.py before the settings class, verbatim ---------

def _env_float(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_path(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return os.path.abspath(os.path.expanduser(value))


def _sim_env_bool(name, default):
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"} if raw else default


def _sim_env_int(name, default):
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _sim_env_float(name, default):
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _get(name, *default):
    return os.environ.get(name, *default)


def _secret(value):
    return value.get_secret_value()


def _old_run_store_path():
    configured = os.environ.get("BESSER_LLM_RUN_STORE_PATH", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(tempfile.gettempdir(), "x.db")


def _new_session_store_path(s):
    return s.session_store_path if s.session_store_path is not None else "<default>"


def _new_smtp_username(s):
    return s.smtp_username if s.smtp_username is not None else "first@example.org"


# (field, variable, default, old read, new read)
ROWS = [
    ("buml_emit_ids", "BESSER_BUML_EMIT_IDS", False,
     lambda: _env_bool("BESSER_BUML_EMIT_IDS", False), lambda s: s.buml_emit_ids),
    ("deploy_from_repo", "BESSER_DEPLOY_FROM_REPO", False,
     lambda: _env_bool("BESSER_DEPLOY_FROM_REPO", False), lambda s: s.deploy_from_repo),
    ("docker_deploy_enabled", "BESSER_ENABLE_DOCKER_DEPLOY", False,
     lambda: _env_bool("BESSER_ENABLE_DOCKER_DEPLOY", False), lambda s: s.docker_deploy_enabled),
    ("llm_max_cost_usd_hard_cap", "BESSER_LLM_MAX_COST_USD_HARD_CAP", 5.0,
     lambda: _env_float("BESSER_LLM_MAX_COST_USD_HARD_CAP", 5.0), lambda s: s.llm_max_cost_usd_hard_cap),
    ("llm_max_runtime_seconds_hard_cap", "BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP", 2400,
     lambda: _env_int("BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP", 2400),
     lambda s: s.llm_max_runtime_seconds_hard_cap),
    ("llm_max_turns_hard_cap", "BESSER_LLM_MAX_TURNS_HARD_CAP", 150,
     lambda: _env_int("BESSER_LLM_MAX_TURNS_HARD_CAP", 150), lambda s: s.llm_max_turns_hard_cap),
    ("llm_default_max_cost_usd", "BESSER_LLM_DEFAULT_MAX_COST_USD", 5.0,
     lambda: min(_env_float("BESSER_LLM_DEFAULT_MAX_COST_USD", 5.0),
                 _env_float("BESSER_LLM_MAX_COST_USD_HARD_CAP", 5.0)),
     lambda s: s.llm_default_max_cost_usd),
    ("llm_default_max_runtime_seconds", "BESSER_LLM_DEFAULT_MAX_RUNTIME_SECONDS", 2400,
     lambda: min(_env_int("BESSER_LLM_DEFAULT_MAX_RUNTIME_SECONDS", 2400),
                 _env_int("BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP", 2400)),
     lambda s: s.llm_default_max_runtime_seconds),
    ("llm_default_max_turns", "BESSER_LLM_DEFAULT_MAX_TURNS", 120,
     lambda: min(_env_int("BESSER_LLM_DEFAULT_MAX_TURNS", 120), _env_int("BESSER_LLM_MAX_TURNS_HARD_CAP", 150)),
     lambda s: s.llm_default_max_turns),
    ("llm_watchdog_grace_seconds", "BESSER_LLM_WATCHDOG_GRACE_SECONDS", 120,
     lambda: _env_int("BESSER_LLM_WATCHDOG_GRACE_SECONDS", 120), lambda s: s.llm_watchdog_grace_seconds),
    ("llm_download_ttl_seconds", "BESSER_LLM_DOWNLOAD_TTL_SECONDS", 1800,
     lambda: _env_int("BESSER_LLM_DOWNLOAD_TTL_SECONDS", 1800), lambda s: s.llm_download_ttl_seconds),
    ("llm_cost_emitter_interval_seconds", "BESSER_LLM_COST_EMITTER_INTERVAL_SECONDS", 2.0,
     lambda: _env_float("BESSER_LLM_COST_EMITTER_INTERVAL_SECONDS", 2.0),
     lambda s: s.llm_cost_emitter_interval_seconds),
    ("llm_cancel_abandoned_runs", "BESSER_LLM_CANCEL_ABANDONED_RUNS", True,
     lambda: _env_bool("BESSER_LLM_CANCEL_ABANDONED_RUNS", True), lambda s: s.llm_cancel_abandoned_runs),
    ("llm_disconnected_grace_seconds", "BESSER_LLM_DISCONNECTED_GRACE_SECONDS", 300,
     lambda: max(0, _env_int("BESSER_LLM_DISCONNECTED_GRACE_SECONDS", 300)),
     lambda s: s.llm_disconnected_grace_seconds),
    ("llm_run_workspace_root", "BESSER_LLM_RUN_WORKSPACE_ROOT", None,
     lambda: _env_path("BESSER_LLM_RUN_WORKSPACE_ROOT"), lambda s: s.llm_run_workspace_root),
    ("llm_max_concurrent_runs", "BESSER_LLM_MAX_CONCURRENT_RUNS", 10,
     lambda: _env_int("BESSER_LLM_MAX_CONCURRENT_RUNS", 10), lambda s: s.llm_max_concurrent_runs),
    ("llm_run_store_path", "BESSER_LLM_RUN_STORE_PATH", None,
     _old_run_store_path,
     lambda s: s.llm_run_store_path or os.path.join(tempfile.gettempdir(), "x.db")),
    ("llm_enable_tracing", "BESSER_LLM_ENABLE_TRACING", True,
     lambda: _env_bool("BESSER_LLM_ENABLE_TRACING", True), lambda s: s.llm_enable_tracing),
    ("llm_enable_checkpointing", "BESSER_LLM_ENABLE_CHECKPOINTING", True,
     lambda: _env_bool("BESSER_LLM_ENABLE_CHECKPOINTING", True), lambda s: s.llm_enable_checkpointing),
    ("llm_per_write_diagnostics", "BESSER_LLM_PER_WRITE_DIAGNOSTICS", True,
     lambda: _env_bool("BESSER_LLM_PER_WRITE_DIAGNOSTICS", True), lambda s: s.llm_per_write_diagnostics),
    ("llm_enable_toolchain_validation", "BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION", False,
     lambda: _env_bool("BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION", False),
     lambda s: s.llm_enable_toolchain_validation),
    ("llm_enable_auto_fix", "BESSER_LLM_ENABLE_AUTO_FIX", True,
     lambda: _env_bool("BESSER_LLM_ENABLE_AUTO_FIX", True), lambda s: s.llm_enable_auto_fix),
    ("llm_enable_shell_tools", "BESSER_LLM_ENABLE_SHELL_TOOLS", False,
     lambda: _env_bool("BESSER_LLM_ENABLE_SHELL_TOOLS", False), lambda s: s.llm_enable_shell_tools),
    ("llm_enable_import_smoke_check", "BESSER_LLM_ENABLE_IMPORT_SMOKE_CHECK", True,
     lambda: _env_bool("BESSER_LLM_ENABLE_IMPORT_SMOKE_CHECK", True), lambda s: s.llm_enable_import_smoke_check),
    ("llm_enable_requirements_ledger", "BESSER_LLM_ENABLE_REQUIREMENTS_LEDGER", True,
     lambda: _env_bool("BESSER_LLM_ENABLE_REQUIREMENTS_LEDGER", True),
     lambda s: s.llm_enable_requirements_ledger),
    ("llm_allow_custom_base_url", "BESSER_LLM_ALLOW_CUSTOM_BASE_URL", False,
     lambda: _env_bool("BESSER_LLM_ALLOW_CUSTOM_BASE_URL", False), lambda s: s.llm_allow_custom_base_url),
    ("demo_token", "BESSER_DEMO_TOKEN", "",
     lambda: _get("BESSER_DEMO_TOKEN", "").strip(), lambda s: _secret(s.demo_token).strip()),
    ("telemetry_enabled", "BESSER_TELEMETRY_ENABLED", False,
     lambda: _get("BESSER_TELEMETRY_ENABLED", "").strip().lower() in ("1", "true"),
     lambda s: telemetry.telemetry_enabled()),
    ("telemetry_dir", "BESSER_TELEMETRY_DIR", None,
     lambda: (_get("BESSER_TELEMETRY_DIR", "/app/telemetry"), _get("BESSER_TELEMETRY_DIR")),
     lambda s: (telemetry._configured_dir(), s.telemetry_dir)),
    ("telemetry_admin_token", "BESSER_TELEMETRY_ADMIN_TOKEN", "",
     lambda: _get("BESSER_TELEMETRY_ADMIN_TOKEN", "").strip(), lambda s: _secret(s.telemetry_admin_token).strip()),
    ("incident_log_dir", "BESSER_INCIDENT_LOG_DIR", None,
     lambda: (_get("BESSER_INCIDENT_LOG_DIR", "/app/incidents"), _get("BESSER_INCIDENT_LOG_DIR")),
     lambda s: ("/app/incidents" if s.incident_log_dir is None else s.incident_log_dir, s.incident_log_dir)),
    ("sdmx_timeout_seconds", "BESSER_SDMX_TIMEOUT", 60.0,
     lambda: _env_float("BESSER_SDMX_TIMEOUT", 60.0), lambda s: s.sdmx_timeout_seconds),
    ("sdmx_max_mb", "BESSER_SDMX_MAX_MB", 32.0,
     lambda: _env_float("BESSER_SDMX_MAX_MB", 32.0), lambda s: s.sdmx_max_mb),
    ("sdmx_max_total_mb", "BESSER_SDMX_MAX_TOTAL_MB", 64.0,
     lambda: _env_float("BESSER_SDMX_MAX_TOTAL_MB", 64.0), lambda s: s.sdmx_max_total_mb),
    ("sdmx_max_concurrent_imports", "BESSER_SDMX_MAX_CONCURRENT_IMPORTS", 2,
     lambda: _env_int("BESSER_SDMX_MAX_CONCURRENT_IMPORTS", 2), lambda s: s.sdmx_max_concurrent_imports),
    ("sdmx_allow_custom_url", "BESSER_SDMX_ALLOW_CUSTOM_URL", False,
     lambda: _env_bool("BESSER_SDMX_ALLOW_CUSTOM_URL", False), lambda s: s.sdmx_allow_custom_url),
    ("agent_simulator_url", "AGENT_SIMULATOR_URL", "http://besser-wme-agent-simulator:8001",
     lambda: (_get("AGENT_SIMULATOR_URL") or "http://besser-wme-agent-simulator:8001").rstrip("/"),
     lambda s: s.agent_simulator_url),
    ("agent_simulator_api_token", "AGENT_SIMULATOR_API_TOKEN", "",
     lambda: _get("AGENT_SIMULATOR_API_TOKEN", "").strip(), lambda s: _secret(s.agent_simulator_api_token).strip()),
    ("agent_simulator_require_auth", "AGENT_SIMULATOR_REQUIRE_AUTH", True,
     lambda: _sim_env_bool("AGENT_SIMULATOR_REQUIRE_AUTH", True), lambda s: s.agent_simulator_require_auth),
    ("agent_simulator_restrict_custom_code", "AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", True,
     lambda: _sim_env_bool("AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE", True),
     lambda s: s.agent_simulator_restrict_custom_code),
    ("agent_simulator_rate_limit_window_seconds", "AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS", 60,
     lambda: _sim_env_int("AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS", 60),
     lambda s: s.agent_simulator_rate_limit_window_seconds),
    ("agent_simulator_rate_limit_max_requests", "AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS", 12,
     lambda: _sim_env_int("AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS", 12),
     lambda s: s.agent_simulator_rate_limit_max_requests),
    ("agent_simulator_rate_limit_max_keys", "AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS", 10000,
     lambda: _sim_env_int("AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS", 10000),
     lambda s: s.agent_simulator_rate_limit_max_keys),
    ("agent_simulator_max_sessions_per_actor", "AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR", 1,
     lambda: _sim_env_int("AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR", 1),
     lambda s: s.agent_simulator_max_sessions_per_actor),
    ("agent_simulator_memory_mb", "AGENT_SIMULATOR_MEMORY_MB", None,
     lambda: _sim_env_int("AGENT_SIMULATOR_MEMORY_MB", None), lambda s: s.agent_simulator_memory_mb),
    ("agent_simulator_cpu_cores", "AGENT_SIMULATOR_CPU_CORES", None,
     lambda: _sim_env_float("AGENT_SIMULATOR_CPU_CORES", None), lambda s: s.agent_simulator_cpu_cores),
    ("agent_simulator_disk_mb", "AGENT_SIMULATOR_DISK_MB", None,
     lambda: _sim_env_int("AGENT_SIMULATOR_DISK_MB", None), lambda s: s.agent_simulator_disk_mb),
    ("agent_simulator_session_lifetime_seconds", "AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS", None,
     lambda: _sim_env_int("AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS", None),
     lambda s: s.agent_simulator_session_lifetime_seconds),
    ("agent_simulator_quota_enabled", "AGENT_SIMULATOR_QUOTA_ENABLED", False,
     lambda: _sim_env_bool("AGENT_SIMULATOR_QUOTA_ENABLED", False), lambda s: s.agent_simulator_quota_enabled),
    ("github_client_id", "GITHUB_CLIENT_ID", "",
     lambda: _get("GITHUB_CLIENT_ID", ""), lambda s: s.github_client_id),
    ("github_client_secret", "GITHUB_CLIENT_SECRET", "",
     lambda: _get("GITHUB_CLIENT_SECRET", ""), lambda s: _secret(s.github_client_secret)),
    ("github_redirect_uri", "GITHUB_REDIRECT_URI", "http://localhost:9000/besser_api/github/auth/callback",
     lambda: _get("GITHUB_REDIRECT_URI", "http://localhost:9000/besser_api/github/auth/callback"),
     lambda s: s.github_redirect_uri),
    ("deployment_url", "DEPLOYMENT_URL", "http://localhost:8080",
     lambda: _get("DEPLOYMENT_URL", "http://localhost:8080"), lambda s: s.deployment_url),
    ("oauth_sessions_path", "OAUTH_SESSIONS_PATH", None,
     lambda: _get("OAUTH_SESSIONS_PATH"), lambda s: s.oauth_sessions_path),
    ("user_tokens_path", "USER_TOKENS_PATH", None,
     lambda: _get("USER_TOKENS_PATH"), lambda s: s.user_tokens_path),
    ("session_store_path", "SESSION_STORE_PATH", None,
     lambda: _get("SESSION_STORE_PATH", "<default>"), _new_session_store_path),
    ("cors_origins", "CORS_ORIGINS", "",
     lambda: _get("CORS_ORIGINS", ""), lambda s: s.cors_origins),
    ("wme_node_server_url", "WME_NODE_SERVER_URL", "http://localhost:8080",
     lambda: _get("WME_NODE_SERVER_URL", "http://localhost:8080").rstrip("/"),
     lambda s: s.wme_node_server_url.rstrip("/")),
    ("feedback_email", "FEEDBACK_EMAIL", "",
     lambda: _get("FEEDBACK_EMAIL", "").strip(), lambda s: s.feedback_email.strip()),
    ("smtp_host", "SMTP_HOST", "smtp.gmail.com",
     lambda: _get("SMTP_HOST", "smtp.gmail.com"), lambda s: s.smtp_host),
    ("smtp_port", "SMTP_PORT", "587",
     lambda: int(_get("SMTP_PORT", "587")), lambda s: int(s.smtp_port)),
    ("smtp_username", "SMTP_USERNAME", None,
     lambda: _get("SMTP_USERNAME", "first@example.org"), _new_smtp_username),
    # Unset was None and is now "": the mail is sent only when it is truthy.
    ("smtp_password", "SMTP_PASSWORD", "",
     lambda: bool(_get("SMTP_PASSWORD")), lambda s: bool(_secret(s.smtp_password))),
]

EDGE_VALUES = [
    "", " ", "0", "-1", "1", "5", " 7 ", "2.5", "1e3", "1_000", "nan", "inf", "abc",
    "true", "TRUE", " yes ", "on", "no", "false", "off", "~/somewhere", "/srv/data/", "http://host:1/",
]

ALL_VARIABLES = [row[1] for row in ROWS]


@pytest.fixture
def clean_env(monkeypatch):
    """No backend variable set."""
    for name in ALL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _outcome(read):
    """The value read, typed (True is not 1, NaN equals NaN), or the error raised."""
    try:
        value = read()
    except ValueError as exc:
        return type(exc)
    return type(value), repr(value)


def test_every_field_has_a_row():
    assert sorted(row[0] for row in ROWS) == sorted(BackendSettings.model_fields)


@pytest.mark.parametrize("field, variable, default", [row[:3] for row in ROWS], ids=[row[0] for row in ROWS])
def test_field_names_its_variable_and_default(field, variable, default):
    info = BackendSettings.model_fields[field]
    assert info.validation_alias == variable
    value = info.default.get_secret_value() if hasattr(info.default, "get_secret_value") else info.default
    assert value == default
    assert info.description


@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
def test_unset_reads_as_before(clean_env, row):
    _, _, _, old, new = row
    assert _outcome(lambda: new(BackendSettings())) == _outcome(old)


@pytest.mark.parametrize("value", EDGE_VALUES, ids=repr)
@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
def test_edge_values_read_as_before(clean_env, row, value):
    _, variable, _, old, new = row
    clean_env.setenv(variable, value)
    assert _outcome(lambda: new(BackendSettings())) == _outcome(old)


@pytest.mark.parametrize("default, cap", [("9", "3"), ("3", "9"), ("0", "3"), ("abc", "-2")])
def test_defaults_are_clamped_to_the_hard_caps_as_before(clean_env, default, cap):
    for kind in ("COST_USD", "RUNTIME_SECONDS", "TURNS"):
        clean_env.setenv(f"BESSER_LLM_DEFAULT_MAX_{kind}", default)
        clean_env.setenv(f"BESSER_LLM_MAX_{kind}_HARD_CAP", cap)
    settings = BackendSettings()
    for field, _, _, old, _ in ROWS:
        if field.startswith("llm_default_max_"):
            assert getattr(settings, field) == old()


def test_a_malformed_value_never_stops_the_backend(clean_env):
    for name in ALL_VARIABLES:
        clean_env.setenv(name, "é not a value ;")
    BackendSettings()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows environment names ignore case")
def test_names_are_case_sensitive(clean_env):
    # Neither the lower-cased variable nor the bare field name is read.
    clean_env.setenv("besser_llm_max_concurrent_runs", "3")
    clean_env.setenv("llm_max_concurrent_runs", "3")
    assert BackendSettings().llm_max_concurrent_runs == 10


def test_get_settings_is_read_once_and_per_call_fields_follow_the_environment(clean_env):
    get_settings.cache_clear()
    try:
        first = get_settings()
        clean_env.setenv("BESSER_LLM_MAX_CONCURRENT_RUNS", "3")
        clean_env.setenv("BESSER_TELEMETRY_ENABLED", "1")
        assert get_settings() is first
        assert get_settings().llm_max_concurrent_runs == 10
        assert telemetry.telemetry_enabled() is True
    finally:
        get_settings.cache_clear()


_STARTUP_OPTIONAL = ["GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "SMTP_PASSWORD", "OPENAI_API_KEY",
                     "AGENT_SIMULATOR_API_TOKEN"]


@pytest.mark.parametrize("value", [None, "", " ", "x"], ids=repr)
def test_startup_warns_for_each_optional_variable_unset_or_empty(monkeypatch, caplog, value):
    """The same rule as the old ``not os.environ.get(name)``: unset or empty warns."""
    import logging

    from besser.utilities.web_modeling_editor.backend.backend import _validate_env_vars

    for name in _STARTUP_OPTIONAL:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING):
        _validate_env_vars()
    warned = [name for name in _STARTUP_OPTIONAL if f"variable {name} is not set" in caplog.text]
    assert warned == (_STARTUP_OPTIONAL if not value else [])
