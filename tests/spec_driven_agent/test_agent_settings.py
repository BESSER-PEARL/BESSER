"""AgentSettings reads every variable exactly as the code it replaced did.

Each row pairs the expression the agent used before the settings class
(``old``, kept verbatim, reading ``os.environ``) with how the agent reads the
same value now (``new``). Every row is checked unset and with a list of edge
values, so a change to a variable's name, default or parsing fails here.
"""

import importlib
import logging
import os
import subprocess
import sys

import pytest

import besser
from besser.spec_driven_agent.settings import AgentSettings


def _get(name, *default):
    return os.environ.get(name, *default)


def _secret(value):
    return value.get_secret_value()


def _old_env_int(name, default):
    """shell_session._env_int before the settings class, without its log line."""
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _old_compaction():
    threshold = max(8_000, int(os.environ.get("BESSER_LLM_COMPACT_THRESHOLD", "80000") or 80_000))
    return threshold, threshold if os.environ.get("BESSER_LLM_COMPACT_THRESHOLD") else None


def _new_compaction(s):
    return s.llm_compact_threshold or 80_000, s.llm_compact_threshold


def _strip(field):
    return lambda s: getattr(s, field).strip()


# (field, variable, default, old read, new read)
ROWS = [
    ("model_catalog", "BESSER_MODEL_CATALOG", "",
     lambda: _get("BESSER_MODEL_CATALOG", "").strip(), _strip("model_catalog")),
    ("llm_planning_model", "BESSER_LLM_PLANNING_MODEL", None,
     lambda: _get("BESSER_LLM_PLANNING_MODEL"), lambda s: s.llm_planning_model),
    ("free_llm_base_url", "BESSER_FREE_LLM_BASE_URL", "",
     lambda: _get("BESSER_FREE_LLM_BASE_URL", "").strip(), _strip("free_llm_base_url")),
    ("free_llm_model", "BESSER_FREE_LLM_MODEL", "",
     lambda: _get("BESSER_FREE_LLM_MODEL", "").strip(), _strip("free_llm_model")),
    ("free_llm_token", "BESSER_FREE_LLM_TOKEN", "",
     lambda: _get("BESSER_FREE_LLM_TOKEN", "").strip(), lambda s: _secret(s.free_llm_token).strip()),
    ("free_llm_alt_models", "BESSER_FREE_LLM_ALT_MODELS", "",
     lambda: _get("BESSER_FREE_LLM_ALT_MODELS", ""), lambda s: s.free_llm_alt_models),
    ("free_llm_fallback_base_url", "BESSER_FREE_LLM_FALLBACK_BASE_URL", "",
     lambda: _get("BESSER_FREE_LLM_FALLBACK_BASE_URL", "").strip(), _strip("free_llm_fallback_base_url")),
    ("free_llm_fallback_model", "BESSER_FREE_LLM_FALLBACK_MODEL", "",
     lambda: _get("BESSER_FREE_LLM_FALLBACK_MODEL", "").strip(), _strip("free_llm_fallback_model")),
    ("free_llm_fallback_token", "BESSER_FREE_LLM_FALLBACK_TOKEN", "",
     lambda: _get("BESSER_FREE_LLM_FALLBACK_TOKEN", "").strip(),
     lambda s: _secret(s.free_llm_fallback_token).strip()),
    ("pilot_llm_model", "BESSER_PILOT_LLM_MODEL", "",
     lambda: _get("BESSER_PILOT_LLM_MODEL", "").strip(), _strip("pilot_llm_model")),
    ("sponsored_llm_base_url", "BESSER_SPONSORED_LLM_BASE_URL", "",
     lambda: _get("BESSER_SPONSORED_LLM_BASE_URL", "").strip(), _strip("sponsored_llm_base_url")),
    ("sponsored_llm_model", "BESSER_SPONSORED_LLM_MODEL", "",
     lambda: _get("BESSER_SPONSORED_LLM_MODEL", "").strip(), _strip("sponsored_llm_model")),
    ("sponsored_llm_token", "BESSER_SPONSORED_LLM_TOKEN", "",
     lambda: _get("BESSER_SPONSORED_LLM_TOKEN", "").strip(), lambda s: _secret(s.sponsored_llm_token).strip()),
    ("llm_call_timeout_seconds", "BESSER_LLM_CALL_TIMEOUT_SECONDS", 300.0,
     lambda: float(_get("BESSER_LLM_CALL_TIMEOUT_SECONDS", "300")), lambda s: s.llm_call_timeout_seconds),
    ("llm_from_scratch_max_tokens", "BESSER_LLM_FROM_SCRATCH_MAX_TOKENS", 32768,
     lambda: int(_get("BESSER_LLM_FROM_SCRATCH_MAX_TOKENS", "32768")),
     lambda s: s.llm_from_scratch_max_tokens),
    ("llm_modify_max_tokens", "BESSER_LLM_MODIFY_MAX_TOKENS", None,
     lambda: int(_get("BESSER_LLM_MODIFY_MAX_TOKENS", "32768")),
     lambda s: 32768 if s.llm_modify_max_tokens is None else s.llm_modify_max_tokens),
    ("llm_rolling_cache", "BESSER_LLM_ROLLING_CACHE", True,
     lambda: _get("BESSER_LLM_ROLLING_CACHE", "1") == "1", lambda s: s.llm_rolling_cache),
    ("llm_compact_threshold", "BESSER_LLM_COMPACT_THRESHOLD", None,
     _old_compaction, _new_compaction),
    ("llm_max_compact_threshold", "BESSER_LLM_MAX_COMPACT_THRESHOLD", 200_000,
     lambda: int(_get("BESSER_LLM_MAX_COMPACT_THRESHOLD", "200000") or 200_000),
     lambda s: s.llm_max_compact_threshold),
    ("llm_history_eviction", "BESSER_LLM_HISTORY_EVICTION", False,
     lambda: _get("BESSER_LLM_HISTORY_EVICTION", "0") == "1", lambda s: s.llm_history_eviction),
    ("llm_inline_scaffold", "BESSER_LLM_INLINE_SCAFFOLD", False,
     lambda: _get("BESSER_LLM_INLINE_SCAFFOLD", "0").lower() in ("1", "true"), lambda s: s.llm_inline_scaffold),
    ("llm_shell_sandbox", "BESSER_LLM_SHELL_SANDBOX", None,
     lambda: (_get("BESSER_LLM_SHELL_SANDBOX") or "auto").strip().lower(),
     lambda s: (s.llm_shell_sandbox or "auto").strip().lower()),
    ("llm_shell_network", "BESSER_LLM_SHELL_NETWORK", None,
     lambda: (_get("BESSER_LLM_SHELL_NETWORK") or "auto").strip().lower(),
     lambda s: (s.llm_shell_network or "auto").strip().lower()),
    ("llm_shell_runbook", "BESSER_LLM_SHELL_RUNBOOK", True,
     lambda: _get("BESSER_LLM_SHELL_RUNBOOK", "1").strip().lower() not in ("0", "false", "no"),
     lambda s: s.llm_shell_runbook),
    ("llm_shell_session_idle_seconds", "BESSER_LLM_SHELL_SESSION_IDLE_SECONDS", 900,
     lambda: _old_env_int("BESSER_LLM_SHELL_SESSION_IDLE_SECONDS", 900),
     lambda s: s.llm_shell_session_idle_seconds),
    ("llm_shell_session_max", "BESSER_LLM_SHELL_SESSION_MAX", 10,
     lambda: _old_env_int("BESSER_LLM_SHELL_SESSION_MAX", 10),
     lambda s: s.llm_shell_session_max),
]

EDGE_VALUES = [
    "", " ", "0", "-1", "1", " 1", "5", " 7 ", "2.5", "1e3", "1_000", "nan", "abc", "true", "TRUE",
    " yes ", "on", "no", " no ", " 0 ", "False", "false", "off", "primary", "Private", " shared ", "a, b,,c ", "/srv/catalog.json",
]

ALL_VARIABLES = [row[1] for row in ROWS]

# Read into module constants; a malformed value fails the import.
IMPORT_TIME_NUMBERS = {
    "BESSER_LLM_CALL_TIMEOUT_SECONDS": "besser.spec_driven_agent.providers.llm_client",
    "BESSER_LLM_FROM_SCRATCH_MAX_TOKENS": "besser.spec_driven_agent.providers.llm_client",
    "BESSER_LLM_MODIFY_MAX_TOKENS": "besser.spec_driven_agent.providers.llm_client",
    "BESSER_LLM_COMPACT_THRESHOLD": "besser.spec_driven_agent.agent.compaction",
    "BESSER_LLM_MAX_COMPACT_THRESHOLD": "besser.spec_driven_agent.agent.compaction",
}


@pytest.fixture
def clean_env(monkeypatch):
    """No agent variable set."""
    for name in ALL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _outcome(read):
    """The value read, typed (True is not 1, NaN equals NaN), or ValueError.

    The settings raise pydantic's ValidationError, a ValueError that names the
    variable, where the old code raised a bare ValueError.
    """
    try:
        value = read()
    except ValueError:
        return ValueError
    return type(value), repr(value)


def test_every_field_has_a_row():
    assert sorted(row[0] for row in ROWS) == sorted(AgentSettings.model_fields)


@pytest.mark.parametrize("field, variable, default", [row[:3] for row in ROWS], ids=[row[0] for row in ROWS])
def test_field_names_its_variable_and_default(field, variable, default):
    info = AgentSettings.model_fields[field]
    assert info.validation_alias == variable
    value = info.default.get_secret_value() if hasattr(info.default, "get_secret_value") else info.default
    assert value == default
    assert info.description


@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
def test_unset_reads_as_before(clean_env, row):
    _, _, _, old, new = row
    assert _outcome(lambda: new(AgentSettings())) == _outcome(old)


@pytest.mark.parametrize("value", EDGE_VALUES, ids=repr)
@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
def test_edge_values_read_as_before(clean_env, row, value):
    _, variable, _, old, new = row
    clean_env.setenv(variable, value)
    assert _outcome(lambda: new(AgentSettings())) == _outcome(old)


def test_a_malformed_value_never_stops_the_other_settings(clean_env):
    for name in ALL_VARIABLES:
        if name not in IMPORT_TIME_NUMBERS:
            clean_env.setenv(name, "é not a value ;")
    AgentSettings()


@pytest.mark.parametrize("variable", sorted(IMPORT_TIME_NUMBERS))
def test_a_malformed_import_time_number_raises_naming_the_variable(clean_env, variable):
    clean_env.setenv(variable, "abc")
    with pytest.raises(ValueError, match=variable):
        AgentSettings()


@pytest.mark.parametrize("variable", ["BESSER_LLM_CALL_TIMEOUT_SECONDS", "BESSER_LLM_COMPACT_THRESHOLD"])
def test_a_malformed_import_time_number_still_fails_the_import(variable):
    root = os.path.dirname(os.path.dirname(os.path.abspath(besser.__file__)))
    env = {**os.environ, "PYTHONPATH": root, variable: "abc"}
    result = subprocess.run([sys.executable, "-c", f"import {IMPORT_TIME_NUMBERS[variable]}"],
                            env=env, cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "ValidationError" in result.stderr and variable in result.stderr


@pytest.mark.parametrize("field, variable, default", [
    ("llm_shell_session_idle_seconds", "BESSER_LLM_SHELL_SESSION_IDLE_SECONDS", 900),
    ("llm_shell_session_max", "BESSER_LLM_SHELL_SESSION_MAX", 10),
])
def test_a_malformed_shell_session_number_warns_and_gives_the_default(clean_env, caplog, field, variable, default):
    clean_env.setenv(variable, "abc")
    with caplog.at_level(logging.WARNING, logger="besser.spec_driven_agent.settings"):
        assert getattr(AgentSettings(), field) == default
    assert f"{variable}='abc' is not an integer; using {default}" in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="Windows environment names ignore case")
def test_names_are_case_sensitive(clean_env):
    # Neither the lower-cased variable nor the bare field name is read.
    clean_env.setenv("besser_free_llm_model", "x")
    clean_env.setenv("free_llm_model", "x")
    assert AgentSettings().free_llm_model == ""


def test_module_level_values_are_read_at_import(clean_env):
    """A reload sees the new environment, as before; the per-call reads see it at once."""
    from besser.spec_driven_agent.agent import compaction
    from besser.spec_driven_agent.providers import llm_client

    clean_env.setenv("BESSER_LLM_COMPACT_THRESHOLD", "50000")
    clean_env.setenv("BESSER_FREE_LLM_MODEL", "  m1 ")
    try:
        importlib.reload(compaction)
        assert compaction.COMPACT_TOKEN_THRESHOLD == 50_000
        assert compaction._OPERATOR_CAP == 50_000
        assert llm_client.free_tier_model() == "m1"
    finally:
        clean_env.delenv("BESSER_LLM_COMPACT_THRESHOLD")
        importlib.reload(compaction)
