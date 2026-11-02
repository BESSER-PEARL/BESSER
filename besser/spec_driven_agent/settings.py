"""Environment configuration of the Spec-Driven Agent.

Every ``BESSER_*`` variable the agent reads is declared once on
:class:`AgentSettings` with its type, default and meaning. ``AgentSettings()``
reads the environment when it is built: module-level values build it at
import, the others at each use, so a change to the environment reaches the
next call (and a reloaded module).

A variable whose old reading raised on a bad value, or logged a warning, is
kept as text and parsed where it is used, so it still fails or warns there and
never stops the other settings from loading.

The editor backend's own variables (including ``BESSER_TELEMETRY_DIR`` and
``BESSER_INCIDENT_LOG_DIR``) are on ``BackendSettings`` in the backend.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _exactly_one(value) -> bool:
    return value == "1"


def _one_or_true(value) -> bool:
    return str(value).lower() in ("1", "true")


def _not_off(value) -> bool:
    return str(value).strip().lower() not in ("0", "false", "no")


_ExactlyOne = Annotated[bool, BeforeValidator(_exactly_one)]


class AgentSettings(BaseSettings):
    """The Spec-Driven Agent's environment variables (all optional)."""

    # Defaults are final values, not raw text, so they are not parsed again.
    # No protected namespace: pydantic before 2.10 warns about ``model_catalog``.
    model_config = SettingsConfigDict(
        case_sensitive=True, extra="ignore", validate_default=False,
        protected_namespaces=())

    # -- Models -----------------------------------------------------------------
    model_catalog: str = Field(
        "", validation_alias="BESSER_MODEL_CATALOG",
        description="JSON file merged over the default model catalog; read once per process.")
    llm_planning_model: Optional[str] = Field(
        None, validation_alias="BESSER_LLM_PLANNING_MODEL",
        description="Model for planning calls; 'primary' uses the run's own model. Read per call.")

    # -- Keyless and sponsored tiers ----------------------------------------------
    free_llm_base_url: str = Field(
        "", validation_alias="BESSER_FREE_LLM_BASE_URL",
        description="Free tier endpoint (OpenAI-compatible); the tier needs it and a model. Read per call.")
    free_llm_model: str = Field(
        "", validation_alias="BESSER_FREE_LLM_MODEL",
        description="Model the free tier serves. Read per call.")
    free_llm_token: SecretStr = Field(
        SecretStr(""), validation_alias="BESSER_FREE_LLM_TOKEN",
        description="Bearer token of the free tier endpoint. Read per call.")
    free_llm_alt_models: str = Field(
        "", validation_alias="BESSER_FREE_LLM_ALT_MODELS",
        description="Comma-separated extra models on the free tier endpoint. Read per call.")
    free_llm_fallback_base_url: str = Field(
        "", validation_alias="BESSER_FREE_LLM_FALLBACK_BASE_URL",
        description="Last-resort free tier endpoint; needs a model too. Read per call.")
    free_llm_fallback_model: str = Field(
        "", validation_alias="BESSER_FREE_LLM_FALLBACK_MODEL",
        description="Model of the last-resort free tier endpoint. Read per call.")
    free_llm_fallback_token: SecretStr = Field(
        SecretStr(""), validation_alias="BESSER_FREE_LLM_FALLBACK_TOKEN",
        description="Bearer token of the last-resort free tier endpoint. Read per call.")
    pilot_llm_model: str = Field(
        "", validation_alias="BESSER_PILOT_LLM_MODEL",
        description="Free tier model a study session starts on, if the server offers it. Read per call.")
    sponsored_llm_base_url: str = Field(
        "", validation_alias="BESSER_SPONSORED_LLM_BASE_URL",
        description="Sponsored (demo) tier endpoint. Read per call.")
    sponsored_llm_model: str = Field(
        "", validation_alias="BESSER_SPONSORED_LLM_MODEL",
        description="Default model of the sponsored tier. Read per call.")
    sponsored_llm_token: SecretStr = Field(
        SecretStr(""), validation_alias="BESSER_SPONSORED_LLM_TOKEN",
        description="Bearer token of the sponsored tier endpoint. Read per call.")

    # -- Provider calls (read at import; a malformed number fails the import) ------
    llm_call_timeout_seconds: str = Field(
        "300", validation_alias="BESSER_LLM_CALL_TIMEOUT_SECONDS",
        description="Per-request timeout of the provider SDKs, in seconds.")
    llm_from_scratch_max_tokens: str = Field(
        "32768", validation_alias="BESSER_LLM_FROM_SCRATCH_MAX_TOKENS",
        description="Output-token ceiling of a from-scratch run.")
    llm_modify_max_tokens: Optional[str] = Field(
        None, validation_alias="BESSER_LLM_MODIFY_MAX_TOKENS",
        description="Output-token ceiling of a modify run (unset: the from-scratch ceiling).")
    llm_rolling_cache: _ExactlyOne = Field(
        True, validation_alias="BESSER_LLM_ROLLING_CACHE",
        description="Rolling prompt-cache breakpoint on Anthropic calls; on only for exactly '1'.")

    # -- Context (read at import; a malformed number fails the import) -------------
    llm_compact_threshold: Optional[str] = Field(
        None, validation_alias="BESSER_LLM_COMPACT_THRESHOLD",
        description="Tokens before compaction (unset: 80000, at least 8000); set, it also caps "
                    "the adaptive threshold.")
    llm_max_compact_threshold: str = Field(
        "200000", validation_alias="BESSER_LLM_MAX_COMPACT_THRESHOLD",
        description="Ceiling of the adaptive compaction threshold; blank is 200000.")
    llm_history_eviction: _ExactlyOne = Field(
        False, validation_alias="BESSER_LLM_HISTORY_EVICTION",
        description="Stub stale file bodies at the compaction checkpoint; on only for exactly '1'.")
    llm_inline_scaffold: Annotated[bool, BeforeValidator(_one_or_true)] = Field(
        False, validation_alias="BESSER_LLM_INLINE_SCAFFOLD",
        description="Inline the Phase 1 scaffold into the Phase 2 prompt (1 or true). Read per run.")

    # -- Shell tools ------------------------------------------------------------------
    llm_shell_sandbox: Optional[str] = Field(
        None, validation_alias="BESSER_LLM_SHELL_SANDBOX",
        description="'auto' (bubblewrap where available) or 'off'; anything else is 'auto'. Read per call.")
    llm_shell_network: Optional[str] = Field(
        None, validation_alias="BESSER_LLM_SHELL_NETWORK",
        description="Shell session network: 'auto', 'private' or 'shared'. Read per call.")
    llm_shell_runbook: Annotated[bool, BeforeValidator(_not_off)] = Field(
        True, validation_alias="BESSER_LLM_SHELL_RUNBOOK",
        description="Add the boot-and-verify procedure to the prompt; off for 0, false or no. Read per call.")
    # Text: a value that is not an integer logs a warning where it is read.
    llm_shell_session_idle_seconds: Optional[str] = Field(
        None, validation_alias="BESSER_LLM_SHELL_SESSION_IDLE_SECONDS",
        description="Idle seconds before a shell session closes (unset: 900; 0 never). Read per call.")
    llm_shell_session_max: Optional[str] = Field(
        None, validation_alias="BESSER_LLM_SHELL_SESSION_MAX",
        description="Live shell sessions per process (unset: 10). Read per call.")
