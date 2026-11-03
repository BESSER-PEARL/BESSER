"""Environment configuration of the LLM layer.

The Spec-Driven Agent's :class:`~besser.spec_driven_agent.settings.AgentSettings`
extends :class:`LLMSettings`, so each variable is still declared once.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """The LLM layer's environment variables (all optional)."""

    # Defaults are final values, not raw text, so they are not parsed again.
    # No protected namespace: pydantic before 2.10 warns about ``model_catalog``.
    model_config = SettingsConfigDict(
        case_sensitive=True, extra="ignore", validate_default=False,
        protected_namespaces=())

    model_catalog: str = Field(
        "", validation_alias="BESSER_MODEL_CATALOG",
        description="JSON file merged over the default model catalog; read once per process.")
