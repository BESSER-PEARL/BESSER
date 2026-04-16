"""Pydantic request model for the smart-generation SSE endpoint.

The ``api_key`` field is a ``SecretStr`` so Pydantic never renders it in
its default ``repr`` / ``model_dump(mode='python')`` output — it shows
as ``**********``. Even if a future developer adds a debug-level
``logger.info("request: %s", request)`` somewhere in the endpoint, the
key will not leak. The runner calls ``resolved_api_key()`` exactly once,
when constructing the LLM client, and never stores the plaintext.

The ``max_cost_usd`` and ``max_runtime_seconds`` fields are clamped by
field validators to the server-side hard caps so clients cannot exceed
them.
"""

from __future__ import annotations

import math
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator

from besser.utilities.web_modeling_editor.backend.constants.constants import (
    LLM_DEFAULT_MAX_COST_USD,
    LLM_DEFAULT_MAX_RUNTIME_SECONDS,
    LLM_MAX_COST_USD_HARD_CAP,
    LLM_MAX_RUNTIME_SECONDS_HARD_CAP,
)
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput

# Model IDs are provider-specific but all follow a conservative
# identifier grammar: letters, digits, dashes, dots, underscores,
# forward slash (for Claude's vendor/model format). Reject anything
# else to catch typos and prompt-injection attempts embedded in the
# model name.
_LLM_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")


class SmartGenerateRequest(BaseModel):
    """Body of ``POST /besser_api/smart-generate``.

    Attributes
    ----------
    project
        The full ``ProjectInput`` payload (same shape as the existing
        ``/generate-output-from-project`` endpoint).
    instructions
        Natural-language description of what to build. Typically refined
        by the modeling agent before being sent.
    api_key
        The user's Anthropic or OpenAI API key. BYOK — sent only in the
        POST body, never in the URL, never logged, never persisted.
    provider
        Which provider the key is for.
    llm_model
        Optional model override (e.g. ``claude-sonnet-4-5``, ``gpt-4o``).
    max_cost_usd
        Soft cap for LLM spend in USD. Clamped to the server hard cap.
    max_runtime_seconds
        Soft cap for total wall-clock runtime. Clamped to the server hard cap.
    """

    project: ProjectInput
    instructions: str = Field(..., min_length=1, max_length=8000)
    api_key: SecretStr
    provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: Optional[str] = Field(default=None, max_length=120)
    max_cost_usd: float = Field(default=LLM_DEFAULT_MAX_COST_USD, gt=0.0)
    max_runtime_seconds: int = Field(default=LLM_DEFAULT_MAX_RUNTIME_SECONDS, gt=0)

    @field_validator("instructions")
    @classmethod
    def _validate_instructions_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions cannot be empty or whitespace-only")
        return value

    @field_validator("api_key")
    @classmethod
    def _validate_api_key_not_empty(cls, value: SecretStr) -> SecretStr:
        # SecretStr accepts empty strings by default. An empty key would
        # fall through to create_llm_client and produce a generic error;
        # catch it here with a clear message instead.
        if not value.get_secret_value().strip():
            raise ValueError("api_key cannot be empty")
        return value

    @field_validator("llm_model")
    @classmethod
    def _validate_llm_model_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not _LLM_MODEL_NAME_RE.match(value):
            raise ValueError(
                "llm_model must contain only letters, digits, dashes, dots, "
                "underscores, or forward slashes"
            )
        return value

    @field_validator("max_cost_usd")
    @classmethod
    def _validate_and_clamp_cost(cls, value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            raise ValueError("max_cost_usd must be a finite positive number")
        return min(value, LLM_MAX_COST_USD_HARD_CAP)

    @field_validator("max_runtime_seconds")
    @classmethod
    def _validate_and_clamp_runtime(cls, value: int) -> int:
        # Pydantic already rejects non-int values, so `gt=0` is enough.
        return min(value, LLM_MAX_RUNTIME_SECONDS_HARD_CAP)

    def resolved_api_key(self) -> str:
        """Return the plaintext API key.

        Callers must pass the returned string directly to the LLM client
        and never log, store, or echo it elsewhere.
        """
        return self.api_key.get_secret_value()
