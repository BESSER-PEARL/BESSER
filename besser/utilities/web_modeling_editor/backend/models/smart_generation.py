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
        The user's Anthropic, OpenAI, or Mistral API key. BYOK — sent only
        in the POST body, never in the URL, never logged, never persisted.
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
    provider: Literal["anthropic", "openai", "mistral"] = "anthropic"
    llm_model: Optional[str] = Field(default=None, max_length=120)
    max_cost_usd: float = Field(default=LLM_DEFAULT_MAX_COST_USD, gt=0.0)
    max_runtime_seconds: int = Field(default=LLM_DEFAULT_MAX_RUNTIME_SECONDS, gt=0)
    # Optional plan override from the preview screen. When the user
    # clicks "Adjust" and picks a different primary model or target
    # generator, those overrides travel here. Unset values fall back to
    # auto-detection — the backend never forces a primary.
    primary_kind_override: Optional[
        Literal["class", "gui", "agent", "state_machine", "object", "quantum"]
    ] = None
    # Optional binding choice of the Phase-1 deterministic generator.
    # When set (e.g. from an approved /smart-preview plan), the
    # orchestrator skips its own LLM/keyword selection entirely — the
    # plan the user approved is the plan that runs, and one paid LLM
    # call is saved. Validated against the registered generator tools.
    target_generator_override: Optional[str] = Field(default=None, max_length=80)
    # Incremental vibe-modify. When ``mode == "modify"`` and ``base_run_id``
    # points at a still-downloadable previous run, the new run is SEEDED
    # from that run's generated files and edits them in place, instead of
    # rebuilding from scratch. ``base_run_id`` is the hex run id returned
    # in the earlier run's ``done`` event; the pattern keeps a malformed
    # id out of the registry lookup. When the base has expired the runner
    # warns and falls back to a normal from-scratch generation, so an
    # invalid pairing degrades gracefully rather than failing the request.
    base_run_id: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    mode: Literal["generate", "modify"] = "generate"

    @field_validator("instructions")
    @classmethod
    def _validate_instructions_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions cannot be empty or whitespace-only")
        return value

    @field_validator("target_generator_override")
    @classmethod
    def _validate_target_generator(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        from besser.generators.llm.tools import GENERATOR_TOOLS

        registered = {tool["name"] for tool in GENERATOR_TOOLS}
        if value not in registered:
            raise ValueError(
                f"target_generator_override must be one of: "
                f"{', '.join(sorted(registered))}"
            )
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


class SmartPushDeployConfig(BaseModel):
    """Deployment target for ``POST /besser_api/push-smart-to-github``.

    Mirrors the ``deploy_config`` block the existing ``/deploy-webapp``
    endpoint reads from its body, but as a typed model. ``is_private``
    defaults to ``True`` — a vibe/smart-generation run is customized,
    unreviewed LLM output and should not be world-readable by accident.
    """

    repo_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=350)
    is_private: bool = True
    use_existing: bool = False
    # Target branch. When omitted the endpoint resolves the repo default
    # rather than blindly taking the first branch.
    branch: Optional[str] = Field(default=None, max_length=250)
    commit_message: Optional[str] = Field(default=None, max_length=500)


class PushSmartToGitHubRequest(BaseModel):
    """Body of ``POST /besser_api/push-smart-to-github``.

    Pushes the *stored* artifact of a finished vibe/smart-generation run
    (identified by ``run_id``) plus the re-importable model source, rather
    than re-generating deterministically (which would discard the LLM
    customizations). The run's code is read from ``SMART_RUN_REGISTRY``;
    only the model source travels in the request.

    Attributes
    ----------
    run_id
        The hex run id returned in the smart-generate ``done`` event.
    projectExport
        The frontend V2 project-export envelope (a re-importable
        ``diagrams.json``). Also used to rebuild the B-UML model files
        written under ``buml/``. Optional — absent just means the pushed
        repo won't carry the model source.
    deploy_config
        Repository/branch/commit target.
    """

    run_id: str = Field(..., min_length=1, max_length=64)
    projectExport: Optional[dict] = None
    deploy_config: SmartPushDeployConfig


class PushSmartToGitHubResponse(BaseModel):
    """Response of ``POST /besser_api/push-smart-to-github``."""

    success: bool
    repo_url: str
    owner: str
    repo_name: str
    # True when this created a fresh repo (``use_existing=False``), False
    # when it appended a commit to an existing one.
    is_first_push: bool
    files_uploaded: int


class ImportGitHubRunRequest(BaseModel):
    """Body of ``POST /besser_api/import-github-run``.

    Points the editor at an existing repo that BESSER created (so it
    carries ``buml/diagrams.json`` + the generated code). The endpoint
    downloads the repo, registers its code tree as a run — so the returned
    ``run_id`` can be used as a modify seed (``base_run_id``) — and returns
    the repo's re-importable model.

    Attributes
    ----------
    owner
        Repository owner (login).
    repo
        Repository name.
    branch
        Optional branch/ref to import. When omitted the repo default is
        resolved.
    """

    owner: str = Field(..., min_length=1, max_length=100)
    repo: str = Field(..., min_length=1, max_length=100)
    branch: Optional[str] = Field(default=None, max_length=250)


class ImportGitHubRunResponse(BaseModel):
    """Response of ``POST /besser_api/import-github-run``."""

    # A fresh run id whose stored files are the repo's code tree; usable as
    # the ``base_run_id`` of a subsequent smart-generate modify run.
    run_id: str
    # The repo's re-importable V2 project export (``buml/diagrams.json``),
    # or ``None`` when the repo carries no BESSER model.
    project: Optional[dict] = None
    # True when the repo carried a re-importable model.
    has_model: bool
    owner: str
    repo: str
    branch: str
    # Human-readable hint when there is no model (or it was unreadable).
    message: Optional[str] = None


class SmartPreviewRequest(BaseModel):
    """Body of ``POST /besser_api/smart-preview``.

    Same shape as ``SmartGenerateRequest`` minus the ``api_key`` —
    preview is a pure-local computation (no LLM call) so we don't want
    users leaking their API key just to see the pre-flight plan.
    """

    project: ProjectInput
    instructions: str = Field(..., min_length=1, max_length=8000)
    max_cost_usd: float = Field(default=LLM_DEFAULT_MAX_COST_USD, gt=0.0)
    max_runtime_seconds: int = Field(default=LLM_DEFAULT_MAX_RUNTIME_SECONDS, gt=0)
    primary_kind_override: Optional[
        Literal["class", "gui", "agent", "state_machine", "object", "quantum"]
    ] = None

    @field_validator("instructions")
    @classmethod
    def _validate_instructions_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions cannot be empty or whitespace-only")
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
        return min(value, LLM_MAX_RUNTIME_SECONDS_HARD_CAP)
