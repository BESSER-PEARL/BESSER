"""Pydantic request model for the spec-driven generation SSE endpoint.

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
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from besser.spec_driven_agent.planning.specification import MAX_SPECIFICATION_CHARS
from besser.utilities.path_utils import is_repo_json_path
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.models.responses import CommitStatus, ResponseModel
from besser.utilities.web_modeling_editor.backend.settings import get_settings

# Model IDs are provider-specific but all follow a conservative
# identifier grammar: letters, digits, dashes, dots, underscores,
# forward slash (for Claude's vendor/model format). Reject anything
# else to catch typos and prompt-injection attempts embedded in the
# model name.
_LLM_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:\-/]+$")

# GitHub login / repository name charset.
_GITHUB_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


class SmartGenerateRequest(BaseModel):
    """Body of ``POST /besser_api/spec-driven/generate``.

    Attributes
    ----------
    project
        The full ``ProjectInput`` payload (same shape as the existing
        ``/generate-output-from-project`` endpoint).
    instructions
        Natural-language description of what to build. Typically refined
        by the modeling agent before being sent.
    api_key
        The user's Anthropic, OpenAI, Mistral, or Nebius Token Factory API
        key. BYOK — sent only
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
    instructions: str = Field(..., min_length=1, max_length=MAX_SPECIFICATION_CHARS)
    # Optional: the ``"free"`` provider needs no key (the server injects a
    # hosted open-weight model). Every other provider requires a non-empty key,
    # enforced provider-aware in ``_validate_key_matches_provider`` below.
    api_key: Optional[SecretStr] = None
    # "pia" and "gateway" are gateways whose endpoints come from the server's
    # model catalog (BESSER_MODEL_CATALOG); a server without them rejects the run.
    provider: Literal[
        "anthropic", "openai", "mistral", "nebius", "free", "sponsored", "pia", "gateway"
    ] = "anthropic"
    llm_model: Optional[str] = Field(default=None, max_length=120)
    # OpenAI-compatible base URL for the frontend's 'Local (self-hosted)'
    # provider (arrives as provider="openai" + this URL). Passed to the OpenAI
    # SDK. SECURITY: having the server open a user-supplied
    # URL is an SSRF surface, so a request carrying base_url is REJECTED unless
    # the deploy opts in via BESSER_LLM_ALLOW_CUSTOM_BASE_URL (see the runner) —
    # these providers are meant for local / single-tenant use.
    base_url: Optional[str] = Field(default=None, max_length=500)
    max_cost_usd: float = Field(default=get_settings().llm_default_max_cost_usd, gt=0.0)
    max_runtime_seconds: int = Field(default=get_settings().llm_default_max_runtime_seconds, gt=0)
    max_turns: int = Field(default=get_settings().llm_default_max_turns, gt=0)
    # Optional plan override from the preview screen. When the user
    # clicks "Adjust" and picks a different primary model or target
    # generator, those overrides travel here. Unset values fall back to
    # auto-detection — the backend never forces a primary.
    primary_kind_override: Optional[
        Literal[
            "class", "gui", "agent", "state_machine", "object",
            "bpmn", "nn", "quantum",
        ]
    ] = None
    # Optional binding choice of the Phase-1 deterministic generator.
    # When set (e.g. from an approved /spec-driven/preview plan), the
    # orchestrator skips its own LLM/keyword selection entirely — the
    # plan the user approved is the plan that runs, and one paid LLM
    # call is saved. Validated against the registered generator tools.
    target_generator_override: Optional[str] = Field(default=None, max_length=80)
    # ``target_generator_override=None`` normally means auto-select. Preview
    # also needs to bind an explicit "LLM from scratch" decision, so that
    # distinct state travels as a boolean instead of being collapsed to null.
    skip_deterministic_generator: bool = False
    # Incremental modify. When ``mode == "modify"`` and ``base_run_id``
    # points at a still-downloadable previous run, the new run is SEEDED
    # from that run's generated files and edits them in place, instead of
    # rebuilding from scratch. ``base_run_id`` is the hex run id returned
    # in the earlier run's ``done`` event; the pattern keeps a malformed
    # id out of the registry lookup. When the base has expired the runner
    # warns and falls back to a normal from-scratch generation, so an
    # invalid pairing degrades gracefully rather than failing the request.
    base_run_id: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    mode: Literal["generate", "modify"] = "generate"
    # Opt-in study telemetry labels (services/spec_driven/telemetry.py).
    # Optional and deliberately fail-open: a generation must NEVER fail
    # because of telemetry, so values that don't match the collection
    # patterns are silently nulled instead of rejected. Validated
    # independently — a valid participant with an invalid session (or
    # vice versa) keeps the valid half; the runner records a run summary
    # only when BOTH survive sanitization.
    telemetry_session: Optional[str] = None
    telemetry_participant: Optional[str] = None
    # Shared secret authorising the server-paid ``sponsored`` tier. Demo links
    # carry it as ``?demo=<token>``; the router compares it against
    # BESSER_DEMO_TOKEN. SecretStr for the same reason as api_key — it must
    # never reach a log or an SSE frame.
    demo_token: Optional[SecretStr] = Field(default=None, max_length=200)

    @field_validator("instructions")
    @classmethod
    def _validate_instructions_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions cannot be empty or whitespace-only")
        return value

    @field_validator("telemetry_session", mode="before")
    @classmethod
    def _sanitize_telemetry_session(cls, value: object) -> Optional[str]:
        # Lazy import to avoid a circular import at module load (the
        # services.spec_driven package imports this module via its runner).
        from besser.utilities.web_modeling_editor.backend.services.spec_driven.telemetry import (
            sanitize_session,
        )
        return sanitize_session(value)

    @field_validator("telemetry_participant", mode="before")
    @classmethod
    def _sanitize_telemetry_participant(cls, value: object) -> Optional[str]:
        from besser.utilities.web_modeling_editor.backend.services.spec_driven.telemetry import (
            sanitize_participant,
        )
        return sanitize_participant(value)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not re.match(r"^https?://", value, re.IGNORECASE):
            raise ValueError("base_url must be an http:// or https:// URL")
        return value

    @field_validator("target_generator_override")
    @classmethod
    def _validate_target_generator(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        from besser.spec_driven_agent.agent.tools import GENERATOR_TOOLS

        registered = {tool["name"] for tool in GENERATOR_TOOLS}
        if value not in registered:
            raise ValueError(
                f"target_generator_override must be one of: "
                f"{', '.join(sorted(registered))}"
            )
        return value

    @model_validator(mode="after")
    def _validate_generator_plan(self) -> "SmartGenerateRequest":
        if self.skip_deterministic_generator and self.target_generator_override:
            raise ValueError(
                "skip_deterministic_generator cannot be combined with "
                "target_generator_override"
            )
        return self

    @model_validator(mode="after")
    def _validate_key_matches_provider(self) -> "SmartGenerateRequest":
        # The keyless "free" tier must NOT carry a key (the server injects the
        # hosted endpoint + token); every other provider requires a non-empty
        # one. SecretStr accepts empty strings, so an empty/whitespace key would
        # otherwise fall through to create_llm_client and produce a generic
        # error — catch it here with a clear, provider-aware message.
        has_key = bool(self.api_key and self.api_key.get_secret_value().strip())
        if self.provider in ("free", "sponsored"):
            # Ignore any accidentally-sent key rather than erroring — these
            # server-configured tiers never use a client key. ``llm_model`` is
            # deliberately KEPT: for "free" it may name the server's fallback
            # model (the only id besides the primary the factory honors —
            # anything else pins to the primary), and for "sponsored" it picks
            # from the aggregator's catalog.
            self.api_key = None
        elif not has_key:
            raise ValueError(
                f"api_key is required for provider '{self.provider}'"
            )
        if self.provider in ("pia", "gateway"):
            # The gateway's endpoint is the server's model catalog, never the
            # request's, so a stray base_url must not trip the SSRF gate.
            self.base_url = None
        return self

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
        return min(value, get_settings().llm_max_cost_usd_hard_cap)

    @field_validator("max_runtime_seconds")
    @classmethod
    def _validate_and_clamp_runtime(cls, value: int) -> int:
        # Pydantic already rejects non-int values, so `gt=0` is enough.
        return min(value, get_settings().llm_max_runtime_seconds_hard_cap)

    @field_validator("max_turns")
    @classmethod
    def _validate_and_clamp_turns(cls, value: int) -> int:
        return min(value, get_settings().llm_max_turns_hard_cap)

    def resolved_api_key(self) -> str:
        """Return the plaintext API key, or ``""`` for the keyless free tier.

        Callers must pass the returned string directly to the LLM client
        and never log, store, or echo it elsewhere. For ``provider == "free"``
        there is no user key — the factory ignores this value and injects the
        server-side hosted endpoint instead.
        """
        if self.api_key is None:
            return ""
        return self.api_key.get_secret_value()

    def resolved_demo_token(self) -> str:
        """The plaintext demo token, or ``""`` when the request carries none."""
        if self.demo_token is None:
            return ""
        return self.demo_token.get_secret_value()


class SmartPushDeployConfig(BaseModel):
    """Deployment target for ``POST /besser_api/spec-driven/push-to-github``.

    Mirrors the ``deploy_config`` block the existing ``/deploy-webapp``
    endpoint reads from its body, but as a typed model. ``is_private``
    defaults to ``True`` — a spec-driven generation run is customized,
    unreviewed LLM output and should not be world-readable by accident.
    """

    repo_name: str = Field(..., min_length=1, max_length=100)
    # Owner of an existing repo (an organization, or a user who made the
    # signed-in user a collaborator). Omitted: the signed-in user. A new repo
    # is always created under the signed-in user.
    owner: Optional[str] = Field(default=None, min_length=1, max_length=100, pattern=r"^[A-Za-z0-9-]+$")
    description: Optional[str] = Field(default=None, max_length=350)
    is_private: bool = True
    use_existing: bool = False
    # Target branch. When omitted the endpoint resolves the repo default
    # rather than blindly taking the first branch.
    branch: Optional[str] = Field(default=None, max_length=250)
    commit_message: Optional[str] = Field(default=None, max_length=500)
    # The commit the run's code was derived from (the import, or the last
    # push). When the branch has moved past it, the push is merged with the
    # newer commits instead of replacing them.
    base_commit_sha: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    # "commit" lands on ``branch``; "pull_request" lands on a new branch and
    # opens a pull request into ``branch``.
    push_mode: Literal["commit", "pull_request"] = "commit"


class PushSmartToGitHubRequest(BaseModel):
    """Body of ``POST /besser_api/spec-driven/push-to-github``.

    Pushes the *stored* artifact of a finished spec-driven generation run
    (identified by ``run_id``) plus the re-importable model source, rather
    than re-generating deterministically (which would discard the LLM
    customizations). The run's code is read from the run registry;
    only the model source travels in the request.

    Attributes
    ----------
    run_id
        The hex run id returned in the spec-driven generate ``done`` event.
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


class RecapRun(ResponseModel):
    """One earlier run on the app, as ``run_recap.build_recap`` reads it."""

    # The user's own words; empty when the recipe no longer holds them.
    request: str
    date: str
    mode: Literal["create", "modify"]
    file_count: int


class RecapOpenItem(ResponseModel):
    """Something the last run left open."""

    kind: Literal["rolled_back", "missing", "not_checked", "left_out"]
    text: str


class RecapCommit(ResponseModel):
    """A commit the repo gained after the last run."""

    sha: str
    message: str
    author: str


class RepoRecap(ResponseModel):
    """What was done on a repo's app and where it stands (``run_recap.build_recap``).

    The editor's chat shows it; every value is text from the repository.
    """

    runs: list[RecapRun]
    open_items: list[RecapOpenItem]
    # The true total; ``open_items`` is capped.
    open_item_count: int
    commits_since_run: list[RecapCommit]
    # The true total; ``commits_since_run`` is capped.
    commit_count: int


class PushSmartToGitHubResponse(ResponseModel):
    """Response of ``POST /besser_api/spec-driven/push-to-github``."""

    success: bool
    repo_url: str
    owner: str
    repo_name: str
    # True when this created a fresh repo (``use_existing=False``), False
    # when it appended a commit to an existing one.
    is_first_push: bool
    files_uploaded: int
    branch: str = ""
    # Head of ``branch`` after the push: the next push's ``base_commit_sha``.
    commit_sha: str = ""
    # "pushed" (fast-forward), "merged" (merged with commits made since
    # ``base_commit_sha``), "conflict" (left on ``conflict_branch``) or
    # "pull_request" (opened ``pull_request_url``).
    status: Literal[CommitStatus, "pull_request"] = "pushed"
    conflict_branch: Optional[str] = None
    pull_request_url: Optional[str] = None
    # The branch the pull request merges (``status`` "pull_request").
    pull_request_branch: Optional[str] = None
    # Set when the push merged in other commits: a run holding the merged
    # repo, to use as the next modify run's ``base_run_id`` so the agent
    # edits the merged code rather than its own pre-merge copy.
    base_run_id: Optional[str] = None
    # Problems the user should know about, e.g. a layout change that keeps
    # the app from deploying as before.
    warnings: list[str] = []
    # The model written to ``buml/`` when the run changed it (a modify run
    # that added classes); None when it is the request's own export. The
    # editor loads it, so its model matches the repo.
    project_export: Optional[dict] = None
    # What the pushed recipe records, and ``AGENTS.md`` too unless a warning says
    # otherwise; None for a run that left no recipe.
    recap: Optional[RepoRecap] = None


class ImportGitHubRunRequest(BaseModel):
    """Body of ``POST /besser_api/spec-driven/import-github-run``.

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
    demo_token
        The demo-link secret (``?demo=<token>``). When valid, a repo without
        readable data classes is read by the sponsored (demo) model instead
        of the keyless one.
    allow_ai_reading
        The user allowed BESSER to send the repo's code to an AI model when it
        cannot read the data classes directly. Without it no model is called;
        the response offers the reading instead (``ai_reading_offered``).
    """

    owner: str = Field(..., min_length=1, max_length=100)
    repo: str = Field(..., min_length=1, max_length=100)
    branch: Optional[str] = Field(default=None, max_length=250)
    demo_token: Optional[SecretStr] = Field(default=None, max_length=200)
    allow_ai_reading: bool = False
    # The model file the user picked when the repo holds several; the first
    # found (``buml/diagrams.json``, else an older editor's save) when unset.
    model_path: Optional[str] = Field(default=None, max_length=500)

    @field_validator("owner", "repo")
    @classmethod
    def _github_name(cls, value: str) -> str:
        """Stripped; both are interpolated into GitHub API URL paths."""
        value = value.strip()
        if not _GITHUB_NAME_RE.fullmatch(value) or not value.strip("."):
            raise ValueError("Invalid GitHub owner or repository name.")
        return value

    @field_validator("model_path")
    @classmethod
    def _model_path_in_repo(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not is_repo_json_path(value):
            raise ValueError("model_path must be a relative repo path to a .json file")
        return value

    def resolved_demo_token(self) -> str:
        """The plaintext demo token, or ``""`` when the request carries none."""
        return "" if self.demo_token is None else self.demo_token.get_secret_value()


class CodeAnalysisResponse(ResponseModel):
    """What a repo's code holds: ``CodeAnalysis.to_dict()`` (reverse_engineering)."""

    stack: list[str]
    # "SQLAlchemy models", "Django models", ...; None when no model code was read.
    model_source: Optional[str]
    source_files: list[str]
    entities: list[str]
    # True when an LLM proposed the model from the code, not the exact reader.
    ai: bool
    # Relationships the model could not take, as "Source -> Target".
    skipped_relationships: list[str]


class ProposedDiagramResponse(ResponseModel):
    """A class diagram read from a repo's code, in the shape the editor imports."""

    title: str
    model: dict[str, Any]
    diagramType: str


class ImportGitHubRunResponse(ResponseModel):
    """Response of ``POST /besser_api/spec-driven/import-github-run``."""

    # A fresh run id whose stored files are the repo's code tree; usable as
    # the ``base_run_id`` of a subsequent spec-driven generate modify run.
    run_id: str
    # The repo's re-importable V2 project export (``buml/diagrams.json``), the
    # bare project an older editor saved, or ``None`` when there is no model.
    project: Optional[dict] = None
    # True when the repo carried a re-importable model.
    has_model: bool
    # Where ``project`` was read; a file other than ``buml/diagrams.json`` is
    # an older editor's save, which the next model save replaces.
    model_path: Optional[str] = None
    owner: str
    repo: str
    branch: str
    # The commit that was imported: the first push's ``base_commit_sha``.
    commit_sha: str = ""
    # Repos without a BESSER model: what their code holds ({stack,
    # model_source, source_files, entities, ai}) and a class diagram read from
    # their data classes, in the shape the editor imports, for the user to
    # accept. ``ai`` is true when no data classes were readable and an LLM
    # proposed the diagram from the code instead. proposed_diagram is None
    # when the repo has a model, or neither reading produced one.
    analysis: Optional[CodeAnalysisResponse] = None
    proposed_diagram: Optional[ProposedDiagramResponse] = None
    # Human-readable hint when there is no model (or it was unreadable).
    message: Optional[str] = None
    # True when an AI model could read the code for a better diagram and the
    # request did not allow it: import again with ``allow_ai_reading`` to read it.
    ai_reading_offered: bool = False
    # False for a repo holding only a saved model: a run has no app to continue from.
    has_code: bool = False
    # Where the repo's app stands; None when no Spec-Driven run was pushed to it.
    recap: Optional[RepoRecap] = None


class SmartPreviewRequest(BaseModel):
    """Body of ``POST /besser_api/spec-driven/preview``.

    Same shape as ``SmartGenerateRequest`` minus the ``api_key`` —
    preview is a pure-local computation (no LLM call) so we don't want
    users leaking their API key just to see the pre-flight plan.
    """

    project: ProjectInput
    instructions: str = Field(..., min_length=1, max_length=MAX_SPECIFICATION_CHARS)
    max_cost_usd: float = Field(default=get_settings().llm_default_max_cost_usd, gt=0.0)
    max_runtime_seconds: int = Field(default=get_settings().llm_default_max_runtime_seconds, gt=0)
    mode: Literal["generate", "modify"] = "generate"
    base_run_id: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    primary_kind_override: Optional[
        Literal[
            "class", "gui", "agent", "state_machine", "object",
            "bpmn", "nn", "quantum",
        ]
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
        return min(value, get_settings().llm_max_cost_usd_hard_cap)

    @field_validator("max_runtime_seconds")
    @classmethod
    def _validate_and_clamp_runtime(cls, value: int) -> int:
        return min(value, get_settings().llm_max_runtime_seconds_hard_cap)

    @model_validator(mode="after")
    def _validate_preview_mode(self) -> "SmartPreviewRequest":
        if self.mode == "modify" and self.base_run_id is None:
            raise ValueError("base_run_id is required when preview mode is modify")
        if self.mode == "generate" and self.base_run_id is not None:
            raise ValueError("base_run_id is only valid when preview mode is modify")
        return self


class SmartPreviewModelSummaryResponse(ResponseModel):
    """``AssembledModels.summary()``: the diagrams present, counts only."""

    primary: str
    # One entry per model present: ``kind`` plus that kind's counts.
    present: list[dict[str, Any]]
    assembly_issues: list[dict[str, str]]


class RenamedElementResponse(ResponseModel):
    """A class ("Room") or attribute ("Room.floor") kept under a new name."""

    from_: str = Field(alias="from")
    to: str


class ModelChangeResponse(ResponseModel):
    """``ModelChange.to_dict()``: how a modify's model relates to its base app's."""

    verdict: Literal["small", "large", "unrelated"]
    # "attributes": full models compared; "classes": an older recipe's class names only.
    basis: Literal["attributes", "classes"]
    gone: float
    removed: list[str]
    added: list[str]
    renamed: list[RenamedElementResponse]
    changes: list[str]


class SmartPreviewResponse(ResponseModel):
    """Response of ``POST /besser_api/spec-driven/preview``: ``PreviewPlan.to_dict()`` and the model."""

    execution_mode: Literal["generate", "modify"]
    primary_kind: str
    auxiliary_kinds: list[str]
    # None when the run writes the code from scratch.
    target_generator: Optional[str]
    target_generator_confidence: float
    summary: str
    estimated_turns: int
    estimated_cost_usd: float
    estimated_duration_seconds: int
    notes: list[str]
    model_summary: SmartPreviewModelSummaryResponse
    # Modify with a known base run only; None = unknown, keep modifying.
    model_change: Optional[ModelChangeResponse]


class SpecDrivenCapsResponse(ResponseModel):
    max_cost_usd_hard_cap: float
    max_runtime_seconds_hard_cap: int
    max_turns_hard_cap: int
    default_max_cost_usd: float
    default_max_runtime_seconds: int
    default_max_turns: int


class SpecDrivenConcurrencyResponse(ResponseModel):
    max_concurrent_runs: int


class SpecDrivenFeaturesResponse(ResponseModel):
    tracing_enabled: bool
    checkpointing_enabled: bool
    resume_enabled: bool
    toolchain_validation_enabled: bool
    shell_tools_enabled: bool
    per_write_diagnostics_enabled: bool
    durable_runs_enabled: bool
    event_replay_enabled: bool
    cancel_abandoned_runs: bool
    disconnected_grace_seconds: int


class CatalogModelResponse(ResponseModel):
    id: str
    label: str
    default: bool


class CatalogProviderResponse(ResponseModel):
    """A model-catalog provider as the key dialog shows it (``public_providers``)."""

    id: str
    label: str
    hint: Optional[str]
    key_prefix: Optional[str]
    models: list[CatalogModelResponse]


class FreeTierModelResponse(ResponseModel):
    id: str
    default: bool


class FreeTierResponse(ResponseModel):
    available: bool
    model: Optional[str]
    models: list[FreeTierModelResponse]
    pilot_model: Optional[str]


class SpecDrivenConfigResponse(ResponseModel):
    """Response of ``GET /besser_api/spec-driven/config``: public caps and feature flags."""

    caps: SpecDrivenCapsResponse
    download_ttl_seconds: int
    cost_emitter_interval_seconds: float
    concurrency: SpecDrivenConcurrencyResponse
    features: SpecDrivenFeaturesResponse
    supported_providers: list[str]
    providers: list[CatalogProviderResponse]
    default_models: dict[str, str]
    free_tier: FreeTierResponse
