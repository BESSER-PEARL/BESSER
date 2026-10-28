"""Request models of the Knowledge Graph endpoints (``/besser_api/kg/...``).

The KG-only fields used to sit on the shared :class:`DiagramInput`, which every
other diagram endpoint also parses. They live here instead, on a subclass that
only the KG endpoints accept.

Naming rule: every **data** field is camelCase, like the rest of the API
(``kgSignature``, ``llmIssues``, ``nodeIds``, ``issueId``). The LLM
**credentials** keep the snake_case names of
:class:`~besser.utilities.web_modeling_editor.backend.models.spec_driven.SmartGenerateRequest`
(``provider``, ``api_key``, ``llm_model``) so one key form serves every LLM
feature of the editor; they are declared once, on :class:`KGLLMCredentials`.
The key is a ``SecretStr`` so Pydantic never renders it in a ``repr`` or a
validation error, and it is only unwrapped by
:meth:`KGLLMCredentials.resolved_api_key`.

Only the *shape* is validated here (a 422 on failure). Semantic checks (a blank
key, an empty or oversized ``nodeIds``, a decision other than accept/skip) stay
in the endpoints and services as ``ValidationError`` / ``ConversionError``,
because a Pydantic failure is a 422 whose ``detail`` is a list, and the editor's
``apiClient`` shows the ``detail`` of a failed call as the message.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator

from besser.utilities.web_modeling_editor.backend.constants.constants import DEFAULT_KG_LLM_MODEL
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.models.responses import KGIssueResponse
from besser.utilities.web_modeling_editor.backend.models.spec_driven import _LLM_MODEL_NAME_RE

#: Providers a user's own key can be used with. ``free`` / ``sponsored`` are
#: server-paid tiers and the ``base_url`` providers (PIA, local) open a
#: user-chosen URL (an SSRF surface), so none of them are offered here.
KGLLMProvider = Literal["anthropic", "openai", "mistral", "nebius"]


class KGResolutionInput(BaseModel):
    """The user's choice for one preflight or LLM issue.

    ``decision`` is ``"accept"`` (apply the issue's recommended action) or
    ``"skip"`` (apply its skip action); any other value is refused by the
    resolver with HTTP 400.
    """

    issueId: str = Field(min_length=1)
    decision: str


class KGDiagramInput(DiagramInput):
    """A Knowledge Graph diagram plus the refinement fields of the KG endpoints."""

    # The user's choices for the issues of ``/kg/preflight`` (or, with
    # ``source="llm"``, of ``llmIssues``). ``kgSignature`` is the value echoed
    # from that response, so the backend can detect a stale payload (KG mutated
    # between the analysis and the apply).
    resolutions: Optional[List[KGResolutionInput]] = None
    kgSignature: Optional[str] = None
    # LLM-cleanup round-trip: the issue list returned by ``/kg/llm-clean`` or
    # ``/kg/classify-orphans``, sent back unchanged so ``/kg/apply-refinement``
    # can rebuild the issues without calling the LLM again.
    llmIssues: Optional[List[KGIssueResponse]] = None
    # Which issues ``/kg/apply-refinement`` resolves: ``"static"`` re-runs the
    # preflight and applies ``resolutions`` to its issues, ``"llm"`` applies
    # them to ``llmIssues``.
    source: Optional[Literal["static", "llm"]] = None
    # KG RDF export: which constraint vocabularies to emit. ``"owl"`` emits
    # only ``owl:Restriction`` blank nodes; ``"shacl"`` emits only
    # ``sh:NodeShape`` / ``sh:PropertyShape``; ``"both"`` (default) emits
    # both where each spec has the required semantics.
    vocab: Optional[Literal["owl", "shacl", "both"]] = None


class KGLLMCredentials(BaseModel):
    """The user's own LLM key, its provider and optionally a model."""

    provider: KGLLMProvider = "openai"
    api_key: SecretStr
    #: ``None`` means the provider's default model (see :meth:`resolved_model`).
    llm_model: Optional[str] = Field(default=None, max_length=120)

    @field_validator("llm_model")
    @classmethod
    def _validate_llm_model_format(cls, value: Optional[str]) -> Optional[str]:
        """Same model-id grammar as the spec-driven agent; a blank value means the default."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not _LLM_MODEL_NAME_RE.match(value):
            raise ValueError(
                "llm_model must contain only letters, digits, dashes, dots, "
                "underscores, colons, or forward slashes"
            )
        return value

    def resolved_api_key(self) -> str:
        """The key, unwrapped. Pass it straight to the provider client; never log or echo it."""
        return self.api_key.get_secret_value()

    def resolved_model(self) -> str:
        """The requested model, else the provider's default."""
        if self.llm_model:
            return self.llm_model
        if self.provider == "openai":
            return DEFAULT_KG_LLM_MODEL
        # Imported here: the provider SDKs are heavy and only this branch needs them.
        from besser.spec_driven_agent.providers.llm_client import DEFAULT_MODELS

        return DEFAULT_MODELS[self.provider]


class KGLLMRequest(KGLLMCredentials):
    """Body of ``/kg/llm-clean``: the graph, optional context, and the user's LLM key."""

    diagram: KGDiagramInput
    #: What the system is, in the user's words. Optional: when blank the model
    #: infers the target domain from the graph.
    description: str = Field(default="", max_length=20_000)


class KGOrphanClassifyRequest(KGLLMRequest):
    """Body of ``/kg/classify-orphans``: the LLM request plus the nodes to place."""

    nodeIds: List[str]
