"""Request models of the Knowledge Graph endpoints.

The KG-only fields used to sit on the shared :class:`DiagramInput`, which every
other diagram endpoint also parses. They live here instead, on a subclass that
only the KG endpoints accept.

The LLM endpoints follow the provider / model / ``SecretStr`` shape of
:class:`~besser.utilities.web_modeling_editor.backend.models.spec_driven.SmartGenerateRequest`.
The key is a ``SecretStr`` so Pydantic never renders it in a ``repr`` or a
validation error, and it is only unwrapped by :meth:`KGLLMRequest.resolved_api_key`.

Only the *shape* is validated here. Semantic checks (a blank key, an empty or
oversized ``node_ids``) stay in the endpoints as ``ValidationError``, because a
Pydantic failure is a 422 whose ``detail`` is a list, and the editor's
``apiClient`` shows the ``detail`` of a failed call as the message.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, SecretStr

from besser.utilities.web_modeling_editor.backend.constants.constants import DEFAULT_KG_LLM_MODEL
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput

#: Providers a user's own key can be used with. ``free`` / ``sponsored`` are
#: server-paid tiers and the ``base_url`` providers (PIA, local) open a
#: user-chosen URL (an SSRF surface), so none of them are offered here.
KGLLMProvider = Literal["anthropic", "openai", "mistral", "nebius"]


class KGDiagramInput(DiagramInput):
    """A Knowledge Graph diagram plus the refinement fields of the KG endpoints."""

    # KG → BUML preflight integration. ``resolutions`` lets the frontend
    # send back user choices for issues surfaced by
    # ``/analyze-kg-for-buml-conversion``; ``kgSignature`` is the value
    # echoed from the analyze response so the backend can detect a
    # stale resolution payload (KG mutated between analyze & convert).
    resolutions: Optional[List[Dict[str, Any]]] = None
    kgSignature: Optional[str] = None
    # LLM-cleanup round-trip: ``llmIssues`` carries the issue list returned
    # by ``/llm-clean-kg`` so ``/apply-kg-refinement`` can reconstruct the
    # KGIssue / KGAction objects without re-calling the LLM.
    llmIssues: Optional[List[Dict[str, Any]]] = None
    # KG refinement (unified Refine KG modal): ``source`` selects which
    # apply path the unified ``/apply-kg-refinement`` endpoint takes —
    # ``"static"`` re-runs the static analyzer and dispatches accept/skip
    # decisions, ``"llm"`` reconstructs LLM-issue objects from
    # ``llmIssues`` and dispatches them.
    source: Optional[Literal["static", "llm"]] = None
    # KG RDF export: which constraint vocabularies to emit. ``"owl"`` emits
    # only ``owl:Restriction`` blank nodes; ``"shacl"`` emits only
    # ``sh:NodeShape`` / ``sh:PropertyShape``; ``"both"`` (default) emits
    # both where each spec has the required semantics.
    vocab: Optional[Literal["owl", "shacl", "both"]] = None


class KGLLMRequest(BaseModel):
    """Body of ``/llm-clean-kg``: the graph, optional context, and the user's LLM key."""

    diagram: KGDiagramInput
    #: What the system is, in the user's words. Optional: when blank the model
    #: infers the target domain from the graph.
    description: str = Field(default="", max_length=20_000)
    provider: KGLLMProvider = "openai"
    api_key: SecretStr
    #: ``None`` means the provider's default model (see :meth:`resolved_model`).
    llm_model: Optional[str] = Field(default=None, max_length=120)

    def resolved_api_key(self) -> str:
        """The key, unwrapped. Pass it straight to the provider client; never log or echo it."""
        return self.api_key.get_secret_value()

    def resolved_model(self) -> str:
        """The requested model, else the provider's default."""
        chosen = (self.llm_model or "").strip()
        if chosen:
            return chosen
        if self.provider == "openai":
            return DEFAULT_KG_LLM_MODEL
        # Imported here: the provider SDKs are heavy and only this branch needs them.
        from besser.spec_driven_agent.providers.llm_client import DEFAULT_MODELS

        return DEFAULT_MODELS[self.provider]


class KGOrphanClassifyRequest(KGLLMRequest):
    """Body of ``/classify-orphans-with-llm``: the LLM request plus the nodes to place."""

    node_ids: List[str]
