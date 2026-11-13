"""Request models for the live agent simulation endpoints (/besser_api/simulation/*)."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator


class SimulationSessionInput(BaseModel):
    """Agent diagram (plus optional agent configuration) to simulate or validate."""
    title: str
    model: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    configYaml: Optional[str] = None
    # Optional LLM provider keys forwarded to the sandboxed agent as env vars:
    # openAiApiKey, huggingFaceToken, replicateApiKey.
    credentials: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def reject_legacy_v3_model(self) -> "SimulationSessionInput":
        """Reject a legacy v3 agent diagram with HTTP 400 (see ``DiagramInput``)."""
        from besser.utilities.web_modeling_editor.backend.services.validators.legacy_format import (
            ensure_not_legacy_model,
        )
        ensure_not_legacy_model(self.model, "AgentDiagram", self.title)
        return self
