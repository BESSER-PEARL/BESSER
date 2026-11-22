"""Request models for the live agent simulation endpoints (/besser_api/simulation/*)."""
from typing import Any, Dict, Optional

from pydantic import BaseModel


class SimulationSessionInput(BaseModel):
    """Agent diagram (plus optional agent configuration) to simulate or validate."""
    title: str
    model: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    configYaml: Optional[str] = None
    # Optional LLM provider keys forwarded to the sandboxed agent as env vars:
    # openAiApiKey, huggingFaceToken, replicateApiKey.
    credentials: Optional[Dict[str, str]] = None
