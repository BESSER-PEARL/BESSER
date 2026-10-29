"""Request and response models of the traceability endpoints (``/besser_api/traceability/...``).

The frontend side of this contract is ``shared/api/traceability.ts`` in the
webapp; the ``trace`` payload is a trace file fragment, typed there as
``TraceFile``.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

from besser.utilities.web_modeling_editor.backend.constants.constants import (
    TRACE_ANONYMOUS_ACTOR,
    TRACE_MAX_ACTOR_NAME_LENGTH,
)
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput


class TracedGenerationInput(DiagramInput):
    """A ``/generate-output`` request whose response also carries the trace of the run.

    ``id`` (the diagram id) is required by the endpoint: the trace refers to the
    model by it. ``actorName`` is the project owner; blank means anonymous.
    """
    actorName: str = Field(default=TRACE_ANONYMOUS_ACTOR, max_length=TRACE_MAX_ACTOR_NAME_LENGTH)

    @field_validator("actorName")
    @classmethod
    def _blank_is_anonymous(cls, value: str) -> str:
        return value.strip() or TRACE_ANONYMOUS_ACTOR


class TracedGenerationResponse(BaseModel):
    """The generated file, as the plain download would deliver it, plus the trace fragment of the run."""
    fileName: str
    mediaType: str
    contentBase64: str
    trace: Dict[str, Any]
