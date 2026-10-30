"""
Traceability Router

The traceability endpoints of the BESSER web modeling editor backend, all under
``/besser_api/traceability``. The editor records modeling traces itself and
keeps the trace file in the browser; the backend records what only it can see,
the trace links of a generation run. Logic lives in ``services/traceability``.
"""

from fastapi import APIRouter

from besser.utilities.web_modeling_editor.backend.models.traceability import (
    TracedGenerationInput,
    TracedGenerationResponse,
)
from besser.utilities.web_modeling_editor.backend.routers.error_handler import handle_endpoint_errors
from besser.utilities.web_modeling_editor.backend.services.traceability import generate_with_trace

router = APIRouter(prefix="/besser_api/traceability", tags=["traceability"])


@router.post("/generate-output", response_model=TracedGenerationResponse)
@handle_endpoint_errors("generate_traced_output")
async def generate_traced_output(input_data: TracedGenerationInput) -> TracedGenerationResponse:
    """Generate code from a class diagram and return the file together with the trace of the run.

    Same input as ``/generate-output`` plus ``actorName``, for a generator in
    ``TRACED_GENERATORS``. The file comes back base64-encoded in JSON so the
    trace can travel with it.
    """
    return await generate_with_trace(input_data)
