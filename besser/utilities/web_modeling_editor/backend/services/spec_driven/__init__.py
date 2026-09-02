"""Smart generation service package.

Glue between the LLM orchestrator (``besser.generators.llm``) and the
FastAPI SSE endpoint. Exposes:

- Event models and the ``format_sse`` serializer used by the router.
- The ``SmartGenerationRunner`` that drives a run and yields SSE bytes.
- The in-memory ``SmartRunRegistry`` that tracks generated-output paths
  awaiting download.
- ``assemble_models_from_project`` which converts a ``ProjectInput`` into
  the BUML models the LLM orchestrator needs.
"""

from .sse_events import (
    BaseSseEvent,
    CostEvent,
    DoneEvent,
    ErrorEvent,
    PhaseEvent,
    StartEvent,
    TextDeltaEvent,
    ToolCallEvent,
    format_sse,
)
from .model_assembly import AssembledModels, assemble_models_from_project
from .runner import SMART_RUN_REGISTRY, SmartGenerationRunner, SmartRunEntry, SmartRunRegistry

__all__ = [
    # Events
    "BaseSseEvent",
    "StartEvent",
    "PhaseEvent",
    "TextDeltaEvent",
    "ToolCallEvent",
    "CostEvent",
    "DoneEvent",
    "ErrorEvent",
    "format_sse",
    # Model assembly
    "AssembledModels",
    "assemble_models_from_project",
    # Runner
    "SmartGenerationRunner",
    "SmartRunRegistry",
    "SmartRunEntry",
    "SMART_RUN_REGISTRY",
]
