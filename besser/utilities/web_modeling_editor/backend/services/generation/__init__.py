"""Service layer of the code generation endpoints (see ``routers/generation_router.py``).

- ``dispatch``: picks the generator path from its registry entry and runs it.
- ``agents``: agent (BAF) generation.
- ``downloads``: :class:`GeneratedOutput`, the ZIP or single file a request downloads.
"""

from .agents import generate_agent_files
from .dispatch import generate_from_diagram, generate_from_project
from .downloads import GeneratedOutput

__all__ = [
    "GeneratedOutput",
    "generate_agent_files",
    "generate_from_diagram",
    "generate_from_project",
]
