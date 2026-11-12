"""The ``besser`` command-line interface.

``besser new``, ``init``, ``guide``, ``schema``, ``validate``, ``info``, ``model``,
``generators``, ``generate``, ``regenerate``, ``build``, ``apply``, ``convert``,
``completion`` and ``mcp``, for people and for coding agents
(``--json``, or the same commands as MCP tools with ``besser mcp``).
Loading a B-UML ``.py`` model executes it; see ``besser --help``.
"""

from besser.cli.main import main

__all__ = ["main"]
