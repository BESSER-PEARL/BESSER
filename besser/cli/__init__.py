"""The ``besser`` command-line interface.

``besser new``, ``validate``, ``info``, ``generators``, ``generate``,
``regenerate``, ``convert`` and ``completion``, for people and for coding agents
(``--json``, or the same commands as MCP tools with ``besser mcp``).
Loading a B-UML ``.py`` model executes it; see ``besser --help``.
"""

from besser.cli.main import main

__all__ = ["main"]
