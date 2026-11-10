"""``besser mcp``: the CLI's commands as tools of a Model Context Protocol server on stdio."""

from __future__ import annotations

from besser.cli.output import Result, failure


def serve(root: str, exec_py: bool = False) -> Result:
    """Serve the tools on standard input and output until the client disconnects."""
    return failure("mcp", "not implemented yet")  # TODO(mcp agent)
