"""``besser mcp``: the CLI's commands as tools of a Model Context Protocol server on stdio.

Each tool calls the ``commands`` function the command line calls and returns
the command's ``--json`` object (``Result.as_json()``), as structured content
and as JSON text, flagged as an error when ``ok`` is false.

The tools work on files, so the server confines them: every path a tool is
given is resolved (``..`` and symbolic links included) and must be inside
``--root``; standard input (``-``) is not a model file here; and a ``.py``
model, which is Python code executed to load it, is refused unless the server
was started with ``--exec-py``.

The MCP SDK (``besser[mcp]``) is imported by ``serve`` only, so the rest of
the command line never needs it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from besser.cli import commands
from besser.cli.loading import STDIN
from besser.cli.output import EXIT_FAILED, Result, failure, stdout_to_stderr
from besser.generators.registry import GENERATORS
from besser.utilities.provenance import get_besser_version

SERVER_NAME = "besser"
MCP_EXTRA = 'pip install "besser[mcp]"'

_PATH = {"type": "string", "description": "a path under the server's root, relative to it or absolute"}
_MODEL = {**_PATH, "description": "a model file under the server's root: an editor .json export, "
                                  "or a B-UML .py file if the server allows it (--exec-py)"}
_OPTIONS = {
    "description": "generator options, as {name: value} or as a list of 'name=value' strings "
                   "(a list value is comma-separated there); the 'generators' tool lists each generator's options",
    "anyOf": [
        {"type": "object", "additionalProperties": {
            "type": ["string", "integer", "boolean", "array"], "items": {"type": ["string", "integer", "boolean"]},
        }},
        {"type": "array", "items": {"type": "string"}},
    ],
}
_USE = {"type": "array", "items": {"type": "string"},
        "description": "the model to read, by name, when the file holds several of a kind ('info' lists them)"}
_DRY_RUN = {"type": "boolean", "default": False, "description": "report the files it would write, and write nothing"}

# What every tool returns: the keys of the CLI's --json object; the command adds its own.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "command": {"type": "string"},
        "exit_code": {"type": "integer", "description": "the CLI's exit status: 0 ok, 1 failed, 2 could not run"},
        "errors": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "files": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ok", "command", "exit_code", "errors", "warnings", "notes", "files"],
}


class _Refused(Exception):
    """A tool argument the server does not accept; carries the message."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the client sees, and the command it runs."""

    name: str
    description: str
    input_schema: dict[str, Any]
    # The tool writes files (generate, regenerate, convert).
    writes: bool
    run: Callable[[Tools, dict[str, Any]], Result]


class Tools:
    """The tools of one server: its root and whether ``.py`` models may be loaded."""

    def __init__(self, root: Path, exec_py: bool = False):
        self.root = root.resolve()
        self.exec_py = exec_py

    # ── paths ───────────────────────────────────────────────────────────────

    def path(self, value: str, what: str) -> Path:
        """``value`` resolved against the root, refused unless it stays inside it.

        Resolving follows ``..`` and every symbolic link, so a link to a file
        outside the root is refused too; the command is then given the resolved path.
        """
        if value == STDIN:
            raise _Refused(f"{what}: '-' (standard input) is not available over MCP; name a file under {self.root}")
        try:
            resolved = (self.root / value).resolve()
        except (OSError, ValueError, RuntimeError) as exc:  # a null byte, a symbolic link loop
            raise _Refused(f"{what} {value!r} is not a usable path: {exc}") from None
        if not resolved.is_relative_to(self.root):
            raise _Refused(f"{what} {value!r} is outside the server's root {self.root}; "
                           "the tools only read and write files under it")
        return resolved

    def model_path(self, value: str, what: str = "model") -> Path:
        """A model file under the root; a ``.py`` one only when the server executes them."""
        resolved = self.path(value, what)
        if resolved.suffix.lower() == ".py" and not self.exec_py:
            raise _Refused(
                f"{what} {value!r} is a B-UML .py model, which is Python code and is executed to load it; "
                "this server loads .json models only. Restart it with 'besser mcp --exec-py' to allow .py models"
            )
        if resolved.is_dir():
            self.check_links(resolved)
        return resolved

    def output_path(self, value: str, what: str) -> Path:
        """A file or folder to write under the root, holding no link that leads out of it."""
        resolved = self.path(value, what)
        if resolved.is_dir():
            self.check_links(resolved)
        return resolved

    def check_links(self, folder: Path) -> None:
        """Refuse ``folder`` if a symbolic link in it resolves outside the root.

        A command walking or writing into the folder would follow it.
        """
        for directory, subdirectories, files in os.walk(folder, followlinks=False):
            for name in (*subdirectories, *files):
                entry = Path(directory, name)
                if entry.is_symlink() and not entry.resolve().is_relative_to(self.root):
                    raise _Refused(f"{entry} is a symbolic link to a path outside the server's root {self.root}")

    # ── calls ───────────────────────────────────────────────────────────────

    def call(self, name: str, arguments: dict[str, Any] | None) -> Result:
        """Run the tool ``name`` with ``arguments``; a refused or invalid argument is a usage failure (2)."""
        import jsonschema  # a dependency of the MCP SDK, which serve() has imported

        spec = TOOLS_BY_NAME[name]
        arguments = arguments or {}
        try:
            jsonschema.validate(arguments, spec.input_schema)
        except jsonschema.ValidationError as exc:
            where = ".".join(str(part) for part in exc.absolute_path)
            return failure(name, f"invalid arguments: {where + ': ' if where else ''}{exc.message}")
        try:
            return spec.run(self, arguments)
        except _Refused as exc:
            return failure(name, str(exc))


def _option_assignments(options: dict[str, Any] | list[str] | None) -> list[str]:
    """Tool options as the CLI's ``name=value`` arguments, which ``commands.parse_options`` reads."""
    if options is None:
        return []
    if isinstance(options, list):
        return list(options)

    def text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return ",".join(text(item) for item in value)
        return str(value)

    return [f"{name}={text(value)}" for name, value in options.items()]


def _validate(tools: Tools, args: dict[str, Any]) -> Result:
    paths = [str(tools.model_path(path)) for path in args["paths"]]
    return commands.validate(paths, strict=args.get("strict", False), exec_py=tools.exec_py,
                             ocl=args.get("ocl", False))


def _info(tools: Tools, args: dict[str, Any]) -> Result:
    return commands.info(str(tools.model_path(args["model"])))


def _generators(tools: Tools, args: dict[str, Any]) -> Result:
    return commands.generators()


def _generate(tools: Tools, args: dict[str, Any]) -> Result:
    return commands.generate(
        args["generator"], str(tools.model_path(args["model"])), str(tools.output_path(args["output_dir"], "output_dir")),
        _option_assignments(args.get("options")), args.get("use", []), args.get("force", False),
        dry_run=args.get("dry_run", False),
    )


def _regenerate(tools: Tools, args: dict[str, Any]) -> Result:
    return commands.regenerate(
        args["generator"], str(tools.model_path(args["model"])), str(tools.output_path(args["output_dir"], "output_dir")),
        _option_assignments(args.get("options")), args.get("use", []), args.get("adopt", False),
        dry_run=args.get("dry_run", False),
    )


def _convert(tools: Tools, args: dict[str, Any]) -> Result:
    return commands.convert(str(tools.model_path(args["input"], "input")),
                            str(tools.output_path(args["output"], "output")), dry_run=args.get("dry_run", False))


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_GENERATOR_IDS = [spec.id for spec in GENERATORS]
_REGENERATE_IDS = [spec.id for spec in GENERATORS if spec.supports_regenerate]

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "validate",
        "Load model files and run validate() on every model in them (the CLI's 'besser validate'). "
        "A directory is searched for model files. ok is false on errors, or on warnings with strict.",
        _object({
            "paths": {"type": "array", "minItems": 1, "items": _MODEL,
                      "description": "model files or directories of them, under the server's root"},
            "strict": {"type": "boolean", "default": False, "description": "fail on warnings too"},
            "ocl": {"type": "boolean", "default": False,
                    "description": "also check OCL constraints: parse each, and evaluate them on object models"},
        }, ["paths"]),
        False, _validate,
    ),
    ToolSpec(
        "info",
        "Summarise a model file: its models, the classes, attributes and associations of a domain model, "
        "and the ids of the generators that can run on it (the CLI's 'besser info').",
        _object({"model": _MODEL}, ["model"]),
        False, _info,
    ),
    ToolSpec(
        "generators",
        "List the generators: id, the models each reads, its options as JSON Schema, and whether "
        "regenerate supports it (the CLI's 'besser generators').",
        _object({}, []),
        False, _generators,
    ),
    ToolSpec(
        "generate",
        "Validate the models a generator reads, run it into output_dir and list the files written "
        "(the CLI's 'besser generate'). A model with validation errors is not generated from. "
        "A non-empty output_dir is refused unless force is true, which overwrites; to keep hand edits "
        "in generated code, use regenerate.",
        _object({
            "generator": {"type": "string", "enum": _GENERATOR_IDS, "description": "the generator id"},
            "model": _MODEL,
            "output_dir": {**_PATH, "description": "the folder to write to, under the server's root"},
            "options": _OPTIONS,
            "use": _USE,
            "force": {"type": "boolean", "default": False,
                      "description": "write into a folder that is not empty, overwriting files in it"},
            "dry_run": _DRY_RUN,
        }, ["generator", "model", "output_dir"]),
        True, _generate,
    ),
    ToolSpec(
        "regenerate",
        "Bring generated code in output_dir up to a changed model and keep the edits made in it: a "
        "three-way merge with the output recorded in .besser_generated.json (the CLI's 'besser regenerate'). "
        "Conflicts get conflict markers and ok false. Needs besser[regen].",
        _object({
            "generator": {"type": "string", "enum": _REGENERATE_IDS,
                          "description": "the generator id; only these generators support regenerate"},
            "model": _MODEL,
            "output_dir": {**_PATH, "description": "the folder holding the generated code, under the server's root"},
            "options": _OPTIONS,
            "use": _USE,
            "adopt": {"type": "boolean", "default": False,
                      "description": "record a folder generated without a record as this model's output, "
                                     "changing no file"},
            "dry_run": _DRY_RUN,
        }, ["generator", "model", "output_dir"]),
        True, _regenerate,
    ),
    ToolSpec(
        "convert",
        "Convert a B-UML .py model to an editor .json project, or an editor .json export to a B-UML .py "
        "file; the direction follows the two extensions (the CLI's 'besser convert'). The output is overwritten.",
        _object({
            "input": {**_MODEL, "description": "the file to convert (.json, or .py if the server allows it)"},
            "output": {**_PATH, "description": "the .py or .json file to write, under the server's root"},
            "dry_run": _DRY_RUN,
        }, ["input", "output"]),
        True, _convert,
    ),
)
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def _instructions(tools: Tools) -> str:
    loads = ".json and .py models" if tools.exec_py else ".json models only (.py models need --exec-py)"
    return (
        "BESSER's command line as tools, on B-UML model files: validate, info, generators, generate, "
        f"regenerate, convert. Paths are under {tools.root}. This server loads {loads}. "
        "Each tool returns the CLI's --json object: ok, errors, warnings, notes, files and its own keys."
    )


def serve(root: str, exec_py: bool = False) -> Result:
    """Serve the tools on standard input and output until the client disconnects."""
    try:
        import anyio
        from anyio import to_thread
        import mcp_types as types
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server
        from mcp.shared.exceptions import MCPError
    except ImportError as exc:
        return failure("mcp", f"besser mcp needs the MCP SDK, which is not installed ({exc}): {MCP_EXTRA}")

    if not Path(root).is_dir():
        return failure("mcp", f"--root {root}: no such directory")
    tools = Tools(Path(root), exec_py=exec_py)
    listed = [
        types.Tool(
            name=spec.name, description=spec.description, input_schema=spec.input_schema,
            output_schema=_OUTPUT_SCHEMA,
            annotations=types.ToolAnnotations(read_only_hint=not spec.writes, destructive_hint=spec.writes,
                                              open_world_hint=False),
        )
        for spec in TOOLS
    ]
    # One command at a time: loading a model and running a generator use process-wide state
    # (standard output, the module cache), and commands may write the same files.
    one_at_a_time = threading.Lock()

    def run_tool(name: str, arguments: dict[str, Any] | None) -> Result:
        with one_at_a_time, stdout_to_stderr():
            try:
                return tools.call(name, arguments)
            except Exception as exc:  # a defect in BESSER, as main() reports it
                traceback.print_exc(file=sys.stderr)
                return failure(name, f"unexpected {type(exc).__name__}: {exc}", EXIT_FAILED)

    async def list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx: Any, params: Any) -> Any:
        if params.name not in TOOLS_BY_NAME:
            raise MCPError(types.INVALID_PARAMS, f"unknown tool {params.name!r}; tools: {', '.join(TOOLS_BY_NAME)}")
        # In a worker thread, so the server keeps answering (pings) while a generator runs.
        result = await to_thread.run_sync(run_tool, params.name, params.arguments)
        output = result.as_json()
        return types.CallToolResult(
            content=[types.TextContent(text=json.dumps(output, indent=2))],
            structured_content=output, is_error=not output["ok"],
        )

    server = Server(SERVER_NAME, version=get_besser_version(), instructions=_instructions(tools),
                    on_list_tools=list_tools, on_call_tool=call_tool)

    async def run() -> None:
        # stdio_server serves the protocol on a private copy of standard output and points
        # descriptor 1 at standard error while it runs; run_tool redirects again, for sys.stdout too.
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    try:
        anyio.run(run)
    except KeyboardInterrupt:
        pass
    return Result(command="mcp")
