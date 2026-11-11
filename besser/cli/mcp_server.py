"""``besser mcp``: the CLI's commands as tools of a Model Context Protocol server on stdio.

Each tool calls the ``commands`` function the command line calls and returns
the command's ``--json`` object (``Result.as_json()``), as structured content
and as JSON text, flagged as an error when ``ok`` is false. The object is
worded for the client (:func:`for_client`): its paths are relative to ``--root``,
and where a message names a CLI flag it names the tool argument instead.

The tools work on files, so the server confines them: every path a tool is
given is resolved (``..`` and symbolic links included) and must be inside
``--root``; standard input (``-``) is not a model file here; and a ``.py``
model, which is Python code executed to load it, is refused unless the server
was started with ``--exec-py``.

The MCP SDK (``besser[mcp]``) is imported by ``serve`` only, so the rest of
the command line never needs it.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from besser.cli import commands
from besser.cli.importing import IMPORT_FORMATS
from besser.cli.loading import STDIN
from besser.cli.output import EXIT_FAILED, Result, failure, stdout_to_stderr
from besser.cli.starters import DEFAULT_KIND, STARTERS
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
    # The tool writes files (new, generate, regenerate, convert).
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
                if not entry.is_symlink():
                    continue
                try:
                    target = entry.resolve()
                except (OSError, RuntimeError):  # a symbolic link loop leads nowhere, so not out of the root
                    continue
                if not target.is_relative_to(self.root):
                    raise _Refused(f"{entry} is a symbolic link to a path outside the server's root {self.root}")

    # ── calls ───────────────────────────────────────────────────────────────

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Run the tool ``name`` with ``arguments`` and return its JSON object, worded for the client.

        A refused or invalid argument is a usage failure (2).
        """
        import jsonschema  # a dependency of the MCP SDK, which serve() has imported

        spec = tool_specs()[name]
        arguments = arguments or {}
        try:
            jsonschema.validate(arguments, spec.input_schema)
        except jsonschema.ValidationError as exc:
            where = ".".join(str(part) for part in exc.absolute_path)
            return failure(name, f"invalid arguments: {where + ': ' if where else ''}{exc.message}").as_json()
        try:
            result = spec.run(self, arguments)
        except _Refused as exc:  # its message names the root on purpose
            return failure(name, str(exc)).as_json()
        return for_client(result.as_json(), self.root)


# ── wording for the client ──────────────────────────────────────────────────

# A CLI command quoted in a message ('besser regenerate sql model.py -o out --adopt').
# In single quotes, or double quotes when the command quotes an option value itself.
_QUOTED_COMMAND = re.compile(r"""(['"])besser ([a-z]+)(.*?)\1""")
# A flag the CLI's message names, and the tool argument that does the same.
_FLAG_ARGUMENTS = (
    ("pass --exec-py", "restart the server with --exec-py"),
    ("--use NAME", "the `use` argument"),
    ("--option", "the `options` argument"),
)
# CLI flags that are a tool's boolean argument of the same name ("--dry-run" -> "dry_run: true").
_BOOLEAN_FLAG = re.compile(r"--(force|adopt|strict|ocl|dry-run)\b")
_MESSAGE_KEYS = frozenset({"errors", "warnings", "notes"})


def _boolean_argument(match: re.Match) -> str:
    return f"`{match.group(1).replace('-', '_')}: true`"


# The positional arguments of a quoted command, as tool arguments in order.
_POSITIONAL_ARGUMENTS = {
    "generate": ("generator", "model"), "regenerate": ("generator", "model"),
    "info": ("model",), "convert": ("input",), "validate": ("paths",), "new": ("path",), "schema": ("topic",),
}
# A placeholder in a quoted command ("<that model>"): one argument.
_PLACEHOLDER = re.compile(r"<[^<>]*>")


def _tool_call(tool: str, command_line: str) -> str | None:
    """The tool arguments of the CLI ``command_line`` (what follows 'besser <tool>'), as JSON; ``None`` if it
    does not parse."""
    try:
        words = shlex.split(_PLACEHOLDER.sub(lambda match: shlex.quote(match.group(0)), command_line))
    except ValueError:
        return None
    arguments: dict[str, Any] = {}
    positional = iter(_POSITIONAL_ARGUMENTS.get(tool, ()))
    words = iter(words)
    for word in words:
        if word in ("-o", "--output"):
            arguments["output" if tool == "convert" else "output_dir"] = next(words, "")
        elif word.startswith(("--option", "--use")):
            name, _, value = word.partition("=")
            value = value or next(words, "")
            arguments.setdefault("options" if name == "--option" else "use", []).append(value)
        elif _BOOLEAN_FLAG.fullmatch(word):
            arguments[word[2:].replace("-", "_")] = True
        elif word.startswith("-"):
            return None
        else:
            name = next(positional, None)
            if name is None:
                return None
            arguments[name] = [word] if name == "paths" else word
    return json.dumps(arguments)


def _tool_wording(message: str) -> str:
    """``message`` with the CLI commands and flags it names replaced by the tools and arguments."""
    def tool(match: re.Match) -> str:
        if match.group(2) not in tool_specs():
            return match.group(0)
        arguments = _tool_call(match.group(2), match.group(3))
        if arguments is None:
            flags = [_boolean_argument(flag) for flag in _BOOLEAN_FLAG.finditer(match.group(3))]
            return f"the `{match.group(2)}` tool" + (f" with {', '.join(flags)}" if flags else "")
        return f"the `{match.group(2)}` tool with {arguments}"

    message = _QUOTED_COMMAND.sub(tool, message)
    for flag, argument in _FLAG_ARGUMENTS:
        message = message.replace(flag, argument)
    return _BOOLEAN_FLAG.sub(_boolean_argument, message)


def _root_relative(text: str, root: Path) -> str:
    """``text`` with every path under ``root`` written relative to it (``root`` itself as ``.``)."""
    if root == Path(root.anchor):  # --root / : every path is under it, and absolute is clearer
        return text
    for spelling in dict.fromkeys((str(root), root.as_posix())):
        # The root where a path starts (the text's start, or after a space, a quote or a bracket),
        # followed by a separator, or by where a path ends (the text's end, a space, a quote, ...).
        text = re.sub(r"""(?<![^\s'"(\[{,=])""" + re.escape(spelling) + r"""(?:[/\\]|(?=$|[\s'"):;,\]}]))""",
                      lambda match: "" if match.group(0)[-1] in "/\\" else ".", text)
    return text


def for_client(value: Any, root: Path, key: str = "") -> Any:
    """A tool's JSON object as the client reads it.

    The commands are given absolute paths, so their output holds them: each is made relative to
    ``root``, the paths the client passes. The messages (``errors``, ``warnings``, ``notes``, at any
    depth) name CLI commands and flags: they name the tools and their arguments instead.
    """
    if isinstance(value, dict):
        return {_root_relative(name, root): for_client(item, root, name) for name, item in value.items()}
    if isinstance(value, list):
        return [for_client(item, root, key) for item in value]
    if isinstance(value, str):
        text = _root_relative(value, root)
        return _tool_wording(text) if key in _MESSAGE_KEYS else text
    return value


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
        commands.option_assignments(args.get("options")), args.get("use", []), args.get("force", False),
        dry_run=args.get("dry_run", False),
    )


def _regenerate(tools: Tools, args: dict[str, Any]) -> Result:
    return commands.regenerate(
        args["generator"], str(tools.model_path(args["model"])), str(tools.output_path(args["output_dir"], "output_dir")),
        commands.option_assignments(args.get("options")), args.get("use", []), args.get("adopt", False),
        dry_run=args.get("dry_run", False),
    )


def _convert(tools: Tools, args: dict[str, Any]) -> Result:
    inputs = [str(tools.model_path(value, "input")) for value in [args["input"], *args.get("more_inputs", [])]]
    return commands.convert(inputs, str(tools.output_path(args["output"], "output")),
                            dry_run=args.get("dry_run", False), source_format=args.get("from"),
                            class_diagram=args.get("class_diagram", False))


def _new(tools: Tools, args: dict[str, Any]) -> Result:
    # The tool writes the one file it names; an MCP client learns the workflow from the tools' descriptions.
    return commands.new(str(tools.output_path(args["path"], "path")), kind=args.get("kind", DEFAULT_KIND),
                        force=args.get("force", False), agent_files=False)


def _schema(tools: Tools, args: dict[str, Any]) -> Result:
    return commands.schema(args.get("topic"))


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


@functools.cache
def tool_specs() -> dict[str, ToolSpec]:
    """The tools, by name.

    Built on first use: which generators support regenerate is known by importing each one's class.
    """
    generator_ids = [spec.id for spec in GENERATORS]
    regenerate_ids = [spec.id for spec in GENERATORS if spec.supports_regenerate]
    return {tool.name: tool for tool in _tools(generator_ids, regenerate_ids)}


def _tools(generator_ids: list[str], regenerate_ids: list[str]) -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            "new",
            "Write a starter model of a kind to edit: a small, commented B-UML .py script, or the same model "
            "as an editor .json project (the CLI's 'besser new'). An existing file is refused unless force is true.",
            _object({
                "path": {**_PATH, "description": "the .py or .json file to write, under the server's root"},
                "kind": {"type": "string", "enum": list(STARTERS), "default": DEFAULT_KIND,
                         "description": "the kind of model to start from"},
                "force": {"type": "boolean", "default": False, "description": "overwrite the file if it exists"},
            }, ["path"]),
            True, _new,
        ),
        ToolSpec(
            "schema",
            "Describe a format: the B-UML classes of a model kind with their constructor arguments, the JSON "
            "Schema of an editor project ('project') or of what every tool returns ('result'), or the "
            "besser.toml file ('config'); without a topic, the topics (the CLI's 'besser schema').",
            _object({"topic": {"type": "string", "enum": commands.schema_topics()}}, []),
            False, _schema,
        ),
        ToolSpec(
            "validate",
            "Load model files and run validate() on every model in them (the CLI's 'besser validate'). "
            "A directory is searched for model files. ok is false on errors, or on warnings with strict.",
            _object({
                "paths": {"type": "array", "minItems": 1, "items": _MODEL,
                          "description": "model files or directories of them, under the server's root"},
                "strict": {"type": "boolean", "default": False, "description": "fail on warnings too"},
                "ocl": {"type": "boolean", "default": False,
                        "description": "also check OCL constraints: parse each, and evaluate the invariants on object models (for editor JSON only when the server was started with --exec-py)"},
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
                "generator": {"type": "string", "enum": generator_ids, "description": "the generator id"},
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
                "generator": {"type": "string", "enum": regenerate_ids,
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
            "Convert a B-UML .py model to an editor .json project or the reverse, or import another notation "
            f"as a model ({', '.join(fmt.name for fmt in IMPORT_FORMATS)}); a knowledge graph is also written "
            "as RDF (.ttl, .owl, .rdf). What is read follows the input's extension, or 'from'; what is written, "
            "the output's (the CLI's 'besser convert'). The output is overwritten.",
            _object({
                "input": {**_MODEL, "description": "the file to convert (.json, or .py if the server allows it) "
                                                   "or to import; a directory for from: code"},
                "more_inputs": {"type": "array", "items": _PATH, "description": "further files, for from: csv"},
                "output": {**_PATH, "description": "the .py, .json or RDF file to write, under the server's root"},
                "from": {"type": "string", "enum": [fmt.name for fmt in IMPORT_FORMATS],
                         "description": "the notation of the input, when its extension does not say"},
                "class_diagram": {"type": "boolean", "default": False,
                                  "description": "write the class diagram derived from each knowledge graph"},
                "dry_run": _DRY_RUN,
            }, ["input", "output"]),
            True, _convert,
        ),
    )


def _instructions(tools: Tools) -> str:
    loads = ".json and .py models" if tools.exec_py else ".json models only (.py models need --exec-py)"
    return (
        "BESSER's command line as tools, on B-UML model files: new, schema, validate, info, generators, "
        f"generate, regenerate, convert. Start with new (a starter model) and schema (its classes). Paths are under {tools.root}. This server loads {loads}. "
        "Each tool returns the CLI's --json object: ok, errors, warnings, notes, files and its own keys."
    )


def serve(root: str, exec_py: bool = False) -> Result | None:
    """Serve the tools on standard input and output until the client disconnects.

    Returns the failure that kept it from serving, or ``None`` once it has served: standard
    output was the protocol's, so nothing is printed after the session.
    """
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
    specs = tool_specs()
    listed = [
        types.Tool(
            name=spec.name, description=spec.description, input_schema=spec.input_schema,
            output_schema=_OUTPUT_SCHEMA,
            annotations=types.ToolAnnotations(read_only_hint=not spec.writes, destructive_hint=spec.writes,
                                              open_world_hint=False),
        )
        for spec in specs.values()
    ]
    # One command at a time: loading a model and running a generator use process-wide state
    # (standard output, the module cache), and commands may write the same files.
    one_at_a_time = threading.Lock()

    def run_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        with one_at_a_time, stdout_to_stderr():
            try:
                return tools.call(name, arguments)
            except Exception as exc:  # a defect in BESSER, as main() reports it
                traceback.print_exc(file=sys.stderr)
                return failure(name, f"unexpected {type(exc).__name__}: {exc}", EXIT_FAILED).as_json()

    async def list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx: Any, params: Any) -> Any:
        if params.name not in specs:
            raise MCPError(types.INVALID_PARAMS, f"unknown tool {params.name!r}; tools: {', '.join(specs)}")
        # In a worker thread, so the server keeps answering (pings) while a generator runs.
        output = await to_thread.run_sync(run_tool, params.name, params.arguments)
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
    return None
