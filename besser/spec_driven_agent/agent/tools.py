"""
Tool definitions for the LLM-augmented generator.

Four categories:
1. **Generator tools** — BESSER's built-in code generators
2. **File tools** — read, write, modify, search files in the workspace
3. **Execution tools** — run commands, install deps, test code
4. **Validation tools** — check the application, test its API, keep the checklist
"""

import json
from typing import Any

from besser.generators.registry import agent_tools


# ======================================================================
# Generator tools
# ======================================================================

# Declared in besser.generators.registry, with the web editor's generators.
GENERATOR_TOOLS: list[dict[str, Any]] = [tool.schema() for tool in agent_tools()]


# ======================================================================
# File tools
# ======================================================================

# The write tools' argument for a file outside a fix attempt's files. Offered only
# while an attempt's edits are limited (``with_scope_widening``), not on every turn.
_NEEDED_FOR = {"type": "string", "description": (
    "Only when this fix attempt refused the file as outside its files: the listed "
    "blocker the file is needed to fix. Omit it otherwise.")}
_SCOPED_WRITE_TOOLS = frozenset({"write_file", "modify_file", "replace_file_lines", "delete_file"})


def with_scope_widening(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``tools`` with ``needed_for`` on the write tools, for a fix attempt whose edits are
    limited to its files."""
    widened = []
    for tool in tools:
        if tool.get("name") in _SCOPED_WRITE_TOOLS:
            schema = tool["input_schema"]
            tool = {**tool, "input_schema": {
                **schema, "properties": {**schema["properties"], "needed_for": _NEEDED_FOR}}}
        widened.append(tool)
    return widened

FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": (
            "List the workspace's files as '<path> <size in bytes>' lines; dependencies, "
            "virtualenvs and build output are skipped, and past 300 files the rest are counted."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": (
            "Read a workspace file; without offset and limit, the whole file. Every line "
            "comes back as 'N| code' with N its 1-based line number (quote the code without "
            "the prefix). offset is a 0-based skip: offset=50, limit=30 shows lines 51-80. "
            "Returns start_line, end_line and a read_id for replace_file_lines on the lines "
            "shown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the workspace"},
                "offset": {
                    "type": "integer",
                    "description": "Lines to SKIP before the first displayed line (0-based). Default: 0",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read. Default: all remaining lines, which is "
                                   "usually what you want",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file, or replace an existing file's whole content. An existing "
            "file must have been read in full (or written) this run. Refused, the file "
            "unchanged, when a parseable file would no longer parse or import, or when "
            "a rewrite of a file this run did not create (generated and over 200 lines; "
            "in a modify run, over 15 lines) drops its code: over 40% of its lines, a "
            "top-level declaration or class member, a route or registration, or code left as "
            "a placeholder. For a small change to a large file, modify_file is cheaper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the workspace"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "modify_file",
        "description": (
            "Replace old_text with new_text in an existing file. old_text must match exactly "
            "one place (indentation included) unless replace_all; keep it short, the lines "
            "that change plus enough neighbours to be unique, one call per change site, all "
            "in the same turn. Calls on one file apply in the order given, each to the file "
            "the previous left; calls on different files run in parallel. A miss changes "
            "nothing and returns the nearest lines (did_you_mean) or the span it located "
            "(located_range, with a read_id for replace_file_lines). already_applied: this "
            "exact edit landed before and the file is unchanged since; possible_replay: a "
            "duplicate-looking insertion, refused. An edit that would leave a parseable file "
            "unparseable or unimportable is refused."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Replace every exact occurrence. Leave false for normal edits; "
                        "true only when all matches intentionally need the same change."
                    ),
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "replace_file_lines",
        "description": (
            "Replace lines start_line..end_line (1-based, inclusive) of a file with new_text, "
            "without quoting them. read_id comes from a read_file or a located_range that "
            "showed those lines; once the file changes it is stale and refused, so read again "
            "before another range edit on the same file. new_text is the complete literal "
            "replacement (empty deletes the lines; to insert, include the anchor line). "
            "Refused, the file unchanged, for unseen lines, abbreviated code, or a replacement "
            "that breaks the file's syntax or import."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the workspace"},
                "read_id": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "new_text": {"type": "string"},
            },
            "required": ["path", "read_id", "start_line", "end_line", "new_text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_in_files",
        "description": (
            "Search workspace files for a case-insensitive regex (an invalid regex is "
            "searched as literal text). Returns up to 50 matches as {file, line, text}, "
            "each line clipped to 200 chars, with truncated=true when the cap is hit. "
            "node_modules, .git, venv/.venv, __pycache__, dist, build and lock files are skipped. "
            "Use it to locate code before read_file; it does not return file contents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "file_glob": {
                    "type": "string",
                    "description": "Optional filename glob, matched against the file NAME only (e.g. '*.py', '*.tsx'); a pattern containing '/' is refused.",
                    "default": "*",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "delete_file",
        "description": (
            "Delete one regular file (not a directory), by its path relative to the "
            "workspace root. The result lists imports of a top-level module the delete "
            "removed (still_imported). To change a generated file, edit it instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
]

# ======================================================================
# Execution tools — the key differentiator from a simple code writer
# ======================================================================

EXECUTION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_command",
        "description": (
            "Run a shell command in this run's shell and return stdout + stderr. Use it to "
            "test code, run linters and verify builds. Every command starts in the workspace "
            "root, or in working_dir: a `cd` lasts for its own command only. The shell is one "
            "session for the whole run: exported variables (`export X=1`, "
            "`source .venv/bin/activate`) carry over to the next command, and a process "
            "started in the background (`python app.py > app.log 2>&1 &`) keeps running "
            "after its command returns, so a server started in one command can be tested "
            "with curl in the next. What a background process prints after its command "
            "returned is discarded: redirect it to a file. Each command has a 120-second "
            "timeout unless it sets `timeout`; a command still running then is killed with everything it started, "
            "while the session and earlier background processes keep running, so a server "
            "started in the foreground blocks for the full timeout: start servers in the "
            "background instead. When a result's notes "
            "say the session was restarted or the command ran in a one-off sandbox, earlier "
            "background processes are gone. A result's note says when exit_code is only that of the "
            "last of several commands joined with ';' and an earlier one failed. Destructive or exfiltrating commands "
            "are refused with an error. stdout/stderr are truncated (~15k chars total); "
            "when cut, full_output_path names a file holding the complete log. A command "
            "whose runtime is not installed returns success=false, missing_command=true: "
            "treat its check as failed, not as passing. git is not installed and the workspace is "
            "not a git repository. Several run_command calls in one turn run "
            "one after another in the order given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_dir": {
                    "type": "string",
                    "description": "Directory for this command, relative to the workspace root. "
                                   "Default: the workspace root.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Seconds before this command is killed: max 600, and never past "
                        "the run's remaining time. Raise it only for a command known to be "
                        "slow (a long build: cargo build, gradle; a first npm or pip install: "
                        "300)."
                    ),
                },
            },
            "required": ["command"],
        },
    },
]

# ======================================================================
# Validation tools
# ======================================================================

# One drop or blocked entry: drop says why the user did not ask for the item,
# blocked what remains unresolved and why.
_TASK_REASON = {"type": "object", "required": ["id", "reason"],
                "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}}}

VALIDATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "test_api",
        "description": (
            "Exercise a real generated FastAPI workflow in an isolated copy with a fresh SQLite "
            "database. Submit up to 20 requests; state is shared within this call only. "
            "Use prior response JSON via {{n.path}}, e.g. {{0.room.id}}, in later JSON values, paths, "
            "header values or expected_fields values, also inside lists; a list item is a path "
            "part, e.g. {{2.items.0.id}}. "
            "To test a protected API, register and log in within the same scenario and pass the "
            "token as a header on each later request, e.g. headers={\"Authorization\": "
            "\"Bearer {{1.access_token}}\"} when request 1 was the login (use the token field the "
            "login response really has). Allowed headers: Authorization, Accept, Content-Type, "
            "Accept-Language, If-Match, If-None-Match, Idempotency-Key and X-* API headers such as "
            "X-API-Key; cookies set by a response are kept for the rest of the scenario. "
            "Credential header values are redacted from results. "
            "Omitted expected_status means 2xx. Assert business results with expected_fields "
            "(dotted JSON path -> expected value). Test happy paths AND invalid input/state transitions "
            "from the original specification. Valid scenarios (at most 10) are retained and rerun after "
            "code changes; failures block completion. Give each workflow a scenario_id; to correct "
            "a mistaken test payload/assertion reuse that ID with correction_reason and resubmit "
            "the whole scenario: a correction that leaves out a request which passed is refused, so "
            "put unrelated checks under their own scenario_id. Never "
            "weaken the specification to make broken code pass. Derive expected values from the "
            "original specification, not the current implementation. action='list' lists retained "
            "scenarios; action='get' with scenario_id reads its exact requests, expectations and "
            "last report without executing it. action='run' (default) with just scenario_id replays "
            "the saved definition. Inspect failed tests before changing either code or expectations. "
            "action='retire' with scenario_id and correction_reason drops a scenario another one replaced; "
            "it no longer runs and is reported as retired. "
            "No shell or external URLs. "
            "Passing proves only the submitted scenario, not every requirement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["run", "list", "get", "retire"]},
                "scenario_id": {"type": "string", "description": "Stable workflow name (max 80 characters)"},
                "correction_reason": {"type": "string", "description": "Required when changing or retiring a saved scenario; explain why the previous test was wrong or what replaced it"},
                "backend": {"type": "string", "description": "Optional relative generated backend directory"},
                "requests": {
                    "type": "array", "minItems": 1, "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                            "path": {"type": "string", "description": "Local API path beginning with /"},
                            "json": {},
                            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                            "expected_status": {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]},
                            "expected_fields": {"type": "object", "additionalProperties": True},
                        },
                        "required": ["method", "path"], "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "validate_app",
        "description": (
            "Run the harness's application checks now: Python declarations, schema/router "
            "contracts, and (when enabled) isolated backend startup and create requests. "
            "No arbitrary shell commands or package installation. Call after a coherent "
            "code change, not again without one: ending your turn runs it too. Fix the "
            "returned blockers, then rerun. "
            "A passing result covers these checks only, not all business requirements. "
            "citations: the whole line a 'requirement unverified' blocker asks you to cite."
        ),
        "input_schema": {"type": "object", "properties": {"citations": {"type": "array", "items": {
            "type": "object", "required": ["requirement", "path", "line"],
            "properties": {"requirement": {"type": "integer"}, "path": {"type": "string"},
                           "line": {"type": "string"}}}}}},
    },
    {
        "name": "task_list",
        "description": (
            "Your checklist for this run. Every field is optional and one call may combine "
            "them: done marks items complete, add appends items you found, drop closes an "
            "item the user did not ask for, blocked records required work you cannot finish. "
            "A call with no fields lists every item. Every result lists the open items. The "
            "run does not finish while items are open. An item with a machine check closes "
            "by itself once the check passes, and done on it is refused while it fails; any "
            "other item closes when you mark it done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "done": {"type": "array", "items": {"type": "integer"}},
                "add": {"type": "array", "items": {"type": "string"}},
                "drop": {"type": "array", "items": _TASK_REASON},
                "blocked": {"type": "array", "items": _TASK_REASON},
            },
        },
    },
]


def get_all_tools_including_generators() -> list[dict[str, Any]]:
    """Return ALL tools (including generators) — for no-generator mode."""
    return (
        GENERATOR_TOOLS
        + FILE_TOOLS
        + EXECUTION_TOOLS
        + VALIDATION_TOOLS
    )


# Which tool names require which model. Kept here so ``get_tools_for``
# below can filter without re-parsing descriptions at runtime. Every
# entry in GENERATOR_TOOLS should appear here; failure modes are loud
# (``assert`` below catches drift in tests).
_TOOL_MODEL_REQUIREMENTS: dict[str, frozenset[str]] = {
    # Generators: from their registry entry
    **{tool.name: tool.requires for tool in agent_tools()},
}


def get_available_generator_names(
    has_domain_model: bool = True,
    has_gui_model: bool = False,
    has_agent_model: bool = False,
    has_state_machines: bool = False,
    has_quantum_circuit: bool = False,
    has_object_model: bool = False,
    has_bpmn_model: bool = False,
    has_nn_model: bool = False,
) -> list[str]:
    """Names of the generators whose required models are present.

    Used by the orchestrator's Phase-1 selector so the LLM is only ever
    offered generators that can actually run — offering e.g.
    ``generate_web_app`` without a GUI model lets the model pick it,
    Phase 1 fails, and the run silently degrades to expensive
    from-scratch generation.
    """
    available: set[str] = set()
    if has_domain_model:
        available.add("domain")
    if has_gui_model:
        available.add("gui")
    if has_agent_model:
        available.add("agent")
    if has_state_machines:
        available.add("state_machine")
    if has_quantum_circuit:
        available.add("quantum")
    if has_object_model:
        available.add("object")
    if has_bpmn_model:
        available.add("bpmn")
    if has_nn_model:
        available.add("nn")
    return [
        tool["name"]
        for tool in GENERATOR_TOOLS
        if _TOOL_MODEL_REQUIREMENTS.get(tool["name"], frozenset()).issubset(available)
    ]


_SHELL_TOOLS = frozenset({"run_command"})

# A provider whose tool-call arguments fail to parse passes
# ``{INVALID_ARGUMENTS_KEY: "<parse error>"}`` instead of ``{}``, so the
# executor refuses the call rather than running it with no arguments.
INVALID_ARGUMENTS_KEY = "__invalid_arguments__"


def dumps_for_model(value: Any) -> str:
    """JSON text of a tool result or tool call as the model reads it.

    Non-ASCII stays literal: ``é`` is one token, ``\\u00e9`` several (a TS file
    with accented strings: 4,459 chars escaped, 2,459 literal). A lone
    surrogate, which a request body cannot carry as UTF-8, stays a ``\\uXXXX``
    escape, so ``json.loads`` still returns the same value.
    """
    text = json.dumps(value, default=str, ensure_ascii=False)
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def get_tools_for(
    has_domain_model: bool = True,
    has_gui_model: bool = False,
    has_agent_model: bool = False,
    has_state_machines: bool = False,
    has_quantum_circuit: bool = False,
    has_object_model: bool = False,
    has_bpmn_model: bool = False,
    has_nn_model: bool = False,
    allow_shell: bool = True,
) -> list[dict[str, Any]]:
    """Return the tool list scoped to which models are actually loaded.

    When a project has no domain model, every generator that takes one
    is dropped from the tool list — the LLM then defaults to
    ``write_file`` / ``run_command`` and builds from the primary model
    (state machine, agent, GUI-only, etc.) in natural code. This
    matches the ``_require_domain_model`` guards in ``ToolExecutor`` so
    the LLM never sees a tool it would immediately get an error from.

    Parameters
    ----------
    has_domain_model
        Whether a ``DomainModel`` is available.
    has_gui_model, has_agent_model, has_state_machines, has_quantum_circuit
        Whether the corresponding BUML model is populated. Currently only
        GUI affects filtering (for React/WebApp); agent / SM /
        quantum flags are accepted for forward compatibility as future
        tools key on them.
    """
    available: set[str] = set()
    if has_domain_model:
        available.add("domain")
    if has_gui_model:
        available.add("gui")
    if has_agent_model:
        available.add("agent")
    if has_state_machines:
        available.add("state_machine")
    if has_quantum_circuit:
        available.add("quantum")
    if has_object_model:
        available.add("object")
    if has_bpmn_model:
        available.add("bpmn")
    if has_nn_model:
        available.add("nn")

    def _keep(tool: dict[str, Any]) -> bool:
        # Arbitrary-shell tools are a hosted-RCE surface: drop them unless the
        # caller (a trusted local/CLI/bench run) explicitly opts in.
        if not allow_shell and tool["name"] in _SHELL_TOOLS:
            return False
        requirements = _TOOL_MODEL_REQUIREMENTS.get(tool["name"], frozenset())
        return requirements.issubset(available)

    tools = [tool for tool in get_all_tools_including_generators() if _keep(tool)]
    return [_with_network_note(tool) if tool["name"] == "run_command" else tool
            for tool in tools]


_PRIVATE_NETWORK_NOTE = (
    " The shell has a private network: localhost is this run's alone, so any port is "
    "free, and the server's own services and other runs cannot be reached from it; "
    "outbound internet (package installs) works.")
_SHARED_NETWORK_NOTE = (
    " The shell shares localhost with other runs on this server: when a port is "
    "already in use, pick another one.")


def _with_network_note(tool: dict[str, Any]) -> dict[str, Any]:
    """run_command's text states the network the sessions actually get here."""
    from besser.spec_driven_agent.execution.sandbox import shell_network_is_private

    note = _PRIVATE_NETWORK_NOTE if shell_network_is_private() else _SHARED_NETWORK_NOTE
    return {**tool, "description": tool["description"] + note}
