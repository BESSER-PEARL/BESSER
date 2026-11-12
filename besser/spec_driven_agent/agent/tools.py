"""
Tool definitions for the LLM-augmented generator.

Four categories:
1. **Generator tools** — BESSER's built-in code generators
2. **File tools** — read, write, modify, search files in the workspace
3. **Execution tools** — run commands, install deps, test code
4. **Validation tools** — check model validity and code syntax
"""

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

FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List all files in the output workspace with sizes in bytes.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": (
            "Read a file from the workspace. Every line comes back prefixed with "
            "its line number, as 'N| '. When quoting a region as modify_file's "
            "old_text, drop that prefix and quote the code itself; a quote that "
            "keeps the prefix is accepted too. "
            "Read the WHOLE function or class you intend to change in one call. "
            "A 600-line file fits in a single read; crawling it in 10-line windows "
            "spends one turn per window and still leaves you without the enclosing "
            "block. Only paginate a file too large to read at once, and then in "
            "large slices. "
            "Returns read_id for revision-bound replace_file_lines edits. "
            "offset is a 0-BASED SKIP; the numbers printed beside each line are "
            "1-based. offset=50, limit=30 displays lines 51-80, so a following "
            "replace_file_lines uses start_line=51, not 50 - or simply copy the "
            "start_line/end_line this call returns."
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
            "Create a new file, or replace an existing file's full contents. "
            "Use it for NEW files (configs, new modules, pages), and to rewrite "
            "the whole of any existing file you have just read in full - a "
            "first-class way to change a file here, not a last resort: it is "
            "the most reliable write tool. Use modify_file for a small edit to a file you would "
            "rather not reproduce in full. Do not rewrite a file you have not "
            "read this run."
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
            "Apply a targeted search-and-replace edit to an existing file. "
            "This is the tool for changing existing files: one modify_file "
            "per change site, as many as the file needs, issued together in "
            "the SAME turn. Keep old_text short - just the lines that change "
            "plus a few neighbours so it matches exactly one place; do not "
            "quote long unchanging runs. Calls on DIFFERENT files run in "
            "parallel; several calls on the SAME file are applied in the "
            "order you list them, so each one must match the file as the "
            "previous one left it. old_text must match exactly, including "
            "indentation. Completed replacement regions are skipped on replay, "
            "including insertions that retain old_text. An already_applied result "
            "requires a receipt for this exact edit and unchanged post-edit file; "
            "it confirms the edit only, not the whole requirement. A possible_replay "
            "result refuses a duplicate-looking insertion without claiming success. "
            "After a miss, use its located_range with replace_file_lines; after two refused "
            "edits, read the whole file and write_file it back. Do not keep retyping a failing old_text."
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
            "Replace a specific range of an existing file WITHOUT quoting old_text. "
            "Use this with the located_range a refused edit returns, or for a complete function/block replacement. "
            "First call read_file on the target block; copy its read_id and select the displayed "
            "1-based start_line and end_line (both inclusive). new_text is the complete literal "
            "replacement, with correct indentation and no numbered prefixes or omitted code. "
            "Use an empty new_text to delete the selected lines. For insertion, include the selected "
            "anchor line in new_text. Only the selected lines change. Unread ranges, stale reads "
            "and syntax-breaking replacements are refused. After each successful edit, read again "
            "before another range edit on the same file; do not batch same-file edits using one read_id."
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
            "node_modules, .git, venv/.venv, __pycache__, dist and build are skipped. "
            "Use it to locate code before read_file; it does not return file contents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "file_glob": {
                    "type": "string",
                    "description": "Optional filename glob, matched against the file NAME only (e.g. '*.py', '*.tsx'); a pattern containing '/' matches nothing.",
                    "default": "*",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "delete_file",
        "description": (
            "Delete a file from the workspace. Use it only for a file YOU "
            "created this run and no longer want — a scratch module, a "
            "duplicate you superseded. Never delete a generated scaffold "
            "file: the framework is fixed, and a protected run refuses it. "
            "To change a generated file, edit it with modify_file (or "
            "write_file after reading it in full). "
            "Only deletes regular files (not directories). Path must be "
            "relative to the workspace root."
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
            "test code, run linters and verify builds. The shell is one session for the "
            "whole run, like a terminal: `cd` and exported variables (`export X=1`, "
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
            "background processes are gone. A result's cwd field is the shell's directory "
            "when it is not the workspace root. Destructive or exfiltrating commands "
            "are refused with an error. stdout/stderr are truncated (~15k chars total); "
            "when cut, full_output_path names a file holding the complete log. A command "
            "whose runtime is not installed returns success=false, missing_command=true: "
            "treat its check as failed, not as passing. Several run_command calls in one turn run "
            "one after another in the order given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_dir": {
                    "type": "string",
                    "description": (
                        "Directory for this command only, relative to the workspace root; "
                        "the shell's own directory does not move. Default: the shell's "
                        "current directory (the workspace root until a command cd's elsewhere)."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Seconds before this command is killed: max 600, and never past "
                        "the run's remaining time. Raise it only for a command known to be "
                        "slow (a long build: cargo build, gradle; a first npm install)."
                    ),
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "install_dependencies",
        "description": (
            "Install dependencies for a project. Detects the package manager automatically: "
            "if requirements.txt exists, runs pip install -r requirements.txt; "
            "if package.json exists, runs npm install (both, when both exist). "
            "Returns an error when neither file is found. A custom command runs through "
            "run_command instead of auto-detection. Every install has a 300-second timeout, "
            "shares run_command's shell session, and leaves the shell's directory and "
            "environment as they were."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "working_dir": {
                    "type": "string",
                    "description": "Subdirectory containing the project (relative to workspace)",
                    "default": ".",
                },
                "command": {
                    "type": "string",
                    "description": "Custom install command",
                },
            },
        },
    },
]

# ======================================================================
# Validation tools
# ======================================================================

VALIDATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "test_api",
        "description": (
            "Exercise a real generated FastAPI workflow in an isolated copy with a fresh SQLite "
            "database. Submit up to 20 requests; state is shared within this call only. "
            "Use prior response JSON via {{n.path}}, e.g. {{0.room.id}}, in later JSON values, paths "
            "or header values. "
            "To test a protected API, register and log in within the same scenario and pass the "
            "token as a header on each later request, e.g. headers={\"Authorization\": "
            "\"Bearer {{1.access_token}}\"} when request 1 was the login (use the token field the "
            "login response really has). Allowed headers: Authorization, Accept, Content-Type, "
            "Accept-Language, If-Match, If-None-Match, Idempotency-Key and X-* API headers such as "
            "X-API-Key; cookies set by a response are kept for the rest of the scenario. "
            "Credential header values are redacted from results. "
            "Omitted expected_status means 2xx. Assert business results with expected_fields "
            "(dotted JSON path -> literal). Test happy paths AND invalid input/state transitions "
            "from the original specification. Valid scenarios are retained and rerun after code "
            "changes; failures block completion. Give each workflow a scenario_id; to correct "
            "a mistaken test payload/assertion reuse that ID with correction_reason, never "
            "weaken the specification to make broken code pass. Derive expected values from the "
            "original specification, not the current implementation. action='list' lists retained "
            "scenarios; action='get' with scenario_id reads its exact requests, expectations and "
            "last report without executing it. action='run' (default) with just scenario_id replays "
            "the saved definition. Inspect failed tests before changing either code or expectations. "
            "No shell or external URLs. "
            "Passing proves only the submitted scenario, not every requirement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["run", "list", "get"]},
                "scenario_id": {"type": "string", "description": "Stable workflow name (max 80 characters)"},
                "correction_reason": {"type": "string", "description": "Required when changing a saved scenario; explain why the previous test was wrong"},
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
            "change and before claiming completion. Fix the returned blockers, then rerun. "
            "A passing result covers these checks only, not all business requirements."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "validate_model",
        "description": "Run structural validation on the domain model. Returns errors and warnings.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_syntax",
        "description": (
            "Parse a Python file and report the first syntax error with its line. "
            "Python only; it does not import or execute the file, so missing imports and "
            "undefined names are not detected (validate_app checks those)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to Python file to check"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "task_list",
        "description": (
            "Your work checklist for this run. action='list' shows every item "
            "with its status; action='done' marks items complete — pass "
            "`ids=[1,2,3]` for SEVERAL AT ONCE, or `id` for one; "
            "action='add' appends items you discovered — pass `texts=[...]` "
            "for SEVERAL AT ONCE, or `text` for one. "
            "action='drop' closes an item the user did NOT ask for (pass `id` and "
            "`reason`: why it was not requested) — never mark such an item done. "
            "action='blocked' records required but unresolved work (pass `id` or "
            "`ids` and `reason`: what remains unresolved and why); use this when you "
            "cannot implement or verify it. "
            "action='mixed' does SEVERAL OF THE ABOVE IN ONE CALL — the turn "
            "you finish two items AND discover a new one AND drop one you didn't "
            "need should be ONE task_list call, not four. Pass any combination of "
            "`done_ids` (+ top-level `evidence`/`existing`, same as action='done'), "
            "`add_texts` (same as action='add' `texts`), `drop` (a list of "
            "{id, reason} objects), and `blocked` (a list of {id, reason} objects). "
            "The reply nests each verb's own result under `results` — check every "
            "key you used; a mixed call can succeed on one verb and be refused on "
            "another in the same response. "
            "For done without an attached verifier, supply evidence=[{id, path, quote}]: "
            "current exact quotes from your successful writes. For an already-existing "
            "implementation, read the source first and set existing=true with exact "
            "executable evidence (citing code you read, without claiming you wrote it); "
            "do not make unnecessary edits just to close a task. This records "
            "implementation only, not verified acceptance. "
            "Mark items done as you complete them — the run does not finish "
            "while items are open. Batch them: one task per call wastes a turn "
            "each, and you have a limited number of turns. Some items are "
            "checked before they are accepted; if one is refused, do the work "
            "and try again — after a few refusals it is recorded as blocked "
            "and you should move on rather than retry it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "done", "add", "drop", "blocked", "mixed"],
                },
                "existing": {"type": "boolean"},
                "id": {"type": "integer"},
                "reason": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "path": {"type": "string"},
                            "quote": {"type": "string", "maxLength": 4000},
                        },
                        "required": ["id", "path", "quote"],
                    },
                },
                "ids": {"type": "array", "items": {"type": "integer"}},
                "text": {"type": "string"},
                "texts": {"type": "array", "items": {"type": "string"}},
                "done_ids": {"type": "array", "items": {"type": "integer"}},
                "add_texts": {"type": "array", "items": {"type": "string"}},
                "drop": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}},
                        "required": ["id", "reason"],
                    },
                },
                "blocked": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}},
                        "required": ["id", "reason"],
                    },
                },
            },
            "required": ["action"],
        },
    },
]


# ======================================================================
# Model query tools — lightweight random-access into the domain model
# so the LLM doesn't have to keep the full JSON in context.
# ======================================================================

MODEL_QUERY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "query_class",
        "description": (
            "Return the full definition of a single class from the domain "
            "model: attributes, methods (with inheritance flattened), "
            "parents, and association ends. Use this when you need deeper "
            "detail on one class than the system-prompt summary provides."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Class name (case-sensitive).",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_classes_with",
        "description": (
            "List every class in the domain model matching a simple "
            "predicate. Keeps context small when the model has dozens of "
            "classes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "predicate": {
                    "type": "string",
                    "description": (
                        "One of: is_abstract | is_root (no parents) | has_constraint | "
                        "has_attribute:<name> | has_method:<name> | "
                        "extends:<parent_name>"
                    ),
                },
            },
            "required": ["predicate"],
        },
    },
    {
        "name": "get_constraints_for",
        "description": (
            "Return all constraints (OCL expressions) whose context is the "
            "given class. Use this to translate OCL constraints into "
            "runtime validators (Pydantic / Zod / serializers)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "Class name to look up constraints for.",
                },
            },
            "required": ["class_name"],
        },
    },
]


def get_all_tools_including_generators() -> list[dict[str, Any]]:
    """Return ALL tools (including generators) — for no-generator mode."""
    return (
        GENERATOR_TOOLS
        + FILE_TOOLS
        + EXECUTION_TOOLS
        + MODEL_QUERY_TOOLS
        + VALIDATION_TOOLS
    )


# Which tool names require which model. Kept here so ``get_tools_for``
# below can filter without re-parsing descriptions at runtime. Every
# entry in GENERATOR_TOOLS should appear here; failure modes are loud
# (``assert`` below catches drift in tests).
_TOOL_MODEL_REQUIREMENTS: dict[str, frozenset[str]] = {
    # Generators: from their registry entry
    **{tool.name: tool.requires for tool in agent_tools()},
    # Model-query tools need the domain model
    "query_class":              frozenset({"domain"}),
    "list_classes_with":        frozenset({"domain"}),
    "get_constraints_for":      frozenset({"domain"}),
    "validate_model":           frozenset({"domain"}),
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


_SHELL_TOOLS = frozenset({"run_command", "install_dependencies"})

# A provider whose tool-call arguments fail to parse passes
# ``{INVALID_ARGUMENTS_KEY: "<parse error>"}`` instead of ``{}``, so the
# executor refuses the call rather than running it with no arguments.
INVALID_ARGUMENTS_KEY = "__invalid_arguments__"


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
