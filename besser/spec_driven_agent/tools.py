"""
Tool definitions for the LLM-augmented generator.

Four categories:
1. **Generator tools** — BESSER's built-in code generators
2. **File tools** — read, write, modify, search files in the workspace
3. **Execution tools** — run commands, install deps, test code
4. **Validation tools** — check model validity and code syntax
"""

from typing import Any


# ======================================================================
# Generator tools
# ======================================================================

GENERATOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "generate_pydantic",
        "description": (
            "Generate Pydantic BaseModel classes from the domain model. "
            "Produces validated data models with type hints and field validators. "
            "Output: pydantic_classes.py"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "backend": {
                    "type": "boolean",
                    "description": "Generate 'Create' suffixed models for API request bodies",
                    "default": False,
                },
                "nested_creations": {
                    "type": "boolean",
                    "description": "Allow nested entity creation in request bodies",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "generate_sqlalchemy",
        "description": (
            "Generate SQLAlchemy ORM models from the domain model. "
            "Produces table definitions with relationships, foreign keys, and inheritance. "
            "Output: sql_alchemy.py"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dbms": {
                    "type": "string",
                    "enum": ["sqlite", "postgresql", "mysql", "mssql", "mariadb", "oracle"],
                    "description": "Target database system",
                    "default": "sqlite",
                },
            },
        },
    },
    {
        "name": "generate_fastapi_backend",
        "description": (
            "Generate a complete, MODULAR FastAPI backend with CRUD endpoints, SQLAlchemy ORM, "
            "and Pydantic models. This is the most complete backend generator. "
            "Output: main_api.py (slim app) + database.py + routers/<Class>.py (endpoints per "
            "class) + sql_alchemy.py + pydantic_classes.py + bal_stdlib.py + requirements.txt"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "http_methods": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                    "description": "HTTP methods to generate endpoints for",
                    "default": ["GET", "POST", "PUT", "DELETE"],
                },
            },
        },
    },
    {
        "name": "generate_django",
        "description": (
            "Generate a full Django project with models, views, templates, URL routing, "
            "and admin panel. Output: full Django project directory"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "default": "myproject"},
                "app_name": {"type": "string", "default": "myapp"},
            },
        },
    },
    {
        "name": "generate_python_classes",
        "description": (
            "Generate plain Python classes with __init__, getters/setters, relationships. "
            "Output: classes.py"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_java_classes",
        "description": "Generate Java classes from the domain model. Output: .java files zipped.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_sql",
        "description": "Generate raw SQL CREATE TABLE statements. Output: tables.sql",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_json_schema",
        "description": "Generate JSON Schema from the domain model. Output: json_schema.json",
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["regular", "smart_data"],
                    "default": "regular",
                },
            },
        },
    },
    {
        "name": "generate_rest_api",
        "description": (
            "Generate FastAPI REST endpoints + Pydantic models. "
            "Lighter than generate_fastapi_backend (no ORM). "
            "Output: rest_api.py + pydantic_classes.py"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "http_methods": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                    "default": ["GET", "POST", "PUT", "DELETE"],
                },
            },
        },
    },
    {
        "name": "generate_react",
        "description": (
            "Generate a React TypeScript frontend from the domain model + GUI model. "
            "REQUIRES a GUI model — fails without one. Output: full React app directory"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_flutter",
        "description": (
            "Generate a Flutter mobile app from the domain model + GUI model. "
            "REQUIRES a GUI model. Output: full Flutter app directory"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_web_app",
        "description": (
            "Generate a complete full-stack web app: React frontend + FastAPI backend "
            "+ Docker deployment. REQUIRES a GUI model. "
            "Output: frontend/ + backend/ + docker-compose.yml"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_rdf",
        "description": "Generate RDF/OWL vocabulary (Turtle format). Output: vocabulary.ttl",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_qiskit",
        "description": (
            "Generate a Qiskit Python script from the Quantum Circuit model. "
            "REQUIRES a quantum circuit. Output: qiskit_circuit.py"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "backend_type": {
                    "type": "string",
                    "description": "Qiskit backend (default: aer_simulator)",
                },
                "shots": {
                    "type": "integer",
                    "description": "Number of shots for the simulation (default: 1024)",
                },
            },
        },
    },
    {
        "name": "generate_supabase",
        "description": (
            "Generate Supabase-flavoured PostgreSQL DDL from the domain model — "
            "UUID primary keys, auth.users mirroring, and Row-Level-Security (RLS) "
            "policies. Output: a .sql file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_root": {
                    "type": "string",
                    "description": "Class mapped to Supabase auth users (default: User)",
                },
            },
        },
    },
    {
        "name": "generate_json_object",
        "description": (
            "Generate a JSON document of the object (instance) model — the concrete "
            "instances and their attribute/relationship values (useful as seed data / "
            "fixtures). REQUIRES an object model. Output: <model>.json"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_baf",
        "description": (
            "Generate a BESSER Agent Framework (BAF) chatbot/agent project from the "
            "agent model. REQUIRES an agent model. Output: a Python BAF agent project."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_bpmn",
        "description": (
            "Generate vendor-neutral BPMN 2.0 XML from the BPMN process model. "
            "REQUIRES a BPMN model. Output: bpmn_diagram.bpmn"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_pytorch",
        "description": (
            "Generate PyTorch neural-network training/eval code from the NN model. "
            "REQUIRES a neural-network model (and PyTorch installed). Output: pytorch_nn.py"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "generation_type": {
                    "type": "string",
                    "enum": ["subclassing", "sequential"],
                    "description": "PyTorch code style (default: subclassing)",
                },
            },
        },
    },
    {
        "name": "generate_tensorflow",
        "description": (
            "Generate TensorFlow/Keras neural-network training/eval code from the NN "
            "model. REQUIRES a neural-network model (and TensorFlow installed). "
            "Output: tf_nn.py"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "generation_type": {
                    "type": "string",
                    "enum": ["subclassing", "sequential"],
                    "description": "TensorFlow code style (default: subclassing)",
                },
            },
        },
    },
]

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
                    "description": "Lines to SKIP before the first displayed line (0-based). "
                                   "offset=50 makes line 51 the first one shown. Default: 0",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read. Default: all remaining lines, which is "
                                   "usually what you want. Prefer the whole enclosing block over "
                                   "a narrow window",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file, or replace an existing file's full contents. "
            "Use it for NEW files (configs, new modules, pages), and for any "
            "existing file you have just read in full: rewrite the whole file "
            "when you have just read it in full. That is a first-class way to "
            "change a file here, not a last resort - it is the most reliable "
            "write tool. Use modify_file for a small edit to a file you would "
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
            "After matching failures, read the target block and use replace_file_lines; "
            "do not keep retyping a failing old_text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to find"},
                "new_text": {"type": "string", "description": "Replacement text"},
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Replace every exact occurrence. Leave false for normal edits; "
                        "use true only when all matches intentionally need the same change."
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
            "Use this after modify_file matching failures or for a complete function/block replacement. "
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
                "read_id": {"type": "string", "description": "Exact read_id returned by read_file for this file"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "new_text": {"type": "string", "description": "Complete replacement for the selected inclusive lines"},
            },
            "required": ["path", "read_id", "start_line", "end_line", "new_text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_in_files",
        "description": (
            "Search for a text pattern across all files in the workspace. "
            "Returns matching lines with file paths and line numbers. "
            "Useful for finding where to make modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                "file_glob": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g. '*.py', '**/*.ts')",
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
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to delete",
                },
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
            "Run a shell command in the workspace directory and return stdout + stderr. "
            "Use this to: test code, run linters, check imports, verify builds. "
            "Commands run with a 120-second timeout. "
            "Examples: 'python main_api.py --help', 'npm run build', 'python -m pytest'. "
            "The working directory is the workspace root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Subdirectory to run in (relative to workspace). Default: workspace root.",
                    "default": ".",
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
            "if package.json exists, runs npm install. "
            "Or specify a custom command."
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
                    "description": "Custom install command (overrides auto-detection)",
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
            "Use prior response JSON via {{0.room.id}} in later JSON values or paths. "
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
            "No shell, credentials, or external URLs. "
            "Passing proves only the submitted scenario, not every requirement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["run", "list", "get"], "description": "Default run; list/get inspect saved workflow definitions without execution"},
                "scenario_id": {"type": "string", "description": "Stable workflow name (max 80 characters); reuse to correct the same test"},
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
            "Check if a Python file has valid syntax. "
            "Use after modifying Python files to catch errors early."
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
            "`reason`) — never mark such an item done. "
            "action='blocked' records required but unresolved work (pass `id` or "
            "`ids` and `reason`); use this when you cannot implement or verify it. "
            "action='mixed' does SEVERAL OF THE ABOVE IN ONE CALL — the turn "
            "you finish two items AND discover a new one AND drop one you didn't "
            "need should be ONE task_list call, not four. Pass any combination of "
            "`done_ids` (+ `evidence`/`existing`, same as action='done'), "
            "`add_texts` (same as action='add' `texts`), `drop` (a list of "
            "{id, reason} objects), and `blocked` (a list of {id, reason} objects). "
            "The reply nests each verb's own result under `results` — check every "
            "key you used; a mixed call can succeed on one verb and be refused on "
            "another in the same response. "
            "For done without an attached verifier, supply evidence=[{id, path, quote}] "
            "from your successful writes. For an already-existing implementation, read the "
            "source first and set existing=true with exact executable evidence; do not make "
            "unnecessary edits just to close a task. This records implementation only, not verified acceptance. "
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
                    "description": "list | done | add | drop | blocked | mixed",
                },
                "existing": {"type": "boolean", "description": "For done: cite an already-existing implementation that you have read, without claiming you wrote it. Acceptance remains unverified."},
                "id": {
                    "type": "integer",
                    "description": (
                        "A single item id to mark done. Prefer `ids` when you "
                        "have finished more than one."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "For drop: why not requested. For blocked: what remains unresolved and why.",
                },
                "evidence": {
                    "type": "array",
                    "maxItems": 20,
                    "description": "For done without a verifier: current exact quotes from files you successfully changed; not acceptance proof.",
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
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Item ids to mark done in ONE call, e.g. [1,2,3]. "
                        "Use this instead of one call per item."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": (
                        "New item text for action='add'. Prefer `texts` when "
                        "you have more than one."
                    ),
                },
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Several new items to append in ONE call. Use this "
                        "instead of one call per item."
                    ),
                },
                "done_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "For action='mixed': item ids to mark done, together "
                        "with top-level `evidence` (and optional `existing`) "
                        "exactly as action='done' uses them."
                    ),
                },
                "add_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For action='mixed': new items to append, same as action='add' `texts`.",
                },
                "drop": {
                    "type": "array",
                    "description": "For action='mixed': items to drop, each with its own reason.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "reason": {"type": "string", "description": "Why the user did not ask for it."},
                        },
                        "required": ["id", "reason"],
                    },
                },
                "blocked": {
                    "type": "array",
                    "description": "For action='mixed': items to record as blocked, each with its own reason.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "reason": {"type": "string", "description": "What remains unresolved and why."},
                        },
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
            "classes. Supported predicates: 'is_abstract', 'is_root' "
            "(no parents), 'has_constraint', 'has_attribute:<name>', "
            "'has_method:<name>', 'extends:<parent_name>'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "predicate": {
                    "type": "string",
                    "description": (
                        "One of: is_abstract | is_root | has_constraint | "
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


def get_all_tools() -> list[dict[str, Any]]:
    """Return tools available to the LLM in Phase 2.

    Generator tools are NOT included — the orchestrator calls them
    directly in Phase 1.  The LLM only gets file, execution, model-query,
    and validation tools for customizing the generated output.
    """
    return FILE_TOOLS + EXECUTION_TOOLS + MODEL_QUERY_TOOLS + VALIDATION_TOOLS


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
    # Generators that need only the domain model
    "generate_pydantic":        frozenset({"domain"}),
    "generate_sqlalchemy":      frozenset({"domain"}),
    "generate_fastapi_backend": frozenset({"domain"}),
    "generate_django":          frozenset({"domain"}),
    "generate_python_classes":  frozenset({"domain"}),
    "generate_java_classes":    frozenset({"domain"}),
    "generate_sql":             frozenset({"domain"}),
    "generate_json_schema":     frozenset({"domain"}),
    "generate_rest_api":        frozenset({"domain"}),
    "generate_rdf":             frozenset({"domain"}),
    # Generators that need domain + GUI
    "generate_react":           frozenset({"domain", "gui"}),
    "generate_flutter":         frozenset({"domain", "gui"}),
    "generate_web_app":         frozenset({"domain", "gui"}),
    # Supabase DDL is domain-only, like the other database generators
    "generate_supabase":        frozenset({"domain"}),
    # Generators driven by other models
    "generate_qiskit":          frozenset({"quantum"}),
    "generate_json_object":     frozenset({"object"}),
    "generate_baf":             frozenset({"agent"}),
    "generate_bpmn":            frozenset({"bpmn"}),
    "generate_pytorch":         frozenset({"nn"}),
    "generate_tensorflow":      frozenset({"nn"}),
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
        GUI affects filtering (for React/Flutter/WebApp); agent / SM /
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

    return [
        tool for tool in get_all_tools_including_generators() if _keep(tool)
    ]
