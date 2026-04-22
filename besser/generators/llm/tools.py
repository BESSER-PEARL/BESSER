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
            "Generate a complete FastAPI backend with CRUD endpoints, SQLAlchemy ORM, "
            "and Pydantic models. This is the most complete backend generator. "
            "Output: main_api.py + sql_alchemy.py + pydantic_classes.py + requirements.txt"
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
            "Read a file from the workspace. For large files (>200 lines), use "
            "offset and limit to read specific line ranges instead of the whole file. "
            "Example: offset=50, limit=30 reads lines 50-79."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the workspace"},
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (0-indexed). Default: 0 (start of file)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of lines to read. Default: all remaining lines",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file or completely overwrite an existing file. "
            "Use for new files (Dockerfiles, configs, auth modules, README). "
            "For modifying generated code, prefer modify_file instead."
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
            "PREFERRED over write_file for modifying generated code. "
            "old_text must match exactly (including whitespace)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to find"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
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
    # Model-query tools need the domain model
    "query_class":              frozenset({"domain"}),
    "list_classes_with":        frozenset({"domain"}),
    "get_constraints_for":      frozenset({"domain"}),
    "validate_model":           frozenset({"domain"}),
}


def get_tools_for(
    has_domain_model: bool = True,
    has_gui_model: bool = False,
    has_agent_model: bool = False,
    has_state_machines: bool = False,
    has_quantum_circuit: bool = False,
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

    def _keep(tool: dict[str, Any]) -> bool:
        requirements = _TOOL_MODEL_REQUIREMENTS.get(tool["name"], frozenset())
        return requirements.issubset(available)

    return [
        tool for tool in get_all_tools_including_generators() if _keep(tool)
    ]
