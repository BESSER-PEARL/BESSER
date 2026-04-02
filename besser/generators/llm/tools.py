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
            "Read the contents of a file from the workspace. "
            "Always read generated code before modifying it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the workspace"},
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


def get_all_tools() -> list[dict[str, Any]]:
    """Return all tool definitions combined."""
    return GENERATOR_TOOLS + FILE_TOOLS + EXECUTION_TOOLS + VALIDATION_TOOLS
