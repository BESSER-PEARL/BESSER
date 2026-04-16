"""
Sandboxed tool executor for LLM-augmented generation.

Executes tool calls in an isolated workspace directory with:
- Path-traversal protection on all file operations
- Shell command execution with timeout and output capture
- Dependency installation with auto-detection
- File search (grep) across the workspace
- All BESSER generators as callable tools

Errors are returned as strings (not raised) so the LLM can handle
them intelligently — e.g., read the error, fix the code, retry.
"""

import ast
import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
from typing import Any

from besser.BUML.metamodel.structural import DomainModel

logger = logging.getLogger(__name__)

# Maximum time a shell command can run (seconds)
COMMAND_TIMEOUT = 120

# Maximum output size returned to the LLM (chars).
# Lower = less context bloat = more turns before compaction needed.
MAX_OUTPUT_SIZE = 15_000

# Maximum file content returned by read_file (chars)
MAX_FILE_READ = 20_000

# Environment variables that are safe to expose to LLM-invoked subprocesses.
# These are needed for basic tooling to work (PATH for binaries, HOME for
# tool caches, LANG/LC_* for locale-aware tools, TMPDIR for scratch space,
# SystemRoot/USERPROFILE on Windows). Everything else — including provider
# API keys, deployment credentials, SMTP passwords, OAuth secrets — is
# stripped so the LLM cannot `printenv` them into generated code.
_SAFE_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "TMP", "TEMP",
    # Windows
    "SystemRoot", "SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "COMSPEC", "PATHEXT", "ProgramFiles", "ProgramData",
    # Python (harmless, often needed by tools)
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
    # Node (harmless, often needed by tools)
    "NODE_PATH",
})

# Variable-name substrings that identify secret-like env vars. Even if
# a variable is accidentally in the allowlist, names matching these
# patterns are always stripped.
_SECRET_SUBSTRINGS: tuple[str, ...] = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "PRIVATE",
    "API_", "AUTH", "CERT", "SESSION",
)


def _normalize_path_for_comparison(path: str) -> str:
    """Strip the Windows ``\\\\?\\`` extended-path prefix.

    On Windows, ``os.path.realpath`` sometimes returns paths with the
    ``\\\\?\\`` (or ``\\\\?\\UNC\\``) extended-length prefix, especially
    when long-path support is enabled at the OS level. This prefix can
    appear on child paths but NOT on the workspace root (or vice versa)
    depending on whether the path exists at resolution time. When that
    happens, a naive ``full.startswith(workspace)`` check fails for
    legitimate paths inside the workspace and the caller sees a bogus
    "Path traversal blocked" error — which is how the Rails output
    ended up missing half its subdirectory files on Windows.

    We strip the prefix from BOTH sides before comparison so the
    containment check is semantically correct regardless of whether
    either path has been normalized to extended form.

    No-op on Unix where the prefix doesn't exist.
    """
    if path.startswith("\\\\?\\UNC\\"):
        # Extended form for UNC paths: \\?\UNC\server\share → \\server\share
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        # Extended form for local paths: \\?\C:\foo → C:\foo
        return path[4:]
    return path


def _safe_subprocess_env() -> dict[str, str]:
    """Return a minimal subprocess environment with secrets stripped.

    Never pass the full ``os.environ`` to an LLM-invoked subprocess —
    that would leak provider API keys, OAuth secrets, SMTP credentials,
    and any other server-side configuration. This helper constructs a
    new environment by copying only allowlisted variables and
    deliberately drops anything whose name contains a secret-like
    substring, even if it's in the allowlist.
    """
    safe: dict[str, str] = {}
    for name, value in os.environ.items():
        if name not in _SAFE_ENV_ALLOWLIST:
            continue
        upper = name.upper()
        if any(substr in upper for substr in _SECRET_SUBSTRINGS):
            continue
        safe[name] = value
    # Ensure PATH exists even if the parent somehow didn't have it.
    safe.setdefault("PATH", os.defpath)
    # Suppress .pyc writes — same behaviour as the old `env={**os.environ,...}`.
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe


class ToolExecutor:
    """
    Executes LLM tool calls in a sandboxed workspace.

    All file paths are resolved relative to the workspace and validated
    against path traversal.  Shell commands run with a timeout and their
    working directory locked to the workspace.

    Args:
        workspace: Absolute path to the output directory.
        domain_model: The BUML domain model.
        gui_model: Optional GUI model.
        agent_model: Optional agent model.
        agent_config: Optional agent config dict.
    """

    def __init__(
        self,
        workspace: str,
        domain_model: DomainModel,
        gui_model: Any = None,
        agent_model: Any = None,
        agent_config: dict | None = None,
    ):
        self.workspace = _normalize_path_for_comparison(os.path.realpath(workspace))
        self.domain_model = domain_model
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
        # Track files created by generators — used to warn if LLM overwrites them
        self._generator_files: set[str] = set()

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        Execute a tool call.  Returns a JSON string.

        Errors are returned as ``{"error": "..."}`` — never raised.
        """
        handler = self._handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(self, arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e, exc_info=True)
            return json.dumps({"error": f"{tool_name} failed: {e}"})

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    def _safe_path(self, rel_path: str) -> str:
        """Resolve a relative path within the workspace.  Blocks traversal."""
        full = os.path.realpath(os.path.join(self.workspace, rel_path))
        # Strip Windows extended-path prefix from BOTH sides — otherwise
        # a workspace stored as ``C:\...`` and a resolved full path as
        # ``\\?\C:\...`` (or vice versa) fails the containment check
        # for legitimate in-workspace paths. See
        # ``_normalize_path_for_comparison`` for the full rationale.
        full_cmp = _normalize_path_for_comparison(full)
        full_norm = os.path.normcase(full_cmp)
        ws_norm = os.path.normcase(self.workspace)
        if not (
            full_norm == ws_norm
            or full_norm.startswith(ws_norm + os.sep)
            or full_norm.startswith(ws_norm + "/")
        ):
            logger.error(
                "Path check failed: rel=%s full=%s workspace=%s full_norm=%s ws_norm=%s",
                rel_path, full, self.workspace, full_norm, ws_norm,
            )
            raise ValueError(f"Path traversal blocked: {rel_path}")
        return full

    def _safe_cwd(self, rel_dir: str = ".") -> str:
        """Resolve a working directory within the workspace."""
        cwd = os.path.realpath(os.path.join(self.workspace, rel_dir))
        cwd_cmp = _normalize_path_for_comparison(cwd)
        cwd_norm = os.path.normcase(cwd_cmp)
        ws_norm = os.path.normcase(self.workspace)
        if not (
            cwd_norm == ws_norm
            or cwd_norm.startswith(ws_norm + os.sep)
            or cwd_norm.startswith(ws_norm + "/")
        ):
            raise ValueError(f"Path traversal blocked: {rel_dir}")
        if not os.path.isdir(cwd):
            os.makedirs(cwd, exist_ok=True)
        return cwd

    def _list_dir(self, directory: str) -> list[dict]:
        """List files recursively, relative to workspace."""
        files = []
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                abs_path = os.path.join(root, f)
                rel = os.path.relpath(abs_path, self.workspace)
                size = os.path.getsize(abs_path)
                files.append({"path": rel.replace("\\", "/"), "size": size})
        return sorted(files, key=lambda x: x["path"])

    def _gen_dir(self, name: str) -> str:
        """Create and return a generator output subdirectory."""
        d = os.path.join(self.workspace, name)
        os.makedirs(d, exist_ok=True)
        return d

    def _track_generated_files(self, directory: str) -> list[dict]:
        """List files in a generator output dir and mark them as generator-created."""
        files = self._list_dir(directory)
        for f in files:
            self._generator_files.add(f["path"])
        return files

    @staticmethod
    def _truncate(text: str, limit: int = MAX_OUTPUT_SIZE, keep_tail: bool = False) -> str:
        """
        Smart truncation.

        For error output (tracebacks), keeps the tail since that's where
        the actual error message is.  For normal output, keeps the head.

        Args:
            text: Text to truncate.
            limit: Maximum characters.
            keep_tail: If True, preserve the end (for error output).
        """
        if len(text) <= limit:
            return text
        if keep_tail:
            # Keep first 20% + last 60% — the error is usually at the end
            head_size = int(limit * 0.2)
            tail_size = int(limit * 0.6)
            return (
                text[:head_size]
                + f"\n\n... [{len(text) - head_size - tail_size} chars truncated] ...\n\n"
                + text[-tail_size:]
            )
        return text[:limit] + f"\n\n... [truncated, {len(text)} chars total]"

    # ------------------------------------------------------------------
    # Generator tools
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Modification guides — tell the LLM HOW to modify each generator's output
    # ------------------------------------------------------------------

    _MODIFICATION_GUIDES: dict[str, str] = {
        "generate_pydantic": (
            "To add validators: use modify_file to add @field_validator methods inside the class. "
            "To add new fields: find the class definition and insert a new field line."
        ),
        "generate_sqlalchemy": (
            "To change the database: modify the create_engine() call. "
            "To add indexes: find the Column definition and add index=True. "
            "To add constraints: add __table_args__ to the class."
        ),
        "generate_fastapi_backend": (
            "This generated 3 files: main_api.py (endpoints), sql_alchemy.py (ORM), pydantic_classes.py (schemas). "
            "To add auth: create a NEW auth.py file, then use modify_file to add Depends(get_current_user) to endpoints in main_api.py. "
            "To add pagination: use modify_file to add skip/limit parameters to GET list endpoints. "
            "To add a new endpoint: find the last @app route in main_api.py and insert after it. "
            "NEVER rewrite these files — use modify_file for surgical edits."
        ),
        "generate_django": (
            "To add DRF: create a NEW serializers.py and viewsets.py, then modify urls.py to add router. "
            "To customize admin: use modify_file on admin.py to add list_display etc."
        ),
        "generate_react": (
            "To add theming: modify App.tsx to wrap with ThemeProvider. "
            "To add new pages: create new component files and modify the router."
        ),
        "generate_web_app": (
            "This generated frontend/ + backend/ + docker-compose.yml. "
            "Modify each subdirectory's files separately. Never rewrite generated files."
        ),
    }

    def _gen_pydantic(self, args: dict) -> dict:
        from besser.generators.pydantic_classes import PydanticGenerator
        out = self._gen_dir("pydantic")
        PydanticGenerator(
            model=self.domain_model, output_dir=out,
            backend=args.get("backend", False),
            nested_creations=args.get("nested_creations", False),
        ).generate()
        return {
            "status": "ok",
            "files": self._track_generated_files(out),
            "guide": self._MODIFICATION_GUIDES.get("generate_pydantic", ""),
        }

    def _gen_sqlalchemy(self, args: dict) -> dict:
        from besser.generators.sql_alchemy import SQLAlchemyGenerator
        out = self._gen_dir("sqlalchemy")
        SQLAlchemyGenerator(model=self.domain_model, output_dir=out).generate(
            dbms=args.get("dbms", "sqlite")
        )
        return {
            "status": "ok",
            "files": self._track_generated_files(out),
            "guide": self._MODIFICATION_GUIDES.get("generate_sqlalchemy", ""),
        }

    def _gen_fastapi_backend(self, args: dict) -> dict:
        from besser.generators.backend import BackendGenerator
        out = self._gen_dir("backend")
        BackendGenerator(
            model=self.domain_model, output_dir=out,
            http_methods=args.get("http_methods", ["GET", "POST", "PUT", "DELETE"]),
        ).generate()
        return {
            "status": "ok",
            "files": self._track_generated_files(out),
            "guide": self._MODIFICATION_GUIDES.get("generate_fastapi_backend", ""),
        }

    def _gen_django(self, args: dict) -> dict:
        from besser.generators.django import DjangoGenerator
        out = self._gen_dir("django")
        DjangoGenerator(
            model=self.domain_model,
            project_name=args.get("project_name", "myproject"),
            app_name=args.get("app_name", "myapp"),
            gui_model=self.gui_model, output_dir=out,
        ).generate()
        return {
            "status": "ok",
            "files": self._track_generated_files(out),
            "guide": self._MODIFICATION_GUIDES.get("generate_django", ""),
        }

    def _gen_python_classes(self, args: dict) -> dict:
        from besser.generators.python_classes import PythonGenerator
        out = self._gen_dir("python")
        PythonGenerator(model=self.domain_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_java_classes(self, args: dict) -> dict:
        from besser.generators.java_classes import JavaGenerator
        out = self._gen_dir("java")
        JavaGenerator(model=self.domain_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_sql(self, args: dict) -> dict:
        from besser.generators.sql import SQLGenerator
        out = self._gen_dir("sql")
        SQLGenerator(model=self.domain_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_json_schema(self, args: dict) -> dict:
        from besser.generators.json import JSONSchemaGenerator
        out = self._gen_dir("jsonschema")
        JSONSchemaGenerator(
            model=self.domain_model, output_dir=out,
            mode=args.get("mode", "regular"),
        ).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_rest_api(self, args: dict) -> dict:
        from besser.generators.rest_api import RESTAPIGenerator
        out = self._gen_dir("rest_api")
        RESTAPIGenerator(
            model=self.domain_model, output_dir=out,
            http_methods=args.get("http_methods", ["GET", "POST", "PUT", "DELETE"]),
        ).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_react(self, args: dict) -> dict:
        if not self.gui_model:
            return {"error": "No GUI model available. The React generator requires a GUI model."}
        from besser.generators.react import ReactGenerator
        out = self._gen_dir("react")
        ReactGenerator(model=self.domain_model, gui_model=self.gui_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_flutter(self, args: dict) -> dict:
        if not self.gui_model:
            return {"error": "No GUI model available. The Flutter generator requires a GUI model."}
        from besser.generators.flutter import FlutterGenerator
        out = self._gen_dir("flutter")
        FlutterGenerator(model=self.domain_model, gui_model=self.gui_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_web_app(self, args: dict) -> dict:
        if not self.gui_model:
            return {"error": "No GUI model available. The WebApp generator requires a GUI model."}
        from besser.generators.web_app import WebAppGenerator
        out = self._gen_dir("web_app")
        WebAppGenerator(
            model=self.domain_model, gui_model=self.gui_model, output_dir=out,
            agent_model=self.agent_model, agent_config=self.agent_config,
        ).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_rdf(self, args: dict) -> dict:
        from besser.generators.rdf import RDFGenerator
        out = self._gen_dir("rdf")
        RDFGenerator(model=self.domain_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    # ------------------------------------------------------------------
    # File tools
    # ------------------------------------------------------------------

    def _list_files(self, args: dict) -> dict:
        files = self._list_dir(self.workspace)
        # Exclude internal files
        files = [f for f in files if not f["path"].startswith(".besser_")]
        return {"files": files, "total": len(files)}

    def _read_file(self, args: dict) -> dict:
        """
        Read a file with optional line-based pagination.

        Supports offset (start line, 0-indexed) and limit (number of lines)
        for reading specific sections of large files without dumping
        everything into context.
        """
        path = self._safe_path(args["path"])
        if not os.path.isfile(path):
            return {"error": f"File not found: {args['path']}"}
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            size = os.path.getsize(path)
            return {"error": f"Binary file ({size} bytes), cannot read as text: {args['path']}"}

        lines = content.split("\n")
        total_lines = len(lines)
        offset = args.get("offset", 0) or 0
        limit = args.get("limit")

        # Apply line-based slicing if offset or limit specified
        start = min(offset, total_lines)
        end = total_lines
        if limit is not None:
            end = min(start + limit, total_lines)

        selected = "\n".join(lines[start:end])

        # Truncate if still too large
        if len(selected) > MAX_FILE_READ:
            selected = self._truncate(selected, MAX_FILE_READ)

        result = {"content": selected}

        # Include metadata so the LLM knows about pagination
        if offset > 0 or limit is not None:
            result["start_line"] = start + 1  # 1-indexed for display
            result["end_line"] = end
            result["total_lines"] = total_lines
            result["lines_read"] = end - start
        elif total_lines > 200:
            # Hint for large files: tell the LLM it can paginate
            result["total_lines"] = total_lines
            result["hint"] = "Large file. Use offset/limit to read specific sections."

        return result

    def _write_file(self, args: dict) -> dict:
        rel_path = args["path"].replace("\\", "/")
        path = self._safe_path(rel_path)

        # Guardrail for small generated files: suggest modify_file instead.
        # For large files (>200 lines), allow rewriting — it's more efficient
        # than 20 modify_file calls on a 5000-line file.
        if rel_path in self._generator_files and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                existing_lines = f.read().count("\n") + 1
            if existing_lines <= 200:
                return {
                    "error": (
                        f"'{rel_path}' was created by a BESSER generator ({existing_lines} lines). "
                        "Use modify_file for targeted edits on small generated files."
                    ),
                }
            # Large file — allow rewrite but log it
            logger.info("Allowing rewrite of large generated file: %s (%d lines)", rel_path, existing_lines)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return {"status": "written", "path": args["path"], "size": len(args["content"])}

    def _modify_file(self, args: dict) -> dict:
        path = self._safe_path(args["path"])
        if not os.path.isfile(path):
            return {"error": f"File not found: {args['path']}"}
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        old_text = args["old_text"]
        if old_text not in content:
            return {
                "error": f"old_text not found in {args['path']}. "
                         f"File has {content.count(chr(10))+1} lines, {len(content)} chars. "
                         f"Make sure old_text matches exactly including whitespace/indentation.",
            }
        new_content = content.replace(old_text, args["new_text"], 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"status": "modified", "path": args["path"]}

    def _search_in_files(self, args: dict) -> dict:
        """Search for a pattern across workspace files."""
        pattern = args["pattern"]
        file_glob = args.get("file_glob", "*")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Fall back to literal search
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        matches = []
        for root, _, filenames in os.walk(self.workspace):
            for fname in filenames:
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, self.workspace).replace("\\", "/")
                # Skip binary files and internal files
                if rel_path.startswith(".besser_"):
                    continue
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append({
                                    "file": rel_path,
                                    "line": i,
                                    "text": line.rstrip()[:200],
                                })
                                if len(matches) >= 50:
                                    return {"matches": matches, "truncated": True}
                except (OSError, UnicodeDecodeError):
                    continue

        return {"matches": matches, "total": len(matches)}

    # ------------------------------------------------------------------
    # Execution tools — the generate-test-fix loop enabler
    # ------------------------------------------------------------------

    def _run_command(self, args: dict) -> dict:
        """
        Run a shell command in the workspace.

        Security:
        - Working directory locked to workspace (or subdirectory)
        - Timeout enforced
        - Output truncated to prevent context blow-up
        """
        command = args["command"]
        working_dir = self._safe_cwd(args.get("working_dir", "."))

        logger.info("Running command: %s (in %s)", command, working_dir)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
                env=_safe_subprocess_env(),
            )

            stdout = self._truncate(result.stdout, MAX_OUTPUT_SIZE // 2)
            # For stderr (errors), keep the tail where the actual error message is
            stderr = self._truncate(result.stderr, MAX_OUTPUT_SIZE // 2, keep_tail=True)

            return {
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out after {COMMAND_TIMEOUT} seconds",
                "command": command,
            }
        except Exception as e:
            return {"error": f"Failed to run command: {e}"}

    def _install_dependencies(self, args: dict) -> dict:
        """
        Install project dependencies with auto-detection.

        Checks for requirements.txt (pip) or package.json (npm) and runs
        the appropriate install command.
        """
        working_dir = self._safe_cwd(args.get("working_dir", "."))
        custom_command = args.get("command")

        if custom_command:
            return self._run_command({"command": custom_command, "working_dir": args.get("working_dir", ".")})

        # Auto-detect
        requirements_txt = os.path.join(working_dir, "requirements.txt")
        package_json = os.path.join(working_dir, "package.json")

        results = []

        if os.path.isfile(requirements_txt):
            pip_cmd = f"{sys.executable} -m pip install -r requirements.txt --quiet"
            pip_result = self._run_command({"command": pip_cmd, "working_dir": args.get("working_dir", ".")})
            results.append({"type": "pip", "result": pip_result})

        if os.path.isfile(package_json):
            npm_result = self._run_command({"command": "npm install --quiet", "working_dir": args.get("working_dir", ".")})
            results.append({"type": "npm", "result": npm_result})

        if not results:
            return {"error": "No requirements.txt or package.json found. Specify a custom install command."}

        return {"installs": results}

    # ------------------------------------------------------------------
    # Validation tools
    # ------------------------------------------------------------------

    def _validate_model(self, args: dict) -> dict:
        try:
            result = self.domain_model.validate()
            return {"validation": result}
        except Exception as e:
            return {"error": f"Validation failed: {e}"}

    def _check_syntax(self, args: dict) -> dict:
        path = self._safe_path(args["path"])
        if not os.path.isfile(path):
            return {"error": f"File not found: {args['path']}"}
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source, filename=args["path"])
            return {"status": "ok", "message": "Syntax is valid"}
        except SyntaxError as e:
            return {"error": f"Syntax error at line {e.lineno}: {e.msg}"}

    # ------------------------------------------------------------------
    # Handler dispatch table
    # ------------------------------------------------------------------

    _handlers: dict[str, Any] = {
        # Generators
        "generate_pydantic": _gen_pydantic,
        "generate_sqlalchemy": _gen_sqlalchemy,
        "generate_fastapi_backend": _gen_fastapi_backend,
        "generate_django": _gen_django,
        "generate_python_classes": _gen_python_classes,
        "generate_java_classes": _gen_java_classes,
        "generate_sql": _gen_sql,
        "generate_json_schema": _gen_json_schema,
        "generate_rest_api": _gen_rest_api,
        "generate_react": _gen_react,
        "generate_flutter": _gen_flutter,
        "generate_web_app": _gen_web_app,
        "generate_rdf": _gen_rdf,
        # Files
        "list_files": _list_files,
        "read_file": _read_file,
        "write_file": _write_file,
        "modify_file": _modify_file,
        "search_in_files": _search_in_files,
        # Execution
        "run_command": _run_command,
        "install_dependencies": _install_dependencies,
        # Validation
        "validate_model": _validate_model,
        "check_syntax": _check_syntax,
    }
