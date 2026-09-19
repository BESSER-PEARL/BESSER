"""
LLM generation orchestrator -- three-phase architecture.

Phase 1 (deterministic, no LLM):
  - Select the best generator based on available models
  - Run the generator
  - Inventory the output (what files, what they contain)
  - Analyze what the user asked for vs what was generated (gap analysis)

Phase 2 (LLM, scoped tasks):
  - Give the LLM a focused task list based on the gap analysis
  - The LLM only writes what's missing (auth, config, Docker, README)
  - Parallel tool execution when multiple independent calls are made

Phase 3 (validation & fix):
  - Validate source, startup, data entry, workflows and requirement evidence
  - Repair and recheck within the remaining turn/cost/runtime budgets
  - Preserve unresolved findings as incomplete, never verified completion
"""

import ast as _ast
import hashlib
import json
import logging
import os
import re as _re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from besser.generators.llm.compaction import (
    COMPACT_RESERVE_TOKENS,
    # Re-exported on purpose: callers and tests import the threshold from
    # here as well as from compaction, and assert the two agree so the
    # orchestrator's budget cannot drift from the compactor's.
    COMPACT_TOKEN_THRESHOLD,  # noqa: F401
    _estimate_tokens,
    effective_threshold,
    maybe_compact,
    _summarize_messages,
)
from besser.generators.llm.user_request import user_request
from besser.generators.llm.specification import validate_specification
from besser.generators.llm.model_serializer import serialize_domain_model
from besser.generators.llm.history_eviction import evict_stale_file_bodies, without_rejected_edit_drafts
from besser.generators.llm.validation.frontend_contract import (
    _method_button_source_issues as _method_button_source_issues,
    collect_frontend_contract_issues,
)
from besser.generators.llm import requirements_ledger as _requirements_ledger
from besser.generators.llm.scaffold_repair import (
    _DEFAULT_BACKEND_REQUIREMENTS as _DEFAULT_BACKEND_REQUIREMENTS,
    _IMPORT_TO_REQUIREMENT as _IMPORT_TO_REQUIREMENT,
    _ensure_requirements_txt,
    _is_dockerfile,
    _project_has_npm_lockfile,
    _strip_missing_lockfile_copy,
)
from besser.generators.llm.checkpoint import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SCHEMA_VERSION,
    _SNAPSHOT_DIR,
    Checkpoint,
    compute_fingerprint,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from besser.generators.llm.errors import (
    CheckpointMismatchError,
    EmptyInstructionsError,
    InvalidApiKeyError,
)
from besser.generators.llm.gap_analyzer import analyze_gaps_via_llm
from besser.generators.llm.action_inventory import (
    collect_action_endpoints, format_action_inventory,
    action_gap_tasks, action_implementation_issues, merge_action_tasks,
)
from besser.generators.llm.llm_client import (
    ClaudeLLMClient,
    FROM_SCRATCH_MAX_TOKENS,
    MODIFY_MAX_TOKENS,
    _is_free_local_model,
)
from besser.generators.llm.prompt_builder import (
    build_scaffold_snapshot,
    build_system_prompt,
    build_inventory,
    build_endpoint_manifest,
)
from besser.generators.llm.stack_metadata import (
    detect_stack,
    pre_generate_metadata,
    stack_label,
)
from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.execution.process import _safe_subprocess_env
# Preserve established imports while lower-level consumers use these leaves.
from besser.generators.llm.validation.issues import (
    ValidationIssue,
    _check_did_not_run,
    _classify_issue,
    _hard_blockers,
    is_completion_issue,
    required_check_unverified,
    required_dependency_setup,
    _RUFF_STYLE_CODES as _RUFF_STYLE_CODES,
    _RUFF_BLOCKER_CODES as _RUFF_BLOCKER_CODES,
    _RUFF_LINE_RE as _RUFF_LINE_RE,
)
from besser.generators.llm.mutation_inventory import build_mutation_manifest
from besser.generators.llm.validation.python_source import (
    _create_schema_router_mismatches,
    _python_files,
    _CREATE_MODEL_RE as _CREATE_MODEL_RE,
    _CREATE_FIELD_RE as _CREATE_FIELD_RE,
    _ROUTER_READ_RE as _ROUTER_READ_RE,
)
from besser.generators.llm.tracing import (
    EVENT_CHECKPOINT,
    EVENT_COST_UPDATE,
    EVENT_PHASE_ENTER,
    EVENT_PHASE_EXIT,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_ROLLBACK,
    EVENT_SNAPSHOT,
    EVENT_TOOL_CALL,
    EVENT_TURN_START,
    EVENT_VALIDATION_ISSUE,
    TRACE_FILENAME,
    NullTraceWriter,
    TraceWriter,
)

logger = logging.getLogger(__name__)


# Where a rollback parks the current tree while it restores. Living inside
# output_dir keeps the move on one filesystem, so it is a rename, not a copy.
_ROLLBACK_DISCARD_DIR = ".besser_rollback_discard"

# Run bookkeeping that a rollback must NOT revert. The snapshot predates them,
# so restoring it would rewind the append-only trace and resurrect a stale
# checkpoint, making a later resume replay work already on disk.
_ROLLBACK_PRESERVED = {
    TRACE_FILENAME,
    CHECKPOINT_FILENAME,
    ".besser_recipe.json",
    _SNAPSHOT_DIR,
    _ROLLBACK_DISCARD_DIR,
}

# Dependency / build directories excluded from the recipe's output_files
# manifest (mirrors the web runner's zip exclusions).
_RECIPE_EXCLUDED_DIRS = {
    "target", "node_modules", "__pycache__", ".git", "dist", "build",
    ".next", ".gradle", "venv", ".venv", _SNAPSHOT_DIR,
    _ROLLBACK_DISCARD_DIR,
}


# Languages / frameworks BESSER has NO code generator for. An explicitly-named
# one must be built from scratch by the LLM (Phase 2) rather than scaffolded by
# the nearest built-in generator — otherwise a "C++ classes" request scaffolds
# Python and the customise loop yields a Python/C++ mishmash. Kept in sync with
# the modeling-agent classifier guard (unified_classifier._names_unsupported_stack).
_UNSUPPORTED_STACK_RE = _re.compile(
    r"\b(rust|kotlin|swift|scala|elixir|golang|ruby|php|dart|perl|haskell|zig|"
    r"nim|crystal|cpp|csharp|dotnet|fsharp|rails|nestjs|nextjs|express|"
    r"springboot|spring|laravel|symfony|angular|vue|svelte|nuxt|flutter|"
    r"objective-?c)\b",
    _re.I,
)
_UNSUPPORTED_STACK_LITERALS = ("c++", "c#", ".net", "f#")
_BARE_LANG_RE = _re.compile(
    r"\b(?:c|go)\b[\s\-]{0,3}(?:classes|class|code|program|programs|language|"
    r"structs?|headers?|files?|app|application)\b",
    _re.I,
)


def _names_unsupported_stack(instructions: str) -> bool:
    """True when the request explicitly names a language/stack BESSER has no
    generator for, so Phase 1 must be skipped and the LLM builds from scratch."""
    low = (instructions or "").lower()
    if any(tok in low for tok in _UNSUPPORTED_STACK_LITERALS):
        return True
    if _UNSUPPORTED_STACK_RE.search(low):
        return True
    if _BARE_LANG_RE.search(low):
        return True
    return False






# Tool-call detail shown in progress events.
_TOOL_DETAIL_MAX_CHARS = 160
# Argument keys worth putting in the stream, per tool. An allow list on purpose:
# file CONTENT must never reach the event stream, but path/target/action are what
# make a run readable afterwards.
_TOOL_DETAIL_KEYS = (
    "path", "file_path", "filename", "target", "action", "id", "ids",
    "text", "command", "pattern", "query", "class_name", "generator",
)


def _tool_call_detail(tool_name: str, tool_input: object, blocks_in_turn: int) -> str:
    """One short line saying what this tool call was about.

    Streamed alongside the tool name so a finished run can be read back from the
    durable event store. ``blocks_in_turn`` is included because 1 means the model
    batched nothing, and every turn costs a full prompt prefill - across a 10-run
    live batch every single turn carried exactly one call.
    """
    parts: list[str] = []
    if isinstance(tool_input, dict):
        for key in _TOOL_DETAIL_KEYS:
            if key not in tool_input:
                continue
            value = tool_input[key]
            if isinstance(value, (list, tuple)):
                rendered = ",".join(str(v) for v in value)
            else:
                rendered = str(value)
            rendered = " ".join(rendered.split())          # collapse newlines
            if not rendered:
                continue
            if len(rendered) > 60:
                rendered = rendered[:57] + "..."
            parts.append(f"{key}={rendered}")
    if blocks_in_turn and blocks_in_turn > 1:
        parts.append(f"batched={blocks_in_turn}")
    detail = " ".join(parts)
    return detail[:_TOOL_DETAIL_MAX_CHARS]


# The stdlib half of the allowlist, taken from the interpreter rather than
# hand-maintained: the hand-written set was missing argparse, sqlite3, glob,
# zipfile and ~180 others, each of which would be reported as "this app
# cannot start".
_STDLIB_ROOTS = frozenset(getattr(sys, "stdlib_module_names", ())) | {
    "typing_extensions",  # not stdlib, but ubiquitous and always installed
}

# Third-party roots a generated Python app legitimately imports. This is a
# fallback: the authoritative source is the app's own manifest (see
# ``_declared_dependency_roots``). Kept because a generator can emit an
# import before the manifest entry, and a missing manifest should not turn
# every framework import into a blocker.
#
# Every non-FastAPI family used to be absent here, which is why a 2-class
# Django app whose requirements.txt declares Django was reported with 10
# blockers and "cannot start" (2026-09-17). All 20 generators were affected.
_EXTERNAL_IMPORT_ROOTS = frozenset({
    # FastAPI / Starlette
    "fastapi", "pydantic", "pydantic_settings", "pydantic_core", "sqlalchemy",
    "starlette", "uvicorn", "alembic", "sqlmodel", "databases", "anyio",
    "sniffio", "greenlet", "mako",
    # Django
    "django", "rest_framework", "corsheaders", "django_filters", "drf_yasg",
    "drf_spectacular", "django_extensions", "storages", "environ",
    # Flask
    "flask", "flask_sqlalchemy", "flask_cors", "flask_migrate",
    "flask_login", "flask_jwt_extended", "flask_restful", "werkzeug",
    "jinja2", "markupsafe", "itsdangerous", "click", "marshmallow",
    # Drivers / infra
    "psycopg", "psycopg2", "pymysql", "mysql", "aiosqlite", "asyncpg",
    "pymongo", "motor", "redis", "celery", "boto3", "botocore", "gunicorn",
    # HTTP / auth / serialisation
    "httpx", "requests", "aiohttp", "websockets", "urllib3", "certifi",
    "idna", "charset_normalizer", "jose", "jwt", "passlib", "bcrypt",
    "argon2", "cryptography", "nacl", "dotenv", "multipart",
    "email_validator", "yaml", "toml", "orjson", "ujson", "jsonschema",
    "dateutil", "pytz", "attr", "attrs",
    # Common utility packages
    "loguru", "structlog", "rich", "typer", "tenacity", "cachetools",
    "slugify", "phonenumbers", "validators", "shortuuid", "ulid", "qrcode",
    "PIL", "openpyxl", "reportlab", "markdown", "bleach", "babel",
    "numpy", "pandas", "openai", "anthropic", "stripe", "sentry_sdk",
    "prometheus_client", "apscheduler",
    # Test tooling
    "pytest", "pytest_asyncio", "faker", "factory", "freezegun", "responses",
    "httpretty", "hypothesis",
})

# Distribution name → import root, for the cases where they differ. The
# default rule (lowercase, '-' → '_') handles everything not listed.
_DIST_TO_IMPORT_ROOT = {
    "djangorestframework": "rest_framework",
    "django-cors-headers": "corsheaders",
    "django-environ": "environ",
    "python-jose": "jose",
    "pyjwt": "jwt",
    "python-dotenv": "dotenv",
    "python-multipart": "multipart",
    "python-slugify": "slugify",
    "psycopg2-binary": "psycopg2",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "python-dateutil": "dateutil",
    "sentry-sdk": "sentry_sdk",
    "mysqlclient": "MySQLdb",
    "python-docx": "docx",
    "attrs": "attr",
}

_REQUIREMENT_NAME_RE = _re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _declared_dependency_roots(output_dir: str) -> set[str]:
    """Import roots the app's own manifest declares.

    Reading the manifest is what makes this check work for all 20
    generators instead of only the stack whose packages someone
    remembered to hardcode. A dependency the app declares is a
    dependency the app has.
    """
    roots: set[str] = set()

    def _add(dist: str) -> None:
        dist = dist.strip().strip('"\'').lower()
        if not dist:
            return
        roots.add(_DIST_TO_IMPORT_ROOT.get(dist, dist.replace("-", "_")))

    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [
            d for d in dirs
            if d not in ("node_modules", _SNAPSHOT_DIR, "__pycache__", ".git")
        ]
        for name in files:
            low = name.lower()
            path = os.path.join(root, name)
            try:
                if low.startswith("requirements") and low.endswith(".txt"):
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            line = line.split("#", 1)[0].strip()
                            if not line or line.startswith("-"):
                                continue
                            match = _REQUIREMENT_NAME_RE.match(line)
                            if match:
                                _add(match.group(1))
                elif low in ("pyproject.toml", "pipfile", "setup.py", "setup.cfg"):
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                    # Deliberately loose: any quoted requirement-looking
                    # token. A false extra allowlist entry costs nothing;
                    # a missed one costs a spurious blocker.
                    for token in _re.findall(r"[\"']([A-Za-z0-9._-]{2,})"
                                             r"[><=!~\s\"']", text):
                        _add(token)
            except OSError:
                continue
    return roots




def _unresolvable_local_imports(output_dir: str) -> list[str]:
    """Local imports that name a module the app does not ship where it is used.

    Ruff cannot find these: ``from sql_alchemy import *`` makes it report
    "unable to detect undefined names", which EXCUSES every name instead of
    flagging it, so a missing MODULE is invisible to F821. Live 2026-09-11: an
    app imported ``sql_alchemy`` and ``pydantic_classes`` with neither file
    present and still reported "0 blockers / 23 total", status=done.

    Resolution rules, each learned from a false positive against a working app:

    * A service runs with its OWN folder as cwd (``uvicorn main_api:app`` from
      ``backend/``), so that folder is importable from everything beneath it.
      Resolving only against the file's own directory condemned six imports.
    * Any directory holding ``.py`` files is importable - Python 3 implicit
      namespace packages need no ``__init__.py``.
    * The module existing in SOME OTHER directory does not count: a copy under
      ``pydantic/`` is not reachable from ``backend/``.
    """
    try:
        py_files = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(output_dir)
            for name in files
            if name.endswith(".py")
            if not any(part in ("node_modules", _SNAPSHOT_DIR, "__pycache__")
                       for part in root.split(os.sep))
        ]
    except OSError:
        return []

    provided: dict[str, set[str]] = {}
    for path in py_files:
        provided.setdefault(os.path.dirname(path), set()).add(
            os.path.splitext(os.path.basename(path))[0]
        )
    package_dirs = set(provided)

    # A directory holding .py files is an importable package FROM ITS
    # PARENT, not only as a sibling of the importing file. Without this,
    # `from core.config import settings` in backend/routers/x.py was
    # reported missing even though backend/core/ exists and backend/ is
    # the service's cwd.
    for directory in package_dirs:
        parent = os.path.dirname(directory)
        if parent and parent != directory:
            provided.setdefault(parent, set()).add(os.path.basename(directory))

    declared = _declared_dependency_roots(output_dir)

    problems: list[str] = []
    for path in py_files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                tree = _ast.parse(handle.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue  # syntax errors are reported by their own check

        # Everything importable from this file: its own folder plus every
        # ancestor folder up to the app root (the service's cwd).
        reachable: set[str] = set()
        probe = os.path.dirname(path)
        while True:
            reachable |= provided.get(probe, set())
            if os.path.normpath(probe) == os.path.normpath(output_dir):
                break
            parent = os.path.dirname(probe)
            if not parent or parent == probe:
                break
            probe = parent

        roots: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                if node.level:            # explicit relative import
                    continue
                if node.module:
                    roots.append(node.module.split(".")[0])
            elif isinstance(node, _ast.Import):
                roots.extend(alias.name.split(".")[0] for alias in node.names)

        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        for root_name in sorted(set(roots)):
            if (root_name in _EXTERNAL_IMPORT_ROOTS
                    or root_name in _STDLIB_ROOTS
                    or root_name in declared
                    or root_name in reachable):
                continue
            if any(os.path.basename(d) == root_name and os.path.dirname(d) ==
                   os.path.dirname(path) for d in package_dirs):
                continue
            problems.append(
                f"missing module: {rel} imports '{root_name}', which the app "
                f"does not provide where it is used - this app cannot start"
            )
    return problems


_STAR_IMPORT_RE = _re.compile(r"^\s*from\s+\S+\s+import\s+\*", _re.MULTILINE)


def _star_import_undefined_names(output_dir: str) -> list[str]:
    """``undefined name:`` blockers for a name a star-importing module uses but
    neither defines nor receives from the modules it star-imports.

    ruff cannot report these: with ``from x import *`` in scope every
    unresolved name is F405 ("may be undefined"), never F821, so the delivered
    app of 2026-09-18 shipped six NameError routes as "0 blockers". Files
    without a star import are left to ruff, which reports them as F821
    already. Same resolver as the write-time check the model was shown.
    """
    import importlib

    try:
        importlib.import_module("pyflakes.checker")
    except ImportError:
        return [_check_did_not_run(
            "the star-import name check", "pyflakes is not installed",
        )]
    from besser.generators.llm.write_diagnostics import diagnose_written_content

    issues: list[str] = []
    for path in _python_files(output_dir):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not _STAR_IMPORT_RE.search(content):
            continue
        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        for finding in diagnose_written_content(rel, content, workspace=output_dir):
            if finding.get("code") != "UndefinedName":
                continue
            detail = finding["message"].removeprefix("undefined name ")
            issues.append(f"undefined name: {rel} line {finding.get('line')}: {detail}")
    return issues


# The generated ORM module is imported and its mappers configured in a
# subprocess. SQLAlchemy resolves the strings in relationship() lazily, on the
# first query, so ast.parse and ruff both passed the 2026-09-18 live run
# 52befadf while every database request returned 500: a class-body
# ``relationship("Guest", secondary="booking_guest", ...)`` named a table that
# does not exist. Import-time NameErrors surface here as well - the class ruff
# excuses under a star import. ~0.5s per module, no database, server or
# network: the template keeps create_all under __main__.
_IMPORT_SMOKE_TIMEOUT_SECONDS = 30
_TRACEBACK_FRAME_RE = _re.compile(r'^\s*File "(.+?)", line (\d+)', _re.M)
_QUOTED_NAME_RE = _re.compile(r"'([^']+)'")


def _import_smoke_issues(output_dir: str) -> list[str]:
    """``mapper config:`` blockers for ORM modules that fail to import or
    configure; ``_check_did_not_run`` notes when the check itself could not."""
    import subprocess

    issues: list[str] = []
    for path in _python_files(output_dir):
        if os.path.basename(path) != "sql_alchemy.py":
            continue
        folder = os.path.dirname(path)
        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        modules = ["sql_alchemy"]
        if os.path.isfile(os.path.join(folder, "pydantic_classes.py")):
            modules.append("pydantic_classes")
        code = (
            f"import {', '.join(modules)}\n"
            "from sqlalchemy.orm import configure_mappers\n"
            "configure_mappers()\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                timeout=_IMPORT_SMOKE_TIMEOUT_SECONDS,
                cwd=folder, env=_safe_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            issues.append(_check_did_not_run(
                "the import smoke check",
                f"timed out after {_IMPORT_SMOKE_TIMEOUT_SECONDS}s on {rel}",
            ))
            continue
        except OSError as exc:
            issues.append(_check_did_not_run(
                "the import smoke check", f"could not be launched: {exc}",
            ))
            continue
        if result.returncode == 0:
            continue
        stderr = result.stderr or ""
        lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
        error = (lines[-1] if lines else "exited non-zero with no error output")[:400]
        if error.startswith("ModuleNotFoundError:"):
            # The harness interpreter is not the app's venv: a third-party
            # module missing HERE says nothing about the app. A missing local
            # module is _unresolvable_local_imports' finding.
            issues.append(_check_did_not_run(
                "the import smoke check", f"{rel}: {error}",
            ))
            continue
        where = _import_smoke_location(output_dir, folder, rel, stderr, error)
        issues.append(f"mapper config: {where}: {error}")
    return issues


def _import_smoke_location(output_dir, folder, rel, stderr, error) -> str:
    """``<file> line N`` when it can be found, else ``<file>``.

    An import-time error's traceback ends in the app's own file. A mapper
    configuration error's ends inside sqlalchemy, so the quoted name from the
    message (``'booking_guest'``) is looked up in the module instead.
    """
    real_folder = os.path.normcase(os.path.realpath(folder))
    located = None
    for m in _TRACEBACK_FRAME_RE.finditer(stderr):
        frame = os.path.realpath(os.path.join(folder, m.group(1)))
        inside = os.path.normcase(frame).startswith(real_folder + os.sep)
        if inside and os.path.isfile(frame):
            located = (frame, int(m.group(2)))
    if located is not None:
        frame, line_no = located
        frame_rel = os.path.relpath(
            frame, os.path.realpath(output_dir)
        ).replace("\\", "/")
        return f"{frame_rel} line {line_no}"
    try:
        with open(os.path.join(folder, "sql_alchemy.py"), "r",
                  encoding="utf-8", errors="ignore") as fh:
            module_lines = fh.read().splitlines()
    except OSError:
        return rel
    for name in _QUOTED_NAME_RE.findall(error):
        token = _re.compile(rf"\b{_re.escape(name)}\b")
        for idx, text in enumerate(module_lines, 1):
            if token.search(text):
                return f"{rel} line {idx}"
    return rel





# Tools that are read-only and shouldn't count for loop detection
_READONLY_TOOLS = frozenset({
    "read_file", "list_files", "search_in_files", "check_syntax", "validate_app", "test_api",
    # Checklist bookkeeping — marking several items done back-to-back is
    # exactly what the end_turn gate asks for, never a stuck loop.
    "task_list",
})

# The two ways the LLM edits an existing file. Both must count toward the
# per-file streak guard: the edit-recovery ladder deliberately pushes a
# flailing model from the first to the second, so counting only the first
# would mean reaching recovery silently disarms the guard.
_EDIT_TOOLS = frozenset({"modify_file", "replace_file_lines"})
# Tools whose ``path`` is recorded for that guard — the edits themselves,
# plus a re-read of the file being edited (part of the flail, not a break).
_EDIT_STREAK_TOOLS = _EDIT_TOOLS | {"read_file"}

# How many ruff lines Phase 3 reports. Blocker-code lines are always kept
# even past this, then files the LLM edited, then the rest.
_RUFF_MAX_REPORTED = 20

# Maximum workers for parallel tool execution
_MAX_PARALLEL_WORKERS = 4

# Phase 3 toolchain-fix outer cap. The LLM gets up to this many
# (collect → fix-loop → re-collect) iterations before we accept the
# remaining toolchain errors and move on. Each iteration is the existing
# 5-turn LLM fix loop, and the actual spend is bounded by ``max_cost_usd``
# (the inner loop stops when the cost cap is hit), so this cap just keeps
# a truly stuck run from looping forever. It is deliberately NOT tiny: as
# long as each round keeps reducing blockers (or we still have cost
# budget), the loop should keep going rather than abandon a run that is
# steadily converging — the no-progress-streak guard below handles the
# stuck case.
_MAX_TOOLCHAIN_FIX_ITERATIONS = 5

# Turns per fix attempt. An attempt that reaches the cap, or ends in prose,
# without one successful write gets exactly one more turn with modify_file
# forced (run 7f918e11, 2026-09-18: two attempts, ten turns, no edit).
_PHASE3_FIX_TURNS = 10
_PHASE3_NO_EDIT_REMINDER = (
    "<system-reminder>This attempt has not edited any file, and the blocker is "
    "still there. Explaining the fix does not apply it. Your next call must be "
    "modify_file on the file the blocker names (quote old_text exactly from the "
    "excerpt), or write_file if the file has to be rewritten. Then keep going "
    "until every blocker is fixed.</system-reminder>"
)

# Checkpoint history eviction (see history_eviction.py). When enabled, stale
# file bodies in older messages are stubbed at the compaction checkpoint to cut
# the re-sent context. OFF by default: it rewrites history the provider
# re-serializes and hasn't been live-verified (gen is rate-limited). Enable with
# BESSER_LLM_HISTORY_EVICTION=1 after a verification run.
_HISTORY_EVICTION_ENABLED = os.environ.get("BESSER_LLM_HISTORY_EVICTION", "0") == "1"

# Sub-generator tools that a chosen PRIMARY generator already bundles, so
# offering them to the Phase-2 agent only lets it scatter redundant top-level
# _gen_dir folders (e.g. a FastAPI backend already contains SQLAlchemy models,
# Pydantic schemas and REST routers inside backend/ — the standalone
# generate_pydantic / generate_sqlalchemy / generate_rest_api tools would emit
# duplicate pydantic/ sqlalchemy/ rest_api/ dirs next to it). The Phase-1
# SELECTOR is already told "generate_fastapi_backend includes SQLAlchemy +
# Pydantic — don't pick those separately", but that guidance never reached the
# Phase-2 agent; this removes the tools so it CANNOT call them.
# Generators that produce a WHOLE application. Once one of them has built the
# scaffold, none of them may run again: re-running the primary regenerates
# over every edit Phase 2 has made, and a rival stack drops a second
# application beside the assembled one - the failure the frontend contract's
# "rival framework imported into the scaffold" check exists to catch.
# Measured across twelve runs the model never called one, so this removes a
# risk and ~200 tokens per request rather than a capability it was using.
# Single-artefact generators (rdf, supabase, java/python classes) stay: a user
# can legitimately ask for one alongside the app.
_APPLICATION_PRIMARY_TOOLS = frozenset({
    "generate_web_app", "generate_django", "generate_flutter",
    "generate_fastapi_backend", "generate_rest_api",
})

_REDUNDANT_GENERATOR_TOOLS_BY_PRIMARY = {
    "generate_fastapi_backend": {
        "generate_pydantic", "generate_sqlalchemy", "generate_rest_api",
        "generate_json_schema", "generate_sql",
    },
    "generate_django": {
        "generate_pydantic", "generate_sqlalchemy", "generate_json_schema",
        "generate_sql",
    },
    "generate_web_app": {
        "generate_pydantic", "generate_sqlalchemy", "generate_rest_api",
        "generate_fastapi_backend", "generate_react", "generate_json_schema",
        "generate_sql",
    },
}

class LLMOrchestrator:
    """
    Three-phase orchestrator for LLM-augmented code generation.

    Phase 1 runs deterministically (no LLM): selects generator, runs it,
    inventories output, performs gap analysis.

    Phase 2 gives the LLM a scoped task list based on the gaps. The LLM
    only implements what's missing -- it doesn't rewrite the generator output.
    Multiple independent tool calls are executed in parallel.

    Phase 3 validates the output and gives the LLM a few turns to fix issues.
    Uses snapshot/rollback if fixes make things worse.
    """

    MAX_TURNS = 120

    # How many times the end_turn checklist gate sends the model back to
    # its open items before letting the run finish anyway.
    _MAX_TASK_NUDGES = 2
    _PHASE2_STAGNANT_TURNS = 10
    _PHASE2_INSPECTION_TURNS = 20

    def __init__(
        self,
        llm_client: ClaudeLLMClient,
        domain_model=None,
        gui_model=None,
        agent_model=None,
        agent_config: dict | None = None,
        output_dir: str | None = None,
        max_turns: int | None = None,
        max_cost_usd: float = 5.0,
        max_runtime_seconds: int = 1200,
        on_progress: Callable[..., None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_phase_details: Callable[[str, str], None] | None = None,
        use_streaming: bool = True,
        object_model=None,
        state_machines=None,
        quantum_circuit=None,
        bpmn_model=None,
        nn_model=None,
        auto_fix_issues: bool = False,
        should_continue: Callable[[], bool] | None = None,
        primary_kind: str | None = None,
        run_id: str = "",
        enable_tracing: bool = True,
        enable_checkpointing: bool = True,
        # Both default OFF: they enable run_command / install_dependencies, the
        # arbitrary-shell capability the hosted gate exists to withhold. Opt in
        # explicitly. (The 20-run experiment on 2026-09-11/12 produced its apps
        # with shell tools off, so the old permissive default bought nothing.)
        enable_toolchain_validation: bool = False,
        allow_shell_tools: bool = False,
        target_generator: str | None = None,
        target_generator_bound: bool = False,
        source_project_export: dict | None = None,
        per_write_diagnostics: bool = True,
        enable_import_smoke_check: bool = True,
        enable_requirements_ledger: bool = True,
        assembly_issues: list[dict] | None = None,
    ):
        self.client = llm_client
        self.domain_model = domain_model
        self.gui_model = gui_model
        self._assembly_issues = [
            {key: str(issue.get(key, ""))[:limit] for key, limit in (
                ("diagram_id", 120), ("diagram_type", 80), ("diagnostic", 160),
            )}
            for issue in assembly_issues or [] if isinstance(issue, dict)
        ]
        self.agent_model = agent_model
        self.agent_config = agent_config
        self.object_model = object_model
        self.bpmn_model = bpmn_model
        self.nn_model = nn_model
        # Normalise to a list so the prompt builder can iterate uniformly.
        if state_machines is None:
            self.state_machines: list = []
        elif isinstance(state_machines, (list, tuple, set)):
            self.state_machines = [sm for sm in state_machines if sm is not None]
        else:
            self.state_machines = [state_machines]
        self.quantum_circuit = quantum_circuit
        # Classify the primary driver. Explicit override wins; otherwise we
        # pick the first populated model in the standard preference order.
        # This is the anchor for generator selection, prompt framing, and
        # the plan preview endpoint.
        self.primary_kind = primary_kind or self._auto_detect_primary_kind()
        if self.primary_kind is None:
            raise ValueError(
                "LLMOrchestrator requires at least one model (domain, gui, "
                "agent, state_machine, object, BPMN, neural network, or quantum)"
            )
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="besser_llm_")
        self.max_turns = max_turns or self.MAX_TURNS
        self.max_cost_usd = max_cost_usd
        self.max_runtime_seconds = max_runtime_seconds
        self.on_progress = on_progress
        self.on_text = on_text
        self.on_phase_details = on_phase_details
        self.use_streaming = use_streaming
        # Weak / free-tier open models (e.g. qwen3-coder) ignore "don't switch
        # frameworks" and delete the whole deterministic scaffold to rebuild it.
        # Make that scaffold delete-proof for them only; capable cloud models are
        # unaffected and keep full delete_file.
        _model_name = (getattr(self.client, "model", "") or "").lower()
        self.executor = ToolExecutor(
            workspace=self.output_dir,
            domain_model=domain_model,
            gui_model=gui_model,
            agent_model=agent_model,
            agent_config=agent_config,
            quantum_circuit=quantum_circuit,
            object_model=object_model,
            bpmn_model=bpmn_model,
            nn_model=nn_model,
            protect_scaffold=_is_free_local_model(_model_name),
            per_write_diagnostics=per_write_diagnostics,
            # The executor enforces this too. Hiding the tools from the
            # advertised list is not a gate: the model can name a tool it was
            # never offered, and the dispatch table used to run it anyway.
            allow_shell=allow_shell_tools,
        )
        self.executor.app_validator = self._validate_app
        self.executor.api_tester = self._test_api
        self._app_validation_cache: tuple[str, dict] | None = None
        self._api_scenarios: dict[str, dict] = {}
        # Give the LLM tools scoped to the models it actually has. Tools
        # that need a domain model (pydantic/sqlalchemy/django/react/…)
        # are hidden when there isn't one so the LLM doesn't waste turns
        # calling generators that will just error. See tools.get_tools_for.
        # ``allow_shell_tools=False`` drops run_command/install_dependencies —
        # the arbitrary-shell tools. On a hosted, multi-tenant, BYOK box those
        # are user-steerable RCE (the cwd lock + denylist are UX, not a
        # sandbox), so the web runner disables them; trusted local/CLI/bench
        # runs keep them for self-verification.
        self.allow_shell_tools = allow_shell_tools
        from besser.generators.llm.tools import get_tools_for
        self.tools = get_tools_for(
            has_domain_model=self.domain_model is not None,
            has_gui_model=self.gui_model is not None,
            has_agent_model=self.agent_model is not None,
            has_state_machines=bool(self.state_machines),
            has_quantum_circuit=self.quantum_circuit is not None,
            has_object_model=self.object_model is not None,
            has_bpmn_model=self.bpmn_model is not None,
            has_nn_model=self.nn_model is not None,
            allow_shell=allow_shell_tools,
        )
        # Phase 3 auto-fix policy. False = report-only (industry default
        # for static analysers — fix on request, never blindly).
        self.auto_fix_issues = auto_fix_issues
        # Phase 3 toolchain checks (tsc / cargo / kotlinc) compile real
        # projects and can add minutes of wall-clock per run. The web
        # runner disables them per deploy (BESSER_LLM_ENABLE_TOOLCHAIN_
        # VALIDATION); library users keep the default. The cheap checks
        # In-process checks always run; shell/toolchain checks are gated.
        self.enable_toolchain_validation = enable_toolchain_validation
        # Phase 3 import smoke check: import the generated ORM module in a
        # subprocess and configure its mappers (~0.5s). Deliberately NOT tied
        # to allow_shell_tools - the hosted deploy has that off, and it is
        # where a mapper that fails on first use ships as a green run.
        self.enable_import_smoke_check = enable_import_smoke_check
        # Requirements ledger: the user's verbatim request turned into atomic
        # requirements once, then judged against the code on every Phase 3
        # pass (see requirements_ledger.py). ``_requirements`` is None until
        # extracted; the verdicts of the last pass go to the recipe.
        self.enable_requirements_ledger = enable_requirements_ledger
        self._requirements: list[dict] | None = None
        self._requirement_verdicts: list[dict] = []
        self._requirement_judgments: dict[str, list[dict]] = {}
        self._requirement_evidence_retries: set[str] = set()
        self._requirement_extraction_attempts = 0
        self._action_endpoints = None
        self._recent_tool_failures: list[dict] = []
        # Binding Phase-1 generator choice (e.g. from a user-approved
        # preview plan). A bound ``None`` explicitly skips Phase 1; an
        # unbound ``None`` keeps auto-selection. Either bound state avoids a
        # paid selector call and executes exactly the approved plan.
        self.target_generator = target_generator
        self.target_generator_bound = (
            target_generator_bound or target_generator is not None
        )
        # Cooperative cancellation hook. The orchestrator polls this at
        # the top of each Phase 2 turn. Returning False causes the loop
        # to exit cleanly — used by the SSE runner to honour
        # ``POST /cancel-smart-gen/{run_id}`` without killing the thread.
        self._should_continue = should_continue
        self.tool_calls_log: list[dict] = []
        self.total_turns = 0
        # (loop key, succeeded) per non-readonly call. Success matters: run
        # 0c537a4e (2026-09-18) was told "called 4 times in a row. Move on."
        # on two SUCCESSFUL edits because only the names were counted.
        self._recent_tool_calls: list[tuple[str, bool]] = []
        # Parallel ring buffer of (tool_name, path) entries used by the
        # per-file modify-loop guard. ``path`` is None for tools that
        # don't operate on a single file (e.g. ``list_files``,
        # ``run_command``) — those entries break any in-progress
        # modify_file streak. Kept separate from ``_recent_tool_calls``
        # so the legacy uniform-tool ``_is_stuck`` heuristic stays
        # exactly as it was.
        self._recent_modify_targets: list[tuple[str, str | None]] = []
        # Path most recently warned about — prevents the per-file
        # reminder from firing turn after turn while the LLM is still
        # working on the SAME file. Resets when a different file or
        # tool is observed.
        self._last_modify_warning_path: str | None = None
        self._phase2_inspection_handoff = ""
        # Escalation for an edit the executor has already rejected and the
        # model sends again (see _escalate_repeat_rejection): the tool the
        # next request must call, and per path the repeat count already acted on.
        self._force_tool_next: str | None = None
        self._repeat_escalations: dict[str, int] = {}
        self._compaction_count = 0
        self._generator_used: str | None = None
        # When Phase 1 selected a generator but the generator FAILED,
        # the reason ("<generator>: <error>") is stored here and woven
        # into the gap-analyser fallback + SSE stream, so neither the
        # user nor Phase 2 is left guessing why the scaffold is missing.
        self._phase1_failure_reason: str | None = None
        # Phase 0.5 stack id (e.g. ``"nextjs"``) and the list of metadata
        # files it created. Both stay None / [] when Phase 0.5 didn't
        # run (Python stacks, or unknown target). Used by the inventory
        # builder to surface "these files were pre-created" to the LLM.
        self._phase0_5_stack: str | None = None
        self._phase0_5_files: list[str] = []
        self._inventory: str = ""
        # True once the client's per-call output-token limit was raised
        # above its default: by ``_apply_adaptive_budget`` for a pure
        # from-scratch ``run()``, or by ``_apply_modify_budget`` for a
        # ``modify()`` run. The legacy recipe field name is retained for
        # compatibility; cost and runtime limits are never raised.
        self._adaptive_budget_applied: bool = False
        # Output-truncation retries in this Phase 2, as a PER-RUN total.
        # Deliberately not reset on a good turn: a model that truncates
        # repeatedly is not adapting, and an unbounded allowance would let it
        # spend the cost cap rediscovering that.
        self._truncation_retries: int = 0
        self._start_time: float | None = None
        # Stored as ValidationIssue objects so the recipe captures severity.
        # Cast to strings via `[str(i) for i in self._validation_issues]`
        # when emitting JSON.
        self._validation_issues: list[ValidationIssue] = []
        # True when Phase 3's repair was discarded because it ended
        # worse than it began; the recipe must not read as a clean fix.
        self._phase3_rolled_back = False
        self._previous_errors: list[str] = []  # track errors to avoid re-attempting
        # Last model name observed on the client. The provider's outage
        # fallback (``OpenAIProvider._activate_fallback``) can swap the
        # client's model mid-run; ``_notify_model_switch`` compares
        # against this after each LLM call and surfaces the change
        # through ``on_progress`` so the UI can show the actual model.
        self._last_seen_model: str | None = getattr(llm_client, "model", None)

        # Observability + crash recovery. Both are output-dir-local and
        # opt-out via the constructor flags so unit tests that don't
        # care (or that use in-memory ``tempfile.mkdtemp`` workspaces)
        # can disable them without polluting the working tree.
        self.run_id = run_id
        self._trace: TraceWriter | NullTraceWriter = (
            TraceWriter(self.output_dir, run_id=run_id, primary_kind=self.primary_kind)
            if enable_tracing
            else NullTraceWriter()
        )
        self._checkpointing_enabled = enable_checkpointing
        # Fingerprint is stable over the run; compute once so resume
        # validation doesn't re-hash on every turn.
        self._project_fingerprint = compute_fingerprint(
            instructions="",  # filled in once run() knows the instructions
            primary_kind=self.primary_kind,
            domain_model=domain_model,
            state_machines=self.state_machines,
            gui_model=gui_model,
            agent_model=agent_model,
            object_model=object_model,
            quantum_circuit=quantum_circuit,
            bpmn_model=bpmn_model,
            nn_model=nn_model,
        )
        self._resume_from_turn: int = 0
        self._resume_messages: list[dict] | None = None
        self._checkpoint_phase = "phase2"
        self._phase3_interrupted = False
        self._repair_progress: dict = {}
        # ``True`` once Phase 2 exits via end_turn (LLM said it's done).
        # Anything else — API error, cost cap, timeout, cancellation —
        # leaves this ``False`` so the checkpoint is preserved for a
        # possible resume. Kept separate from ``self._validation_issues``
        # because those are about Phase 3 quality, not run completion.
        self._phase2_exited_cleanly: bool = False
        # Why Phase 2 stopped. "completed" only when the LLM signalled
        # end_turn; otherwise one of: "api_error", "cost_cap", "timeout",
        # "cancelled", "max_turns". The runner reads this to decide whether
        # to warn the user that the downloaded output may be incomplete.
        self._phase2_stop_reason: str = "max_turns"
        # Short provider error string captured when stop_reason == "api_error",
        # surfaced to the user so a rate-limit reads as such (not a mystery).
        self._phase2_api_error: str = ""
        # end_turn checklist gate: how many times we've already sent the
        # model back to its open task_list items (bounded by
        # ``_MAX_TASK_NUDGES`` so a stubborn model can't loop the budget).
        self._end_turn_task_nudges: int = 0
        # Model-derived acceptance matrix (per entity: route/page/create),
        # computed by Phase 3 and saved in the recipe. Report-only.
        self._acceptance_matrix: dict | None = None
        # The run's instructions, kept for Phase 3 checks that depend on
        # what was ASKED (e.g. "web app" requested but no frontend files).
        self._instructions: str = ""

        # Incremental vibe-modify state. ``modify()`` seeds ``output_dir``
        # from a previous run's files and edits them in place instead of
        # rebuilding from scratch. ``_modify_mode`` is the single flag that
        # threads through ``_build_system_prompt`` to prepend the
        # "preserve what works" directive; it stays False on the run() /
        # resume() paths so those prompts are byte-identical to today's.
        # ``_seed_generator_used`` records which deterministic generator
        # first produced the seeded base (read back from the seed's
        # recipe) so the inventory / new recipe frame the run correctly.
        self._modify_mode: bool = False
        self._seed_generator_used: str | None = None
        # Unresolved blockers carried over from the seed run's recipe
        # (modify() only). A modify run must pay down the seed's known
        # debt, not just layer new edits on top of it — otherwise every
        # defect the seed run shipped survives forever because later
        # runs never look at it again.
        self._seed_unresolved_issues: list[str] = []
        # Model-sync during vibe-MODIFY (class-diagram only). When a
        # ``modify()`` instruction implies new domain entities (e.g. "add
        # authentication" → a ``User`` class), ``_derive_and_apply_model_deltas``
        # mutates ``self.domain_model`` IN PLACE and re-serialises an updated
        # project export here so the push writes ``buml/`` from the UPDATED
        # model. Both stay untouched on the ``run()`` / ``resume()`` paths —
        # the delta step is invoked ONLY from ``modify()`` — so from-scratch
        # generation is byte-identical. ``_source_project_export`` is the run's
        # original export (passed by the web runner); ``_updated_project_export``
        # is ``None`` unless at least one new class was actually added.
        self._source_project_export: dict | None = source_project_export
        self._updated_project_export: dict | None = None

        # Fix/modify success gate (modify() only). When a modify run is
        # seeded by a user-reported error (a pasted traceback, a broken
        # endpoint, "it 400s"), the reported failure IS the run's success
        # criterion: the loop must attempt exactly that and must not report
        # a clean success while it is unresolved. ``_fix_target`` is the
        # parsed :class:`ReportedTarget`; ``_is_fix_run`` gates every branch
        # so from-scratch run()/resume() are byte-identical (they never set
        # these). ``_fix_target_resolved`` / ``_fix_target_message`` are the
        # end-of-run verdict the runner surfaces honestly.
        self._is_fix_run: bool = False
        self._fix_target = None  # ReportedTarget | None
        self._fix_target_resolved: bool | None = None
        self._fix_target_message: str | None = None

    _LOOP_THRESHOLD = 4
    # Tighter, per-file threshold for the modify_file streak guard.
    # Long sequences of small modify_file edits on the same path are
    # the typical "death by a thousand cuts" failure mode: the LLM
    # keeps making forward progress, so the legacy uniform-tool
    # ``_is_stuck`` warning (a soft note appended to a tool_result)
    # doesn't change behaviour. At 3 consecutive single-file edits we
    # inject a high-salience reminder before the NEXT LLM call.
    #
    # Derived, not re-declared: this guard reads the same counter
    # (``consecutive_modify_misses``) that the executor refuses on, so two
    # independent 3s in two files could be tuned apart and silently disagree.
    _PER_FILE_MODIFY_THRESHOLD = ToolExecutor._MAX_MODIFY_MISSES
    # How many output-token truncations one Phase 2 may recover from in
    # total, never reset. Past this the model is not adapting and resume is a
    # better answer than burning the cost cap.
    _MAX_TRUNCATION_RETRIES = 4

    def _auto_detect_primary_kind(self) -> str | None:
        """Pick the primary model kind from whatever is present.

        Same order as the service-layer assembler: class diagrams are
        preferred when present because they drive the most mature
        deterministic generators. We duplicate the ordering here rather
        than importing from the web-layer assembler so the orchestrator
        stays independent of the HTTP surface.
        """
        if self.domain_model is not None:
            return "class"
        if self.gui_model is not None:
            return "gui"
        if self.agent_model is not None:
            return "agent"
        if self.state_machines:
            return "state_machine"
        if self.object_model is not None:
            return "object"
        if self.bpmn_model is not None:
            return "bpmn"
        if self.nn_model is not None:
            return "nn"
        if self.quantum_circuit is not None:
            return "quantum"
        return None

    # ==================================================================
    # Main entry point
    # ==================================================================

    def run(self, instructions: str) -> str:
        """Run the three-phase generation. Returns path to output directory."""
        if not instructions or not instructions.strip():
            raise EmptyInstructionsError("Instructions cannot be empty")
        validate_specification(instructions)
        self._instructions = instructions

        self._start_time = time.monotonic()
        # Re-compute fingerprint now that we know the instructions — the
        # constructor-time value was hashed with an empty string.
        self._project_fingerprint = compute_fingerprint(
            instructions=instructions,
            primary_kind=self.primary_kind,
            domain_model=self.domain_model,
            state_machines=self.state_machines,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            object_model=self.object_model,
            quantum_circuit=self.quantum_circuit,
            bpmn_model=self.bpmn_model,
            nn_model=self.nn_model,
        )
        self._trace.write(
            EVENT_RUN_START,
            # bounded: trace display only, never a decision input
            instructions=instructions[:500],
            max_cost_usd=self.max_cost_usd,
            max_runtime_seconds=self.max_runtime_seconds,
            max_turns=self.max_turns,
        )

        # -- Phase 0: the model itself ------------------------------------
        # Runs before anything is generated: a defect here is one no amount
        # of Phase 2 or Phase 3 work can repair, because the code will be a
        # faithful rendering of an impossible specification.
        self._collect_model_contract_issues()

        # -- Phase 1: Deterministic generation ----------------------------
        self._trace.write(EVENT_PHASE_ENTER, phase="phase1")
        self._run_phase1(instructions)
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="phase1",
            generator_used=self._generator_used,
        )

        # -- Phase 0.5: Stack-metadata floor (only when no Phase 1 ran) ---
        # When Phase 1 picked a Python generator (Django / FastAPI /
        # SQLAlchemy / Pydantic / plain Python), the deterministic
        # generator already emitted the manifest. Phase 0.5 only
        # intervenes when Phase 1 was a no-op — i.e. the target is a
        # stack BESSER doesn't generate (Next.js, Rust, Kotlin / Spring).
        # This keeps the Python paths byte-identical to today's output.
        self._run_phase0_5_metadata(instructions)

        # -- Adaptive budget: raise the cap for from-scratch runs ---------
        # Must run after Phase 1 (needs ``self._generator_used``) and
        # after Phase 0.5 (needs ``self._phase0_5_stack`` for logging) but
        # before Phase 2, since that's the loop the raised cost/runtime
        # cap and the raised client max_tokens actually apply to.
        self._apply_adaptive_budget()

        # -- Phase 1.5: Validate Phase 1 output ---------------------------
        phase1_issues = self._validate_phase1_output()
        for issue in phase1_issues:
            self._trace.write(EVENT_VALIDATION_ISSUE, phase="phase1_5", message=issue)

        # -- Phase 2: LLM customization -----------------------------------
        self._trace.write(EVENT_PHASE_ENTER, phase="phase2")
        self._run_phase2(instructions, extra_issues=phase1_issues)
        self._trace.write(EVENT_PHASE_EXIT, phase="phase2", turns=self.total_turns)

        # -- Snapshot BEFORE Phase 3 (preserves all Phase 2 work) ---------
        # If Phase 3 fixes make things worse, we roll back here
        # (keeping Phase 2 work intact), not back to Phase 1.
        self._create_snapshot()
        self._trace.write(EVENT_SNAPSHOT, before_phase="phase3")

        # -- Phase 3: Validate & fix --------------------------------------
        if self._phase2_exited_cleanly or self._phase2_stop_reason == "validation_required":
            self._save_phase3_checkpoint()
        self._trace.write(EVENT_PHASE_ENTER, phase="phase3")
        self._run_phase3_validation()
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="phase3",
            unresolved_blockers=sum(
                1 for i in self._validation_issues if i.severity == "blocker"
            ),
        )

        elapsed = time.monotonic() - self._start_time
        logger.info(
            "LLM generation finished: %d turns, %.1fs, %d tool calls, "
            "generator=%s, compactions=%d",
            self.total_turns, elapsed, len(self.tool_calls_log),
            self._generator_used or "none", self._compaction_count,
        )

        # Log cost
        logger.info("Cost: %s", self.client.usage)

        self._save_recipe(instructions, elapsed)

        # Clean up snapshot
        self._remove_snapshot()

        self._finish_checkpoint()

        self._trace.write(
            EVENT_RUN_END,
            elapsed_seconds=round(elapsed, 2),
            total_turns=self.total_turns,
            estimated_cost_usd=float(self.client.usage.estimated_cost),
            validation_issues=len(self._validation_issues),
        )

        return self.output_dir

    # ==================================================================
    # Resume entry point
    # ==================================================================

    def resume(self, instructions: str) -> str:
        """Resume a previously-crashed run from its checkpoint.

        Loads ``.besser_checkpoint.json`` from ``self.output_dir`` and
        continues the saved phase. Phase 1 is skipped entirely. Repair
        checkpoints contain state, not a conversation to replay: they go
        directly to fresh validation against the current files.

        Raises
        ------
        FileNotFoundError
            No checkpoint in the output dir — nothing to resume.
        ValueError
            The checkpoint's project fingerprint disagrees with the
            current project/instructions. We refuse rather than silently
            resuming against a different spec.
        """
        validate_specification(instructions)
        self._instructions = instructions
        checkpoint = load_checkpoint(self.output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint at {self.output_dir}; nothing to resume"
            )

        expected = compute_fingerprint(
            instructions=instructions,
            primary_kind=self.primary_kind,
            domain_model=self.domain_model,
            state_machines=self.state_machines,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            object_model=self.object_model,
            quantum_circuit=self.quantum_circuit,
            bpmn_model=self.bpmn_model,
            nn_model=self.nn_model,
        )
        if expected != checkpoint.project_fingerprint:
            raise CheckpointMismatchError(
                "Checkpoint fingerprint does not match the supplied "
                "project/instructions — refusing to resume. Start a "
                "fresh run if you changed the model or the request."
            )

        # Re-hydrate the counters that drive Phase 2 semantics.
        self._start_time = time.monotonic()
        self.total_turns = checkpoint.total_turns
        # Seed the fresh UsageTracker with what the crashed run already
        # spent — otherwise the cost cap only covers post-resume spend
        # and a crash-resume cycle could legally double the user's bill.
        try:
            self.client.usage.seed_cost(float(checkpoint.estimated_cost_usd or 0.0))
        except (AttributeError, TypeError, ValueError):
            # Older checkpoints / mock clients without seed_cost — keep
            # resuming rather than failing the run over cost accounting.
            logger.debug("Could not seed resumed cost", exc_info=True)
        self._checkpoint_phase = checkpoint.phase
        self._resume_from_turn = checkpoint.turn if checkpoint.phase == "phase2" else 0
        self._resume_messages = checkpoint.messages if checkpoint.phase == "phase2" else None
        self._phase2_stop_reason = checkpoint.phase2_stop_reason
        self._phase2_exited_cleanly = checkpoint.phase2_exited_cleanly
        self._phase3_interrupted = False
        # On-disk changes survive interruption. Old scheduling history is only
        # relevant to those exact bytes; validation results are always rebuilt.
        self._repair_progress = (
            dict(checkpoint.repair_progress)
            if checkpoint.source_revision == self._workspace_revision() else {}
        )
        self._inventory = checkpoint.inventory
        self._generator_used = checkpoint.generator_used
        self._compaction_count = checkpoint.compaction_count
        self.tool_calls_log = list(checkpoint.tool_calls_log)
        self._validation_issues = [
            ValidationIssue(i.get("severity", "warning"), i.get("message", ""))
            for i in checkpoint.validation_issues
        ]
        self._project_fingerprint = checkpoint.project_fingerprint
        from besser.generators.llm.checkpoint import restore_api_scenarios
        self._api_scenarios = restore_api_scenarios(checkpoint.api_scenarios)
        # A resumed loop must retain the same definition of done. Rebuild the
        # harness-owned verifier callables from the current workspace/model and
        # reattach them by task text; ordinary LLM-planned tasks need no callable.
        self.executor.restore_tasks(
            checkpoint.tasks,
            verification_tasks=self._deterministic_gap_tasks(),
        )
        self._trace.write(
            EVENT_RUN_START,
            resumed=True,
            resume_from_turn=checkpoint.turn,
            resume_phase=checkpoint.phase,
            saved_at=checkpoint.saved_at,
        )

        # -- Adaptive budget: same rule as a fresh run (``_generator_used``
        # was just restored from the checkpoint above). Resuming is the
        # common path for a run that previously broke out on a cost_cap /
        # timeout / max_tokens truncation, so this matters here too.
        self._apply_adaptive_budget()

        if checkpoint.phase == "phase2":
            self._trace.write(EVENT_PHASE_ENTER, phase="phase2_resume")
            self._run_phase2(instructions, extra_issues=[])
            self._trace.write(EVENT_PHASE_EXIT, phase="phase2_resume", turns=self.total_turns)
        else:
            self._drop_redundant_generator_tools()
            self.executor.set_scaffold_family(self._scaffold_family())

        self._create_snapshot()
        self._trace.write(EVENT_SNAPSHOT, before_phase="phase3_resume")
        if (checkpoint.phase == "phase3" or self._phase2_exited_cleanly
                or self._phase2_stop_reason == "validation_required"):
            self._save_phase3_checkpoint()
        self._trace.write(EVENT_PHASE_ENTER, phase="phase3")
        self._run_phase3_validation()
        self._trace.write(EVENT_PHASE_EXIT, phase="phase3")

        elapsed = time.monotonic() - self._start_time
        self._save_recipe(instructions, elapsed)
        self._remove_snapshot()
        self._finish_checkpoint()
        self._trace.write(EVENT_RUN_END, resumed=True, elapsed_seconds=round(elapsed, 2))
        return self.output_dir

    # ==================================================================
    # Incremental vibe-modify entry point
    # ==================================================================

    def modify(self, instructions: str) -> str:
        """Edit a seeded workspace in place instead of rebuilding it.

        Modelled on ``run()`` (NOT ``resume()``): there is no checkpoint
        load, no fingerprint gate, and no crash-recovery replay. The runner
        has already copied a previous run's generated files into
        ``self.output_dir`` (stripping that run's checkpoint + snapshot but
        KEEPING its ``.besser_recipe.json``). This method:

          * SKIPS Phase 1 entirely — the deterministic generator would
            overwrite the customised files the user wants to keep.
          * Re-derives the inventory + generator-file tags from the seed
            so Phase 2 sees the real on-disk state.
          * Drives Phase 2 with ``modify_mode=True`` so the system prompt
            biases the LLM toward the smallest surgical change.
          * Runs Phase 1.5 validation, Phase 3 validation, and the recipe
            save exactly like ``run()``; drops the checkpoint on clean exit.

        The user may also have edited the model between runs;
        ``self.domain_model`` reflects that. For the MVP this is a pure
        code-edit — the inventory and gap analyzer already surface the
        model's classes vs. the files on disk, so Phase 2 authors any
        deltas via write_file / modify_file (no scaffold merge yet).

        Returns the path to ``self.output_dir``.
        """
        if not instructions or not instructions.strip():
            raise EmptyInstructionsError("Instructions cannot be empty")
        validate_specification(instructions)
        self._instructions = instructions

        self._start_time = time.monotonic()
        self._modify_mode = True
        # Edit-first guardrail: in a modify run, whole-file rewrites of
        # existing files are rejected until targeted edits were tried —
        # rewrites are where modify-run regressions come from.
        self.executor.enable_modify_guard()

        # Give the modify/fix path the wider per-call output ceiling: a
        # single-turn file rewrite here overruns the client's default cap
        # and truncates mid-file. Set once, before Phase 2. Cost/runtime
        # caps are untouched; only the response-size ceiling is raised.
        self._apply_modify_budget()

        # Re-hydrate generator-file tags + the seed's generator name from
        # the copied recipe BEFORE building the inventory (which needs a
        # generator name for its framing line).
        self._seed_generator_files_from_recipe()
        # Frame the run as editing an existing project. When the seed came
        # from a deterministic generator we adopt that name (accurate — the
        # base + prior LLM edits descend from it) so gap analysis, the
        # scaffold-snapshot inlining, and the saved recipe all line up.
        self._generator_used = self._seed_generator_used
        self.executor.set_scaffold_family(self._scaffold_family())

        # -- Model-sync: derive + apply class-diagram deltas implied by the
        # instruction BEFORE building the inventory, so a genuinely new
        # domain entity (e.g. a ``User`` class for "add authentication")
        # flows into Phase 2's inventory AND the updated model reaches the
        # push path. Fully guarded (see the method): an empty/failed/bad
        # delta leaves the run proceeding EXACTLY as before — no model
        # change, no crash. Scoped to modify() only; run()/_run_phase1
        # never invoke it, so from-scratch output is byte-identical.
        self._derive_and_apply_model_deltas(instructions)

        self._inventory = build_inventory(
            self.output_dir,
            self.domain_model,
            self._seed_generator_used or "existing project",
        )
        # Session memory: open the run knowing what was asked and changed
        # in every previous run on this app (recipe history), instead of
        # rediscovering it from file contents.
        session_recap = self._render_seed_history()
        if session_recap:
            self._inventory = f"{self._inventory}\n{session_recap}"

        # -- Fix/modify success gate: parse the user-reported failure ------
        # A modify run seeded by a reported error (traceback / broken
        # endpoint) makes that failure the run's success criterion. Detect
        # + parse it once here; every downstream branch is gated on
        # ``_is_fix_run`` so run()/resume() stay byte-identical.
        self._detect_fix_target(instructions)

        self._trace.write(
            EVENT_RUN_START,
            mode="modify",
            # bounded: trace display only, never a decision input
            instructions=instructions[:500],
            max_cost_usd=self.max_cost_usd,
            max_runtime_seconds=self.max_runtime_seconds,
            max_turns=self.max_turns,
            fix_run=self._is_fix_run,
            fix_target=(self._fix_target.descriptor if self._fix_target else None),
        )

        # -- Phase 1.5: Validate the seeded output (no Phase 1 run) --------
        phase1_issues = self._validate_phase1_output()
        for issue in phase1_issues:
            self._trace.write(EVENT_VALIDATION_ISSUE, phase="phase1_5", message=issue)

        # -- Phase 2: LLM edits the seeded files in place ------------------
        # Forward the seed run's unresolved blockers (D1): they ride along
        # with the fresh Phase 1.5 findings so the modify run pays down
        # the known debt instead of preserving it forever.
        if self._seed_unresolved_issues:
            logger.info(
                "Forwarding %d unresolved blocker(s) from the seed run",
                len(self._seed_unresolved_issues),
            )
            for issue in self._seed_unresolved_issues:
                self._trace.write(
                    EVENT_VALIDATION_ISSUE, phase="seed_forward", message=issue,
                )
        # Feed the tool's OWN structural findings that match the reported
        # target into Phase 2 as explicit fix instructions, so the LLM acts
        # on the concrete defect (e.g. "the Watchlist create form is not
        # wired") instead of only the user's paraphrase. Empty on a
        # non-fix modify run and on run()/resume().
        fix_seed_issues = self._fix_run_scoped_issues()
        if fix_seed_issues:
            for issue in fix_seed_issues:
                self._trace.write(
                    EVENT_VALIDATION_ISSUE, phase="fix_target_seed", message=issue,
                )
        self._trace.write(EVENT_PHASE_ENTER, phase="phase2_modify")
        self._run_phase2(
            instructions,
            extra_issues=(
                phase1_issues + self._seed_unresolved_issues + fix_seed_issues
            ),
        )
        self._trace.write(
            EVENT_PHASE_EXIT, phase="phase2_modify", turns=self.total_turns,
        )

        # -- Snapshot BEFORE Phase 3 (preserves all Phase 2 edits) --------
        self._create_snapshot()
        self._trace.write(EVENT_SNAPSHOT, before_phase="phase3")

        # -- Phase 3: Validate & fix --------------------------------------
        if self._phase2_exited_cleanly or self._phase2_stop_reason == "validation_required":
            self._save_phase3_checkpoint()
        self._trace.write(EVENT_PHASE_ENTER, phase="phase3")
        self._run_phase3_validation()
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="phase3",
            unresolved_blockers=sum(
                1 for i in self._validation_issues if i.severity == "blocker"
            ),
        )

        # -- Fix/modify success gate --------------------------------------
        # After Phase 3, decide whether the reported failure is plausibly
        # addressed. A matching finding still present marks the run
        # incomplete with an honest, target-specific message instead of a
        # clean "success / 0 blockers".
        self._evaluate_fix_target_gate()

        elapsed = time.monotonic() - self._start_time
        logger.info(
            "LLM modify finished: %d turns, %.1fs, %d tool calls, "
            "seed_generator=%s, compactions=%d",
            self.total_turns, elapsed, len(self.tool_calls_log),
            self._seed_generator_used or "none", self._compaction_count,
        )
        logger.info("Cost: %s", self.client.usage)

        self._save_recipe(instructions, elapsed)
        self._remove_snapshot()

        self._finish_checkpoint()

        self._trace.write(
            EVENT_RUN_END,
            mode="modify",
            elapsed_seconds=round(elapsed, 2),
            total_turns=self.total_turns,
            estimated_cost_usd=float(self.client.usage.estimated_cost),
            validation_issues=len(self._validation_issues),
        )
        return self.output_dir

    def _render_seed_history(self) -> str:
        """Format the seed recipe's session history for the inventory."""
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        history = self._load_recipe_history(recipe_path)
        if not history:
            return ""
        lines = ["\nPrevious work on this app (oldest first):"]
        for i, entry in enumerate(history, 1):
            request = (entry.get("instructions") or "").strip().replace("\n", " ")
            mode = entry.get("mode") or "run"
            files = entry.get("files_touched") or []
            file_note = ""
            if files:
                shown = ", ".join(files[:8])
                more = f" (+{len(files) - 8} more)" if len(files) > 8 else ""
                file_note = f" — touched: {shown}{more}"
            lines.append(f"  {i}. [{mode}] \"{request}\"{file_note}")
        lines.append(
            "Respect this history: those changes are deliberate and must "
            "survive your edits unless the new request says otherwise."
        )
        return "\n".join(lines)

    def _seed_generator_files_from_recipe(self) -> None:
        """Pre-load generator-file tags from a seeded run's recipe.

        ``modify()`` runs against ``output_dir`` copied from a previous
        run, and that copy KEEPS the previous ``.besser_recipe.json`` whose
        ``output_files`` entries are tagged ``source: generator|llm``. We
        replay the ``generator`` tags into
        ``self.executor._generator_files`` so the write-tool guardrail
        still protects deterministically-generated files, and
        ``_save_recipe`` re-tags them ``generator`` for the new run. Also
        records ``generator_used`` so the caller can frame the run.

        Best-effort: a missing / unreadable / malformed recipe just leaves
        every file tagged ``llm`` (harmless — the guardrail relaxes and the
        LLM can still edit anything).
        """
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        if not os.path.isfile(recipe_path):
            return
        try:
            with open(recipe_path, "r", encoding="utf-8") as fh:
                recipe = json.load(fh)
        except Exception:
            logger.debug(
                "Seed recipe unreadable; treating all seeded files as llm",
                exc_info=True,
            )
            return
        if not isinstance(recipe, dict):
            return
        self._seed_generator_used = recipe.get("generator_used")
        # Seed-issue forwarding: blockers the seed run could not fix are
        # THIS run's opening tasks. Blockers only — warnings/style would
        # bloat the prompt with ruff noise and dilute the real debt.
        try:
            self._seed_unresolved_issues = [
                f"Unresolved from the previous run: {i.get('message', '')}"
                for i in recipe.get("validation_issues", [])
                if isinstance(i, dict)
                and i.get("severity") == "blocker"
                and i.get("message")
            ]
        except Exception:
            logger.debug("Seed recipe validation_issues malformed", exc_info=True)
        try:
            for entry in recipe.get("output_files", []):
                if (
                    isinstance(entry, dict)
                    and entry.get("source") == "generator"
                    and isinstance(entry.get("path"), str)
                ):
                    self.executor._generator_files.add(entry["path"])
        except Exception:
            logger.debug("Seed recipe output_files malformed", exc_info=True)

    # ==================================================================
    # Fix/modify success gate (modify() only)
    # ==================================================================

    # Prefix stamped on a finding promoted because it matches the reported
    # failure. Self-contained and stable so the end-of-run gate can
    # identify a promoted finding regardless of re-collection.
    _FIX_TARGET_PREFIX = "reported-failure: "

    def _detect_fix_target(self, instructions: str) -> None:
        """MODIFY-only: parse the user-reported failure into a target.

        Sets ``self._fix_target`` and ``self._is_fix_run``. Best-effort — a
        parse failure (or a modify run with no fix/error vocabulary) leaves
        the run ungated, so a plain feature-add modify behaves exactly as
        before. NEVER invoked from run()/resume(); from-scratch generation
        is untouched.
        """
        try:
            from besser.generators.llm.fix_target import parse_reported_target
            self._fix_target = parse_reported_target(
                instructions, self.domain_model,
            )
        except Exception:
            logger.debug("Fix-target parse failed", exc_info=True)
            self._fix_target = None
        self._is_fix_run = self._fix_target is not None
        if self._is_fix_run:
            logger.info(
                "Fix/modify run: reported target = %s (kind=%s, entities=%s)",
                self._fix_target.descriptor,
                self._fix_target.kind,
                ", ".join(self._fix_target.entities) or "none",
            )

    def _acceptance_seed_issues(self) -> list[str]:
        """Acceptance-matrix findings on the current (seed) workspace."""
        try:
            from besser.generators.llm.acceptance import (
                build_acceptance_matrix,
                matrix_issues,
            )
            matrix = build_acceptance_matrix(self.output_dir, self.domain_model)
            return matrix_issues(matrix)
        except Exception:
            logger.debug("Seed acceptance computation failed", exc_info=True)
            return []

    def _fix_run_scoped_issues(self) -> list[str]:
        """Explicit Phase-2 fix instructions derived from the reported target.

        Always leads with a headline naming the reported failure (so the
        LLM knows the success criterion even when nothing structural
        matched), then appends the tool's OWN structural findings on the
        SEED workspace that match the target (acceptance matrix +
        data-contract lint). Empty on a non-fix run and on run()/resume().
        """
        if not self._is_fix_run or self._fix_target is None:
            return []
        from besser.generators.llm.fix_target import finding_matches_target

        issues: list[str] = [
            "The user reported this failure and it must be resolved in this "
            f"run: {self._fix_target.descriptor}. Reproduce the cause, fix "
            "it, and confirm the reported request/flow now succeeds; do not "
            "end the run while it is unresolved."
        ]
        raw = list(self._acceptance_seed_issues()) + list(
            self._collect_data_contract_issues()
        )
        for finding in raw:
            if finding_matches_target(finding, self._fix_target):
                issues.append(
                    "Concrete defect behind the reported failure — "
                    f"{finding}. Repair this so the reported request succeeds."
                )
        return issues

    def _promote_fix_target_findings(
        self, issues: list[ValidationIssue],
    ) -> list[ValidationIssue]:
        """Promote target-matching findings from warning to blocker.

        Scoped to a fix run only; a no-op otherwise (so from-scratch
        Phase 3 classification is identical). A promoted finding is
        rewritten with a stable prefix + the target descriptor so it reads
        honestly in the recipe and the end-of-run gate can identify it.
        """
        if not self._is_fix_run or self._fix_target is None:
            return issues
        from besser.generators.llm.fix_target import finding_matches_target

        promoted: list[ValidationIssue] = []
        for issue in issues:
            if (
                issue.severity != "blocker"
                and not issue.message.startswith(self._FIX_TARGET_PREFIX)
                and finding_matches_target(issue.message, self._fix_target)
            ):
                promoted.append(ValidationIssue(
                    "blocker",
                    f"{self._FIX_TARGET_PREFIX}{issue.message} "
                    f"(matches the reported failure: {self._fix_target.descriptor})",
                ))
            else:
                promoted.append(issue)
        return promoted

    def _evaluate_fix_target_gate(self) -> None:
        """Decide whether the reported failure is plausibly addressed.

        Sets ``_fix_target_resolved`` / ``_fix_target_message`` from the
        FINAL Phase-3 issue list. A blocker that matches the target (either
        one we promoted or a pre-existing data-contract blocker about the
        same entity) means the run must NOT claim a clean success.

        A soft target (no entities we could ground) is left as ``None`` —
        we neither confirm nor deny a specific defect, rather than
        over-claiming either way.
        """
        if not self._is_fix_run or self._fix_target is None:
            return
        if self._fix_target.is_soft:
            self._fix_target_resolved = None
            logger.info(
                "Fix run: soft target %s — no structural signal to verify",
                self._fix_target.descriptor,
            )
            return
        from besser.generators.llm.fix_target import finding_matches_target

        unresolved = [
            i for i in self._validation_issues
            if i.severity == "blocker"
            and (
                i.message.startswith(self._FIX_TARGET_PREFIX)
                or finding_matches_target(i.message, self._fix_target)
            )
        ]
        if unresolved:
            self._fix_target_resolved = False
            detail = unresolved[0].message
            if detail.startswith(self._FIX_TARGET_PREFIX):
                detail = detail[len(self._FIX_TARGET_PREFIX):]
            self._fix_target_message = (
                "I changed the app, but could not confirm the reported "
                f"failure is fixed ({self._fix_target.descriptor}). "
                f"Outstanding issue: {detail}"
            )
            logger.warning(
                "Fix run: reported target NOT confirmed fixed — %s",
                self._fix_target.descriptor,
            )
        else:
            self._fix_target_resolved = True
            logger.info(
                "Fix run: reported target plausibly addressed — %s",
                self._fix_target.descriptor,
            )
        self._trace.write(
            EVENT_PHASE_EXIT,
            phase="fix_target_gate",
            resolved=self._fix_target_resolved,
            target=self._fix_target.descriptor,
        )

    # ==================================================================
    # Model-sync during vibe-MODIFY (class-diagram only)
    # ==================================================================

    # Common attribute-type spellings the LLM might return, mapped to the
    # B-UML primitive-type names (``PrimitiveDataType`` only accepts these).
    # Anything unrecognised falls back to ``str`` — a safe, lossless default.
    _PRIMITIVE_TYPE_ALIASES = {
        "str": "str", "string": "str", "text": "str", "varchar": "str",
        "char": "str", "uuid": "str", "email": "str", "url": "str",
        "int": "int", "integer": "int", "number": "int", "long": "int",
        "float": "float", "double": "float", "decimal": "float", "real": "float",
        "bool": "bool", "boolean": "bool",
        "datetime": "datetime", "timestamp": "datetime",
        "date": "date", "time": "time", "timedelta": "timedelta",
        "any": "any", "object": "any", "json": "any",
    }

    @staticmethod
    def _is_valid_model_name(name) -> bool:
        """Cheap pre-check mirroring the metamodel ``NamedElement`` rules.

        The name setter rejects None / empty / whitespace / spaces /
        hyphens; we filter those here so a bad LLM name is skipped
        silently instead of forcing a ValueError through the try/except.
        """
        return (
            isinstance(name, str)
            and name.strip() != ""
            and " " not in name
            and "-" not in name
        )

    def _resolve_primitive_type(self, type_str):
        """Map an LLM-supplied type string to a ``PrimitiveDataType``."""
        from besser.BUML.metamodel.structural import PrimitiveDataType

        key = type_str.strip().lower() if isinstance(type_str, str) else ""
        return PrimitiveDataType(self._PRIMITIVE_TYPE_ALIASES.get(key, "str"))

    def _derive_and_apply_model_deltas(self, instructions: str) -> None:
        """MODIFY-only: sync the domain model with the modification intent.

        Asks the orchestrator's LLM for genuinely-new domain entities
        implied by ``instructions`` (e.g. "add authentication" → a ``User``
        class), applies them to ``self.domain_model`` IN PLACE, and
        re-serialises an updated project export onto
        ``self._updated_project_export`` so the GitHub push writes
        ``buml/diagrams.json`` + ``buml/*.py`` from the UPDATED model.

        Fully guarded — this is the load-bearing safety contract:
          * Skipped entirely when there is no class diagram to sync
            (``self.domain_model is None``) or the client is a test/mock
            double (same ``_client`` gate the generator selector uses), so
            unit tests driving the full ``modify()`` loop with a scripted
            client are unaffected.
          * Any failure (LLM error, malformed delta, serialisation error)
            is swallowed: ``modify()`` proceeds EXACTLY as it does today —
            no model change beyond what already applied, no crash. An empty
            result is the common, expected case.

        NEVER invoked from ``run()`` / ``resume()`` / ``_run_phase1`` — the
        from-scratch path is untouched and byte-identical.
        """
        # Class-diagram-only MVP: nothing to sync without a domain model.
        if self.domain_model is None:
            return
        # Skip mock/duck-typed clients that don't look like a real provider
        # (mirrors _select_generator_with_llm's gate). Keeps the full
        # modify() loop deterministic under a scripted test client.
        if not hasattr(self.client, "_client"):
            return
        if not self._client_supports_structured_chat():
            return

        try:
            new_classes = self._request_model_deltas(instructions)
        except Exception:
            logger.debug(
                "modify: model-delta LLM call failed; proceeding without "
                "model sync", exc_info=True,
            )
            return

        if not new_classes:
            return

        added = self._apply_new_classes(new_classes)
        if added == 0:
            return

        logger.info("modify: model-sync added %d new class(es)", added)

        # Re-serialise the (now-mutated) model and slot it into the run's
        # original project export for the push path. A failure here still
        # leaves the domain-model mutation in place (it already improved
        # Phase 2's inventory) — only the push export falls back.
        try:
            from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
                class_buml_to_json,
            )

            updated_class_json = class_buml_to_json(self.domain_model)
            self._updated_project_export = self._build_updated_project_export(
                updated_class_json
            )
        except Exception:
            logger.warning(
                "modify: failed to serialise updated model; the push will "
                "fall back to the request's projectExport", exc_info=True,
            )
            self._updated_project_export = None

    def _request_model_deltas(self, instructions: str) -> list[dict]:
        """Structured LLM call returning ``new_classes`` implied by the edit.

        Uses the same forced-tool structured-prediction infra as
        ``_select_generator_structured``. Returns a (possibly empty) list
        of ``{"name": str, "attributes": [{"name": str, "type": str}]}``.
        """
        existing = sorted(c.name for c in self.domain_model.get_classes())
        delta_tool = {
            "name": "derive_model_deltas",
            "description": (
                "Report genuinely-new domain entities the underlying data "
                "MODEL should gain because of a modification request."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "new_classes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "attributes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                        },
                                        "required": ["name", "type"],
                                    },
                                },
                            },
                            "required": ["name"],
                        },
                    },
                },
                "required": ["new_classes"],
            },
        }
        request = user_request(instructions)
        prompt = (
            "An existing app is being MODIFIED with this instruction:\n\n"
            f"'{request}'\n\n"
            "The app's current domain model has these classes: "
            f"{', '.join(existing) if existing else '(none)'}.\n\n"
            "List ONLY genuinely NEW domain entities the MODEL should gain "
            "because of this change — e.g. adding authentication implies a "
            "User/Account entity with username/password/role. Do NOT repeat "
            "entities that already exist. Return an EMPTY list if the change "
            "is purely code-level (styling, responsiveness, routing, copy, "
            "config, performance). An empty list is the common, expected "
            "answer."
        )
        planning_model = getattr(self.client, "planning_model", None)
        response = self.client.chat(
            system=(
                "You extract new domain-model entities implied by a code "
                "modification. Call derive_model_deltas with your answer."
            ),
            messages=[{"role": "user", "content": prompt}],
            tools=[delta_tool],
            force_tool="derive_model_deltas",
            model_override=planning_model,
        )
        for block in response.get("content", []):
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "tool_use":
                continue
            payload = getattr(block, "input", None) or (
                block.get("input") if isinstance(block, dict) else None
            )
            classes = (payload or {}).get("new_classes", [])
            if isinstance(classes, list):
                return classes
        return []

    def _apply_new_classes(self, new_classes: list[dict]) -> int:
        """Apply ``new_classes`` to ``self.domain_model`` in place.

        Returns the number of classes actually added. Duplicates (name
        collides with an existing type) and invalid names are skipped
        silently; the model set setter also raises on duplicates, which
        the per-class try/except absorbs.
        """
        from besser.BUML.metamodel.structural import Class, Property

        # Existing type names (classes + enums + primitives) — adding a
        # type whose name already exists raises in the ``types`` setter.
        existing_type_names = {t.name for t in self.domain_model.types}
        added = 0
        for spec in new_classes:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not self._is_valid_model_name(name):
                continue
            if name in existing_type_names:
                continue  # skip duplicates silently
            try:
                new_cls = Class(name=name)
                for attr in spec.get("attributes") or []:
                    if not isinstance(attr, dict):
                        continue
                    attr_name = attr.get("name")
                    if not self._is_valid_model_name(attr_name):
                        continue
                    prop_type = self._resolve_primitive_type(attr.get("type"))
                    try:
                        new_cls.add_attribute(
                            Property(name=attr_name, type=prop_type)
                        )
                    except Exception:
                        # Duplicate attribute name, etc. — skip that attr.
                        continue
                self.domain_model.add_type(new_cls)
            except Exception:
                logger.debug(
                    "modify: skipped invalid model delta %r", name, exc_info=True,
                )
                continue
            existing_type_names.add(name)
            added += 1
        return added

    def _build_updated_project_export(self, updated_class_json: dict):
        """Slot ``updated_class_json`` into a copy of the run's export.

        Replaces the active ``ClassDiagram`` entry's ``model`` with the
        re-serialised class diagram. Returns ``None`` (push falls back to
        the request's projectExport) when there is no source export or it
        has no ClassDiagram entry to update.
        """
        import copy

        source = self._source_project_export
        if not isinstance(source, dict):
            return None
        diagrams = source.get("diagrams")
        if not isinstance(diagrams, dict):
            return None
        class_entries = diagrams.get("ClassDiagram")
        if not isinstance(class_entries, list) or not class_entries:
            return None

        # Resolve the active index the same way ProjectInput.get_active_diagram
        # does (currentDiagramIndices, clamped into range).
        idx = 0
        indices = source.get("currentDiagramIndices")
        if isinstance(indices, dict):
            maybe_idx = indices.get("ClassDiagram")
            if isinstance(maybe_idx, int):
                idx = maybe_idx
        idx = min(max(idx, 0), len(class_entries) - 1)

        export = copy.deepcopy(source)
        entry = export["diagrams"]["ClassDiagram"][idx]
        if not isinstance(entry, dict):
            return None
        entry["model"] = updated_class_json
        return export

    # ==================================================================
    # Phase 1: Deterministic generation (no LLM)
    # ==================================================================

    def _run_phase1(self, instructions: str) -> None:
        """Select and run the best generator, then inventory the output."""
        # Almost every BESSER generator needs a domain model. When the
        # user drove smart-generation from a state-machine / agent /
        # quantum-only project, there is nothing for Phase 1 to do —
        # skip straight to Phase 2, where the LLM writes from the
        # primary model using write_file / run_command.
        if not any((
            self.domain_model is not None,
            self.agent_model is not None,
            self.object_model is not None,
            self.quantum_circuit is not None,
            self.bpmn_model is not None,
            self.nn_model is not None,
        )):
            logger.info(
                "Phase 1: skipped (no domain_model or quantum_circuit — "
                "primary_kind=%s). LLM writes from scratch in Phase 2.",
                self.primary_kind,
            )
            if self.on_progress:
                # Surface the skip so the smart-gen card shows a `generate`
                # row with a clear "skipped — no model" message instead of
                # silently jumping from `select` to `gap`.
                self.on_progress(0, "__skipped__", "no_model")
            return

        generator_name = self._select_generator(instructions)

        if generator_name:  # non-empty string = use this generator
            logger.info("Phase 1: Running %s generator", generator_name)
            if self.on_progress:
                self.on_progress(0, generator_name, "generating")

            try:
                result = json.loads(self.executor.execute(generator_name, {}))
            except (json.JSONDecodeError, TypeError):
                result = {"status": "failed", "error": "Generator returned invalid response"}
            if result.get("status") == "ok":
                self._generator_used = generator_name
                self.executor.set_scaffold_family(self._scaffold_family())
                self.tool_calls_log.append({
                    "turn": 0, "tool": generator_name,
                    "input": {}, "success": True,
                })
                self._inventory = build_inventory(
                    self.output_dir, self.domain_model, generator_name,
                )
                logger.info("Phase 1: Generated %d files", len(result.get("files", [])))
            else:
                error_text = str(result.get("error") or "unknown error")
                self._phase1_failure_reason = f"{generator_name}: {error_text}"
                logger.warning("Phase 1: Generator failed: %s", error_text)
                # Surface the failure on the SSE stream — without this
                # the smart-gen card shows "generating" forever and the
                # user never learns why the scaffold was skipped.
                if self.on_progress:
                    self.on_progress(
                        0, generator_name, f"failed: {error_text[:120]}"
                    )
        else:
            logger.info("Phase 1: No matching generator -- LLM will write from scratch")
            if self.on_progress:
                self.on_progress(0, "__skipped__", "no_generator")

    def _select_generator(self, instructions: str = "") -> str | None:
        """
        Pick the best generator using a cheap LLM call or keyword fallback.

        Tries a quick LLM call first (if available), falls back to keyword
        matching. Respects what the user asked for — if they want NestJS,
        returns None so the LLM writes from scratch.
        """
        # A binding override (user-approved preview plan) wins outright —
        # no LLM call, no keywords. Validated upstream against the
        # registered generator tools.
        if self.target_generator_bound:
            if self.target_generator is None:
                logger.info(
                    "Phase 1: caller explicitly selected no deterministic generator"
                )
                return None
            logger.info(
                "Phase 1: using caller-specified generator %s",
                self.target_generator,
            )
            return self.target_generator

        # Hard override: an explicitly-named language/stack BESSER has no
        # generator for (Rust, C, C++, Kotlin, Go, ...) must build from scratch.
        # The LLM selector below is unreliable here — it picks the nearest
        # built-in (Python/Java) — so decide this deterministically first.
        if _names_unsupported_stack(instructions):
            logger.info(
                "Phase 1: request names an unsupported-for-BESSER stack — "
                "building from scratch (no deterministic generator)"
            )
            return None

        # LLM decides first
        llm_result = self._select_generator_with_llm(instructions)

        if llm_result and llm_result != "":
            # LLM picked a specific generator — trust it
            return llm_result

        # LLM said "none" or failed — run keywords as safety net
        keyword_result = self._select_generator_keyword(instructions)

        if keyword_result and keyword_result != "":
            # Keywords found a match — override LLM's "none"
            logger.info("Phase 1: Keywords override LLM → %s", keyword_result)
            return keyword_result

        if llm_result == "" and (keyword_result is None or keyword_result == ""):
            # Both LLM and keywords agree: no generator
            return None

        # Last resort: default based on the primary specialist model.
        if self.primary_kind == "bpmn" and self.bpmn_model is not None:
            return "generate_bpmn"
        if self.primary_kind == "nn" and self.nn_model is not None:
            lower = instructions.lower()
            if "tensorflow" in lower or "keras" in lower:
                return "generate_tensorflow"
            return "generate_pytorch"
        if self.primary_kind == "agent" and self.agent_model is not None:
            return "generate_baf"
        if self.primary_kind == "object" and self.object_model is not None:
            return "generate_json_object"

        # Legacy specialist/default ordering.
        if self.quantum_circuit is not None:
            return "generate_qiskit"
        if self.gui_model and self.domain_model is not None:
            return "generate_web_app"
        if self.domain_model is not None:
            try:
                if self.domain_model.get_classes():
                    return "generate_fastapi_backend"
            except Exception:
                pass
        return None

    def _select_generator_with_llm(self, instructions: str) -> str | None:
        """Use a cheap LLM call to pick the best generator for Phase 1."""
        try:
            from besser.generators.llm.tools import get_available_generator_names

            classes = [c.name for c in self.domain_model.get_classes()] if self.domain_model else []

            # Inventory of every available editor model so the selector LLM
            # can prefer generators that match (e.g. quantum circuit → qiskit,
            # state machines present → backend with state-pattern wiring).
            available_models = [f"Domain model: {len(classes)} classes ({', '.join(classes[:10])})"]
            available_models.append(f"GUI model: {'YES' if self.gui_model else 'NO'}")
            available_models.append(f"Agent model: {'YES' if self.agent_model else 'NO'}")
            if self.object_model is not None:
                available_models.append(
                    "Object model: YES (instance data — useful as seeders / fixtures)"
                )
            else:
                available_models.append("Object model: NO")
            if self.state_machines:
                sm_names = ", ".join(getattr(sm, "name", "?") for sm in self.state_machines[:5])
                available_models.append(
                    f"State machines: YES ({len(self.state_machines)}: {sm_names}) — "
                    "behavioural specs that should drive transition guards / event handlers"
                )
            else:
                available_models.append("State machines: NO")
            if self.quantum_circuit is not None:
                available_models.append(
                    "Quantum circuit: YES — prefer generate_qiskit for the circuit code"
                )
            else:
                available_models.append("Quantum circuit: NO")
            if self.bpmn_model is not None:
                available_models.append(
                    "BPMN model: YES - prefer generate_bpmn for executable process XML"
                )
            else:
                available_models.append("BPMN model: NO")
            if self.nn_model is not None:
                available_models.append(
                    "Neural-network model: YES - choose PyTorch or TensorFlow"
                )
            else:
                available_models.append("Neural-network model: NO")

            # Build the generator menu from the tool registry, restricted
            # to generators whose required models are actually loaded.
            # Offering an unavailable generator (e.g. generate_web_app
            # with no GUI model) lets the LLM pick it, Phase 1 fails,
            # and the run silently degrades to expensive from-scratch
            # generation — seen in production logs.
            from besser.generators.llm.tools import GENERATOR_TOOLS
            selectable_names = get_available_generator_names(
                has_domain_model=self.domain_model is not None,
                has_gui_model=self.gui_model is not None,
                has_agent_model=self.agent_model is not None,
                has_state_machines=bool(self.state_machines),
                has_quantum_circuit=self.quantum_circuit is not None,
                has_object_model=self.object_model is not None,
                has_bpmn_model=self.bpmn_model is not None,
                has_nn_model=self.nn_model is not None,
            )
            gen_lines = [
                f"- {tool['name']} → {tool['description']}"
                for tool in GENERATOR_TOOLS
                if tool["name"] in selectable_names
            ]

            prompt = (
                f"User request: {user_request(instructions)}\n\n"
                "Available editor models:\n"
                + "\n".join(f"  • {line}" for line in available_models) + "\n\n"
                "Available BESSER generators:\n"
                + "\n".join(gen_lines) + "\n"
                "- NONE → write from scratch (for frameworks BESSER doesn't support: NestJS, Next.js, Express, Spring Boot, Go, etc.)\n\n"
                "RULES:\n"
                "- Pick the generator that best covers the MAIN part of the request\n"
                "- Even if the user asks for more than one thing (backend + frontend), pick the generator for the biggest part\n"
                "- generate_fastapi_backend includes SQLAlchemy + Pydantic — don't pick those separately\n"
                "- generate_web_app includes React + FastAPI + Docker — most complete if GUI available\n"
                "- If a Quantum circuit is present and the user asks for quantum/Qiskit code → generate_qiskit\n"
                "- If state machines are present, pick the generator that fits the rest of the request — "
                "the LLM in Phase 2 will wire state transitions on top of the generator output\n"
                "- BPMN/workflow/process requests with a BPMN model: generate_bpmn\n"
                "- PyTorch/torch requests with an NN model: generate_pytorch\n"
                "- TensorFlow/Keras requests with an NN model: generate_tensorflow\n"
                "- Agent/chatbot/BAF requests with an Agent model: generate_baf\n"
                "- JSON fixture/seed requests with an Object model: generate_json_object\n"
                "- Supabase requests with a Domain model: generate_supabase\n"
                "- ONLY answer NONE if the user explicitly asks for a framework like NestJS, Next.js, Express, Spring Boot, Go, Rust\n"
                "- If the user says 'backend', 'API', 'FastAPI', or 'REST' → answer generate_fastapi_backend\n"
                "- If the user says 'Django' → answer generate_django\n"
                "- NEVER answer NONE for a Python/FastAPI/Django request\n\n"
                "Reply with ONLY the generator name or NONE. One word. Nothing else."
            )

            # Use the main client with a short response.
            # Skip if client doesn't look like a real provider (e.g. mock in tests).
            if not hasattr(self.client, '_client'):
                return None

            registered_names = selectable_names

            # Preferred path: force a choose_generator tool call so the
            # answer is exact by construction (an enum), routed to the
            # cheap planning sibling when one exists. Falls back to the
            # legacy free-text protocol for clients that don't support
            # tool_choice (older providers, gateways, duck-typed mocks).
            if self._client_supports_structured_chat():
                choice = self._select_generator_structured(prompt, registered_names)
                if choice is not None:
                    return choice
                # Structured path failed entirely — fall through to text.

            response = self.client.chat(
                system="You select the best code generator. Reply with only the generator name or NONE.",
                messages=[{"role": "user", "content": prompt}],
                tools=[],
            )

            # Extract the answer
            answer = ""
            for block in response.get("content", []):
                if hasattr(block, "text"):
                    answer = block.text.strip()
                elif isinstance(block, dict) and block.get("type") == "text":
                    answer = block["text"].strip()

            answer = answer.strip().lower().replace("`", "").replace("'", "").replace('"', '')
            logger.info("Phase 1 (LLM): Raw answer: '%s'", answer[:100])

            # Match against EVERY registered generator. Sort by name
            # length descending so longer prefixes win first (e.g.
            # ``generate_python_classes`` before ``generate_python``).
            for gen in sorted(registered_names, key=len, reverse=True):
                if gen in answer:
                    logger.info("Phase 1 (LLM): Selected %s", gen)
                    return gen

            # Only treat as "no generator" if answer is exactly "none"
            # (not just contains "none" — avoids false matches)
            if answer.strip() == "none":
                logger.info("Phase 1 (LLM): Explicitly no generator")
                return ""

            logger.warning("Phase 1 (LLM): Could not parse answer: '%s'", answer[:100])
            return None  # fall through to keyword matching

        except Exception as e:
            logger.debug("Phase 1: LLM selection skipped (%s), using keywords", e)
            return None

    def _client_supports_structured_chat(self) -> bool:
        """True when ``client.chat`` accepts force_tool / model_override."""
        import inspect
        try:
            sig = inspect.signature(self.client.chat)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return True
        return "force_tool" in params and "model_override" in params

    def _select_generator_structured(
        self, prompt: str, registered_names: list[str],
    ) -> str | None:
        """Selection via a forced choose_generator tool call.

        Returns the generator name, ``""`` for an explicit NONE, or
        ``None`` when the structured path failed (caller falls back to
        the free-text protocol).
        """
        choose_tool = {
            "name": "choose_generator",
            "description": (
                "Select the best BESSER generator for this request, or "
                "NONE to write from scratch."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "generator": {
                        "type": "string",
                        "enum": registered_names + ["NONE"],
                    },
                },
                "required": ["generator"],
            },
        }
        planning_model = getattr(self.client, "planning_model", None)
        for model_override in dict.fromkeys([planning_model, None]):
            try:
                response = self.client.chat(
                    system=(
                        "You select the best code generator. Call "
                        "choose_generator with your selection."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    tools=[choose_tool],
                    force_tool="choose_generator",
                    model_override=model_override,
                )
            except Exception as exc:
                logger.info(
                    "Phase 1 (LLM): structured selection failed on %s (%s)",
                    model_override or "primary", exc,
                )
                continue
            for block in response.get("content", []):
                block_type = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if block_type != "tool_use":
                    continue
                payload = getattr(block, "input", None) or (
                    block.get("input") if isinstance(block, dict) else None
                )
                choice = (payload or {}).get("generator", "")
                if choice == "NONE":
                    logger.info("Phase 1 (LLM): Explicitly no generator")
                    return ""
                if choice in registered_names:
                    logger.info("Phase 1 (LLM): Selected %s", choice)
                    return choice
            # No usable tool_use block (gateway ignored tool_choice?) —
            # don't retry the other model for this; bail to text protocol.
            return None
        return None

    def _select_generator_keyword(self, instructions: str) -> str | None:
        """Keyword-based generator selection. Returns generator name, '' for none, or None if undecided."""
        import re as _re
        lower = instructions.lower()

        def _has(word: str) -> bool:
            return bool(_re.search(r'\b' + _re.escape(word) + r'\b', lower))

        # Frameworks with NO BESSER generator → write from scratch
        for fw in ("nestjs", "next.js", "nextjs", "express", "spring boot",
                    "springboot", "laravel", "rails", "golang", "axum",
                    "actix", "angular", "vue", "svelte", "nuxt"):
            if _has(fw):
                return ""

        # Specialist generators for the newer modeling artifacts.
        if self.bpmn_model is not None and (
            self.primary_kind == "bpmn"
            or _has("bpmn")
            or _has("workflow")
            or "business process" in lower
            or "process model" in lower
        ):
            return "generate_bpmn"
        if self.nn_model is not None:
            if _has("tensorflow") or _has("keras"):
                return "generate_tensorflow"
            if (
                self.primary_kind == "nn"
                or _has("pytorch")
                or _has("torch")
                or "neural network" in lower
            ):
                return "generate_pytorch"
        if self.agent_model is not None and (
            self.primary_kind == "agent"
            or _has("baf")
            or _has("chatbot")
            or "agent framework" in lower
        ):
            return "generate_baf"
        if self.object_model is not None and (
            self.primary_kind == "object"
            or "json object" in lower
            or _has("fixture")
            or _has("fixtures")
            or "seed data" in lower
        ):
            return "generate_json_object"
        if self.domain_model is not None and _has("supabase"):
            return "generate_supabase"

        # Positive matches — check what user explicitly asked for
        if _has("django"):
            return "generate_django"
        if _has("fastapi") or _has("fast api"):
            if self.gui_model:
                return "generate_web_app"  # full-stack is better when GUI available
            return "generate_fastapi_backend"
        if _has("backend") or _has("api") or _has("rest"):
            if self.gui_model:
                return "generate_web_app"
            return "generate_fastapi_backend"
        if _has("full-stack") or _has("fullstack") or _has("full stack") or _has("web app"):
            if self.gui_model:
                return "generate_web_app"
            return "generate_fastapi_backend"
        if _has("pydantic") and not _has("api") and not _has("backend"):
            return "generate_pydantic"
        if _has("sqlalchemy"):
            return "generate_sqlalchemy"

        # No clear keyword match — let caller decide
        return None

    # ==================================================================
    # Phase 0.5: Stack-metadata pre-generation
    # ==================================================================

    def _run_phase0_5_metadata(self, instructions: str) -> None:
        """Pre-create a minimal build-metadata file for non-Python stacks.

        BESSER's deterministic generators only cover the Python family.
        For Next.js / Rust / Kotlin requests, the customise loop has
        historically been expected to invent ``tsconfig.json`` /
        ``Cargo.toml`` / ``build.gradle.kts`` from scratch — which it
        occasionally forgets, breaking the per-project compile check.

        This step writes a stack-appropriate manifest as a floor under
        the customise loop. The LLM is free to extend it (adding
        dependencies, scripts, etc.) but doesn't have to remember to
        create it.

        Guarantees:
          - No-op when Phase 1 actually ran a generator (Python stacks
            are byte-identical to today's output).
          - No-op when the target stack isn't one we have a template
            for (e.g. Go, Ruby, Express — left to the customise loop
            as before, until templates are added).
          - Strictly additive: if a file already exists at the target
            path, it is preserved (covers the rare case where the
            executor wrote one before Phase 0.5 ran).
        """
        # Don't second-guess BESSER's own generators. If Phase 1 ran a
        # Python generator, the manifest is already on disk and almost
        # certainly more tailored than our static template would be.
        if self._generator_used:
            return

        stack_id = detect_stack(instructions)
        if stack_id is None:
            logger.debug(
                "Phase 0.5: no recognised non-Python stack in instructions; skipping"
            )
            return

        try:
            written = pre_generate_metadata(stack_id, self.output_dir)
        except Exception as exc:  # pragma: no cover - defensive
            # A broken template must not abort the whole run — log and
            # fall through to the customise loop, which can still try
            # to author the files itself.
            logger.warning(
                "Phase 0.5 template write failed for %s: %s", stack_id, exc,
            )
            return

        if not written:
            return

        self._phase0_5_stack = stack_id
        self._phase0_5_files = list(written)
        # Tell the customise loop these files already exist. Otherwise
        # ``_inventory`` is empty for non-Python stacks (Phase 1 skipped)
        # and the LLM has no signal that the manifest is on disk —
        # so it might rewrite it from scratch, the very thing this
        # phase is here to prevent.
        bullet_files = "\n".join(f"  - {p}" for p in written)
        self._inventory = (
            f"Phase 0.5 pre-generated a minimal {stack_label(stack_id)} "
            f"project manifest:\n{bullet_files}\n\n"
            "These are MINIMAL but VALID build-config files. Build your "
            "application on top of them — read them with `read_file` and "
            "use `modify_file` to add dependencies as needed. Do NOT "
            "rewrite them from scratch."
        )
        self._trace.write(
            EVENT_PHASE_ENTER,
            phase="phase0_5",
            stack=stack_id,
            files=list(written),
        )
        self._trace.write(EVENT_PHASE_EXIT, phase="phase0_5", stack=stack_id)
        if self.on_progress:
            # Use the same "skipped" sentinel shape so the smart-gen
            # progress card stays compact — we don't want a new top-level
            # row for what is essentially a tiny scaffolding step.
            self.on_progress(
                0,
                "__metadata__",
                f"{stack_label(stack_id)}: {', '.join(written)}",
            )

    # ==================================================================
    # Adaptive response sizing for from-scratch runs
    # ==================================================================

    def _apply_adaptive_budget(self) -> None:
        """Raise only the output-token ceiling for from-scratch runs.

        Called from ``run()`` / ``resume()`` after Phase 1 (and Phase 0.5)
        have run, so ``self._generator_used`` is authoritative:

        - ``None`` means Phase 1 found nothing to run -- either there was
          no domain/quantum model at all (state-machine/agent-only
          projects), or Phase 1 explicitly decided no registered BESSER
          generator matches the request (e.g. the user asked for Next.js,
          Rust, or Kotlin -- stacks BESSER doesn't scaffold). Either way,
          Phase 2 has NOTHING to build on top of: it authors the entire
          application from nothing, which legitimately needs bigger
          individual responses than the common case (a Python scaffold
          Phase 2 only patches).
        - Anything else means a deterministic generator actually ran --
          the common, cheaper case -- so no response-size adaptation is
          needed.

        Cost and runtime are user-authorised safety rails. They are never
        changed here: a from-scratch run may need more budget, but it must
        stop at the explicit cap rather than silently spending more.
        """
        # Scaffolded runs get the wider ceiling too: a customisation turn writes
        # whole NEW files the scaffold never emitted (React pages, auth modules).
        # Observed live 2026-09-10: a scaffolded run overran 16_384 on its FIRST
        # customisation turn and Phase 2 exited with zero LLM writes.

        # Widen the per-call output-token limit: a from-scratch run
        # writes large files with no scaffold underneath them, which is
        # far more likely to hit the provider's default max_tokens
        # mid-``write_file`` than the scaffolded case. See
        # llm_client.FROM_SCRATCH_MAX_TOKENS for why raising the limit
        # (rather than chunking/continuing a truncated tool call) is the
        # chosen, lower-risk fix.
        try:
            current_max_tokens = self.client.max_tokens
        except (AttributeError, NotImplementedError):
            # Defensive: a test double / older client without the
            # max_tokens property. Don't fail the run over telemetry.
            current_max_tokens = None
        if current_max_tokens is not None and current_max_tokens < FROM_SCRATCH_MAX_TOKENS:
            logger.info(
                "Adaptive response sizing: raising output-token limit %d -> %d "
                "(generator_used=%s)",
                current_max_tokens, FROM_SCRATCH_MAX_TOKENS, self._generator_used,
            )
            self.client.max_tokens = FROM_SCRATCH_MAX_TOKENS
            self._adaptive_budget_applied = True

    def _apply_modify_budget(self) -> None:
        """Raise only the output-token ceiling for modify/fix runs.

        Called once at the start of ``modify()`` (before Phase 2). A
        modify/fix run frequently rewrites a whole existing file in a
        single ``write_file`` turn -- a targeted edit that still touches
        most of the file, or a smaller model electing a full rewrite over
        a surgical patch -- which is exactly the large-single-response
        case that overruns the client's default per-call output cap and
        truncates mid-file (see the ``stop_reason in ("max_tokens",
        "length")`` handling in ``_run_customization_loop``).

        Mirrors ``_apply_adaptive_budget`` (raises only the per-call
        output limit; the caller-authorised cost and runtime caps are
        never touched), but is keyed on the modify path rather than
        ``self._generator_used`` -- a modify run adopts the seed's
        generator name, so that "no scaffold ran" signal isn't available
        here. NEVER invoked from ``run()`` / ``resume()``, so
        first-generation output sizing (scaffolded stays at the client
        default; pure from-scratch uses ``_apply_adaptive_budget``) is
        unchanged.
        """
        try:
            current_max_tokens = self.client.max_tokens
        except (AttributeError, NotImplementedError):
            # Defensive: a test double / older client without the
            # max_tokens property. Don't fail the run over telemetry.
            current_max_tokens = None
        if current_max_tokens is not None and current_max_tokens < MODIFY_MAX_TOKENS:
            logger.info(
                "Adaptive response sizing: raising output-token limit %d -> %d for "
                "modify/fix run",
                current_max_tokens, MODIFY_MAX_TOKENS,
            )
            self.client.max_tokens = MODIFY_MAX_TOKENS
            self._adaptive_budget_applied = True

    # ==================================================================
    # Phase 1.5: Validate Phase 1 output
    # ==================================================================

    def _validate_phase1_output(self) -> list[str]:
        """
        Validate Phase 1 generator output before handing off to the LLM.

        Checks:
        - Python syntax on all .py files (ast.parse)
        - Dockerfiles reference files that actually exist

        Returns a list of issue strings to feed into gap analysis.
        """
        issues = []

        for root, _, files in os.walk(self.output_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.output_dir).replace("\\", "/")

                # Check Python syntax
                if fname.endswith(".py"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            _ast.parse(f.read(), filename=rel)
                    except SyntaxError as e:
                        issues.append(
                            f"Fix syntax error in {rel} line {e.lineno}: {e.msg}"
                        )

                # Check Dockerfiles reference files that exist
                if fname == "Dockerfile":
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read()
                        docker_dir = os.path.dirname(fpath)
                        if "requirements.txt" in content:
                            req = os.path.join(docker_dir, "requirements.txt")
                            if not os.path.isfile(req):
                                if _ensure_requirements_txt(docker_dir):
                                    logger.info(
                                        "Auto-fixed: restored missing requirements.txt for %s", rel
                                    )
                                else:
                                    issues.append(
                                        f"{rel} references requirements.txt but it doesn't exist -- "
                                        f"create it or fix the Dockerfile"
                                    )
                        if "package.json" in content or "package*.json" in content:
                            pkg = os.path.join(docker_dir, "package.json")
                            if not os.path.isfile(pkg):
                                issues.append(
                                    f"{rel} references package.json but it doesn't exist -- "
                                    f"create it or fix the Dockerfile"
                                )
                    except Exception:
                        pass

        if issues:
            logger.warning("Phase 1 validation found %d issues: %s", len(issues), issues)
        else:
            logger.info("Phase 1 validation passed -- no issues found")

        return issues

    # ==================================================================
    # Phase 2: LLM customization (scoped tasks)
    # ==================================================================

    _GITIGNORE = (
        "# Generated by BESSER\n"
        "__pycache__/\n*.py[cod]\n*.egg-info/\n"
        ".venv/\nvenv/\nenv/\n"
        "node_modules/\ndist/\nbuild/\n.next/\n"
        "*.db\n*.sqlite\n*.sqlite3\n*.db-journal\n"
        ".env\n.env.*\n"
        "*.log\n.pytest_cache/\n.DS_Store\n"
    )

    def _drop_redundant_generator_tools(self) -> None:
        """Remove sub-generator tools the chosen primary already bundles.

        No-op unless ``self._generator_used`` is a known bundling primary
        (see ``_REDUNDANT_GENERATOR_TOOLS_BY_PRIMARY``). Prevents the Phase-2
        agent from emitting duplicate ``pydantic/`` / ``sqlalchemy/`` /
        ``rest_api/`` dirs alongside the assembled ``backend/``.
        """
        redundant = _REDUNDANT_GENERATOR_TOOLS_BY_PRIMARY.get(self._generator_used or "")
        if not redundant:
            return
        before = len(self.tools)
        self.tools = [
            t for t in self.tools
            if (t.get("name") if isinstance(t, dict) else None) not in redundant
        ]
        if len(self.tools) != before:
            logger.info(
                "Phase 2: dropped %d redundant generator tool(s) already "
                "bundled by %s (avoids duplicate output dirs)",
                before - len(self.tools), self._generator_used,
            )

    def _ensure_gitignore(self) -> None:
        """Write a .gitignore into the output root if the run didn't author one."""
        try:
            path = os.path.join(self.output_dir, ".gitignore")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(self._GITIGNORE)
        except OSError:
            logger.debug("could not write .gitignore", exc_info=True)

    def _run_phase2(self, instructions: str, extra_issues: list[str] | None = None) -> None:
        """Run the Phase 2 customisation loop.

        Before starting, do ONE cheap LLM call (~$0.01) to scope the
        work into a focused task list. If that call fails, fall back to
        no checklist — the Phase 2 LLM is smart enough to plan from
        instructions alone.

        Phase 1 validator findings are passed as ``scoped_issues`` so
        they're presented as bugs to fix (separate from the user's
        feature request).
        """
        # Drop sub-generator tools already covered by the chosen primary, so
        # the agent can't scatter redundant pydantic/ sqlalchemy/ rest_api/
        # dirs next to the assembled backend/ (observed on every FastAPI run).
        self._drop_redundant_generator_tools()

        # Ship a .gitignore so a pushed/cloned repo doesn't carry caches, a
        # runtime DB, node_modules, or a leaked .env. Deterministic — no LLM.
        self._ensure_gitignore()

        scoped_issues = list(extra_issues) if extra_issues else []
        scoped_issues.extend(self._validate_app()["issues"])

        # On resume we skip gap analysis entirely — the checkpoint's
        # message history already contains whatever task-list the
        # original run established. Running a new gap-analyzer call
        # now would contradict the mid-run reasoning.
        resumed = self._resume_messages is not None
        if resumed:
            # Skip gap analysis — the checkpoint's message history already
            # contains whatever task list the original run established.
            gap_tasks: list[str] | None = None
            messages = list(self._resume_messages or [])
        else:
            gap_tasks = analyze_gaps_via_llm(
                instructions=self._planner_instructions(instructions),
                generator_used=self._generator_used,
                domain_model=self.domain_model,
                inventory=self._inventory,
                llm_client=self.client,
                on_progress=self.on_progress,
                on_phase_details=self.on_phase_details,
                generator_failure=self._phase1_failure_reason,
                modify_mode=self._modify_mode,
                workspace_files=self._workspace_file_list(),
                action_endpoints=self._expected_action_endpoints(),
            )
            # The planning call may have switched the client to its
            # outage fallback model — surface that before Phase 2 turns.
            self._notify_model_switch()
            messages = [{"role": "user", "content": instructions}]

            # Short-circuit: the planner explicitly judged the scaffold
            # sufficient (empty list — distinct from None, which means
            # the analysis failed). Only trusted when a deterministic
            # generator actually ran clean and Phase 1.5 found nothing
            # to fix; Phase 3 validation still runs as the safety net.
            if (
                gap_tasks == []
                and self._generator_used
                and not scoped_issues
                # A vibe-modify run must never skip Phase 2: the user asked
                # to add/change a feature on top of the seeded app, so an
                # empty gap list ("scaffold already covers it") is never a
                # reason to no-op here. ``_modify_mode`` is False on the
                # from-scratch path, keeping run() behaviour identical.
                and not self._modify_mode
            ):
                # Backstop: the deterministic scaffold never includes auth,
                # security, payments, email, integrations, or custom styling.
                # A weak planner sometimes returns [] even when the request
                # clearly asks for one of these — don't trust an empty list
                # in that case; run Phase 2 from the instructions instead.
                _markers = (
                    "auth", "login", "log in", "sign in", "signin",
                    "sign up", "signup", "register", "jwt", "oauth",
                    "session", "password", "secur", "authoriz", "authentic",
                    "permission", "payment", "stripe", "checkout", "email",
                    "webhook", "integrat", "upload", "theme", "styling",
                    " colour", " color",
                )
                _needs_custom = any(
                    m in (instructions or "").lower() for m in _markers
                ) or bool(self._deterministic_gap_tasks())
                if _needs_custom:
                    logger.warning(
                        "Phase 2: gap analysis returned empty, but the request "
                        "asks for something the deterministic scaffold never "
                        "produces (auth/security/custom) — running Phase 2 "
                        "anyway instead of trusting the empty checklist."
                    )
                    gap_tasks = None  # don't tell Phase 2 "no gaps were found"
                else:
                    logger.info(
                        "Phase 2: skipped — gap analysis found the %s scaffold "
                        "already covers the request",
                        self._generator_used,
                    )
                    self._phase2_exited_cleanly = True
                    if self.on_progress:
                        self.on_progress(
                            1, "__customize_skipped__",
                            "scaffold already covers the request",
                        )
                    return

        # Seed the executor's checklist from the gap tasks PLUS the
        # harness's own deterministic items (e.g. "build the frontend"
        # when a web app was asked and the scaffold has none). The LLM
        # manages it through the ``task_list`` tool and the end_turn
        # gate below refuses to finish while items are open — "done"
        # becomes "the checklist is closed", not "the model said done".
        if not resumed:
            deterministic_tasks = self._deterministic_gap_tasks()
            if deterministic_tasks:
                logger.info(
                    "Seeding %d harness-owned checklist task(s) (e.g. %r)",
                    len(deterministic_tasks),
                    (deterministic_tasks[0].get("text")
                     if isinstance(deterministic_tasks[0], dict)
                     else deterministic_tasks[0])[:80],
                )
                gap_tasks = merge_action_tasks(
                    deterministic_tasks + (gap_tasks or []), self._expected_action_endpoints(),
                )
        if gap_tasks:
            self.executor.set_tasks(gap_tasks)

        system = self._build_system_prompt(
            instructions=instructions,
            scoped_issues=scoped_issues,
            gap_tasks=gap_tasks,
        )
        system += (
            "\nAfter each coherent backend change call validate_app. It returns actual "
            "startup and data-entry failures, even when shell tools are disabled. "
            "Fix import/DDL errors first, then schema-router contracts and business "
            "behavior. Schema edits require updating all consumers and forms. "
            "Use test_api to exercise complete workflows with persisted values and negative "
            "cases from the original specification, not just health endpoints. Read routes "
            "and request schemas first. Its response references let later requests use created "
            "IDs. Assertions must follow the user specification, not the current implementation. "
            "Never announce the app is functional before verification."
        )

        _cost_warning_fired = False
        start_turn = self._resume_from_turn if resumed else 0
        # Clear resume state so a subsequent run() on the same instance
        # starts fresh.
        self._resume_from_turn = 0
        self._resume_messages = None

        last_progress_revision = self._workspace_revision()
        no_source_progress = 0
        no_information_progress = 0
        inspected: set[tuple[str, str]] = set()
        observed_plans = {self._repair_obligations_revision()}
        inspection_nudged = False
        self._phase2_inspection_handoff = ""
        for turn in range(start_turn, self.max_turns):
            self.total_turns = turn + 1
            self._trace.write(EVENT_TURN_START, turn=turn + 1)

            # -- Cooperative cancellation -------------------------------
            # The runner sets the underlying flag when a user POSTs to
            # /cancel-smart-gen/{run_id}. Bail out at the next turn
            # boundary rather than killing the worker thread.
            if self._should_continue is not None and not self._should_continue():
                logger.warning("Cancellation requested — stopping Phase 2 loop")
                self._phase2_stop_reason = "cancelled"
                break

            # -- Runtime timeout check ------------------------------------
            if self._start_time is not None:
                elapsed = time.monotonic() - self._start_time
                if elapsed > self.max_runtime_seconds:
                    logger.warning(
                        "Runtime timeout: %.1fs > %ds", elapsed, self.max_runtime_seconds,
                    )
                    self._phase2_stop_reason = "timeout"
                    break

            # -- Pre-call cost check ---------------------------------------
            # The post-call check below catches the overrun; this one
            # prevents firing ANOTHER billable request when the budget
            # is already spent (e.g. after an expensive streaming turn).
            if self.client.usage.estimated_cost > self.max_cost_usd:
                logger.warning(
                    "Cost cap already reached before turn %d: $%.4f > $%.4f",
                    turn + 1, self.client.usage.estimated_cost, self.max_cost_usd,
                )
                self._phase2_stop_reason = "cost_cap"
                break

            logger.info("LLM generation turn %d/%d", turn + 1, self.max_turns)

            messages = self._maybe_compact(messages)

            try:
                response = self._chat_with_pending_force(system, messages)
            except InvalidApiKeyError:
                # Auth failures must PROPAGATE so the runner reports INVALID_KEY,
                # not a misleading INTERNAL/api_error or a fake "incomplete
                # success" (#27 — the runner's INVALID_KEY branch was dead because
                # this blanket except swallowed it into api_error).
                raise
            except Exception as e:
                logger.error("LLM API call failed on turn %d: %s", turn + 1, e)
                self._phase2_stop_reason = "api_error"
                self._phase2_api_error = str(e)
                break

            # The call may have switched the client to its outage
            # fallback model mid-flight — surface that to the UI.
            self._notify_model_switch()

            # -- Cost cap check (after each API call) ---------------------
            current_cost = self.client.usage.estimated_cost
            if not _cost_warning_fired and current_cost > self.max_cost_usd * 0.8:
                logger.warning(
                    "Cost at 80%% of cap: $%.4f / $%.4f",
                    current_cost, self.max_cost_usd,
                )
                _cost_warning_fired = True
            if current_cost > self.max_cost_usd:
                logger.warning(
                    "Cost cap reached: $%.4f > $%.4f",
                    current_cost, self.max_cost_usd,
                )
                self._phase2_stop_reason = "cost_cap"
                break

            if response["stop_reason"] == "end_turn":
                # Checklist gate: the run is not done while gap-analysis
                # items are open. Nudge the model back to work (bounded —
                # a stubborn model that ignores two nudges is let through
                # rather than looping the user's budget away; Phase 3
                # still validates whatever state it left).
                execution_report = self._validate_app()
                if execution_report["blocker_count"]:
                    self._phase2_stop_reason = "validation_required"
                    self._phase2_exited_cleanly = False
                    break
                open_items = self.executor.open_tasks()
                _has_verified_open = any(t.get("verify") for t in open_items)
                _nudge_cap = (self._MAX_TASK_NUDGES + 2 if _has_verified_open
                              else self._MAX_TASK_NUDGES)
                if open_items and self._end_turn_task_nudges < _nudge_cap:
                    self._end_turn_task_nudges += 1
                    logger.info(
                        "end_turn with %d open checklist item(s) — nudge %d/%d",
                        len(open_items), self._end_turn_task_nudges,
                        self._MAX_TASK_NUDGES,
                    )
                    messages.append(
                        {"role": "assistant", "content": response["content"]}
                    )
                    listing = "\n".join(
                        f"  {t['id']}. {t['text']}" for t in open_items
                    )
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": (
                            "You ended the turn, but these checklist items "
                            "are still OPEN:\n"
                            f"{listing}\n"
                            "Finish each one now. If an item is already "
                            "complete, mark it with task_list(action='done', "
                            "id=N, evidence=[{id, path, quote}]) when no verifier "
                            "is attached. If required work cannot be finished, "
                            "record task_list(action='blocked', id=N, reason=...). "
                            "If the user did not ask for it, close it "
                            "honestly with task_list(action='drop', id=N, "
                            "reason=...) - never mark undone work done. End "
                            "the turn only when every item is closed."
                        )}],
                    })
                    continue
                logger.info("LLM completed after %d turns", turn + 1)
                self._phase2_exited_cleanly = True
                self._phase2_stop_reason = "completed"
                break

            if response["stop_reason"] == "tool_use":
                messages.append({"role": "assistant", "content": response["content"]})

                # Collect tool_use blocks
                tool_blocks = [
                    block for block in response["content"]
                    if hasattr(block, "type") and block.type == "tool_use" and getattr(block, "name", None)
                ]

                tool_results = self._execute_tool_blocks(tool_blocks, turn)
                messages.append({"role": "user", "content": tool_results})
                revision = self._workspace_revision()
                source_changed = revision != last_progress_revision
                if source_changed:
                    inspected.clear()
                    observed_plans.clear()
                novel_inspection = self._record_novel_inspections(tool_blocks, tool_results, inspected)
                plan = self._repair_obligations_revision()
                novel_plan = plan not in observed_plans
                observed_plans.add(plan)
                no_source_progress = 0 if source_changed else no_source_progress + 1
                no_information_progress = (0 if source_changed or novel_inspection or novel_plan
                                           else no_information_progress + 1)
                last_progress_revision = revision
                handoff = (no_information_progress >= self._PHASE2_STAGNANT_TURNS
                           or no_source_progress >= self._PHASE2_INSPECTION_TURNS)
                if not inspection_nudged and no_source_progress >= self._PHASE2_STAGNANT_TURNS:
                    inspection_nudged = True
                    messages.append({"role": "user", "content": [{"type": "text", "text": (
                        "<system-reminder>You have inspected/planned without changing source for "
                        f"{no_source_progress} turns. New successful reads are useful, but inspection is bounded. "
                        "Use the code already inspected to implement one coherent missing behavior, "
                        "or use test_api/task_list evidence to verify behavior already present. "
                        "Do not make cosmetic edits to reset progress or repeat unchanged reads. "
                        "If a prerequisite prevents progress, record the concrete blocker. "
                        "The next stage will retain unresolved work, not treat inspection as completion.</system-reminder>"
                    )}]})
                if handoff:
                    self._phase2_stop_reason = "validation_required"
                    paths = sorted({path for path, _ in inspected})
                    self._phase2_inspection_handoff = (
                        f"Phase 2 inspected {len(inspected)} distinct successful source excerpts and "
                        f"{len(observed_plans)} task/scenario states since the last source change. "
                        f"No source change for {no_source_progress} turns; no new inspection/plan for "
                        f"{no_information_progress} turns. Inspected paths: {', '.join(paths[:12]) or 'none'}"
                        + (" (additional paths omitted)" if len(paths) > 12 else "")
                        + ". This is an inspection handoff, not verification. Use the listed source and "
                        "concrete diagnostics to implement or verify the next unresolved behavior; avoid restarting a full-file tour."
                    )
                    logger.warning("Phase 2: bounded inspection/no-progress limit; switching to diagnostic-led repair")

                # Per-file modify-loop guard. After the tool_results are
                # appended, check whether the LLM has just made a streak
                # of modify_file calls on a single path. If so, inject a
                # high-salience reminder as a separate user message
                # BEFORE the next LLM call, so the model sees it at
                # response-time (not buried inside a tool_result blob).
                if self._apply_edit_loop_guards(messages, where="phase 2"):
                    break

                # Save a checkpoint at the end of every full turn so a
                # crash AFTER tool execution doesn't make the LLM
                # re-execute the same tool calls on resume. We save
                # after appending results so the rehydrated message
                # list starts cleanly with the next assistant turn.
                self._trace.write(
                    EVENT_COST_UPDATE,
                    turn=turn + 1,
                    estimated_cost_usd=float(current_cost),
                )
                self._save_checkpoint_for_turn(
                    turn=turn + 1,
                    messages=messages,
                    instructions=instructions,
                )
                if handoff:
                    break
            elif response["stop_reason"] in ("max_tokens", "length"):
                # The model hit its OUTPUT token limit mid-turn (typically a
                # large write_file). That is not a provider failure — report it
                # honestly as truncation instead of the misleading
                # "unexpected stop_reason: length" provider error. (#29)
                # The truncated tool call is intentionally NOT appended to
                # ``messages`` / executed — a partial write_file's JSON
                # arguments are usually invalid, so nothing gets written to
                # disk in a half-finished state. The checkpoint from the
                # last COMPLETE turn stays on disk (Phase 2 didn't exit
                # cleanly, see ``run()``), so the run is resumable.
                current_max_tokens = getattr(self.client, "max_tokens", None)
                logger.warning(
                    "Output token limit reached on turn %d (max_tokens=%s, "
                    "adaptive_budget_applied=%s)",
                    turn + 1, current_max_tokens, self._adaptive_budget_applied,
                )
                # Recoverable: the model can simply emit less next turn. Ending
                # Phase 2 on the FIRST truncation threw whole runs away (live
                # 2026-09-10: died on turn 1 with zero LLM writes). Feed the
                # truncation back, bounded by _MAX_TRUNCATION_RETRIES.
                if self._truncation_retries < self._MAX_TRUNCATION_RETRIES:
                    self._truncation_retries += 1
                    messages.append({"role": "user", "content": [{
                        "type": "text",
                        "text": (
                            "Your previous response was CUT OFF at the output "
                            "token limit"
                            + (f" ({current_max_tokens} tokens)"
                               if current_max_tokens else "")
                            + ", so it was discarded and NOTHING was written to "
                            "disk. Do not repeat it as-is. Emit a SMALLER turn: "
                            "one file per tool call, and at most one or two tool "
                            "calls in this turn. Continue from where you left "
                            "off — re-state only what you still need to write."
                        ),
                    }]})
                    logger.warning(
                        "Output truncation recovery %d/%d — asking for a smaller turn",
                        self._truncation_retries, self._MAX_TRUNCATION_RETRIES,
                    )
                    continue

                self._phase2_stop_reason = "api_error"
                self._phase2_api_error = (
                    "The model hit its output token limit"
                    + (f" ({current_max_tokens} tokens)" if current_max_tokens else "")
                    + f" on {self._truncation_retries + 1} consecutive turns, so "
                    "the generated code may be truncated. The run can be resumed "
                    "to continue from the last completed step, or try a smaller "
                    "scope / fewer files per run."
                )
                break
            else:
                logger.warning("Unexpected stop_reason: %s", response["stop_reason"])
                self._phase2_stop_reason = "api_error"
                self._phase2_api_error = f"unexpected stop_reason: {response['stop_reason']}"
                break

    @staticmethod
    def _record_novel_inspections(tool_blocks, tool_results, seen) -> bool:
        """Credit successful new read output, not new arguments or failed reads."""
        blocks = {block.id: block for block in tool_blocks}
        novel = False
        for result in tool_results:
            block = blocks.get(result.get("tool_use_id"))
            if block is None or block.name != "read_file" or not isinstance(block.input, dict):
                continue
            try:
                payload = json.loads(result.get("content", ""))
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict) or payload.get("error") or not isinstance(payload.get("content"), str):
                continue
            content = payload["content"]
            path = block.input.get("path")
            if not content.strip() or not isinstance(path, str):
                continue
            path = os.path.normcase(os.path.normpath(path.replace("\\", "/"))).replace("\\", "/")
            key = (path, hashlib.sha256(content.encode("utf-8")).hexdigest())
            if key not in seen and len(seen) < 1000:
                seen.add(key)
                novel = True
        return novel

    # Keys worth keeping in the trace when a tool reports them. From a real
    # post-mortem (2026-09-11): a run burned 11 of 80 turns on failed
    # modify_file calls and the trace recorded only ``status: error`` with no
    # reason, so the failures could not be diagnosed afterwards at all.
    _TRACE_DIAG_TEXT = ("error", "note", "advice", "warning", "matched_by",
                        "did_you_mean", "diagnostic_message", "rejection_kind", "edit_recovery")
    _TRACE_DIAG_MAX_CHARS = 400

    def _emit_progress(self, turn: int, tool: str, status: str,
                       detail: str | None = None) -> None:
        """Fire ``on_progress``, tolerating a callback that predates `detail`.

        ``on_progress`` is public API of a published package, so a caller may
        still supply the original three-argument callback. Passing four
        arguments unconditionally raised TypeError for them; the detail is a
        diagnostic nicety and must never break a run.
        """
        if not self.on_progress:
            return
        if not getattr(self, "_progress_takes_detail", True):
            self.on_progress(turn, tool, status)
            return
        try:
            self.on_progress(turn, tool, status, detail)
        except TypeError:
            self._progress_takes_detail = False
            try:
                self.on_progress(turn, tool, status)
            except Exception:
                logger.debug("on_progress callback raised; continuing", exc_info=True)
        except Exception:
            logger.debug("on_progress callback raised; continuing", exc_info=True)

    @classmethod
    def _trace_diagnostics(cls, result: str) -> dict:
        """Bounded diagnostic fields from a tool result, for the trace.

        Deliberately truncated: a trace line must stay greppable, and
        ``did_you_mean`` can carry a whole file excerpt. Never raises — a
        malformed result must not break the run it is describing.
        """
        try:
            obj = json.loads(result)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        if not isinstance(obj, dict):
            return {}
        diag: dict = {}
        for key in cls._TRACE_DIAG_TEXT:
            value = obj.get(key)
            if value:
                text = value if isinstance(value, str) else str(value)
                if len(text) > cls._TRACE_DIAG_MAX_CHARS:
                    text = text[:cls._TRACE_DIAG_MAX_CHARS] + "…[truncated]"
                diag[key] = text
        # Per-write diagnostics are a list; the COUNT is the useful signal,
        # the bodies are already in the tool_result the model saw.
        findings = obj.get("diagnostics")
        if isinstance(findings, list) and findings:
            diag["diagnostics_count"] = len(findings)
        return diag

    # Tools whose effect is a WRITE to a specific path. Two of these on the
    # same path inside one turn must not run concurrently: each does
    # read -> transform -> write, so racing them silently drops the earlier
    # edit (last writer wins). The tool description used to invite exactly
    # that ("different sections ... in the SAME turn ... in parallel").
    _WRITE_TOOLS = frozenset({"modify_file", "replace_file_lines", "write_file", "delete_file"})

    def _serial_key(self, block) -> str:
        """Group key for execution: writes to one path share a key (so they run
        in order), everything else gets a unique key (so it stays parallel)."""
        name = getattr(block, "name", "")
        args = getattr(block, "input", None)
        if name in self._WRITE_TOOLS and isinstance(args, dict):
            path = args.get("path")
            if isinstance(path, str) and path.strip():
                normalized = os.path.normcase(
                    os.path.normpath(path.replace("\\", "/").strip())
                ).replace("\\", "/")
                return "path:" + normalized
        return "id:" + str(getattr(block, "id", id(block)))

    def _execute_tool_blocks(self, tool_blocks: list, turn: int) -> list[dict]:
        # Set before any block runs so every tool_call event in this turn can
        # report it. A model that emits ONE call per turn needs ~4x the turns of
        # one that batches, which makes MAX_TURNS mean wildly different things
        # per model — invisible until this is recorded (see 2026-09-11: 80 turns,
        # 80 calls, 18 files).
        self._blocks_in_turn = len(tool_blocks)
        """
        Execute tool call blocks, in parallel where that is SAFE.

        Blocks are grouped by write target: calls that write the same path run
        sequentially in the order the model emitted them, while independent
        groups still run concurrently. Without this, two ``modify_file`` calls
        on one file in a single turn each read the pre-turn content and the
        second write overwrites the first edit.

        Args:
            tool_blocks: List of tool_use content blocks from the LLM response.
            turn: Current turn number.

        Returns:
            List of tool_result dicts, ordered to match ``tool_blocks``.
        """
        if not tool_blocks:
            return []

        if len(tool_blocks) == 1:
            return [self._execute_single_tool(tool_blocks[0], turn)]

        # Validation must see the completed batch, not race an in-flight edit.
        validation_names = {"validate_app", "test_api", "task_list"}
        validations = [b for b in tool_blocks if b.name in validation_names]
        if validations:
            results = self._execute_tool_blocks([b for b in tool_blocks if b.name not in validation_names], turn)
            results.extend(self._execute_single_tool(b, turn) for b in validations)
            order = {b.id: index for index, b in enumerate(tool_blocks)}
            return sorted(results, key=lambda result: order[result["tool_use_id"]])

        groups: dict[str, list] = {}
        for block in tool_blocks:
            groups.setdefault(self._serial_key(block), []).append(block)

        serialized = sum(1 for blocks in groups.values() if len(blocks) > 1)
        if serialized:
            logger.info(
                "Executing %d tool calls in %d group(s); %d group(s) serialized "
                "(same write target)", len(tool_blocks), len(groups), serialized,
            )
        else:
            logger.info("Executing %d tool calls in parallel", len(tool_blocks))

        def _run_group(blocks: list) -> list[dict]:
            # Sequential within a group so same-path edits compose. Keep one
            # result per call even if tracing/progress code around a tool raises
            # unexpectedly; a sibling result must never disappear.
            results: list[dict] = []
            for block in blocks:
                try:
                    results.append(self._execute_single_tool(block, turn))
                except Exception as exc:
                    logger.exception(
                        "Tool orchestration failed for %s",
                        getattr(block, "name", "?"),
                    )
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": f"Execution failed: {exc}"}),
                    })
            return results

        tool_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_WORKERS) as pool:
            futures = {pool.submit(_run_group, blocks): blocks
                       for blocks in groups.values()}
            for future in as_completed(futures):
                blocks = futures[future]
                try:
                    tool_results.extend(future.result())
                except Exception as e:
                    logger.error("Parallel tool execution failed for %s: %s",
                                 getattr(blocks[0], "name", "?"), e)
                    for block in blocks:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"error": f"Execution failed: {e}"}),
                        })

        # Restore the model's block order so tool_result pairing is stable.
        block_id_order = {block.id: i for i, block in enumerate(tool_blocks)}
        tool_results.sort(key=lambda r: block_id_order.get(r["tool_use_id"], 0))
        return tool_results

    def _execute_single_tool(self, block, turn: int) -> dict:
        """Execute a single tool call and return the tool_result dict."""
        tool_name = block.name
        logger.info("Executing tool: %s", tool_name)

        if tool_name not in _READONLY_TOOLS:
            # Loop detection keys on (tool, path) when the tool targets a
            # file: a from-scratch run legitimately calls write_file
            # dozens of times in a row on DIFFERENT paths — that is
            # progress, not a loop. Same tool on the SAME path (or a
            # path-less tool like run_command repeated verbatim by name)
            # still trips the guard.
            loop_key = tool_name
            if isinstance(block.input, dict):
                raw_loop_path = block.input.get("path")
                if isinstance(raw_loop_path, str) and raw_loop_path.strip():
                    loop_key = f"{tool_name}:{raw_loop_path.replace(chr(92), '/').strip()}"
        elif tool_name == "task_list":
            # Bookkeeping is not a loop when it works, which is why task_list
            # is read-only for loop purposes. Nine consecutive REFUSED calls
            # is a loop: run n_6i2i5r spent turns 12-20 marking tasks 9-17
            # done, one per turn, every one rejected for supplying no evidence
            # at all. _is_stuck only fires when every call in the window
            # failed, so a healthy batch still never trips it.
            loop_key = "task_list"
        else:
            loop_key = None

        # Track (tool, path) for the per-file modify streak guard. We record
        # ALL tools so that unrelated work between modify calls breaks the
        # streak. ``read_file`` captures its path too: re-reading the very
        # file you have just failed to edit is part of the flail, not a
        # break from it, and treating it as a break is what let run
        # a5dce952 alternate modify/read on one file for 38 pairs.
        # ``replace_file_lines`` counts as an edit: the recovery ladder
        # steers a flailing model straight into it, so recording it as a
        # path-less tool would let reaching recovery disarm this guard.
        # Every other tool gets ``path=None`` and so trips the check in
        # ``_consecutive_modify_on_same_file``.
        target_path = None
        if tool_name in _EDIT_STREAK_TOOLS and isinstance(block.input, dict):
            raw_path = block.input.get("path")
            if isinstance(raw_path, str):
                # Normalise for stable comparison across mixed
                # separators (Windows ``\`` vs POSIX ``/``).
                target_path = raw_path.replace("\\", "/").strip()
        self._recent_modify_targets.append((tool_name, target_path))
        # Room for the threshold's worth of modify calls PLUS an interleaved
        # read after each, so the alternating shape still fits the window.
        _window = self._PER_FILE_MODIFY_THRESHOLD * 4
        if len(self._recent_modify_targets) > _window:
            self._recent_modify_targets = self._recent_modify_targets[-_window:]

        if self.on_progress:
            # `detail` carries WHAT the call was about plus this turn's batch
            # count; the durable run store keeps only the stream.
            self._emit_progress(
                turn + 1, tool_name, "executing",
                _tool_call_detail(tool_name, block.input,
                                  getattr(self, "_blocks_in_turn", 1)),
            )

        execution = self.executor.execute_typed(tool_name, block.input)
        # A landed edit may have satisfied a checkable item. Close it here
        # rather than making the model spend a turn arguing for it: run
        # ys4gfj4v made 22 task_list calls against 9 edits, 12 refused.
        if execution.succeeded and tool_name in _WRITE_TOOLS_ON_RECORD:
            closed = self.executor.autoclose_verified_tasks()
            if closed:
                execution.payload["checklist_closed"] = closed
                logger.info("Auto-closed %d verified checklist item(s): %s",
                            len(closed), [item["id"] for item in closed])
        recovery = execution.payload.get("edit_recovery", {})
        if recovery.get("next_tool") in {"read_file", "replace_file_lines", "modify_file"}:
            self._force_tool_next = recovery["next_tool"]
        result = execution.to_json()
        success = execution.succeeded

        if loop_key is not None:
            self._recent_tool_calls.append((loop_key, success))
            # Cap the ring buffer so long runs don't leak memory. Keeping
            # 2x the loop threshold is plenty — _is_stuck() only checks
            # the tail.
            if len(self._recent_tool_calls) > self._LOOP_THRESHOLD * 2:
                self._recent_tool_calls = self._recent_tool_calls[-self._LOOP_THRESHOLD * 2 :]

        if loop_key is not None and not success and self._is_stuck():
            logger.warning("Possible loop: %s", tool_name)
            try:
                result_obj = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                result_obj = result
            if not isinstance(result_obj, dict):
                result_obj = {"result": result_obj}
            result_obj["warning"] = (
                f"'{tool_name}' has failed {self._LOOP_THRESHOLD} times in a row "
                "on the same target. Do not repeat it; take a different action."
                if tool_name != "task_list" else
                f"{self._LOOP_THRESHOLD} task_list calls in a row were refused. "
                "Marking an item done is bookkeeping, not progress, and the "
                "checklist will keep refusing items you have not implemented. "
                "Stop closing items: open the file named by the next unresolved "
                "task and make the change, then cite it."
            )
            result = json.dumps(result_obj)

        if not success:
            try:
                failure = json.loads(result)
            except (TypeError, ValueError):
                failure = {}
            self._recent_tool_failures.append({
                "tool": tool_name,
                "path": block.input.get("path", "") if isinstance(block.input, dict) else "",
                "error": str(failure.get("error", execution.status))[:600],
                "rejection_kind": str(failure.get("rejection_kind", ""))[:80],
            })
            self._recent_tool_failures = self._recent_tool_failures[-8:]

        self.tool_calls_log.append({
            "turn": turn + 1, "tool": tool_name,
            "input": _sanitize_for_log(block.input),
            "success": success,
            "status": execution.status,
        })
        if tool_name in _WRITE_TOOLS_ON_RECORD and self._trace.path:
            self._record_full_tool_input(turn + 1, tool_name, block.input, success, execution.status)
        self._trace.write(
            EVENT_TOOL_CALL,
            turn=turn + 1,
            tool=tool_name,
            success=success,
            status=execution.status,
            input=_sanitize_for_log(block.input),
            # How many calls the model batched into this turn (1 = no batching).
            blocks_in_turn=getattr(self, "_blocks_in_turn", 1),
            # WHY it failed, and which edit tier matched when it succeeded —
            # see _trace_diagnostics for why this is not optional.
            **self._trace_diagnostics(result),
        )

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        }

    # ==================================================================
    # Checkpointing
    # ==================================================================

    def _save_checkpoint_for_turn(
        self,
        turn: int,
        messages: list[dict],
        instructions: str,
        *,
        phase: str = "phase2",
    ) -> None:
        """Persist mid-run state so a crash after ``turn`` can recover.

        No-op when checkpointing was disabled in the constructor (tests).
        Failures are logged at debug level — instrumentation must not
        break the run.
        """
        if not self._checkpointing_enabled:
            return
        try:
            from besser.generators.llm.checkpoint import api_scenario_snapshot
            if phase not in {"phase2", "phase3"}:
                raise ValueError("Unknown checkpoint phase")
            self._checkpoint_phase = phase
            ckpt = Checkpoint(
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                run_id=self.run_id,
                instructions=instructions,
                primary_kind=self.primary_kind,
                turn=turn,
                total_turns=self.total_turns,
                messages=messages if phase == "phase2" else [],
                tool_calls_log=self.tool_calls_log,
                validation_issues=[
                    {"severity": i.severity, "message": i.message}
                    for i in self._validation_issues
                ],
                inventory=self._inventory,
                generator_used=self._generator_used,
                estimated_cost_usd=float(self.client.usage.estimated_cost),
                compaction_count=self._compaction_count,
                project_fingerprint=self._project_fingerprint,
                saved_at=time.time(),
                tasks=self.executor.task_snapshot(),
                api_scenarios=api_scenario_snapshot(self._api_scenarios.values()),
                phase=phase,
                source_revision=self._workspace_revision() if phase == "phase3" else "",
                phase2_stop_reason=self._phase2_stop_reason,
                phase2_exited_cleanly=self._phase2_exited_cleanly,
                repair_progress=self._repair_progress if phase == "phase3" else {},
            )
            path = save_checkpoint(self.output_dir, ckpt)
            if path:
                self._trace.write(EVENT_CHECKPOINT, turn=turn, phase=phase, path=path)
        except Exception as exc:
            logger.debug("Checkpoint write failed on turn %d: %s", turn, exc)

    def _save_phase3_checkpoint(self) -> None:
        """Persist repair state without replaying a repair transcript on resume."""
        self._checkpoint_phase = "phase3"
        self._save_checkpoint_for_turn(
            self.total_turns, [], self._instructions, phase="phase3",
        )

    def _finish_checkpoint(self) -> None:
        """Keep recovery state until both customization and validation finished."""
        if (self._phase2_exited_cleanly and not self._phase3_interrupted
                and not any(is_completion_issue(issue) for issue in self._validation_issues)):
            delete_checkpoint(self.output_dir)
        elif self._checkpoint_phase == "phase3":
            self._save_phase3_checkpoint()
        elif self._checkpointing_enabled:
            # An unfinished Phase 2 must remain resumable as Phase 2. Refresh
            # costs/tasks after validation without replacing its conversation.
            checkpoint = load_checkpoint(self.output_dir)
            if checkpoint is not None:
                self._save_checkpoint_for_turn(
                    checkpoint.turn, checkpoint.messages, self._instructions,
                )

    # ==================================================================
    # Phase 3: Post-generation validation & fix
    # ==================================================================

    def _complete_repair_if_verified(self) -> None:
        if self._phase2_stop_reason == "validation_required":
            self._phase2_exited_cleanly = True
            self._phase2_stop_reason = "completed"

    def _phase3_stop_requested(self, *, check_turn_budget: bool = True) -> str | None:
        """One stop gate for repair turns and the validation calls between them."""
        reason = None
        if self._start_time is not None and time.monotonic() - self._start_time > self.max_runtime_seconds:
            reason = "runtime budget exhausted"
        elif self.max_cost_usd is not None and self.client.usage.estimated_cost >= self.max_cost_usd:
            reason = "cost budget exhausted"
        elif check_turn_budget and self.total_turns >= self.max_turns:
            # No further edit turn, but final verification of the last accepted
            # tool batch may still run. Do not poison that check as cancelled.
            return "turn budget exhausted"
        elif self._should_continue is not None and not self._should_continue():
            reason = "cancellation requested"
        elif self._phase3_interrupted:
            reason = "repair interrupted before verification completed"
        if reason:
            self._phase3_interrupted = True
        return reason

    def _repair_obligations_revision(self) -> str:
        """Test corrections and checklist evidence are progress without source edits."""
        payload = {"tasks": self.executor.task_snapshot(),
                   "scenarios": [record["scenario"] for record in self._api_scenarios.values()]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _run_phase3_validation(self) -> None:
        """
        Lightweight validation of generated output. If issues found,
        give the LLM a few turns to fix them.

        Checks (no network, no Docker, instant unless the toolchain
        runs — tsc / cargo / kotlinc each have their own timeout):
        - Python syntax on all .py files
        - Dockerfiles reference files that exist
        - package.json exists if Dockerfile uses npm
        - npm ci -> npm install (common LLM mistake)
        - ``ruff`` lint (if installed)
        - the app booted in a subprocess: mappers configure, and every
          entity can be created through its own create endpoint
        - ``tsc --noEmit`` on every tsconfig (if tsc installed)
        - ``cargo check`` on every Cargo.toml (if cargo installed)
        - ``kotlinc`` on every Kotlin source root (if kotlinc installed)

        Per-project failures feed a repair/recheck loop bounded by the
        remaining turn, cost and runtime budgets. Two consecutive unchanged
        or repeated source states stop retries; unresolved blockers remain
        explicitly incomplete, never accepted as verified output. This
        closes the gap where Phase 3 used to surface tsc errors as
        warnings (no fix attempt) and never invoked cargo / kotlinc
        at all, leaving the per-project compile-pass at 0/n for TS /
        Rust / Kotlin runs.
        """
        # The final allowed editing turn still deserves validation. The turn
        # cap prevents another repair interaction, not local checks/final review.
        stop_reason = self._phase3_stop_requested(check_turn_budget=False)
        if stop_reason:
            logger.warning("Skipping Phase 3 -- %s", stop_reason)
            self._validation_issues.append(ValidationIssue(
                "blocker", f"requirement unverified: Phase 3 validation did not run: {stop_reason}. "
                "The current application has not completed final verification.",
            ))
            if self._checkpoint_phase == "phase3":
                self._save_phase3_checkpoint()
            return

        issues = self._collect_validation_issues()

        if not issues:
            logger.info("Phase 3: Validation passed -- no issues found")
            self._validation_issues = []
            self._complete_repair_if_verified()
            if self._checkpoint_phase == "phase3":
                self._save_phase3_checkpoint()
            return

        # Always record everything in the recipe — severity decides what
        # gets fixed automatically.
        blockers_before = [i for i in issues if i.severity == "blocker"]
        warnings_before = [i for i in issues if i.severity == "warning"]
        styles_before = [i for i in issues if i.severity == "style"]
        logger.warning(
            "Phase 3: Found %d issues (%d blocker / %d warning / %d style)",
            len(issues), len(blockers_before), len(warnings_before), len(styles_before),
        )
        for issue in issues:
            logger.warning("  [%s] %s", issue.severity, issue.message)
        self._validation_issues = list(issues)
        if not blockers_before:
            self._complete_repair_if_verified()
        if self._checkpoint_phase == "phase3":
            self._save_phase3_checkpoint()

        if self.on_progress:
            self.on_progress(
                self.total_turns,
                "validation",
                f"{len(blockers_before)} blockers / {len(issues)} total",
            )

        # Auto-fix is opt-in (default off). Industry pattern: report by
        # default, fix on request. Avoid the LLM running ``npm install`` —
        # post-install hooks execute arbitrary code from chosen packages.
        if not self.auto_fix_issues:
            logger.info(
                "Phase 3: auto_fix_issues=False — issues recorded, no LLM fix loop."
            )
            return

        # Auto-fix only consumes BLOCKER issues. Style warnings (unused
        # imports, line length) and soft warnings (tsc type hints) are
        # left as-is; they don't justify burning LLM turns.
        if not blockers_before:
            logger.info(
                "Phase 3: auto_fix_issues=True but no blocker-class issues — "
                "skipping LLM fix loop. %d non-blocker issue(s) recorded.",
                len(warnings_before) + len(styles_before),
            )
            return

        # Repair within the remaining budgets. Fixing an upstream failure can
        # expose downstream failures, so blocker counts are not a progress
        # metric. Two unchanged/repeated source states stop an unproductive loop.
        current_blockers = blockers_before
        prev_blocker_count = len(blockers_before)
        # Only a repair that actually wrote something can be rolled back.
        source_ever_changed = False
        last_issues = list(issues)
        progress = self._repair_progress
        attempts_run = progress.get("attempts_run", 0)
        last_validated_revision = self._workspace_revision()
        last_obligations_revision = self._repair_obligations_revision()
        same_validated_state = (
            progress.get("last_validated_revision") == last_validated_revision
            and progress.get("last_obligations_revision") == last_obligations_revision
        )
        no_progress_streak = progress.get("no_progress_streak", 0) if same_validated_state else 0
        seen_states = {
            (source, obligations, tuple(messages))
            for source, obligations, messages in progress.get("seen_states", [])
        }

        def checkpoint_progress() -> None:
            self._repair_progress = {
                "attempts_run": attempts_run,
                "no_progress_streak": no_progress_streak,
                "seen_states": [
                    [source, obligations, list(messages)]
                    for source, obligations, messages in sorted(seen_states)
                ],
                "last_validated_revision": last_validated_revision,
                "last_obligations_revision": last_obligations_revision,
            }
            self._save_phase3_checkpoint()

        max_attempts = max(_MAX_TOOLCHAIN_FIX_ITERATIONS, self.max_turns - self.total_turns)
        for _ in range(max_attempts):
            stop_reason = self._phase3_stop_requested()
            if stop_reason:
                logger.warning("Phase 3: %s; preserving unresolved findings", stop_reason)
                break
            is_first_attempt = attempts_run == 0
            attempts_run += 1
            self._trace.write(
                EVENT_PHASE_ENTER,
                phase="phase3_fix_attempt",
                attempt=attempts_run,
                blockers=len(current_blockers),
            )
            revision_before = self._workspace_revision()
            obligations_before = self._repair_obligations_revision()
            checkpoint_progress()
            edits = self._invoke_phase3_fix_loop(current_blockers, is_first_attempt)
            source_changed = revision_before != self._workspace_revision()
            source_ever_changed = source_ever_changed or source_changed
            obligations_changed = obligations_before != self._repair_obligations_revision()
            checkpoint_progress()
            if not edits:
                # The one line that was missing from run 7f918e11's log: the
                # attempt burned its turns and changed nothing.
                logger.warning(
                    "Phase 3: attempt %d ended with no successful edit (verification changed=%s)",
                    attempts_run, obligations_changed,
                )

            # A stop during a repair is not permission to issue another paid
            # coverage judgment. Preserve last-known findings and the checkpoint.
            if self._phase3_stop_requested(check_turn_budget=False):
                break
            # Re-validate. The bench's per-project compile-pass score
            # only cares about a clean toolchain, so re-running these
            # is what actually drives the metric.
            issues_after = self._collect_validation_issues()
            last_issues = issues_after
            self._validation_issues = list(issues_after)
            last_validated_revision = self._workspace_revision()
            last_obligations_revision = self._repair_obligations_revision()
            blockers_after = [i for i in issues_after if i.severity == "blocker"]
            self._trace.write(
                EVENT_PHASE_EXIT,
                phase="phase3_fix_attempt",
                attempt=attempts_run,
                blockers_remaining=len(blockers_after),
                source_changed=source_changed,
                successful_writes=edits or 0,
            )

            if not blockers_after:
                logger.info(
                    "Phase 3: All blockers fixed after %d attempt(s) "
                    "(%d non-blocker remain).",
                    attempts_run, len(issues_after),
                )
                self._validation_issues = list(issues_after)
                self._complete_repair_if_verified()
                checkpoint_progress()
                return

            # Fixing one import can expose several previously unreachable CRUD
            # errors. A larger count is not evidence of regression. Continue on
            # new source states; only unchanged/repeated states count as stalls.
            state = (
                last_validated_revision, last_obligations_revision,
                tuple(i.message for i in _hard_blockers(blockers_after)),
            )
            if (not source_changed and not obligations_changed) or state in seen_states:
                # A single stalled round can need another pass, so don't
                # bail immediately: retry while we still have cost budget,
                # and only give up after two consecutive no-progress rounds
                # (or when the cost cap leaves nothing to retry with). The
                # outer attempt cap still bounds the worst case.
                no_progress_streak += 1
                budget_left = (
                    self.max_cost_usd is None
                    or self.client.usage.estimated_cost < self.max_cost_usd
                )
                if no_progress_streak >= 2 or not budget_left:
                    logger.warning(
                        "Phase 3: Attempt %d made no progress (%d -> %d "
                        "blockers); ending fix loop (%d consecutive "
                        "no-progress round(s), budget_left=%s).",
                        attempts_run, prev_blocker_count,
                        len(blockers_after), no_progress_streak, budget_left,
                    )
                    break
                logger.info(
                    "Phase 3: Attempt %d made no progress (%d -> %d "
                    "blockers); retrying once more (budget remains).",
                    attempts_run, prev_blocker_count, len(blockers_after),
                )
                # Re-attempt against the current state on the next round.
                current_blockers = blockers_after
                checkpoint_progress()
                continue

            # Progress this round: reset the stall counter and keep going.
            seen_states.add(state)
            no_progress_streak = 0
            prev_blocker_count = len(blockers_after)
            current_blockers = blockers_after
            checkpoint_progress()

        # We get here either by ending the loop early (no progress)
        # or by exhausting the attempt cap. Record whatever the final
        # state is so the recipe surfaces it.
        if self._rollback_phase3_if_worse(blockers_before, last_issues, source_ever_changed):
            last_issues = list(self._validation_issues)
        else:
            self._validation_issues = list(last_issues)
        checkpoint_progress()
        remaining_blockers = [
            i for i in last_issues if i.severity == "blocker"
        ]
        if remaining_blockers:
            logger.warning(
                "Phase 3: %d blocker(s) remain after %d attempt(s); "
                "preserving partial output, not verified completion.",
                len(remaining_blockers), attempts_run,
            )
            for issue in last_issues:
                logger.warning("  [%s] %s", issue.severity, issue.message)

    # Blocker classes that mean the application does not start at all: the
    # ORM will not map, a module will not import, a name is undefined. Unlike
    # a missing feature these are never a reasonable price for a repair.
    _STARTUP_BLOCKER_PREFIXES = (
        "mapper config:", "application startup:", "python contract:",
        "missing module:", "undefined name:",
    )

    @classmethod
    def _startup_blockers(cls, issues: list[ValidationIssue]) -> set[str]:
        """The startup-class blocker messages present in ``issues``."""
        return {i.message for i in issues
                if i.message.lower().startswith(cls._STARTUP_BLOCKER_PREFIXES)}

    def _rollback_phase3_if_worse(
        self, entry_blockers: list[ValidationIssue],
        final_issues: list[ValidationIssue],
        source_changed: bool,
    ) -> bool:
        """Ship the pre-Phase-3 tree when repair ended worse than it began.

        Three conditions, and all are needed. The repair must have actually
        written something: a blocker that appears while nothing was edited is
        newly-exposed truth or judge variance, and rolling back would hide it
        (there would also be nothing to undo). A rising count *during* the
        loop is expected, since fixing an import exposes the errors behind it,
        so only the final state counts. And only hard blockers count, because
        two judge passes on one app returned 12 then 22 missing requirements.

        Run trilraak entered Phase 3 with 11 blockers; attempt 2 wrote six
        files, added an association table using ``Table`` without importing
        it, took the count to 45, and the run shipped that: ``sql_alchemy.py``
        no longer imported, so every router that star-imports it was dead.
        ``_restore_snapshot`` existed and was unit-tested, but nothing in
        production ever called it.

        The code is reverted, the findings are not: what the repaired tree
        revealed is recorded, so a real defect a partial fix exposed does not
        become invisible again.
        """
        if not source_changed:
            return False
        final_blockers = [i for i in final_issues if i.severity == "blocker"]
        entry_hard = len(_hard_blockers(entry_blockers))
        final_hard = len(_hard_blockers(final_blockers))
        # A count cannot see a TRADE. Run mbzbzhq9 held 13 blockers flat across
        # six attempts while swapping a hard blocker for a broken ORM mapper,
        # so "not more than we started with" was true and the run shipped an
        # app whose every endpoint returned 500. Introducing a blocker that
        # stops the app starting is never an acceptable trade, at any count.
        broke_startup = (self._startup_blockers(final_blockers)
                         - self._startup_blockers(entry_blockers))
        if final_hard <= entry_hard and not broke_startup:
            return False
        if broke_startup:
            logger.warning(
                "Phase 3 introduced %d blocker(s) that stop the app starting: %s",
                len(broke_startup), "; ".join(sorted(broke_startup))[:300],
            )
        logger.warning(
            "Phase 3 ended worse than it began (%d -> %d hard blockers); "
            "restoring the pre-Phase-3 tree.", entry_hard, final_hard,
        )
        if not self._restore_snapshot():
            logger.error(
                "Phase 3 regressed but the snapshot could not be restored; "
                "keeping the repaired tree and reporting it as it stands.",
            )
            self._validation_issues = list(final_issues)
            return False
        self._phase3_rolled_back = True
        # The tree went back; the checklist did not. Anything whose verifier
        # no longer passes was undone by this restore and must stop reporting
        # itself complete.
        reopened = self.executor.reopen_unverifiable_tasks()
        if reopened:
            logger.warning(
                "Phase 3 rollback discarded the implementation of %d checklist "
                "item(s), now reopened: %s", len(reopened), reopened,
            )
        restored = self._collect_validation_issues()
        discarded = sorted({i.message for i in final_blockers})[:10]
        undone = (f" The restore also undid completed work: {len(reopened)} checklist "
                  f"item(s) verified during the repair are open again." if reopened else "")
        self._validation_issues = list(restored) + [_classify_issue(
            "validation: the Phase 3 repair was rolled back - it ended with "
            f"{final_hard} hard blockers against {entry_hard} on entry, so the "
            f"pre-repair output is what ships.{undone} Findings seen only in the "
            "discarded tree (they may still be real): " + "; ".join(discarded)
        )]
        restored_blockers = [i for i in restored if i.severity == "blocker"]
        self._trace.write(
            EVENT_ROLLBACK, phase="phase3",
            # All three are HARD counts so they compare; the totals include
            # ledger/checklist verdicts the decision deliberately ignores.
            hard_blockers_on_entry=entry_hard,
            hard_blockers_after_repair=final_hard,
            hard_blockers_after_rollback=len(_hard_blockers(restored_blockers)),
            total_blockers_after_rollback=len(restored_blockers),
        )
        return True

    def _invoke_phase3_fix_loop(
        self,
        blockers: list[ValidationIssue],
        is_first_attempt: bool,
    ) -> int:
        """Run one LLM fix attempt against ``blockers``; return the number
        of successful write-tool calls it made.

        Factored out of ``_run_phase3_validation`` so the outer
        toolchain-fix iteration cap can call it more than once. Each
        invocation builds a fresh prompt (so the LLM doesn't see
        stale context from a previous attempt) and exits when the LLM
        emits ``end_turn`` or hits the per-attempt turn budget - except
        that an attempt with no successful edit is re-prompted once,
        with ``modify_file`` forced where the client supports
        ``tool_choice`` and a reminder in the message either way.
        Tool calls run through ``_execute_tool_blocks`` so they are
        recorded like Phase 2's.

        The prompt always reproduces the current blocker list verbatim
        — including any ``tsc [...]:`` / ``cargo [...]:`` /
        ``kotlinc [...]:`` lines. When toolchain blockers are present,
        a high-salience reminder is appended instructing the LLM to
        re-run the toolchain via ``run_command`` after each edit, so
        the model verifies its own fixes instead of stopping at the
        first plausible-looking change.
        """
        if not blockers:
            return 0

        # Citation/coverage obligations and model-authored tests can be wrong.
        # A verified existing implementation or corrected test need not mutate
        # production code. Keep forced edits only for concrete code defects.
        verification_only = all(issue.message.startswith((
            "requirement unverified:", "task unverified:", "api scenario:",
            "runtime unverified:", "create unverified:", "verification setup:",
        )) for issue in blockers)

        # Only the toolchain half is needed: it drives the re-run reminder
        # below. Every blocker, toolchain or not, is listed in the prompt.
        toolchain_blockers = [
            i for i in blockers
            if i.message.startswith(("tsc [", "cargo [", "kotlinc ["))
        ]

        prompt_parts: list[str] = []
        if self._instructions:
            prompt_parts.extend([
                "## Original request (the authority for required behavior)",
                self._instructions, "",
            ])
        if self._phase2_inspection_handoff:
            prompt_parts.extend(["## Prior inspection handoff", self._phase2_inspection_handoff, ""])
        if self.domain_model is not None:
            prompt_parts.extend([
                "## Domain model and conversion losses",
                "conversion_issues are NOT implemented constraints. Recover their "
                "intended behavior using actual relationship names; the original "
                "request takes precedence over conflicting model expressions.",
                json.dumps(serialize_domain_model(self.domain_model), ensure_ascii=False), "",
            ])
        requirements = self._requirements_for_validation()
        if requirements:
            prompt_parts.extend([
                "## Requirements to verify (including conversion recovery)",
                _requirements_ledger.render_requirements(requirements), "",
            ])
        prompt_parts.extend([
            "## Current files and symbols (use these paths; do not invent models/ or schemas/ folders)",
            build_inventory(self.output_dir, self.domain_model, self._generator_used or "existing workspace"),
            "", "## Actual action handlers",
            format_action_inventory(collect_action_endpoints(self.output_dir)), "",
        ])
        mutation_manifest = build_mutation_manifest(
            self.output_dir, scenario_records=self._api_scenarios.values(),
            current_revision=self._workspace_revision(),
        )
        if mutation_manifest:
            prompt_parts.extend([mutation_manifest, ""])
        if self._recent_tool_failures:
            prompt_parts.extend([
                "## Recent rejected operations (do not repeat unchanged requests)",
                json.dumps(self._recent_tool_failures, ensure_ascii=False), "",
            ])
        if is_first_attempt:
            prompt_parts.append(
                "Post-generation validation found these unresolved issues. "
                "Distinguish observed execution failures from unverified evidence "
                "and model judgments; resolve each against the original specification:"
            )
        else:
            prompt_parts.append(
                "After your previous fixes, these BLOCKER issues "
                "still remain. Fix every one:"
            )

        prompt_parts.append("")
        prompt_parts.append(
            "Repair in dependency order: first import/database startup, then "
            "schema/router mismatches and failed creates, then business behavior. "
            "Removing client-writable derived fields also requires implementing "
            "their server defaults/computation and updating every route/form that "
            "uses them. Do not delete business rules just to make startup pass. "
            "Call validate_app after each coherent change for fresh diagnostics. "
            "A model-authored API assertion is not the specification: inspect a "
            "failed scenario with test_api(action='get', scenario_id=...) before "
            "changing code. If its expectation contradicts the original request, "
            "correct that same scenario with a specification-grounded correction_reason. "
            "Do not change correct behavior to satisfy a mistaken assertion. "
            "Do not invent authentication, new roles, or destructive data deletion "
            "from a generic usability or release requirement."
        )
        prompt_parts.extend(f"- {i.message}" for i in sorted(blockers, key=self._repair_priority))

        # Show the offending lines. Measured 2026-09-18 on the model that had
        # just failed here: with only "file line N" it spends a turn on
        # read_file (6/6); with the excerpt it calls modify_file immediately
        # (6/6). A BOUNDED window is the point -- SWE-agent's ablation scores
        # a 100-line window above the whole file (18.0 vs 12.7 on SWE-bench
        # Lite), so this never pastes an entire file.
        excerpts = self._excerpts_for(blockers)
        if excerpts:
            prompt_parts.append("")
            prompt_parts.append(
                "The offending lines, verbatim from disk. Quote old_text from "
                "here exactly — never abbreviate with '...':"
            )
            prompt_parts.extend(excerpts)

        prompt_parts.append("")
        prompt_parts.append(
            "Resolve authorized dependency setup through install_dependencies, then "
            "validate_app; resolve missing evidence using current exact citations "
            "or test_api. Do not manufacture source changes to "
            "satisfy bookkeeping; make an edit only if behavior is actually missing. "
            "The harness will recheck evidence when this attempt ends."
            if verification_only else
            "Fix actual code defects with modify_file, replace_file_lines, or write_file. "
            "After repeated text matching failures, read the target block and use "
            "replace_file_lines with its read_id and inclusive line numbers instead of "
            "quoting old_text again. Reading first "
            "is fine, but you are not done until defects are repaired and verified. "
            "An evidence/citation error alone does not justify changing correct code. "
            "Do NOT touch anything unrelated."
        )

        # When the blockers include toolchain errors, instruct the LLM
        # to drive the toolchain itself with run_command — that's the
        # only way to know whether a fix actually compiles, and it's
        # the bench's per-project compile-pass criterion. We do this
        # as additional text in the same user turn (vs. a separate
        # message) so the LLM sees the request as part of the brief.
        if toolchain_blockers:
            cmds = self._toolchain_commands_for(toolchain_blockers)
            cmd_lines = "\n".join(f"  - {c}" for c in cmds)
            prompt_parts.append("")
            prompt_parts.append(
                "After each edit, re-run the relevant toolchain check "
                "using run_command to confirm the error is gone:"
            )
            prompt_parts.append(cmd_lines)
            prompt_parts.append(
                "Keep iterating (edit -> re-run) until the toolchain "
                "reports zero errors. Do not declare done while any "
                "compile / type error is still reported."
            )

        fix_prompt = "\n".join(prompt_parts)
        system = (
            "You are fixing validation errors in generated code. "
            "Fix each issue concisely. Call validate_app to recheck startup and "
            "data entry after a coherent repair. When shell tools are unavailable, "
            "validate_app and test_api are the supported verification tools. Use test_api "
            "for specification-based workflow assertions and invalid inputs. When shell tools "
            "are available and the report contains "
            "toolchain errors (tsc / cargo / kotlinc), you MUST "
            "verify your fix by re-running the toolchain with "
            "run_command — do not declare done based on the diff alone."
        )
        messages: list[dict] = [{"role": "user", "content": fix_prompt}]

        # Inject a high-salience reminder as a separate user message
        # right after the prompt. Matches the pattern used by the
        # per-file modify-loop guard: a <system-reminder>-tagged block
        # the LLM sees at response-time, not buried inside the prompt.
        if toolchain_blockers:
            reminder = self._build_toolchain_reminder(toolchain_blockers)
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": reminder}],
            })

        edits = 0                 # successful write-tool calls this attempt
        nudged = False            # the one re-prompt an edit-less attempt gets
        force_next: str | None = None
        turn_cap = _PHASE3_FIX_TURNS
        turn = 0
        while True:
            stop_reason = self._phase3_stop_requested()
            if stop_reason:
                logger.warning("Phase 3: %s", stop_reason)
                return edits
            if turn >= turn_cap:
                if edits or nudged or verification_only:
                    return edits
                # Read until the cap and wrote nothing: one more turn, and
                # it has to be the edit.
                nudged, force_next = True, "modify_file"
                turn_cap += 1
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": _PHASE3_NO_EDIT_REMINDER}],
                })
            turn += 1
            self.total_turns += 1
            # Recovery is shared with Phase 2. Do not force the failing text
            # strategy again after the executor requested a fresh read/range edit.
            force = self._force_tool_next or force_next
            self._force_tool_next, force_next = None, None
            request_messages = without_rejected_edit_drafts(messages)
            try:
                if force and self._client_supports_structured_chat():
                    response = self.client.chat(
                        system=system, messages=request_messages, tools=self.tools,
                        force_tool=force,
                    )
                else:
                    response = self.client.chat(
                        system=system, messages=request_messages, tools=self.tools,
                    )
            except Exception as exc:
                # Surface the failure instead of silently exiting the fix
                # loop — callers and logs need to see why validation bailed.
                logger.warning(
                    "Phase 3: LLM call failed on fix turn %d, aborting fix loop: %s",
                    turn, exc,
                )
                self._phase3_interrupted = True
                return edits
            # A stop can arrive while a provider request is in flight. Do not
            # apply its returned mutations after cancellation. The final allowed
            # turn may still execute its tools; it is not a new provider call.
            if self._phase3_stop_requested(check_turn_budget=False):
                return edits
            if response["stop_reason"] == "end_turn":
                if edits or nudged or verification_only:
                    return edits
                # Ended in prose with nothing written. Say so once, force the
                # edit where tool_choice is honoured, and let the reminder
                # carry it where the gateway ignores tool_choice.
                nudged, force_next = True, "modify_file"
                if response["content"]:
                    messages.append({"role": "assistant", "content": response["content"]})
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": _PHASE3_NO_EDIT_REMINDER}],
                })
                continue
            if response["stop_reason"] != "tool_use":
                # Without this the loop appends nothing and re-sends an
                # identical request until the cap. Phase 2 already breaks here.
                logger.warning(
                    "Phase 3: unexpected stop_reason %r on fix turn %d — "
                    "stopping the fix loop instead of re-sending the same "
                    "request", response["stop_reason"], turn,
                )
                self._phase3_interrupted = True
                return edits
            messages.append({"role": "assistant", "content": response["content"]})
            tool_blocks = [
                block for block in response["content"]
                if hasattr(block, "type") and block.type == "tool_use"
                and getattr(block, "name", None)
            ]
            # Through the Phase 2 path, so the trace, recipe and sidecar see
            # these calls; run 7f918e11's ten turns went through the raw
            # executor and left no record of what they were.
            logged_before = len(self.tool_calls_log)
            tool_results = self._execute_tool_blocks(tool_blocks, self.total_turns - 1)
            edits += sum(
                1 for entry in self.tool_calls_log[logged_before:]
                if entry["tool"] in _WRITE_TOOLS_ON_RECORD and entry["success"]
            )
            messages.append({"role": "user", "content": tool_results})
            # The same streak/repeat guards Phase 2 and the fix cycle get.
            # Omitting them here left the bounded repair loop running on
            # _is_stuck alone, on a tighter budget than either.
            if self._apply_edit_loop_guards(messages, where="phase 3 repair"):
                break
            self._save_phase3_checkpoint()

    _FILE_LINE_RE = _re.compile(
        r"(?<![\w./\\-])([\w./\\-]+\.\w+)(?: line |:|\()(\d+)"
    )

    def _excerpts_for(
        self, blockers: list[ValidationIssue], context: int = 5, limit: int = 3
    ) -> list[str]:
        """Numbered source windows around each blocker that names file+line."""
        out: list[str] = []
        seen: set[tuple[str, int]] = set()
        for issue in blockers:
            match = self._FILE_LINE_RE.search(issue.message)
            if not match:
                continue
            rel, line_no = match.group(1), int(match.group(2))
            if (rel, line_no) in seen or len(seen) >= limit:
                continue
            path = os.path.realpath(os.path.join(self.output_dir, rel.replace("/", os.sep)))
            workspace = os.path.realpath(self.output_dir)
            try:
                if os.path.commonpath([workspace, path]) != workspace:
                    continue
            except ValueError:
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    lines = fh.read().splitlines()
            except (OSError, UnicodeError):
                continue
            if not 1 <= line_no <= len(lines):
                continue
            seen.add((rel, line_no))
            lo = max(0, line_no - 1 - context)
            hi = min(len(lines), line_no + context)
            body = chr(10).join(
                f"{n + 1:>5}| {lines[n]}" for n in range(lo, hi)
            )
            header = f"{rel} (lines {lo + 1}-{hi}):"
            fence = "```"
            out.append(chr(10).join(["", header, fence, body, fence]))
        return out

    def _toolchain_commands_for(
        self, toolchain_blockers: list[ValidationIssue]
    ) -> list[str]:
        """Pick the right re-run command for each toolchain in the report.

        Looks at the prefix on each blocker message (``tsc [...]:``,
        ``cargo [...]:``, ``kotlinc [...]:``) and returns the matching
        command (with the project sub-path) the LLM should invoke via
        ``run_command`` to verify its fix. The list is deduplicated so
        the same command isn't suggested twice for a multi-error report.
        """
        commands: list[str] = []
        seen: set[str] = set()
        for issue in toolchain_blockers:
            msg = issue.message
            # Extract the bracketed sub-path: ``tsc [frontend]:`` -> ``frontend``
            match = _re.match(r"(tsc|cargo|kotlinc) \[([^\]]+)\]:", msg)
            if not match:
                continue
            tool, path = match.group(1), match.group(2)
            if tool == "tsc":
                # ``npx tsc --noEmit`` works whether or not tsc is on
                # PATH globally — npm projects nearly always have it
                # installed locally as a devDependency.
                cmd = (
                    f"run_command: command='npx tsc --noEmit', "
                    f"working_dir='{path}'"
                )
            elif tool == "cargo":
                cmd = (
                    f"run_command: command='cargo check', "
                    f"working_dir='{path}'"
                )
            elif tool == "kotlinc":
                # The .kt files under the module root, compiled to
                # /dev/null. The bench uses kotlinc directly too.
                cmd = (
                    f"run_command: command='kotlinc -nowarn "
                    f"-d /tmp/out $(find {path} -name \"*.kt\")', "
                    f"working_dir='.'"
                )
            else:  # pragma: no cover - defensive
                continue
            if cmd not in seen:
                commands.append(cmd)
                seen.add(cmd)
        return commands

    def _build_toolchain_reminder(
        self, toolchain_blockers: list[ValidationIssue]
    ) -> str:
        """High-salience reminder text for the toolchain-fix loop.

        Mirrors the shape of ``_build_modify_loop_reminder`` (the
        per-file modify-loop guard the LLM already recognises) so the
        model treats the toolchain errors with the same urgency as a
        rewrite-or-quit warning. The verbatim error lines are quoted
        in the reminder so they sit in the model's working memory
        right before its next response.
        """
        # Cap the reminder body so a runaway report (e.g. 100 type
        # errors after a single missing import) doesn't blow out the
        # context window. The full list is already in the prompt.
        lines = [i.message for i in toolchain_blockers[:8]]
        bulleted = "\n".join(f"  - {ln}" for ln in lines)
        more = (
            f"\n  - (+{len(toolchain_blockers) - 8} more — see prompt for full list)"
            if len(toolchain_blockers) > 8 else ""
        )
        return (
            "<system-reminder>"
            "The generated project does not compile on its own toolchain. "
            "The bench's per-project compile-pass score is currently 0 "
            "for this run because of these errors:\n"
            f"{bulleted}{more}\n\n"
            "You MUST drive these to zero. After EACH edit, invoke "
            "run_command with the appropriate toolchain check (npx tsc "
            "--noEmit / cargo check / kotlinc) and read the output. "
            "Only call end_turn once the toolchain reports zero errors, "
            "or after you have made a clear good-faith attempt that the "
            "remaining errors require dependencies you cannot add."
            "</system-reminder>"
        )

    @staticmethod
    def _repair_priority(issue: ValidationIssue) -> tuple[int, str]:
        message = issue.message.lower()
        if message.startswith(("syntax", "python contract:", "mapper config:", "application startup:", "missing module:")):
            return 0, message
        if message.startswith(("data contract:", "create contract:", "undefined name:", "runtime unverified:")):
            return 1, message
        if message.startswith(("requirement", "task unverified:")):
            return 3, message
        return 2, message

    def _collect_execution_issues(self) -> list[str]:
        """Cheap source checks first, then isolated startup/data-entry probes.

        This path never calls an LLM, installs packages, or grants shell access.
        It is available during editing as well as at the final verification gate.
        """
        from besser.generators.llm.write_diagnostics import diagnose_written_content

        raw: list[str] = []
        for path in _python_files(self.output_dir):
            rel = os.path.relpath(path, self.output_dir).replace("\\", "/")
            try:
                with open(path, encoding="utf-8-sig") as fh:
                    source = fh.read()
            except (OSError, UnicodeError):
                continue
            for finding in diagnose_written_content(rel, source, workspace=self.output_dir, limit=25):
                raw.append(f"python contract: {rel} line {finding.get('line', 1)}: {finding['message']}")
        raw.extend(_create_schema_router_mismatches(self.output_dir))
        from besser.generators.llm.validation.frontend_schema import collect_frontend_schema_issues

        raw.extend(collect_frontend_schema_issues(self.output_dir))
        raw.extend(_unresolvable_local_imports(self.output_dir))
        # Don't repeatedly boot an application already proven to be broken.
        if raw:
            return list(dict.fromkeys(raw))
        if self.enable_import_smoke_check:
            mapper_issues = _import_smoke_issues(self.output_dir)
            raw.extend(mapper_issues)
            if not any(i.startswith("mapper config:") for i in mapper_issues):
                try:
                    from besser.generators.llm.constructibility import collect_constructibility_issues
                    raw.extend(collect_constructibility_issues(self.output_dir))
                except Exception as exc:
                    raw.append(f"runtime unverified: isolated app verification failed: {type(exc).__name__}: {exc}")
            raw = ["runtime unverified: " + item if item.startswith("validation:") else item for item in raw]
            # Guessed fixtures can legitimately violate business rules. Keep
            # that unknown distinct from a crash, and let a real create/read
            # scenario supply evidence without weakening application validation.
            raw.extend(self._collect_api_scenario_issues())
            from besser.generators.llm.api_probe import confirmed_create_paths
            revision = self._workspace_revision()
            confirmed = {
                (record["report"].get("backend"), path)
                for record in self._api_scenarios.values() if record["revision"] == revision
                for path in confirmed_create_paths(record["report"])
            }
            checked = []
            for item in raw:
                if item.startswith("create unverified: "):
                    match = _re.match(r"create unverified: (.+?): POST (\S+) -", item)
                    if match and (match.group(1), match.group(2).rstrip("/")) in confirmed:
                        continue
                    item = "runtime unverified: " + item
                checked.append(item)
            raw = checked
        elif any(os.path.basename(p) == "main_api.py" for p in _python_files(self.output_dir)):
            raw.append("runtime unverified: backend startup/data-entry checks are disabled; this app has not been runtime verified")
        return list(dict.fromkeys(raw))

    def _validate_app(self) -> dict:
        """Model-facing, revision-bound diagnostics with no additional LLM cost."""
        revision = self._workspace_revision()
        if self._app_validation_cache and self._app_validation_cache[0] == revision:
            return self._app_validation_cache[1]
        issues = sorted((_classify_issue(s) for s in self._collect_execution_issues()), key=self._repair_priority)
        blockers = [issue for issue in issues if issue.severity == "blocker"]
        result = {
            "verified": not blockers,
            "blocker_count": len(blockers),
            "issues": [issue.message for issue in issues[:40]],
            "remaining_issue_count": max(0, len(issues) - 40),
            "scope": "Python declarations, schema/router contracts, backend startup and create probes where supported; not complete business acceptance",
            "next_step": ("Repair concrete failures and call validate_app again. For create unverified (guessed fixtures), "
                          "use test_api with valid unique fixtures and a GET of the persisted record; do not remove "
                          "business validation to satisfy guessed samples.") if blockers else "These checks pass. Verify the remaining user requirements and business workflows before completion.",
        }
        self._app_validation_cache = (revision, result)
        return result

    def _test_api(self, args: dict) -> dict:
        """Run and retain bounded declarative workflow checks, never arbitrary commands."""
        action = args.get("action", "run")
        if action not in ("run", "list", "get"):
            return {"error": "action must be run, list, or get"}
        scenario_id = args.get("scenario_id")
        if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id.strip() or len(scenario_id) > 80):
            return {"error": "scenario_id must be a nonempty workflow name of at most 80 characters."}
        if action == "list":
            revision = self._workspace_revision()
            return {"scenarios": [{"scenario_id": record.get("scenario_id"),
                                   "backend": record["scenario"].get("backend"),
                                   "request_count": len(record["scenario"]["requests"]),
                                   "last_status": record["report"].get("status"),
                                   "current_revision": record["revision"] == revision}
                                  for record in self._api_scenarios.values()],
                    "next_step": "Use action=get with scenario_id to inspect exact expectations before a repair."}
        previous = self._api_scenarios.get("named:" + scenario_id) if scenario_id else None
        if action == "get":
            if previous is None:
                return {"error": "Unknown scenario_id; use action=list to find retained workflows."}
            # Return a detached snapshot; inspecting a test must not mutate it or
            # turn a stale report into fresh verification.
            return json.loads(json.dumps({
                "scenario_id": scenario_id, **previous["scenario"],
                "last_report": previous["report"],
                "current_revision": previous["revision"] == self._workspace_revision(),
                "correction_history": previous.get("correction_history", []),
                "authority": "Original user specification. A generated assertion can be wrong; explain any correction against that specification.",
            }))
        if not self.enable_import_smoke_check:
            return {"error": "Runtime execution is disabled; API workflows have not been verified."}
        from besser.generators.llm.api_probe import probe_api_scenario

        if "requests" not in args:
            if previous is None:
                return {"error": "Provide requests for a new workflow, or scenario_id to replay a retained one."}
            if "backend" in args and args["backend"] != previous["scenario"].get("backend"):
                return {"error": "Changing a scenario backend requires its requests and correction_reason."}
            args = {**args, **previous["scenario"]}
        # Freeze the caller's values: later argument mutations must not silently
        # weaken retained assertions or bypass explicit scenario correction.
        try:
            scenario = json.loads(json.dumps({"requests": args.get("requests"), "backend": args.get("backend")}, allow_nan=False))
        except (ValueError, TypeError, RecursionError):
            return {"error": "Scenario arguments must be finite JSON values."}
        scenario_id = scenario_id or "scenario-" + hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()[:16]
        key = "named:" + scenario_id
        previous = self._api_scenarios.get(key)
        changed = previous is not None and previous["scenario"] != scenario
        reason = args.get("correction_reason")
        if changed and (not isinstance(reason, str) or not reason.strip()):
            return {"error": "Changing a saved scenario requires correction_reason explaining the mistaken test; do not relax the original specification."}
        if key not in self._api_scenarios and len(self._api_scenarios) >= 10:
            return {"error": "Ten workflow scenarios are already retained. Rerun an existing scenario or use validate_app."}
        revision = self._workspace_revision()
        report = probe_api_scenario(self.output_dir, scenario["requests"], backend=scenario["backend"])
        # Invalid tool arguments must not become impossible-to-repair app failures.
        if report.get("boot") != "not_started":
            history = list(previous.get("correction_history", [])) if previous else []
            if changed:
                report["correction"] = {"reason": reason[:1000], "previous_status": previous["report"].get("status")}
                history.append(report["correction"])
            self._api_scenarios[key] = {"scenario": scenario, "scenario_id": scenario_id, "revision": revision,
                                        "report": report, "correction_history": history[-5:]}
            self._app_validation_cache = None
        if scenario_id:
            report["scenario_id"] = scenario_id
        return report

    def _collect_api_scenario_issues(self) -> list[str]:
        """Replay retained scenarios against changed source so old green results cannot hide regressions."""
        revision = self._workspace_revision()
        issues = []
        for index, record in enumerate(list(self._api_scenarios.values()), 1):
            if record["revision"] != revision:
                record["report"] = self._test_api({**record["scenario"], "scenario_id": record.get("scenario_id")})
                record["revision"] = revision
            report = record["report"]
            if report.get("status") != "passed":
                detail = report.get("error") or json.dumps({
                    "assertion_failures": report.get("assertion_failures", []),
                    "request_errors": [{"path": response.get("path"), "error": response["error"]}
                                       for response in report.get("responses", []) if response.get("error")],
                })
                scenario_id = record.get("scenario_id") or f"workflow {index}"
                issues.append(
                    f"api scenario: {scenario_id} failed: {detail[:2500]}. "
                    "Inspect its exact requests and expected values with test_api(action='get', "
                    f"scenario_id={scenario_id!r}). The original specification is authoritative: "
                    "fix the application if its behavior is wrong; if the generated test is wrong, "
                    "resubmit this scenario with correction_reason grounded in the specification."
                )
        return issues

    def _collect_validation_issues(self) -> list[ValidationIssue]:
        """Collect all validation issues from the output directory.

        Returns ``ValidationIssue`` records with severity. The
        Phase 3 fix loop only acts on ``blocker`` items when
        ``auto_fix_issues`` is enabled.
        """
        raw_issues: list[str] = [
            required_check_unverified(
                "model assembly",
                f"{issue['diagram_type']} [{issue['diagram_id']}]: {issue['diagnostic']}; "
                "correct the project input and regenerate",
            ) for issue in self._assembly_issues
        ]

        for root, _, files in os.walk(self.output_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.output_dir).replace("\\", "/")

                # Skip snapshot directory
                if rel.startswith(_SNAPSHOT_DIR):
                    continue

                # Check Python syntax
                if fname.endswith(".py"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            _ast.parse(f.read(), filename=rel)
                    except SyntaxError as e:
                        raw_issues.append(f"Syntax error in {rel} line {e.lineno}: {e.msg}")

                # Check Dockerfiles. Matching only the exact name "Dockerfile"
                # skipped every multi-service layout: an app with
                # Dockerfile.frontend / Dockerfile.backend ran none of the checks
                # below and failed `docker compose build` on `npm ci`
                # (2026-09-11).
                if _is_dockerfile(fname):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read()
                        docker_dir = os.path.dirname(fpath)
                        # npm ci without a lock file -> should be npm install.
                        # Searched across the WHOLE project, not just beside the
                        # Dockerfile: a root Dockerfile.frontend typically COPYs
                        # from a frontend/ subdirectory, so docker_dir alone
                        # missed lockfiles that existed.
                        if "npm ci" in content:
                            if not _project_has_npm_lockfile(self.output_dir):
                                # Auto-fix this common mistake
                                fixed = content.replace("npm ci", "npm install")
                                with open(fpath, "w", encoding="utf-8") as f:
                                    f.write(fixed)
                                logger.info("Auto-fixed: %s: npm ci -> npm install (no lock file)", rel)
                        # COPY of a package-lock.json that does not exist is a
                        # HARD build failure ("failed to compute cache key"), and
                        # the LLM cannot write a lockfile - npm resolves it from
                        # the registry. Strip the reference; package.json alone
                        # is enough for `npm install`.
                        if "package-lock.json" in content and not _project_has_npm_lockfile(
                            self.output_dir
                        ):
                            stripped = _strip_missing_lockfile_copy(content)
                            if stripped != content:
                                content = stripped
                                with open(fpath, "w", encoding="utf-8") as f:
                                    f.write(content)
                                logger.info(
                                    "Auto-fixed: %s: dropped COPY of a "
                                    "package-lock.json that does not exist", rel,
                                )

                        # Check COPY references
                        if "package.json" in content or "package*.json" in content:
                            pkg = os.path.join(docker_dir, "package.json")
                            if not os.path.isfile(pkg):
                                raw_issues.append(f"{rel} references package.json but it doesn't exist")
                        if "requirements.txt" in content:
                            req = os.path.join(docker_dir, "requirements.txt")
                            if not os.path.isfile(req):
                                if _ensure_requirements_txt(docker_dir):
                                    logger.info(
                                        "Auto-fixed: restored missing requirements.txt for %s", rel
                                    )
                                else:
                                    raw_issues.append(f"{rel} references requirements.txt but it doesn't exist")
                    except Exception:
                        pass

        # Auto-fix known critical incompatibility: passlib + bcrypt>=4.1
        # This is a belt-and-suspenders fix — the pip dry-run below should
        # also catch it, but this is instant and doesn't need network.
        for root, _, files in os.walk(self.output_dir):
            for fname in files:
                if fname == "requirements.txt":
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, self.output_dir).replace("\\", "/")
                    if _SNAPSHOT_DIR in rel:
                        continue
                    try:
                        with open(fpath, "r") as f:
                            content = f.read()
                        if "passlib" in content:
                            import re as _re
                            new_content = _re.sub(r'bcrypt[><=!]+[^\n]*', 'bcrypt==4.0.1', content)
                            if "bcrypt" not in new_content:
                                new_content += "\nbcrypt==4.0.1\n"
                            if new_content != content:
                                with open(fpath, "w") as f:
                                    f.write(new_content)
                                logger.info("Auto-fixed: %s: pinned bcrypt==4.0.1 (passlib compat)", rel)
                    except Exception:
                        pass

        # Dependency resolution may execute untrusted build backends and
        # access the network. Keep it behind the same explicit shell-tools
        # trust gate as run_command/install_dependencies; hosted mode has
        # this disabled by default and therefore never invokes pip.
        if self.allow_shell_tools:
            for root, _, files in os.walk(self.output_dir):
                for fname in files:
                    if fname == "requirements.txt":
                        req_path = os.path.join(root, fname)
                        rel = os.path.relpath(req_path, self.output_dir).replace("\\", "/")
                        if rel.startswith(_SNAPSHOT_DIR):
                            continue
                        try:
                            import subprocess
                            # Dry-run install to check for conflicts
                            req_dir = os.path.dirname(req_path)
                            result = subprocess.run(
                                [sys.executable, "-m", "pip", "install",
                                 "--dry-run", "-r", "requirements.txt", "--quiet"],
                                capture_output=True, text=True, timeout=30,
                                cwd=req_dir,
                                # Never expose provider keys / OAuth secrets to a
                                # (possibly untrusted) requirements.txt's build
                                # backend, which pip may execute to resolve sdists.
                                env=_safe_subprocess_env(),
                            )
                            if result.returncode != 0:
                                # Extract the meaningful error
                                err_lines = [line for line in result.stderr.strip().split("\n")
                                             if line.strip() and "WARNING" not in line]
                                if err_lines:
                                    err = "\n".join(err_lines[-3:])
                                    raw_issues.append(f"Dependency conflict in {rel}:\n{err}")
                        except Exception:
                            pass  # pip not available or timeout — skip

        # Static checks catch per-project compile errors that would
        # otherwise only surface at deploy time. ruff is near-instant
        # and always runs; the project compilers (tsc / cargo /
        # kotlinc) can add minutes of wall-clock and are gated behind
        # ``enable_toolchain_validation`` so the web deployment can
        # opt out per deploy. Required TS/frontend checks record an explicit
        # verification gap when opted out; optional lint remains advisory.
        raw_issues.extend(self._collect_frontend_contract_issues())
        raw_issues.extend(_method_button_source_issues(self.output_dir))
        raw_issues.extend(action_implementation_issues(
            self.output_dir, self._expected_action_endpoints(),
        ))
        try:
            from besser.generators.llm.endpoint_coherence import (
                collect_endpoint_coherence_issues,
            )

            # Warning mode for the first campaign: _classify_issue deliberately
            # leaves this prefix at the conservative warning default. Promote to
            # blocker only after measured false-positive review.
            raw_issues.extend(collect_endpoint_coherence_issues(self.output_dir))
        except Exception:
            logger.debug("Endpoint coherence validation failed", exc_info=True)
        raw_issues.extend(self._collect_framework_switch_issues())
        raw_issues.extend(self._collect_missing_frontend_issue())
        raw_issues.extend(self._collect_data_contract_issues())

        # Model-derived acceptance matrix: per entity — route present,
        # page present, create wired. REPORT-ONLY (warnings + recipe
        # field): a GUI-scoped run may legitimately omit entities, so
        # these are visibility, never blockers.
        try:
            from besser.generators.llm.acceptance import build_acceptance_matrix, matrix_issues
            self._acceptance_matrix = build_acceptance_matrix(
                self.output_dir, self.domain_model,
            )
            raw_issues.extend(matrix_issues(self._acceptance_matrix))
        except Exception:
            logger.debug("Acceptance matrix computation failed", exc_info=True)

        execution_issues = self._collect_execution_issues()
        # An enabled ledger that extracted nothing verified nothing. Saying so
        # is the difference between "the user asked for nothing" and "we never
        # looked" — run 7aybctis reported the former on an empty extraction.
        if (self.enable_requirements_ledger and self._requirements is None
                and self._requirement_extraction_attempts):
            raw_issues.append(_check_did_not_run(
                "the requirements ledger",
                f"{self._requirement_extraction_attempts} extraction attempt(s) on "
                f"{getattr(self.client, 'model', 'this model')} returned nothing",
            ))
        if not self.enable_requirements_ledger or not any(_classify_issue(s).severity == "blocker" for s in execution_issues):
            raw_issues.extend(self._collect_requirement_issues())
        else:
            raw_issues.append("validation: business-requirement judgment deferred until startup/data-entry blockers are fixed")
        raw_issues.extend(self._collect_task_issues())
        raw_issues.extend(self._collect_ruff_issues())
        raw_issues.extend(execution_issues)
        # The TS collector reports a relevant disabled check as unknown; it
        # never enables tooling. Already-authorized dependency setup is an
        # actionable verification prerequisite, not a request for source edits.
        raw_issues.extend(self._collect_tsc_issues())
        if self.enable_toolchain_validation:
            raw_issues.extend(self._collect_cargo_issues())
            raw_issues.extend(self._collect_kotlinc_issues())
        else:
            logger.info(
                "Phase 3: toolchain validation (tsc/cargo/kotlinc) disabled "
                "for this run"
            )

        from besser.generators.llm.validation.frontend_build import collect_frontend_build_issues
        build_cache = getattr(self, "_successful_frontend_builds", {})
        self._successful_frontend_builds = build_cache
        remaining = self.max_runtime_seconds - (
            time.monotonic() - self._start_time if self._start_time is not None else 0)
        raw_issues.extend(collect_frontend_build_issues(
            self.output_dir, enabled=self.enable_toolchain_validation,
            allow_shell=self.allow_shell_tools, source_revision=self._workspace_revision,
            successful_builds=build_cache, can_run=self._verification_call_allowed,
            timeout=remaining,
        ))

        issues = [_classify_issue(s) for s in raw_issues]
        # For the duration of a fix/modify run, promote findings that match
        # the user-reported target from warning to blocker so the Phase 3
        # fix loop is driven to resolve them and the success gate keys on
        # them. Non-matching findings keep their severity. No-op on
        # from-scratch runs (``_is_fix_run`` is only set in modify()).
        return sorted(self._promote_fix_target_findings(issues), key=self._repair_priority)

    _SCAFFOLD_FAMILIES = {
        "generate_fastapi_backend": "fastapi",
        "generate_web_app": "fastapi",
        "generate_rest_api": "fastapi",
        "generate_django": "django",
    }

    def _scaffold_family(self) -> str | None:
        return self._SCAFFOLD_FAMILIES.get(self._generator_used or "")

    def _collect_framework_switch_issues(self) -> list[str]:
        """BLOCKER when generated code imports a rival framework.

        Live finding (2026-09-02): the free model rewrote a FastAPI
        scaffold into a Flask hybrid via write_file (delete-protection
        never fired), burned the runtime cap mid-restructure, and shipped
        an unbootable mix. The HARD-CONSTRAINTS prompt forbids this; now
        Phase 3 enforces it.
        """
        family = self._scaffold_family()
        rivals = {"fastapi": ("flask", "django"),
                  "django": ("flask", "fastapi")}.get(family or "")
        if not rivals:
            return []
        offenders: list[str] = []
        for root, dirs, files in os.walk(self.output_dir):
            dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", "build")]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.output_dir).replace("\\", "/")
                if rel.startswith(_SNAPSHOT_DIR) or rel.startswith(".besser_"):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                for rival in rivals:
                    if _re.search(rf"^\s*(?:from|import)\s+{rival}\b", content, _re.MULTILINE):
                        offenders.append(f"{rel} (imports {rival})")
                        break
        if not offenders:
            return []
        shown = ", ".join(offenders[:6])
        more = f" (+{len(offenders) - 6} more)" if len(offenders) > 6 else ""
        return [
            f"frontend contract: framework switch — the scaffold is {family} "
            f"but these files import a rival framework: {shown}{more}. "
            f"Remove the rewrite and extend the existing {family} app."
        ]

    # "web application" was the miss that mattered: the trailing \b could
    # not match after "app" when the word continued into "lication", so the
    # single most natural phrasing of the request slipped past and 9 of 10
    # sweep runs shipped backend-only while reporting success. Every
    # alternative here has to tolerate the word being spelled out.
    # Deliberately NOT included: "spa" — a hotel spec has one.
    _WEBAPP_ASK_RE = _re.compile(
        r"\b(web[ -]?app(?:lication)?s?|front[ -]?end|web ?site|"
        r"web ?interface|single[ -]page app(?:lication)?s?|ui|"
        r"user interface|dashboard|portal)\b")

    def _has_frontend_files(self) -> bool:
        for root, dirs, files in os.walk(self.output_dir):
            dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", "build")]
            rel_root = os.path.relpath(root, self.output_dir).replace("\\", "/")
            if rel_root.startswith(_SNAPSHOT_DIR):
                continue
            for fname in files:
                if fname.endswith((".js", ".jsx", ".ts", ".tsx", ".html")) or fname == "package.json":
                    return True
        return False

    _FRONTEND_CHECKLIST_TASK = (
        "Create the React frontend (none exists yet) — write these files "
        "with write_file: frontend/package.json (react, react-dom, "
        "react-router-dom, vite), frontend/index.html, "
        "frontend/src/main.jsx, frontend/src/App.jsx (router with a home "
        "route + nav), frontend/src/api.js (fetch helpers for the backend "
        "routes), and per entity frontend/src/pages/<Entity>List.jsx with "
        "a table plus working Create/Edit/Delete wired to the API. This "
        "task cannot be marked done until frontend files exist on disk."
    )

    def _deterministic_gap_tasks(self) -> list[str]:
        """Checklist items the harness ADDS regardless of the planner.

        Devstral A/B finding: with a backend-only scaffold, no layer
        explicitly ORDERS the frontend — the gap planner assumes the
        scaffold has screens, and prompt Rule 15 is prose a terse model
        skips. Making it a checklist item puts it behind the end_turn
        gate, which is enforcement, not prose.
        """
        low = (self._instructions or "").lower()
        tasks: list = []
        if self._WEBAPP_ASK_RE.search(low) and not self._has_frontend_files():
            tasks.append({
                "text": self._FRONTEND_CHECKLIST_TASK,
                # Cheat-proof: done is refused until frontend files exist.
                "verify": self._has_frontend_files,
            })
        tasks.extend(action_gap_tasks(self.output_dir, self._expected_action_endpoints()))
        tasks.extend(self._unenforced_rule_tasks())
        tasks.extend(item["text"] for item in self._requirements_for_validation()
                     if item.get("conversion_issue_id"))
        return tasks

    def _expected_action_endpoints(self):
        # Keep initial obligations even if a handler is deleted, and discover
        # handlers introduced after an initially empty/from-scratch scaffold.
        known = {(item.path, item.http_method, item.route, item.router_binding, item.router_prefix): item
                 for item in self._action_endpoints or []}
        for item in collect_action_endpoints(self.output_dir):
            known.setdefault((item.path, item.http_method, item.route, item.router_binding, item.router_prefix), item)
        self._action_endpoints = list(known.values())
        return self._action_endpoints

    def _collect_task_issues(self) -> list[str]:
        conversion_requirements = [item for item in self._requirements_for_validation()
                                   if item.get("conversion_issue_id")]
        verified_conversion_texts = set()
        if self.enable_requirements_ledger and conversion_requirements:
            # Phase 3 may implement a rule without closing its Phase 2 task.
            # Only the current source+requirements judgment can discharge it;
            # a stale verdict or a task_list claim cannot.
            verdicts = self._requirement_judgments.get(
                self._requirement_cache_key(self._requirements_for_validation()), [],
            )
            verified_ids = {item["id"] for item in _requirements_ledger.verify_evidence(
                verdicts, self.output_dir,
            ) if item["status"] == "implemented"}
            verified_conversion_texts = {item["text"] for item in conversion_requirements
                                         if item["id"] in verified_ids}
        issues = []
        for task in self.executor.open_tasks() + self.executor.blocked_tasks():
            if task["text"] in verified_conversion_texts:
                task.update(done=True, blocked=False, verification="evidence_checked")
                task.pop("blocked_reason", None)
                continue
            verify = task.get("verify")
            if verify is not None:
                try:
                    if verify():
                        # A Phase 3 edit can satisfy a formerly blocked structural
                        # task; evaluate the real callback, never trust the claim.
                        self.executor._task_list({"action": "done", "id": task["id"]})
                        continue
                except Exception:
                    pass  # The task remains unresolved, not successful.
            issues.append(
                f"task unverified: task {task['id']} remains unresolved: {task['text']}. "
                f"{task.get('blocked_reason', '')} Inspect and implement the required work; "
                "record exact current evidence with task_list (existing=true is allowed "
                "for an implementation already present; acceptance still needs verification)."
            )
        return issues

    def _unenforced_rule_tasks(self) -> list[str]:
        """One task per modeled OCL rule the generators could not enforce.

        The pydantic generator declines a constraint that spans relationships
        and leaves a NOTE in the schema (pydantic_classes_template.py.j2). The
        rule is the user's own words in the model, so it is checklist work,
        not a comment: run 19h35 (2026-09-18) shipped without the guest-
        capacity rule its model carried. FastAPI scaffolds only - the
        placement names a router file.
        """
        if self._scaffold_family() != "fastapi" or self.domain_model is None:
            return []
        from besser.generators.pydantic_classes.ocl_utils import parse_ocl_constraint
        tasks: list[str] = []
        for constraint in list(getattr(self.domain_model, "constraints", None) or []):
            context = getattr(constraint, "context", None)
            if context is None or getattr(constraint, "language", "OCL") != "OCL":
                continue
            try:
                parsed = parse_ocl_constraint(constraint, self.domain_model)
            except Exception:
                parsed = None
            if not (isinstance(parsed, dict) and parsed.get("skipped")):
                continue
            cls = context.name
            tasks.append(
                f"Enforce the modeled rule '{constraint.name}' of {cls} - OCL: "
                f"{constraint.expression}. The generators could not express it "
                "(it spans relationships), so nothing enforces it yet: implement "
                f"the check in the create and update endpoints of routers/{cls.lower()}.py "
                "and in every modeled method that changes the values involved, and "
                "refuse a violating request with HTTP 400 naming the rule."
            )
        return tasks

    def _collect_missing_frontend_issue(self) -> list[str]:
        """BLOCKER when the user asked for a web app and got no frontend.

        The hotel audit's defect #4: 'build a hotel reservation web app'
        shipped an API-only tree — presence of a backend read as success.
        High-precision: fires only when the instructions explicitly name a
        web app / frontend / UI AND the workspace holds not a single
        frontend artifact (js/ts/tsx/jsx/html or a package.json).
        """
        low = (self._instructions or "").lower()
        if not _re.search(r"\b(web ?app|frontend|front-end|website|\bui\b|user interface)\b", low):
            return []
        for root, dirs, files in os.walk(self.output_dir):
            dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", "build")]
            rel_root = os.path.relpath(root, self.output_dir).replace("\\", "/")
            if rel_root.startswith(_SNAPSHOT_DIR):
                continue
            for fname in files:
                if fname.endswith((".js", ".jsx", ".ts", ".tsx", ".html")) or fname == "package.json":
                    return []
        return [
            "frontend contract: the instructions request a web app / "
            "frontend, but the output contains no frontend files at all "
            "(no js/ts/html, no package.json). Build the frontend — a "
            "backend-only tree does not satisfy a web-app request."
        ]

    def _collect_data_contract_issues(self) -> list[str]:
        """Sweep the workspace with the model-derived data-contract lint.

        Same checks the executor already ran per-write (contract_checks
        module) — this catches what slipped through anyway: files the
        model wrote before a violation pattern existed in them, Phase 1
        scaffold output, and violations introduced by one edit into
        another file's assumptions. Blocker findings feed the Phase 3
        fix loop via the ``data contract:`` prefix; advisory findings
        are reported as warnings.
        """
        try:
            from besser.generators.llm.contract_checks import build_data_contract, lint_file
            contract = build_data_contract(self.domain_model)
        except Exception as exc:
            # Returning [] here reported "no data-contract violations" when the
            # truth was that the contract could not be built and nothing was
            # checked. Say which it is.
            logger.warning("Data-contract check could not run: %s", exc, exc_info=True)
            return [_check_did_not_run("the data-contract check", str(exc)[:200])]
        if contract is None:
            # A legitimately empty contract (no domain model / no classes):
            # nothing to check, not a failure.
            return []

        issues: list[str] = []
        exts = (".py", ".js", ".jsx", ".ts", ".tsx")
        for root, dirs, files in os.walk(self.output_dir):
            dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", "build")]
            for fname in files:
                if not fname.endswith(exts):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.output_dir).replace("\\", "/")
                if rel.startswith(_SNAPSHOT_DIR) or rel.startswith(".besser_"):
                    continue
                try:
                    if os.path.getsize(fpath) > 1_000_000:
                        continue
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                for finding in lint_file(rel, content, contract):
                    prefix = "data contract:" if finding.blocker else "data contract (advisory):"
                    issues.append(
                        f"{prefix} {finding.path} line {finding.line}: {finding.message}"
                    )
        return issues

    def _planner_instructions(self, instructions: str) -> str:
        """The request, then the requirements ledger as numbered lines.

        The planner already reads the verbatim spec, and on the 19h35 model
        it still skipped the unique room number and the extra charges. A
        numbered list is something to diff against, not prose to skim. The
        extraction happens here, once per run, so Phase 2 and Phase 3 hold
        the model to the same list.
        """
        if self.enable_requirements_ledger and self._requirements is None:
            self._extract_requirements(instructions)
        requirements = self._requirements_for_validation()
        if not requirements:
            return instructions
        return (
            f"{instructions}\n\n"
            "## Requirements the user stated (each is verified after generation)\n\n"
            f"{_requirements_ledger.render_requirements(requirements)}"
        )

    def _requirements_for_validation(self) -> list[dict]:
        """Keep rejected model rules as obligations, independent of LLM extraction.

        Do not insert them into the executable model or turn failed extraction
        into a successful empty ledger. Model losses remain visible even when
        the planner omits them or a checklist item is dropped.
        """
        requirements = list(self._requirements or [])
        next_id = max((item["id"] for item in requirements), default=0) + 1
        for issue in getattr(self.domain_model, "conversion_issues", []) or []:
            if not isinstance(issue, dict):
                continue
            expression = issue.get("expression") or issue.get("original_text") or ""
            source = issue.get("source") or {}
            label = issue.get("name") or issue.get("context") or issue.get("id") or "unnamed rule"
            contract_kind = issue.get("kind") or "constraint"
            target = issue.get("context") or "unknown context"
            if issue.get("method"):
                target += f"::{issue['method']}"
            if contract_kind == "postcondition":
                obligation = "The action must guarantee the intended state or result after successful execution. "
            elif contract_kind == "precondition":
                obligation = "The action must check the intended condition before execution and refuse when it fails. "
            else:
                obligation = "The application must enforce the intended behavior and reject violations. "
            text = (
                f"Recover model constraint '{label}' rejected during conversion. "
                f"Contract: {contract_kind} on {target}. {obligation}"
                "Use the actual model/API relationship names. "
                "Resolve conflicts in favor of the original user specification "
                "and its requirements above; the rejected expression is source "
                "intent, not executable code. "
                f"Original OCL: {expression}. Conversion reason: {issue.get('reason', 'unknown')}. "
                f"Source diagram: {source.get('diagram_title') or source.get('diagram_id') or 'unknown'}; "
                f"element: {source.get('element_id') or 'unknown'}. "
                "Verify the runtime enforcement, not just the presence of a model, "
                "comment, task completion claim, or renamed expression. If the "
                "intended behavior cannot be recovered from the original spec and "
                "model, keep it unresolved rather than inventing a rule."
            )
            requirements.append({
                "id": next_id, "kind": "action" if contract_kind == "postcondition" else "rule",
                "text": text,
                "conversion_issue_id": issue.get("id") or f"conversion-{next_id}",
            })
            next_id += 1
        return requirements

    def _collect_requirement_issues(self) -> list[str]:
        """``requirement:`` blockers for what the user asked for and the code
        does not do. Run 19h35 (2026-09-18) planned the guest-capacity rule and
        shipped without it, and never planned the unique room number or the
        extra charges; nothing checked the app against the request itself.
        """
        if not self.enable_requirements_ledger:
            findings = ([required_check_unverified(
                "original-specification coverage", "requirements ledger is disabled",
            )] if _requirements_ledger.original_request(self._instructions).strip() else [])
            findings.extend(
                required_check_unverified("model conversion recovery", f"{item['text']} Requirements ledger is disabled")
                for item in self._requirements_for_validation() if item.get("conversion_issue_id")
            )
            return findings
        if self._requirements is None:
            self._extract_requirements(self._instructions)
        if self._requirements is None:
            if (_requirements_ledger._is_real_provider(self.client)
                    or getattr(self.domain_model, "conversion_issues", None)):
                return [
                    "requirement unverified: requirement extraction failed; "
                    "the original specification has not been checked. Do not "
                    "treat this as an empty requirement list or verified completion."
                ]
            return []  # Offline clients cannot run the optional LLM judge.
        requirements = self._requirements_for_validation()
        if not requirements:
            return []
        digest = _requirements_ledger.build_app_digest(self.output_dir)
        cache_key = self._requirement_cache_key(requirements)
        verdicts = self._requirement_judgments.get(cache_key)
        cached = verdicts is not None
        if verdicts is None:
            if not self._verification_call_allowed():
                return ["requirement unverified: verification stopped or budget exhausted; original-specification coverage is not verified."]
            verdicts = _requirements_ledger.judge_coverage(
                requirements, digest, self.client, original_spec=self._instructions,
            )
        if verdicts is None:
            return [
                "requirement unverified: the requirements judge returned no verdicts; "
                "coverage of the original specification has not been verified."
            ]
        # Cache successful judgments only, keyed to all source/config bytes, not
        # the clipped judge prompt. An unchanged app must not 'improve' by chance.
        self._requirement_judgments[cache_key] = verdicts
        if len(self._requirement_judgments) > 8:
            self._requirement_judgments.pop(next(iter(self._requirement_judgments)))
        self._requirement_verdicts = _requirements_ledger.verify_evidence(
            verdicts, self.output_dir,
        )
        unknown_ids = {item["id"] for item in self._requirement_verdicts
                       if item["status"] == "unverified"}
        if (cached and unknown_ids and cache_key not in self._requirement_evidence_retries
                and self._verification_call_allowed()):
            # Bad citations need a bounded way to recover without gratuitous
            # code edits. Rejudge only unknown entries once per revision; never
            # reroll missing/verified judgments on unchanged source.
            self._requirement_evidence_retries.add(cache_key)
            focus_paths = [str(item.get("evidence", "")).split(":", 1)[0].strip()
                           for item in self._requirement_verdicts if item["id"] in unknown_ids]
            focus_paths = [path for item in self._requirement_verdicts
                           if item["id"] in unknown_ids
                           for path in item.get("inspection_paths", [])] + focus_paths
            focused_digest = _requirements_ledger.build_app_digest(
                self.output_dir, focus_paths=focus_paths,
            )
            repair = _requirements_ledger.judge_coverage(
                [item for item in requirements if item["id"] in unknown_ids],
                focused_digest, self.client, original_spec=self._instructions,
                previous_verdicts=[item for item in self._requirement_verdicts
                                   if item["id"] in unknown_ids],
            )
            repaired = _requirements_ledger.verify_evidence(repair or [], self.output_dir)
            valid_ids = {item["id"] for item in repaired if item["status"] == "implemented"}
            replacements = {item["id"]: item for item in repair or [] if item["id"] in valid_ids}
            if replacements:
                verdicts = [replacements.get(item["id"], item) for item in verdicts]
                self._requirement_judgments[cache_key] = verdicts
                self._requirement_verdicts = _requirements_ledger.verify_evidence(verdicts, self.output_dir)
        return _requirements_ledger.ledger_issues(self._requirement_verdicts)

    def _collect_frontend_contract_issues(self) -> list[str]:
        return collect_frontend_contract_issues(self.output_dir)

    def _requirement_cache_key(self, requirements: list[dict]) -> str:
        return hashlib.sha256((
            self._workspace_revision() + json.dumps(requirements, sort_keys=True)
        ).encode("utf-8")).hexdigest()

    def _extract_requirements(self, instructions: str) -> None:
        # One initial attempt and one recovery attempt; failed extraction stays
        # unknown instead of being cached forever as a successful empty list.
        if self._requirement_extraction_attempts >= 2 or not self._verification_call_allowed():
            return
        self._requirement_extraction_attempts += 1
        self._requirements = _requirements_ledger.extract_requirements(instructions, self.client)
        if self._requirements is None:
            # extract_requirements has five silent None paths. Run 7aybctis
            # (Qwen) recorded requirements: [] in the recipe, which reads as
            # "the user asked for nothing" rather than "we never looked".
            logger.warning(
                "Requirements ledger: extraction returned nothing on attempt %d "
                "(model %s); requirement verification will not run",
                self._requirement_extraction_attempts, getattr(self.client, "model", "?"),
            )

    def _verification_call_allowed(self) -> bool:
        """Do not start a paid extraction/judgment after a stop or spend cap."""
        return not (
            (self.max_cost_usd is not None and self.client.usage.estimated_cost >= self.max_cost_usd)
            or (self._start_time is not None and time.monotonic() - self._start_time >= self.max_runtime_seconds)
            or (self._should_continue is not None and not self._should_continue())
        )

    def _workspace_revision(self) -> str:
        """Hash source/config state, excluding traces, caches and runtime data."""
        fingerprint = hashlib.sha256()
        workspace = os.path.realpath(self.output_dir)
        extensions = {
            ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".html", ".css",
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".txt", ".lock",
            ".sql", ".rs", ".kt", ".java", ".go", ".c", ".cpp", ".h", ".rb", ".php",
        }
        for rel in sorted(self._workspace_file_list()):
            if os.path.splitext(rel)[1].lower() not in extensions and os.path.basename(rel) != "Dockerfile":
                continue
            path = os.path.realpath(os.path.join(workspace, rel))
            try:
                if os.path.commonpath([workspace, path]) != workspace:
                    continue
                fingerprint.update(rel.encode("utf-8"))
                with open(path, "rb") as source:
                    for chunk in iter(lambda: source.read(65536), b""):
                        fingerprint.update(chunk)
            except (OSError, ValueError):
                fingerprint.update(b"<unreadable>")
            fingerprint.update(b"\0")
        return fingerprint.hexdigest()


    def _collect_ruff_issues(self) -> list[str]:
        """Run ``ruff check`` across the workspace when available.

        Returns a list of concise issue strings. Times out / unparseable
        output are skipped silently (nice-to-have checks). A MISSING ruff
        binary, however, is reported loudly: ``_classify_issue`` promotes
        ruff's undefined-name findings to blockers ("ships green, boots
        dead"), so when ruff is absent that whole class of defect is
        invisible and a "0 blockers" result is not the verification it
        looks like. Found live 2026-09-10: ruff was never in the hosted
        image (only in CI), so this path returned [] for every pilot run —
        including the two that shipped a backend NameError-ing on import.
        """
        import shutil as _shutil
        import subprocess

        ruff_bin = _shutil.which("ruff")
        if not ruff_bin:
            if not getattr(self, "_warned_ruff_missing", False):
                self._warned_ruff_missing = True
                logger.warning(
                    "Phase 3: ruff is not installed on this host — Python "
                    "undefined-name/import checks are SKIPPED, so '0 blockers' "
                    "does not cover import-time NameErrors. Install ruff in the "
                    "image to enable them."
                )
            # A visible (non-blocking) validation note, deliberately phrased
            # without a ruff rule code so _classify_issue keeps it a warning.
            return [
                "validation: ruff is not installed on this host - Python "
                "undefined-name/import checks were skipped (install ruff to "
                "enable them)"
            ]

        try:
            result = subprocess.run(
                [
                    ruff_bin, "check",
                    "--output-format=concise",
                    "--no-cache",
                    "--exit-zero",
                    "--exclude", _SNAPSHOT_DIR,
                    self.output_dir,
                ],
                capture_output=True, text=True, timeout=30,
                env=_safe_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            return [_check_did_not_run("ruff", "timed out after 30s")]
        except OSError as exc:
            return [_check_did_not_run("ruff", f"could not be launched: {exc}")]

        # --exit-zero means findings alone never set a non-zero status, so a
        # non-zero code here is ruff itself failing (unreadable config, a
        # panic). Without this the run reported clean having checked nothing.
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            reason = detail[-1][:200] if detail else f"exit code {result.returncode}"
            return [_check_did_not_run("ruff", reason)]

        lines = [line.strip() for line in (result.stdout or "").strip().splitlines()
                 if line.strip()]
        if not lines:
            return []
        # The cap used to take ruff's first 20 lines, which are sorted by
        # path: on a 585-issue workspace that is always the same scaffold
        # files, and a real F821 late in the alphabet never reached the fix
        # loop at all. Keep every blocker-code line, then spend what is left
        # of the budget on files the LLM actually edited this run.
        touched = self._llm_edited_paths()
        blockers, edited, rest = [], [], []
        for line in lines:
            match = _RUFF_LINE_RE.search(line)
            if match and match.group(1) in _RUFF_BLOCKER_CODES:
                blockers.append(line)
            elif self._ruff_line_path(line) in touched:
                edited.append(line)
            else:
                rest.append(line)
        ordered = blockers + edited + rest
        kept = ordered[:max(_RUFF_MAX_REPORTED, len(blockers))]
        issues = [f"ruff: {line}" for line in kept]
        if len(ordered) > len(kept):
            issues.append(f"ruff: (+{len(ordered) - len(kept)} more issues truncated)")
        return issues

    def _llm_edited_paths(self) -> set[str]:
        """Absolute, normalised paths the LLM wrote to in this run."""
        edited = set()
        for call in self.tool_calls_log:
            if call.get("tool") not in _WRITE_TOOLS_ON_RECORD or call.get("success") is False:
                continue
            path = (call.get("input") or {}).get("path")
            if isinstance(path, str) and path.strip():
                edited.add(os.path.normcase(os.path.normpath(
                    os.path.join(self.output_dir, path.replace("\\", "/")))))
        return edited

    @staticmethod
    def _ruff_line_path(line: str) -> str:
        """The file part of a concise ruff line, or "" when unparseable.

        ``<path>:<line>:<col>: <CODE> <message>`` — a Windows drive letter
        puts an extra colon in the path, so split from the right.
        """
        head = line.rsplit(":", 3)
        if len(head) != 4:
            return ""
        return os.path.normcase(os.path.normpath(head[0]))

    def _collect_tsc_issues(self) -> list[str]:
        """Run ``tsc --noEmit`` for any TypeScript project in the workspace.

        Looks for ``tsconfig.json`` files (skipping the snapshot dir) and
        runs ``tsc --noEmit`` against each project root. A disabled or missing
        compiler is unverified, not a source defect for automatic repair.
        """
        import shutil as _shutil
        import subprocess

        tsconfigs: list[str] = []
        workspace = os.path.realpath(self.output_dir)
        for root, dirs, files in os.walk(self.output_dir):
            retained = []
            for directory in dirs:
                if directory in _RECIPE_EXCLUDED_DIRS or directory.startswith(".besser_"):
                    continue
                try:
                    if os.path.commonpath([workspace, os.path.realpath(os.path.join(root, directory))]) == workspace:
                        retained.append(directory)
                except ValueError:
                    continue
            dirs[:] = retained
            rel_root = os.path.relpath(root, self.output_dir).replace("\\", "/")
            if rel_root.startswith(_SNAPSHOT_DIR):
                continue
            # Skip node_modules — running tsc there is both meaningless
            # and extremely slow.
            if "node_modules" in rel_root.split("/"):
                continue
            if "tsconfig.json" in files:
                try:
                    if os.path.commonpath([workspace, os.path.realpath(os.path.join(root, "tsconfig.json"))]) == workspace:
                        tsconfigs.append(root)
                except ValueError:
                    continue

        if not tsconfigs:
            return []

        global_tsc = _shutil.which("tsc") or _shutil.which("tsc.cmd")
        issues: list[str] = []
        for project_dir in tsconfigs:
            rel = os.path.relpath(project_dir, self.output_dir).replace("\\", "/") or "."
            if not self.enable_toolchain_validation:
                issues.append(required_check_unverified(
                    f"tsc [{rel}]", "toolchain validation is disabled"))
                continue
            deps_installed = os.path.isdir(os.path.join(project_dir, "node_modules"))
            tsc_bin = global_tsc
            if self.allow_shell_tools:
                # Project executables are package-authored code. Only prefer
                # them when shell execution was explicitly authorized.
                local_tsc = os.path.join(project_dir, "node_modules", ".bin",
                                         "tsc.cmd" if os.name == "nt" else "tsc")
                if os.path.isfile(local_tsc):
                    tsc_bin = local_tsc
            setup_allowed = (
                not deps_installed and self.allow_shell_tools
                and os.path.isfile(os.path.join(project_dir, "package.json"))
                and bool(_shutil.which("npm") or _shutil.which("npm.cmd"))
            )
            if setup_allowed:
                issues.append(required_dependency_setup(f"tsc [{rel}]", rel))
            if not tsc_bin:
                if not setup_allowed:
                    issues.append(required_check_unverified(f"tsc [{rel}]", "TypeScript compiler is unavailable"))
                continue
            if not deps_installed and not setup_allowed:
                issues.append(required_check_unverified(f"tsc [{rel}]", "dependencies are not installed; only partial source checks are possible"))
            project_arg, cleanup = self._tsc_project_arg(project_dir, deps_installed)
            try:
                result = subprocess.run(
                    [tsc_bin, "--noEmit", "-p", project_arg],
                    capture_output=True, text=True, timeout=60,
                    cwd=project_dir,
                    env=_safe_subprocess_env(),
                )
            except subprocess.TimeoutExpired:
                issues.append(required_check_unverified(f"tsc [{rel}]", "timed out after 60s"))
                continue
            except OSError as exc:
                issues.append(
                    required_check_unverified(f"tsc [{rel}]", f"could not be launched: {exc}")
                )
                continue
            finally:
                cleanup()

            # tsc emits errors on stdout (not stderr) in the classic
            # ``file(line,col): error TSxxxx: message`` format.
            output = (result.stdout or "").strip().splitlines()
            err_lines = [ln.strip() for ln in output if ln.strip() and "error" in ln.lower()]
            if not err_lines and result.returncode == 0:
                continue
            if not err_lines:
                # Non-zero exit with nothing parseable — a missing typescript
                # install, a broken tsconfig. Reporting clean here is how a
                # frontend that does not compile passes validation.
                detail = (
                    (result.stderr or "").strip().splitlines()
                    + (result.stdout or "").strip().splitlines()
                )
                tail = detail[-1][:200] if detail else f"exit code {result.returncode}"
                issues.append(f"tsc [{rel}]: {tail}")
                continue
            if not deps_installed:
                err_lines = self._demote_tsc_without_deps(err_lines, rel, issues)
                if not err_lines:
                    continue
            # One name repeated 8 times would fill the cap and hide the other
            # 7 distinct names behind "truncated", costing a whole fix round.
            err_lines = self._collapse_repeated_names(err_lines)
            for line in err_lines[:10]:
                issues.append(f"tsc [{rel}]: {line}")
            if len(err_lines) > 10:
                issues.append(f"tsc [{rel}]: (+{len(err_lines) - 10} more errors truncated)")
        return issues

    @classmethod
    def _collapse_repeated_names(cls, err_lines: list[str]) -> list[str]:
        """Keep the first occurrence of each undefined name per file."""
        seen: set[tuple[str, str]] = set()
        kept: list[str] = []
        for line in err_lines:
            match = cls._TSC_UNDEFINED_NAME_RE.match(line.strip())
            if not match:
                kept.append(line)
                continue
            key = (match.group("file"), match.group("name"))
            if key in seen:
                continue
            seen.add(key)
            kept.append(line)
        return kept

    # A relative import names a file the run was supposed to write; a bare
    # one names a package. Only the first is checkable without an install.
    _TSC_MISSING_MODULE_RE = _re.compile(
        r"error TS2307:.*?Cannot find module ['\"](?P<spec>[^'\"]+)['\"]")

    # Written next to the real tsconfig so its relative include/exclude/baseUrl
    # still resolve, then removed.
    _TSC_PROBE_NAME = "tsconfig.besser-probe.json"
    # target/moduleResolution are overridden too: TypeScript 7 REMOVED
    # `target: es5` and `moduleResolution: node`, which every CRA-era
    # scaffold still carries, and a removed option is TS5108 at config
    # time -- the same zero-files-checked abort this probe exists to
    # avoid. Neither option affects TS2304, the only code promoted here.
    _TSC_PROBE_BODY = (
        '{\n  "extends": "./tsconfig.json",\n'
        '  "compilerOptions": {\n'
        '    "types": [], "noEmit": true,\n'
        '    "target": "es2020", "module": "esnext", "moduleResolution": "bundler"\n'
        '  }\n}\n'
    )

    def _tsc_project_arg(self, project_dir: str, deps_installed: bool):
        """Return the ``-p`` target for tsc, plus a cleanup callable.

        A ``types`` entry naming an uninstalled package (``"types":
        ["vite/client"]``, which every Vite scaffold carries) makes tsc abort
        at config resolution: it emits TS2688 and type-checks ZERO files. Run
        36e9c8a6 shipped a frontend with 23 real errors whose entire tsc
        output was that one line, so Phase 3 saw a clean frontend.

        Clearing ``types`` costs nothing when deps are missing — those types
        cannot resolve either way — and lets tsc actually read the source.
        There is no CLI equivalent: ``--types ""`` is TS6044 and
        ``--typeRoots`` does not suppress an explicit ``types`` entry.
        """
        if deps_installed:
            return ".", lambda: None

        probe = os.path.join(project_dir, self._TSC_PROBE_NAME)
        try:
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write(self._TSC_PROBE_BODY)
        except OSError:
            return ".", lambda: None

        def cleanup():
            try:
                os.remove(probe)
            except OSError:
                pass

        return self._TSC_PROBE_NAME, cleanup

    # TS2304 means a name has no binding in scope. That is true whether or not
    # packages are installed -- EXCEPT for names a package contributes as a
    # global type declaration, which are unresolvable only because of the
    # missing install.
    _TSC_PACKAGE_GLOBALS = frozenset({
        # test runners (vitest / jest globals)
        "describe", "it", "test", "expect", "vi", "jest", "suite", "assert",
        "beforeEach", "afterEach", "beforeAll", "afterAll", "afterFile",
        # node
        "process", "Buffer", "__dirname", "__filename", "global", "require",
        "module", "exports", "NodeJS", "globalThis",
        # react UMD global / JSX namespace
        "React", "JSX",
    })

    _TSC_UNDEFINED_NAME_RE = _re.compile(
        r"^(?P<file>[^(]+)\(.*?error TS2304: Cannot find name '(?P<name>[^']+)'"
    )

    @classmethod
    def _is_real_undefined_name(cls, line: str) -> bool:
        """True for a TS2304 that a ``npm install`` would not have fixed."""
        match = cls._TSC_UNDEFINED_NAME_RE.match(line.strip())
        if not match:
            return False
        if match.group("name") in cls._TSC_PACKAGE_GLOBALS:
            return False
        path = match.group("file").replace("\\", "/").lower()
        base = path.rsplit("/", 1)[-1]
        # Test files pull in runner globals we cannot enumerate; skip them.
        if "__tests__" in path or "__mocks__" in path:
            return False
        return not any(
            f".{kind}." in base for kind in ("test", "spec", "stories", "cy")
        )

    def _demote_tsc_without_deps(
        self, err_lines: list[str], rel: str, issues: list[str]
    ) -> list[str]:
        """Keep only the tsc errors that survive a missing ``node_modules``.

        We never run ``npm install`` during validation — it needs network
        the host may not have, and on a proxied corporate network it fails
        outright. So tsc runs against an uninstalled tree, where every
        package import is unresolvable and the type errors cascade from
        there. On the 2026-09-17 hotel run that produced
        ``error TS2688: Cannot find type definition file for 'vite/client'``
        and a "3 blocker-level issues remain — may not run as-is" verdict
        on a frontend that starts and renders perfectly once installed.

        What tsc CAN still tell us truthfully is whether a locally
        referenced file exists: ``import Foo from './components/Foo'``
        where the run never wrote that file is a genuine break either way.
        Everything else is reported, but as advisory — it must not drive
        the Phase-3 fix loop, which would spend its budget "fixing"
        imports that are already correct.
        """
        real: list[str] = []
        demoted = 0
        for line in err_lines:
            match = self._TSC_MISSING_MODULE_RE.search(line)
            if match and match.group("spec").startswith("."):
                real.append(line)
            elif self._is_real_undefined_name(line):
                real.append(line)
            else:
                demoted += 1
                if demoted <= 5:
                    issues.append(f"tsc-advisory [{rel}]: {line}")
        if demoted:
            issues.append(
                f"tsc-advisory [{rel}]: {demoted} type error(s) reported without "
                "node_modules installed — package imports and their types cannot "
                "resolve, so these are advisory, not blockers. Run npm install "
                "before trusting them."
            )
        return real

    def _collect_cargo_issues(self) -> list[str]:
        """Run ``cargo check`` for any Rust crate in the workspace.

        Mirrors ``_collect_tsc_issues`` for the Rust toolchain. Looks
        for ``Cargo.toml`` files at any depth (skipping the snapshot
        dir and any ``target/`` build output) and runs ``cargo check
        --message-format=short`` per crate. Skips silently if ``cargo``
        is not on PATH — matches the soft-skip pattern the bench uses
        when the toolchain isn't installed on the run host.

        ``cargo check`` is used in preference to ``cargo build``: it
        runs the front-end and type-checker without producing artifacts,
        which is what the per-project compile-pass criterion actually
        cares about and is ~3-5× faster.
        """
        import shutil as _shutil
        import subprocess

        cargo_bin = _shutil.which("cargo") or _shutil.which("cargo.exe")
        if not cargo_bin:
            return []

        crates: list[str] = []
        for root, _, files in os.walk(self.output_dir):
            rel_root = os.path.relpath(root, self.output_dir).replace("\\", "/")
            if rel_root.startswith(_SNAPSHOT_DIR):
                continue
            # ``target/`` is the cargo build cache — running cargo
            # inside it is meaningless. Also skip any vendored deps.
            parts = rel_root.split("/")
            if "target" in parts or "vendor" in parts:
                continue
            if "Cargo.toml" in files:
                crates.append(root)

        if not crates:
            return []

        # Redirect cargo's build cache OUT of the user workspace: without
        # this, ``target/`` (thousands of files for a typical axum crate)
        # lands inside the output dir — bloating the download zip — and
        # every check cold-compiles all dependency crates from scratch.
        # A shared per-host cache dir makes repeat checks incremental.
        # Strip secrets from the env handed to cargo (it can execute build.rs /
        # proc-macros from generated crates). Keeps PATH/HOME so cargo still
        # resolves its toolchain + ~/.cargo.
        cargo_env = {**_safe_subprocess_env()}
        cargo_env.setdefault(
            "CARGO_TARGET_DIR",
            os.path.join(tempfile.gettempdir(), "besser_cargo_cache"),
        )

        issues: list[str] = []
        for crate_dir in crates:
            rel = os.path.relpath(crate_dir, self.output_dir).replace("\\", "/") or "."
            try:
                result = subprocess.run(
                    [
                        cargo_bin, "check",
                        "--message-format=short",
                        "--quiet",
                    ],
                    capture_output=True, text=True, timeout=180,
                    cwd=crate_dir,
                    env=cargo_env,
                )
            except subprocess.TimeoutExpired:
                issues.append(_check_did_not_run(f"cargo [{rel}]", "timed out after 180s"))
                continue
            except OSError as exc:
                issues.append(
                    _check_did_not_run(f"cargo [{rel}]", f"could not be launched: {exc}")
                )
                continue

            # cargo emits diagnostics on stderr in short format like:
            #   src/main.rs:12:5: error[E0308]: mismatched types
            err_lines = []
            for ln in (result.stderr or "").splitlines():
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("error") or ": error" in s:
                    err_lines.append(s)
            if not err_lines and result.returncode == 0:
                continue
            if not err_lines:
                # Non-zero exit with no parseable error lines (rare —
                # network failure resolving deps, missing rustc, etc.).
                # Surface a single summary line so the LLM can decide
                # whether to address it.
                summary = (result.stderr or "").strip().splitlines()
                tail = summary[-1] if summary else "cargo check failed with no output"
                issues.append(f"cargo [{rel}]: {tail[:200]}")
                continue
            for line in err_lines[:10]:
                issues.append(f"cargo [{rel}]: {line}")
            if len(err_lines) > 10:
                issues.append(
                    f"cargo [{rel}]: (+{len(err_lines) - 10} more errors truncated)"
                )
        return issues

    def _collect_kotlinc_issues(self) -> list[str]:
        """Run ``kotlinc`` against ``.kt`` sources in the workspace.

        Kotlin / Spring projects from Phase 0.5 ship with a Gradle
        build, but invoking the Gradle wrapper would pull the network
        on first run and is far too slow for an inner-loop check. We
        instead run the standalone ``kotlinc`` compiler on the
        ``src/main/kotlin`` tree with no class-path (Spring annotations
        and missing imports still surface as compile errors).

        Limitations (documented for the caller, not bugs):
          - Type references to external Maven deps will show up as
            unresolved-reference errors. That's the right call here —
            it tells the LLM the import / dep listing is wrong, and
            the project will fail Gradle in the same way.
          - We only walk one source root per Kotlin module to keep
            the invocation cheap. Multi-module projects compile one
            module at a time.

        Soft-skips when ``kotlinc`` is not on PATH (no warning in the
        recipe — the bench host either has it or doesn't).
        """
        import shutil as _shutil
        import subprocess

        kotlinc_bin = (
            _shutil.which("kotlinc")
            or _shutil.which("kotlinc.bat")
            or _shutil.which("kotlinc.cmd")
        )
        if not kotlinc_bin:
            return []

        # Locate Kotlin source roots. We look for ``src/main/kotlin``
        # under any directory containing a Gradle build file, which is
        # the convention every Phase 0.5 Kotlin template lands in.
        modules: list[str] = []
        for root, dirs, files in os.walk(self.output_dir):
            rel_root = os.path.relpath(root, self.output_dir).replace("\\", "/")
            if rel_root.startswith(_SNAPSHOT_DIR):
                continue
            parts = rel_root.split("/")
            if "build" in parts or ".gradle" in parts:
                # Don't recurse into build output / Gradle caches.
                dirs[:] = []
                continue
            has_gradle = (
                "build.gradle.kts" in files
                or "build.gradle" in files
            )
            if not has_gradle:
                continue
            src_main_kotlin = os.path.join(root, "src", "main", "kotlin")
            if os.path.isdir(src_main_kotlin):
                modules.append(src_main_kotlin)

        if not modules:
            return []

        issues: list[str] = []
        for src_root in modules:
            module_rel = (
                os.path.relpath(src_root, self.output_dir).replace("\\", "/") or "."
            )
            # Collect every .kt file under the source root. Limited to
            # 200 sources per invocation to keep the command line in
            # bounds on Windows; if a project exceeds that, the rest
            # are skipped (and the LLM still sees the first batch).
            kt_files: list[str] = []
            for kt_root, _, kt_files_in_dir in os.walk(src_root):
                for fname in kt_files_in_dir:
                    if fname.endswith(".kt"):
                        kt_files.append(os.path.join(kt_root, fname))
                        if len(kt_files) >= 200:
                            break
                if len(kt_files) >= 200:
                    break
            if not kt_files:
                continue

            try:
                result = subprocess.run(
                    [kotlinc_bin, "-nowarn", "-d", os.devnull, *kt_files],
                    capture_output=True, text=True, timeout=180,
                    cwd=self.output_dir,
                    env=_safe_subprocess_env(),
                )
            except (subprocess.TimeoutExpired, OSError):
                continue

            # kotlinc reports diagnostics on stderr as
            #   /abs/path/Foo.kt:12:5: error: unresolved reference: Bar
            err_lines = []
            for ln in (result.stderr or "").splitlines():
                s = ln.strip()
                if not s or ": warning:" in s:
                    continue
                if ": error:" in s or s.startswith("error:"):
                    # Strip the absolute path prefix so the LLM sees
                    # the location relative to the workspace.
                    err_lines.append(
                        s.replace(self.output_dir + os.sep, "")
                         .replace(self.output_dir + "/", "")
                    )
            if not err_lines and result.returncode == 0:
                continue
            if not err_lines:
                tail = (result.stderr or "").strip().splitlines()
                summary = tail[-1] if tail else "kotlinc failed with no output"
                issues.append(f"kotlinc [{module_rel}]: {summary[:200]}")
                continue
            for line in err_lines[:10]:
                issues.append(f"kotlinc [{module_rel}]: {line}")
            if len(err_lines) > 10:
                issues.append(
                    f"kotlinc [{module_rel}]: "
                    f"(+{len(err_lines) - 10} more errors truncated)"
                )
        return issues

    # ==================================================================
    # Snapshot / Rollback
    # ==================================================================

    def _create_snapshot(self) -> None:
        """Create a lightweight snapshot of the output directory after Phase 1."""
        snapshot_path = os.path.join(self.output_dir, _SNAPSHOT_DIR)
        try:
            if os.path.exists(snapshot_path):
                shutil.rmtree(snapshot_path)

            # Copy everything except the snapshot dir and the run bookkeeping
            # a rollback must never revert (see _ROLLBACK_PRESERVED).
            for item in os.listdir(self.output_dir):
                if item in _ROLLBACK_PRESERVED:
                    continue
                src = os.path.join(self.output_dir, item)
                dst = os.path.join(snapshot_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)

            logger.info("Snapshot created at %s", snapshot_path)
        except Exception as e:
            logger.warning("Failed to create snapshot: %s", e)

    def _restore_snapshot(self) -> bool:
        """Restore the output directory from the post-Phase-1 snapshot.

        Returns True only if the workspace now holds the snapshot's content.

        The current tree is *moved* aside and only discarded once the restore
        succeeds; if anything fails it is moved back. Deleting first and copying
        after leaves a half-erased workspace with no way back when the copy dies
        partway (full disk, locked file), which the run then packages as the
        deliverable.
        """
        snapshot_path = os.path.join(self.output_dir, _SNAPSHOT_DIR)
        if not os.path.isdir(snapshot_path):
            logger.warning("No snapshot to restore from")
            return False

        discard_path = os.path.join(self.output_dir, _ROLLBACK_DISCARD_DIR)
        moved: list[tuple[str, str]] = []
        try:
            if os.path.exists(discard_path):
                shutil.rmtree(discard_path)
            os.makedirs(discard_path, exist_ok=True)

            # Park the current tree instead of deleting it. Same filesystem,
            # so os.replace is a rename and costs nothing.
            for item in os.listdir(self.output_dir):
                if item in _ROLLBACK_PRESERVED:
                    continue
                src = os.path.join(self.output_dir, item)
                dst = os.path.join(discard_path, item)
                os.replace(src, dst)
                moved.append((src, dst))

            for item in os.listdir(snapshot_path):
                src = os.path.join(snapshot_path, item)
                dst = os.path.join(self.output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        except Exception as e:
            logger.error("Rollback failed, reverting to the pre-rollback tree: %s", e)
            try:
                # Undo the partial restore, then put the parked tree back.
                for _src, dst in moved:
                    landed = os.path.join(self.output_dir, os.path.basename(dst))
                    if os.path.isdir(landed):
                        shutil.rmtree(landed, ignore_errors=True)
                    elif os.path.isfile(landed):
                        os.remove(landed)
                for src, dst in moved:
                    os.replace(dst, src)
                shutil.rmtree(discard_path, ignore_errors=True)
            except Exception as revert_exc:
                # Both directions failed. Say so loudly and leave the parked
                # copy in place — it is the only intact tree left.
                logger.error(
                    "Could not revert the rollback either; the Phase 2 output "
                    "is preserved under %s: %s", _ROLLBACK_DISCARD_DIR, revert_exc,
                )
            return False

        shutil.rmtree(discard_path, ignore_errors=True)
        logger.info("Restored from snapshot")
        return True

    def _remove_snapshot(self) -> None:
        """Clean up the snapshot directory."""
        snapshot_path = os.path.join(self.output_dir, _SNAPSHOT_DIR)
        if os.path.isdir(snapshot_path):
            try:
                shutil.rmtree(snapshot_path)
            except Exception:
                pass

    # ==================================================================
    # Interactive error feedback
    # ==================================================================

    def fix_error(self, error_message: str) -> str:
        """
        Fix a user-reported error. The LLM analyzes it and either:
        - Explains what the user should do (environment issues)
        - Fixes the code (code bugs)

        Returns the LLM's explanation/summary of what it did.
        """
        if not error_message or not error_message.strip():
            raise ValueError("Error message cannot be empty")

        # Check if this looks like an actual error (not a question)
        error_keywords = {"error", "traceback", "exception", "failed", "fatal",
                          "cannot", "not found", "denied", "refused", "timeout",
                          "syntax", "import", "module", "attribute", "type"}
        lower = error_message.lower()
        is_error = any(kw in lower for kw in error_keywords)

        if not is_error:
            return (
                "That doesn't look like an error message. "
                "Paste the actual error/traceback from your terminal."
            )

        # Check if we already tried to fix this exact error.
        # Extract the actual error line (last meaningful line), not the
        # traceback boilerplate which looks similar across different errors.
        lines = [ln.strip() for ln in error_message.strip().splitlines() if ln.strip()]
        error_line = ""
        for line in reversed(lines):
            # Skip traceback frame lines and empty lines
            if line and not line.startswith(("File ", "^", "Traceback", "---")):
                error_line = line[:200]
                break
        if not error_line:
            error_line = error_message.strip()[:200]

        already_tried = False
        for prev in self._previous_errors:
            if prev == error_line:
                already_tried = True
                break
        self._previous_errors.append(error_line)

        if self._start_time is None:
            self._start_time = time.monotonic()

        retry_note = ""
        if already_tried:
            retry_note = (
                "\n\nIMPORTANT: I already tried to fix this same error before but it persists. "
                "This strongly suggests it's an ENVIRONMENT issue, not a code bug. "
                "Do NOT modify code again. Just explain what the user needs to do "
                "in their environment (restart, rebuild, clear cache, reset database, etc.).\n"
            )

        fix_prompt = (
            "The user ran the generated code and got this error:\n\n"
            f"```\n{error_message}\n```\n\n"
            f"{retry_note}"
            "FIRST: Analyze whether this is a CODE bug or an ENVIRONMENT issue.\n\n"
            "If ENVIRONMENT (stale DB, old Docker volume, wrong env var, port conflict):\n"
            "→ Do NOT modify code. Just explain what the user should do.\n\n"
            "If CODE BUG (syntax error, wrong import, logic error, bad config):\n"
            "→ Fix with modify_file. Explain what you changed.\n\n"
            "ALWAYS end with a clear summary of what you did or what the user should do."
        )
        system = (
            "You are debugging an error. Not all errors need code fixes. "
            "If it's an environment issue (stale data, 'already exists', "
            "'connection refused'), explain what the user should do. "
            "If it's a code bug, fix it. ALWAYS explain what you did."
        )
        messages: list[dict] = [{"role": "user", "content": fix_prompt}]

        max_fix_turns = 10
        llm_explanation = ""

        for turn in range(max_fix_turns):
            self.total_turns += 1

            if self._start_time is not None:
                elapsed = time.monotonic() - self._start_time
                if elapsed > self.max_runtime_seconds:
                    break

            try:
                response = self._chat_with_pending_force(system, messages)
            except Exception as e:
                logger.error("Fix cycle API call failed: %s", e)
                break

            # Fix-cycle calls can trigger the outage fallback too.
            self._notify_model_switch()

            # Capture LLM's text explanation
            for block in response.get("content", []):
                if hasattr(block, "text") and block.text:
                    llm_explanation = block.text

            if response["stop_reason"] == "end_turn":
                logger.info("Fix cycle completed after %d turns", turn + 1)
                break

            if response["stop_reason"] == "tool_use":
                messages.append({"role": "assistant", "content": response["content"]})
                tool_blocks = [
                    block for block in response["content"]
                    if hasattr(block, "type") and block.type == "tool_use" and getattr(block, "name", None)
                ]
                tool_results = self._execute_tool_blocks(tool_blocks, self.total_turns - 1)
                messages.append({"role": "user", "content": tool_results})

                # Per-file modify-loop guard (mirrors Phase 2). The fix
                # cycle is the most common offender — the LLM gets a
                # single error to fix and starts dribbling out one-line
                # modify_file calls instead of rewriting the file.
                if self._apply_edit_loop_guards(messages, where="fix cycle"):
                    break
            else:
                break

        return llm_explanation or "Fix cycle completed."

    # ==================================================================
    # Delegating methods
    # ==================================================================

    def _build_system_prompt(
        self,
        instructions: str,
        scoped_issues: list[str] | None = None,
        gap_tasks: list[str] | None = None,
    ) -> str:
        """Delegate to prompt_builder module.

        ``instructions`` is the user's verbatim request — embedded in the
        prompt so the LLM plans its own work. ``scoped_issues`` are
        validator findings from Phase 1 that the LLM must address as
        concrete bugs (not user requests). ``gap_tasks`` is the optional
        focused checklist produced by the cheap gap-analyzer LLM call.
        """
        # Optionally inline small scaffold files (BESSER_LLM_INLINE_SCAFFOLD=1)
        # to save the first read_file turns. OFF by default since 2026-09-17:
        # the copy is a per-run constant, so after the model edits a file it is
        # stale, and quoting from it is what produced the modify_file misses.
        # Like other coding agents, file text now reaches the model only
        # through read_file, which is current; reads batch four to a turn.
        scaffold_snapshot = ""
        if (
            os.environ.get("BESSER_LLM_INLINE_SCAFFOLD", "0").lower()
            in ("1", "true")
            and (self._generator_used or self._phase0_5_files)
        ):
            try:
                scaffold_snapshot = build_scaffold_snapshot(self.output_dir)
                # Inlined files count as read: a miss on any other file is a
                # quote from memory and the executor says so.
                self.executor.mark_known(
                    _re.findall(r"^### `(.+?)`$", scaffold_snapshot, _re.M)
                )
            except Exception:
                logger.debug("Scaffold snapshot build failed", exc_info=True)

        # Exact backend endpoint manifest so the LLM-authored frontend targets
        # real routes instead of reconstructing (and drifting from) them. Best
        # effort — never let a parse failure abort prompt construction.
        endpoint_manifest = ""
        try:
            endpoint_manifest = build_endpoint_manifest(self.output_dir)
        except Exception:
            logger.debug("Endpoint manifest build failed", exc_info=True)

        return build_system_prompt(
            domain_model=self.domain_model,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            inventory=self._inventory,
            instructions=instructions,
            scoped_issues=scoped_issues or [],
            gap_tasks=gap_tasks or [],
            max_turns=self.max_turns,
            object_model=self.object_model,
            state_machines=self.state_machines,
            quantum_circuit=self.quantum_circuit,
            bpmn_model=self.bpmn_model,
            nn_model=self.nn_model,
            primary_kind=self.primary_kind,
            scaffold_snapshot=scaffold_snapshot,
            endpoint_manifest=endpoint_manifest,
            requirements=_requirements_ledger.render_requirements(self._requirements_for_validation()),
            # ``_modify_mode`` is False on the run()/resume() paths, so the
            # from-scratch prompt stays byte-identical; only ``modify()``
            # flips it to prepend the "preserve what works" directive.
            modify_mode=self._modify_mode,
        )

    def _maybe_compact(self, messages: list[dict]) -> list[dict]:
        """Delegate to compaction module.

        When history eviction is enabled (``BESSER_LLM_HISTORY_EVICTION=1``),
        first run a lossless checkpoint eviction: stub stale write_file/read_file
        bodies in older messages (the files are on disk, re-readable). This runs
        ONLY at the compaction checkpoint (when history already exceeds the
        threshold), never per turn — so it doesn't repeatedly bust the prompt
        cache. It's lighter-touch than summarization and often drops the history
        back under the threshold so no summarize is needed; if not, the summarize
        below still runs on top. Gated OFF by default and unverified live — see
        history_eviction.py.
        """
        model = getattr(self.client, "model", None)
        # The reserve must match the output the model is actually ALLOWED to
        # produce this run. The from-scratch and modify paths raise
        # client.max_tokens to FROM_SCRATCH_MAX_TOKENS (32_768), which is
        # double the COMPACT_RESERVE_TOKENS default - so a constant reserve
        # leaves only half the headroom the response may need.
        reserve = max(
            COMPACT_RESERVE_TOKENS, int(getattr(self.client, "max_tokens", 0) or 0)
        )
        if _HISTORY_EVICTION_ENABLED and _estimate_tokens(messages) >= effective_threshold(
            model, reserve=reserve
        ):
            messages, evicted = evict_stale_file_bodies(messages)
            if evicted:
                self._eviction_count = getattr(self, "_eviction_count", 0) + 1
                logger.info(
                    "Checkpoint eviction: stubbed %d stale file body/bodies "
                    "(history now ~%d tokens)", evicted, _estimate_tokens(messages),
                )
        result, did_compact = maybe_compact(
            messages=messages,
            tool_calls_log=self.tool_calls_log,
            output_dir=self.output_dir,
            domain_model=self.domain_model,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            state_machines=self.state_machines,
            object_model=self.object_model,
            quantum_circuit=self.quantum_circuit,
            bpmn_model=self.bpmn_model,
            nn_model=self.nn_model,
            primary_kind=self.primary_kind,
            # Clamps the threshold to the model's context window — the
            # fixed default overflows genuinely small local models long
            # before it trips. See HARNESS_LIMITS_AUDIT.md for why the
            # window table must never guess LOW.
            model=model,
            reserve=reserve,
        )
        if did_compact:
            self._compaction_count += 1
        return result

    def _summarize_messages(self, messages: list[dict]) -> str:
        """Delegate to compaction module."""
        return _summarize_messages(messages, self.tool_calls_log, self.output_dir)

    # ==================================================================
    # Model-switch visibility
    # ==================================================================

    def _notify_model_switch(self) -> None:
        """Surface a mid-run model change to the progress channel.

        The provider's outage fallback (``OpenAIProvider._activate_fallback``)
        swaps the client's model sticky-for-the-run when the primary
        endpoint stays down past the retry budget. That happens inside a
        ``chat``/``chat_stream`` call, so the orchestrator only sees it
        afterwards: compare ``self.client.model`` against the last value
        we saw and, on change, emit the ``__model_switch__`` sentinel via
        ``on_progress`` (the SSE runner translates it into a
        ``model_update`` event). Cheap enough to call after every LLM
        call; a no-op when nothing changed.
        """
        current = getattr(self.client, "model", None)
        if not current or current == self._last_seen_model:
            return
        logger.info(
            "LLM model changed mid-run: %s -> %s",
            self._last_seen_model, current,
        )
        self._last_seen_model = current
        if self.on_progress:
            try:
                self.on_progress(0, "__model_switch__", current)
            except Exception:
                logger.debug(
                    "on_progress failed for model switch", exc_info=True
                )

    # ==================================================================
    # Streaming
    # ==================================================================

    def _call_streaming(self, system: str, messages: list[dict]) -> dict:
        collected_content = []
        stop_reason = "end_turn"
        for event in self.client.chat_stream(
            system=system, messages=messages, tools=self.tools,
        ):
            if event["type"] == "text_delta" and self.on_text:
                self.on_text(event["text"])
            elif event["type"] == "message_done":
                stop_reason = event.get("stop_reason", "end_turn")
                if event.get("content"):
                    collected_content = event["content"]
        return {"stop_reason": stop_reason, "content": collected_content}

    # ==================================================================
    # Loop detection
    # ==================================================================

    def _collect_model_contract_issues(self) -> None:
        """Promote the domain model's constructibility warnings to blockers.

        ``DomainModel.validate`` only warns about a mandatory creation cycle:
        an association with 1..1 on both ends is legal UML, and BESSER's own
        ``user_reference_domain_model`` ships three of them. Here the intent
        IS to generate a CRUD API, and a cycle makes that API unusable — live
        2026-09-18, ``BookingCreate`` required a ReservedRoom id while
        ``ReservedRoomCreate`` required a Booking id, so the delivered app
        served 69 paths and could not create either class.

        Recorded once, before Phase 1, and deliberately NOT part of
        ``_collect_validation_issues``: the Phase 3 fix loop edits code, and
        no edit to the generated code can repair the specification it was
        generated from. Surfacing it early is the whole value.
        """
        if self.domain_model is None:
            return
        try:
            result = self.domain_model.validate(raise_exception=False)
        except Exception:
            logger.debug("Domain-model validation raised; skipping", exc_info=True)
            return
        for warning in result.get("warnings", []):
            if not warning.lower().startswith("mandatory creation cycle"):
                continue
            issue = _classify_issue(f"model contract: {warning}")
            self._validation_issues.append(issue)
            logger.warning("Phase 0: %s", issue.message)
            self._trace.write(
                EVENT_VALIDATION_ISSUE, phase="phase0",
                severity=issue.severity, message=issue.message,
            )

    def _workspace_file_list(self) -> list[str]:
        """Every file in the output tree, relative and ``/``-separated.

        Feeds the gap analyser's path repair. Unlike the inventory string
        (capped at 30 entries) this is the complete list, which is the
        point: the paths a planner invents are the ones it could not see.
        """
        files: list[str] = []
        try:
            for root, dirs, fnames in os.walk(self.output_dir):
                dirs[:] = [d for d in dirs if d not in _RECIPE_EXCLUDED_DIRS]
                for f in fnames:
                    if f.startswith(".besser_"):
                        continue
                    rel = os.path.relpath(os.path.join(root, f), self.output_dir)
                    files.append(rel.replace("\\", "/"))
        except Exception:
            # Path repair is a best-effort assist, never a run blocker.
            logger.debug("Workspace walk failed; skipping path repair", exc_info=True)
            return []
        return files

    def _chat_with_pending_force(self, system: str, messages: list[dict]) -> dict:
        """One chat call. When an escalation asked for a specific tool, make
        that request non-streaming with ``force_tool`` (only clients whose
        ``chat`` accepts it; a plain client just gets the message)."""
        messages = without_rejected_edit_drafts(messages)
        force = self._force_tool_next
        self._force_tool_next = None
        if force and self._client_supports_structured_chat():
            return self.client.chat(
                system=system, messages=messages, tools=self.tools, force_tool=force,
            )
        if self.use_streaming and self.on_text and hasattr(self.client, "chat_stream"):
            return self._call_streaming(system, messages)
        return self.client.chat(system=system, messages=messages, tools=self.tools)

    # Repeats of an edit the executor already rejected, per path. Two live
    # runs on 2026-09-18 (0c537a4e: 13 identical misses; 57160293: 16
    # identical no-ops) alternated read_file / modify_file for ~30 turns while
    # every advisory guard was ignored or never fired. Aider stops after three
    # reflections and hands the prompt to a human; headless, the runtime has
    # to change editing strategy, not permanently close the file before the
    # model has a way to recover. Ignoring recovery remains bounded.
    _REPEAT_FORCE_AT = 3
    _REPEAT_STOP_AT = 7

    def _apply_edit_loop_guards(self, messages: list[dict], *, where: str) -> bool:
        """Per-file modify streak + repeat-rejection escalation. True = stop.

        One mechanism with three callers. It was pasted into Phase 2's loop
        and the fix cycle and simply omitted from ``_invoke_phase3_fix_loop``,
        so the bounded repair loop - the one place a model is asked to fix its
        own mistakes on a 10-turn budget - ran on ``_is_stuck`` alone, the
        weakest of the three, while the README claimed recovery was shared
        across phases. Only the low-level executor ladder actually was.
        """
        stuck_path = self._consecutive_modify_on_same_file()
        if stuck_path is not None:
            logger.warning(
                "Per-file modify loop (%s): %d consecutive edits on %s - "
                "injecting reminder", where, self._PER_FILE_MODIFY_THRESHOLD, stuck_path,
            )
            messages.append({"role": "user", "content": [
                {"type": "text", "text": self._build_modify_loop_reminder(stuck_path)}]})
            # Don't re-fire while the model is still on the same file.
            self._last_modify_warning_path = stuck_path
        elif self._recent_modify_targets:
            last_tool, last_path = self._recent_modify_targets[-1]
            if last_tool != "modify_file" or last_path != self._last_modify_warning_path:
                # Streak broken; a fresh one on this path may warn again.
                self._last_modify_warning_path = None
        return self._escalate_repeat_rejection(messages)

    def _escalate_repeat_rejection(self, messages: list[dict]) -> bool:
        """Act on ``executor.last_repeat``. Returns True when the caller's
        loop must stop."""
        hit = getattr(self.executor, "last_repeat", None)
        if not hit:
            return False
        path, seen = hit
        if self._repeat_escalations.get(path) == seen:
            return False
        self._repeat_escalations[path] = seen
        tool = next((item.get("tool") for item in reversed(self._recent_tool_failures)
                     if item.get("tool") in _EDIT_TOOLS
                     and str(item.get("path", "")).replace("\\", "/").strip() == path),
                    "modify_file")
        if seen >= self._REPEAT_STOP_AT:
            logger.warning(
                "Stuck edit loop: the same rejected %s on %s was sent %d "
                "times; ending the phase", tool, path, seen,
            )
            self._phase2_stop_reason = "stuck_edit_loop"
            return True
        if seen >= self._REPEAT_FORCE_AT:
            # Only steer toward the range editor when the failing strategy is
            # text quotation. Sending a repeating range edit back through
            # read -> replace_file_lines is the loop it is already in.
            #
            # _force_tool_next is a single slot the executor's recovery ladder
            # also writes, earlier in the same turn. An unconditional
            # assignment here silently discarded that hint - including the
            # "go back to modify_file" reversal - purely by write order. The
            # executor saw the actual refusal, so leave its choice alone.
            if self._force_tool_next is None:
                self._force_tool_next = "read_file" if tool == "modify_file" else None
            strategy = (
                "Read the target block, then use replace_file_lines with the returned "
                "read_id and inclusive line numbers."
                if tool == "modify_file" else
                "Re-selecting the same lines will fail the same way. Correct new_text "
                "itself — complete lines, real indentation, balanced brackets — or "
                "select the whole enclosing block."
            )
            text = (
                f"<system-reminder>{tool} on `{path}` was rejected {seen} times "
                f"with the same arguments; the executor will not apply it. {strategy} "
                "The file remains editable. For an already-present change, verify "
                "behavior instead of inserting it again. A rejected edit is not "
                "completion; do not mark it done or drop the requirement."
                "</system-reminder>"
            )
        else:
            return False
        logger.warning("Repeat rejection on %s (%d): %s", path, seen, text[:80])
        messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        return False

    def _record_full_tool_input(
        self, turn: int, tool_name: str, tool_input: object, success: bool, status: str,
    ) -> None:
        """Append the UNTRUNCATED input of a write tool to the sidecar. The
        trace, checkpoint and recipe are bounded by ``_sanitize_for_log``;
        this is where a run's real edits can be read back and cut into
        fixtures."""
        try:
            row = {"turn": turn, "tool": tool_name, "success": success,
                   "status": status, "input": tool_input}
            with open(os.path.join(self.output_dir, TOOL_INPUTS_FILENAME), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            logger.debug("tool-input sidecar write failed", exc_info=True)

    def _is_stuck(self) -> bool:
        recent = self._recent_tool_calls[-self._LOOP_THRESHOLD:]
        return (
            len(recent) >= self._LOOP_THRESHOLD
            and len({key for key, _ in recent}) == 1
            and not any(ok for _, ok in recent)
        )

    def _consecutive_modify_on_same_file(self) -> str | None:
        """Return the file path being repeatedly modified, or None.

        Fires when the tail of the recent tool history is
        ``_PER_FILE_MODIFY_THRESHOLD`` ``modify_file`` calls on the SAME
        (normalised) path that were ALL refused (the executor resets its
        miss count on a successful edit, so N good edits to one file never
        fire). A ``read_file`` on that same path does NOT break the streak:
        re-reading the file you cannot edit is the flail's own rhythm — live
        run a5dce952 alternated modify/read on one file for 38 pairs, 85
        turns and $0.70 while this guard stayed silent. Any other tool, and
        a read of a DIFFERENT file, still breaks it: those are real movement.

        Resets / suppresses repeat firing: once we've warned about a
        path, ``_last_modify_warning_path`` is set; subsequent identical
        streaks return None until the LLM either switches files or
        switches tools.
        """
        n = self._PER_FILE_MODIFY_THRESHOLD
        # The most recent edit call fixes which file the streak is about.
        path = next(
            (p for tool, p in reversed(self._recent_modify_targets)
             if tool in _EDIT_TOOLS),
            None,
        )
        if path is None:
            return None
        # Walk back over modify calls on that file, stepping over a re-read
        # of the SAME file (part of the flail — see the note at the
        # recording site). Anything else, including a read of a different
        # file, is real movement and ends the streak.
        streak = 0
        for tool, entry in reversed(self._recent_modify_targets):
            if tool in _EDIT_TOOLS and entry == path:
                streak += 1
            elif tool == "read_file" and entry == path:
                continue
            else:
                break
        if streak < n:
            return None
        if path is None:
            # modify_file without a parseable path argument — skip.
            return None
        if self.executor.consecutive_modify_misses(path) < n:
            # Successful edits are ordinary work on one file, not a flail.
            return None
        if path == self._last_modify_warning_path:
            # Already warned about this streak; wait for a real change
            # of file or tool before firing again.
            return None
        return path

    def _build_modify_loop_reminder(self, path: str) -> str:
        """Name the actual refusal reason; a syntax rejection is not a text miss."""
        n = self._PER_FILE_MODIFY_THRESHOLD
        failure = next((item for item in reversed(self._recent_tool_failures)
                        if item.get("tool") in _EDIT_TOOLS
                        and str(item.get("path", "")).replace("\\", "/").strip() == path), {})
        tool = failure.get("tool", "modify_file")
        error = str(failure.get("error", ""))[:400]
        kind = failure.get("rejection_kind")
        if kind == "syntax_error" or "syntax" in error.lower():
            advice = (
                "The proposed edit was refused by the syntax guard; this is not evidence that old_text failed to match. "
                "The proposal was NOT applied. Use the CURRENT ON-DISK excerpt, or read_file around the enclosing "
                "function/try/except block. Preserve its complete indentation and control-flow structure; "
                "do not copy the rejected would_write proposal as current source. Correct the replacement, "
                "then retry one focused edit."
            )
        else:
            advice = (
                "Use the reported reason before retrying. For a missing/ambiguous target, call read_file "
                "on the affected region, then switch to replace_file_lines with its read_id "
                "and exact inclusive line numbers instead of quoting old_text again. "
                "For another refusal, resolve that specific guard instead of repeating unchanged arguments. "
                "Verify any already-present change from current source before marking it done."
            )
        if tool == "replace_file_lines":
            # Re-reading and range-editing again is what just failed N times,
            # so do not send the model back around that same loop.
            advice = (
                "Re-reading and selecting the same range again is what just failed. "
                "The refusal reason above is about the replacement text, not the line "
                "numbers: correct new_text (complete lines, real indentation, balanced "
                "brackets), or select the whole enclosing block. If the change is "
                "already present, verify the behavior instead of editing again."
            )
        return (
            f"<system-reminder>Your last {n} {tool} calls on `{path}` "
            f"were refused. Latest reason: {error or 'inspect the tool response'}. "
            f"{advice} Do NOT rewrite `{path}` from memory.</system-reminder>"
        )

    # ==================================================================
    # Recipe
    # ==================================================================

    @staticmethod
    def _load_recipe_history(recipe_path: str) -> list[dict]:
        """Read the session history from an existing recipe, best-effort.

        Legacy recipes (written before the history field existed) get one
        entry synthesized from their own instructions + tool log, so the
        first modify after this ships still sees the seed run.
        """
        if not os.path.isfile(recipe_path):
            return []
        try:
            with open(recipe_path, "r", encoding="utf-8") as fh:
                prior = json.load(fh)
        except Exception:
            return []
        if not isinstance(prior, dict):
            return []
        history = prior.get("history")
        if isinstance(history, list) and history:
            return [h for h in history if isinstance(h, dict)]
        # Legacy recipe — synthesize the seed entry.
        instructions = prior.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            return []
        touched = sorted({
            (tc.get("input") or {}).get("path")
            for tc in prior.get("tool_calls", []) or []
            if isinstance(tc, dict)
            and tc.get("tool") in ("write_file", "modify_file", "replace_file_lines")
            and isinstance((tc.get("input") or {}).get("path"), str)
        })
        return [{
            # bounded: recipe history display only
            "instructions": instructions[:300],
            "saved_at": "",
            "mode": "create",
            "files_touched": touched[:20],
        }]

    def _save_recipe(self, instructions: str, elapsed: float) -> None:
        # Build file manifest. Dependency / build directories are pruned:
        # an LLM-run ``npm install`` would otherwise put thousands of
        # node_modules entries in the manifest, ballooning the recipe
        # past the SSE embed cap (a production run hit 4.9 MB and the
        # whole recipe was dropped from the done event).
        output_files = []
        generator_files = self.executor._generator_files if hasattr(self.executor, '_generator_files') else set()
        # ``source`` records who CREATED the file, and resume re-seeds the
        # scaffold guardrail from it, so it must keep its two values. It
        # therefore cannot answer "did the LLM change this?": run 7aybctis
        # landed 13 edits and still reported from_llm=1, because all but one
        # were edits to generator files. That is the number that says which
        # work regeneration would overwrite, so record it separately.
        llm_edited = self._llm_edited_paths()
        try:
            for root, dirs, fnames in os.walk(self.output_dir):
                dirs[:] = [d for d in dirs if d not in _RECIPE_EXCLUDED_DIRS]
                for f in fnames:
                    if f.startswith(".besser_"):
                        continue
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, self.output_dir).replace("\\", "/")
                    entry = {
                        "path": rel,
                        "size": os.path.getsize(full),
                        "source": "generator" if rel in generator_files else "llm",
                    }
                    if os.path.normcase(os.path.normpath(full)) in llm_edited:
                        entry["llm_modified"] = True
                    output_files.append(entry)
        except Exception:
            pass

        # Build model summary. Every field is populated best-effort —
        # a state-machine-only or agent-only run still gets a recipe,
        # it just doesn't claim there were classes/enums/associations
        # that weren't actually part of the project.
        recipe_model: dict[str, Any] = {"primary_kind": self.primary_kind}
        if self.domain_model is not None:
            try:
                recipe_model.update({
                    "name": getattr(self.domain_model, "name", None),
                    "classes": [c.name for c in self.domain_model.get_classes()],
                    "enumerations": [e.name for e in self.domain_model.get_enumerations()],
                    "associations": len(self.domain_model.associations),
                })
            except Exception:
                # Domain model is present but malformed — don't fail
                # the whole recipe write over it.
                logger.debug("Skipping domain model summary in recipe (malformed)")
        if self.state_machines:
            recipe_model["state_machines"] = [
                getattr(sm, "name", "unnamed") for sm in self.state_machines
            ]
        if self.agent_model is not None:
            recipe_model["agent_present"] = True
        if self.gui_model is not None:
            recipe_model["gui_present"] = True
        if self.quantum_circuit is not None:
            recipe_model["quantum_present"] = True
        if self.bpmn_model is not None:
            recipe_model["bpmn"] = {
                "name": getattr(self.bpmn_model, "name", None),
                "processes": len(getattr(self.bpmn_model, "processes", []) or []),
            }
        if self.nn_model is not None:
            recipe_model["neural_network"] = {
                "name": getattr(self.nn_model, "name", None),
                "modules": len(getattr(self.nn_model, "modules", []) or []),
            }

        # Session history (pi's branch-summary mechanic): each run appends
        # a compact entry — request, mode, files it touched — on top of
        # whatever the seed recipe already carried. The next modify run
        # opens with this history in its inventory, so the agent knows
        # what was asked and changed before, instead of starting amnesiac.
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        history = self._load_recipe_history(recipe_path)
        touched = sorted({
            (tc.get("input") or {}).get("path")
            for tc in self.tool_calls_log
            if tc.get("tool") in ("write_file", "modify_file", "replace_file_lines")
            and isinstance((tc.get("input") or {}).get("path"), str)
        })
        history.append({
            "instructions": (instructions or "")[:300],
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "modify" if self._modify_mode else "create",
            "files_touched": touched[:20],
        })
        history = history[-10:]

        from besser.generators.llm.checkpoint import api_scenario_snapshot
        recipe = {
            "instructions": instructions,
            "history": history,
            "model": recipe_model,
            "llm_model": self.client.model,
            "generator_used": self._generator_used,
            "turns": self.total_turns,
            "tool_calls_count": len(self.tool_calls_log),
            "compactions": self._compaction_count,
            "elapsed_seconds": round(elapsed, 1),
            "max_cost_usd": self.max_cost_usd,
            "max_runtime_seconds": self.max_runtime_seconds,
            # Legacy field name: true when the per-call response-token
            # ceiling was raised (from-scratch or modify/fix run).
            # Cost/runtime caps remain caller-authorised.
            "adaptive_budget_applied": self._adaptive_budget_applied,
            # Items the model declined as not requested (task_list drop) with
            # its stated reasons: the reviewer sees what was NOT built and why.
            "dropped_tasks": [
                {"id": t["id"], "text": t["text"], "reason": t["dropped"]}
                for t in getattr(self.executor, "_tasks", []) if t.get("dropped")
            ],
            "tasks": self.executor.task_snapshot(),
            # The user's requirements with the last Phase 3 verdict on each:
            # what was NOT built is read here, not inferred from the code.
            "requirements": self._requirement_verdicts,
            "api_scenarios": api_scenario_snapshot(self._api_scenarios.values(), include_reports=True),
            "model_conversion_issues": getattr(self.domain_model, "conversion_issues", []) or [],
            "model_assembly_issues": self._assembly_issues,
            "usage": self.client.usage.summary(),
            "validation_issues": [
                {"severity": i.severity, "message": i.message}
                for i in self._validation_issues
            ],
            # Per-entity route/page/create facts (None when no domain
            # model or Phase 3 didn't run). The UI/recipe reader can
            # render this as the model-derived definition of done.
            "acceptance_matrix": self._acceptance_matrix,
            # The Phase 3 edits in tool_calls are on disk only when this is
            # False; a rollback discarded them and kept the Phase 2 output.
            "phase3_rolled_back": self._phase3_rolled_back,
            "output_files": sorted(output_files, key=lambda f: f["path"]),
            "output_summary": {
                "total_files": len(output_files),
                "from_generator": sum(1 for f in output_files if f["source"] == "generator"),
                "from_llm": sum(1 for f in output_files if f["source"] == "llm"),
                # Generator files the LLM changed: the work a regeneration
                # would overwrite. from_llm alone counts only new files.
                "generator_files_edited_by_llm": sum(
                    1 for f in output_files
                    if f.get("llm_modified") and f["source"] == "generator"),
                "total_bytes": sum(f["size"] for f in output_files),
            },
            "tool_calls": self.tool_calls_log,
            # Pointer to the structured trace file alongside this recipe.
            # Clients that want per-turn detail (tool calls, cost ticks,
            # phase transitions) can read it instead of re-parsing the
            # recipe's flattened summaries.
            "trace_file": TRACE_FILENAME if self._trace.path else None,
        }
        # Fix/modify success-gate outcome. Present ONLY on a fix run so the
        # from-scratch recipe stays byte-identical; records the reported
        # target and whether it was confirmed fixed, so a later "what did
        # the fix run actually do" is answerable from the recipe alone.
        if self._is_fix_run and self._fix_target is not None:
            recipe["fix_run"] = {
                "detected": True,
                "target": self._fix_target.descriptor,
                "kind": self._fix_target.kind,
                "entities": list(self._fix_target.entities),
                "target_resolved": self._fix_target_resolved,
                "target_message": self._fix_target_message,
            }
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        try:
            with open(recipe_path, "w", encoding="utf-8") as f:
                json.dump(recipe, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save recipe: %s", e)


# ======================================================================
# Helpers
# ======================================================================

# Per-value budget for the trace, the checkpoint's tool_calls_log and the
# recipe. Untruncated write-tool inputs go to TOOL_INPUTS_FILENAME.
_LOG_VALUE_BUDGET = 500
_WRITE_TOOLS_ON_RECORD = frozenset({"modify_file", "replace_file_lines", "write_file", "delete_file"})
TOOL_INPUTS_FILENAME = ".besser_tool_inputs.jsonl"


def _sanitize_for_log(data: Any) -> Any:
    """Bound string values for the logs with a marker that cannot be read as
    code. The old ``v[:500] + "..."`` was twice diagnosed as a model-written
    elision (runs 3f9a34b8 and 57160293, 2026-09-18). The marker names the
    cut, its size and a fingerprint, so two different over-budget inputs
    never render identically."""
    if isinstance(data, dict):
        out: dict = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > _LOG_VALUE_BUDGET:
                digest = hashlib.sha256(v.encode("utf-8", "replace")).hexdigest()[:12]
                v = (
                    v[:_LOG_VALUE_BUDGET]
                    + f"\n<<truncated {len(v) - _LOG_VALUE_BUDGET} of {len(v)} chars, "
                    f"sha256 {digest}>>"
                )
            out[k] = v
        return out
    return data
