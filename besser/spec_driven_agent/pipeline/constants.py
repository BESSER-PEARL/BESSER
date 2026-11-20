"""Tuning constants and small pure helpers for the orchestrator.

Nothing here holds run state; every value is a constant or a pure function of
its arguments, which is what lets the orchestrator's mixins import it without
a cycle back to the module they are mixed into.

The comments explaining WHY each threshold has the value it does are the
point of this file -- several are derived from mining recorded runs, and a
number without its derivation is an invitation to "tune" it blindly.
"""

from __future__ import annotations

import fnmatch

from besser.spec_driven_agent.execution.process import COMMAND_OUTPUT_DIR
from besser.spec_driven_agent.state.checkpoint import (
    CHECKPOINT_FILENAME,
    _SNAPSHOT_DIR,
)
from besser.spec_driven_agent.state.tracing import TRACE_FILENAME
from besser.spec_driven_agent.validation.issues import RUNTIME_CHECK_UNVERIFIED


# Where a rollback parks the current tree while it restores. Living inside
# output_dir keeps the move on one filesystem, so it is a rename, not a copy.
_ROLLBACK_DISCARD_DIR = ".besser_rollback_discard"

# Where a new snapshot is assembled before it replaces the current one. The
# swap is a rename, so a failed copy never leaves the run without a rollback
# target. Phase 3 re-snapshots every time it reaches a better tree.
_SNAPSHOT_STAGING_DIR = ".besser_snapshot_staging"

# Installed dependency caches, never LLM-authored, excluded from the snapshot.
# Phase 3 re-snapshots on every improvement, and copying an installed
# node_modules several times per run is gigabytes of pointless IO and disk.
_SNAPSHOT_IGNORED_DIRS = ("node_modules", "__pycache__", ".venv", "venv")

# Directories a workspace walk skips: dependency caches, VCS and build output.
WORKSPACE_SKIP_DIRS = frozenset(
    {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build"}
)

# Append-only sidecar of untruncated write-tool inputs (see _LOG_VALUE_BUDGET).
TOOL_INPUTS_FILENAME = ".besser_tool_inputs.jsonl"

# Run bookkeeping that a rollback must NOT revert. The snapshot predates them,
# so restoring it would rewind the append-only trace and tool-input sidecar and
# resurrect a stale checkpoint, making a later resume replay work already on disk.
_ROLLBACK_PRESERVED = {
    TRACE_FILENAME,
    CHECKPOINT_FILENAME,
    TOOL_INPUTS_FILENAME,
    ".besser_recipe.json",
    _SNAPSHOT_DIR,
    _SNAPSHOT_STAGING_DIR,
    _ROLLBACK_DISCARD_DIR,
}

# A run's temporary workspace folder starts with this; it is removed after the run.
RUN_DIR_PREFIX = "besser_llm_"

# Dependency, build and run-internal directories that are not part of the app.
_RECIPE_EXCLUDED_DIRS = {
    "target", "node_modules", "__pycache__", ".git", "dist", "build",
    ".next", ".gradle", "venv", ".venv", ".pytest_cache", _SNAPSHOT_DIR,
    _SNAPSHOT_STAGING_DIR, _ROLLBACK_DISCARD_DIR, COMMAND_OUTPUT_DIR,
}

# Files a build or a run of the app leaves behind (fnmatch globs).
_EXCLUDED_OUTPUT_FILE_GLOBS = ("*.zip", "*.db", "*.sqlite", "*.sqlite3", "*.db-journal", "*.pyc", "*.log")

# Base names left out of everything a run delivers: the recipe's output_files
# manifest, the download, the seed of a later run and a GitHub push. One list,
# so the manifest cannot name a file the user does not receive.
EXCLUDED_OUTPUT_NAMES = frozenset({*_RECIPE_EXCLUDED_DIRS, *_EXCLUDED_OUTPUT_FILE_GLOBS})

# What installing and building put in a workspace, which nothing delivers or seeds:
# the excluded folders without the run's own records.
INSTALLED_AND_BUILT_DIRS = frozenset(_RECIPE_EXCLUDED_DIRS - {
    _SNAPSHOT_DIR, _SNAPSHOT_STAGING_DIR, _ROLLBACK_DISCARD_DIR, COMMAND_OUTPUT_DIR})


def is_excluded_output_file(name: str) -> bool:
    """True for a file name the delivered output leaves out."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_OUTPUT_NAMES)

# Written by a build or by running the app, never authored. Invisible to
# ``_workspace_revision`` (not even by presence): the frontend build and the
# boot probe both compare the revision across their own run, and their own
# leftovers must not read back as a source edit.
_REVISION_IGNORED_SUFFIXES = (
    ".tsbuildinfo", ".log", ".db", ".sqlite", ".sqlite3", ".pyc", ".coverage",
    # SQLite's sidecar files next to a database (app.db-journal, -wal, -shm).
    "-journal", "-wal", "-shm",
)
# Caches a test, lint or build run writes wherever it runs. Nobody authors
# source under these dot-names, so they count at any depth (``.dart_tool`` is
# Flutter's).
_REVISION_IGNORED_DIRS = frozenset({
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".dart_tool",
})
# Report and build output under an ordinary folder name, ignored only in the
# project the tool ran in: next to one of these manifests (a leading dot means
# any file with that suffix) or at the workspace root. Anywhere else, e.g.
# ``src/features/coverage/``, the folder is authored source.
_REVISION_PROJECT_OUTPUT_DIRS = {
    "coverage": ("package.json",),  # jest / vitest / c8
    "htmlcov": ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
                "pytest.ini", "tox.ini"),  # coverage.py / pytest-cov
    "obj": (".csproj", ".fsproj", ".vbproj"),  # MSBuild intermediate output
}
# Data and media a running app writes (a CSV store, an uploaded photo): their
# path counts, their content does not, so the app's own writes are no edit.
_REVISION_PATH_ONLY_SUFFIXES = (
    ".csv", ".tsv", ".jsonl", ".parquet", ".pkl", ".npy",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".mp3", ".mp4", ".wav",
)
# Bytes of each file the revision reads; the size covers the rest.
_REVISION_HASH_CAP = 1024 * 1024


# Tool-call detail shown in progress events.
_TOOL_DETAIL_MAX_CHARS = 160
# Argument keys worth putting in the stream, per tool. An allow list on purpose:
# file CONTENT must never reach the event stream, but path/target/action are what
# make a run readable afterwards.
_TOOL_DETAIL_KEYS = (
    "path", "file_path", "filename", "target", "action", "id", "ids",
    "done", "add", "drop", "blocked", "text", "command", "pattern", "query", "class_name", "generator",
)


def _runtime_verdict(messages) -> int:
    """Worst runtime state the findings in ``messages`` establish.

    ``_RUNTIME_FAILED`` only on an OBSERVED failure, ``_RUNTIME_UNVERIFIED``
    when the probe ran but could not settle a create/action or a runtime check
    did not run, ``_RUNTIME_OK``
    otherwise. Callers must not read OK as "the app works" unless a backend
    was actually booted - see ``_probeable_backends``.
    """
    verdict = _RUNTIME_OK
    for entry in messages:
        lower = str(getattr(entry, "message", entry)).strip().lower()
        if lower.startswith(_RUNTIME_FAILURE_PREFIXES):
            return _RUNTIME_FAILED
        if lower.startswith(_RUNTIME_UNVERIFIED_PREFIXES):
            verdict = _RUNTIME_UNVERIFIED
    return verdict


def _tool_use_blocks(content) -> list:
    """Every tool call in an assistant reply, a nameless one included.

    Each call needs its result in the next message, or both providers reject
    the request that follows; a call with no name gets the executor's
    "unknown tool" error like any other name it does not know.
    """
    return [block for block in content or [] if getattr(block, "type", None) == "tool_use"]


def _tool_call_detail(tool_name: str, tool_input: object, blocks_in_turn: int) -> str:
    """One short line saying what this tool call was about.

    Streamed alongside the tool name so a finished run can be read back from the
    durable event store. ``blocks_in_turn`` is included because 1 means the model
    batched nothing, and every turn costs a full prompt prefill.
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







# Tools that are read-only and shouldn't count for loop detection
_READONLY_TOOLS = frozenset({
    "read_file", "list_files", "search_in_files", "validate_app", "test_api",
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

# Maximum workers for parallel tool execution
_MAX_PARALLEL_WORKERS = 4

# Phase 3 toolchain-fix outer cap. The LLM gets up to this many
# (collect → fix-loop → re-collect) iterations before we accept the
# remaining toolchain errors and move on. Each iteration is the existing
# 5-turn LLM fix loop, and the actual spend is bounded by ``max_cost_usd``
# (the inner loop stops when the cost cap is hit), so this cap just keeps
# a truly stuck run from looping forever. It is deliberately NOT tiny: as
# long as each kept attempt retires blockers, the loop keeps going; the
# keep-or-revert rule in phase3_repair ends a stuck one.
_MAX_TOOLCHAIN_FIX_ITERATIONS = 5

# Ceiling on the one-off Phase 1 dependency install. A cold npm install of a
# generated Vite frontend measured 11s; this only stops a wedged registry call
# from delaying the whole run before the model has done anything.
_SCAFFOLD_INSTALL_TIMEOUT_SECONDS = 180

# Work is counted in executed tool calls: each turn of ``max_turns`` allows this
# many, so a model that makes one call per turn gets the work one that batches
# four got. Turns stay a runaway guard at the same multiple.
_CALLS_PER_TURN = 4
# Executed tool calls per fix attempt (ten turns of four). No tool is forced: an
# attempt ends when the model ends its turn.
_PHASE3_FIX_CALLS = 10 * _CALLS_PER_TURN
# Sent when a fix attempt stops with listed creates not yet read back.
_PHASE3_CREATE_EVIDENCE_REMINDER = (
    "<system-reminder>These listed creates still have no read-back: {paths}. "
    "Each is settled only by a test_api scenario that POSTs it and then, in that "
    "same scenario, GETs the record by its id under the same path. A scenario "
    "holds at most 20 requests, so split the endpoints across several scenarios, "
    "each logging in first where the API requires it and creating the parents it "
    "needs. No source edit is needed for these; your next call must be test_api."
    "</system-reminder>"
)

# Share of each budget (seconds, cost and, with auto-fix on, tool calls) Phase 2
# leaves to Phase 3's checks, judge and repair. Basis, the runs recorded on
# 2026-11-17/18: where Phase 3 repaired, it made a median 25% of the run's tool
# calls (10 runs); without an install hang it took 15-992 s (median 128 s, one
# judge call 56-154 s), and a quarter of the 2400 s cap covers 4 of those 5.
_PHASE3_BUDGET_SHARE = 0.25

# Runtime-gate verdicts, ordered worst-last so a tuple compares as a score.
_RUNTIME_OK, _RUNTIME_UNVERIFIED, _RUNTIME_FAILED = 0, 1, 2

# The probe ran the application and WATCHED it fail: the ORM would not map,
# the app would not start, a create route crashed, an action handler crashed.
#
# ``api scenario:`` is deliberately NOT here. A retained workflow is a
# model-authored assertion, and the tool that owns it says so ("a generated
# assertion can be wrong; explain any correction against that specification").
# An app can pass every acceptance check and still fail its own generated
# scenario, so a gate keyed on it would refuse a working application.
_RUNTIME_OBSERVED_FAILURE_PREFIXES = (
    "mapper config:", "application startup:", "create contract:", "action call:",
)
# Source the app cannot survive at runtime even where no probe reached it: it
# does not parse, imports a module that is not there, or uses a name nothing
# defines. Deterministic, not a guess about untried input.
_RUNTIME_FATAL_SOURCE_PREFIXES = (
    "syntax error in", "python contract:", "missing module:",
)
_RUNTIME_FAILURE_PREFIXES = (
    _RUNTIME_OBSERVED_FAILURE_PREFIXES + _RUNTIME_FATAL_SOURCE_PREFIXES
)
# The app ran but the probe could not establish the result: a guessed fixture a
# business rule legitimately refused, an action whose reachable state it could
# not construct, or a probe that did not run at all. Not evidence of a defect,
# and not evidence of a working app either. Only a spec-derived ``test_api``
# scenario (see ``confirmed_create_paths``) turns one of these green. A runtime
# check that did not run is a warning, not a blocker, but still unverified.
_RUNTIME_UNVERIFIED_PREFIXES = (
    "runtime unverified:", "create unverified:", "action unverified:",
    "api scenario:", RUNTIME_CHECK_UNVERIFIED,
)

# Generators that produce a WHOLE application, and the workspace folder each
# writes. Once one of them has built the scaffold, none of them may run again:
# re-running the primary regenerates over every edit Phase 2 has made, and a
# rival stack drops a second application beside the assembled one.
# Single-artefact generators (rdf, supabase, java/python classes) stay: a user
# can legitimately ask for one alongside the app.
_APPLICATION_OUTPUT_DIRS = {
    "generate_web_app": "web_app",
    "generate_django": "django",
    "generate_fastapi_backend": "backend",
    "generate_rest_api": "rest_api",
}
_APPLICATION_PRIMARY_TOOLS = frozenset(_APPLICATION_OUTPUT_DIRS)
# The generators a modify run regenerates and merges after a model change (#659):
# on 44 FastAPI runs, regenerating kept every app working. The web app's React
# output has no determinism test yet, so it is not regenerated.
REGENERABLE_GENERATORS = frozenset({"generate_fastapi_backend"})

# Sub-generator tools that a chosen PRIMARY generator already bundles (e.g. a
# FastAPI backend already contains SQLAlchemy models, Pydantic schemas and REST
# routers). Removed from the Phase-2 toolset so the agent CANNOT emit duplicate
# top-level pydantic/ sqlalchemy/ rest_api/ folders next to the scaffold.
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


# Per-value budget for the trace, the checkpoint's tool_calls_log and the
# recipe. Untruncated write-tool inputs go to TOOL_INPUTS_FILENAME.
_LOG_VALUE_BUDGET = 500
_WRITE_TOOLS_ON_RECORD = frozenset({"modify_file", "replace_file_lines", "write_file", "delete_file"})
