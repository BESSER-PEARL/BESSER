"""Tuning constants and small pure helpers for the orchestrator.

Extracted from ``orchestrator.py``, which had grown to 6,771 lines with three
hundred of them being module-level configuration before the class even began.
Nothing here holds run state; every value is a constant or a pure function of
its arguments, which is what lets the orchestrator's mixins import it without
a cycle back to the module they are mixed into.

The comments explaining WHY each threshold has the value it does are the
point of this file -- several are derived from mining recorded runs, and a
number without its derivation is an invitation to "tune" it blindly.
"""

from __future__ import annotations

import os
import re as _re

from besser.spec_driven_agent.execution.process import COMMAND_OUTPUT_DIR
from besser.spec_driven_agent.state.checkpoint import (
    CHECKPOINT_FILENAME,
    _SNAPSHOT_DIR,
)
from besser.spec_driven_agent.state.tracing import TRACE_FILENAME


# Where a rollback parks the current tree while it restores. Living inside
# output_dir keeps the move on one filesystem, so it is a rename, not a copy.
_ROLLBACK_DISCARD_DIR = ".besser_rollback_discard"

# Where a new snapshot is assembled before it replaces the current one. The
# swap is a rename, so a failed copy never leaves the run without a rollback
# target. Phase 3 re-snapshots every time it reaches a better tree.
_SNAPSHOT_STAGING_DIR = ".besser_snapshot_staging"

# Installed dependency caches, never LLM-authored, excluded from the snapshot.
# Phase 3 now re-snapshots on every improvement, and copying an installed
# node_modules several times per run is gigabytes of pointless IO on a host
# whose C: drive has hit zero free bytes mid-session before.
_SNAPSHOT_IGNORED_DIRS = ("node_modules", "__pycache__", ".venv", "venv")

# Run bookkeeping that a rollback must NOT revert. The snapshot predates them,
# so restoring it would rewind the append-only trace and resurrect a stale
# checkpoint, making a later resume replay work already on disk.
_ROLLBACK_PRESERVED = {
    TRACE_FILENAME,
    CHECKPOINT_FILENAME,
    ".besser_recipe.json",
    _SNAPSHOT_DIR,
    _SNAPSHOT_STAGING_DIR,
    _ROLLBACK_DISCARD_DIR,
}

# Dependency / build directories excluded from the recipe's output_files
# manifest (mirrors the web runner's zip exclusions).
_RECIPE_EXCLUDED_DIRS = {
    "target", "node_modules", "__pycache__", ".git", "dist", "build",
    ".next", ".gradle", "venv", ".venv", _SNAPSHOT_DIR,
    _SNAPSHOT_STAGING_DIR, _ROLLBACK_DISCARD_DIR, COMMAND_OUTPUT_DIR,
}

# Written by a build or by running the app, never authored. Invisible to
# ``_workspace_revision`` (not even by presence): the frontend build and the
# boot probe both compare the revision across their own run, and their own
# leftovers must not read back as a source edit.
_REVISION_IGNORED_SUFFIXES = (
    ".tsbuildinfo", ".log", ".db", ".sqlite", ".sqlite3",
)


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


def _runtime_verdict(messages) -> int:
    """Worst runtime state the findings in ``messages`` establish.

    ``_RUNTIME_FAILED`` only on an OBSERVED failure, ``_RUNTIME_UNVERIFIED``
    when the probe ran but could not settle a create/action, ``_RUNTIME_OK``
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

# Ceiling on the one-off Phase 1 dependency install. A cold npm install of a
# generated Vite frontend measured 11s; this only stops a wedged registry call
# from delaying the whole run before the model has done anything.
_SCAFFOLD_INSTALL_TIMEOUT_SECONDS = 180

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

# Consecutive no-progress rounds tolerated before the repair loop ends.
#
# Was 1, which is right for a round that is genuinely a replay of the last one
# and wrong for every other kind. The loop now decides that per round (see
# ``replay`` below): a round that never reached for the editor still ends the
# loop on the spot, and only a round that DID something it can carry into the
# next prompt gets this allowance.
#
# Measured over the 221 spec-iteration runs recorded before the zero-write
# stop existed (verification/spec-iterations, 2026-09-19..20), on the round
# that FOLLOWED a round which wrote nothing and left the tree byte-identical:
#
#   the round attempted edits, all rejected  n= 30  next wrote 57%, cut 20%
#   the round made tool calls, none an edit  n=209  next wrote 18%, cut 11%
#   the round called no tool at all (prose)  n=  4  next wrote  0%, cut  0%
#
# and after TWO consecutive such rounds, whatever their kind, the next wrote
# 6% and cut 4% (n=116) - so the streak still ends at two.
_PHASE3_NO_PROGRESS_ROUNDS = 2

# Rounds that edit the tree without improving its score before the loop ends.
# The unchanged-state guard cannot see these: its key includes a content hash
# of every source file, so ANY write - including a different useless one each
# round - reads as a new state and the guard never fires.
#
# Was 2, which stopped the loop one round before the payoff. Same corpus,
# counting a round as non-improving when the blocker count did not fall: after
# two consecutive non-improving rounds the NEXT round still improved 36% of
# the time (n=125) against a 46% baseline with no plateau behind it; only at
# three does the payoff halve (23%, n=60) and at four collapse (15%, n=107).
_PHASE3_PLATEAU_ROUNDS = 3

# Runtime-gate verdicts, ordered worst-last so a tuple compares as a score.
_RUNTIME_OK, _RUNTIME_UNVERIFIED, _RUNTIME_FAILED = 0, 1, 2

# The probe ran the application and WATCHED it fail: the ORM would not map,
# the app would not start, a create route crashed, an action handler crashed.
#
# ``api scenario:`` is deliberately NOT here. A retained workflow is a
# model-authored assertion, and the tool that owns it says so ("a generated
# assertion can be wrong; explain any correction against that specification").
# Run dynioweu delivers an app that passes 11/11 corrected acceptance checks
# and still fails its own booking_overlap_violation scenario, so a gate keyed
# on it refuses a working application.
_RUNTIME_OBSERVED_FAILURE_PREFIXES = (
    "mapper config:", "application startup:", "create contract:", "action call:",
)
# Source the app cannot survive at runtime even where no probe reached it: it
# does not parse, imports a module that is not there, or uses a name nothing
# defines. Deterministic, not a guess about untried input.
_RUNTIME_FATAL_SOURCE_PREFIXES = (
    "syntax error in", "python contract:", "missing module:", "undefined name:",
)
_RUNTIME_FAILURE_PREFIXES = (
    _RUNTIME_OBSERVED_FAILURE_PREFIXES + _RUNTIME_FATAL_SOURCE_PREFIXES
)
# The app ran but the probe could not establish the result: a guessed fixture a
# business rule legitimately refused, an action whose reachable state it could
# not construct, or a probe that did not run at all. Not evidence of a defect,
# and not evidence of a working app either. Only a spec-derived ``test_api``
# scenario (see ``confirmed_create_paths``) turns one of these green.
_RUNTIME_UNVERIFIED_PREFIXES = (
    "runtime unverified:", "create unverified:", "action unverified:",
    "api scenario:",
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


# Per-value budget for the trace, the checkpoint's tool_calls_log and the
# recipe. Untruncated write-tool inputs go to TOOL_INPUTS_FILENAME.
_LOG_VALUE_BUDGET = 500
_WRITE_TOOLS_ON_RECORD = frozenset({"modify_file", "replace_file_lines", "write_file", "delete_file"})
TOOL_INPUTS_FILENAME = ".besser_tool_inputs.jsonl"
