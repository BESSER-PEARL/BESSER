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
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import count
from typing import Any, Literal
from uuid import uuid4

from besser.BUML.metamodel.structural import DomainModel
# The canonical set, shared with get_tools_for() so the advertised list and the
# dispatch gate can never disagree about which tools are shell tools.
from besser.spec_driven_agent.agent.tools import _SHELL_TOOLS as _SHELL_TOOL_NAMES
from besser.spec_driven_agent.execution.process import (
    COMMAND_OUTPUT_DIR,
    _safe_subprocess_env,
    _SAFE_ENV_ALLOWLIST as _SAFE_ENV_ALLOWLIST,
    _SECRET_SUBSTRINGS as _SECRET_SUBSTRINGS,
)
from besser.spec_driven_agent.execution.sandbox import (
    SandboxUnavailable,
    sandboxed_command,
)
from besser.spec_driven_agent.agent.edit_apply import (
    AmbiguousEdit,
    describe_escape_mismatch,
    elided_lines,
    find_elision,
    find_similar_lines,
    locate_anchored_span,
    locate_chunk,
    replace_most_similar_chunk,
    replacement_spans,
    _strip_line_numbers,
)
from besser.spec_driven_agent.validation import frontend_source

logger = logging.getLogger(__name__)


# OpenCode edit.ts serializes the complete read/validate/write transaction by
# resolved path. Share locks across executors too; weak references prevent the
# registry retaining every generated path for the worker's entire lifetime.
_FILE_LOCKS = weakref.WeakValueDictionary()
_FILE_LOCKS_GUARD = threading.Lock()


def _file_lock(path: str):
    key = os.path.normcase(os.path.realpath(path))
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _FILE_LOCKS[key] = lock
        return lock


ToolExecutionStatus = Literal["ok", "error", "skipped"]


# A task whose verifier keeps refusing is retried at most this many times;
# unbounded retries can livelock a run until the turn cap.
_MAX_TASK_VERIFY_ATTEMPTS = 3


@dataclass(frozen=True)
class ToolExecutionResult:
    """Typed result used by the harness around the model-facing JSON payload.

    Tool payloads intentionally retain their historical shapes (``status`` may
    be ``written``, ``modified``, and so on), but orchestration must never infer
    success by searching the first bytes of a JSON string. This envelope gives
    loop guards, tracing, and future telemetry one canonical outcome while
    :meth:`ToolExecutor.execute` remains backward compatible for callers that
    expect a JSON string.
    """

    status: ToolExecutionStatus
    payload: dict[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    def to_json(self) -> str:
        return json.dumps(self.payload, default=str)

# Maximum time a shell command can run (seconds)
COMMAND_TIMEOUT = 120

# 1:1 typographic → ASCII map used by modify_file's fallback matching.
# Models routinely type an ASCII apostrophe/quote/dash where the file has
# the typographic variant (’ etc.); the exact match then fails on
# every retry. Every mapping is same-length so match
# indices in the normalized text are valid in the original.
_TYPOGRAPHIC_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ",
})

# Obvious destructive / exfiltration patterns we refuse to run. This is
# NOT a full sandbox — a committed attacker with prompt injection on the
# instructions can still cause damage within the workspace. It's a cheap
# filter that stops the easy mistakes: fork bombs, root-fs rm, curl-pipe-
# sh one-liners, writes to system paths, and credential theft from the
# user's home dir. Strong isolation requires a container/VM.
_COMMAND_DENY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(?:^|[\s;&|])sudo(?:\s|$)"),
    re.compile(r"(?:^|[\s;&|])su\s+-"),
    re.compile(r"rm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+(?:/|~|\$HOME|%USERPROFILE%)(?:\s|$)"),
    re.compile(r"\|\s*(?:sh|bash|zsh|ksh|fish|cmd|powershell|pwsh)\b"),
    re.compile(r"(?:curl|wget|iwr|Invoke-WebRequest)\b[^\n|]*\|\s*(?:sh|bash|zsh)\b"),
    re.compile(r":\(\)\s*\{.*:\|:&.*\};\s*:"),  # classic fork bomb
    re.compile(r"(?:^|\s)chmod\s+-R\s+[0-7]*777\s+/"),
    re.compile(r">\s*/etc/"),
    re.compile(r">\s*/dev/(?:sd|nvme|disk)"),
    re.compile(r"(?:~/|\$HOME/|%USERPROFILE%\\)\.ssh", re.IGNORECASE),
    re.compile(r"(?:~/|\$HOME/|%USERPROFILE%\\)\.aws", re.IGNORECASE),
)


def _check_command_safety(command: str) -> str | None:
    """Return a human-readable reason if ``command`` matches a deny pattern."""
    for pattern in _COMMAND_DENY_PATTERNS:
        match = pattern.search(command)
        if match:
            return (
                "Refused: command matches a security denylist "
                f"({match.group(0)!r}). Break the task into smaller, "
                "explicit commands."
            )
    return None

# Maximum output size returned to the LLM (chars).
# Lower = less context bloat = more turns before compaction needed.
MAX_OUTPUT_SIZE = 15_000

# Per-stream ceiling on a spilled command log (chars). The spill exists to
# keep what truncation drops, so it is deliberately far above MAX_OUTPUT_SIZE;
# the bound is only there so a runaway command cannot fill the disk.
MAX_SPILL_SIZE = 2_000_000

# Directories search_in_files never descends into: installed dependencies and
# build output, none of which the model wrote or can usefully edit.
_UNSEARCHED_DIRS = frozenset({
    "node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build",
})

# Maximum file content returned by read_file (chars)
# A generated router runs to ~35k chars; 60k covers the largest observed
# generated file with headroom and costs ~15k tokens against an 80k
# compaction threshold.
MAX_FILE_READ = 60_000


def _normalize_path_for_comparison(path: str) -> str:
    """Strip the Windows ``\\\\?\\`` extended-path prefix.

    On Windows, ``os.path.realpath`` sometimes returns paths with the
    ``\\\\?\\`` (or ``\\\\?\\UNC\\``) extended-length prefix, especially
    when long-path support is enabled at the OS level. This prefix can
    appear on child paths but NOT on the workspace root (or vice versa)
    depending on whether the path exists at resolution time. When that
    happens, a naive ``full.startswith(workspace)`` check fails for
    legitimate paths inside the workspace and the caller sees a bogus
    "Path traversal blocked" error.

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


# Stderr fragments that indicate the requested binary isn't installed in
# the execution container. The runner image has Python +
# Node + a few essentials; everything else (ruby, go, rustc, dotnet, ...)
# routinely falls into this bucket. We treat these as soft skips so the
# LLM doesn't surface "X is not installed" to the user as if it were a
# real failure — the user can't act on it and it's just noise.
_RUNTIME_NOT_INSTALLED_PATTERNS: tuple[str, ...] = (
    "command not found",
    "not recognized as an internal or external command",  # Windows shell
)
# A bare "no such file or directory" is NOT one of these: it is also what a
# missing data file, requirements.txt or package.json produces. Only the
# exec-failure form names an absent binary.
_EXEC_ENOENT_RE = re.compile(
    r"^\s*exec(?:ve)?:.*no such file or directory", re.IGNORECASE | re.MULTILINE,
)


def _looks_like_command_not_found(stderr: str) -> bool:
    """True if the subprocess stderr matches a "binary missing" message."""
    if not isinstance(stderr, str) or not stderr:
        return False
    lowered = stderr.lower()
    if any(pat in lowered for pat in _RUNTIME_NOT_INSTALLED_PATTERNS):
        return True
    return bool(_EXEC_ENOENT_RE.search(stderr))




_DEF_HEADER_RE = re.compile(r"^\s*(?:async\s+)?(def|class)\s+(\w+)", re.MULTILINE)

# Recovery for an anchor the file does not hold. Never "use write_file": that
# wording turns targeted edits into whole-file rewrites of scaffold code
# (see Orchestrator._build_modify_loop_reminder).
_ANCHOR_ADVICE = (
    "To add code that does not exist yet, anchor on a line that IS in the file: "
    "old_text = that line copied from read_file output, new_text = that line "
    "followed by the new code. To change existing code, copy the real lines from "
    "read_file output."
)


_IMPORT_BREAK_ADVICE = (
    "Do not resend it unchanged. Either define the missing name in this same "
    "file before the code that uses it, or call read_file on the file to see "
    "the name it actually exports and use that spelling."
)


def _missing_definition(old_text: str, content: str) -> str:
    """``"There is no `def X` in this file. "`` when old_text quotes a
    definition header the file never had, else ``""``."""
    match = _DEF_HEADER_RE.search(old_text)
    if not match:
        return ""
    keyword, name = match.groups()
    if re.search(rf"^\s*(?:async\s+)?{keyword}\s+{re.escape(name)}\b", content, re.MULTILINE):
        return ""
    return f"There is no `{keyword} {name}` in this file. "


def _occurrence_locations(content: str, old_text: str, limit: int = 10) -> list[str]:
    """``"line N (in <def>)"`` for each occurrence of ``old_text``."""
    lines = content.split("\n")
    starts = [i for i, ln in enumerate(lines) if re.match(r"\s*(?:async\s+def|def|class)\s+\w+", ln)]
    out: list[str] = []
    pos = content.find(old_text)
    while pos != -1 and len(out) < limit:
        line_no = content.count("\n", 0, pos)
        owner = next((lines[i].strip() for i in reversed(starts) if i <= line_no), None)
        out.append(f"line {line_no + 1}" + (f" (in {owner})" if owner else ""))
        pos = content.find(old_text, pos + max(1, len(old_text)))
    return out


def _anchored_occurrences(content: str, old_text: str) -> list[int]:
    """Start offsets of ``old_text`` that do not begin inside a line's
    indentation. A quote whose first line is under-indented still matches as
    a substring a few characters into the file's line; replacing that span
    keeps the file's leading spaces and writes the rest at the quote's indent
    (e.g. a second ``try:`` at the enclosing level). A hit at column 0, or
    after real text (a partial-line edit), is fine."""
    out: list[int] = []
    pos = content.find(old_text)
    while pos != -1:
        line_start = content.rfind("\n", 0, pos) + 1
        if pos == line_start or content[line_start:pos].strip():
            out.append(pos)
        pos = content.find(old_text, pos + max(1, len(old_text)))
    return out


def _new_syntax_error(rel_path: str, before: str, after: str) -> tuple[str, int] | None:
    """``(message, line)`` when ``after`` no longer parses although ``before``
    did; ``None`` otherwise (unsupported language, or already broken).

    Python compiles; TS/TSX/JS/JSX go through the structural scanner in
    ``validation.frontend_source``, so an edit cannot corrupt a frontend file
    that was valid as generated.
    """
    if frontend_source.supports(rel_path):
        return frontend_source.new_syntax_error(before, after)
    if not rel_path.lower().endswith(".py"):
        return None
    try:
        compile(before, rel_path, "exec")
    except Exception:
        return None
    try:
        compile(after, rel_path, "exec")
    except SyntaxError as exc:
        return (exc.msg or "syntax error", exc.lineno or 0)
    except Exception:
        return None
    return None


# The ORM module every router star-imports. Breaking it does not break one
# file, it takes the whole application down, and the Phase 3 repair loop
# edits it often; a rollback catches that, but wastes the whole attempt.
_STRUCTURAL_MODULES = ("sql_alchemy.py", "pydantic_classes.py")
_IMPORT_SMOKE_TIMEOUT = 20


def _breaks_module_import(path: str, before: str, after: str) -> str | None:
    """``reason`` when ``after`` makes a structural module unimportable.

    The same contract as ``_new_syntax_error``, one level up: an edit that
    leaves the file parseable but the module unimportable (a name used before
    definition, a relationship naming a property that no longer exists) is
    refused rather than written. Only judged when the module imported BEFORE
    the edit, so a file that was already broken can still be repaired.
    """
    import subprocess

    name = os.path.basename(path)
    if name not in _STRUCTURAL_MODULES:
        return None
    folder = os.path.dirname(path) or "."
    probe = (f"import {name[:-3]}\n"
             "try:\n"
             "    from sqlalchemy.orm import configure_mappers; configure_mappers()\n"
             "except ImportError:\n"
             "    pass\n")

    def imports(source: str) -> tuple[bool, str]:
        original = None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                original = handle.read()
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(source)
            result = subprocess.run(
                [sys.executable, "-c", probe], cwd=folder, capture_output=True,
                text=True, timeout=_IMPORT_SMOKE_TIMEOUT, env=_safe_subprocess_env(),
            )
            return result.returncode == 0, (result.stderr or "").strip().splitlines()[-1:] and \
                (result.stderr or "").strip().splitlines()[-1] or ""
        except Exception:
            # The probe itself failed; never turn that into a refusal.
            return True, ""
        finally:
            if original is not None:
                with open(path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(original)

    was_ok, _ = imports(before)
    if not was_ok:
        return None
    # Judge what will actually land. A write finishes by repairing a
    # forgotten framework import, so an edit that only needs `Table` adding
    # to the sqlalchemy line must not be refused here - the two guards would
    # otherwise cancel out, the import one winning because it runs first.
    try:
        from besser.spec_driven_agent.repair.import_repair import repair_missing_imports

        repaired, _notes = repair_missing_imports(path, after)
    except Exception:
        repaired = after
    now_ok, reason = imports(repaired)
    return None if now_ok else (reason or "the module no longer imports")


def _changed_region(old: str, new: str, context: int = 2, cap: int = 60) -> str:
    """The lines of ``new`` that differ from ``old``, numbered, with
    ``context`` lines either side."""
    old_lines, new_lines = old.split("\n"), new.split("\n")
    first = 0
    while first < min(len(old_lines), len(new_lines)) and old_lines[first] == new_lines[first]:
        first += 1
    tail = 0
    while (tail < min(len(old_lines), len(new_lines)) - first
           and old_lines[-1 - tail] == new_lines[-1 - tail]):
        tail += 1
    start = max(0, first - context)
    end = min(len(new_lines), len(new_lines) - tail + context)
    rows = [f"{i + 1:>4}| {new_lines[i]}" for i in range(start, end)]
    if len(rows) > cap:
        rows = rows[:cap] + [f"   ...| ({len(rows) - cap} more changed lines)"]
    return "\n".join(rows)


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
        domain_model: DomainModel | None = None,
        gui_model: Any = None,
        agent_model: Any = None,
        agent_config: dict | None = None,
        quantum_circuit: Any = None,
        object_model: Any = None,
        bpmn_model: Any = None,
        nn_model: Any = None,
        protect_scaffold: bool = False,
        per_write_diagnostics: bool = True,
        allow_shell: bool = False,
    ):
        self.workspace = _normalize_path_for_comparison(os.path.realpath(workspace))
        self.allow_shell = allow_shell
        # Serial number for spilled command logs (see _spill_command_output).
        self._command_log_count = 0
        self.domain_model = domain_model
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
        self.quantum_circuit = quantum_circuit
        self.object_model = object_model
        self.bpmn_model = bpmn_model
        self.nn_model = nn_model
        # Track files created by generators — used to warn if LLM overwrites them
        self._generator_files: set[str] = set()
        # Per-path modify_file counts. After >= 2 targeted edits on a small
        # generator file the write_file guardrail relaxes, so a deliberate
        # rewrite is not bounced with "use modify_file".
        self._modify_counts: dict[str, int] = {}
        # Consecutive failed modify_file attempts per path. Escalates the
        # error message from "not found" to "stop retyping old_text, read the
        # file".
        self._failed_modifies: dict[str, int] = {}
        # The old_text of the last miss per path. A byte-identical resend
        # cannot succeed.
        self._last_missed_old_text: dict[str, str] = {}
        # Every rejected modify_file this run, (path, old_text, new_text) ->
        # times seen, and per path the highest repeat count; no-op calls
        # (old_text == new_text) count too. Only a successful edit on the path
        # clears these.
        self._rejected_edits: dict[tuple[str, str, str], int] = {}
        # The same TARGET refused again, whatever the draft: (path, anchor)
        # where anchor is old_text, or "lines N-M" for a range edit.
        self._rejected_targets: dict[tuple[str, str], int] = {}
        self._repeat_hits: dict[str, int] = {}
        # (path, times seen) when the LAST call repeated a rejected edit, for
        # the orchestrator's escalation; None otherwise.
        self.last_repeat: tuple[str, int] | None = None
        # Paths the orchestrator closed for the rest of the run -> reason.
        self._frozen_paths: dict[str, str] = {}
        # Files the model has actually SEEN this run: read, written, modified,
        # or inlined in the scaffold snapshot. A miss on any other path is a
        # quote from memory, and the reply says so first.
        self._known_paths: set[str] = set()
        # Finding replacement text in a file is not proof that an edit happened.
        # Receipts bind the exact request to the entire successful post-write
        # state. They are intentionally not inferred from resumed file contents.
        self._edit_receipts: dict[tuple[str, str, str, bool], str] = {}
        self._successful_writes: dict[str, str] = {}
        # A range edit names a server-issued read, not model-reconstructed old
        # code. Bind the visible lines to the resolved file and its full digest.
        # These capabilities deliberately expire on resume (read again).
        self._read_sequence = count(1)
        self._read_epoch = uuid4().hex[:12]
        # Reads per path, to catch a model crawling one file in tiny windows.
        self._read_counts: dict[str, int] = {}
        self._read_views: dict[str, tuple[str, str, int, int]] = {}
        self._edit_recovery: dict[str, int] = {}
        # Range-edit refusals per path; past _RANGE_EDIT_GIVE_UP the
        # ladder stops steering this file toward replace_file_lines.
        self._range_edit_failures: dict[str, int] = {}
        self.app_validator = None  # supplied by the orchestrator; no arbitrary command input
        self.api_tester = None
        # When True (weak / free-tier models only), the deterministic Phase-1
        # scaffold is IMMUTABLE to delete_file: the model may edit those files
        # in place but cannot tear them down and rebuild in another framework.
        # Observed with the free qwen tier deleting a whole FastAPI backend to
        # rewrite it in Flask; capable cloud models keep full delete_file.
        self._protect_scaffold = protect_scaffold
        self._per_write_diagnostics = per_write_diagnostics
        # Data contract from the domain model (id types, server-owned
        # fields). Every write_file/modify_file result carries the lint
        # findings for the new content, so the model sees a violation in
        # the SAME turn it wrote it — far cheaper than waiting for the
        # Phase 3 sweep to send it back with cold context.
        try:
            from besser.spec_driven_agent.validation.contract_checks import build_data_contract
            self._data_contract = build_data_contract(domain_model)
        except Exception:  # never let contract extraction break the executor
            self._data_contract = None
        # The run's work checklist (seeded from gap analysis via
        # ``set_tasks``). The LLM manages it through the ``task_list``
        # tool; the orchestrator's end_turn gate refuses to finish while
        # items are open.
        self._tasks: list[dict] = []
        # Edit-first guardrail for MODIFY runs (set via
        # ``enable_modify_guard``). In a modify run every existing file
        # is a user's working app, so whole-file rewrites are rejected
        # until the model has demonstrably tried targeted edits — the
        # rewrite habit is where modify-run regressions come from.
        self._modify_guard = False

    def enable_modify_guard(self) -> None:
        """Turn on the edit-first guardrail (modify runs only)."""
        self._modify_guard = True

    def set_scaffold_family(self, family: str | None, instructions: str = "") -> None:
        """Record which framework the Phase-1 scaffold committed to.

        Used by the per-write lint: content that imports a RIVAL framework
        (a Flask rewrite of a FastAPI scaffold) gets an immediate warning —
        the free tier's framework-switch habit, caught at write time
        instead of after the run burned its budget.
        """
        self._scaffold_family = family
        self._scaffold_instructions = instructions

    def _framework_switch_warning(self, rel_path: str, content: str) -> str | None:
        from besser.spec_driven_agent.planning.stack_metadata import effective_rivals

        family = getattr(self, "_scaffold_family", None)
        # Shares one rule with Phase 3 so a user who asked for Flask is not
        # warned at write time and then blocked at validation for the same file.
        rivals = effective_rivals(family, getattr(self, "_scaffold_instructions", ""))
        if not rivals or not rel_path.endswith(".py"):
            return None
        for rival in rivals:
            if re.search(rf"^\s*(?:from|import)\s+{rival}\b", content, re.MULTILINE):
                return (
                    f"FRAMEWORK SWITCH: this file imports {rival}, but the "
                    f"scaffold is {family} — a HARD constraint violation. "
                    f"Extend the existing {family} app; do not rewrite it "
                    f"in {rival}."
                )
        return None

    def set_tasks(self, tasks: list) -> None:
        """Seed the checklist (one entry per gap-analysis task).

        An entry may be a plain string, or a dict ``{"text": ...,
        "verify": callable}``. A deterministic verifier must pass before the
        item is verified. Without one, explicit current write evidence can
        record implementation only; acceptance remains unverified.
        """
        self._tasks = []
        for i, t in enumerate(tasks or []):
            if isinstance(t, dict):
                text = str(t.get("text", "")).strip()
                verify = t.get("verify")
                kind = t.get("kind")
                planning_notes = [note for note in (t.get("planning_notes") or []) if isinstance(note, str)]
            else:
                text, verify = str(t).strip(), None
                kind = None
                planning_notes = []
            if text:
                self._tasks.append(
                    {"id": len(self._tasks) + 1, "text": text,
                     "done": False, "verify": verify,
                     "verification": "unverified",
                     **({"kind": kind} if isinstance(kind, str) and kind else {}),
                     **({"planning_notes": planning_notes} if planning_notes else {}),
                     "attempts": 0, "blocked": False}
                )

    def autoclose_verified_tasks(self) -> list[dict]:
        """Close open items whose deterministic verifier now passes.

        A task with a real verifier does not need the model to argue for it:
        the harness can see the endpoint is implemented, and checklist
        bookkeeping otherwise costs more turns than the edits. Checking here
        also means a model that never learns the checklist protocol still gets
        credit for what it built.

        Only verifier-backed items qualify. An evidence-only task still needs
        the model to cite its work, because nothing here can judge it.
        """
        closed = []
        for task in self._tasks:
            verify = task.get("verify")
            if verify is None or task.get("done") or task.get("dropped"):
                continue
            try:
                passed = bool(verify())
            except Exception:
                continue
            if passed:
                task.update(done=True, verification="verified", blocked=False)
                task.pop("blocked_reason", None)
                closed.append({"id": task["id"], "text": task["text"][:120]})
        return closed

    def reopen_unverifiable_tasks(self) -> list[int]:
        """Re-run verifiers on completed items and reopen those that now fail.

        Restoring the pre-Phase-3 tree undoes the code but not the checklist,
        so work Phase 3 implemented would otherwise stay done/verified over
        restored stubs. Only items with a deterministic verifier can be
        re-judged; evidence-only completions are left alone rather than
        guessed at.
        """
        reopened = []
        for task in self._tasks:
            verify = task.get("verify")
            if verify is None or not task.get("done") or task.get("dropped"):
                continue
            try:
                still_true = bool(verify())
            except Exception:
                still_true = False
            if not still_true:
                task.update(done=False, verification="unverified", blocked=False)
                task.pop("blocked_reason", None)
                reopened.append(task["id"])
        return reopened

    def task_snapshot(self) -> list[dict]:
        """Return the serializable checklist state for crash recovery.

        Verifier callables are process-local and intentionally excluded. The
        orchestrator supplies freshly reconstructed deterministic verifiers to
        :meth:`restore_tasks` when resuming.
        """
        return [
            {"id": t["id"], "text": t["text"], "done": bool(t["done"]),
             "attempts": int(t.get("attempts") or 0),
             "blocked": bool(t.get("blocked")),
             "verification": t.get("verification", "unverified"),
             **{key: t[key] for key in ("blocked_reason", "dropped", "implementation_evidence", "planning_notes", "kind") if key in t}}
            for t in self._tasks
        ]

    def restore_tasks(
        self,
        snapshot: list[dict],
        verification_tasks: list[dict] | None = None,
    ) -> None:
        """Restore a checkpointed checklist without reopening completed work."""
        verifiers = {
            str(item.get("text", "")).strip(): item.get("verify")
            for item in (verification_tasks or [])
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        }
        restored: list[dict] = []
        for item in snapshot or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            raw_id = item.get("id")
            task_id = (
                raw_id
                if isinstance(raw_id, int) and raw_id > 0
                else len(restored) + 1
            )
            restored.append({
                "id": task_id,
                "text": text,
                "done": item.get("done") is True,
                "verify": verifiers.get(text),
                "attempts": int(item.get("attempts") or 0),
                "blocked": item.get("blocked") is True,
                "verification": item.get("verification", "unverified"),
                **{key: item[key] for key in ("blocked_reason", "dropped", "implementation_evidence", "planning_notes", "kind") if key in item},
            })
        self._tasks = restored

    def open_tasks(self) -> list[dict]:
        """Checklist items still to do.

        A BLOCKED item is excluded: its verifier has refused
        ``_MAX_TASK_VERIFY_ATTEMPTS`` times, so leaving it open would keep the
        run going until the turn cap for work that cannot be signed off. It is
        reported through :meth:`blocked_tasks` instead.
        """
        return [t for t in self._tasks if not t["done"] and not t.get("blocked")]

    def blocked_tasks(self) -> list[dict]:
        """Unresolved items, explicitly blocked or refused too many times."""
        return [t for t in self._tasks if t.get("blocked")]

    def unverified_tasks(self) -> list[dict]:
        """Implemented checklist items whose acceptance has not been checked."""
        return [t for t in self._tasks if t["done"] and not t.get("dropped")
                and t.get("verification", "unverified") != "verified"]

    def _task_implementation_evidence(self, task_id: int, args: dict) -> tuple[list[dict], str | None]:
        """Validate modification evidence, never treat it as acceptance proof."""
        evidence = args.get("evidence")
        if not isinstance(evidence, list) or not evidence or len(evidence) > 20:
            return [], "no verifier is attached; provide evidence=[{id, path, quote}] from actual successful writes, or mark blocked with a reason"
        accepted = []
        for item in evidence:
            if not isinstance(item, dict) or item.get("id") != task_id:
                continue
            rel, quote = item.get("path"), item.get("quote")
            if not isinstance(rel, str) or not isinstance(quote, str) or not quote.strip() or len(quote) > 4000:
                return [], "each evidence item requires a path and a non-empty exact quote of at most 4000 characters"
            try:
                path = self._safe_path(rel)
                with open(path, "r", encoding="utf-8") as source:
                    content = source.read()
            except (ValueError, OSError, UnicodeError):
                return [], f"evidence path is not a readable workspace file: {rel}"
            # read_file numbers every line, so a quote copied straight out of
            # it never matches; strip that prefix as modify_file's ladder does,
            # or implemented tasks end up BLOCKED.
            if quote not in content:
                unnumbered = _strip_line_numbers(quote.splitlines())
                if unnumbered is not None and "\n".join(unnumbered) in content:
                    quote = "\n".join(unnumbered)
            digest = self._content_digest(content)
            origin = "written"
            if self._successful_writes.get(os.path.normcase(path)) != digest:
                if args.get("existing") is not True:
                    return [], (f"no successful write matches the current contents of {rel}; "
                                "if this implementation already exists, read it and submit "
                                "existing=true with exact executable evidence (acceptance remains unverified)")
                if os.path.relpath(path, self.workspace).replace("\\", "/") not in self._known_paths:
                    return [], f"read {rel} before citing an existing implementation"
                from besser.spec_driven_agent.planning.requirements_ledger import verify_evidence
                task = next((t for t in self._tasks if t["id"] == task_id), {})
                evidence_kind = task.get("kind")
                if not isinstance(evidence_kind, str) or evidence_kind not in (
                    "validation", "uniqueness", "rule", "computed", "transition", "action", "ui",
                ):
                    evidence_kind = "implementation"
                checked = verify_evidence([{
                    "id": task_id, "text": task.get("text", ""),
                    # An untyped planning task is not necessarily an action:
                    # unique columns, defaults and declarative UI bindings can
                    # already implement it. Evidence is not acceptance proof.
                    "kind": evidence_kind,
                    "status": "implemented", "evidence": f"{rel}: {quote}",
                }], self.workspace)
                if not checked or checked[0]["status"] != "implemented":
                    detail = checked[0].get("note", "") if checked else "no valid citation"
                    return [], ("existing implementation evidence could not be verified: "
                                f"{detail}; cite the actual implementation; acceptance remains unverified")
                origin = "existing"
            if quote not in content:
                return [], f"evidence quote is not present in {rel}"
            accepted.append({"path": os.path.relpath(path, self.workspace).replace("\\", "/"),
                             "quote": quote, "sha256": digest, "origin": origin})
        if not accepted:
            return [], f"no implementation evidence supplied for task {task_id}"
        return accepted, None

    @staticmethod
    def _requested_task_ids(args: dict) -> tuple[list[int], list]:
        """Ids from ``ids=[...]`` or a single ``id=N``; also the unusable ones.

        Accepting a list is the point: marking items done one per turn made
        bookkeeping 40% of all turns across a 10-run batch (83% in the worst
        run), because every turn pays a full prompt prefill.
        """
        raw = args.get("ids")
        if raw is None:
            single = args.get("id")
            raw = [] if single is None else [single]
        elif not isinstance(raw, (list, tuple)):
            raw = [raw]
        ids: list[int] = []
        bad: list = []
        for value in raw:
            if isinstance(value, bool):      # bool is an int subclass; not an id
                bad.append(value)
                continue
            try:
                ids.append(int(value))
            except (TypeError, ValueError):
                bad.append(value)
        seen, unique = set(), []
        for i in ids:                        # keep order, drop repeats
            if i not in seen:
                seen.add(i)
                unique.append(i)
        return unique, bad

    def _task_list(self, args: dict) -> dict:
        action = args.get("action")
        if action == "list":
            return {
                "tasks": [
                    {"id": t["id"], "text": t["text"],
                     "status": ("dropped" if t.get("dropped") else "blocked" if t.get("blocked")
                                else "done" if t["done"] and t.get("verification") == "verified"
                                else "implemented" if t["done"] else "open"),
                     "verification": t.get("verification", "unverified"),
                     **({"planning_notes": t["planning_notes"]} if t.get("planning_notes") else {}),
                     **({"reason": t["dropped"]} if t.get("dropped") else
                        {"reason": t["blocked_reason"]} if t.get("blocked_reason") else {})}
                    for t in self._tasks
                ],
                "open": len(self.open_tasks()),
                "unverified": len(self.unverified_tasks()),
            }
        if action == "done":
            return self._do_done(args)
        if action == "blocked":
            return self._do_blocked(args)
        if action == "drop":
            return self._do_drop(args)
        if action == "add":
            return self._do_add(args)
        if action == "mixed":
            return self._do_mixed(args)
        return {"error": f"Unknown action '{action}'. Use list | done | add | drop | blocked | mixed."}

    def _do_done(self, args: dict) -> dict:
        """action='done' — also the `done_ids`+`evidence` verb inside a mixed call."""
        ids, bad = self._requested_task_ids(args)
        if not ids:
            return {"error": (
                "action='done' requires `id` (an integer) or `ids` "
                "(a list of integers). Use action='list' to see ids."
                + (f" Not integers: {bad}." if bad else "")
            )}

        done: list[int] = []
        refused: list[dict] = []
        blocked: list[dict] = []
        already_blocked: list[dict] = []
        unknown: list[int] = []
        by_id = {t["id"]: t for t in self._tasks}

        for task_id in ids:
            t = by_id.get(task_id)
            if t is None:
                unknown.append(task_id)
                continue
            was_blocked = bool(t.get("blocked"))
            if t["done"]:
                done.append(task_id)
                continue
            verify = t.get("verify")
            reason = None
            if verify is not None:
                try:
                    verified = bool(verify())
                except Exception as exc:
                    verified = False
                    reason = f"verification could not run ({type(exc).__name__}): {str(exc)[:300]}"
                if verified:
                    t["verification"] = "verified"
                elif reason is None:
                    reason = f"its check still fails: {t['text']}"
            else:
                evidence, reason = self._task_implementation_evidence(task_id, args)
                if reason is None:
                    t["implementation_evidence"] = evidence
                    t["verification"] = "unverified"
            if reason is not None:
                if was_blocked:
                    # A later repair may make the check pass. Re-evaluate
                    # it, but do not reopen another retry budget on failure.
                    already_blocked.append({"id": task_id, "text": t["text"],
                                            "reason": t.get("blocked_reason", reason)})
                    continue
                # Bound retries, including broken/missing verifiers.
                t["attempts"] = int(t.get("attempts") or 0) + 1
                if t["attempts"] >= _MAX_TASK_VERIFY_ATTEMPTS:
                    # Preserve the unresolved requirement without livelock.
                    t["blocked"] = True
                    t["blocked_reason"] = reason
                    blocked.append({"id": task_id, "text": t["text"], "reason": reason})
                else:
                    refused.append({
                        "id": task_id,
                        "attempts": t["attempts"],
                        "remaining_attempts": _MAX_TASK_VERIFY_ATTEMPTS - t["attempts"],
                        "reason": reason,
                    })
                continue
            t["done"] = True
            t["blocked"] = False
            t.pop("blocked_reason", None)
            done.append(task_id)

        remaining = self.open_tasks()
        result: dict = {
            "status": "done" if done else "refused",
            "done_ids": done,
            "verified_ids": [task_id for task_id in done if by_id[task_id].get("verification") == "verified"],
            "unverified_ids": [task_id for task_id in done if by_id[task_id].get("verification") != "verified"],
            "open_remaining": len(remaining),
            "open_items": [{"id": r["id"], "text": r["text"]} for r in remaining],
        }
        if len(ids) == 1:
            result["id"] = ids[0]
        if result["unverified_ids"]:
            result["verification_note"] = "Write evidence records implementation only; requirement acceptance is still unverified."
        if already_blocked:
            result["already_blocked"] = already_blocked
            if not done and not refused and not blocked and not unknown:
                result["status"] = "blocked"
        if refused:
            result["refused"] = refused
            result["advice"] = (
                "Do the work for the refused items, then mark them done. "
                "You may pass several ids at once: ids=[1,2,3]."
            )
        if blocked:
            result["blocked"] = blocked
            result["advice"] = (
                f"These items failed their check {_MAX_TASK_VERIFY_ATTEMPTS} "
                "times and are now recorded as BLOCKED. Do NOT call task_list "
                "for them again - move on to the remaining work."
            )
        if unknown:
            result["unknown_ids"] = unknown

        if not done:
            # ``_result_status`` treats an outcome as an error ONLY when the
            # payload carries an "error" key, and the loop guards, tracing
            # and "don't re-attempt" memory all depend on seeing a call that
            # accepted nothing as one.
            parts = []
            for item in refused:
                parts.append(
                    f"Task {item['id']} is NOT done - {item['reason']} "
                    f"(attempt {item['attempts']} of "
                    f"{_MAX_TASK_VERIFY_ATTEMPTS}). Do the work first, "
                    "then mark it done."
                )
            for item in blocked:
                parts.append(
                    f"Task {item['id']} is NOT done and is now BLOCKED "
                    f"after {_MAX_TASK_VERIFY_ATTEMPTS} failed checks: "
                    f"{item['text']}"
                )
            for task_id in unknown:
                parts.append(
                    f"No task with id {task_id}. Use action='list' to see ids."
                )
            if parts:
                result["error"] = " ".join(parts)
        return result

    def _do_blocked(self, args: dict) -> dict:
        """action='blocked' — also one entry of the `blocked` list inside a mixed call."""
        ids, bad = self._requested_task_ids(args)
        reason = str(args.get("reason") or "").strip()
        if not ids or bad or not reason:
            return {"error": "action='blocked' requires id or ids and a non-empty reason explaining what remains unresolved."}
        by_id = {t["id"]: t for t in self._tasks}
        if any(task_id not in by_id for task_id in ids):
            return {"error": "Unknown task id. Use action='list' to see ids."}
        for task_id in ids:
            task = by_id[task_id]
            task.pop("dropped", None)
            task.update(done=False, blocked=True, blocked_reason=reason[:2000], verification="unverified")
        return {"status": "blocked", "blocked_ids": ids, "reason": reason[:2000], "open": len(self.open_tasks())}

    def _do_drop(self, args: dict) -> dict:
        """action='drop' — also one entry of the `drop` list inside a mixed call.

        The honest exit for an item the user never asked for. Without
        it the end-turn gate leaves only a false 'done'.
        """
        task_id = args.get("id")
        reason = str(args.get("reason") or "").strip()
        if not isinstance(task_id, int) or not reason:
            return {"error": (
                "action='drop' requires `id` (an integer) and a non-empty `reason` "
                "saying why the user did not ask for it."
            )}
        for t in self._tasks:
            if t["id"] != task_id:
                continue
            if t["done"]:
                return {"error": f"Task {task_id} is already closed."}
            t["done"] = True
            t["dropped"] = reason
            t["blocked"] = False
            t.pop("blocked_reason", None)
            logger.info("Checklist item %d dropped (%s): %r", task_id, reason, t["text"][:80])
            return {"status": "dropped", "id": task_id, "open": len(self.open_tasks())}
        return {"error": f"Unknown task id {task_id}. Use action='list' to see ids."}

    def _do_add(self, args: dict) -> dict:
        """action='add' — also the `add_texts` verb inside a mixed call.

        `texts` appends several in one call (one add per turn wastes turns,
        as with ``done``); `text` keeps the single shape existing prompts and
        checkpoints use.
        """
        raw = args.get("texts")
        batched = isinstance(raw, list)
        if not batched:
            raw = [args.get("text")]
        items = [t.strip() for t in raw if isinstance(t, str) and t.strip()]
        if not items:
            return {"error": (
                "action='add' requires non-empty `text`, or `texts` with "
                "at least one non-empty item."
            )}
        ids: list[int] = []
        appended: set[int] = set()
        for text in items:
            # Same item already listed (case, whitespace, trailing period
            # aside): hand back its id rather than a second copy the model
            # will later notice and work through again.
            key = " ".join(text.lower().split()).rstrip(".")
            existing = next(
                (t for t in self._tasks
                 if " ".join(t["text"].lower().split()).rstrip(".") == key),
                None,
            )
            if existing is not None:
                ids.append(existing["id"])
                continue
            new_id = max((t["id"] for t in self._tasks), default=0) + 1
            self._tasks.append({"id": new_id, "text": text, "done": False})
            ids.append(new_id)
            appended.add(new_id)
        if batched:
            return {"status": "added", "ids": ids, "open": len(self.open_tasks())}
        # Single-item shape unchanged, including the "exists" status a
        # caller may branch on.
        only = ids[0]
        return {
            "status": "added" if only in appended else "exists",
            "id": only,
            "open": len(self.open_tasks()),
        }

    def _do_mixed(self, args: dict) -> dict:
        """action='mixed' — several checklist verbs in ONE call.

        Mixing a ``done`` and an ``add`` in the same turn would otherwise need
        two calls (plus often a ``list``) because ``action`` is a single enum.
        Each verb here
        reuses the EXACT single-action handler that ``action='done'`` /
        ``'add'`` / ``'drop'`` / ``'blocked'`` calls, so every gate
        (evidence, deterministic verifiers, the block-after-N-attempts guard,
        drop's reason requirement) still applies per item, unchanged.

        The response nests each verb's own result under ``results`` so a
        model can tell which parts succeeded and which were refused: a mixed
        call that half-succeeds carries no top-level ``error`` (real progress
        happened, the same convention a partial ``done`` batch already uses)
        but still shows the failing verb's own error underneath.
        """
        done_ids = args.get("done_ids")
        add_texts = args.get("add_texts")
        drop_items = args.get("drop")
        blocked_items = args.get("blocked")
        if done_ids is None and add_texts is None and drop_items is None and blocked_items is None:
            return {"error": (
                "action='mixed' requires at least one of `done_ids`, `add_texts`, "
                "`drop`, or `blocked`. Use the single-verb actions to do just one thing."
            )}

        results: dict[str, dict] = {}
        succeeded = False
        failed = False

        def _record(verb: str, sub: dict) -> None:
            nonlocal succeeded, failed
            results[verb] = sub
            if self._result_status(sub) == "error":
                failed = True
            else:
                succeeded = True

        if done_ids is not None:
            _record("done", self._do_done({
                "ids": done_ids, "evidence": args.get("evidence"), "existing": args.get("existing"),
            }))

        if add_texts is not None:
            _record("add", self._do_add({"texts": add_texts}))

        if drop_items is not None:
            if not isinstance(drop_items, list) or not drop_items:
                _record("drop", {"error": "`drop` must be a non-empty list of {id, reason} objects."})
            else:
                dropped_ids: list[int] = []
                errors: list[str] = []
                for item in drop_items:
                    sub = self._do_drop(item if isinstance(item, dict) else {})
                    if self._result_status(sub) == "error":
                        errors.append(sub.get("error", "drop failed"))
                    else:
                        dropped_ids.append(sub["id"])
                # ``_result_status`` only recognizes a singular "error" key, so
                # a verb where EVERY entry failed must carry one too, or the
                # harness (and _record's own success/failure split) would
                # misread total failure here as success.
                _record("drop", {
                    "dropped_ids": dropped_ids,
                    **({"errors": errors} if errors else {}),
                    **({"error": " | ".join(errors)} if errors and not dropped_ids else {}),
                })

        if blocked_items is not None:
            if not isinstance(blocked_items, list) or not blocked_items:
                _record("blocked", {"error": "`blocked` must be a non-empty list of {id, reason} objects."})
            else:
                blocked_ids: list[int] = []
                errors: list[str] = []
                for item in blocked_items:
                    sub = self._do_blocked(item if isinstance(item, dict) else {})
                    if self._result_status(sub) == "error":
                        errors.append(sub.get("error", "blocked failed"))
                    else:
                        blocked_ids.extend(sub.get("blocked_ids", []))
                _record("blocked", {
                    "blocked_ids": blocked_ids,
                    **({"errors": errors} if errors else {}),
                    **({"error": " | ".join(errors)} if errors and not blocked_ids else {}),
                })

        result: dict = {
            "status": "ok" if succeeded and not failed else "partial" if succeeded else "refused",
            "results": results,
            "open": len(self.open_tasks()),
        }
        if not succeeded:
            parts = []
            for verb, sub in results.items():
                if sub.get("error"):
                    parts.append(f"{verb}: {sub['error']}")
                elif sub.get("errors"):
                    parts.append(f"{verb}: " + " | ".join(sub["errors"]))
            result["error"] = " ".join(parts) if parts else "Nothing in this mixed task_list call was accepted."
        return result

    def _contract_warnings(self, rel_path: str, content: str) -> str | None:
        """Lint freshly-written content against the model's data contract."""
        switch = self._framework_switch_warning(rel_path, content)
        if self._data_contract is None:
            return switch
        try:
            from besser.spec_driven_agent.validation.contract_checks import format_findings, lint_file
            findings = lint_file(rel_path, content, self._data_contract)
        except Exception:
            return switch
        if not findings:
            return switch
        note = (
            "DATA-CONTRACT VIOLATIONS in the content you just wrote — "
            "fix them now, while you still have the file in context:\n"
            + format_findings(findings)
        )
        return f"{switch}\n{note}" if switch else note

    def _repair_imports(self, rel_path: str, content: str, result: dict) -> str:
        """Write back the same file with the forgotten import added.

        Returns the content now on disk. Silent on failure: this is a
        convenience, and Phase 3 remains the backstop.
        """
        try:
            from besser.spec_driven_agent.repair.import_repair import repair_missing_imports

            repaired, notes = repair_missing_imports(rel_path, content)
            if not notes or repaired == content:
                return content
            path = self._safe_path(rel_path)
            with open(path, "w", encoding="utf-8", newline="\n") as target:
                target.write(repaired)
            self._successful_writes[os.path.normcase(path)] = self._content_digest(repaired)
            result["auto_imports"] = notes
            return repaired
        except Exception:
            logger.debug("import repair failed on %s", rel_path, exc_info=True)
            return content

    def _append_write_feedback(
        self,
        result: dict,
        rel_path: str,
        content: str,
    ) -> None:
        """Attach bounded contract and parser feedback to a successful write.

        A framework name the file never imported is repaired here first, so
        the break cannot cascade: an ORM module that stops importing kills
        every router that star-imports it.
        """
        content = self._repair_imports(rel_path, content, result)
        warnings = self._contract_warnings(rel_path, content)
        if warnings:
            result["contract_warnings"] = warnings
        if not self._per_write_diagnostics:
            return
        try:
            from besser.spec_driven_agent.validation.write_diagnostics import diagnose_written_content

            diagnostics = diagnose_written_content(rel_path, content, workspace=self.workspace)
            if rel_path.endswith(".py"):
                # A schema edit can break untouched routers. Report those
                # consumers immediately, not only at final validation.
                from besser.spec_driven_agent.validation.python_source import _create_schema_router_mismatches

                diagnostics.extend({
                    "source": "project-contract", "severity": "error",
                    "code": "schema-consumer-mismatch", "message": message,
                } for message in _create_schema_router_mismatches(self.workspace)[:10])
        except Exception:
            diagnostics = []
        if rel_path.endswith((".py", ".js", ".jsx", ".ts", ".tsx")):
            try:
                from besser.spec_driven_agent.validation.frontend_schema import collect_frontend_schema_diagnostics

                diagnostics.extend(collect_frontend_schema_diagnostics(
                    self.workspace, changed_path=rel_path, content=content,
                )[:10])
            except Exception:
                # An optional cross-file check must not erase parser findings
                # already collected for the successful write.
                logger.debug("Frontend/schema write diagnostics failed", exc_info=True)
        if diagnostics:
            result["diagnostics"] = diagnostics
            result["diagnostic_message"] = (
                "The content was written, but diagnostics found errors. Fix them "
                "now while this file is still in context."
            )

    def _require_domain_model(self, tool_name: str) -> dict | None:
        """Return an error dict if no domain model is loaded, else None.

        Called at the top of every handler that needs ``self.domain_model``.
        Surfaces a clear "you can't do this without a ClassDiagram" error
        to the LLM instead of raising an AttributeError mid-handler.
        """
        if self.domain_model is None:
            return {
                "error": (
                    f"{tool_name} requires a domain model (ClassDiagram) "
                    "but this project does not include one. Skip this tool "
                    "and use write_file / run_command instead."
                ),
            }
        return None

    @staticmethod
    def _result_status(payload: dict[str, Any]) -> ToolExecutionStatus:
        """Classify a handler payload without relying on its presentation."""
        if (
            payload.get("error") is not None
            or payload.get("status") == "error"
            or payload.get("success") is False
        ):
            return "error"
        if payload.get("skipped") is True or payload.get("status") == "skipped":
            return "skipped"
        return "ok"

    def execute_typed(
        self, tool_name: str, arguments: dict
    ) -> ToolExecutionResult:
        """Execute a tool and return a machine-readable harness outcome.

        Handler failures stay data, never exceptions. The model-facing payload
        is deliberately unchanged; only the harness gains an explicit
        ``ok | error | skipped`` status.
        """
        # The shell gate is enforced HERE, not only in the advertised tool list.
        # `get_tools_for(allow_shell=False)` filters the menu, but the handler
        # table still holds these two and a model can name one anyway — on an
        # OpenAI-compatible endpoint the tool name comes verbatim out of the
        # model's output, and the Phase-3 fix prompt names `run_command`
        # unconditionally. Filtering the menu is not a gate; refusing is.
        if tool_name in _SHELL_TOOL_NAMES and not self.allow_shell:
            logger.warning(
                "Refused %s: shell tools are disabled for this run", tool_name,
            )
            return ToolExecutionResult("error", {
                "error": (
                    f"{tool_name} is not available in this run. Shell access is "
                    "disabled. Use the file tools (write_file / modify_file) "
                    "instead — do not try to run commands."
                ),
            })

        handler = self._handlers.get(tool_name)
        if not handler:
            payload = {"error": f"Unknown tool: {tool_name}"}
            return ToolExecutionResult("error", payload)
        try:
            file_tool = tool_name in {"read_file", "modify_file", "replace_file_lines", "write_file", "delete_file"}
            lock = _file_lock(self._safe_path(arguments["path"])) if file_tool else nullcontext()
            with lock:
                # Aliases of a file must share rejection/freeze/read state too.
                if file_tool:
                    arguments = {**arguments, "path": os.path.relpath(
                        self._safe_path(arguments["path"]), self.workspace,
                    ).replace("\\", "/")}
                raw_result = handler(self, arguments)
                if file_tool and isinstance(raw_result, dict):
                    self._add_edit_recovery(tool_name, arguments, raw_result)
            payload = raw_result if isinstance(raw_result, dict) else {"result": raw_result}
            return ToolExecutionResult(self._result_status(payload), payload)
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e, exc_info=True)
            return ToolExecutionResult(
                "error", {"error": f"{tool_name} failed: {e}"}
            )

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        Execute a tool call.  Returns a JSON string.

        Errors are returned as ``{"error": "..."}`` — never raised.
        """
        return self.execute_typed(tool_name, arguments).to_json()

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
        # Return the NORMALISED path, not the raw realpath. self.workspace is
        # stored prefix-free, so a returned "\\?\C:\..." makes every downstream
        # os.path.relpath(path, self.workspace) raise "path is on mount
        # '\\?\C:', start on mount 'C:'". realpath keeps the prefix when
        # _getfinalpathname fails transiently (e.g. WinError 1450), so the
        # failure is intermittent. Containment was already proven against
        # full_cmp, so returning it is also the self-consistent answer.
        return full_cmp

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
        if not os.path.isdir(cwd_cmp):
            os.makedirs(cwd_cmp, exist_ok=True)
        # Same reason as _safe_path: the raw realpath can carry the Windows
        # extended prefix, and this value becomes a subprocess cwd and the
        # sandbox's --chdir.
        return cwd_cmp

    def _list_dir(self, directory: str) -> list[dict]:
        """List files recursively, relative to workspace.

        Installed dependencies and caches are pruned: a single npm install
        can put 10k node_modules paths into one listing and overflow the
        model's context on the next turn.
        """
        from besser.spec_driven_agent.agent.prompt_builder import _SNAPSHOT_SKIP_DIRS

        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in _SNAPSHOT_SKIP_DIRS]
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
            "This generated a MODULAR FastAPI app: main_api.py (slim — CORS, middleware, "
            "exception handlers, system endpoints, and include_router() calls), database.py "
            "(engine/session/get_db), routers/<Class>.py (one APIRouter per class holding "
            "that class's CRUD/relationship/method endpoints, decorated with @router — NOT "
            "@app), sql_alchemy.py (ORM), pydantic_classes.py (schemas), and bal_stdlib.py "
            "(helpers). "
            "To add auth: create a NEW auth.py file, then use modify_file to add "
            "Depends(get_current_user) to the endpoints in the relevant routers/<Class>.py. "
            "To add pagination: use modify_file to add skip/limit parameters to the GET list "
            "endpoints in routers/<Class>.py. "
            "To add a new endpoint: add it to the appropriate routers/<Class>.py (use @router), "
            "or create a new router file and register it with include_router() in main_api.py. "
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
        err = self._require_domain_model("generate_pydantic")
        if err:
            return err
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
        err = self._require_domain_model("generate_sqlalchemy")
        if err:
            return err
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
        err = self._require_domain_model("generate_fastapi_backend")
        if err:
            return err
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
        err = self._require_domain_model("generate_django")
        if err:
            return err
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
        err = self._require_domain_model("generate_python_classes")
        if err:
            return err
        from besser.generators.python_classes import PythonGenerator
        out = self._gen_dir("python")
        PythonGenerator(model=self.domain_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_qiskit(self, args: dict) -> dict:
        if self.quantum_circuit is None:
            return {
                "error": (
                    "generate_qiskit requires a quantum circuit model but this "
                    "project does not include one. Skip this tool and use "
                    "write_file / run_command instead."
                ),
            }
        from besser.generators.qiskit.qiskit_generator import QiskitGenerator
        out = self._gen_dir("qiskit")
        QiskitGenerator(
            model=self.quantum_circuit, output_dir=out,
            backend_type=args.get("backend_type", "aer_simulator"),
            shots=int(args.get("shots", 1024)),
        ).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_supabase(self, args: dict) -> dict:
        err = self._require_domain_model("generate_supabase")
        if err:
            return err
        from besser.generators.supabase.supabase_generator import SupabaseGenerator
        out = self._gen_dir("supabase")
        SupabaseGenerator(
            model=self.domain_model, output_dir=out,
            user_root=args.get("user_root", "User"),
        ).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_json_object(self, args: dict) -> dict:
        if self.object_model is None:
            return {
                "error": (
                    "generate_json_object requires an object model (ObjectDiagram) "
                    "but this project does not include one. Skip this tool and use "
                    "write_file instead."
                ),
            }
        from besser.generators.json.json_object_generator import JSONObjectGenerator
        out = self._gen_dir("json_object")
        JSONObjectGenerator(model=self.object_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_baf(self, args: dict) -> dict:
        if self.agent_model is None:
            return {
                "error": (
                    "generate_baf requires an agent model (AgentDiagram) but this "
                    "project does not include one. Skip this tool and use write_file "
                    "instead."
                ),
            }
        # Pure templated generation (GenerationMode.FULL without a config emits the
        # templated agent code); personalization needs a provider key we don't hold
        # here, so we deliberately skip it. Wrapped so a generator error surfaces as
        # a tool error rather than aborting the run.
        try:
            from besser.generators.agents.baf_generator import BAFGenerator
            out = self._gen_dir("baf")
            BAFGenerator(model=self.agent_model, output_dir=out).generate()
        except Exception as exc:
            return {"status": "error", "error": f"BAF generation failed: {exc}"}
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_bpmn(self, args: dict) -> dict:
        if self.bpmn_model is None:
            return {
                "error": (
                    "generate_bpmn requires a BPMN model (BPMNDiagram) but this "
                    "project does not include one. Skip this tool and use write_file "
                    "instead."
                ),
            }
        from besser.generators.bpmn.bpmn_generator import BPMNGenerator
        out = self._gen_dir("bpmn")
        BPMNGenerator(model=self.bpmn_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_pytorch(self, args: dict) -> dict:
        if self.nn_model is None:
            return {
                "error": (
                    "generate_pytorch requires a neural-network model (NNDiagram) "
                    "but this project does not include one. Skip this tool and use "
                    "write_file instead."
                ),
            }
        # torch is an OPTIONAL dependency, not installed in the hosted image —
        # import lazily and degrade to a clear tool error rather than crashing.
        try:
            from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator
            out = self._gen_dir("pytorch")
            PytorchGenerator(
                model=self.nn_model, output_dir=out,
                generation_type=args.get("generation_type", "subclassing"),
            ).generate()
        except ImportError as exc:
            return {"status": "error", "error": f"PyTorch is not installed in this environment ({exc})."}
        except Exception as exc:
            return {"status": "error", "error": f"PyTorch generation failed: {exc}"}
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_tensorflow(self, args: dict) -> dict:
        if self.nn_model is None:
            return {
                "error": (
                    "generate_tensorflow requires a neural-network model (NNDiagram) "
                    "but this project does not include one. Skip this tool and use "
                    "write_file instead."
                ),
            }
        # tensorflow is an OPTIONAL dependency — import lazily and degrade cleanly.
        try:
            from besser.generators.nn.tf.tf_code_generator import TFGenerator
            out = self._gen_dir("tensorflow")
            TFGenerator(
                model=self.nn_model, output_dir=out,
                generation_type=args.get("generation_type", "subclassing"),
            ).generate()
        except ImportError as exc:
            return {"status": "error", "error": f"TensorFlow is not installed in this environment ({exc})."}
        except Exception as exc:
            return {"status": "error", "error": f"TensorFlow generation failed: {exc}"}
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_java_classes(self, args: dict) -> dict:
        err = self._require_domain_model("generate_java_classes")
        if err:
            return err
        from besser.generators.java_classes import JavaGenerator
        out = self._gen_dir("java")
        JavaGenerator(model=self.domain_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_sql(self, args: dict) -> dict:
        err = self._require_domain_model("generate_sql")
        if err:
            return err
        from besser.generators.sql import SQLGenerator
        out = self._gen_dir("sql")
        SQLGenerator(model=self.domain_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_json_schema(self, args: dict) -> dict:
        err = self._require_domain_model("generate_json_schema")
        if err:
            return err
        from besser.generators.json import JSONSchemaGenerator
        out = self._gen_dir("jsonschema")
        JSONSchemaGenerator(
            model=self.domain_model, output_dir=out,
            mode=args.get("mode", "regular"),
        ).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_rest_api(self, args: dict) -> dict:
        err = self._require_domain_model("generate_rest_api")
        if err:
            return err
        from besser.generators.rest_api import RESTAPIGenerator
        out = self._gen_dir("rest_api")
        RESTAPIGenerator(
            model=self.domain_model, output_dir=out,
            http_methods=args.get("http_methods", ["GET", "POST", "PUT", "DELETE"]),
        ).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_react(self, args: dict) -> dict:
        err = self._require_domain_model("generate_react")
        if err:
            return err
        if not self.gui_model:
            return {"error": "No GUI model available. The React generator requires a GUI model."}
        from besser.generators.react import ReactGenerator
        out = self._gen_dir("react")
        ReactGenerator(model=self.domain_model, gui_model=self.gui_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_flutter(self, args: dict) -> dict:
        err = self._require_domain_model("generate_flutter")
        if err:
            return err
        if not self.gui_model:
            return {"error": "No GUI model available. The Flutter generator requires a GUI model."}
        from besser.generators.flutter import FlutterGenerator
        out = self._gen_dir("flutter")
        FlutterGenerator(model=self.domain_model, gui_model=self.gui_model, output_dir=out).generate()
        return {"status": "ok", "files": self._track_generated_files(out)}

    def _gen_web_app(self, args: dict) -> dict:
        err = self._require_domain_model("generate_web_app")
        if err:
            return err
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
        err = self._require_domain_model("generate_rdf")
        if err:
            return err
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

    def _mint_read_view(self, path: str, content: str, start_line: int, end_line: int) -> str:
        """Authorize a 1-based inclusive line span of ``content`` for a range edit."""
        read_id = f"{self._read_epoch}:{next(self._read_sequence)}"
        self._read_views[read_id] = (
            os.path.normcase(path), self._content_digest(content), start_line, end_line,
        )
        if len(self._read_views) > 128:
            del self._read_views[next(iter(self._read_views))]
        return read_id

    def _read_file(self, args: dict) -> dict:
        """
        Read a file with optional line-based pagination, numbered ``   7| code``.

        Supports offset (start line, 0-indexed) and limit (number of lines)
        for reading specific sections of large files without dumping
        everything into context.
        """
        path = self._safe_path(args["path"])
        if not os.path.isfile(path):
            return self._missing_file_result(args["path"])
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
        if (type(offset) is not int or offset < 0
                or (limit is not None and (type(limit) is not int or limit < 1))):
            return {"error": "offset must be a non-negative integer and limit a positive integer."}

        # Apply line-based slicing if offset or limit specified
        start = min(offset, total_lines)
        end = total_lines
        if limit is not None:
            end = min(start + limit, total_lines)

        # Prefix every line with its TRUE file line number, in the same format
        # the post-edit echo uses, so a region quoted straight out of this
        # output lands on modify_file's line-number tier instead of a miss.
        selected = "\n".join(
            f"{n:>4}| {line}"
            for n, line in enumerate(lines[start:end], start=start + 1)
        )

        # Only complete, actually displayed lines are eligible for a range
        # edit. Never authorize the unseen tail of a truncated read.
        truncated = len(selected) > MAX_FILE_READ
        if len(selected) > MAX_FILE_READ:
            boundary = selected.rfind("\n", 0, MAX_FILE_READ)
            selected = selected[:boundary] if boundary >= 0 else ""
            end = start + (selected.count("\n") + 1 if selected else 0)

        result = {"content": selected}
        result["read_id"] = self._mint_read_view(path, content, start + 1, end)
        result["start_line"] = start + 1
        result["end_line"] = end
        if truncated:
            result["truncated"] = True
            result["content"] += "\n... [truncated; read the next range to continue]"
            result["hint"] = f"Read continues at offset={end}; unseen lines cannot be edited with this read_id."
        self._known_paths.add(args["path"].replace("\\", "/").strip())

        # Include metadata so the LLM knows about pagination
        if offset > 0 or limit is not None:
            result["start_line"] = start + 1  # 1-indexed for display
            result["end_line"] = end
            result["total_lines"] = total_lines
            result["lines_read"] = end - start
        elif total_lines > 200:
            # A whole large file still fits in MAX_FILE_READ; pagination is for
            # the ones that do not. Inviting it unconditionally teaches the
            # model to crawl a file in tiny windows.
            result["total_lines"] = total_lines

        # Crawling a file that fits in one read in 10-line windows can burn a
        # third of the turn budget. Prose in the tool description does not
        # stop it, so say it where the behaviour happens.
        rel = args["path"].replace("\\", "/").strip()
        self._read_counts[rel] = self._read_counts.get(rel, 0) + 1
        window = end - start
        if (self._read_counts[rel] >= 3 and not truncated
                and 0 < window < 40 and total_lines > 2 * window):
            result["hint"] = (
                f"This is read {self._read_counts[rel]} of {rel}, covering {window} "
                f"of {total_lines} lines. Read the whole function or class you intend "
                "to change in one call - a file this size fits in a single read. "
                "Each narrow window costs a turn and still omits the enclosing block."
            )

        return result

    # Consecutive range-edit refusals on one path before the ladder sends the
    # model back to text matching. Measured on Qwen, modify_file lands 86-92%
    # of edits while replace_file_lines lands 15-43%, so the ladder must not
    # keep steering into a failing range edit.
    _RANGE_EDIT_GIVE_UP = 3

    def _add_edit_recovery(self, tool: str, args: dict, result: dict) -> None:
        """Change strategy after two refusals, in every orchestration phase.

        The ladder runs both ways: quoting text, then a revision-bound range
        edit, then back to quoting when the range edit is the thing failing.
        Keep this in the typed execution path so a reworded/no-op/ambiguous
        failure cannot escape recovery. A read does not count as a write.
        """
        path = args["path"].replace("\\", "/")
        if tool in {"modify_file", "replace_file_lines", "write_file", "delete_file"}:
            if self._result_status(result) == "ok":
                self._edit_recovery.pop(path, None)
                self._failed_modifies.pop(path, None)
                self._last_missed_old_text.pop(path, None)
                self._range_edit_failures.pop(path, None)
                self._clear_rejections(path)
                self.last_repeat = None
            elif (tool in {"modify_file", "replace_file_lines"}
                    and result.get("status") not in {"already_applied", "possible_replay"}):
                self._edit_recovery[path] = self._edit_recovery.get(path, 0) + 1
                if tool == "replace_file_lines":
                    # A range edit carries no old_text for _modify_file's own
                    # fingerprinting, so key the rejection on the selected
                    # range. Without this, identical refused range edits leave
                    # last_repeat None and neither _REPEAT_STOP_AT nor the
                    # per-file streak guard can fire.
                    self._failed_modifies[path] = self._failed_modifies.get(path, 0) + 1
                    self._note_rejection(
                        path, f"lines {args.get('start_line')}-{args.get('end_line')}",
                        args.get("new_text") or "",
                    )
                    self._range_edit_failures[path] = self._range_edit_failures.get(path, 0) + 1
        exhausted = self._range_edit_failures.get(path, 0) >= self._RANGE_EDIT_GIVE_UP
        # _modify_file already bracketed the target and issued a read_id for it,
        # so steer on the FIRST miss: the span saves a re-read and a guess at
        # line numbers.
        located = result.get("located_range") if tool == "modify_file" else None
        # The located-range assist is a FIRST-miss aid only. Once two edits on
        # this path have been refused the ladder escalates to a whole-file
        # rewrite rather than to replace_file_lines.
        if located and self._edit_recovery.get(path, 0) < 2:
            result["edit_recovery"] = {
                "next_tool": "replace_file_lines", "path": path,
                "read_id": located["read_id"],
                "start_line": located["start_line"], "end_line": located["end_line"],
                "instruction": located["instruction"],
            }
            return
        if self._edit_recovery.get(path, 0) < 2 or self._frozen(path):
            return
        # Two refused edits on this path -> rewrite the whole file.
        # NOT because write_file is more reliable (it is not): successful range
        # edits on a drifted view overwrite neighbouring classes, so ending the
        # ladder at replace_file_lines measurably loses scaffold code.
        if tool in {"modify_file", "replace_file_lines", "read_file"}:
            result["edit_recovery"] = {
                "next_tool": "write_file", "path": path,
                "instruction": "Two edits on this file have been refused. Stop editing it "
                               "piecemeal: call read_file on the WHOLE file (no offset/limit), "
                               "then write_file the complete file back with your change "
                               "applied. Reproduce every line you read - do not summarise, "
                               "elide, or drop code you were not asked to change. No rejected "
                               "edit was applied; do not mark the requirement done.",
            }
            return
        if tool == "read_file" and "read_id" in result and not exhausted:
            result["edit_recovery"] = {
                "next_tool": "replace_file_lines", "path": path,
                "read_id": result["read_id"],
                "instruction": "Select the exact 1-based inclusive lines shown here and supply their complete replacement. "
                               "Do not quote old_text again. Keep surrounding code unchanged; a read is not an implementation.",
            }
        elif tool in {"modify_file", "replace_file_lines"} and self._result_status(result) == "error":
            if exhausted:
                result["edit_recovery"] = {
                    "next_tool": "modify_file", "path": path,
                    "instruction": "Range edits on this file keep being refused, so stop using them here. "
                                   "Go back to modify_file with the SMALLEST unique old_text that "
                                   "brackets your change - one or two lines copied from the latest "
                                   "read - instead of rewriting a whole block. No rejected edit was "
                                   "applied; do not mark the requirement done.",
                }
            else:
                result["edit_recovery"] = {
                    "next_tool": "read_file", "path": path,
                    "instruction": "Read the target function/block with offset/limit, then use replace_file_lines with its "
                                   "read_id and line numbers. Stop retrying the same text replacement. "
                                   "No rejected edit was applied; do not mark the requirement done.",
                }

    def _replace_file_lines(self, args: dict) -> dict:
        """Replace a deliberately selected range from an unchanged, visible read.

        No fuzzy targeting and no re-quotation of old code. Stale handles cannot
        overwrite intervening edits or replay an insertion, including on resume.
        """
        path = self._safe_path(args["path"])
        frozen = self._frozen(args["path"])
        if frozen:
            return frozen
        view = self._read_views.get(args.get("read_id", ""))
        if view is None or view[0] != os.path.normcase(path):
            return {"error": "Unknown read_id for this file. Call read_file on the target region first.",
                    "rejection_kind": "unread_range"}
        if not os.path.isfile(path):
            return self._missing_file_result(args["path"])
        with open(path, encoding="utf-8") as source:
            before = source.read()
        if self._content_digest(before) != view[1]:
            return {"error": "Stale read_id: the file changed after this read. No edit applied. "
                             "Read the region again; do not reuse old line numbers or mark the task done.",
                    "rejection_kind": "stale_read"}
        start, end = args.get("start_line"), args.get("end_line")
        if (type(start) is not int or type(end) is not int
                or not view[2] <= start <= end <= view[3]):
            # read_file's ``offset`` is a 0-based skip; the numbers printed
            # beside each line are 1-based. Models confuse the two and resend
            # the identical call, so name the correction, don't restate the
            # range.
            hint = ""
            if type(start) is int and start == view[2] - 1:
                hint = (f" You selected {start}, one before the first displayed line: "
                        "read_file's offset is a 0-based skip, while the numbers shown "
                        "beside each line are the 1-based ones to use here. Use the "
                        "start_line/end_line that read_file returned.")
            return {"error": f"Select 1-based inclusive lines within the displayed range {view[2]}-{view[3]}."
                             f"{hint} Unseen or truncated lines cannot be replaced; read the complete target block first.",
                    "rejection_kind": "unread_range",
                    "displayed_start_line": view[2], "displayed_end_line": view[3]}
        replacement = args.get("new_text")
        if not isinstance(replacement, str):
            return {"error": "new_text must be a string containing the complete replacement."}
        replacement = replacement.replace("\r\n", "\n")
        # read_file numbers what the model sees, and a model that selected a
        # range by those numbers often pastes them back. Strip them as
        # modify_file's ladder does, or " 101|       </nav>" lands on disk as
        # invalid source that nothing downstream may catch (e.g. with tsc off).
        numbering = _strip_line_numbers(replacement.split("\n"))
        stripped_numbering = numbering is not None
        if stripped_numbering:
            replacement = "\n".join(numbering)
        # Match read_file's newline-only numbering (splitlines would also
        # split form feeds / Unicode separators inside a source line).
        parts = before.split("\n")
        # Keep the terminal empty line if read_file displayed it. It is an
        # explicit zero-width EOF anchor, not an unseen/out-of-range line.
        lines = [line + "\n" for line in parts[:-1]] + [parts[-1]]
        old = "".join(lines[start - 1:end])
        elision = find_elision(replacement)
        if elision and elision[1].strip() not in elided_lines(old):
            return {"error": "new_text abbreviates the code. Write every replacement line in full; no '...' placeholders.",
                    "rejection_kind": "elision"}
        # Preserve the boundary to the next line, not the caller's indentation.
        if replacement and not replacement.endswith("\n") and old.endswith("\n"):
            replacement += "\n"
        after = "".join(lines[:start - 1]) + replacement + "".join(lines[end:])
        if after == before:
            return {"error": "This range edit makes no change; it is not evidence of implementation. "
                             "The lines you selected already read exactly like new_text. Call "
                             "read_file on the region to see what is actually there, then send "
                             "the change you meant - or mark the task blocked with a reason.",
                    "status": "no_change", "replacements": 0}
        broke = _new_syntax_error(args["path"], before, after)
        if broke:
            message, line = broke
            return {"error": f"Refused: replacement would make the file unparseable ({message} at line {line}). "
                             "The file was left unchanged. Correct new_text or select the complete enclosing block.",
                    "rejection_kind": "syntax_error", "syntax_line": line,
                    "would_write": "PROPOSED ONLY - NOT APPLIED:\n" + _changed_region(before, after),
                    "current_source": "CURRENT ON-DISK CONTENT:\n" + _changed_region(after, before)}
        unimportable = _breaks_module_import(path, before, after)
        if unimportable:
            return {"error": f"Refused: this edit leaves {args['path']} parseable but no longer "
                             f"importable ({unimportable}). Every router star-imports it, so this "
                             "would take the whole application down. The file was left unchanged. "
                             + _IMPORT_BREAK_ADVICE,
                    "rejection_kind": "breaks_import",
                    "would_write": "PROPOSED ONLY - NOT APPLIED:\n" + _changed_region(before, after)}
        with open(path, "w", encoding="utf-8", newline="\n") as target:
            target.write(after)
        self._successful_writes[os.path.normcase(path)] = self._content_digest(after)
        result = {"status": "modified", "path": args["path"], "replacements": 1,
                  "matched_by": "read_bound_range", "snippet": _changed_region(before, after)}
        if stripped_numbering:
            result["note"] = (
                "Your new_text carried read_file's 'NNN| ' line numbers on every "
                "line; they were stripped before writing. Send the code itself, "
                "not the numbered display."
            )
        self._append_write_feedback(result, args["path"], after)
        return result

    def _missing_file_result(self, requested: str) -> dict:
        """Suggest real workspace paths, without guessing or remapping the read."""
        requested = requested.replace("\\", "/")
        result = {
            "error": f"File not found: {requested}",
            "advice": "Use list_files or read one of the existing paths below; no path was substituted.",
        }
        candidates: list[tuple[float, str]] = []
        ignored = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build"}
        inspected = 0
        for root, dirs, names in os.walk(self.workspace):
            dirs[:] = sorted(d for d in dirs if d not in ignored and not d.startswith("."))
            for name in sorted(names):
                if name.startswith("."):
                    continue
                rel = os.path.relpath(os.path.join(root, name), self.workspace).replace("\\", "/")
                try:
                    self._safe_path(rel)  # Do not expose outside-workspace symlink targets.
                except ValueError:
                    continue
                score = SequenceMatcher(None, requested.lower(), rel.lower()).ratio()
                if name.lower() == requested.rsplit("/", 1)[-1].lower():
                    score += 1
                candidates.append((score, rel))
                inspected += 1
                if inspected >= 2000:
                    break
            if inspected >= 2000:
                break
        result["suggested_paths"] = [rel for _, rel in sorted(candidates, reverse=True)[:8]]
        return result

    @staticmethod
    def _content_digest(content: str) -> str:
        return hashlib.sha256(content.replace("\r\n", "\n").encode("utf-8")).hexdigest()

    def _write_file(self, args: dict) -> dict:
        rel_path = args["path"].replace("\\", "/")
        path = self._safe_path(rel_path)
        frozen = self._frozen(rel_path)
        if frozen:
            return frozen

        # read_file numbers what the model sees and Qwen pastes the gutter
        # back. Strip it as modify_file and replace_file_lines do, or a rewrite
        # arriving as "   1| import re" is refused as "unexpected indent at
        # line 1" and resent unchanged (most of Qwen's write_file refusals).
        # Safe because the guard is "most content lines carry NNN| ", which
        # no real source file satisfies.
        content = args.get("content")
        if isinstance(content, str) and content:
            unnumbered = _strip_line_numbers(content.split("\n"))
            if unnumbered is not None:
                args = {**args, "content": "\n".join(unnumbered)}

        # Rewriting a file the model has never seen this run is a rewrite
        # from memory, which loses scaffold code.
        if os.path.isfile(path) and rel_path.strip() not in self._known_paths:
            return {"error": (
                f"{args['path']} exists and you have not read it this run. Call "
                "read_file first, then modify_file for targeted changes (write_file "
                "only if a full rewrite is really needed)."
            )}

        # Edit-first guardrail (MODIFY runs only): every existing
        # non-trivial file is part of the user's working app — reject a
        # whole-file rewrite until the model has tried at least two
        # targeted modify_file edits on it. The rejection text teaches
        # the model the recovery path, and the >= 2 relaxation keeps the
        # modify-streak reminder ("switch to write_file") consistent.
        if (
            self._modify_guard
            and os.path.isfile(path)
            and self._modify_counts.get(rel_path, 0) < 2
        ):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                existing_lines = f.read().count("\n") + 1
            if existing_lines > 15:
                return {
                    "error": (
                        f"'{rel_path}' already exists ({existing_lines} lines) and "
                        "this is a MODIFY run — edit it in place with modify_file "
                        "(exact old_text → new_text). A full rewrite risks dropping "
                        "the customisations the app already carries. write_file "
                        "unlocks for this file after two modify_file attempts, as "
                        "a last resort."
                    ),
                }

        # Size rule: a whole-file rewrite is cheapest and
        # safest on a SMALL file (<=200 lines), which the model can reproduce
        # faithfully after one read; it is a large file that risks dropping
        # code. So small generated files unlock write_file immediately and
        # large ones keep the targeted-edit requirement until two modify_file
        # attempts have been made.
        if rel_path in self._generator_files and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                existing_lines = f.read().count("\n") + 1
            if existing_lines > 200 and self._modify_counts.get(rel_path, 0) < 2:
                return {
                    "error": (
                        f"'{rel_path}' was created by a BESSER generator and is large "
                        f"({existing_lines} lines) - too large to reproduce faithfully "
                        "from one read. Use modify_file for targeted edits. write_file "
                        "unlocks for this file after two modify_file attempts."
                    ),
                }
            # Rewrite allowed (large file, or repeated modifies already tried) — log it
            logger.info(
                "Allowing rewrite of generated file: %s (%d lines, %d prior modifies)",
                rel_path, existing_lines, self._modify_counts.get(rel_path, 0),
            )

        before = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                before = f.read()
            broke = _new_syntax_error(rel_path, before, args["content"])
            if broke:
                msg, line_no = broke
                return {
                    "error": (
                        f"Refused: this rewrite would make {args['path']} unparseable "
                        f"({msg} at line {line_no}); the file was left unchanged. Fix "
                        "the syntax and resend the full content."
                    ),
                    "rejection_kind": "syntax_error",
                    "syntax_line": line_no,
                    "would_write": "PROPOSED ONLY - NOT APPLIED; these are not current file contents:\n"
                    + _changed_region(before, args["content"]),
                    "current_source": "CURRENT ON-DISK CONTENT - unchanged by this refused rewrite:\n"
                    + _changed_region(args["content"], before),
                }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(args["content"])
        if before != args["content"]:
            self._successful_writes[os.path.normcase(path)] = self._content_digest(args["content"])
        result = {"status": "written", "path": args["path"], "size": len(args["content"])}
        self._known_paths.add(args["path"].replace("\\", "/").strip())
        self._append_write_feedback(result, rel_path, args["content"])
        return result

    def mark_known(self, paths) -> None:
        """Record files whose contents the model was shown (scaffold snapshot)."""
        for p in paths:
            self._known_paths.add(str(p).replace("\\", "/").strip())

    # Three is the figure Cline (consecutiveMistakeCount) and Roo
    # (DEFAULT_CONSECUTIVE_MISTAKE_LIMIT) both settled on.
    _MAX_MODIFY_MISSES = 3

    def consecutive_modify_misses(self, path: str) -> int:
        """Failed modify_file attempts on ``path`` since its last successful edit."""
        key = path.replace("\\", "/").strip()
        for rel_path, misses in self._failed_modifies.items():
            if rel_path.strip() == key:
                return misses
        return 0

    def repeat_rejections(self, path: str) -> int:
        """Highest number of times one already-rejected edit was sent to ``path``."""
        return self._repeat_hits.get(path.replace("\\", "/").strip(), 0)

    def freeze_path(self, path: str, reason: str) -> None:
        """Close ``path`` to modify_file / write_file for the rest of the run."""
        self._frozen_paths[path.replace("\\", "/").strip()] = reason

    def _frozen(self, rel_path: str) -> dict | None:
        reason = self._frozen_paths.get(rel_path.strip())
        if reason is None:
            return None
        return {
            "error": (
                f"{rel_path} is closed for edits for the rest of this run "
                f"({reason}). Continue with other work or finish."
            ),
        }

    # Refusals at one target before the orchestrator's escalation is told,
    # counting redrafts. One above the exact-text threshold (_REPEAT_FORCE_AT
    # = 3): alternating drafts slow the exact-text counter, yet a third
    # genuinely different edit to one range can still land. Advisory - it
    # steers strategy, it refuses nothing.
    _TARGET_REPEAT_AT = 4

    def _note_rejection(self, rel_path: str, old_text: str, new_text: str) -> int:
        """Record a rejected edit; return how often this exact call has now been seen."""
        key = (rel_path.strip(), old_text, new_text)
        seen = self._rejected_edits.get(key, 0) + 1
        self._rejected_edits[key] = seen
        target = (key[0], old_text)
        hits = self._rejected_targets.get(target, 0) + 1
        self._rejected_targets[target] = hits
        reported = max(seen if seen > 1 else 0,
                       hits if hits >= self._TARGET_REPEAT_AT else 0)
        if reported:
            self._repeat_hits[key[0]] = max(self._repeat_hits.get(key[0], 0), reported)
            self.last_repeat = (key[0], reported)
        return seen

    def _clear_rejections(self, rel_path: str) -> None:
        key_path = rel_path.strip()
        for key in [k for k in self._rejected_edits if k[0] == key_path]:
            del self._rejected_edits[key]
        for key in [k for k in self._rejected_targets if k[0] == key_path]:
            del self._rejected_targets[key]
        self._repeat_hits.pop(key_path, None)

    def _modify_file(self, args: dict) -> dict:
        """Targeted search-and-replace, with a flexible apply ladder.

        Match tiers, most literal first (never similarity-scored): exact,
        line-anchored -> typographic normalization -> ``edit_apply`` (uniform-
        indent correction, spurious leading blank line, blank-run count,
        line-number prefix). On a miss the error carries a "did you mean"
        window of the closest real lines so the retry can copy them verbatim.
        An edit that would turn a parseable Python file into an unparseable
        one is refused and the file left untouched. Every rejection is
        fingerprinted; a repeat of a rejected call after three misses is
        refused outright and only a successful edit on the path clears that.
        """
        path = self._safe_path(args["path"])
        if not os.path.isfile(path):
            return self._missing_file_result(args["path"])
        rel_path = args["path"].replace("\\", "/")
        self.last_repeat = None
        # read() normalizes file newlines; normalize quoted text identically.
        old_text = args["old_text"].replace("\r\n", "\n")
        new_text = args["new_text"].replace("\r\n", "\n")
        frozen = self._frozen(rel_path)
        if frozen:
            self._note_rejection(rel_path, old_text, new_text)
            return frozen
        self._modify_counts[rel_path] = self._modify_counts.get(rel_path, 0) + 1
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        replace_all = args.get("replace_all") is True
        receipt_key = (os.path.normcase(path), old_text, new_text, replace_all)
        if not old_text.strip():
            return {
                "error": "old_text is empty or whitespace only. To create a file "
                         "or append to it, use write_file instead.",
            }
        if self._edit_receipts.get(receipt_key) == self._content_digest(content):
            self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
            self._note_rejection(rel_path, old_text, new_text)
            line_no = locate_chunk(content, new_text) if new_text.strip() else None
            result = {
                "status": "already_applied", "replacements": 0,
                "error": (
                    f"This exact edit was successfully applied to {args['path']} and "
                    "the file still matches its recorded post-edit state. "
                    + (f"The replacement is already in the file at line {line_no}. " if line_no else "")
                    + "Do not resend it. "
                    "This confirms the edit only, not completion of the requirement."
                ),
            }
            hint = find_similar_lines(new_text, content) if new_text.strip() else None
            if hint:
                result["did_you_mean"] = "Current file lines for a different edit:\n" + hint
            return result
        # Sticky stop. After three misses a call the executor has ALREADY
        # rejected is refused without running the ladder, and only a
        # successful edit on this path clears it (a refusal must never reset
        # its own counter).
        # A NEW anchor still goes to the ladder below.
        if (
            self.consecutive_modify_misses(rel_path) >= self._MAX_MODIFY_MISSES
            and self._rejected_edits.get((rel_path.strip(), old_text, new_text), 0) > 0
        ):
            self._note_rejection(rel_path, old_text, new_text)
            return {
                "error": (
                    f"modify_file has missed {self._MAX_MODIFY_MISSES} times in a row on "
                    f"{args['path']} and this exact call was already rejected, so it is "
                    "refused. Do not send it again. A DIFFERENT edit on this file is "
                    "still accepted: call read_file on the region and copy old_text "
                    "verbatim from that output."
                ),
            }
        if old_text == new_text:
            # A no-op is a rejection like any other and must reach the counters.
            self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
            seen = self._note_rejection(rel_path, old_text, new_text)
            where = ""
            # locate_chunk, not `in`: the ladder re-indents new_text on a
            # tier-2 apply, and a byte-exact check then misses its own output.
            line_no = locate_chunk(content, new_text) if new_text.strip() else None
            if line_no:
                where = (
                    f" The identical text is already in the file at line {line_no}; "
                    "this is not evidence of an implemented change."
                )
            if seen > 1:
                return {
                    "error": (
                        f"This identical no-op call was already rejected {seen - 1} "
                        f"time(s) on {args['path']} and cannot succeed.{where} Do not "
                        "send it again. Make a real change or mark the task blocked with a reason."
                    ),
                }
            return {
                "error": (
                    "old_text and new_text are identical, so this call changes "
                    f"nothing.{where} Do not resend it. For a different change, send "
                    "old_text = the current lines copied from read_file output and "
                    "new_text = the changed lines."
                ),
            }
        # Refuse to write an abbreviation (e.g. `contact_id:...`) into the
        # file. Gemini CLI rejects this pre-flight for the same reason; the
        # carve-out is theirs too -- a placeholder already present in old_text
        # is being preserved, not introduced.
        # Compare the lines rather than mere presence, so one ellipsis in
        # old_text does not license new ones.
        new_elision = find_elision(new_text)
        if new_elision and new_elision[1].strip() not in elided_lines(old_text):
            line_no, line = new_elision
            self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
            self._note_rejection(rel_path, old_text, new_text)
            return {
                "error": (
                    f"new_text abbreviates the code at line {line_no}: {line.strip()!r}. "
                    "new_text must be the complete literal text to write - an editor "
                    "cannot expand \"...\". Write out every line in full."
                ),
            }

        matched_by = "exact"
        if old_text not in content:
            # Typographic fallback: match with smart quotes/dashes/nbsp
            # normalized on both sides, then substitute the original span
            # so the file's own characters are what gets replaced.
            norm_content = content.translate(_TYPOGRAPHIC_TRANSLATION)
            norm_old = old_text.translate(_TYPOGRAPHIC_TRANSLATION)
            idx = norm_content.find(norm_old)
            if idx >= 0:
                old_text = content[idx:idx + len(old_text)]
                matched_by = "typographic"

        # Only line-anchored hits (or hits after real text: a partial-line
        # edit) count as exact. A hit inside a line's indentation is the quote
        # under-indented - see _anchored_occurrences.
        # Insertion edits retain their anchor inside new_text; reapplying them
        # must not report success and grow the file forever.
        # Exclude only anchors INSIDE completed replacement regions, not other
        # sites that still need this edit. This guards against duplication after
        # resume, but existing content alone must NEVER claim a successful edit.
        completed = replacement_spans(content, new_text)
        starts = [
            i for i in _anchored_occurrences(content, old_text)
            if not any(lo <= i and i + len(old_text) <= hi for lo, hi in completed)
        ]
        indent_shifted = not starts and old_text in content
        occurrences = len(starts)
        if occurrences:
            # A short anchor appearing more than once is genuinely ambiguous:
            # silently editing "the first one" can edit the wrong place. Refuse
            # and ask for context instead.
            if occurrences > 1 and not replace_all:
                self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
                self._note_rejection(rel_path, old_text, new_text)
                return {
                    "error": (
                        f"old_text occurs {occurrences} times in {args['path']} and is "
                        "ambiguous. Include surrounding lines so it matches exactly "
                        "one place, or set replace_all=true only when every occurrence "
                        "must change."
                    ),
                    # Generated routers are N identical stubs; naming the enclosing
                    # def lets the retry pin one without guessing.
                    "occurrences": _occurrence_locations(content, old_text),
                }
            new_content = content
            for i in (reversed(starts) if replace_all else starts[:1]):
                new_content = new_content[:i] + new_text + new_content[i + len(old_text):]
        else:
            try:
                new_content = replace_most_similar_chunk(
                    content, old_text, new_text, completed, require_unique=True,
                )
            except AmbiguousEdit:
                self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
                self._note_rejection(rel_path, old_text, new_text)
                return {"error": "old_text is ambiguous after whitespace normalization. "
                                 "Include surrounding lines to identify one location. "
                                 "replace_all requires exact quoted occurrences."}
            matched_by = "flexible"

        if new_content is None:
            misses = self._failed_modifies.get(rel_path, 0) + 1
            self._failed_modifies[rel_path] = misses
            self._note_rejection(rel_path, old_text, new_text)
            resent = self._last_missed_old_text.get(rel_path) == old_text
            self._last_missed_old_text[rel_path] = old_text
            hint = find_similar_lines(old_text, content)
            # The lines old_text brackets, if it brackets exactly one region.
            # A LOCATOR: nothing is applied, the model still authors new_text.
            span = locate_anchored_span(content, old_text)
            unread = (
                f"You have not read {args['path']} this run, so old_text cannot be a "
                "copy of it: call read_file (offset/limit for a region) and quote from "
                "that. "
                if rel_path.strip() not in self._known_paths else ""
            )
            # An elided quote can never match, and "check your whitespace" sends
            # the model to re-read and re-elide in a loop. Name the real cause
            # first.
            elision = find_elision(old_text)
            # A retained anchor inside matching replacement content warrants a
            # safe refusal, not a success claim. Multi-line incidental matches
            # are no stronger evidence than a single `pass` or `return True`.
            retained_anchor = any(
                locate_chunk(content[lo:hi], old_text)
                for lo, hi in completed
            )
            if elision:
                line_no, line = elision
                err: dict = {
                    "error": (
                        f"old_text abbreviates the file at line {line_no}: "
                        f"{line.strip()!r}. old_text must be the exact literal text, "
                        "never a shortened quote - call read_file on that region and "
                        "copy the full lines verbatim."
                    ),
                }
            elif retained_anchor:
                err = {
                    "status": "possible_replay",
                    "replacements": 0,
                    "error": (
                        f"Refused possible duplicate insertion in {args['path']}: "
                        "the matching anchor is already inside replacement-shaped content. "
                        "No successful-edit receipt proves this request was applied. "
                        "The file was not changed; do not mark the task done on this basis. "
                        "Read the region and use a distinct surrounding anchor for a different change."
                    ),
                }
            elif resent:
                err = {
                    "error": (
                        f"This exact old_text was already rejected on {args['path']} and "
                        "the file still does not contain it, so resending it cannot "
                        "succeed. Do not send it again. " + _ANCHOR_ADVICE
                    ),
                }
            elif indent_shifted:
                raw = content.find(old_text)
                quoted = len(old_text) - len(old_text.lstrip(" \t"))
                in_file = raw - (content.rfind("\n", 0, raw) + 1) + quoted
                err = {
                    "error": (
                        f"old_text's first line is indented differently from "
                        f"{args['path']}: the file's line has {in_file} leading "
                        f"spaces, the quote has {quoted}. Copy the line's "
                        "indentation exactly from read_file output."
                    ),
                }
            elif not hint and not span:
                # No close match, including when the model quotes a rejected
                # draft as though it was applied. Re-read CURRENT code rather
                # than claiming that re-reading cannot help and then asking
                # for exactly that in the recovery advice below.
                err = {
                    "error": (
                        unread + f"old_text not found in {args['path']}, and no region "
                        "of the current source closely resembles it. The file was not changed. "
                        "Read the target region and quote existing lines verbatim, "
                        "not a proposed edit that was refused. "
                        + _missing_definition(old_text, content) + _ANCHOR_ADVICE
                    ),
                }
            else:
                err = {
                    "error": unread + f"old_text not found in {args['path']}. "
                             f"File has {content.count(chr(10))+1} lines, {len(content)} chars. "
                             f"Make sure old_text matches exactly including whitespace/indentation.",
                }
            # "old_text not found" is true but useless when the quote IS the
            # file's line with every backslash written twice: re-reading the
            # region shows the model text it believes it already copied. Name
            # the actual defect first. The ladder deliberately will not apply
            # this one - see describe_escape_mismatch.
            escaped = describe_escape_mismatch(content, old_text)
            if escaped:
                err["error"] = escaped + " " + err["error"]

            # A bracketed span replaces the recovery detour (miss -> miss ->
            # read_file -> guess a range) with one pre-filled replace_file_lines.
            # Both documented range-edit failure modes are line-number selection
            # errors, so the numbers come from here, not from the model.
            if (span and err.get("status") != "possible_replay"
                    and self._range_edit_failures.get(rel_path, 0) < self._RANGE_EDIT_GIVE_UP):
                numbered = chr(10).join(
                    f"{n:>4}| {line}"
                    for n, line in enumerate(content.split(chr(10))[span[0] - 1:span[1]], span[0])
                )
                if len(numbered) <= MAX_FILE_READ:
                    err["located_range"] = {
                        "path": args["path"],
                        "read_id": self._mint_read_view(path, content, span[0], span[1]),
                        "start_line": span[0], "end_line": span[1],
                        "lines": numbered,
                        "instruction": (
                            "Your old_text brackets exactly these lines. Call "
                            "replace_file_lines with this read_id and these start_line/"
                            "end_line, and new_text = their complete replacement. Do not "
                            "re-quote old_text and do not paste the 'NNN| ' prefixes."
                        ),
                    }
            if hint and "located_range" not in err:
                err["did_you_mean"] = (
                    "Closest lines actually in the file - copy old_text verbatim "
                    "from here:" + chr(10) + hint
                )
            if misses >= 2:
                err["advice"] = (
                    f"This is failed attempt #{misses} on {args['path']}. Call "
                    "read_file on that region and copy old_text out of the output "
                    "instead of retyping it."
                )
            if misses > self._MAX_MODIFY_MISSES:
                err["error"] = (
                    f"modify_file has missed {misses} times in a row on {args['path']}; "
                    "this attempt is refused as well. Call read_file on the region and "
                    "copy old_text verbatim from that output, or move on. "
                    + err["error"]
                )
            return err

        if new_content == content:
            self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
            self._note_rejection(rel_path, old_text, new_text)
            return {
                "status": "no_change", "replacements": 0,
                "error": "This edit makes no change after normalization. Do not resend it; "
                         "this is not evidence of completion. Read the current region for a different change.",
            }

        # A valid file must never leave this tool unparseable.
        broke = _new_syntax_error(rel_path, content, new_content)
        if broke:
            msg, line_no = broke
            self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
            self._note_rejection(rel_path, old_text, new_text)
            return {
                "error": (
                    f"Refused: this edit would make {args['path']} unparseable "
                    f"({msg} at line {line_no}) and the file was left unchanged. "
                    "Check the indentation of new_text against the surrounding code "
                    "and that every block you open is closed."
                ),
                "rejection_kind": "syntax_error",
                "syntax_line": line_no,
                "would_write": "PROPOSED ONLY - NOT APPLIED; these are not current file contents:\n"
                + _changed_region(content, new_content),
                "current_source": "CURRENT ON-DISK CONTENT - unchanged by this refused edit:\n"
                + _changed_region(new_content, content),
            }

        unimportable = _breaks_module_import(path, content, new_content)
        if unimportable:
            self._failed_modifies[rel_path] = self._failed_modifies.get(rel_path, 0) + 1
            self._note_rejection(rel_path, old_text, new_text)
            return {
                "error": (
                    f"Refused: this edit leaves {args['path']} parseable but no longer "
                    f"importable ({unimportable}). Every router star-imports it, so this "
                    "would take the whole application down. The file was left unchanged. "
                    + _IMPORT_BREAK_ADVICE
                ),
                "rejection_kind": "breaks_import",
                "would_write": "PROPOSED ONLY - NOT APPLIED:\n"
                + _changed_region(content, new_content),
            }

        # newline="\n": a text-mode write translates "\n" to os.linesep, which
        # on a Windows host re-encodes the whole file as CRLF - and the React
        # generator copies some templates verbatim out of a CRLF checkout.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_content)
        digest = self._content_digest(new_content)
        self._edit_receipts[receipt_key] = digest
        self._successful_writes[os.path.normcase(path)] = digest
        self._failed_modifies.pop(rel_path, None)
        self._last_missed_old_text.pop(rel_path, None)
        self._clear_rejections(rel_path)
        self._known_paths.add(rel_path.strip())
        result = {
            "status": "modified",
            "path": args["path"],
            "replacements": occurrences if replace_all and occurrences else 1,
        }
        # What is on disk now, numbered: the model quotes from this next time
        # instead of from its memory of the file.
        result["snippet"] = _changed_region(content, new_content)
        if matched_by != "exact":
            # Surfaced for telemetry and as a nudge: old_text did not match
            # byte-for-byte, so the model's copy of the file is drifting.
            result["matched_by"] = matched_by
        self._append_write_feedback(result, rel_path, new_content)
        return result

    def _scaffold_edit_route(self, rel_path: str, path: str) -> str:
        """The write route ``_write_file`` actually permits for a generator file.

        A rewrite of a generator file over 200 lines is itself refused until
        two modify_file edits have been tried, so blanket "or write_file"
        advice in another refusal walks the model into a second refusal.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                existing_lines = fh.read().count("\n") + 1
        except OSError:
            existing_lines = 0
        if existing_lines > 200 and self._modify_counts.get(rel_path, 0) < 2:
            return (
                "Edit it in place with modify_file; write_file on this file "
                "unlocks only after two modify_file attempts."
            )
        return (
            "Edit it in place with modify_file, or read it in full and "
            "write_file it back."
        )

    def _delete_file(self, args: dict) -> dict:
        """Delete a regular file from the workspace.

        Refuses to delete directories — the LLM should never need to
        recursively wipe a tree, and an accidental glob expansion at the
        prompt level would be catastrophic. Path traversal is blocked by
        ``_safe_path`` (same containment check used by every other file
        tool).
        """
        path = self._safe_path(args["path"])
        if not os.path.exists(path):
            return {"error": f"File not found: {args['path']}"}
        if os.path.isdir(path):
            return {
                "error": (
                    f"Refused to delete: {args['path']} is a directory. "
                    "delete_file only removes regular files."
                ),
            }
        # Scaffold protection (weak / free-tier models only): the deterministic
        # Phase-1 scaffold is the app's foundation and must not be torn down and
        # rebuilt in another framework. Files the model created this run are not
        # tracked here, so it can still delete its own junk.
        rel_path = args["path"].replace("\\", "/")
        if self._protect_scaffold and rel_path in self._generator_files:
            return {
                "error": (
                    f"Refused to delete: '{args['path']}' is part of the generated "
                    "scaffold — the app's foundation. Do NOT remove or replace it. "
                    f"{self._scaffold_edit_route(rel_path, path)} Keep the existing "
                    "framework and build on top of what is already generated."
                ),
            }
        try:
            os.remove(path)
        except OSError as exc:
            return {"error": f"Failed to delete {args['path']}: {exc}"}
        return {"status": "deleted", "path": args["path"]}

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
        for root, dirs, filenames in os.walk(self.workspace):
            # Vendored and generated trees are not the model's code, and the
            # 50-match cap below is global: once the frontend's dependencies
            # are installed, a search that walks node_modules fills its whole
            # budget with third-party source and reports "truncated" without
            # ever reaching the app. Prune before the ordering below, which
            # must still run so the spill logs stay last.
            dirs[:] = [name for name in dirs if name not in _UNSEARCHED_DIRS]
            # Spilled command logs are searchable, but visited last: a build
            # log must not eat the 50-match budget before the source files.
            dirs.sort(key=lambda name: name == COMMAND_OUTPUT_DIR)
            for fname in filenames:
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, self.workspace).replace("\\", "/")
                # Skip binary files and internal files. Spilled command logs
                # are the exception: run_command hands the model their path
                # precisely so it can grep the errors truncation dropped.
                if rel_path.startswith(".besser_") and not rel_path.startswith(
                    COMMAND_OUTPUT_DIR + "/"
                ):
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

    @staticmethod
    def _cap_spill(stream: str) -> str:
        """Bound one spilled stream. 2 MB is ~130x the context cap."""
        if len(stream) <= MAX_SPILL_SIZE:
            return stream
        dropped = len(stream) - MAX_SPILL_SIZE
        return stream[:MAX_SPILL_SIZE] + f"\n\n... [{dropped} chars dropped from the spill]"

    def _spill_command_output(
        self, command: str, stdout: str, stderr: str,
    ) -> str | None:
        """Write the untruncated command output into the workspace.

        Returns the workspace-relative path, or None when nothing was cut.
        ``_truncate`` keeps head 20% + tail 60% of stderr, and a failing
        ``tsc`` / ``npm run build`` reports its errors in the discarded
        middle - the one part Phase 3 has to act on. The spill keeps them
        reachable through search_in_files / read_file.
        """
        limit = MAX_OUTPUT_SIZE // 2
        if len(stdout) <= limit and len(stderr) <= limit:
            return None
        self._command_log_count += 1
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", command).strip("-").lower()[:40] or "command"
        rel = f"{COMMAND_OUTPUT_DIR}/{self._command_log_count:03d}-{slug}.log"
        try:
            full = self._safe_path(rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8", errors="replace", newline="") as fh:
                fh.write(
                    f"$ {command}\n\n===== stdout =====\n{self._cap_spill(stdout)}\n"
                    f"===== stderr =====\n{self._cap_spill(stderr)}\n"
                )
        except (OSError, ValueError):
            logger.debug("Could not spill command output for %s", command, exc_info=True)
            return None
        return rel

    def _run_command(self, args: dict) -> dict:
        """
        Run a shell command in the workspace.

        Security:
        - Working directory locked to workspace (or subdirectory)
        - The command itself runs in a namespace sandbox that binds only this
          run's directory and gives it its own PID 1 (see execution.sandbox).
          The path lock covers tool arguments; the sandbox covers the command
          string (e.g. `cd ../<other_run>`).
        - Timeout enforced
        - Output truncated to prevent context blow-up, full log spilled to
          the workspace when it is
        """
        command = args["command"]
        working_dir = self._safe_cwd(args.get("working_dir", "."))

        # Refuse obviously-destructive or exfil commands before we hand
        # them to the shell. Returned as a normal tool error so the LLM
        # can re-plan.
        deny_reason = _check_command_safety(command)
        if deny_reason:
            logger.warning("run_command denied: %s", deny_reason)
            return {
                "error": deny_reason,
                "command": command,
                "exit_code": None,
                "success": False,
            }

        # Fail closed: a command that cannot be confined is not run. Falling
        # back to an unconfined shell would silently restore both holes the
        # sandbox exists to close.
        try:
            sandbox = sandboxed_command(
                command, workspace=self.workspace, cwd=working_dir,
            )
        except SandboxUnavailable as exc:
            logger.error("run_command refused, sandbox unavailable: %s", exc)
            return {
                "error": (
                    "Refused: the shell sandbox could not be started, so this "
                    f"command was not run. {exc}."
                ),
                "command": command,
                "exit_code": None,
                "success": False,
            }

        logger.info(
            "Running command: %s (in %s, sandbox=%s)",
            command, working_dir, sandbox.mode,
        )

        try:
            result = subprocess.run(
                sandbox.argv,
                shell=sandbox.use_shell,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
                env=_safe_subprocess_env(),
            )

            startup_error = sandbox.startup_error(result.returncode, result.stderr)
            if startup_error:
                logger.error("run_command sandbox failed to start: %s", startup_error)
                return {
                    "error": (
                        "Refused: the shell sandbox failed to start, so this "
                        f"command was not run. {startup_error}."
                    ),
                    "command": command,
                    "exit_code": None,
                    "success": False,
                }

            raw_stdout = result.stdout or ""
            raw_stderr = result.stderr or ""
            stdout = self._truncate(raw_stdout, MAX_OUTPUT_SIZE // 2)
            # For stderr (errors), keep the tail where the actual error message is
            stderr = self._truncate(raw_stderr, MAX_OUTPUT_SIZE // 2, keep_tail=True)

            # If the command failed because the runtime isn't installed in
            # this container (e.g. `ruby -c file.rb` when ruby is absent),
            # treat it as a soft skip rather than a real error. Otherwise
            # the LLM dutifully reports "Ruby is not installed in the
            # execution environment" in its user-facing summary, which is
            # noise the user can't act on.
            if result.returncode != 0 and _looks_like_command_not_found(stderr):
                logger.info(
                    "run_command: runtime not available, treating as soft skip: %s",
                    command,
                )
                return {
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "success": True,
                    "skipped": True,
                    "skip_reason": (
                        "Runtime for this command is not installed in the "
                        "execution environment; validation skipped."
                    ),
                }

            payload = {
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "success": result.returncode == 0,
            }
            spilled = self._spill_command_output(command, raw_stdout, raw_stderr)
            if spilled:
                payload["full_output_path"] = spilled
                payload["full_output_note"] = (
                    "stdout/stderr above are truncated and the middle is missing. "
                    f"The complete output is in {spilled}: use search_in_files to "
                    "locate the errors, then read_file with offset/limit. That file "
                    "is run-internal and is not part of the generated project."
                )
            return payload

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
        err = self._require_domain_model("validate_model")
        if err:
            return err
        try:
            result = self.domain_model.validate()
            return {"validation": result}
        except Exception as e:
            return {"error": f"Validation failed: {e}"}

    def _validate_app(self, args: dict) -> dict:
        if self.app_validator is None:
            return {"error": "Application verification is unavailable in this executor; do not claim the app was verified."}
        return self.app_validator()

    def _test_api(self, args: dict) -> dict:
        if self.api_tester is None:
            return {"error": "API workflow verification is unavailable; do not claim workflows were tested."}
        return self.api_tester(args)

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
    # Model query tools
    # ------------------------------------------------------------------

    def _find_class_by_name(self, name: str):
        """Case-sensitive class lookup in the domain model.

        Returns the ``Class`` or ``None`` if not found. Used by every
        model-query tool handler below.
        """
        if not self.domain_model or not name:
            return None
        for cls in self.domain_model.get_classes():
            if cls.name == name:
                return cls
        return None

    def _query_class(self, args: dict) -> dict:
        """Return a richer view of one class than the prompt summary."""
        name = args.get("name", "")
        cls = self._find_class_by_name(name)
        if cls is None:
            return {"error": f"Class {name!r} not found in the domain model"}

        from besser.spec_driven_agent.model_serializer import (
            _attribute_entry,
            _method_entry,
            _type_name,
        )

        attrs = [_attribute_entry(a) for a in sorted(cls.attributes, key=lambda a: a.name)]
        methods = [_method_entry(m) for m in sorted(cls.methods, key=lambda m: m.name)]

        try:
            parents = sorted([p.name for p in cls.parents()])
        except Exception:
            parents = []

        try:
            assoc_ends = []
            for end in cls.all_association_ends():
                assoc_ends.append({
                    "role": end.name,
                    "type": _type_name(end.type),
                    "navigable": getattr(end, "is_navigable", True),
                })
        except Exception:
            assoc_ends = []

        return {
            "name": cls.name,
            "is_abstract": bool(getattr(cls, "is_abstract", False)),
            "attributes": attrs,
            "methods": methods,
            "parents": parents,
            "association_ends": assoc_ends,
        }

    def _list_classes_with(self, args: dict) -> dict:
        """Filter the domain model's classes by a simple predicate."""
        predicate = (args.get("predicate") or "").strip()
        if not predicate:
            return {"error": "predicate is required"}
        if not self.domain_model:
            return {"error": "No domain model loaded"}

        classes = list(self.domain_model.get_classes())

        # Predicate parsing — ``prefix:value`` or single keyword.
        if ":" in predicate:
            key, _, value = predicate.partition(":")
            key = key.strip()
            value = value.strip()
        else:
            key, value = predicate, None

        def _has_constraint(cls) -> bool:
            try:
                return any(
                    c for c in self.domain_model.constraints
                    if getattr(c, "context", None) is cls
                )
            except Exception:
                return False

        matchers = {
            "is_abstract": lambda c: bool(getattr(c, "is_abstract", False)),
            "is_root": lambda c: not list(c.parents()),
            "has_constraint": _has_constraint,
            "has_attribute": lambda c: any(a.name == value for a in c.attributes),
            "has_method": lambda c: any(m.name == value for m in c.methods),
            "extends": lambda c: any(p.name == value for p in c.all_parents()),
        }
        if key not in matchers:
            return {
                "error": (
                    f"Unknown predicate {predicate!r}. Supported: "
                    f"{', '.join(sorted(matchers))}"
                )
            }
        if key in {"has_attribute", "has_method", "extends"} and not value:
            return {"error": f"predicate '{key}' requires a value (e.g. '{key}:name')"}

        matched = sorted([c.name for c in classes if matchers[key](c)])
        return {"predicate": predicate, "matches": matched, "count": len(matched)}

    def _get_constraints_for(self, args: dict) -> dict:
        """Return OCL / constraint expressions scoped to a given class."""
        name = args.get("class_name", "")
        cls = self._find_class_by_name(name)
        if cls is None:
            return {"error": f"Class {name!r} not found in the domain model"}
        if not self.domain_model:
            return {"error": "No domain model loaded"}

        out = []
        for c in self.domain_model.constraints:
            context = getattr(c, "context", None)
            if context is not cls:
                continue
            out.append({
                "name": getattr(c, "name", ""),
                "expression": getattr(c, "expression", ""),
                "language": getattr(c, "language", None),
            })
        return {"class": name, "constraints": out, "count": len(out)}

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
        "generate_qiskit": _gen_qiskit,
        "generate_supabase": _gen_supabase,
        "generate_json_object": _gen_json_object,
        "generate_baf": _gen_baf,
        "generate_bpmn": _gen_bpmn,
        "generate_pytorch": _gen_pytorch,
        "generate_tensorflow": _gen_tensorflow,
        # Files
        "list_files": _list_files,
        "read_file": _read_file,
        "write_file": _write_file,
        "replace_file_lines": _replace_file_lines,
        "modify_file": _modify_file,
        "search_in_files": _search_in_files,
        "delete_file": _delete_file,
        # Execution
        "run_command": _run_command,
        "install_dependencies": _install_dependencies,
        # Model queries
        "query_class": _query_class,
        "list_classes_with": _list_classes_with,
        "get_constraints_for": _get_constraints_for,
        # Validation
        "validate_model": _validate_model,
        "validate_app": _validate_app,
        "test_api": _test_api,
        "check_syntax": _check_syntax,
        # Work checklist
        "task_list": _task_list,
    }
