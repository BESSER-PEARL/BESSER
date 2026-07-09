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
  - Validate generated output (syntax, Dockerfile refs, imports)
  - Give the LLM a few turns to fix any issues
  - Snapshot/rollback if fixes make things worse
"""

import ast as _ast
import json
import logging
import os
import re as _re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Literal

from besser.generators.llm.compaction import (
    COMPACT_TOKEN_THRESHOLD,
    COMPACT_PRESERVE_RECENT,
    _estimate_tokens,
    maybe_compact,
    _summarize_messages,
)
from besser.generators.llm.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
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
from besser.generators.llm.llm_client import ClaudeLLMClient, FROM_SCRATCH_MAX_TOKENS
from besser.generators.llm.prompt_builder import (
    build_scaffold_snapshot,
    build_system_prompt,
    build_inventory,
)
from besser.generators.llm.stack_metadata import (
    detect_stack,
    pre_generate_metadata,
    stack_label,
)
from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.tools import get_all_tools, get_all_tools_including_generators
from besser.generators.llm.tracing import (
    EVENT_CHECKPOINT,
    EVENT_COMPACTION,
    EVENT_COST_UPDATE,
    EVENT_ERROR,
    EVENT_PHASE_ENTER,
    EVENT_PHASE_EXIT,
    EVENT_ROLLBACK,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_SNAPSHOT,
    EVENT_TOOL_CALL,
    EVENT_TURN_START,
    EVENT_VALIDATION_ISSUE,
    TRACE_FILENAME,
    NullTraceWriter,
    TraceWriter,
)

logger = logging.getLogger(__name__)

# Snapshot directory name (inside output_dir)
_SNAPSHOT_DIR = ".besser_snapshot"

# Dependency / build directories excluded from the recipe's output_files
# manifest (mirrors the web runner's zip exclusions).
_RECIPE_EXCLUDED_DIRS = {
    "target", "node_modules", "__pycache__", ".git", "dist", "build",
    ".next", ".gradle", "venv", ".venv", _SNAPSHOT_DIR,
}


@dataclass(frozen=True)
class ValidationIssue:
    """A single Phase 3 validator finding, tagged by severity.

    severity:
      - ``blocker``: prevents the app from running (syntax errors,
        missing required files, dependency conflicts).
      - ``warning``: probably broken at runtime but not certain
        (Dockerfile semantic issues, tsc type errors, missing imports).
      - ``style``: cosmetic / preference (unused imports, line length,
        formatting). Never blocks a release.

    The auto-fix loop (when enabled) only consumes ``blocker`` issues.
    Everything else is reported in the recipe and logs but left alone.
    """

    severity: Literal["blocker", "warning", "style"]
    message: str

    def __str__(self) -> str:  # for legacy log formatting
        return self.message


# ---- Phase 3 issue classification --------------------------------------

# Ruff rule codes that are pure style: dead imports, unused vars, line length.
# Anything else we treat as a warning (could be a real bug).
_RUFF_STYLE_CODES = frozenset({
    "F401", "F811", "F841",      # unused / redefinition
    "E501",                       # line too long
    "W291", "W292", "W293", "W391",  # whitespace
    "E302", "E303", "E305", "E261", "E262", "E266",  # blank lines / comments
    "I001",                       # import order
})
_RUFF_LINE_RE = _re.compile(r"\b([EWFCNI]\d{2,4})\b")


def _classify_issue(message: str) -> ValidationIssue:
    """Map a raw issue string to a ``ValidationIssue`` with severity.

    Heuristics keyed on the prefix our validators produce so the
    classification is stable as new validators are added.
    """
    text = message.strip()
    lower = text.lower()

    # Hard blockers — these prevent the generated app from running.
    if lower.startswith("syntax error in"):
        return ValidationIssue("blocker", text)
    if lower.startswith("dependency conflict in"):
        return ValidationIssue("blocker", text)
    if "but it doesn't exist" in lower:
        # e.g. "Dockerfile references requirements.txt but it doesn't exist"
        return ValidationIssue("blocker", text)

    # Ruff: classify by rule code.
    if text.startswith("ruff:"):
        match = _RUFF_LINE_RE.search(text)
        if match and match.group(1) in _RUFF_STYLE_CODES:
            return ValidationIssue("style", text)
        return ValidationIssue("warning", text)

    # Per-project toolchain failures (tsc / cargo / kotlinc) are
    # blockers: they mean the artifact does not compile on its own
    # toolchain, which is the per-project compile-pass criterion the
    # bench checks. The Phase 3 fix loop must drive these to zero.
    # ``tsc info`` / ``cargo info`` etc. (informational lines our
    # collectors emit when the binary is missing or the project has
    # no errors) are NOT prefixed this way — only real error lines
    # land here.
    if (text.startswith("tsc [")
            or text.startswith("cargo [")
            or text.startswith("kotlinc [")):
        return ValidationIssue("blocker", text)

    # Legacy ``tsc `` (no bracket) prefix — kept as a soft warning so
    # any caller that constructs strings outside the collector path
    # doesn't trip the fix loop unexpectedly.
    if text.startswith("tsc "):
        return ValidationIssue("warning", text)

    # Unknown shape → conservative default: warning.
    return ValidationIssue("warning", text)

# Tools that are read-only and shouldn't count for loop detection
_READONLY_TOOLS = frozenset({"read_file", "list_files", "search_in_files", "check_syntax"})

# Maximum workers for parallel tool execution
_MAX_PARALLEL_WORKERS = 4

# Phase 3 toolchain-fix outer cap. The LLM gets up to this many
# (collect → fix-loop → re-collect) iterations before we accept the
# remaining toolchain errors and move on. Each iteration is the
# existing 5-turn LLM fix loop, so the worst-case extra cost is
# 3 × 5 = 15 LLM turns. Kept low to bound the bill — if tsc / cargo
# / kotlinc errors don't converge in 3 rounds, more rounds aren't
# going to help.
_MAX_TOOLCHAIN_FIX_ITERATIONS = 3

# Sub-generator tools that a chosen PRIMARY generator already bundles, so
# offering them to the Phase-2 agent only lets it scatter redundant top-level
# _gen_dir folders (e.g. a FastAPI backend already contains SQLAlchemy models,
# Pydantic schemas and REST routers inside backend/ — the standalone
# generate_pydantic / generate_sqlalchemy / generate_rest_api tools would emit
# duplicate pydantic/ sqlalchemy/ rest_api/ dirs next to it). The Phase-1
# SELECTOR is already told "generate_fastapi_backend includes SQLAlchemy +
# Pydantic — don't pick those separately", but that guidance never reached the
# Phase-2 agent; this removes the tools so it CANNOT call them.
_REDUNDANT_GENERATOR_TOOLS_BY_PRIMARY = {
    "generate_fastapi_backend": {
        "generate_pydantic", "generate_sqlalchemy", "generate_rest_api",
    },
    "generate_django": {"generate_pydantic", "generate_sqlalchemy"},
    "generate_web_app": {
        "generate_pydantic", "generate_sqlalchemy", "generate_rest_api",
        "generate_fastapi_backend", "generate_react",
    },
}

# Adaptive budget ceiling for FROM-SCRATCH runs (Phase 1 found no
# deterministic generator to run). These are deploy-tunable, so they live
# in the web backend's constants module next to the default cap they
# override -- see constants.py for the full rationale. The import is
# best-effort and deliberately NOT at call time: ``besser.generators.llm``
# is also used as a standalone library (``LLMGenerator``, no web backend
# installed/importable), and this module must keep working there. When
# the web backend isn't importable we fall back to the same defaults
# constants.py itself ships, so behaviour is identical either way -- ops
# just lose the env-var override unless the web backend is present.
try:
    from besser.utilities.web_modeling_editor.backend.constants.constants import (
        LLM_FROM_SCRATCH_MAX_COST_USD_HARD_CAP as _FROM_SCRATCH_MAX_COST_USD,
        LLM_FROM_SCRATCH_MAX_RUNTIME_SECONDS_HARD_CAP as _FROM_SCRATCH_MAX_RUNTIME_SECONDS,
    )
except Exception:  # pragma: no cover - defensive, web backend optional
    _FROM_SCRATCH_MAX_COST_USD = 5.0
    _FROM_SCRATCH_MAX_RUNTIME_SECONDS = 1800


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

    MAX_TURNS = 80

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
        on_progress: Callable[[int, str, str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_phase_details: Callable[[str, str], None] | None = None,
        use_streaming: bool = True,
        object_model=None,
        state_machines=None,
        quantum_circuit=None,
        auto_fix_issues: bool = False,
        should_continue: Callable[[], bool] | None = None,
        primary_kind: str | None = None,
        run_id: str = "",
        enable_tracing: bool = True,
        enable_checkpointing: bool = True,
        enable_toolchain_validation: bool = True,
        target_generator: str | None = None,
        source_project_export: dict | None = None,
    ):
        self.client = llm_client
        self.domain_model = domain_model
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
        self.object_model = object_model
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
                "agent, state_machine, object, or quantum)"
            )
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="besser_llm_")
        self.max_turns = max_turns or self.MAX_TURNS
        self.max_cost_usd = max_cost_usd
        self.max_runtime_seconds = max_runtime_seconds
        self.on_progress = on_progress
        self.on_text = on_text
        self.on_phase_details = on_phase_details
        self.use_streaming = use_streaming
        self.executor = ToolExecutor(
            workspace=self.output_dir,
            domain_model=domain_model,
            gui_model=gui_model,
            agent_model=agent_model,
            agent_config=agent_config,
            quantum_circuit=quantum_circuit,
        )
        # Give the LLM tools scoped to the models it actually has. Tools
        # that need a domain model (pydantic/sqlalchemy/django/react/…)
        # are hidden when there isn't one so the LLM doesn't waste turns
        # calling generators that will just error. See tools.get_tools_for.
        from besser.generators.llm.tools import get_tools_for
        self.tools = get_tools_for(
            has_domain_model=self.domain_model is not None,
            has_gui_model=self.gui_model is not None,
            has_agent_model=self.agent_model is not None,
            has_state_machines=bool(self.state_machines),
            has_quantum_circuit=self.quantum_circuit is not None,
        )
        # Phase 3 auto-fix policy. False = report-only (industry default
        # for static analysers — fix on request, never blindly).
        self.auto_fix_issues = auto_fix_issues
        # Phase 3 toolchain checks (tsc / cargo / kotlinc) compile real
        # projects and can add minutes of wall-clock per run. The web
        # runner disables them per deploy (BESSER_LLM_ENABLE_TOOLCHAIN_
        # VALIDATION); library users keep the default. The cheap checks
        # (ast.parse, Dockerfile refs, ruff, pip dry-run) always run.
        self.enable_toolchain_validation = enable_toolchain_validation
        # Binding Phase-1 generator choice (e.g. from a user-approved
        # preview plan). When set, _select_generator returns it directly —
        # no selection LLM call, no keyword fallback.
        self.target_generator = target_generator
        # Cooperative cancellation hook. The orchestrator polls this at
        # the top of each Phase 2 turn. Returning False causes the loop
        # to exit cleanly — used by the SSE runner to honour
        # ``POST /cancel-smart-gen/{run_id}`` without killing the thread.
        self._should_continue = should_continue
        self.tool_calls_log: list[dict] = []
        self.total_turns = 0
        self._recent_tool_calls: list[str] = []
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
        # Set by ``_apply_adaptive_budget`` when Phase 1 found no
        # deterministic generator to run: the cost / runtime ceiling and
        # the client's output-token limit were raised for this run. Kept
        # for observability (surfaced in the saved recipe).
        self._adaptive_budget_applied: bool = False
        self._start_time: float | None = None
        # Stored as ValidationIssue objects so the recipe captures severity.
        # Cast to strings via `[str(i) for i in self._validation_issues]`
        # when emitting JSON.
        self._validation_issues: list[ValidationIssue] = []
        self._previous_errors: list[str] = []  # track errors to avoid re-attempting

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
        )
        self._resume_from_turn: int = 0
        self._resume_messages: list[dict] | None = None
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

    _LOOP_THRESHOLD = 4
    # Tighter, per-file threshold for the modify_file streak guard.
    # Long sequences of small modify_file edits on the same path are
    # the typical "death by a thousand cuts" failure mode: the LLM
    # keeps making forward progress, so the legacy uniform-tool
    # ``_is_stuck`` warning (a soft note appended to a tool_result)
    # doesn't change behaviour. At 3 consecutive single-file edits we
    # inject a high-salience reminder before the NEXT LLM call.
    _PER_FILE_MODIFY_THRESHOLD = 3

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
        )
        self._trace.write(
            EVENT_RUN_START,
            instructions=instructions[:500],
            max_cost_usd=self.max_cost_usd,
            max_runtime_seconds=self.max_runtime_seconds,
            max_turns=self.max_turns,
        )

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

        # Only drop the checkpoint when Phase 2 ended cleanly (LLM said
        # "done"). If Phase 2 broke out due to an API error, cost cap,
        # timeout, or cancellation, the checkpoint stays on disk so the
        # user can resume via POST /besser_api/resume-smart-gen/{run_id}.
        if self._phase2_exited_cleanly:
            delete_checkpoint(self.output_dir)

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
        continues Phase 2 from the saved turn. Phase 1 is skipped
        entirely — its outputs are already on disk. Phase 3 still runs
        after Phase 2 completes so validation/fix logic gets a chance
        to clean up anything left half-done at the crash point.

        Raises
        ------
        FileNotFoundError
            No checkpoint in the output dir — nothing to resume.
        ValueError
            The checkpoint's project fingerprint disagrees with the
            current project/instructions. We refuse rather than silently
            resuming against a different spec.
        """
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
        self._resume_from_turn = checkpoint.turn
        self._resume_messages = checkpoint.messages
        self._inventory = checkpoint.inventory
        self._generator_used = checkpoint.generator_used
        self._compaction_count = checkpoint.compaction_count
        self.tool_calls_log = list(checkpoint.tool_calls_log)
        self._validation_issues = [
            ValidationIssue(i.get("severity", "warning"), i.get("message", ""))
            for i in checkpoint.validation_issues
        ]
        self._project_fingerprint = checkpoint.project_fingerprint
        self._trace.write(
            EVENT_RUN_START,
            resumed=True,
            resume_from_turn=checkpoint.turn,
            saved_at=checkpoint.saved_at,
        )

        # -- Adaptive budget: same rule as a fresh run (``_generator_used``
        # was just restored from the checkpoint above). Resuming is the
        # common path for a run that previously broke out on a cost_cap /
        # timeout / max_tokens truncation, so this matters here too.
        self._apply_adaptive_budget()

        # -- Phase 2 (continued) ------------------------------------------
        self._trace.write(EVENT_PHASE_ENTER, phase="phase2_resume")
        self._run_phase2(instructions, extra_issues=[])
        self._trace.write(EVENT_PHASE_EXIT, phase="phase2_resume", turns=self.total_turns)

        self._create_snapshot()
        self._trace.write(EVENT_SNAPSHOT, before_phase="phase3_resume")
        self._trace.write(EVENT_PHASE_ENTER, phase="phase3")
        self._run_phase3_validation()
        self._trace.write(EVENT_PHASE_EXIT, phase="phase3")

        elapsed = time.monotonic() - self._start_time
        self._save_recipe(instructions, elapsed)
        self._remove_snapshot()
        delete_checkpoint(self.output_dir)
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

        self._start_time = time.monotonic()
        self._modify_mode = True

        # Re-hydrate generator-file tags + the seed's generator name from
        # the copied recipe BEFORE building the inventory (which needs a
        # generator name for its framing line).
        self._seed_generator_files_from_recipe()
        # Frame the run as editing an existing project. When the seed came
        # from a deterministic generator we adopt that name (accurate — the
        # base + prior LLM edits descend from it) so gap analysis, the
        # scaffold-snapshot inlining, and the saved recipe all line up.
        self._generator_used = self._seed_generator_used

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

        self._trace.write(
            EVENT_RUN_START,
            mode="modify",
            instructions=instructions[:500],
            max_cost_usd=self.max_cost_usd,
            max_runtime_seconds=self.max_runtime_seconds,
            max_turns=self.max_turns,
        )

        # -- Phase 1.5: Validate the seeded output (no Phase 1 run) --------
        phase1_issues = self._validate_phase1_output()
        for issue in phase1_issues:
            self._trace.write(EVENT_VALIDATION_ISSUE, phase="phase1_5", message=issue)

        # -- Phase 2: LLM edits the seeded files in place ------------------
        self._trace.write(EVENT_PHASE_ENTER, phase="phase2_modify")
        self._run_phase2(instructions, extra_issues=phase1_issues)
        self._trace.write(
            EVENT_PHASE_EXIT, phase="phase2_modify", turns=self.total_turns,
        )

        # -- Snapshot BEFORE Phase 3 (preserves all Phase 2 edits) --------
        self._create_snapshot()
        self._trace.write(EVENT_SNAPSHOT, before_phase="phase3")

        # -- Phase 3: Validate & fix --------------------------------------
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
            "LLM modify finished: %d turns, %.1fs, %d tool calls, "
            "seed_generator=%s, compactions=%d",
            self.total_turns, elapsed, len(self.tool_calls_log),
            self._seed_generator_used or "none", self._compaction_count,
        )
        logger.info("Cost: %s", self.client.usage)

        self._save_recipe(instructions, elapsed)
        self._remove_snapshot()

        # Same clean-exit rule as run(): drop the checkpoint Phase 2 wrote
        # only when the LLM signalled it was done. Otherwise it stays on
        # disk so the modify run itself can be resumed.
        if self._phase2_exited_cleanly:
            delete_checkpoint(self.output_dir)

        self._trace.write(
            EVENT_RUN_END,
            mode="modify",
            elapsed_seconds=round(elapsed, 2),
            total_turns=self.total_turns,
            estimated_cost_usd=float(self.client.usage.estimated_cost),
            validation_issues=len(self._validation_issues),
        )
        return self.output_dir

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
        prompt = (
            "An existing app is being MODIFIED with this instruction:\n\n"
            f"'{instructions[:1000]}'\n\n"
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
        if self.domain_model is None and self.quantum_circuit is None:
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
        if self.target_generator:
            logger.info(
                "Phase 1: using caller-specified generator %s",
                self.target_generator,
            )
            return self.target_generator

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

        # Last resort: default based on available models. Quantum and GUI
        # specialise first because they map cleanly to a single generator;
        # bare class diagrams fall through to a backend.
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
            )
            gen_lines = [
                f"- {tool['name']} → {tool['description']}"
                for tool in GENERATOR_TOOLS
                if tool["name"] in selectable_names
            ]

            prompt = (
                f"User request: {instructions[:500]}\n\n"
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
    # Adaptive budget: raise the ceiling for from-scratch runs
    # ==================================================================

    def _apply_adaptive_budget(self) -> None:
        """Raise the cost / runtime / output-token ceiling for from-scratch runs.

        Called from ``run()`` / ``resume()`` after Phase 1 (and Phase 0.5)
        have run, so ``self._generator_used`` is authoritative:

        - ``None`` means Phase 1 found nothing to run -- either there was
          no domain/quantum model at all (state-machine/agent-only
          projects), or Phase 1 explicitly decided no registered BESSER
          generator matches the request (e.g. the user asked for Next.js,
          Rust, or Kotlin -- stacks BESSER doesn't scaffold). Either way,
          Phase 2 has NOTHING to build on top of: it authors the entire
          application from nothing, which legitimately needs more turns,
          more spend, and bigger individual responses than the common
          case (a Python scaffold Phase 2 only patches).
        - Anything else means a deterministic generator actually ran --
          the common, cheaper case -- so the caller-supplied cap (already
          clamped upstream to the web deployment's tight default) is left
          exactly as-is. This keeps the BYOK safety rail unchanged for
          the case it was sized for.

        Uses ``max()`` semantics throughout: this only ever RAISES the
        ceiling, never lowers a caller-supplied value that's already
        higher (e.g. a library caller that explicitly asked for more).
        """
        if self._generator_used is not None:
            return  # scaffolded Python run — caller's cap is untouched

        raised_cost = self.max_cost_usd < _FROM_SCRATCH_MAX_COST_USD
        raised_runtime = self.max_runtime_seconds < _FROM_SCRATCH_MAX_RUNTIME_SECONDS
        if raised_cost or raised_runtime:
            logger.info(
                "Adaptive budget: from-scratch run detected (generator_used=None, "
                "stack=%s) — raising ceiling cost $%.2f -> $%.2f, runtime %ds -> %ds",
                self._phase0_5_stack or "unrecognised",
                self.max_cost_usd, max(self.max_cost_usd, _FROM_SCRATCH_MAX_COST_USD),
                self.max_runtime_seconds,
                max(self.max_runtime_seconds, _FROM_SCRATCH_MAX_RUNTIME_SECONDS),
            )
            self.max_cost_usd = max(self.max_cost_usd, _FROM_SCRATCH_MAX_COST_USD)
            self.max_runtime_seconds = max(
                self.max_runtime_seconds, _FROM_SCRATCH_MAX_RUNTIME_SECONDS,
            )
            self._adaptive_budget_applied = True

        # Also widen the per-call output-token limit: a from-scratch run
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
                "Adaptive budget: raising output-token limit %d -> %d for "
                "from-scratch run",
                current_max_tokens, FROM_SCRATCH_MAX_TOKENS,
            )
            self.client.max_tokens = FROM_SCRATCH_MAX_TOKENS
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
                instructions=instructions,
                generator_used=self._generator_used,
                domain_model=self.domain_model,
                inventory=self._inventory,
                llm_client=self.client,
                on_progress=self.on_progress,
                on_phase_details=self.on_phase_details,
                generator_failure=self._phase1_failure_reason,
            )
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
                )
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

        system = self._build_system_prompt(
            instructions=instructions,
            scoped_issues=scoped_issues,
            gap_tasks=gap_tasks,
        )

        _cost_warning_fired = False
        start_turn = self._resume_from_turn if resumed else 0
        # Clear resume state so a subsequent run() on the same instance
        # starts fresh.
        self._resume_from_turn = 0
        self._resume_messages = None

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
                if self.use_streaming and self.on_text and hasattr(self.client, 'chat_stream'):
                    response = self._call_streaming(system, messages)
                else:
                    response = self.client.chat(
                        system=system, messages=messages, tools=self.tools,
                    )
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

                # Per-file modify-loop guard. After the tool_results are
                # appended, check whether the LLM has just made a streak
                # of modify_file calls on a single path. If so, inject a
                # high-salience reminder as a separate user message
                # BEFORE the next LLM call, so the model sees it at
                # response-time (not buried inside a tool_result blob).
                stuck_path = self._consecutive_modify_on_same_file()
                if stuck_path is not None:
                    reminder_text = self._build_modify_loop_reminder(stuck_path)
                    logger.warning(
                        "Per-file modify loop: %d consecutive modify_file "
                        "on %s — injecting reminder",
                        self._PER_FILE_MODIFY_THRESHOLD, stuck_path,
                    )
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": reminder_text}],
                    })
                    # Record so we don't re-fire on the next turn while
                    # the LLM is still on the same file. The next
                    # _consecutive_modify_on_same_file() call will return
                    # None for ``stuck_path`` until the LLM switches
                    # files / tools.
                    self._last_modify_warning_path = stuck_path
                else:
                    # Streak broken (LLM switched file/tool) — clear so
                    # a fresh streak on the same path could re-warn.
                    if self._recent_modify_targets:
                        last_tool, last_path = self._recent_modify_targets[-1]
                        if last_tool != "modify_file" or last_path != self._last_modify_warning_path:
                            self._last_modify_warning_path = None

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
                self._phase2_stop_reason = "api_error"
                self._phase2_api_error = (
                    "The model hit its output token limit"
                    + (f" ({current_max_tokens} tokens)" if current_max_tokens else "")
                    + ", so the generated code may be truncated. The run can be "
                    "resumed to continue from the last completed step, or try a "
                    "smaller scope / fewer files per run."
                )
                break
            else:
                logger.warning("Unexpected stop_reason: %s", response["stop_reason"])
                self._phase2_stop_reason = "api_error"
                self._phase2_api_error = f"unexpected stop_reason: {response['stop_reason']}"
                break

    def _execute_tool_blocks(self, tool_blocks: list, turn: int) -> list[dict]:
        """
        Execute tool call blocks, using parallel execution when possible.

        If multiple independent tool calls are made in the same turn,
        they are executed concurrently using a thread pool.

        Args:
            tool_blocks: List of tool_use content blocks from the LLM response.
            turn: Current turn number.

        Returns:
            List of tool_result dicts for the API response.
        """
        if not tool_blocks:
            return []

        tool_results = []

        if len(tool_blocks) > 1:
            # Parallel execution for multiple independent tool calls
            logger.info("Executing %d tool calls in parallel", len(tool_blocks))
            with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_WORKERS) as pool:
                futures = {
                    pool.submit(self._execute_single_tool, block, turn): block
                    for block in tool_blocks
                }
                for future in as_completed(futures):
                    block = futures[future]
                    try:
                        result_dict = future.result()
                        tool_results.append(result_dict)
                    except Exception as e:
                        logger.error("Parallel tool execution failed for %s: %s",
                                     getattr(block, "name", "?"), e)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"error": f"Execution failed: {e}"}),
                        })

            # Sort results by block order to maintain deterministic output
            block_id_order = {block.id: i for i, block in enumerate(tool_blocks)}
            tool_results.sort(key=lambda r: block_id_order.get(r["tool_use_id"], 0))
        else:
            # Single tool call -- execute directly
            block = tool_blocks[0]
            tool_results.append(self._execute_single_tool(block, turn))

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
            self._recent_tool_calls.append(loop_key)
            # Cap the ring buffer so long runs don't leak memory. Keeping
            # 2x the loop threshold is plenty — _is_stuck() only checks
            # the tail.
            if len(self._recent_tool_calls) > self._LOOP_THRESHOLD * 2:
                self._recent_tool_calls = self._recent_tool_calls[-self._LOOP_THRESHOLD * 2 :]

        # Track (tool, path) for the per-file modify streak guard. We
        # record ALL tools here — including read-only ones — so that a
        # ``read_file`` / ``list_files`` between modify calls correctly
        # breaks the streak. For ``modify_file`` we capture the path;
        # any other tool gets ``path=None`` and so trips the
        # uniqueness check in ``_consecutive_modify_on_same_file``.
        target_path = None
        if tool_name == "modify_file" and isinstance(block.input, dict):
            raw_path = block.input.get("path")
            if isinstance(raw_path, str):
                # Normalise for stable comparison across mixed
                # separators (Windows ``\`` vs POSIX ``/``).
                target_path = raw_path.replace("\\", "/").strip()
        self._recent_modify_targets.append((tool_name, target_path))
        if len(self._recent_modify_targets) > self._PER_FILE_MODIFY_THRESHOLD * 2:
            self._recent_modify_targets = self._recent_modify_targets[
                -self._PER_FILE_MODIFY_THRESHOLD * 2 :
            ]

        if self.on_progress:
            self.on_progress(turn + 1, tool_name, "executing")

        result = self.executor.execute(tool_name, block.input)

        if self._is_stuck():
            logger.warning("Possible loop: %s", tool_name)
            try:
                result_obj = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                result_obj = result
            result = json.dumps({
                "warning": f"'{tool_name}' called {self._LOOP_THRESHOLD} times in a row. Move on.",
                "result": result_obj,
            })

        success = '"error"' not in result[:100]
        self.tool_calls_log.append({
            "turn": turn + 1, "tool": tool_name,
            "input": _sanitize_for_log(block.input),
            "success": success,
        })
        self._trace.write(
            EVENT_TOOL_CALL,
            turn=turn + 1,
            tool=tool_name,
            success=success,
            input=_sanitize_for_log(block.input),
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
    ) -> None:
        """Persist mid-run state so a crash after ``turn`` can recover.

        No-op when checkpointing was disabled in the constructor (tests).
        Failures are logged at debug level — instrumentation must not
        break the run.
        """
        if not self._checkpointing_enabled:
            return
        try:
            ckpt = Checkpoint(
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                run_id=self.run_id,
                instructions=instructions,
                primary_kind=self.primary_kind,
                turn=turn,
                total_turns=self.total_turns,
                messages=messages,
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
            )
            path = save_checkpoint(self.output_dir, ckpt)
            if path:
                self._trace.write(EVENT_CHECKPOINT, turn=turn, path=path)
        except Exception as exc:
            logger.debug("Checkpoint write failed on turn %d: %s", turn, exc)

    # ==================================================================
    # Phase 3: Post-generation validation & fix
    # ==================================================================

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
        - ``tsc --noEmit`` on every tsconfig (if tsc installed)
        - ``cargo check`` on every Cargo.toml (if cargo installed)
        - ``kotlinc`` on every Kotlin source root (if kotlinc installed)

        Per-project toolchain failures (tsc / cargo / kotlinc) feed
        into a bounded fix loop: the orchestrator runs up to
        ``_MAX_TOOLCHAIN_FIX_ITERATIONS`` (collect → 5-turn LLM fix →
        re-collect) rounds before accepting whatever remains. This
        closes the gap where Phase 3 used to surface tsc errors as
        warnings (no fix attempt) and never invoked cargo / kotlinc
        at all, leaving the per-project compile-pass at 0/n for TS /
        Rust / Kotlin runs.
        """
        # Skip if runtime budget is exhausted
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed > self.max_runtime_seconds:
                logger.warning(
                    "Skipping Phase 3 -- runtime timeout: %.1fs > %ds",
                    elapsed, self.max_runtime_seconds,
                )
                return

        issues = self._collect_validation_issues()

        if not issues:
            logger.info("Phase 3: Validation passed -- no issues found")
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

        # Outer cap: up to _MAX_TOOLCHAIN_FIX_ITERATIONS rounds of
        # (LLM-fix → re-validate). Each round is the existing 5-turn
        # fix loop, so worst-case extra cost is bounded at 3 × 5 = 15
        # LLM turns. Toolchain blockers (tsc / cargo / kotlinc) are
        # the typical reason for multiple rounds: the LLM fixes one
        # type error and uncovers the next downstream of it.
        current_blockers = blockers_before
        prev_blocker_count = len(blockers_before)
        last_issues = list(issues)

        for attempt in range(_MAX_TOOLCHAIN_FIX_ITERATIONS):
            is_first_attempt = attempt == 0
            self._trace.write(
                EVENT_PHASE_ENTER,
                phase="phase3_fix_attempt",
                attempt=attempt + 1,
                blockers=len(current_blockers),
            )
            self._invoke_phase3_fix_loop(current_blockers, is_first_attempt)

            # Re-validate. The bench's per-project compile-pass score
            # only cares about a clean toolchain, so re-running these
            # is what actually drives the metric.
            issues_after = self._collect_validation_issues()
            last_issues = issues_after
            blockers_after = [i for i in issues_after if i.severity == "blocker"]
            self._trace.write(
                EVENT_PHASE_EXIT,
                phase="phase3_fix_attempt",
                attempt=attempt + 1,
                blockers_remaining=len(blockers_after),
            )

            if not blockers_after:
                logger.info(
                    "Phase 3: All blockers fixed after %d attempt(s) "
                    "(%d non-blocker remain).",
                    attempt + 1, len(issues_after),
                )
                self._validation_issues = list(issues_after)
                return

            if len(blockers_after) > len(blockers_before):
                # Fixes made BLOCKERS worse than the original state →
                # rollback to pre-Phase-3 and stop. Comparing against
                # blockers_before (the pre-Phase-3 baseline), not the
                # previous attempt, so a single transient regression
                # mid-loop doesn't trigger the rollback.
                logger.warning(
                    "Phase 3: Fixes made blockers worse (%d -> %d). "
                    "Rolling back to keep Phase 2 work.",
                    len(blockers_before), len(blockers_after),
                )
                self._restore_snapshot()
                self._validation_issues = self._collect_validation_issues()
                for issue in self._validation_issues:
                    logger.warning("  Unfixed [%s]: %s", issue.severity, issue.message)
                return

            if len(blockers_after) >= prev_blocker_count:
                # No progress this round (same or more blockers than
                # we started this attempt with). Spending another
                # 5 turns isn't going to help — exit early instead of
                # exhausting the iteration budget.
                logger.warning(
                    "Phase 3: Attempt %d made no progress (%d -> %d "
                    "blockers); ending fix loop early.",
                    attempt + 1, prev_blocker_count, len(blockers_after),
                )
                break

            prev_blocker_count = len(blockers_after)
            current_blockers = blockers_after

        # We get here either by ending the loop early (no progress)
        # or by exhausting the attempt cap. Record whatever the final
        # state is so the recipe surfaces it.
        self._validation_issues = list(last_issues)
        remaining_blockers = [
            i for i in last_issues if i.severity == "blocker"
        ]
        if remaining_blockers:
            logger.warning(
                "Phase 3: %d blocker(s) remain after %d attempt(s); "
                "accepting as-is.",
                len(remaining_blockers), _MAX_TOOLCHAIN_FIX_ITERATIONS,
            )
            for issue in last_issues:
                logger.warning("  [%s] %s", issue.severity, issue.message)

    def _invoke_phase3_fix_loop(
        self,
        blockers: list[ValidationIssue],
        is_first_attempt: bool,
    ) -> None:
        """Run one 5-turn LLM fix loop against ``blockers``.

        Factored out of ``_run_phase3_validation`` so the outer
        toolchain-fix iteration cap can call it more than once. Each
        invocation builds a fresh prompt (so the LLM doesn't see
        stale context from a previous attempt) and exits as soon as
        the LLM emits ``end_turn`` or hits the per-loop turn budget.

        The prompt always reproduces the current blocker list verbatim
        — including any ``tsc [...]:`` / ``cargo [...]:`` /
        ``kotlinc [...]:`` lines. When toolchain blockers are present,
        a high-salience reminder is appended instructing the LLM to
        re-run the toolchain via ``run_command`` after each edit, so
        the model verifies its own fixes instead of stopping at the
        first plausible-looking change.
        """
        if not blockers:
            return

        toolchain_blockers = [
            i for i in blockers
            if i.message.startswith(("tsc [", "cargo [", "kotlinc ["))
        ]
        other_blockers = [
            i for i in blockers if i not in toolchain_blockers
        ]

        prompt_parts: list[str] = []
        if is_first_attempt:
            prompt_parts.append(
                "Post-generation validation found these BLOCKER issues "
                "(syntax / dependency / missing-file / toolchain). Fix "
                "every one — they prevent the app from running:"
            )
        else:
            prompt_parts.append(
                "After your previous fixes, these BLOCKER issues "
                "still remain. Fix every one:"
            )

        prompt_parts.append("")
        prompt_parts.extend(f"- {i.message}" for i in blockers)
        prompt_parts.append("")
        prompt_parts.append(
            "Use modify_file or write_file as needed. Do NOT touch "
            "anything unrelated."
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
            "Fix each issue concisely. When the report contains "
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

        for turn in range(5):  # max 5 fix turns per attempt
            # Same per-turn guards as the Phase 2 loop — the fix loop bills
            # real turns, so it must honour cancellation and the run's
            # cost/runtime budget instead of assuming the 5-turn cap is
            # small enough to never matter (15 worst-case turns across the
            # outer attempts is NOT small on an expensive model).
            if self._should_continue is not None and not self._should_continue():
                logger.warning(
                    "Phase 3: cancellation requested — stopping fix loop"
                )
                return
            if self._start_time is not None:
                elapsed = time.monotonic() - self._start_time
                if elapsed > self.max_runtime_seconds:
                    logger.warning(
                        "Phase 3: runtime cap reached (%.1fs > %ds) — "
                        "stopping fix loop", elapsed, self.max_runtime_seconds,
                    )
                    return
            if self.client.usage.estimated_cost > self.max_cost_usd:
                logger.warning(
                    "Phase 3: cost cap reached before fix turn %d "
                    "($%.4f > $%.4f) — stopping fix loop",
                    turn + 1, self.client.usage.estimated_cost, self.max_cost_usd,
                )
                return

            self.total_turns += 1
            try:
                response = self.client.chat(
                    system=system, messages=messages, tools=self.tools,
                )
            except Exception as exc:
                # Surface the failure instead of silently exiting the fix
                # loop — callers and logs need to see why validation bailed.
                logger.warning(
                    "Phase 3: LLM call failed on fix turn %d, aborting fix loop: %s",
                    turn + 1, exc,
                )
                return
            if response["stop_reason"] == "end_turn":
                return
            if response["stop_reason"] == "tool_use":
                messages.append({"role": "assistant", "content": response["content"]})
                tool_results = []
                for block in response["content"]:
                    if hasattr(block, "type") and block.type == "tool_use":
                        result = self.executor.execute(
                            getattr(block, "name", ""),
                            getattr(block, "input", {}),
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})

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

    def _collect_validation_issues(self) -> list[ValidationIssue]:
        """Collect all validation issues from the output directory.

        Returns ``ValidationIssue`` records with severity. The
        Phase 3 fix loop only acts on ``blocker`` items when
        ``auto_fix_issues`` is enabled.
        """
        raw_issues: list[str] = []

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

                # Check Dockerfiles
                if fname == "Dockerfile":
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read()
                        docker_dir = os.path.dirname(fpath)
                        # npm ci without lock file -> should be npm install
                        if "npm ci" in content:
                            lock = os.path.join(docker_dir, "package-lock.json")
                            if not os.path.isfile(lock):
                                # Auto-fix this common mistake
                                fixed = content.replace("npm ci", "npm install")
                                with open(fpath, "w", encoding="utf-8") as f:
                                    f.write(fixed)
                                logger.info("Auto-fixed: %s: npm ci -> npm install (no lock file)", rel)
                        # Check COPY references
                        if "package.json" in content or "package*.json" in content:
                            pkg = os.path.join(docker_dir, "package.json")
                            if not os.path.isfile(pkg):
                                raw_issues.append(f"{rel} references package.json but it doesn't exist")
                        if "requirements.txt" in content:
                            req = os.path.join(docker_dir, "requirements.txt")
                            if not os.path.isfile(req):
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

        # Verify Python dependencies actually install without conflicts.
        # This is the ROOT FIX for dependency issues — instead of hardcoding
        # specific workarounds (passlib/bcrypt, etc.), we test ALL deps.
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
                        )
                        if result.returncode != 0:
                            # Extract the meaningful error
                            err_lines = [l for l in result.stderr.strip().split("\n")
                                         if l.strip() and "WARNING" not in l]
                            if err_lines:
                                err = "\n".join(err_lines[-3:])
                                raw_issues.append(f"Dependency conflict in {rel}:\n{err}")
                    except Exception:
                        pass  # pip not available or timeout — skip

        # ``ruff``, ``tsc``, ``cargo``, and ``kotlinc`` are best-effort
        # static checks — each skips silently if the binary isn't on
        # PATH. They catch the per-project compile errors that would
        # otherwise only surface at deploy time. ruff is near-instant
        # and always runs; the project compilers (tsc / cargo /
        # kotlinc) can add minutes of wall-clock and are gated behind
        # ``enable_toolchain_validation`` so the web deployment can
        # opt out per deploy.
        raw_issues.extend(self._collect_ruff_issues())
        if self.enable_toolchain_validation:
            raw_issues.extend(self._collect_tsc_issues())
            raw_issues.extend(self._collect_cargo_issues())
            raw_issues.extend(self._collect_kotlinc_issues())
        else:
            logger.info(
                "Phase 3: toolchain validation (tsc/cargo/kotlinc) disabled "
                "for this run"
            )

        return [_classify_issue(s) for s in raw_issues]

    def _collect_ruff_issues(self) -> list[str]:
        """Run ``ruff check`` across the workspace when available.

        Returns a list of concise issue strings. Skips silently if ruff is
        not installed, hits a timeout, or produces no parseable output —
        these are "nice to have" checks, not run-blockers.
        """
        import shutil as _shutil
        import subprocess

        ruff_bin = _shutil.which("ruff")
        if not ruff_bin:
            return []

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
            )
        except (subprocess.TimeoutExpired, OSError):
            return []

        lines = (result.stdout or "").strip().splitlines()
        if not lines:
            return []
        # Cap to keep Phase 3 feedback scoped — the LLM doesn't need every
        # unused-import warning, it needs the shape of the problem.
        issues = [f"ruff: {line.strip()}" for line in lines[:20] if line.strip()]
        if len(lines) > 20:
            issues.append(f"ruff: (+{len(lines) - 20} more issues truncated)")
        return issues

    def _collect_tsc_issues(self) -> list[str]:
        """Run ``tsc --noEmit`` for any TypeScript project in the workspace.

        Looks for ``tsconfig.json`` files (skipping the snapshot dir) and
        runs ``tsc --noEmit`` against each project root. Skips silently
        if ``tsc`` is not available — tsc isn't installed by default and
        we don't want to fail runs on user machines that don't have it.
        """
        import shutil as _shutil
        import subprocess

        tsc_bin = _shutil.which("tsc") or _shutil.which("tsc.cmd")
        if not tsc_bin:
            return []

        tsconfigs: list[str] = []
        for root, _, files in os.walk(self.output_dir):
            rel_root = os.path.relpath(root, self.output_dir).replace("\\", "/")
            if rel_root.startswith(_SNAPSHOT_DIR):
                continue
            # Skip node_modules — running tsc there is both meaningless
            # and extremely slow.
            if "node_modules" in rel_root.split("/"):
                continue
            if "tsconfig.json" in files:
                tsconfigs.append(root)

        if not tsconfigs:
            return []

        issues: list[str] = []
        for project_dir in tsconfigs:
            rel = os.path.relpath(project_dir, self.output_dir).replace("\\", "/") or "."
            try:
                result = subprocess.run(
                    [tsc_bin, "--noEmit", "-p", "."],
                    capture_output=True, text=True, timeout=60,
                    cwd=project_dir,
                )
            except (subprocess.TimeoutExpired, OSError):
                continue

            # tsc emits errors on stdout (not stderr) in the classic
            # ``file(line,col): error TSxxxx: message`` format.
            output = (result.stdout or "").strip().splitlines()
            err_lines = [ln.strip() for ln in output if ln.strip() and "error" in ln.lower()]
            if not err_lines:
                continue
            for line in err_lines[:10]:
                issues.append(f"tsc [{rel}]: {line}")
            if len(err_lines) > 10:
                issues.append(f"tsc [{rel}]: (+{len(err_lines) - 10} more errors truncated)")
        return issues

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
        cargo_env = {**os.environ}
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
            except (subprocess.TimeoutExpired, OSError):
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

            # Copy all files except the snapshot dir itself
            for item in os.listdir(self.output_dir):
                if item == _SNAPSHOT_DIR:
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

    def _restore_snapshot(self) -> None:
        """Restore output directory from snapshot."""
        snapshot_path = os.path.join(self.output_dir, _SNAPSHOT_DIR)
        if not os.path.isdir(snapshot_path):
            logger.warning("No snapshot to restore from")
            return

        try:
            # Remove everything except the snapshot
            for item in os.listdir(self.output_dir):
                if item == _SNAPSHOT_DIR:
                    continue
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

            # Copy snapshot contents back
            for item in os.listdir(snapshot_path):
                src = os.path.join(snapshot_path, item)
                dst = os.path.join(self.output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            logger.info("Restored from snapshot")
        except Exception as e:
            logger.warning("Failed to restore from snapshot: %s", e)

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
        lines = [l.strip() for l in error_message.strip().splitlines() if l.strip()]
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
                if self.use_streaming and self.on_text and hasattr(self.client, 'chat_stream'):
                    response = self._call_streaming(system, messages)
                else:
                    response = self.client.chat(
                        system=system, messages=messages, tools=self.tools,
                    )
            except Exception as e:
                logger.error("Fix cycle API call failed: %s", e)
                break

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
                stuck_path = self._consecutive_modify_on_same_file()
                if stuck_path is not None:
                    reminder_text = self._build_modify_loop_reminder(stuck_path)
                    logger.warning(
                        "Per-file modify loop (fix cycle): %d consecutive "
                        "modify_file on %s — injecting reminder",
                        self._PER_FILE_MODIFY_THRESHOLD, stuck_path,
                    )
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": reminder_text}],
                    })
                    self._last_modify_warning_path = stuck_path
                else:
                    if self._recent_modify_targets:
                        last_tool, last_path = self._recent_modify_targets[-1]
                        if last_tool != "modify_file" or last_path != self._last_modify_warning_path:
                            self._last_modify_warning_path = None
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
        # Inline small scaffold files so the LLM doesn't burn its first
        # turns on read_file calls. Per-run constant, so it caches like
        # the rest of the prompt. Disable via BESSER_LLM_INLINE_SCAFFOLD=0.
        scaffold_snapshot = ""
        if (
            os.environ.get("BESSER_LLM_INLINE_SCAFFOLD", "1").lower()
            not in ("0", "false")
            and (self._generator_used or self._phase0_5_files)
        ):
            try:
                scaffold_snapshot = build_scaffold_snapshot(self.output_dir)
            except Exception:
                logger.debug("Scaffold snapshot build failed", exc_info=True)

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
            primary_kind=self.primary_kind,
            scaffold_snapshot=scaffold_snapshot,
            # ``_modify_mode`` is False on the run()/resume() paths, so the
            # from-scratch prompt stays byte-identical; only ``modify()``
            # flips it to prepend the "preserve what works" directive.
            modify_mode=self._modify_mode,
        )

    def _build_inventory(self, generator_name: str) -> str:
        """Delegate to prompt_builder module."""
        return build_inventory(self.output_dir, self.domain_model, generator_name)

    def _maybe_compact(self, messages: list[dict]) -> list[dict]:
        """Delegate to compaction module."""
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
            primary_kind=self.primary_kind,
        )
        if did_compact:
            self._compaction_count += 1
        return result

    def _summarize_messages(self, messages: list[dict]) -> str:
        """Delegate to compaction module."""
        return _summarize_messages(messages, self.tool_calls_log, self.output_dir)

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

    def _is_stuck(self) -> bool:
        recent = self._recent_tool_calls[-self._LOOP_THRESHOLD:]
        return len(recent) >= self._LOOP_THRESHOLD and len(set(recent)) == 1

    def _consecutive_modify_on_same_file(self) -> str | None:
        """Return the file path being repeatedly modified, or None.

        Fires when the tail of the recent write-tool history is
        ``_PER_FILE_MODIFY_THRESHOLD`` consecutive ``modify_file`` calls
        on the SAME (normalised) path. Any other write-class tool in
        the window — ``write_file``, ``run_command``, ``delete_file``,
        etc. — breaks the streak because its slot in the buffer has
        ``path=None`` (or a different path), so the uniqueness check
        below fails.

        Resets / suppresses repeat firing: once we've warned about a
        path, ``_last_modify_warning_path`` is set; subsequent identical
        streaks return None until the LLM either switches files or
        switches tools.
        """
        n = self._PER_FILE_MODIFY_THRESHOLD
        recent = self._recent_modify_targets[-n:]
        if len(recent) < n:
            return None
        if any(tool != "modify_file" for tool, _ in recent):
            return None
        paths = {path for _, path in recent}
        if len(paths) != 1:
            return None
        path = next(iter(paths))
        if path is None:
            # modify_file without a parseable path argument — skip.
            return None
        if path == self._last_modify_warning_path:
            # Already warned about this streak; wait for a real change
            # of file or tool before firing again.
            return None
        return path

    def _build_modify_loop_reminder(self, path: str) -> str:
        """High-salience system-style reminder text for the per-file
        modify-streak guard. Worded to push the LLM toward a decisive
        action (rewrite-or-quit) instead of continuing to dribble out
        single-line edits.
        """
        n = self._PER_FILE_MODIFY_THRESHOLD
        return (
            f"<system-reminder>You've called modify_file on `{path}` "
            f"{n} times in a row. Stop incrementally editing this file. "
            "EITHER call write_file with the complete new contents for "
            f"`{path}` in a single call (generator-created files become "
            "rewritable after repeated modify_file edits — an earlier "
            "write_file rejection no longer applies), OR stop modifying "
            "this file and move on to other work (or call end_turn / "
            "produce your final response if the file is satisfactory). "
            "Do NOT issue another modify_file on this same path."
            "</system-reminder>"
        )

    # ==================================================================
    # Recipe
    # ==================================================================

    def _save_recipe(self, instructions: str, elapsed: float) -> None:
        # Build file manifest. Dependency / build directories are pruned:
        # an LLM-run ``npm install`` would otherwise put thousands of
        # node_modules entries in the manifest, ballooning the recipe
        # past the SSE embed cap (a production run hit 4.9 MB and the
        # whole recipe was dropped from the done event).
        output_files = []
        generator_files = self.executor._generator_files if hasattr(self.executor, '_generator_files') else set()
        try:
            for root, dirs, fnames in os.walk(self.output_dir):
                dirs[:] = [d for d in dirs if d not in _RECIPE_EXCLUDED_DIRS]
                for f in fnames:
                    if f.startswith(".besser_"):
                        continue
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, self.output_dir).replace("\\", "/")
                    output_files.append({
                        "path": rel,
                        "size": os.path.getsize(full),
                        "source": "generator" if rel in generator_files else "llm",
                    })
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

        recipe = {
            "instructions": instructions,
            "model": recipe_model,
            "llm_model": self.client.model,
            "generator_used": self._generator_used,
            "turns": self.total_turns,
            "tool_calls_count": len(self.tool_calls_log),
            "compactions": self._compaction_count,
            "elapsed_seconds": round(elapsed, 1),
            "max_cost_usd": self.max_cost_usd,
            "max_runtime_seconds": self.max_runtime_seconds,
            # True when this was a from-scratch run (no deterministic
            # Phase 1 generator) and the cost/runtime/output-token ceiling
            # above was therefore raised past the caller-supplied default.
            # See LLMOrchestrator._apply_adaptive_budget.
            "adaptive_budget_applied": self._adaptive_budget_applied,
            "usage": self.client.usage.summary(),
            "validation_issues": [
                {"severity": i.severity, "message": i.message}
                for i in self._validation_issues
            ],
            "output_files": sorted(output_files, key=lambda f: f["path"]),
            "output_summary": {
                "total_files": len(output_files),
                "from_generator": sum(1 for f in output_files if f["source"] == "generator"),
                "from_llm": sum(1 for f in output_files if f["source"] == "llm"),
                "total_bytes": sum(f["size"] for f in output_files),
            },
            "tool_calls": self.tool_calls_log,
            # Pointer to the structured trace file alongside this recipe.
            # Clients that want per-turn detail (tool calls, cost ticks,
            # phase transitions) can read it instead of re-parsing the
            # recipe's flattened summaries.
            "trace_file": TRACE_FILENAME if self._trace.path else None,
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

def _sanitize_for_log(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            k: (v[:500] + "..." if isinstance(v, str) and len(v) > 500 else v)
            for k, v in data.items()
        }
    return data
