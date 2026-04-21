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
from besser.generators.llm.gap_analyzer import analyze_gaps_via_llm
from besser.generators.llm.llm_client import ClaudeLLMClient
from besser.generators.llm.prompt_builder import (
    build_system_prompt,
    build_inventory,
)
from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.tools import get_all_tools, get_all_tools_including_generators

logger = logging.getLogger(__name__)

# Snapshot directory name (inside output_dir)
_SNAPSHOT_DIR = ".besser_snapshot"


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

    # tsc errors are usually warnings (often missing @types packages,
    # not actual code bugs). Treat them as warnings unless they're
    # clearly fatal — keeping it simple for v1.
    if text.startswith("tsc "):
        return ValidationIssue("warning", text)

    # Unknown shape → conservative default: warning.
    return ValidationIssue("warning", text)

# Tools that are read-only and shouldn't count for loop detection
_READONLY_TOOLS = frozenset({"read_file", "list_files", "search_in_files", "check_syntax"})

# Maximum workers for parallel tool execution
_MAX_PARALLEL_WORKERS = 4


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
        domain_model,
        gui_model=None,
        agent_model=None,
        agent_config: dict | None = None,
        output_dir: str | None = None,
        max_turns: int | None = None,
        max_cost_usd: float = 5.0,
        max_runtime_seconds: int = 1200,
        on_progress: Callable[[int, str, str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        use_streaming: bool = True,
        object_model=None,
        state_machines=None,
        quantum_circuit=None,
        auto_fix_issues: bool = False,
        should_continue: Callable[[], bool] | None = None,
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
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="besser_llm_")
        self.max_turns = max_turns or self.MAX_TURNS
        self.max_cost_usd = max_cost_usd
        self.max_runtime_seconds = max_runtime_seconds
        self.on_progress = on_progress
        self.on_text = on_text
        self.use_streaming = use_streaming
        self.executor = ToolExecutor(
            workspace=self.output_dir,
            domain_model=domain_model,
            gui_model=gui_model,
            agent_model=agent_model,
            agent_config=agent_config,
        )
        # Give the LLM ALL tools including generators — it might need
        # generate_pydantic during customization, or call a generator
        # the orchestrator didn't pick in Phase 1.
        self.tools = get_all_tools_including_generators()
        # Phase 3 auto-fix policy. False = report-only (industry default
        # for static analysers — fix on request, never blindly).
        self.auto_fix_issues = auto_fix_issues
        # Cooperative cancellation hook. The orchestrator polls this at
        # the top of each Phase 2 turn. Returning False causes the loop
        # to exit cleanly — used by the SSE runner to honour
        # ``POST /cancel-smart-gen/{run_id}`` without killing the thread.
        self._should_continue = should_continue
        self.tool_calls_log: list[dict] = []
        self.total_turns = 0
        self._recent_tool_calls: list[str] = []
        self._compaction_count = 0
        self._generator_used: str | None = None
        self._inventory: str = ""
        self._start_time: float | None = None
        # Stored as ValidationIssue objects so the recipe captures severity.
        # Cast to strings via `[str(i) for i in self._validation_issues]`
        # when emitting JSON.
        self._validation_issues: list[ValidationIssue] = []
        self._previous_errors: list[str] = []  # track errors to avoid re-attempting

    _LOOP_THRESHOLD = 4

    # ==================================================================
    # Main entry point
    # ==================================================================

    def run(self, instructions: str) -> str:
        """Run the three-phase generation. Returns path to output directory."""
        if not instructions or not instructions.strip():
            raise ValueError("Instructions cannot be empty")

        self._start_time = time.monotonic()

        # -- Phase 1: Deterministic generation ----------------------------
        self._run_phase1(instructions)

        # -- Phase 1.5: Validate Phase 1 output ---------------------------
        phase1_issues = self._validate_phase1_output()

        # -- Phase 2: LLM customization -----------------------------------
        self._run_phase2(instructions, extra_issues=phase1_issues)

        # -- Snapshot BEFORE Phase 3 (preserves all Phase 2 work) ---------
        # If Phase 3 fixes make things worse, we roll back here
        # (keeping Phase 2 work intact), not back to Phase 1.
        self._create_snapshot()

        # -- Phase 3: Validate & fix --------------------------------------
        self._run_phase3_validation()

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

        return self.output_dir

    # ==================================================================
    # Phase 1: Deterministic generation (no LLM)
    # ==================================================================

    def _run_phase1(self, instructions: str) -> None:
        """Select and run the best generator, then inventory the output."""
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
                logger.warning("Phase 1: Generator failed: %s", result.get("error"))
        else:
            logger.info("Phase 1: No matching generator -- LLM will write from scratch")

    def _select_generator(self, instructions: str = "") -> str | None:
        """
        Pick the best generator using a cheap LLM call or keyword fallback.

        Tries a quick LLM call first (if available), falls back to keyword
        matching. Respects what the user asked for — if they want NestJS,
        returns None so the LLM writes from scratch.
        """
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
        if self.gui_model:
            return "generate_web_app"
        if self.domain_model and self.domain_model.get_classes():
            return "generate_fastapi_backend"
        return None

    def _select_generator_with_llm(self, instructions: str) -> str | None:
        """Use a cheap LLM call to pick the best generator for Phase 1."""
        try:
            from besser.generators.llm.tools import GENERATOR_TOOLS

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

            # Build generator list dynamically from the tool registry.
            gen_lines = []
            for tool in GENERATOR_TOOLS:
                name = tool["name"]
                desc = tool["description"]
                needs_gui = "REQUIRES GUI model" in desc
                available = "AVAILABLE" if (not needs_gui or self.gui_model) else "NOT available (no GUI model)"
                gen_lines.append(f"- {name} → {desc} [{available}]")

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

            # Match against EVERY registered generator (was previously a hand-
            # maintained subset that omitted qiskit, flutter, java, etc.).
            # Sort by name length descending so longer prefixes win first
            # (e.g. ``generate_python_classes`` is checked before
            # ``generate_python``).
            registered_names = sorted(
                (tool["name"] for tool in GENERATOR_TOOLS),
                key=len,
                reverse=True,
            )
            for gen in registered_names:
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
        scoped_issues = list(extra_issues) if extra_issues else []
        gap_tasks = analyze_gaps_via_llm(
            instructions=instructions,
            generator_used=self._generator_used,
            domain_model=self.domain_model,
            inventory=self._inventory,
            llm_client=self.client,
        )
        system = self._build_system_prompt(
            instructions=instructions,
            scoped_issues=scoped_issues,
            gap_tasks=gap_tasks,
        )
        messages: list[dict] = [{"role": "user", "content": instructions}]

        _cost_warning_fired = False

        for turn in range(self.max_turns):
            self.total_turns = turn + 1

            # -- Cooperative cancellation -------------------------------
            # The runner sets the underlying flag when a user POSTs to
            # /cancel-smart-gen/{run_id}. Bail out at the next turn
            # boundary rather than killing the worker thread.
            if self._should_continue is not None and not self._should_continue():
                logger.warning("Cancellation requested — stopping Phase 2 loop")
                break

            # -- Runtime timeout check ------------------------------------
            if self._start_time is not None:
                elapsed = time.monotonic() - self._start_time
                if elapsed > self.max_runtime_seconds:
                    logger.warning(
                        "Runtime timeout: %.1fs > %ds", elapsed, self.max_runtime_seconds,
                    )
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
            except Exception as e:
                logger.error("LLM API call failed on turn %d: %s", turn + 1, e)
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
                break

            if response["stop_reason"] == "end_turn":
                logger.info("LLM completed after %d turns", turn + 1)
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
            else:
                logger.warning("Unexpected stop_reason: %s", response["stop_reason"])
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
            self._recent_tool_calls.append(tool_name)

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

        self.tool_calls_log.append({
            "turn": turn + 1, "tool": tool_name,
            "input": _sanitize_for_log(block.input),
            "success": '"error"' not in result[:100],
        })

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        }

    # ==================================================================
    # Phase 3: Post-generation validation & fix
    # ==================================================================

    def _run_phase3_validation(self) -> None:
        """
        Lightweight validation of generated output. If issues found,
        give the LLM a few turns to fix them.

        Checks (no network, no Docker, instant):
        - Python syntax on all .py files
        - Dockerfiles reference files that exist
        - package.json exists if Dockerfile uses npm
        - npm ci -> npm install (common LLM mistake)
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

        fix_prompt = (
            "Post-generation validation found these BLOCKER issues "
            "(syntax / dependency / missing-file). Fix every one — they "
            "prevent the app from running:\n\n"
            + "\n".join(f"- {i.message}" for i in blockers_before)
            + "\n\nUse modify_file or write_file as needed. Do NOT touch "
            "anything unrelated."
        )
        system = "You are fixing validation errors in generated code. Fix each issue concisely."
        messages: list[dict] = [{"role": "user", "content": fix_prompt}]

        for turn in range(5):  # max 5 fix turns
            self.total_turns += 1
            try:
                response = self.client.chat(system=system, messages=messages, tools=self.tools)
            except Exception:
                break
            if response["stop_reason"] == "end_turn":
                break
            if response["stop_reason"] == "tool_use":
                messages.append({"role": "assistant", "content": response["content"]})
                tool_results = []
                for block in response["content"]:
                    if hasattr(block, "type") and block.type == "tool_use":
                        result = self.executor.execute(getattr(block, "name", ""), getattr(block, "input", {}))
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})

        # Re-evaluate against blocker count specifically. Style/warning
        # changes don't justify a rollback — only a worse blocker count.
        issues_after = self._collect_validation_issues()
        blockers_after = [i for i in issues_after if i.severity == "blocker"]
        if not blockers_after:
            logger.info(
                "Phase 3: All blockers fixed (%d non-blocker remain).",
                len(issues_after),
            )
            self._validation_issues = list(issues_after)
        elif len(blockers_after) > len(blockers_before):
            # Fixes made BLOCKERS worse → rollback to pre-Phase-3 state.
            logger.warning(
                "Phase 3: Fixes made blockers worse (%d -> %d). "
                "Rolling back to keep Phase 2 work.",
                len(blockers_before), len(blockers_after),
            )
            self._restore_snapshot()
            self._validation_issues = self._collect_validation_issues()
            for issue in self._validation_issues:
                logger.warning("  Unfixed [%s]: %s", issue.severity, issue.message)
        else:
            self._validation_issues = list(issues_after)
            if issues_after:
                logger.warning(
                    "Phase 3: %d issues remain after fixes (%d blocker)",
                    len(issues_after), len(blockers_after),
                )
                for issue in issues_after:
                    logger.warning("  [%s] %s", issue.severity, issue.message)

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

        # ``ruff`` and ``tsc`` are best-effort static checks — they skip
        # silently if the binary isn't on PATH. Both catch classes of bug
        # (dead imports, type errors) that would otherwise only surface
        # at runtime, burning LLM turns.
        raw_issues.extend(self._collect_ruff_issues())
        raw_issues.extend(self._collect_tsc_issues())

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

    # ==================================================================
    # Recipe
    # ==================================================================

    def _save_recipe(self, instructions: str, elapsed: float) -> None:
        # Build file manifest
        output_files = []
        generator_files = self.executor._generator_files if hasattr(self.executor, '_generator_files') else set()
        try:
            for root, _, fnames in os.walk(self.output_dir):
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

        # Build model summary
        classes = [c.name for c in self.domain_model.get_classes()]
        enums = [e.name for e in self.domain_model.get_enumerations()]

        recipe = {
            "instructions": instructions,
            "model": {
                "name": self.domain_model.name,
                "classes": classes,
                "enumerations": enums,
                "associations": len(self.domain_model.associations),
            },
            "llm_model": self.client.model,
            "generator_used": self._generator_used,
            "turns": self.total_turns,
            "tool_calls_count": len(self.tool_calls_log),
            "compactions": self._compaction_count,
            "elapsed_seconds": round(elapsed, 1),
            "max_cost_usd": self.max_cost_usd,
            "max_runtime_seconds": self.max_runtime_seconds,
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
