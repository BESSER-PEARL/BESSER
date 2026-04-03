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
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from besser.generators.llm.compaction import (
    COMPACT_TOKEN_THRESHOLD,
    COMPACT_PRESERVE_RECENT,
    _estimate_tokens,
    maybe_compact,
    _summarize_messages,
)
from besser.generators.llm.gap_analyzer import (
    _AUTH_KEYWORDS,
    _DB_KEYWORDS,
    _DOCKER_KEYWORDS,
    _TEST_KEYWORDS,
    _SEARCH_KEYWORDS,
    _PAGINATION_KEYWORDS,
    _ROLE_KEYWORDS,
    analyze_gaps,
    _analyze_gaps_with_llm,
    _analyze_gaps_keyword,
    build_model_summary,
)
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
    ):
        self.client = llm_client
        self.domain_model = domain_model
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
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
        self.tool_calls_log: list[dict] = []
        self.total_turns = 0
        self._recent_tool_calls: list[str] = []
        self._compaction_count = 0
        self._generator_used: str | None = None
        self._inventory: str = ""
        self._start_time: float | None = None
        self._validation_issues: list[str] = []
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

        # Last resort: default based on available models
        if self.gui_model:
            return "generate_web_app"
        if self.domain_model and self.domain_model.get_classes():
            return "generate_fastapi_backend"
        return None

    def _select_generator_with_llm(self, instructions: str) -> str | None:
        """Use a cheap LLM call to pick the best generator for Phase 1."""
        try:
            from besser.generators.llm.tools import GENERATOR_TOOLS

            has_gui = "YES" if self.gui_model else "NO"
            classes = [c.name for c in self.domain_model.get_classes()] if self.domain_model else []

            # Build generator list dynamically from the tool registry
            gen_lines = []
            for tool in GENERATOR_TOOLS:
                name = tool["name"]
                desc = tool["description"]
                needs_gui = "REQUIRES GUI model" in desc
                available = "AVAILABLE" if (not needs_gui or self.gui_model) else "NOT available (no GUI model)"
                gen_lines.append(f"- {name} → {desc} [{available}]")

            prompt = (
                f"User request: {instructions[:500]}\n\n"
                f"Domain model: {len(classes)} classes ({', '.join(classes[:10])})\n"
                f"GUI model: {has_gui}\n\n"
                "Available BESSER generators:\n"
                + "\n".join(gen_lines) + "\n"
                "- NONE → write from scratch (for frameworks BESSER doesn't support: NestJS, Next.js, Express, Spring Boot, Go, etc.)\n\n"
                "RULES:\n"
                "- Pick the generator that best covers the MAIN part of the request\n"
                "- Even if the user asks for more than one thing (backend + frontend), pick the generator for the biggest part\n"
                "- generate_fastapi_backend includes SQLAlchemy + Pydantic — don't pick those separately\n"
                "- generate_web_app includes React + FastAPI + Docker — most complete if GUI available\n"
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

            # Check for a generator name FIRST (before checking for "none",
            # because answer might contain both, e.g. "generate_fastapi_backend, none for frontend")
            for gen in ("generate_web_app", "generate_fastapi_backend", "generate_django",
                        "generate_pydantic", "generate_sqlalchemy", "generate_react",
                        "generate_python_classes", "generate_sql"):
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
        """Give the LLM scoped tasks based on gap analysis."""
        gap_tasks = self._analyze_gaps(instructions)

        # Append any Phase 1 validation issues as extra tasks
        if extra_issues:
            gap_tasks.extend(extra_issues)

        system = self._build_system_prompt(gap_tasks)
        messages: list[dict] = [{"role": "user", "content": instructions}]

        _cost_warning_fired = False

        for turn in range(self.max_turns):
            self.total_turns = turn + 1

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

        error_count_before = len(issues)
        # Log each issue clearly so user can see what's wrong
        logger.warning("Phase 3: Found %d issues:", error_count_before)
        for i, issue in enumerate(issues, 1):
            logger.warning("  Issue %d: %s", i, issue)
        self._validation_issues = list(issues)  # save for recipe

        if self.on_progress:
            self.on_progress(self.total_turns, "validation", f"{error_count_before} issues")

        # Give the LLM a few turns to fix
        fix_prompt = (
            "Post-generation validation found these issues:\n\n"
            + "\n".join(f"- {i}" for i in issues)
            + "\n\nFix each issue. Use modify_file or write_file as needed."
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

        # Check if fixes helped
        issues_after = self._collect_validation_issues()
        if not issues_after:
            logger.info("Phase 3: All issues fixed!")
            self._validation_issues = []
        elif len(issues_after) > error_count_before:
            # Fixes made things worse → rollback to pre-Phase-3 state
            logger.warning(
                "Phase 3: Fixes made things worse (%d -> %d issues). "
                "Rolling back to keep Phase 2 work. Remaining issues:",
                error_count_before, len(issues_after),
            )
            self._restore_snapshot()
            # Re-collect issues from the restored state
            self._validation_issues = self._collect_validation_issues()
            for i, issue in enumerate(self._validation_issues, 1):
                logger.warning("  Unfixed issue %d: %s", i, issue)
        else:
            # Some issues remain but didn't get worse
            self._validation_issues = list(issues_after)
            if issues_after:
                logger.warning("Phase 3: %d issues remain after fixes:", len(issues_after))
                for i, issue in enumerate(issues_after, 1):
                    logger.warning("  Remaining %d: %s", i, issue)

    def _collect_validation_issues(self) -> list[str]:
        """Collect all validation issues from the output directory."""
        issues = []

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
                        issues.append(f"Syntax error in {rel} line {e.lineno}: {e.msg}")

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
                                issues.append(f"{rel} references package.json but it doesn't exist")
                        if "requirements.txt" in content:
                            req = os.path.join(docker_dir, "requirements.txt")
                            if not os.path.isfile(req):
                                issues.append(f"{rel} references requirements.txt but it doesn't exist")
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
                                issues.append(f"Dependency conflict in {rel}:\n{err}")
                    except Exception:
                        pass  # pip not available or timeout — skip

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
    # Delegating methods (backward compatibility)
    # ==================================================================

    def _analyze_gaps(self, instructions: str) -> list[str]:
        """Delegate to gap_analyzer module."""
        return analyze_gaps(
            instructions=instructions,
            generator_used=self._generator_used,
            domain_model=self.domain_model,
            llm_client=self.client,
        )

    def _analyze_gaps_with_llm(self, instructions: str) -> list[str] | None:
        """Delegate to gap_analyzer module."""
        return _analyze_gaps_with_llm(
            instructions=instructions,
            generator_used=self._generator_used or "",
            domain_model=self.domain_model,
            llm_client=self.client,
        )

    def _analyze_gaps_keyword(self, instructions: str) -> list[str]:
        """Delegate to gap_analyzer module."""
        return _analyze_gaps_keyword(
            instructions=instructions,
            generator_used=self._generator_used or "",
        )

    def _build_model_summary(self) -> str:
        """Delegate to gap_analyzer module."""
        return build_model_summary(self.domain_model)

    def _build_system_prompt(self, gap_tasks: list[str]) -> str:
        """Delegate to prompt_builder module."""
        return build_system_prompt(
            domain_model=self.domain_model,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            inventory=self._inventory,
            gap_tasks=gap_tasks,
            max_turns=self.max_turns,
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
            "validation_issues": self._validation_issues,
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
