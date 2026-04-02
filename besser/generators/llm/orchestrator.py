"""
LLM generation orchestrator — the agent loop.

Manages the multi-turn conversation between Claude and the tool executor:

1. Serialize the domain model as structured context for the LLM
2. Build a system prompt explaining the model, available tools, and strategy
3. Send user instructions
4. Claude calls tools (generators, file ops, shell commands) in a loop
5. Execute each tool, feed results back
6. Claude iterates: generate → test → see errors → fix → test again
7. Repeat until Claude signals completion or max turns reached
8. Save a generation recipe (.besser_recipe.json) for audit/replay
"""

import json
import logging
import os
import tempfile
import time
from typing import Any, Callable

from besser.generators.llm.llm_client import ClaudeLLMClient
from besser.generators.llm.model_serializer import (
    serialize_agent_model,
    serialize_domain_model,
    serialize_gui_model,
)
from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.tools import get_all_tools

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Orchestrates LLM-augmented code generation.

    The orchestrator runs an agent loop where Claude:
    1. Calls BESSER generators for components where one exists
    2. Writes code from scratch for components where none exists
    3. Runs the code to test it
    4. Reads errors and fixes them
    5. Repeats until the code works

    This generate → test → fix loop is what makes the agent reliable.

    Args:
        llm_client: Configured Claude client.
        domain_model: The BUML domain model.
        gui_model: Optional GUI model.
        agent_model: Optional agent model.
        agent_config: Optional agent config dict.
        output_dir: Directory for generated files (temp dir if None).
        max_turns: Safety limit on agent loop iterations.
        on_progress: Optional callback(turn, tool_name, status) for UI integration.
    """

    MAX_TURNS = 50

    def __init__(
        self,
        llm_client: ClaudeLLMClient,
        domain_model,
        gui_model=None,
        agent_model=None,
        agent_config: dict | None = None,
        output_dir: str | None = None,
        max_turns: int | None = None,
        on_progress: Callable[[int, str, str], None] | None = None,
    ):
        self.client = llm_client
        self.domain_model = domain_model
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="besser_llm_")
        self.max_turns = max_turns or self.MAX_TURNS
        self.on_progress = on_progress
        self.executor = ToolExecutor(
            workspace=self.output_dir,
            domain_model=domain_model,
            gui_model=gui_model,
            agent_model=agent_model,
            agent_config=agent_config,
        )
        self.tools = get_all_tools()
        self.tool_calls_log: list[dict] = []
        self.total_turns = 0
        self._recent_tool_calls: list[str] = []  # for loop detection

    # Maximum consecutive calls to the same tool before warning the LLM
    _LOOP_THRESHOLD = 4

    def run(self, instructions: str) -> str:
        """
        Run the generation loop.

        Args:
            instructions: Natural language description of what to build.

        Returns:
            Absolute path to the output directory with generated files.
        """
        if not instructions or not instructions.strip():
            raise ValueError("Instructions cannot be empty")

        start_time = time.monotonic()
        system = self._build_system_prompt()
        messages: list[dict] = [{"role": "user", "content": instructions}]

        for turn in range(self.max_turns):
            self.total_turns = turn + 1
            logger.info("LLM generation turn %d/%d", turn + 1, self.max_turns)

            try:
                response = self.client.chat(
                    system=system,
                    messages=messages,
                    tools=self.tools,
                )
            except Exception as e:
                logger.error("LLM API call failed on turn %d: %s", turn + 1, e)
                break

            # LLM is done
            if response["stop_reason"] == "end_turn":
                logger.info("LLM generation completed after %d turns", turn + 1)
                break

            # LLM wants to call tools
            if response["stop_reason"] == "tool_use":
                messages.append({"role": "assistant", "content": response["content"]})

                tool_results = []
                for block in response["content"]:
                    if not (hasattr(block, "type") and block.type == "tool_use"):
                        continue

                    tool_name = block.name
                    logger.info("Executing tool: %s", tool_name)

                    # Loop detection: if same WRITE tool called too many times
                    # in a row, warn the LLM. Read-only tools (read_file,
                    # list_files, search_in_files, check_syntax) are excluded
                    # because reading multiple files sequentially is normal.
                    if tool_name not in ("read_file", "list_files", "search_in_files", "check_syntax"):
                        self._recent_tool_calls.append(tool_name)
                    if self._is_stuck():
                        logger.warning("Possible loop: %s called %d times consecutively",
                                       tool_name, self._LOOP_THRESHOLD)

                    if self.on_progress:
                        self.on_progress(turn + 1, tool_name, "executing")

                    result = self.executor.execute(tool_name, block.input)

                    # If stuck in a loop, prepend a warning to the result
                    if self._is_stuck():
                        result = json.dumps({
                            "warning": (
                                f"You have called '{tool_name}' {self._LOOP_THRESHOLD} times "
                                "in a row. You may be stuck. Try a different approach, "
                                "or if the task is done, stop and summarize."
                            ),
                            "result": json.loads(result),
                        })

                    # Log for recipe/audit
                    self.tool_calls_log.append({
                        "turn": turn + 1,
                        "tool": tool_name,
                        "input": _sanitize_for_log(block.input),
                        "success": '"error"' not in result[:100],
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                logger.warning("Unexpected stop_reason: %s", response["stop_reason"])
                break

        elapsed = time.monotonic() - start_time
        logger.info(
            "LLM generation finished: %d turns, %.1fs, %d tool calls",
            self.total_turns, elapsed, len(self.tool_calls_log),
        )

        self._save_recipe(instructions, elapsed)
        return self.output_dir

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def _is_stuck(self) -> bool:
        """Check if the LLM is calling the same tool repeatedly."""
        recent = self._recent_tool_calls[-self._LOOP_THRESHOLD:]
        return (
            len(recent) >= self._LOOP_THRESHOLD
            and len(set(recent)) == 1
        )

    # ------------------------------------------------------------------
    # System prompt — the most important piece
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt with model context and strategy."""
        model_json = json.dumps(serialize_domain_model(self.domain_model), indent=2)

        gui_section = ""
        if self.gui_model:
            gui_data = serialize_gui_model(self.gui_model)
            if gui_data:
                gui_section = (
                    "\n## GUI Model\n"
                    "The user designed a GUI with these screens and components:\n"
                    f"```json\n{json.dumps(gui_data, indent=2)}\n```\n"
                )

        agent_section = ""
        if self.agent_model:
            agent_data = serialize_agent_model(self.agent_model)
            if agent_data:
                agent_section = (
                    "\n## Agent Model\n"
                    "The user designed a conversational agent:\n"
                    f"```json\n{json.dumps(agent_data, indent=2)}\n```\n"
                )

        return f"""\
You are an expert full-stack developer. You build production-ready applications \
from domain models using the BESSER low-code platform.

You have access to code generators, file operations, and shell commands. \
You can generate code, run it, see errors, and fix them — just like a real developer.

## Domain Model

This is the user's validated data model. Every class, attribute, and relationship \
is correct and intentional.

```json
{model_json}
```
{gui_section}{agent_section}
## How to work

### Phase 1: Generate base code
For each component the user wants:
- **If a BESSER generator exists** (FastAPI, Django, React, Pydantic, SQLAlchemy, etc.): \
call it first. Generator output is tested and reliable — it's your foundation.
- **If no generator exists** (NestJS, Next.js, Express, Spring Boot, etc.): \
write the code from scratch using the domain model above as your specification. \
Every entity, attribute, type, and relationship in your code must match the model.

### Phase 2: Customize
Read the generated files, then modify them to add what the user asked for \
(authentication, pagination, theming, deployment, etc.). Use `modify_file` for \
targeted edits — don't rewrite entire files.

### Phase 3: Verify
- Run `check_syntax` on Python files after modifying them.
- Use `run_command` to test the code: `python -c "import main_api"`, \
`python -m py_compile file.py`, `npm run build`, etc.
- If a command fails, **read the error output carefully**, fix the issue, and try again.
- Install dependencies with `install_dependencies` if needed.

### Phase 4: Finalize
- Write a README.md with clear setup and run instructions.
- List all dependencies (requirements.txt, package.json).
- If relevant, write a Dockerfile and/or docker-compose.yml.

## Rules

1. **Model is truth**: Never invent entities or attributes not in the model. \
If the user needs something not in the model, tell them.
2. **Generators first**: Always prefer calling a generator over writing from scratch.
3. **Read before modify**: Always `read_file` before `modify_file`.
4. **Test your work**: After generating and modifying code, run it to verify. \
The generate → test → fix loop is what makes your output reliable.
5. **Fix errors**: If a command or syntax check fails, read the error, fix the code, \
and verify again. Don't leave broken code.
6. **Be precise with modify_file**: The `old_text` must match exactly including \
whitespace and indentation. Use `read_file` or `search_in_files` to find the exact text.

When done, briefly summarize what you built and how to run it.
"""

    # ------------------------------------------------------------------
    # Recipe logging
    # ------------------------------------------------------------------

    def _save_recipe(self, instructions: str, elapsed: float) -> None:
        """Save the generation recipe for audit/replay."""
        recipe = {
            "instructions": instructions,
            "model_name": self.domain_model.name,
            "llm_model": self.client.model,
            "turns": self.total_turns,
            "tool_calls_count": len(self.tool_calls_log),
            "elapsed_seconds": round(elapsed, 1),
            "tool_calls": self.tool_calls_log,
        }
        recipe_path = os.path.join(self.output_dir, ".besser_recipe.json")
        try:
            with open(recipe_path, "w", encoding="utf-8") as f:
                json.dump(recipe, f, indent=2, default=str)
        except Exception:
            pass  # non-critical


def _sanitize_for_log(data: Any) -> Any:
    """Truncate large values in tool inputs for the recipe log."""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 500:
                result[k] = v[:500] + "... [truncated]"
            else:
                result[k] = v
        return result
    return data
