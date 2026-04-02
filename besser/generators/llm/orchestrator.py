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
        self._write_skills_guide()
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
    # Skills guide — the CLAUDE.md equivalent for generated code
    # ------------------------------------------------------------------

    def _write_skills_guide(self) -> None:
        """Write .besser_guide.md into the workspace for the LLM to read."""
        guide = """\
# BESSER Generator Guide

Read this BEFORE modifying any generated code.

## Golden Rule

**NEVER overwrite generated files with write_file.** Use modify_file for all changes.
The write_file tool will BLOCK you from overwriting generator output.
Use write_file ONLY for NEW files (auth.py, Dockerfile, README, configs).

## What each generator produces

### generate_fastapi_backend
Creates `backend/` with:
- `main_api.py` — FastAPI app with CRUD endpoints for every model entity
  - App created as: `app = FastAPI(...)`
  - Each entity has: GET list, GET by id, POST, PUT, DELETE
  - Endpoints use SQLAlchemy session via `get_db()` dependency
  - Structure: imports → app → get_db → endpoints
- `sql_alchemy.py` — SQLAlchemy ORM models
  - `Base = declarative_base()`
  - One class per entity with Column definitions
  - Relationships defined with `relationship()`
  - `engine = create_engine(...)` at the bottom
- `pydantic_classes.py` — Pydantic request/response schemas
  - One BaseModel per entity
  - Type hints matching the domain model
- `requirements.txt` — pip dependencies

**How to modify main_api.py:**
- Add auth: create NEW auth.py, then modify_file to add `Depends(get_current_user)` to endpoint params
- Add pagination: find `def get_<entity>` endpoints, add `skip: int = 0, limit: int = 20` params
- Add new endpoint: find the last `@app` decorated function, insert your new endpoint after it
- Change DB: find `create_engine(` line, change the connection string

**How to modify sql_alchemy.py:**
- Add index: find `Column(` definition, add `index=True` param
- Add constraint: add `__table_args__` to the class
- Change DB: find `create_engine(` and change the URL

**How to modify pydantic_classes.py:**
- Add validator: find the class, add `@field_validator('field_name')` method
- Add new field: insert a new line in the class body

### generate_pydantic
Creates `pydantic/pydantic_classes.py` — same structure as above but standalone.

### generate_sqlalchemy
Creates `sqlalchemy/sql_alchemy.py` — same structure as above but standalone.

### generate_django
Creates `django/` with a full Django project:
- `manage.py`, `settings.py`, `urls.py`
- `models.py` — Django models for each entity
- `views.py`, `admin.py`
How to add DRF: create NEW `serializers.py` + `viewsets.py`, modify `urls.py` to add router.

### generate_python_classes
Creates `python/classes.py` — plain Python classes with getters/setters.

### generate_react
Creates `react/` with full React TypeScript app. REQUIRES GUI model.

### generate_web_app
Creates `web_app/` with frontend/ + backend/ + docker-compose.yml. REQUIRES GUI model.

## Common patterns

### Adding authentication to FastAPI
1. `write_file("backend/auth.py", ...)` — NEW file with JWT logic
2. `modify_file("backend/main_api.py", ...)` — add `from auth import ...` to imports
3. `modify_file("backend/main_api.py", ...)` — add auth dependency to protected endpoints
4. `modify_file("backend/requirements.txt", ...)` — add python-jose, passlib

### Adding pagination to FastAPI
1. `modify_file("backend/main_api.py", old_text="def get_users(", new_text="def get_users(skip: int = 0, limit: int = 20, ")`
2. `modify_file("backend/main_api.py", old_text=".all()", new_text=".offset(skip).limit(limit).all()")`

### Changing database from SQLite to PostgreSQL
1. `modify_file("backend/sql_alchemy.py", old_text="sqlite:///", new_text="postgresql://user:pass@localhost:5432/dbname")`
2. `modify_file("backend/requirements.txt", ...)` — add psycopg2-binary
"""
        guide_path = os.path.join(self.output_dir, ".besser_guide.md")
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(guide)

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

You have code generators, file operations, and shell commands as tools.

## Domain Model

Every class, attribute, and relationship below is the user's validated specification.

```json
{model_json}
```
{gui_section}{agent_section}
## Skills Guide

A file `.besser_guide.md` is in your workspace. **Read it first** with `read_file` \
before modifying any generated code. It explains the structure of each generator's \
output and shows exactly how to make common modifications (auth, pagination, DB changes).

## CRITICAL RULE: Preserve generator output

BESSER's generators produce tested, reliable code. When you call a generator:

1. **DO NOT rewrite generated files with write_file.** The generator output is your \
foundation. Treat it like code written by a senior colleague — modify it surgically, \
don't throw it away.

2. **Use modify_file for ALL changes to generated files.** Find the exact section that \
needs changing and replace just that section. For example, to add a new endpoint, \
find the last endpoint in main_api.py and insert after it — don't rewrite the whole file.

3. **Only use write_file for NEW files** that no generator produced: auth modules, \
Dockerfiles, CI configs, README, .env files, etc.

## How to work

### Step 1: Generate
Call the appropriate generator(s). If no generator exists for what the user wants \
(e.g., NestJS, Next.js), write from scratch using the model as spec.

### Step 2: Read
Read the generated files to understand the structure, imports, and patterns.

### Step 3: Modify (not rewrite!)
Use `modify_file` to make targeted additions:
- To add auth: find the app initialization, insert middleware after it
- To add pagination: find a GET endpoint, add limit/offset parameters to it
- To change the database: find the create_engine call, change the connection string

### Step 4: Add new files
Use `write_file` ONLY for files that don't exist yet: auth.py, Dockerfile, README.md, etc.

### Step 5: Verify
- `check_syntax` on modified Python files
- `run_command` to test imports and basic functionality
- `install_dependencies` if needed
- If something fails, read the error, fix with `modify_file`, try again

## Rules

1. **NEVER rewrite a generated file.** Use modify_file to change it surgically. \
If you find yourself wanting to write_file on a file that a generator created, \
stop and use modify_file instead.
2. **Model is truth.** Never invent entities not in the model.
3. **Generators first.** Always call a generator before writing code from scratch.
4. **Test your work.** The generate → test → fix loop makes output reliable.
5. **Be precise.** modify_file's old_text must match exactly (whitespace matters).

## Efficiency

You have a limited number of turns. Be efficient:
- Call multiple tools in the same turn when they're independent
- Write complete files in one write_file call (don't write partial files and modify later)
- For the frontend: write each file completely in one go
- Don't over-test: one import check + one basic validation is enough
- Finish with Docker + README — don't run out of turns before those

When done, briefly summarize what you built.
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
