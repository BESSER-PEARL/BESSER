"""
System prompt construction for the LLM orchestrator.

Builds the system prompt that guides the LLM's behavior during Phase 2,
including domain model context, file inventory, and scoped task lists.
"""

import json
import logging
import os
from typing import Any

from besser.generators.llm.model_serializer import (
    serialize_agent_model,
    serialize_domain_model,
    serialize_gui_model,
)

logger = logging.getLogger(__name__)


def build_system_prompt(
    domain_model,
    gui_model,
    agent_model,
    inventory: str,
    gap_tasks: list[str],
    max_turns: int,
) -> str:
    """
    Build the system prompt with domain model, inventory, and scoped tasks.

    Args:
        domain_model: The BUML domain model.
        gui_model: Optional GUI model.
        agent_model: Optional agent model.
        inventory: Description of what the generator already produced.
        gap_tasks: List of specific tasks for the LLM.
        max_turns: Maximum agent loop iterations (for budget hints).

    Returns:
        The full system prompt string.
    """
    model_json = json.dumps(serialize_domain_model(domain_model), indent=2)

    # Build the task list
    if gap_tasks:
        task_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(gap_tasks))
    else:
        task_list = "  No specific customizations detected. Review the output and add any finishing touches."

    # Build inventory section
    inventory_section = ""
    if inventory:
        inventory_section = f"\n## What was already generated\n\n{inventory}\n"

    return f"""\
You are an expert full-stack developer. You make targeted, scoped changes to code.

Read the generated code before changing it and keep changes tightly scoped to the tasks below.
Do not rewrite generated files from scratch — make surgical modifications.

## Domain Model

```json
{model_json}
```
{inventory_section}
## Your tasks

The generator handled the base app (CRUD, ORM, schemas, pages). Your job is to \
implement ONLY the following customizations:

{task_list}

## Rules

1. **Keep changes scoped.** Only implement the tasks above. Don't rewrite generated files.
2. **Use modify_file** for changes to existing files. Use write_file only for NEW files.
3. **Model is truth.** Never invent entities not in the model.
4. **Read before modify.** Read the relevant section of a file before editing it. \
Use offset/limit for large files (>200 lines).
5. **Be efficient.** One read per file. Write complete files in one call. \
Max 2-3 turns for testing. Budget turns for Docker + README at the end.
6. **Don't re-read files** you already read. Remember what's in them.

When done, briefly summarize what you changed.
"""


def build_inventory(output_dir: str, domain_model, generator_name: str) -> str:
    """
    Describe what the generator produced -- for the LLM's context.

    Args:
        output_dir: Path to the output directory.
        domain_model: The BUML domain model.
        generator_name: Name of the generator that was used.

    Returns:
        A human-readable inventory string.
    """
    # List files
    files = []
    for root, _, filenames in os.walk(output_dir):
        for f in filenames:
            rel = os.path.relpath(os.path.join(root, f), output_dir)
            size = os.path.getsize(os.path.join(root, f))
            files.append(f"{rel.replace(chr(92), '/')} ({size:,} bytes)")

    class_names = [c.name for c in domain_model.get_classes()]
    enum_names = [e.name for e in domain_model.get_enumerations()]

    lines = [f"Generator `{generator_name}` produced {len(files)} files:"]
    for f in sorted(files)[:30]:
        lines.append(f"  - {f}")
    if len(files) > 30:
        lines.append(f"  ... and {len(files) - 30} more files")

    lines.append(f"\nEntities with full CRUD: {', '.join(class_names)}")
    if enum_names:
        lines.append(f"Enumerations: {', '.join(enum_names)}")
    lines.append(f"Associations: {len(domain_model.associations)} relationships")

    if generator_name == "generate_web_app":
        lines.append("\nBackend: FastAPI with SQLAlchemy ORM + Pydantic schemas")
        lines.append("Frontend: React + TypeScript + Vite + Tailwind CSS")
        lines.append("Docker: docker-compose.yml + Dockerfiles")
        # List frontend pages
        pages = []
        for root, _, filenames in os.walk(output_dir):
            for f in filenames:
                if "/pages/" in os.path.join(root, f).replace("\\", "/") and f.endswith((".tsx", ".jsx")):
                    pages.append(f.replace(".tsx", "").replace(".jsx", ""))
        if pages:
            lines.append(f"Frontend pages: {', '.join(sorted(pages))}")
    elif generator_name == "generate_fastapi_backend":
        lines.append("\nBackend: FastAPI with SQLAlchemy ORM + Pydantic schemas")
        lines.append("Files: main_api.py, sql_alchemy.py, pydantic_classes.py, requirements.txt")

    return "\n".join(lines)
