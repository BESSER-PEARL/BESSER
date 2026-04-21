"""LLM-based gap analysis for the orchestrator.

A single cheap LLM call ("Haiku-class") that compares the user's request
against the generator output inventory and produces a focused task list
for the Phase 2 LLM to execute.

This is intentionally minimal — no keyword regex, no fallback rules.
If the LLM call fails, the orchestrator skips the checklist and lets
Phase 2 plan its own work from the user instructions alone.

Why keep this when Phase 2 has a frontier model?
  - It costs ~$0.01 per run; Phase 2 burning a turn or two to plan its
    own to-do list is wasteful in comparison.
  - An explicit checklist makes Phase 2 more likely to ship every
    requested feature instead of stopping at "good enough".
  - Failure mode is graceful: no list = the smart Phase 2 model still
    plans, just without the upfront scaffold.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def analyze_gaps_via_llm(
    instructions: str,
    generator_used: str | None,
    domain_model,
    inventory: str,
    llm_client,
) -> list[str]:
    """Return a focused task list for Phase 2.

    Returns:
      - A list of task strings on success.
      - An empty list if the LLM call fails or no work is needed
        (caller should skip the checklist section in that case).
      - A 1-element "build everything from scratch" list if no
        generator was used in Phase 1.
    """
    if not generator_used:
        return [
            "No BESSER generator was used. Build the entire application "
            "from scratch using the domain model as your specification. "
            "Every entity, attribute, type, and relationship in your code "
            "must match the model."
        ]

    if not _is_real_provider(llm_client):
        # Test client / mock — skip the LLM call.
        return []

    classes = [c.name for c in domain_model.get_classes()] if domain_model else []
    summary = (
        f"Generator: {generator_used}. "
        f"Classes: {', '.join(classes[:20])}. "
        f"Output inventory: {inventory[:1500]}"
    )

    user_prompt = (
        f"USER REQUEST:\n{instructions[:1500]}\n\n"
        f"GENERATOR CONTEXT:\n{summary}\n\n"
        "The generator produced the inventory above (CRUD endpoints, ORM, "
        "schemas, basic pages). List ONLY what's still missing as a JSON "
        "array of short task strings (1 line each, max 12 tasks). Skip "
        "anything the generator already provided. Return ONLY the JSON "
        "array — no prose, no markdown fences."
    )
    system_prompt = (
        "You are a senior engineer scoping a refactor. Return ONLY a JSON "
        "array of task strings. No explanation, no markdown."
    )

    try:
        response = llm_client.chat(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[],
        )
    except Exception:
        logger.warning("Gap analyzer LLM call failed; falling back to no checklist")
        return []

    text = _extract_text(response)
    tasks = _parse_task_array(text)
    if tasks is None:
        logger.warning("Gap analyzer returned non-JSON response: %r", text[:200])
        return []
    # Sanity bounds — if the model returns 50 tasks, something is off.
    return [t.strip() for t in tasks[:12] if isinstance(t, str) and t.strip()]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _is_real_provider(llm_client) -> bool:
    """Mock clients in tests don't have ``_client``. Skip the LLM call
    in that case so unit tests don't try to hit the network."""
    return hasattr(llm_client, "_client")


def _extract_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in response.get("content", []):
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def _parse_task_array(text: str) -> list | None:
    """Best-effort parse of a JSON array from a model response.

    Handles raw arrays, arrays wrapped in ```json fences, and trailing
    prose by extracting the first ``[...]`` block.
    """
    if not text:
        return None
    # Strip common markdown fences.
    cleaned = text
    if cleaned.startswith("```"):
        # Drop opening fence (with or without ``json``).
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    # Slice to the first balanced bracket pair if there's surrounding prose.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None
