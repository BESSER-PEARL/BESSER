"""LLM-based gap analysis for the orchestrator.

A single planning LLM call that compares the user's request against the
deterministic generator's output and produces a focused task list for
the customise (Phase 2) loop to execute.

Model selection:
    The call uses ``llm_client``, which is the *same* client (and
    therefore the same provider+model) the orchestrator was constructed
    with — i.e. whatever the user picked in the BYOK dialog. There is no
    separate "cheap" model. Cost scales with the user's chosen tier.

Why keep this when Phase 2 has a frontier model?
    * It anchors Phase 2 on every model class, attribute, relationship,
      and explicit user requirement instead of relying on the LLM to
      re-derive the to-do list from free-text every turn.
    * An explicit checklist makes Phase 2 more likely to ship every
      requested feature instead of stopping at "good enough".
    * It can flag dead files left over from the deterministic generator
      so the customise loop deletes them via the ``delete_file`` tool
      (e.g. a leftover ``main_api.py`` after a switch to Flask).
    * Failure mode is graceful: no list = the smart Phase 2 model still
      plans, just without the upfront scaffold.

Inputs are passed *verbatim* — we no longer truncate the user's
instructions or the file inventory to 1500 chars, since the planner has
to see the whole picture to find genuine gaps. The serialized domain
model (relationships, multiplicities, inheritance, constraints) is fed
in as JSON via :func:`besser.generators.llm.model_serializer.serialize_domain_model`.
"""

import json
import logging
from typing import Any, Callable

from besser.generators.llm.model_serializer import serialize_domain_model

logger = logging.getLogger(__name__)


# Soft budgets — generous, but a runaway 100k-line inventory would burn
# the whole context window before the customise phase gets a chance.
_MAX_INSTRUCTIONS_CHARS = 8_000
_MAX_INVENTORY_CHARS = 8_000
_MAX_MODEL_JSON_CHARS = 12_000
_MAX_TASKS = 16


def analyze_gaps_via_llm(
    instructions: str,
    generator_used: str | None,
    domain_model,
    inventory: str,
    llm_client,
    on_progress: Callable[[int, str, str], None] | None = None,
) -> list[str]:
    """Return a focused task list for Phase 2.

    Args:
        instructions: Raw user request, untruncated.
        generator_used: Name of the deterministic generator that ran in
            Phase 1, or ``None`` if no generator was selected.
        domain_model: The assembled BUML ``DomainModel`` (or ``None``
            for non-class primaries like agent/state-machine runs).
        inventory: Output-tree inventory string from
            ``prompt_builder.build_inventory``.
        llm_client: The orchestrator's primary LLM client. **Same model
            the user selected** — no second cheap model is constructed.
        on_progress: Optional callback so the orchestrator can fire a
            ``gap`` SSE phase event before the LLM call. Signature
            mirrors the orchestrator's existing ``on_progress``:
            ``(turn, tool, status)``.

    Returns:
        - A list of task strings on success.
        - ``[]`` if the LLM call fails or no work is needed.
        - A 1-element "build everything from scratch" list when no
          generator ran.
    """
    if on_progress is not None:
        try:
            on_progress(0, "gap_analysis", "analyzing")
        except Exception:
            # Progress callback must never break the analysis.
            logger.debug("on_progress callback raised; continuing", exc_info=True)

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

    model_json = _safe_serialize_model(domain_model)
    user_prompt = _build_user_prompt(
        instructions=instructions,
        generator_used=generator_used,
        model_json=model_json,
        inventory=inventory,
    )

    try:
        response = llm_client.chat(
            system=_SYSTEM_PROMPT,
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
    return [t.strip() for t in tasks[:_MAX_TASKS] if isinstance(t, str) and t.strip()]


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a senior engineer scoping a customisation pass on top of a "
    "deterministically generated codebase. You have the full domain model "
    "(JSON), the file inventory the generator produced, and the user's "
    "request. Your job is to produce a tight, actionable task list for the "
    "next agent — who has read_file/write_file/modify_file/delete_file/"
    "search_in_files/check_syntax tools — to execute.\n\n"
    "Task-list rules:\n"
    "  * Each task is one line, imperative, scoped to a single file or a "
    "small coherent change.\n"
    "  * Anchor every task to a SPECIFIC model element (class, attribute, "
    "relationship, OCL constraint) or a SPECIFIC line in the user's request. "
    "If you can't tie a task to one of those, drop it.\n"
    "  * If the deterministic generator left files that no longer fit the "
    "customised stack (e.g. FastAPI files when the user asked for Flask, "
    "Pydantic schemas when you'll use Marshmallow), include an explicit "
    "'delete X' task. The next agent has a delete_file tool.\n"
    "  * Skip anything the generator already provided correctly.\n"
    "  * Skip nice-to-haves (auth, tests, Docker) unless the user asked.\n"
    "  * If the user named a target framework or language, every task must "
    "respect it — do not propose tasks for the generator's default stack.\n"
    "  * Hard cap: 16 tasks. Prioritise correctness over completeness.\n\n"
    "Return ONLY a JSON array of task strings. No prose, no markdown."
)


def _build_user_prompt(
    instructions: str,
    generator_used: str,
    model_json: str,
    inventory: str,
) -> str:
    instructions_clipped = _clip(instructions, _MAX_INSTRUCTIONS_CHARS, "instructions")
    inventory_clipped = _clip(inventory, _MAX_INVENTORY_CHARS, "inventory")
    return (
        f"USER REQUEST:\n{instructions_clipped}\n\n"
        f"DETERMINISTIC GENERATOR THAT RAN: {generator_used}\n\n"
        f"DOMAIN MODEL (JSON):\n{model_json}\n\n"
        f"FILE INVENTORY (paths + sizes):\n{inventory_clipped}\n\n"
        "List the missing or incorrect work as a JSON array of short task "
        "strings (max 16). Tie every task to either a specific model element "
        "or a specific user requirement. Include explicit delete tasks for "
        "any generator-output files that no longer fit the customised stack."
    )


def _clip(text: str, limit: int, label: str) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n…[{label} truncated at {limit} chars]"


def _safe_serialize_model(domain_model) -> str:
    """Best-effort JSON dump of the domain model.

    Returns ``"null"`` when there is no model (agent/state-machine runs)
    and an error sentinel string when serialisation fails — never
    raises, since a failed gap analysis is recoverable but a crashed
    one isn't.
    """
    if domain_model is None:
        return "null"
    try:
        data = serialize_domain_model(domain_model)
    except Exception:
        logger.warning("Gap analyzer: serialize_domain_model failed", exc_info=True)
        # Fall back to bare class names — better than nothing.
        try:
            names = [c.name for c in domain_model.get_classes()]
            return json.dumps({"classes": [{"name": n} for n in names]})
        except Exception:
            return "{}"
    payload = json.dumps(data, default=str, separators=(",", ":"))
    if len(payload) <= _MAX_MODEL_JSON_CHARS:
        return payload
    # Pretty-print is more compressible by the LLM than raw truncation,
    # but if we're over budget the safest move is to drop the
    # ``inherited_*`` keys (the bulkiest part) and re-serialise.
    pruned: dict[str, Any] = dict(data)
    for cls in pruned.get("classes", []):
        cls.pop("inherited_attributes", None)
        cls.pop("inherited_methods", None)
    payload = json.dumps(pruned, default=str, separators=(",", ":"))
    if len(payload) <= _MAX_MODEL_JSON_CHARS:
        return payload
    # Last resort — raw clip with an explicit marker so the LLM knows
    # the JSON is partial.
    return payload[:_MAX_MODEL_JSON_CHARS] + '..."__truncated__"'


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
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
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
