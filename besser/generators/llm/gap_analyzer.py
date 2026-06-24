"""LLM-based gap analysis for the orchestrator.

A single planning LLM call that compares the user's request against the
deterministic generator's output and produces a focused task list for
the customise (Phase 2) loop to execute.

Model selection:
    The call prefers the provider's cheap planning sibling
    (``llm_client.planning_model`` — Haiku for Anthropic keys,
    gpt-4o-mini for OpenAI keys, overridable via the
    ``BESSER_LLM_PLANNING_MODEL`` env var, ``primary`` to disable). If
    the cheap model is unavailable on the user's gateway the call is
    retried once on the primary model, so cheap routing can never fail
    a run. Cost is billed at the effective model's own pricing.

Output contract (load-bearing — the orchestrator branches on it):
    * ``None``  — the analysis FAILED (LLM error, unparseable reply,
      mock client). Phase 2 proceeds without a checklist, exactly as
      before.
    * ``[]``    — the model judged the scaffold ALREADY COVERS the
      user's request. The orchestrator may skip Phase 2 entirely.
    * ``[...]`` — a focused task list for Phase 2.

Why keep this when Phase 2 has a frontier model?
    * It anchors Phase 2 on every model class, attribute, relationship,
      and explicit user requirement instead of relying on the LLM to
      re-derive the to-do list from free-text every turn.
    * An explicit checklist makes Phase 2 more likely to ship every
      requested feature instead of stopping at "good enough".
    * It can flag dead files left over from the deterministic generator
      so the customise loop deletes them via the ``delete_file`` tool
      (e.g. a leftover ``main_api.py`` after a switch to Flask).

Inputs are soft-clipped to the ``_MAX_*_CHARS`` budgets below (a
runaway 100k-line inventory would otherwise burn the context window
before the customise phase gets a chance). The serialized domain model
is pruned progressively but is ALWAYS valid JSON — the planner never
receives a raw character clip.
"""

import inspect
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


# Forced-tool schema for the planning reply. Structured-by-construction
# beats parsing free text: prose with brackets can no longer corrupt
# extraction, and "no work needed" is an explicit empty array instead
# of an accident.
_SUBMIT_TASKS_TOOL = {
    "name": "submit_tasks",
    "description": (
        "Submit the final task list for the customisation pass. "
        "Submit an EMPTY list when the generated scaffold already "
        "fully covers the user's request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": _MAX_TASKS,
                "description": (
                    "One-line imperative tasks, each anchored to a "
                    "specific model element or user requirement."
                ),
            },
        },
        "required": ["tasks"],
    },
}


def analyze_gaps_via_llm(
    instructions: str,
    generator_used: str | None,
    domain_model,
    inventory: str,
    llm_client,
    on_progress: Callable[[int, str, str], None] | None = None,
    on_phase_details: Callable[[str, str], None] | None = None,
    generator_failure: str | None = None,
) -> list[str] | None:
    """Return a focused task list for Phase 2.

    Args:
        instructions: Raw user request.
        generator_used: Name of the deterministic generator that ran in
            Phase 1, or ``None`` if no generator ran (none selected, or
            the selected one failed — see ``generator_failure``).
        domain_model: The assembled BUML ``DomainModel`` (or ``None``
            for non-class primaries like agent/state-machine runs).
        inventory: Output-tree inventory string from
            ``prompt_builder.build_inventory``.
        llm_client: The orchestrator's LLM client. The call is routed
            to ``llm_client.planning_model`` when available.
        on_progress: Optional callback so the orchestrator can fire a
            ``gap`` SSE phase event before the LLM call. Signature
            mirrors the orchestrator's existing ``on_progress``:
            ``(turn, tool, status)``.
        on_phase_details: Optional callback surfacing the task list to
            the SSE consumer.
        generator_failure: When Phase 1 selected a generator but it
            FAILED, the reason string (``"<generator>: <error>"``). It
            is woven into the from-scratch fallback task so Phase 2
            knows why the scaffold is missing and can avoid the cause.

    Returns:
        ``None`` on failure, ``[]`` when no work is needed, otherwise
        the task list (see the module docstring for the full contract).
    """
    if on_progress is not None:
        try:
            on_progress(0, "gap_analysis", "analyzing")
        except Exception:
            # Progress callback must never break the analysis.
            logger.debug("on_progress callback raised; continuing", exc_info=True)

    if not generator_used:
        if generator_failure:
            fallback = [
                f"The deterministic generator failed ({generator_failure}). "
                "Build the entire application from scratch using the domain "
                "model as your specification, and avoid the cause of that "
                "failure (e.g. rename reserved identifiers in the code you "
                "write). Every entity, attribute, type, and relationship in "
                "your code must match the model."
            ]
        else:
            fallback = [
                "No BESSER generator was used. Build the entire application "
                "from scratch using the domain model as your specification. "
                "Every entity, attribute, type, and relationship in your code "
                "must match the model."
            ]
        _emit_phase_details(
            on_phase_details, fallback,
            fallback_label=True, generator_failure=generator_failure,
        )
        return fallback

    if not _is_real_provider(llm_client):
        # Test client / mock — skip the LLM call. Counts as "no
        # analysis", not as "scaffold is sufficient".
        return None

    model_json = _safe_serialize_model(domain_model)
    user_prompt = _build_user_prompt(
        instructions=instructions,
        generator_used=generator_used,
        model_json=model_json,
        inventory=inventory,
    )

    tasks = _call_planner(llm_client, user_prompt)
    if tasks is None:
        return None
    cleaned = [t.strip() for t in tasks[:_MAX_TASKS] if isinstance(t, str) and t.strip()]
    _emit_phase_details(on_phase_details, cleaned)
    return cleaned


def _call_planner(llm_client, user_prompt: str) -> list | None:
    """Run the planning call: cheap model first, primary on error, one
    repair retry on an unparseable reply. Returns the raw task list or
    ``None`` on failure."""
    structured = _chat_supports_kwargs(llm_client, "force_tool", "model_override")
    planning_model = getattr(llm_client, "planning_model", None) if structured else None

    def _chat(prompt: str, model_override: str | None):
        messages = [{"role": "user", "content": prompt}]
        if structured:
            return llm_client.chat(
                system=_SYSTEM_PROMPT,
                messages=messages,
                tools=[_SUBMIT_TASKS_TOOL],
                force_tool="submit_tasks",
                model_override=model_override,
            )
        return llm_client.chat(system=_SYSTEM_PROMPT, messages=messages, tools=[])

    response = None
    try:
        response = _chat(user_prompt, planning_model)
    except Exception:
        if planning_model:
            # The cheap sibling may not exist on this gateway — retry
            # once on the primary model before giving up.
            logger.info(
                "Gap analyzer: planning model %r failed; retrying on primary",
                planning_model,
            )
            try:
                response = _chat(user_prompt, None)
            except Exception:
                logger.warning("Gap analyzer LLM call failed; no checklist")
                return None
        else:
            logger.warning("Gap analyzer LLM call failed; no checklist")
            return None

    tasks = _extract_tasks(response)
    if tasks is not None:
        return tasks

    # One repair retry: tell the model exactly what was wrong.
    repair_prompt = (
        f"{user_prompt}\n\n"
        "Your previous reply could not be parsed. Reply with ONLY a JSON "
        "array of task strings (or call the submit_tasks tool). No prose, "
        "no markdown."
    )
    try:
        response = _chat(repair_prompt, planning_model)
    except Exception:
        logger.warning("Gap analyzer repair retry failed; no checklist")
        return None
    tasks = _extract_tasks(response)
    if tasks is None:
        logger.warning("Gap analyzer returned unparseable response twice; no checklist")
    return tasks


def _extract_tasks(response: dict[str, Any]) -> list | None:
    """Pull the task list from a planner response.

    Prefers the forced ``submit_tasks`` tool_use block; falls back to
    parsing a JSON array out of the text (for providers/gateways that
    ignore ``tool_choice``)."""
    if not isinstance(response, dict):
        return None
    for block in response.get("content", []):
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "tool_use":
            continue
        name = getattr(block, "name", None) or (
            block.get("name") if isinstance(block, dict) else None
        )
        if name != _SUBMIT_TASKS_TOOL["name"]:
            continue
        payload = getattr(block, "input", None) or (
            block.get("input") if isinstance(block, dict) else None
        )
        if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            return payload["tasks"]
    return _parse_task_array(_extract_text(response))


def _chat_supports_kwargs(llm_client, *names: str) -> bool:
    """True when ``llm_client.chat`` accepts every kwarg in ``names``.

    Older/mock clients with a positional ``chat(system, messages,
    tools)`` signature get the legacy free-text protocol."""
    try:
        sig = inspect.signature(llm_client.chat)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return all(n in params for n in names)


def _emit_phase_details(
    callback: Callable[[str, str], None] | None,
    tasks: list[str],
    fallback_label: bool = False,
    generator_failure: str | None = None,
) -> None:
    """Best-effort: surface the gap task list to the SSE consumer.

    Builds a markdown bullet list from the tasks so the smart-gen card
    can render it behind a chevron. Silently no-ops if the callback
    raises — the gap analyser must never break the run.
    """
    if callback is None:
        return
    try:
        if not tasks:
            return
        if fallback_label:
            if generator_failure:
                details = (
                    f"The deterministic generator failed ({generator_failure}). "
                    "The LLM will scaffold the entire codebase from the "
                    "domain model."
                )
            else:
                details = (
                    "No deterministic generator ran. The LLM will scaffold the "
                    "entire codebase from the domain model."
                )
        else:
            bullets = "\n".join(f"- {t}" for t in tasks)
            details = (
                f"Identified {len(tasks)} task"
                f"{'s' if len(tasks) != 1 else ''} for the customise loop:\n\n"
                f"{bullets}"
            )
        callback("gap", details)
    except Exception:
        logger.debug("on_phase_details callback raised; ignoring", exc_info=True)


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
    "  * CRITICAL — what the deterministic generator does NOT produce: it "
    "emits ONLY the data model's structure and basic CRUD endpoints/screens. "
    "It NEVER produces authentication, login/registration, JWT/session/token "
    "handling, authorization, roles/permissions, security, payments, email, "
    "file upload, custom business logic, custom UI styling/theming/colours, or "
    "third-party integrations. If the user asked for ANY of these, it is "
    "ALWAYS missing from the scaffold — emit concrete tasks for it and do NOT "
    "return an empty array.\n"
    "  * You may skip tests/Docker/CI unless the user explicitly asked for "
    "them.\n"
    "  * If the user named a target framework or language, every task must "
    "respect it — do not propose tasks for the generator's default stack.\n"
    "  * Return an EMPTY array ONLY when the scaffold genuinely and fully "
    "covers the request — i.e. a plain CRUD API/UI over exactly the model "
    "with no extra features requested. When in doubt, emit tasks.\n"
    f"  * Hard cap: {_MAX_TASKS} tasks. Prioritise correctness over completeness.\n\n"
    "Call the submit_tasks tool with your final list. If tool calling is "
    "unavailable, return ONLY a JSON array of task strings — no prose, no "
    "markdown."
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
        f"strings (max {_MAX_TASKS}). Tie every task to either a specific model "
        "element or a specific user requirement. Include explicit delete tasks "
        "for any generator-output files that no longer fit the customised "
        "stack. Submit an empty array if the scaffold already covers the "
        "request."
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
    and an error sentinel when serialisation fails — never raises, since
    a failed gap analysis is recoverable but a crashed one isn't.

    Whatever happens, the returned string is ALWAYS valid JSON: when the
    full payload exceeds the budget it is pruned progressively
    (inherited members → methods → attribute detail → bare class names)
    with an explicit ``"__truncated__": true`` marker, never raw-clipped
    into broken syntax.
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

    def _dump(payload: Any) -> str:
        return json.dumps(payload, default=str, separators=(",", ":"))

    serialized = _dump(data)
    if len(serialized) <= _MAX_MODEL_JSON_CHARS:
        return serialized

    # Pruning step 1: drop inherited_* keys (the bulkiest part).
    pruned: dict[str, Any] = dict(data)
    pruned["__truncated__"] = True
    for cls in pruned.get("classes", []):
        if isinstance(cls, dict):
            cls.pop("inherited_attributes", None)
            cls.pop("inherited_methods", None)
    serialized = _dump(pruned)
    if len(serialized) <= _MAX_MODEL_JSON_CHARS:
        return serialized

    # Pruning step 2: drop methods entirely.
    for cls in pruned.get("classes", []):
        if isinstance(cls, dict):
            cls.pop("methods", None)
    serialized = _dump(pruned)
    if len(serialized) <= _MAX_MODEL_JSON_CHARS:
        return serialized

    # Pruning step 3: reduce attributes to name + type.
    for cls in pruned.get("classes", []):
        if not isinstance(cls, dict):
            continue
        attrs = cls.get("attributes")
        if isinstance(attrs, list):
            cls["attributes"] = [
                {"name": a.get("name"), "type": a.get("type")}
                if isinstance(a, dict) else a
                for a in attrs
            ]
    serialized = _dump(pruned)
    if len(serialized) <= _MAX_MODEL_JSON_CHARS:
        return serialized

    # Last resort: bare class names — still valid JSON, never a raw clip.
    try:
        names = [
            cls.get("name") for cls in pruned.get("classes", [])
            if isinstance(cls, dict) and cls.get("name")
        ]
    except Exception:
        names = []
    return _dump({"classes": [{"name": n} for n in names], "__truncated__": True})


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

    Tries the whole (fence-stripped) text as JSON first, then falls
    back to extracting the first ``[...]`` block for replies with
    surrounding prose.
    """
    if not text:
        return None
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Whole-text parse first — immune to brackets inside task strings.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

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
