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

Inventories are soft-clipped to the ``_MAX_*_CHARS`` budgets below (a
runaway 100k-line inventory would otherwise burn the context window
before the customise phase gets a chance). The serialized domain model
is pruned progressively but is ALWAYS valid JSON — the planner never
receives a raw character clip. Accepted user instructions are preserved in
full, including any harness-added requirement ledger. The user-input size
limit is enforced at the API/core entry points, before adding derived context.
"""

import inspect
import json
import logging
import re
from typing import Any, Callable

from besser.generators.llm.action_inventory import ActionEndpoint, format_action_inventory
from besser.generators.llm.model_serializer import ordered_literals, serialize_domain_model

logger = logging.getLogger(__name__)


# Soft budgets — generous, but a runaway 100k-line inventory would burn
# the whole context window before the customise phase gets a chance.
_MAX_INVENTORY_CHARS = 8_000
_MAX_MODEL_JSON_CHARS = 12_000
_MAX_ACTION_INVENTORY_CHARS = 8_000
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
    modify_mode: bool = False,
    workspace_files: list[str] | None = None,
    action_endpoints: list[ActionEndpoint] | None = None,
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
        workspace_files: Every file in the output tree, relative and
            ``/``-separated. Used to repair the paths the planner names
            (see ``_resolve_task_paths``); when omitted the task text is
            passed through unchanged. The full list on purpose — the
            ``inventory`` string is capped at 30 entries, so the paths a
            planner most often invents are the ones missing from it.
        action_endpoints: Source-derived operation routes and their handler
            locations. Presence in the domain model is not an implementation;
            this inventory identifies the actual executable extension points.

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
        if modify_mode:
            # A seeded/imported app is already in the workspace (its
            # recipe just didn't survive — e.g. a repo pushed before the
            # recipe was included). "Build from scratch" framing here
            # would invite the LLM to bulldoze the user's app.
            fallback = [
                "An existing application is already present in the "
                "workspace. Do NOT rebuild it from scratch: explore the "
                "code, keep its framework and structure, and implement "
                "the requested change with the smallest set of edits. "
                "Where the domain model and the code disagree, update "
                "the code to match the model."
            ]
        elif generator_failure:
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
            modify_mode=modify_mode,
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
        action_inventory=(format_action_inventory(action_endpoints)
                          if action_endpoints is not None
                          else "Action handlers were not inventoried; inspect their implementations."),
    )

    tasks = _call_planner(llm_client, user_prompt)
    if tasks is None:
        return None
    cleaned = _dedupe([t.strip() for t in tasks if isinstance(t, str) and t.strip()])[:_MAX_TASKS]
    cleaned = _sanitize_tasks(cleaned, generator_used, instructions)
    cleaned = _drop_present_enumerations(cleaned, domain_model)
    cleaned = _drop_present_attributes(cleaned, domain_model)
    cleaned = _note_present_regex_validations(cleaned, domain_model)
    cleaned = _resolve_task_paths(cleaned, workspace_files or [])
    cleaned = _note_dependent_rule_placement(cleaned, domain_model)
    cleaned = _note_model_only_tasks(cleaned, workspace_files or [])
    cleaned = _note_present_relationships(cleaned, domain_model)
    cleaned = _dedupe(_note_action_placement(cleaned, action_endpoints or []))
    # Harness-owned, prepended: a spec-vs-model gap the planner may miss or
    # (worse) judge "nothing to do" — this must survive even an empty list,
    # since [] can short-circuit Phase 2 entirely (see the module docstring).
    cleaned = _dedupe(_note_derived_enum_initial_state(domain_model, instructions) + cleaned)
    cleaned = _dedupe(_note_rejected_constraints(domain_model, cleaned) + cleaned)
    _emit_phase_details(on_phase_details, cleaned)
    return cleaned


# "to the domain model" as a TARGET, not "as defined in the domain model",
# and not "in the model web_app/backend/sql_alchemy.py" — that names an
# ORM source file, which is editable. Run n_6i2i5r annotated three such
# tasks and pointed them away from the very file they named.
_MODEL_TARGET_RE = re.compile(
    r"(?<!as defined )(?<!as described )(?<!as specified )(?<!according to )"
    r"\b(?:to|in|into|on)\s+the\s+(?:b-?uml\s+|besser\s+)?(?:domain\s+)?model\b"
    r"(?!\s+\S*[\w/\\.-]+\.(?:py|ts|tsx|js|jsx|json|sql))",
    re.IGNORECASE,
)
_MODEL_MUTATION_RE = re.compile(
    r"\b(?:add|create|define|introduce|declare|change|update|modify|rename|remove|delete)\b",
    re.IGNORECASE,
)


def _note_model_only_tasks(tasks: list, workspace_files: list) -> list:
    """Redirect a task that asks to edit the B-UML model into the code.

    Phase 2's model tools are query-only — there is no tool that mutates the
    domain model. Live run 7aybctis (2026-09-19): three of sixteen tasks were
    phrased "add an association class ... to the domain model" and "add a
    constraint ... to the Booking class in the domain model". The agent
    understood the intent and cited ``pydantic_classes.py``, but could not
    produce write evidence for a file it had not changed, so each burned its
    three checklist attempts and was recorded BLOCKED — nine of the run's
    thirty-four turns. The requirement stays; only its target is corrected.
    """
    validator = next((path for path in workspace_files
                      if path.replace("\\", "/").endswith("pydantic_classes.py")), None)
    target = f"`{validator}`" if validator else "the generated Pydantic/ORM modules"

    noted: list = []
    for task in tasks:
        if not isinstance(task, str) or not _MODEL_TARGET_RE.search(task) \
                or not _MODEL_MUTATION_RE.search(task):
            noted.append(task)
            continue
        noted.append(f"{task.rstrip()} (The B-UML model is read-only in this phase — "
                     f"no tool edits it. {_placement_advice(task, target)})")
    return noted


# OCL collection navigation: the rule reads a RELATED row, not this payload.
_RELATIONAL_OCL_RE = re.compile(
    r"->\s*(?:collect|forAll|exists|select|reject|includes|isEmpty|notEmpty)\b",
    re.IGNORECASE,
)


def _placement_advice(task: str, target: str) -> str:
    """Where a recovered rule can actually run.

    A Pydantic ``model_validator`` sees only the request payload. A rule that
    navigates a relationship needs the related rows, so it can only run where
    the database session exists. Run kinie9zr implemented
    ``guestsWithinCapacity`` as ``sum(room.maxOccupancy for room in
    self.rooms)`` inside the validator, where ``self.rooms`` holds link
    objects carrying ``target``/``agreedPrice`` and no capacity at all. Every
    POST /booking/ then raised AttributeError, which took the whole booking
    half of the app down. The earlier version of this note named the
    validator file first and so pointed the model straight at it.
    """
    if _RELATIONAL_OCL_RE.search(task):
        return (
            "This rule reads related rows, so it CANNOT run in a Pydantic "
            "validator - that sees only the request payload, where a "
            "relationship is an id or a link object, not the related row. "
            "Enforce it in the router handler that performs the operation, "
            "after loading the related rows through the database session, and "
            "reject the request there."
        )
    return (f"Implement this in the generated code: {target} for a field or "
            "shape constraint, or the router handler that performs the "
            "operation when the rule needs data beyond the request payload.")


def _note_action_placement(tasks: list[str], endpoints: list[ActionEndpoint]) -> list[str]:
    """Keep planning prose anchored to the handlers which actually serve it.

    Run 8efe8fd4 put every action task in sql_alchemy.py; all six action
    endpoints remained 501. Annotate explicit action references without
    discarding requirements or guessing that an ORM method is connected.
    Harness-owned action tasks enforce omissions independently of this hint.
    """
    noted = []
    for task in tasks:
        matches = [item for item in endpoints if re.search(
            rf"\b{_class_name_pattern(item.action)}\b", task, re.IGNORECASE,
        )]
        if matches:
            locations = "; ".join(
                f"{item.http_method} {item.route} is served by {item.path} "
                f"function {item.function}"
                for item in matches
            )
            task = (
                f"{task.rstrip().rstrip('.')}. ACTION HANDOFF: {locations}. "
                "Implement or wire the behavior there; adding a same-named ORM "
                "method alone does not connect the action endpoint."
            )
        noted.append(task)
    return noted


# Scaffold families for the task sanitizer (mirrors the orchestrator's
# framework-switch enforcement).
_GAP_SCAFFOLD_RIVALS = {
    "generate_fastapi_backend": ("flask", "django"),
    "generate_web_app": ("flask", "django"),
    "generate_rest_api": ("flask", "django"),
    "generate_django": ("flask", "fastapi"),
}

_DELETE_SCAFFOLD_RE = re.compile(
    r"\b(delete|remove|drop|uninstall|replace)\b.{0,60}\b(scaffold|"
    r"generated|frontend|react|output|backend framework)\b",
    re.IGNORECASE,
)


def _task_key(text: str) -> str:
    """Identity of a checklist item: case, whitespace and a trailing period aside."""
    return " ".join(text.lower().split()).rstrip(".")


def _dedupe(tasks: list) -> list:
    """Drop repeated tasks, first-seen order kept.

    Live run (2026-09-17): the planner listed the same item three times;
    the model noticed ("tasks 15, 16 and 18 are duplicated"), did the
    work once, and then re-did it for each open copy. Runs before the cap
    so repeats cannot crowd unique work out of the list.
    """
    seen: set = set()
    kept: list = []
    for task in tasks:
        key = _task_key(task)
        if key in seen:
            logger.info("Gap sanitizer dropped duplicate task: %r", task[:100])
            continue
        seen.add(key)
        kept.append(task)
    return kept


def _sanitize_tasks(
    tasks: list, generator_used: str | None, instructions: str
) -> list:
    """Drop checklist items that would demolish the scaffold.

    Live finding (Devstral A/B, 2026-09-02): the planner proposed
    'delete react frontend scaffold' and 'install Flask' — a checklist
    that would fight the Phase-2 HARD CONSTRAINTS and the framework-
    switch blocker for the whole run. The prompt now forbids it; this
    filter guarantees it. Rival-framework mentions are only dropped when
    the USER didn't ask for that framework themselves.
    """
    rivals = _GAP_SCAFFOLD_RIVALS.get(generator_used or "", ())
    low_instr = (instructions or "").lower()
    kept: list = []
    for task in tasks:
        low = task.lower()
        if _DELETE_SCAFFOLD_RE.search(task):
            logger.info("Gap sanitizer dropped scaffold-demolition task: %r", task[:100])
            continue
        rival_hit = next(
            (r for r in rivals if r in low and r not in low_instr), None
        )
        if rival_hit:
            logger.info(
                "Gap sanitizer dropped rival-framework (%s) task: %r",
                rival_hit, task[:100],
            )
            continue
        kept.append(task)
    return kept


# An add/create/define verb within three tokens of "enum": proposes the
# enumeration itself, as opposed to using one ("add a cancel endpoint that
# moves BookingCommercialStatus ...").
_ADD_ENUM_RE = re.compile(
    r"\b(?:add|create|introduce|define|declare)\b(?:\s+\S+){0,3}?\s+enum",
    re.IGNORECASE,
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _drop_present_enumerations(tasks: list, domain_model) -> list:
    """Drop add-enumeration tasks whose literal set the model already has.

    Live finding (2026-09-17): the planner proposed adding 'commercialStatus'
    with AWAITING_PAYMENT/CONFIRMED/CANCELLED while the model carried
    BookingCommercialStatus with exactly those literals. It matched on the
    name - the unreliable part - and ignored the identical member set. The
    proposed name is ignored here on purpose: every literal of an existing
    enumeration (two or more) named in an add-enumeration task is the match.
    """
    if domain_model is None:
        return tasks
    try:
        present = [
            (enum.name, {lit.name.lower() for lit in enum.literals})
            for enum in domain_model.get_enumerations()
        ]
    except Exception:
        return tasks
    kept: list = []
    for task in tasks:
        if _ADD_ENUM_RE.search(task):
            tokens = {t.lower() for t in _IDENT_RE.findall(task)}
            match = next(
                (name for name, lits in present if len(lits) >= 2 and lits <= tokens),
                None,
            )
            if match:
                logger.info(
                    "Gap sanitizer dropped enumeration already in the model as %s: %r",
                    match, task[:100],
                )
                continue
        kept.append(task)
    return kept


# "add/create/... an attribute": proposes declaring an attribute the model
# may already carry. The identifier is read from a token WINDOW around the
# "attribute" keyword, not the whole sentence, so an unrelated quoted token
# elsewhere in a longer task (an enum literal, a requirement quote) is never
# mistaken for the attribute's own name.
_ADD_VERB_RE = re.compile(r"\b(?:add|create|introduce|define|declare)\b", re.IGNORECASE)
_ATTR_KEYWORD_RE = re.compile(r"\battributes?\b", re.IGNORECASE)
_QUOTED_IDENT_RE = re.compile(r"['\"`]([A-Za-z_][A-Za-z0-9_]*)['\"`]")

# Naming the attribute is not the same as asking for the BEHAVIOUR behind it.
# Live counterexample (se7k3zbx, 2026-09-19): "add a computed
# 'commercialStatus' attribute that is derived from the bill's settlement
# status, transitioning from 'awaiting payment' to 'confirmed' when the bill
# is settled" names an attribute the model already has, but the transition
# LOGIC - which the model does not encode - is the real ask. This downgrades
# a drop to an annotation so the requirement itself is never lost.
_ATTR_BEHAVIOR_RE = re.compile(
    r"\b(?:derive[ds]?\s+from|computed?\s+from|transition\w*|based\s+on|"
    r"calculat\w+|whenever|logic)\b",
    re.IGNORECASE,
)
_TYPE_SYNONYMS = {"boolean": "bool", "string": "str", "text": "str", "integer": "int"}
_TYPE_WORD_RE = re.compile(
    r"\b(float|double|int|integer|str|string|text|bool|boolean|date|datetime|time)\b",
    re.IGNORECASE,
)


def _attr_idents_near_keyword(sentence: str) -> list[str]:
    """Quoted identifiers within 6 tokens of the word 'attribute(s)'."""
    tokens = sentence.split()
    idents: list[str] = []
    for i, tok in enumerate(tokens):
        if not _ATTR_KEYWORD_RE.search(tok):
            continue
        window = " ".join(tokens[max(0, i - 6): i + 7])
        idents.extend(_QUOTED_IDENT_RE.findall(window))
    return idents


def _rejected_constraint_names(domain_model) -> set:
    """Names of OCL invariants that were REJECTED during conversion.

    These live in ``domain_model.conversion_issues``, never in
    ``domain_model.constraints`` - but the planner routinely re-describes a
    rejected rule in a LATER, separate task without naming it (e.g.
    "does not exceed the combined capacity of all rooms" instead of
    "guestsWithinCapacity"). Every "already present" family in this module
    checks this set FIRST and backs off entirely on a hit: a false "already
    handled" signal on unimplemented, rejected work is worse than a missed
    drop or annotation on implemented work.
    """
    names: set = set()
    for issue in getattr(domain_model, "conversion_issues", None) or []:
        name = issue.get("name") if isinstance(issue, dict) else None
        if name:
            names.add(str(name))
    return names


def _mentions_rejected_constraint(task: str, rejected_names: set) -> bool:
    return any(re.search(rf"\b{re.escape(n)}\b", task, re.IGNORECASE) for n in rejected_names)


# A model can reject many constraints; the checklist is a scarce resource.
_MAX_REJECTED_CONSTRAINT_TASKS = 6


def _note_rejected_constraints(domain_model, tasks: list) -> list[str]:
    """Harness-owned tasks for OCL invariants the converter threw away.

    The generator now leaves a TODO where it could not enforce a constraint
    (``pydantic_classes/ocl_utils.py``), which makes the omission visible but
    does not make it anyone's job. Nothing turned a rejected invariant into
    work: ``_rejected_constraint_names`` was only ever read defensively, to
    stop an "already present" family from dropping a task that happened to
    mention one. If the planner did not raise it, the rule shipped as a
    comment.

    So the harness raises it. Rejected invariants are exactly the rules the
    deterministic path cannot express - the owner's directive is that the
    agent implements those, not that the generator grows to cover them.

    Only constraints no existing task already names are added, since the
    planner's own phrasing carries the spec's wording and already gets
    placement advice. The rejection reason rides along: it usually says
    which property name did not resolve, and that is the model-vs-spec
    delta the agent needs in order to write the check against the ends the
    model actually declares rather than the ones the OCL text assumed.
    """
    issues = getattr(domain_model, "conversion_issues", None) or []
    added: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        name = issue.get("name")
        expression = issue.get("expression") or issue.get("original_text") or ""
        if not name or not expression:
            continue
        if any(_mentions_rejected_constraint(t, {str(name)}) for t in tasks):
            continue
        context = issue.get("context") or ""
        reason = (issue.get("reason") or "").removeprefix("Warning: ")
        reason = reason.replace(f" in '{expression}'", "", 1).strip()
        where = f"that create or update {context}" if context else "for this operation"
        added.append(
            f"Enforce the '{name}' rule, which the model records but no generated code "
            f"checks: {expression}. The OCL converter rejected it"
            + (f" ({reason})" if reason else "")
            + f", so pydantic_classes.py carries only a TODO for it. Implement the check "
            f"in the router handlers {where}, after loading the related rows via the "
            f"database session, and write it against the relationship names the model "
            f"actually declares - the ones in the OCL text above may not exist."
        )
        if len(added) >= _MAX_REJECTED_CONSTRAINT_TASKS:
            break
    return added


def _mentioned_classes(sentence: str, classes: list) -> list:
    """Model classes named in ``sentence``, tolerating a Create, Update,
    Read or Response schema suffix (``BookingCreate`` still names
    ``Booking``)."""
    found = []
    for cls in classes:
        pattern = _class_name_pattern(cls.name)
        if re.search(rf"\b{pattern}(?:Create|Update|Read|Response)?\b", sentence):
            found.append(cls)
    return found


def _drop_present_attributes(tasks: list, domain_model) -> list:
    """Drop (or, when behaviour is also being asked for, annotate) an
    add-attribute task whose (class, attribute) pair the model already has.

    Verified 2026-09-19 against verification/spec-iterations (18 real
    recipes, one model shared across all of them): run fcdh0s9k's task 11
    "Add a 'extraCharges' attribute to the ReservedRoom class" - ReservedRoom
    already declares agreedPrice AND extraCharges, and the generated
    sql_alchemy.py / pydantic_classes.py of every run already carry both end
    to end.

    Matching requires an exact (case-insensitive) attribute-name match on
    the NAMED class's OWN attributes: not a same-named attribute on an
    unrelated class, not a class the sentence never mentions. A type word
    (float/int/bool/...) that contradicts the model's actual type blocks the
    match entirely - that names a real gap, not a duplicate. Only a task
    that is PURELY a declaration is dropped; one that also asks for derived
    behaviour (see ``_ATTR_BEHAVIOR_RE``) is annotated instead, so the
    requirement is never silently lost.
    """
    if domain_model is None:
        return tasks
    try:
        classes = list(domain_model.get_classes())
        rejected = _rejected_constraint_names(domain_model)
    except Exception:
        return tasks

    noted: list = []
    for task in tasks:
        if not isinstance(task, str) or not _ADD_VERB_RE.search(task) \
                or not _ATTR_KEYWORD_RE.search(task) \
                or _mentions_rejected_constraint(task, rejected):
            noted.append(task)
            continue
        decision = None  # "drop" | "annotate"
        for sentence in _sentences(task):
            if not (_ADD_VERB_RE.search(sentence) and _ATTR_KEYWORD_RE.search(sentence)):
                continue
            quoted = _attr_idents_near_keyword(sentence)
            if not quoted:
                continue
            for cls in _mentioned_classes(sentence, classes):
                attr_by_lower = {a.name.lower(): a for a in getattr(cls, "attributes", None) or []}
                if not all(q.lower() in attr_by_lower for q in quoted):
                    continue
                type_word = _TYPE_WORD_RE.search(sentence)
                if type_word and len(quoted) == 1:
                    word = _TYPE_SYNONYMS.get(type_word.group(1).lower(), type_word.group(1).lower())
                    type_name = getattr(getattr(attr_by_lower[quoted[0].lower()], "type", None), "name", None)
                    if type_name and word != str(type_name).lower():
                        continue  # names a type the model disagrees with - a real gap
                decision = "annotate" if _ATTR_BEHAVIOR_RE.search(sentence) else "drop"
                break
            if decision:
                break
        if decision == "drop":
            logger.info("Gap sanitizer dropped already-present attribute task: %r", task[:100])
            continue
        if decision == "annotate":
            noted.append(
                f"{task.rstrip()} (NOTE: the model already declares this "
                "attribute and the generator already carries it into the generated "
                "ORM/Pydantic code - focus on the requested BEHAVIOUR, not on "
                "re-adding the field.)"
            )
            logger.info("Gap sanitizer annotated already-present attribute task: %r", task[:100])
            continue
        noted.append(task)
    return noted


# ``self.<attr>.matches('<regex>')`` with nothing else: the one shape
# besser/generators/pydantic_classes/ocl_utils.py turns into a
# ``@field_validator`` without skipping it (a constraint touching
# collections/relationships is emitted as a code comment instead).
_OCL_HEADER_RE = re.compile(r"^\s*context\s+\S+(?:::\S+)?\s+inv\s+\S+\s*:\s*", re.IGNORECASE)
_OCL_PURE_MATCH_RE = re.compile(
    r"^self\s*\.\s*(?P<attr>[A-Za-z_]\w*)\s*\.\s*matches\s*\(\s*(?:'[^']*'|\"[^\"]*\")\s*\)$"
)
_REGEX_VALIDATION_MARKER_RE = re.compile(
    r"\b(?:regex|pattern|format|valid(?:ate|ated|ation)?)\b", re.IGNORECASE,
)


def _present_regex_validations(domain_model) -> list[tuple[str, str, str, set]]:
    """(class_name, attribute_name, constraint_name, subclass_names) for
    every successfully-converted OCL invariant that is exactly a
    ``self.<attr>.matches(regex)`` check.

    ``domain_model.constraints`` holds only invariants that PARSED; a
    REJECTED one lives in ``domain_model.conversion_issues`` and is never
    read here, so a task about a rejected constraint can never match.
    """
    triples: list[tuple[str, str, str, set]] = []
    for c in getattr(domain_model, "constraints", None) or []:
        body = _OCL_HEADER_RE.sub("", c.expression or "", count=1).strip()
        m = _OCL_PURE_MATCH_RE.match(body)
        if not m:
            continue
        ctx = getattr(c, "context", None)
        if ctx is None or not getattr(ctx, "name", None):
            continue
        try:
            subclasses = {s.name for s in ctx.all_specializations()}
        except Exception:
            subclasses = set()
        triples.append((ctx.name, m.group("attr"), c.name, subclasses))
    return triples


def _note_present_regex_validations(tasks: list, domain_model) -> list:
    """Annotate (never drop) a task that asks to validate an attribute the
    model already enforces via a successfully-converted OCL regex invariant.

    Verified 2026-09-19: 6 of 18 real runs re-asked for Person email/phone
    validation while the model's validEmail/validPhone invariants already
    convert and the generated PersonCreate already carries their exact
    ``@field_validator`` (inherited by GuestCreate/EmployeeCreate, since both
    subclass PersonCreate in every generated pydantic_classes.py).

    Kept as an ANNOTATION, not a drop: run gpt-5.6-terra-dp3trml9's task 14
    bundles the already-satisfied validEmail/validPhone re-ask together with
    "enforce Person identifyingNumber uniqueness", which the model does NOT
    carry anywhere. Dropping the whole task on the validEmail/validPhone
    match would have silently discarded that second, genuine requirement -
    exactly the failure mode this module must not reproduce.
    """
    if domain_model is None:
        return tasks
    try:
        triples = _present_regex_validations(domain_model)
        rejected = _rejected_constraint_names(domain_model)
    except Exception:
        return tasks
    if not triples:
        return tasks

    noted: list = []
    for task in tasks:
        if not isinstance(task, str) or _mentions_rejected_constraint(task, rejected):
            noted.append(task)
            continue
        hit = None
        for sentence in _sentences(task):
            for cls_name, attr_name, constraint_name, subclasses in triples:
                if re.search(rf"\b{re.escape(constraint_name)}\b", sentence, re.IGNORECASE):
                    hit = (cls_name, attr_name, constraint_name)
                    break
                if not re.search(rf"\b{re.escape(attr_name)}\b", sentence, re.IGNORECASE):
                    continue
                if not _REGEX_VALIDATION_MARKER_RE.search(sentence):
                    continue
                names = {cls_name} | subclasses
                if any(re.search(rf"\b{_class_name_pattern(n)}\b", sentence) for n in names):
                    hit = (cls_name, attr_name, constraint_name)
                    break
            if hit:
                break
        if hit is None:
            noted.append(task)
            continue
        cls_name, attr_name, constraint_name = hit
        noted.append(
            f"{task.rstrip()} (NOTE: {cls_name}.{attr_name} already carries "
            f"the OCL invariant '{constraint_name}'; the generated Pydantic Create "
            "model already enforces it with a @field_validator - verify it in the "
            "generated code before re-adding it.)"
        )
        logger.info(
            "Gap sanitizer annotated already-validated attribute task (%s.%s): %r",
            cls_name, attr_name, task[:100],
        )
    return noted


# Sentences that plausibly name a starting/initial state ("begins",
# "starts out", "initially", "by default", "not set by hand").
_INITIAL_STATE_MARKER_RE = re.compile(
    r"\b(?:starts?|begins?|beginning|initial(?:ly)?|by\s+default|"
    r"default(?:s)?\s+to|not\s+set\s+by\s+hand)\b",
    re.IGNORECASE,
)

# Types the generator (sql_alchemy/templates/helpers.py.j2) already defaults a
# required derived attribute to on INSERT, because the type has an unambiguous
# zero. An enum has none, so its column is emitted nullable instead: the app is
# constructible, but the initial state is still open and only the spec knows it.
_SERVER_DEFAULTABLE_TYPES = {"int", "float", "str", "string", "bool"}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _literal_tokens(literal: str) -> list[str]:
    return re.findall(r"[a-z]+", literal.lower())


def _find_initial_state_quote(
    instructions: str, literals: list[str],
) -> tuple[str, str] | None:
    """The spec's own sentence naming one of ``literals`` as the starting value.

    Requires BOTH an initial-state marker word and every token of the
    literal's name (``NOT_ARRIVED`` -> "not", "arrived") in the same
    sentence — loose enough to match "not yet arrived" against NOT_ARRIVED,
    strict enough that an unrelated mention of e.g. "cancelled" elsewhere in
    the spec is not mistaken for the starting state.
    """
    for sentence in _sentences(instructions):
        if not _INITIAL_STATE_MARKER_RE.search(sentence):
            continue
        low_tokens = set(re.findall(r"[a-z]+", sentence.lower()))
        for literal in literals:
            tokens = _literal_tokens(literal)
            if tokens and all(tok in low_tokens for tok in tokens):
                return sentence, literal
    return None


def _derived_enum_task_text(
    cls_name: str, attr_name: str, enum_name: str, literals: list[str],
    found: tuple[str, str] | None,
) -> str:
    lits = ", ".join(literals)
    if found:
        sentence, literal = found
        resolution = (
            f"THE SPEC NAMES IT: \"{sentence}\" — set the initial value to "
            f"{enum_name}.{literal}."
        )
    else:
        resolution = (
            "Re-read the user's specification for the sentence that names "
            f"the starting state of {cls_name}.{attr_name} and use that "
            "value — do not guess."
        )
    return (
        f"{cls_name}.{attr_name} is a derived, required {enum_name} "
        "attribute with no default and no server-side initial value: an "
        f"enum has no unambiguous zero, so {enum_name}'s first-declared "
        f"literal ({literals[0]}) is only declaration order, not a "
        "decision, and the generator left the column nullable rather than "
        f"invent one. Creates therefore succeed, but every new {cls_name} "
        f"starts with no {attr_name} at all — a state the spec does not "
        "describe, and one that no status check will match. THE SPEC DECIDES "
        f"the initial value here, NOT the model. {resolution} Close this "
        f"in the generated code: give {cls_name}.{attr_name} that value "
        f"wherever a {cls_name} row is created (ORM column default or the "
        f"creation service), so every new {cls_name} starts in that state "
        f"without the client sending {attr_name}. Available literals: {lits}."
    )


def _note_derived_enum_initial_state(domain_model, instructions: str) -> list[str]:
    """Harness-owned tasks for a derived enum attribute the generator left unset.

    The owner's directive: when the model and the user's spec disagree, the
    Spec-Driven Agent closes the gap in the GENERATED CODE — it does not
    wait for a planner call to notice, and it does not ship the model's
    silence as if it were a decision.

    A required derived attribute with an unambiguous zero (int/float/str/bool)
    gets a server-side default at INSERT time; an enum has no zero, so its
    column is emitted nullable instead — "use the first literal" would have
    shipped every new booking already CHECKED_IN while the spec says a booking
    "starts out with the guests not yet arrived". Nullable keeps the app
    constructible without inventing that state, but it does not supply it:
    the row starts with no status, so this task remains the only stage that
    holds the spec text and the model side by side and can say what it is.

    Only the genuinely open case is flagged: derived, required (not
    optional, not the id), no explicit default already on the model, and
    typed as an enumeration — the one type the generator could not default
    to a zero. A derived int/float/str/bool attribute needs nothing here.
    """
    if domain_model is None:
        return []
    try:
        classes = list(domain_model.get_classes())
        enums = {e.name: e for e in domain_model.get_enumerations()}
    except Exception:
        return []
    if not enums:
        return []

    tasks: list = []
    for cls in sorted(classes, key=lambda c: c.name):
        for attr in sorted(getattr(cls, "attributes", None) or [], key=lambda a: a.name):
            if not getattr(attr, "is_derived", False):
                continue
            if getattr(attr, "is_id", False) or getattr(attr, "is_optional", False):
                continue
            if getattr(attr, "default_value", None) is not None:
                continue
            type_name = getattr(getattr(attr, "type", None), "name", None)
            if type_name in _SERVER_DEFAULTABLE_TYPES:
                continue
            enum = enums.get(type_name)
            if enum is None:
                continue
            # Declaration order, not alphabetical: the task text below calls
            # literals[0] the "first-declared" literal, and a name sort makes
            # that claim false exactly when it matters (CANCELLED before
            # SCHEDULED).
            literals = [lit.name for lit in ordered_literals(enum)]
            if not literals:
                continue
            found = _find_initial_state_quote(instructions, literals)
            tasks.append(_derived_enum_task_text(cls.name, attr.name, type_name, literals, found))
    return tasks


# File paths named inside a task: a dotted basename, optionally preceded
# by directory segments. Restricted to extensions the generators actually
# emit so prose like ``domain_model.get_classes`` or a bare ``Booking.`` is
# not mistaken for a file.
_TASK_PATH_RE = re.compile(
    r"(?<![\w/\\.])((?:[\w.\-]+[/\\])*[\w.\-]+"
    r"\.(?:py|tsx|ts|jsx|js|css|scss|html|json|ya?ml|md|sql|txt|toml|cfg|ini|sh))"
    r"(?![\w/\\])",
    re.IGNORECASE,
)
# Minimum shared basename prefix before one file is offered as the "nearest"
# candidate for another. Four characters keeps ``BookingForm`` -> ``Booking``
# while rejecting ``Bill`` -> ``Booking``.
_NEAREST_PREFIX_CHARS = 4
_MAX_NEAREST = 2


def _nearest_existing(path: str, workspace_files: list[str]) -> list[str]:
    """Real files whose name plausibly answers ``path``.

    Same extension and a shared basename-stem prefix, longest first — the
    planner's invented ``BookingForm.tsx`` resolves to the real
    ``pages/Booking.tsx`` this way.
    """
    base = path.rsplit("/", 1)[-1]
    stem, _, ext = base.rpartition(".")
    stem = stem.lower()
    scored: list[tuple[int, str]] = []
    for real in workspace_files:
        real_base = real.rsplit("/", 1)[-1]
        real_stem, _, real_ext = real_base.rpartition(".")
        if real_ext.lower() != ext.lower():
            continue
        real_stem = real_stem.lower()
        shared = 0
        for a, b in zip(stem, real_stem):
            if a != b:
                break
            shared += 1
        if shared >= _NEAREST_PREFIX_CHARS:
            scored.append((shared, real))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [real for _, real in scored[:_MAX_NEAREST]]


def _resolve_task_paths(tasks: list, workspace_files: list) -> list:
    """Repair the file paths a task list names, against the real tree.

    Live run (2026-09-18, Qwen3-30B on a 63-file ``generate_web_app``
    scaffold): the planner named ``frontend/src/components/BookingForm.tsx``
    and ``BookingDetails.tsx``, neither of which exists — the Booking screen
    is ``frontend/src/pages/Booking.tsx``. Phase 2 failed both reads and then
    re-ran the same four searches three times over: 18 turns, zero writes.

    Two deterministic repairs, no LLM call:

    * a path whose suffix matches exactly ONE real file is rewritten to it
      (the planner routinely drops the ``web_app/`` prefix);
    * a path matching NO real file keeps its task — "create this file" is
      legitimate work — but is annotated with that fact and the nearest real
      candidates, which is what stops the hunt.

    An ambiguous suffix is left alone: guessing between two real files is
    worse than leaving the model to look.
    """
    if not workspace_files:
        return tasks

    real = {f.replace("\\", "/").lstrip("./") for f in workspace_files}
    by_suffix: dict[str, set] = {}
    for full in real:
        parts = full.split("/")
        for i in range(len(parts)):
            by_suffix.setdefault("/".join(parts[i:]), set()).add(full)

    repaired: list = []
    for task in tasks:
        text = task
        missing: list[str] = []
        for token in dict.fromkeys(_TASK_PATH_RE.findall(task)):
            norm = token.replace("\\", "/").lstrip("./")
            if norm in real:
                continue
            matches = by_suffix.get(norm, set())
            if len(matches) == 1:
                target = next(iter(matches))
                text = text.replace(token, target)
                logger.info(
                    "Gap sanitizer repaired task path %r -> %r", token, target
                )
            elif not matches:
                missing.append(norm)
        if missing:
            notes = []
            for path in missing:
                nearest = _nearest_existing(path, sorted(real))
                notes.append(
                    f"{path} does not exist in the workspace"
                    + (f" (nearest existing: {', '.join(nearest)})" if nearest else "")
                )
            text = f"{text.rstrip().rstrip('.')}. NOTE: " + "; ".join(notes) + (
                ". Do not search for it — edit the nearest existing file, or "
                "create the file if the task genuinely needs a new one."
            )
            logger.info(
                "Gap sanitizer flagged unresolved path(s) %s in task: %r",
                missing, task[:100],
            )
        repaired.append(text)
    return repaired


# The creation verb, optional articles, then the class name - whose camel
# humps may be split in prose ("booked room", "booked_room").
_CREATE_VERB = r"\bcreat(?:e|es|ed|ing|ion)(?:[\s_-]+(?:a|an|the|new|each|every|of))*[\s_-]+"


def _class_name_pattern(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", lambda _: r"[\s_-]*", re.escape(name))


def _note_dependent_rule_placement(tasks: list, domain_model) -> list:
    """Tell a create-time task about rows that cannot exist yet.

    Live run 9a6063ed (2026-09-18): the planner emitted "add validation in
    'create_booking' to enforce that the total number of guests does not
    exceed the sum of room capacities across all BookedRooms", and Phase 2
    wrote exactly that - at insert time. A BookedRoom needs a Booking id, so
    when create_booking runs there are never any: capacity was 0, every
    POST /booking/ was a 400, and the booking half of the app was dead.

    The dependency is in the model. For a task that creates X and names a
    class whose rows require an X, say where the rule can hold. The rule
    itself stays - only its placement is corrected.
    """
    if domain_model is None:
        return tasks
    try:
        needs = domain_model._mandatory_dependencies()
    except Exception:
        return tasks
    dependents: dict[str, set] = {}
    for cls, required in needs.items():
        for parent in required:
            if parent != cls:
                dependents.setdefault(parent, set()).add(cls)
    if not dependents:
        return tasks

    noted: list = []
    for task in tasks:
        text = task
        squashed = re.sub(r"[\s_-]+", "", task.lower())
        for parent in sorted(dependents):
            name = _class_name_pattern(parent)
            creates_parent = re.search(
                rf"{_CREATE_VERB}{name}\b|\b{name}[\s_-]*creat(?:e|ion)\b|\bpost\s+/{name}/?",
                task, re.IGNORECASE,
            )
            if not creates_parent:
                continue
            named = sorted(c for c in dependents[parent] if c.lower() in squashed)
            if not named:
                continue
            rows = " or ".join(named)
            text = (
                f"{text.rstrip().rstrip('.')}. NOTE: no {rows} row can exist before "
                f"the {parent} it requires, so a {parent} being created has none - a "
                f"rule about them cannot be checked while creating the {parent} "
                f"without rejecting every request. Enforce it where {rows} rows are "
                f"created, updated or deleted (and when the {parent} is updated), "
                f"or create them inline in the same request."
            )
            logger.info(
                "Gap sanitizer noted dependent-row placement (%s) in task: %r",
                parent, task[:100],
            )
            break
        noted.append(text)
    return noted


_RELATIONSHIP_MARKER_RE = re.compile(
    r"\b(?:relationship|reference|cardinalit\w*|association|valid\s+\w+)\b",
    re.IGNORECASE,
)

# A rule that aggregates or compares ACROSS related rows (a capacity sum, an
# overlap check) is not the plain existence/cardinality guard the router
# already emits, even when it is phrased through a real association name.
# Live finding (2026-09-19, several runs): the planner re-describes a
# REJECTED constraint ("does not exceed the combined capacity of all rooms")
# without naming it, so the name-based ``_mentions_rejected_constraint``
# guard alone does not catch it - this catches it on the shape of the rule
# instead. Both rejected constraints in the verified dataset are exactly
# this shape (a capacity sum, a date overlap), so this guard is what keeps
# every one of their paraphrases out of this function's reach.
_AGGREGATE_RULE_MARKER_RE = re.compile(
    r"\b(?:capacity|occupancy|overlap\w*|exceed\w*|combined|aggregate|sum)\b",
    re.IGNORECASE,
)


def _relationship_names_by_class(domain_model) -> dict:
    """class name -> relationship identifiers reachable FROM it: each
    sibling end's role name (the field the generator puts on this class's
    own Create schema) plus the association's own name - how these tasks
    usually refer to it, even when that differs from the generated field
    (the 'handledBy' association's own field on Booking is 'employee')."""
    names: dict = {}
    for assoc in getattr(domain_model, "associations", None) or []:
        ends = list(getattr(assoc, "ends", None) or [])
        if len(ends) != 2:
            continue
        for i, end in enumerate(ends):
            other = ends[1 - i]
            cls_name = getattr(getattr(end, "type", None), "name", None)
            if not cls_name:
                continue
            bucket = names.setdefault(cls_name, set())
            if getattr(other, "name", None):
                bucket.add(other.name.lower())
            if getattr(assoc, "name", None):
                bucket.add(assoc.name.lower())
    return names


def _note_present_relationships(tasks: list, domain_model) -> list:
    """Annotate (never drop) a task that asks to validate an association
    end / relationship reference the model already declares.

    Verified 2026-09-19: run trilraak asked, in 4 separate tasks phrased
    "Ensure the '<end>' relationship in Booking enforces that ... is a
    valid ... by validating ... in the create_booking endpoint", for exactly
    the contact/guests/handledBy/rooms relationships - and the generated
    web_app/backend/routers/booking.py of that same run already has the
    400/404 existence check and the minimum-cardinality check for every one
    of them, deterministically emitted from the model's multiplicities. All
    four tasks sat at attempts=0: the agent recognised they were already
    done and never touched them, but they still occupied a checklist slot.

    Kept as an ANNOTATION rather than a drop: the evidence that the router
    already enforces it is a property of a SPECIFIC generator's template
    (``router.py.j2``, used by the web-app/FastAPI/REST-API generators), not
    a fact derivable from the domain model alone for every generator (e.g.
    Django, or a from-scratch build), and the wording of these tasks is far
    less uniform than the enumeration or attribute families - some ask for
    more than existence (a business-specific check a plain ID-existence
    guard would not cover).
    """
    if domain_model is None:
        return tasks
    try:
        classes = list(domain_model.get_classes())
        rel_names = _relationship_names_by_class(domain_model)
        rejected = _rejected_constraint_names(domain_model)
    except Exception:
        return tasks
    if not rel_names:
        return tasks

    noted: list = []
    for task in tasks:
        if not isinstance(task, str) or _mentions_rejected_constraint(task, rejected) \
                or _AGGREGATE_RULE_MARKER_RE.search(task):
            noted.append(task)
            continue
        hit = None
        for sentence in _sentences(task):
            if not _RELATIONSHIP_MARKER_RE.search(sentence):
                continue
            for cls in _mentioned_classes(sentence, classes):
                idents = rel_names.get(cls.name)
                if not idents:
                    continue
                low = sentence.lower()
                match = next(
                    (rel for rel in idents if re.search(rf"\b{re.escape(rel)}\b", low)), None,
                )
                if match:
                    hit = (cls.name, match)
                    break
            if hit:
                break
        if hit is None:
            noted.append(task)
            continue
        cls_name, rel_name = hit
        noted.append(
            f"{task.rstrip()} (NOTE: '{rel_name}' is a relationship the "
            f"model already declares on {cls_name}. A deterministically generated "
            "router typically already enforces existence and minimum-cardinality "
            "checks for it from the model's multiplicities - verify in the "
            "generated router before re-implementing this.)"
        )
        logger.info(
            "Gap sanitizer annotated already-declared relationship task (%s.%s): %r",
            cls_name, rel_name, task[:100],
        )
    return noted


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
        tasks = _extract_tasks(response)
    except Exception:
        logger.warning("Gap analyzer repair retry failed; trying plain-JSON fallback")
        tasks = None
    if tasks is not None:
        return tasks

    # Final fallback, NO tools: some OpenAI-compatible gateways (ollama
    # for certain models, e.g. devstral) reject a FORCED tool_choice with
    # an error even though ordinary tool calling works — which silently
    # killed the checklist (and with it the end_turn gate) on every free-
    # tier run. A plain call asking for a bare JSON array sidesteps the
    # gateway quirk entirely.
    try:
        response = llm_client.chat(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": repair_prompt}],
            tools=[],
        )
        tasks = _extract_tasks(response)
    except Exception:
        tasks = None
    if tasks is None:
        logger.warning("Gap analyzer unparseable after plain fallback; no checklist")
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
    modify_mode: bool = False,
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
            if modify_mode:
                details = (
                    "Editing the existing app in place — the deterministic "
                    "generator is skipped so your current code is preserved."
                )
            elif generator_failure:
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
    "  * The scaffold's framework is FIXED — it is the stack named in the "
    "inventory. NEVER propose switching frameworks (e.g. FastAPI→Flask), "
    "deleting the scaffold, or removing the generated frontend. Every task "
    "EXTENDS the existing stack. A 'delete X' task is allowed ONLY for a "
    "leftover file that is unused within that SAME stack, and ONLY when the "
    "user's own words asked for a different stack than the scaffold's.\n"
    "  * Skip anything the generator already provided correctly.\n"
    "  * Use the ACTION IMPLEMENTATION INVENTORY to locate actual operation "
    "handlers. A modeled method may still be an HTTP 501 placeholder. Put "
    "action work in its real handler (or wire the handler to a service), not "
    "only in the ORM. Preserve bodies already supplied by the model unless "
    "the request requires changing them. Unresolved handlers also receive "
    "harness-owned tasks independently of your list.\n"
    "  * CRITICAL — what the deterministic generator does NOT produce: it "
    "cannot invent behavior from method signatures alone. It emits the data "
    "model's structure, basic CRUD endpoints/screens, and any explicit method "
    "bodies supplied in the model. "
    "If, and only if, the user's own words asked for one of the following, "
    "check whether it is missing from the scaffold — emit concrete tasks for "
    "missing behavior and do NOT "
    "return an empty array: authentication, login/registration, "
    "JWT/session/token handling, authorization, roles/permissions, security, "
    "payments, email, file upload, custom business logic, custom UI "
    "styling/theming/colours, third-party integrations. Never propose one "
    "the request did not mention.\n"
    "  * CRITICAL — THE DOMAIN MODEL IS NOT THE SPEC. It was produced by an "
    "earlier modelling step that routinely loses what the user stated in "
    "prose: operations, status vocabularies, validity rules, and facts that "
    "belong to a relationship rather than to either entity. The USER REQUEST "
    "is the authority; the model is only what survived. Anything the request "
    "names that is ABSENT from the model is absent from the generated code "
    "too, and no later stage will notice — you are the only step that sees "
    "both. Emit a task for each one, naming the exact words the user used.\n"
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
    action_inventory: str = "",
) -> str:
    # Only derived inventories are clipped; user requirements must not disappear.
    inventory_clipped = _clip(inventory, _MAX_INVENTORY_CHARS, "inventory")
    actions_clipped = _clip(action_inventory, _MAX_ACTION_INVENTORY_CHARS, "action inventory")
    return (
        f"USER REQUEST:\n{instructions}\n\n"
        f"DETERMINISTIC GENERATOR THAT RAN: {generator_used}\n\n"
        f"DOMAIN MODEL (JSON):\n{model_json}\n\n"
        f"FILE INVENTORY (paths + sizes):\n{inventory_clipped}\n\n"
        f"ACTION IMPLEMENTATION INVENTORY (source-derived):\n{actions_clipped}\n\n"
        "Work in two passes.\n\n"
        "PASS 1 — USER REQUEST vs DOMAIN MODEL. Re-read the request and "
        "enumerate every concrete thing it names:\n"
        "  - operations an entity must support (\"a booking can be "
        "cancelled\", \"produces a bill\") — one method each, using the "
        "user's own count and names;\n"
        "  - status or state vocabularies, INCLUDING the exact words used. "
        "Two independent dimensions of status are TWO enumerations, not one "
        "merged one, and the members are the user's words — never invented "
        "ones;\n"
        "  - rules, limits and validity conditions (\"must not exceed\", "
        "\"cannot overlap\", \"must be a valid ...\") — one enforced "
        "constraint each;\n"
        "  - facts that belong to a RELATIONSHIP rather than to either "
        "entity (\"the price agreed for that room in that booking\") — a "
        "link attribute, which needs its own table or column;\n"
        "  - screens, roles, integrations, styling.\n"
        "For each item, check whether it appears in the DOMAIN MODEL JSON "
        "above. Match on MEANING and on member sets, not on exact names: an "
        "enumeration whose literals are the vocabulary the request describes "
        "IS that element even when its name carries a prefix "
        "(BookingCommercialStatus for 'commercial status'); the same goes for "
        "attributes and methods named with a prefix or different casing. "
        "Every item that does NOT appear is a gap the modelling step "
        "lost: emit a task stating the exact names and values from the "
        "request, and where in the stack to add them.\n\n"
        "PASS 2 — DOMAIN MODEL vs FILE INVENTORY. What the model does carry "
        "but the scaffold did not implement, plus anything the request asks "
        "for that the deterministic generator never emits.\n\n"
        "Then submit the combined list as a JSON array of short task strings "
        f"(max {_MAX_TASKS}), Pass-1 gaps FIRST — they are invisible "
        "everywhere else. Tie every task to either a specific model element "
        "or a specific line of the user's request. Include explicit delete "
        "tasks for any generator-output files that no longer fit the "
        "customised stack. Submit an empty array only if the scaffold "
        "already covers the request in full."
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
        fallback = {}
        issues = getattr(domain_model, "conversion_issues", None)
        if issues:
            fallback["conversion_issues"] = issues
        try:
            names = [c.name for c in domain_model.get_classes()]
            fallback["classes"] = [{"name": n} for n in names]
        except Exception:
            pass
        return json.dumps(fallback, default=str)

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
    fallback = {"classes": [{"name": n} for n in names], "__truncated__": True}
    # Explicit model-loss diagnostics are obligations, not expendable summary.
    # Preserve them even when that means exceeding the model-summary budget.
    if data.get("conversion_issues"):
        fallback["conversion_issues"] = data["conversion_issues"]
    return _dump(fallback)


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
