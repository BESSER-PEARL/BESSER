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
    serialize_object_model,
    serialize_quantum_circuit,
    serialize_state_machines,
)

logger = logging.getLogger(__name__)


def build_system_prompt(
    domain_model,
    gui_model,
    agent_model,
    inventory: str,
    instructions: str,
    max_turns: int,
    scoped_issues: list[str] | None = None,
    gap_tasks: list[str] | None = None,
    object_model=None,
    state_machines=None,
    quantum_circuit=None,
    primary_kind: str | None = None,
) -> str:
    """
    Build the system prompt with all available models, inventory, the user's
    request, and any concrete validator-detected issues.

    Each non-empty model is embedded as a JSON section so the LLM can reason
    over the full editor context, not just the class diagram. The LLM plans
    its own task breakdown from ``instructions`` — we no longer pre-compute
    a checklist (the keyword-based gap analyzer was deleted).

    Args:
        domain_model: Optional BUML domain model. When None, the LLM is
            told explicitly that no ClassDiagram was provided and must
            drive the run from whichever ``primary_kind`` model IS
            present — see Rule 3 in the system prompt below.
        gui_model: Optional GUIModel.
        agent_model: Optional AgentModel.
        inventory: Description of what the generator already produced.
        instructions: The user's verbatim natural-language request.
        max_turns: Maximum agent loop iterations (for budget hints).
        scoped_issues: Validator-detected issues from Phase 1 the LLM must
            fix as concrete bugs (separate from user-requested features).
        object_model: Optional ObjectModel (instance data — useful as fixtures/seeders).
        state_machines: Optional list of StateMachine objects.
        quantum_circuit: Optional QuantumCircuit.
        primary_kind: Which model is driving the generation: "class",
            "gui", "agent", "state_machine", "object", "quantum". Used to
            frame the LLM's task (e.g. "this is a state-machine-driven
            run — emit the transition code").

    Returns:
        The full system prompt string.
    """
    model_sections: list[str] = []
    if domain_model is not None:
        model_sections.extend([
            "## Domain Model",
            "",
            "```json",
            json.dumps(serialize_domain_model(domain_model), indent=2),
            "```",
        ])

    gui_json = serialize_gui_model(gui_model)
    if gui_json:
        model_sections.extend([
            "",
            "## GUI Model",
            "",
            "Screens, components and data bindings declared in the editor. "
            "Use this to drive React/Flutter output and wire components to domain classes.",
            "",
            "```json",
            json.dumps(gui_json, indent=2),
            "```",
        ])

    agent_json = serialize_agent_model(agent_model)
    if agent_json:
        model_sections.extend([
            "",
            "## Agent Model",
            "",
            "Conversational agent states and intents. Drives chatbot wiring.",
            "",
            "```json",
            json.dumps(agent_json, indent=2),
            "```",
        ])

    object_json = serialize_object_model(object_model) if object_model is not None else None
    if object_json:
        model_sections.extend([
            "",
            "## Object Model (instance data)",
            "",
            "Concrete instances of domain classes. Use these as seed data, test "
            "fixtures, or example payloads — they represent real expected values.",
            "",
            "```json",
            json.dumps(object_json, indent=2),
            "```",
        ])

    sm_json = serialize_state_machines(state_machines) if state_machines else None
    if sm_json:
        model_sections.extend([
            "",
            "## State Machines",
            "",
            "Behavioural spec for domain entities. When a class is governed by a "
            "state machine, generate a state field + transition guards + event "
            "handlers that respect these transitions.",
            "",
            "```json",
            json.dumps(sm_json, indent=2),
            "```",
        ])

    q_json = serialize_quantum_circuit(quantum_circuit) if quantum_circuit is not None else None
    if q_json:
        model_sections.extend([
            "",
            "## Quantum Circuit",
            "",
            "Quantum circuit definition for Qiskit / Cirq code generation.",
            "",
            "```json",
            json.dumps(q_json, indent=2),
            "```",
        ])

    # Cross-model links — heuristic correlations so the LLM treats related
    # diagrams as a single system, not as independent islands.
    cross_links = _compute_cross_model_links(
        domain_model=domain_model,
        sm_json=sm_json,
        object_json=object_json,
        agent_json=agent_json,
    )
    if cross_links:
        model_sections.extend([
            "",
            "## Cross-model links",
            "",
            "Heuristic associations between the models above. Use these as "
            "hints about which code should cross-reference which model.",
            "",
            "```json",
            json.dumps(cross_links, indent=2),
            "```",
        ])

    models_block = "\n".join(model_sections)

    # Build inventory section
    inventory_section = ""
    if inventory:
        inventory_section = f"\n## What was already generated\n\n{inventory}\n"

    # Phase 1 validator findings — concrete bugs the LLM must fix on top of
    # whatever the user asked for. Kept separate from the user request so
    # the LLM doesn't conflate "user wants X" with "generator produced
    # broken Dockerfile".
    issues_section = ""
    if scoped_issues:
        formatted = "\n".join(f"  - {issue}" for issue in scoped_issues)
        issues_section = (
            "\n## Issues to fix from Phase 1 validation\n\n"
            "These are concrete problems detected in the generator output. "
            "Fix them in addition to implementing the user request:\n\n"
            f"{formatted}\n"
        )

    # ------------------------------------------------------------------
    # Cache strategy
    # ------------------------------------------------------------------
    # Anthropic prompt caching keys on byte-identical prefixes. Everything
    # BEFORE the ``## Variable context`` marker is stable across turns and
    # across runs for the same diagram — so we keep role, rules, tool
    # guidance, and the serialized models up top where they benefit from
    # ephemeral caching. The variable tail (inventory, user request, gap
    # tasks, Phase 1 issues) changes per run and never gets cached. This
    # rewrite was driven by a prompt-caching audit that found the
    # previous order mixed variable content into the middle of the
    # prompt, invalidating the cache on every new request.
    # Primary-kind banner. When the user drove the run from anything
    # other than a ClassDiagram, we tell the LLM up front so it doesn't
    # default to "build a CRUD app". Omitted for class-diagram-driven
    # runs because that IS the default.
    primary_banner = ""
    if primary_kind and primary_kind != "class":
        friendly = {
            "gui": "GUI-driven run — screens and components are the spec; there is no domain/class model.",
            "agent": "Agent-driven run — the chatbot / intent flow is the spec; there is no domain/class model.",
            "state_machine": "State-machine-driven run — transition tables are the spec. Generate transition code / guards.",
            "object": "Object-diagram-driven run — instance data is the spec. Emit seed data / fixtures.",
            "quantum": "Quantum-circuit-driven run — circuit gates are the spec. Emit Qiskit code.",
        }.get(primary_kind, f"Primary model kind: {primary_kind}")
        primary_banner = f"\n> Primary input: {friendly}\n"

    stable_header = f"""\
You are an expert full-stack developer. You make targeted, scoped changes to code.

Read the generated code before changing it and keep changes tightly scoped to the user's request.
Do not rewrite generated files from scratch — make surgical modifications.{primary_banner}

## Rules

1. **Keep changes scoped to the user request.** Don't rewrite generated files
   or add features the user didn't ask for.
2. **Use modify_file** for changes to existing files. Use write_file only for NEW files.
3. **Model is truth.** Never invent entities not in the models above. If a
   detail is missing from the JSON, query it with the tools above before
   guessing.
4. **State machines drive code.** When a state machine governs a class
   (see Cross-model links), generate a state field + transition guards +
   event handlers that respect the declared transitions. Do not invent
   states or transitions.
5. **Object instances are seed data.** Treat the Object Model as concrete
   examples — emit them as fixtures, seeders, or example payloads in the
   generated code, not as new domain rules.
6. **OCL constraints must run.** For every constraint listed under the
   class, emit a runtime check in the language idiomatic to the target
   (Pydantic `@field_validator`, Zod refine, SQL CHECK, etc.).
7. **Standard hygiene.** A deployable app needs a README; a Dockerised app
   needs a docker-compose; an env-dependent app needs `.env.example`.
   Add these without being asked when they're obviously required.
8. **Read before modify.** Read the relevant section of a file before editing it.
   Use offset/limit for large files (>200 lines).
9. **Be efficient.** One read per file. Write complete files in one call.
10. **Don't re-read files** you already read. Remember what's in them.

## Tools for deeper model inspection

When the JSON below is not enough — large model, need a single class in
detail, or want to filter classes by a predicate — use these tools rather
than re-reading or guessing:

- **`query_class(name)`** — full definition of one class: attributes,
  methods, parents, association ends, abstract flag. Use this when you
  need details that don't fit in the summary.
- **`list_classes_with(predicate)`** — find classes matching a simple
  rule. Predicates: `is_abstract`, `is_root`, `has_constraint`,
  `has_attribute:<name>`, `has_method:<name>`, `extends:<parent_name>`.
  Use this for "every class with an OCL constraint" or "every leaf in
  the hierarchy" queries.
- **`get_constraints_for(class_name)`** — OCL constraint expressions
  scoped to a class. **Translate these into runtime validators** in the
  target language (Pydantic field validators, Zod schemas, SQL CHECK
  constraints, etc.) — the generator does not enforce them for you.
"""

    variable_tail = f"""\

## Variable context (per-run — not cached below this line)
{inventory_section}
## User request

The generator handled the base app (CRUD, ORM, schemas, pages). Your job is to
implement what the user asked for, on top of the generator output:

> {instructions.strip()}
{_render_gap_section(gap_tasks)}
Plan your own work from the request — pick the right files to edit, the
right packages to add, the right order. Do NOT exceed the request scope.
{issues_section}
When done, briefly summarize what you changed.
"""

    return f"{stable_header}\n{models_block}\n{variable_tail}"


def _render_gap_section(gap_tasks: list[str] | None) -> str:
    """Render the optional 'Focused checklist' section.

    The cheap gap-analyzer LLM call produces this list. If the call
    failed or wasn't invoked, ``gap_tasks`` is empty and we render
    nothing — the Phase 2 LLM plans its own work.
    """
    if not gap_tasks:
        return ""
    bullets = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(gap_tasks))
    return (
        "\n\n### Focused checklist (from gap analysis)\n"
        "Use this list as a starting point — it captures what the gap "
        "analyzer thinks is missing from the generator output. You can "
        "add or skip items based on the user request.\n\n"
        f"{bullets}\n"
    )


def _compute_cross_model_links(
    domain_model,
    sm_json,
    object_json,
    agent_json,
) -> dict[str, Any] | None:
    """Heuristic cross-references between the models in the prompt.

    * For each state machine, flag the class whose name is contained in
      the state-machine name (case-insensitive substring) — a common
      convention (``OrderStateMachine`` → ``Order``).
    * For the object model, list which classes have instances so the LLM
      can treat those as seed data.
    * For the agent model, signal its presence (the agent typically wires
      into the whole app, not any single class).
    """
    try:
        class_names = [c.name for c in domain_model.get_classes()]
    except Exception:
        class_names = []

    links: dict[str, Any] = {}

    # State machine ↔ class
    if sm_json:
        sm_bindings = []
        for sm in sm_json:
            name = sm.get("name", "")
            name_lower = name.lower()
            match = next(
                (c for c in class_names if c and c.lower() in name_lower and c.lower() != name_lower),
                None,
            )
            if match:
                sm_bindings.append({
                    "state_machine": name,
                    "governs_class": match,
                })
        if sm_bindings:
            links["state_machine_bindings"] = sm_bindings

    # Object model ↔ class
    if object_json:
        classes_with_instances = sorted({
            o.get("class") for o in object_json.get("objects", []) if o.get("class")
        })
        if classes_with_instances:
            links["classes_with_instances"] = classes_with_instances

    # Agent presence
    if agent_json:
        links["agent_present"] = True

    return links or None


def build_inventory(output_dir: str, domain_model, generator_name: str) -> str:
    """
    Describe what the generator produced -- for the LLM's context.

    Args:
        output_dir: Path to the output directory.
        domain_model: The BUML domain model, or None when the run is
            driven by a non-class primary (state machine, agent, etc.).
            When None, the domain-specific lines (entities, enums,
            associations) are omitted rather than showing empty values.
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

    lines = [f"Generator `{generator_name}` produced {len(files)} files:"]
    for f in sorted(files)[:30]:
        lines.append(f"  - {f}")
    if len(files) > 30:
        lines.append(f"  ... and {len(files) - 30} more files")

    if domain_model is not None:
        try:
            class_names = [c.name for c in domain_model.get_classes()]
            enum_names = [e.name for e in domain_model.get_enumerations()]
            lines.append(f"\nEntities with full CRUD: {', '.join(class_names)}")
            if enum_names:
                lines.append(f"Enumerations: {', '.join(enum_names)}")
            lines.append(f"Associations: {len(domain_model.associations)} relationships")
        except Exception:
            # Malformed / partially-constructed domain model — skip the
            # domain-specific lines rather than spraying tracebacks into
            # the LLM prompt.
            pass

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
