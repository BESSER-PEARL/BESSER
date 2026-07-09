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
from besser.generators.llm.stack_metadata import idiom_guidance_section

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
    scaffold_snapshot: str = "",
    modify_mode: bool = False,
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
        modify_mode: When True the run is an incremental vibe-modify — the
            output_dir was seeded from a previous run's generated files and
            the LLM edits them in place. Prepends a directive that biases
            the model toward the smallest surgical change. MUST leave the
            from-scratch prompt byte-identical when False.

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
        if primary_kind == "gui":
            # GUI-driven run: the screens ARE the spec — match them.
            gui_framing = (
                "Screens, components and data bindings declared in the editor. "
                "This is the UI spec — build the React/Flutter output to match "
                "these screens and wire components to the domain classes."
            )
        else:
            # Class/other-driven run: the GUI model here is incidental and is
            # frequently auto-generated (e.g. by the modeling agent), so it's
            # rough and over-literal adherence drags the output down. Frame it
            # as a hint, not a contract — the domain model and good UX win.
            gui_framing = (
                "A rough UI sketch from the editor — often auto-generated and "
                "incomplete. Treat it as a LOOSE HINT for which screens and "
                "fields matter, NOT a spec to reproduce literally. The domain "
                "model and good UX take priority: freely restructure, merge, "
                "add, drop, or relayout screens and components to produce a "
                "clean app. Do NOT degrade the result just to mirror this "
                "sketch, and don't treat its omissions as restrictions."
            )
        model_sections.extend([
            "",
            "## GUI Model",
            "",
            gui_framing,
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

    # Inline scaffold contents — saves the LLM 2-4 read_file round-trips
    # at the start of nearly every run. Constant ACROSS TURNS within a
    # run, so it caches like the rest of the prompt.
    snapshot_section = ""
    if scaffold_snapshot:
        snapshot_section = (
            "\n## Scaffold file contents (snapshot at end of Phase 1)\n\n"
            "These were the contents when customisation started. Files "
            "reproduced here do NOT need a read_file before their first "
            "edit. Once you edit a file, your edits supersede this "
            "snapshot — re-read only files you have modified.\n\n"
            f"{scaffold_snapshot}\n"
        )

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

    # Stack-specific idiom reminders (#4b). ``instructions`` is constant
    # across every turn of a single Phase 2 run, so — like ``primary_banner``
    # above — placing this in the cached stable_header doesn't cost any
    # turn-to-turn cache hits; it only varies between runs, same as the
    # rest of this header already does via primary_kind. See
    # stack_metadata.idiom_guidance_section for the detection + content.
    idiom_section = idiom_guidance_section(instructions)

    stable_header = f"""\
You are an expert full-stack developer. You make targeted, scoped changes to code.

Read the generated code before changing it and keep changes tightly scoped to the user's request.
Do not rewrite generated files from scratch — make surgical modifications.{primary_banner}

## Plan before you implement

Before making any edits, briefly state your plan, then execute it in the same turn:
1. What does the user request imply on top of the generated scaffold?
2. Which generated files/components must change, and which new files are needed?
3. In what order, to avoid broken imports or circular dependencies?
4. Which model elements (classes, attributes, relationships, OCL constraints) must
   be reflected in the result?

Keep the plan short (a few lines), then proceed with surgical edits.

## Rules

1. **Keep changes scoped to the user request.** Don't rewrite generated files
   or add features the user didn't ask for.
2. **Pick the right write tool.** Use `modify_file` for one or two
   localised edits to an existing file. Use `write_file` for new files
   and for any existing file that needs three or more changes — one
   `write_file` is cheaper than a chain of `modify_file` calls.
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
7. **Standard hygiene.** Every project must ship the STANDARD project
   file for its target stack: `pyproject.toml` for Python, `pom.xml`
   for Java with Maven, `package.json` for Node / TypeScript,
   `Cargo.toml` for Rust, `go.mod` for Go, `Gemfile` for Ruby,
   `build.gradle.kts` for Kotlin / Gradle. Ship a brief `README.md`
   with a one-line summary + a runnable quick-start command. Only add
   `tests/`, `Dockerfile`, or `.env.example` when the user request
   mentions them.
8. **Read before modify.** Read the relevant section of a file before editing it.
   Use offset/limit for large files (>200 lines).
9. **Be efficient. Batch tool calls in one turn.** Issue every
   independent tool call you can in the SAME turn — read multiple files
   in one turn, edit multiple files in one turn. When a single file
   needs three or more changes, prefer one `write_file` over a chain of
   `modify_file` calls. Each turn is a separate LLM round-trip, so
   sequential single edits multiply cost; the runner executes batched
   tool calls in parallel.
10. **Don't re-read files** you already read. Remember what's in them.
11. **Check your output against the model before finishing.** Before
    declaring done, verify that every Class's declared attributes and
    methods appear in your output (under whatever name the target
    language uses). If anything is missing, add it. Do not silently
    drop attributes the LLM judges "not needed" — the model is the spec.
12. **Implement every explicit request FULLY — no stubs, no token mentions.**
    When the user asks for a feature, behaviour, or styling, build it
    completely and wire it end-to-end. A requested feature that's only
    half-built — a form with no submit handler, an endpoint with no UI, a
    component that's never imported/rendered, a "TODO" — is a bug. Rule 1
    keeps you to what the user asked for; it does NOT excuse implementing
    it shallowly. Before finishing, re-check that every feature the user
    named actually works in the generated code.
13. **Styling/theme requests must actually RENDER.** When the user names
    colours or a visual theme, define them as concrete CSS — real hex
    values or CSS variables — and apply them consistently across the UI
    (backgrounds, buttons, headers, links, accents), not just one element.
    Map informal colour names to hex: rose → `#f43f5e`, pink → `#ec4899`,
    amber/yellow → `#f59e0b` / `#eab308`, teal → `#14b8a6`, indigo →
    `#6366f1`, emerald → `#10b981`. CRITICAL: `rose`, `amber`, `teal`,
    `indigo`, `emerald` are NOT valid CSS colour keywords — never write
    `color: rose`; use the hex. A theme that's only mentioned in a comment
    but not visibly applied is a failure. Aim for a clean, modern,
    cohesive look (consistent spacing, a primary + accent colour, readable
    contrast).
14. **Authentication, when requested, is COMPLETE and wired.** Generate the
    full flow: a registration / sign-up form AND a login form, secure
    password hashing (bcrypt / passlib / argon2 — never plaintext), token
    or session handling (e.g. JWT), protected routes/endpoints, and the
    frontend forms wired to working backend auth endpoints. "Login" implies
    the user can also CREATE AN ACCOUNT unless they say otherwise. No auth
    stubs — a user must be able to register, then log in, end to end.
15. **A domain-model app needs a COMPLETE, NAVIGABLE CRUD frontend** — unless
    the UI is already specified by GUI screens. Read-only lists are NOT
    enough. For the React frontend, build:
    - A home route (`"/"`) AND a persistent navigation bar/header linking every
      entity's page. The app must NEVER render blank on load — if you use a
      router, give it a landing page and wire the nav links to the routes.
    - Per entity: a list/table view PLUS working **Create** (a real form),
      **Edit** (a pre-filled form), and **Delete** — each wired to the backend's
      POST / PUT / DELETE endpoints, not just the GET list. A page that can only
      read is half-built (rule 12).
    - Loading, empty, and error states on every data fetch.
    - One shared stylesheet applied across the whole app (cards or clean tables,
      consistent spacing, a primary + accent colour, readable contrast) — a
      cohesive modern look, not unstyled browser-default HTML.
    - Internal consistency: `package.json` dependencies match the imports; the
      dev/build scripts actually run the app.

{idiom_section}## Tools for deeper model inspection

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
{inventory_section}{snapshot_section}
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

    # Incremental vibe-modify directive. Prepended (not woven into the
    # cached header) so the from-scratch prompt is byte-identical when
    # ``modify_mode`` is False — the whole point of the hard separation
    # between the from-scratch path and the modify path.
    modify_directive = ""
    if modify_mode:
        modify_directive = (
            "You are MODIFYING an existing, working app. Preserve everything "
            "that already works; make the smallest change that satisfies the "
            "request; prefer `modify_file` over `write_file`; do NOT "
            "regenerate untouched files.\n\n"
        )

    return f"{modify_directive}{stable_header}\n{models_block}\n{variable_tail}"


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


# File extensions that are never worth inlining (binary, locks, archives).
_SNAPSHOT_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz",
    ".woff", ".woff2", ".ttf", ".eot", ".pyc", ".db", ".sqlite",
    ".lock",
}
_SNAPSHOT_SKIP_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock", "Cargo.lock"}
_SNAPSHOT_SKIP_DIRS = {
    ".besser_snapshot", "node_modules", "target", "__pycache__", ".git",
    "dist", "build", ".next", ".gradle", "venv", ".venv",
}


def build_scaffold_snapshot(
    output_dir: str,
    max_file_lines: int = 150,
    total_char_budget: int = 24_000,
) -> str:
    """Render the contents of small scaffold files for the system prompt.

    The inventory only lists paths + sizes, while Rule 8 ("read before
    modify") forces the LLM to spend its first 2-4 turns on read_file
    calls against the same scaffold files. Inlining them up front — the
    system prompt is constant across turns, so this is paid once — cuts
    those round-trips entirely.

    Files are included smallest-first, only when <= ``max_file_lines``
    lines, under a global ``total_char_budget``. Binary / lock files and
    dependency dirs are skipped. Returns "" when nothing qualifies.
    """
    candidates: list[tuple[int, str, str]] = []  # (size, rel_path, content)
    for root, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in _SNAPSHOT_SKIP_DIRS]
        for fname in files:
            if fname.startswith(".besser_") or fname in _SNAPSHOT_SKIP_NAMES:
                continue
            if os.path.splitext(fname)[1].lower() in _SNAPSHOT_SKIP_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, output_dir).replace("\\", "/")
            try:
                size = os.path.getsize(fpath)
                if size > total_char_budget:
                    continue
                with open(fpath, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            if content.count("\n") + 1 > max_file_lines:
                continue
            candidates.append((size, rel, content))

    if not candidates:
        return ""

    sections: list[str] = []
    used = 0
    skipped: list[str] = []
    for size, rel, content in sorted(candidates):
        block = f"### `{rel}`\n\n```\n{content}\n```\n"
        if used + len(block) > total_char_budget:
            skipped.append(rel)
            continue
        used += len(block)
        sections.append(block)

    if not sections:
        return ""
    if skipped:
        sections.append(
            "_Not shown (budget): "
            + ", ".join(f"`{p}`" for p in skipped[:20])
            + " — use read_file for these._\n"
        )
    return "\n".join(sections)


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
        lines.append("\nBackend: FastAPI with SQLAlchemy ORM + Pydantic schemas (modular)")
        lines.append(
            "Files: main_api.py (slim app + include_router calls), database.py "
            "(engine/session), routers/<Class>.py (endpoints per class, @router), "
            "sql_alchemy.py, pydantic_classes.py, bal_stdlib.py, requirements.txt"
        )

    return "\n".join(lines)
