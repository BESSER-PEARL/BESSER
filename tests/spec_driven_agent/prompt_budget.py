"""Measure the fixed prefix every Phase 2 turn re-sends: system prompt + tool schemas.

On a provider without prompt caching that prefix is paid in full on every turn
(up to ``MAX_TURNS``), so its size is a first-order cost. ``phase2_request``
drives the real ``LLMOrchestrator._run_phase2`` up to its first chat call and
returns exactly what that call would send; ``breakdown`` splits it by section
and by tool. ``test_prompt_budget.py`` pins the totals.

Run it directly for the table: ``python -m tests.spec_driven_agent.prompt_budget``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from unittest import mock

from besser.BUML.metamodel.state_machine.state_machine import Event, StateMachine
from besser.BUML.metamodel.structural import (
    BinaryAssociation, BooleanType, Class, DateType, DomainModel, Enumeration, EnumerationLiteral,
    FloatType, IntegerType, Method, MethodImplementationType, Multiplicity, Parameter, Property,
    StringType,
)
from besser.spec_driven_agent.agent.prompt_builder import build_inventory
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

# The same request, planner checklist and requirements for every fixture, so the
# fixtures differ only in their models and scaffold.
INSTRUCTIONS = (
    "Build the application from the model. Add a dashboard page with totals per entity, "
    "validate every required field on create and edit, and style it with a teal theme."
)
GAP_TASKS = [
    "Add a dashboard page with totals per entity",
    "Validate required fields on create and edit",
    "Apply a teal theme across the UI",
]
REQUIREMENTS = [
    {"id": 1, "text": "A dashboard page shows totals per entity."},
    {"id": 2, "text": "Create and edit reject a missing required field."},
    {"id": 3, "text": "The UI uses a teal theme."},
]

# Section headings in the order build_system_prompt emits them; each section
# runs from its heading to the next heading present in the prompt.
_SECTION_MARKERS = (
    ("rules", "\n## Rules\n"),
    ("model_tools", "\n## Tools for deeper model inspection"),
    ("models", ("\n## Domain Model", "\n## GUI Model", "\n## Agent Model", "\n## Object Model",
                "\n## State Machines", "\n## Quantum Circuit", "\n## BPMN Process Model",
                "\n## Neural Network Model")),
    ("data_contract", "\n## Data contract"),
    ("tail_header", "\n## Variable context"),
    ("inventory", "\n## What was already generated"),
    ("endpoint_manifest", "\n## Backend REST API"),
    ("requirements", "\n## Requirements the user stated"),
    ("request", "\n## User request"),
    ("checklist", "\n### Focused checklist"),
    ("phase1_issues", "\n## Issues to fix from Phase 1 validation"),
    # The paragraph _run_phase2 appends after build_system_prompt's last line.
    ("phase2_appendix", "briefly summarize what you changed.\n"),
)


class _RequestCaptured(BaseException):
    """Raised by the stub client: a BaseException, so _run_phase2's ``except Exception`` lets it out."""


class _CaptureClient:
    """Records the first Phase 2 chat request and stops the loop there."""

    model = "prompt-budget-stub"

    def __init__(self):
        self.usage = UsageTracker(self.model)
        self.system: str | None = None
        self.tools: list[dict] | None = None

    def chat(self, system, messages, tools, **_):
        self.system, self.tools = system, tools
        raise _RequestCaptured


@dataclass(frozen=True)
class Fixture:
    """One representative run: its models, and whether Phase 1 scaffolds a FastAPI backend."""

    name: str
    domain_model: object = None
    state_machines: tuple = ()
    scaffold: bool = False


def shelf_fixture() -> Fixture:
    return Fixture("shelf", domain_model=shelf_model(isbn=True, author=True, library_method=True),
                   scaffold=True)


def conference_model() -> DomainModel:
    """Fifteen classes, an enumeration, methods and associations: a mid-size editor model."""
    kind = Enumeration(name="TicketKind", literals={
        EnumerationLiteral(name=n) for n in ("STANDARD", "VIP", "STUDENT")})
    fields = {
        "Conference": [("title", StringType), ("startDate", DateType), ("endDate", DateType)],
        "Track": [("name", StringType), ("color", StringType)],
        "Session": [("title", StringType), ("startsAt", DateType), ("minutes", IntegerType)],
        "Speaker": [("fullName", StringType), ("email", StringType), ("bio", StringType)],
        "Talk": [("title", StringType), ("abstract", StringType), ("accepted", BooleanType)],
        "Attendee": [("fullName", StringType), ("email", StringType)],
        "Ticket": [("code", StringType), ("price", FloatType), ("kind", kind)],
        "Venue": [("name", StringType), ("address", StringType), ("capacity", IntegerType)],
        "Room": [("name", StringType), ("seats", IntegerType)],
        "Sponsor": [("name", StringType), ("tier", StringType), ("amount", FloatType)],
        "Review": [("score", IntegerType), ("comment", StringType)],
        "Organizer": [("fullName", StringType), ("phone", StringType)],
        "Badge": [("label", StringType), ("printed", BooleanType)],
        "Invoice": [("number", StringType), ("total", FloatType), ("paid", BooleanType)],
        "Payment": [("amount", FloatType), ("method", StringType), ("paidOn", DateType)],
    }
    classes = {
        name: Class(name=name, attributes={
            Property(name=attr, type=typ, is_id=(name, attr) == ("Ticket", "code"))
            for attr, typ in attrs})
        for name, attrs in fields.items()
    }
    for owner, method, param, statement in (
            ("Ticket", "reprice", ("price", FloatType, "float"), "this.price = price;"),
            ("Invoice", "markPaid", None, "this.paid = true;"),
            ("Talk", "accept", None, "this.accepted = true;"),
            ("Room", "resize", ("seats", IntegerType, "int"), "this.seats = seats;")):
        parameters = [Parameter(name=param[0], type=param[1])] if param else []
        signature = f"{param[0]}:{param[2]}" if param else ""
        body = Method(name=method, parameters=parameters, type=None,
                      implementation_type=MethodImplementationType.BAL)
        body.code = f"def {method}({signature}) -> nothing {{\n    {statement}\n}}"
        classes[owner].add_method(body)
    links = (("Conference", "tracks", "Track", "conference"), ("Track", "sessions", "Session", "track"),
             ("Session", "talks", "Talk", "session"), ("Speaker", "talks", "Talk", "speaker"),
             ("Conference", "attendees", "Attendee", "conference"), ("Attendee", "tickets", "Ticket", "holder"),
             ("Venue", "rooms", "Room", "venue"), ("Room", "sessions", "Session", "room"),
             ("Conference", "sponsors", "Sponsor", "conference"), ("Talk", "reviews", "Review", "talk"),
             ("Conference", "organizers", "Organizer", "conference"), ("Attendee", "badges", "Badge", "attendee"),
             ("Attendee", "invoices", "Invoice", "attendee"), ("Invoice", "payments", "Payment", "invoice"))
    associations = {
        BinaryAssociation(name=f"{one}_{many_end}", ends={
            Property(name=many_end, type=classes[many], multiplicity=Multiplicity(0, "*")),
            Property(name=one_end, type=classes[one], multiplicity=Multiplicity(0, 1))})
        for one, many_end, many, one_end in links
    }
    return DomainModel(name="Conference", types={*classes.values(), kind}, associations=associations)


def conference_fixture() -> Fixture:
    return Fixture("conference", domain_model=conference_model(), scaffold=True)


def order_state_machine() -> StateMachine:
    """A state-machine-only project: no domain model, so no generator and no scaffold."""
    machine = StateMachine(name="OrderLifecycle")
    draft = machine.new_state("Draft", initial=True)
    placed = machine.new_state("Placed")
    shipped = machine.new_state("Shipped")
    closed = machine.new_state("Closed", final=True)
    for source, event, dest in ((draft, "place", placed), (placed, "ship", shipped),
                                (placed, "cancel", closed), (shipped, "deliver", closed)):
        source.when_event(Event(event)).go_to(dest)
    return machine


def state_machine_fixture() -> Fixture:
    return Fixture("state_machine", state_machines=(order_state_machine(),))


FIXTURES = {"shelf": shelf_fixture, "conference": conference_fixture, "state_machine": state_machine_fixture}


def phase2_request(fixture: Fixture, workspace) -> tuple[str, list[dict]]:
    """The system prompt and tool list of the first Phase 2 chat call, as the orchestrator sends them.

    Phase 1 is reproduced as ``_run_phase1`` leaves it on success (scaffold,
    ``_generator_used``, ``_inventory``); the planner call is replaced by the
    fixed ``GAP_TASKS``, requirement extraction by ``REQUIREMENTS``, and the
    Phase 1 validator findings by none: those depend on the host (a missing
    sandbox or toolchain adds "did not run" findings) and belong to the
    per-run tail, not the prefix this measures. Everything else is the orchestrator's own code: tool filtering, the
    endpoint manifest, the deterministic checklist items, the prompt and the
    paragraph ``_run_phase2`` appends to it.
    """
    client = _CaptureClient()
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=fixture.domain_model,
        state_machines=list(fixture.state_machines) or None,
        output_dir=str(workspace),
        enable_tracing=False,
        enable_checkpointing=False,
        use_streaming=False,
    )
    orch._instructions = INSTRUCTIONS
    orch._requirements = [dict(item) for item in REQUIREMENTS]
    if fixture.scaffold:
        generate_backend(workspace, fixture.domain_model)
        orch._generator_used = GENERATOR
        orch._inventory = build_inventory(str(workspace), fixture.domain_model, GENERATOR)
    with mock.patch.object(orchestrator_module, "analyze_gaps_via_llm", return_value=list(GAP_TASKS)), \
            mock.patch.object(orch, "_validate_app", return_value={"issues": []}):
        try:
            orch._run_phase2(INSTRUCTIONS)
        except _RequestCaptured:
            pass
    if client.system is None:
        raise AssertionError(f"{fixture.name}: Phase 2 ended before its first chat call")
    return client.system, client.tools


def tool_chars(tool: dict) -> int:
    """A tool schema's size as compact JSON, the form a provider request carries it in."""
    return len(json.dumps(tool, ensure_ascii=False, separators=(",", ":")))


def prompt_sections(system: str) -> dict[str, str]:
    """Split the system prompt at its section headings; the parts concatenate back to it."""
    cuts = [("role_plan", 0)]
    position = 0
    for name, markers in _SECTION_MARKERS:
        hits = [system.find(m, position) for m in ((markers,) if isinstance(markers, str) else markers)]
        hits = [h for h in hits if h >= 0]
        if not hits:
            continue
        position = min(hits)
        if name == "phase2_appendix":
            position += len(markers)
        cuts.append((name, position))
    bounds = [pos for _, pos in cuts[1:]] + [len(system)]
    return {name: system[start:end] for (name, start), end in zip(cuts, bounds)}


def breakdown(system: str, tools: list[dict]) -> dict:
    """Totals and per-part sizes in characters."""
    return {
        "system_chars": len(system),
        "tools_chars": sum(tool_chars(t) for t in tools),
        "sections": {name: len(text) for name, text in prompt_sections(system).items()},
        "tools": {t["name"]: tool_chars(t) for t in tools},
    }


def _report() -> None:
    """Print the breakdown of every fixture, with cl100k tokens when tiktoken can load offline."""
    import tempfile

    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        def tokens(text):
            return len(encoding.encode(text))
    except Exception:  # no tokenizer offline: characters only
        def tokens(text):
            return None
    for name, build in FIXTURES.items():
        with tempfile.TemporaryDirectory(prefix="besser_budget_") as workspace:
            system, tools = phase2_request(build(), workspace)
        tools_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        print(f"\n== {name}: system {len(system):,} chars / {tokens(system)} tok; "
              f"tools {len(tools)} = {len(tools_json):,} chars / {tokens(tools_json)} tok")
        for section, text in prompt_sections(system).items():
            print(f"   {section:<18} {len(text):>7,} chars {tokens(text)!s:>6} tok")
        for tool in sorted(tools, key=tool_chars, reverse=True):
            text = json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
            print(f"   tool {tool['name']:<30} {len(text):>6,} chars {tokens(text)!s:>6} tok")


if __name__ == "__main__":
    _report()
