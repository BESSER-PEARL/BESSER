"""A rejected OCL invariant has to become work, once.

The Pydantic generator leaves a TODO where it could not enforce a
constraint. That makes the omission visible; it does not make it anyone's
job. On the hotel model, ``guestsWithinCapacity`` and
``noOverlappingBookings`` are both rejected during conversion, and unless
something seeds them the run ships them as comments.

The harness seeds one recovery item per rejected constraint
(``LLMOrchestrator._deterministic_gap_tasks``); it is the item the
requirements judge can close. The gap analyzer used to prepend a second
item for the same rule, so the model did the work twice.
"""

import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import gap_analyzer
from besser.spec_driven_agent.planning.action_inventory import merge_action_tasks
from besser.spec_driven_agent.providers.llm_client import UsageTracker

GUESTS_OCL = "context Booking inv guestsWithinCapacity: self.guests->size() <= 4"
OVERLAP_OCL = "context Room inv noOverlappingBookings: self.bookings->forAll(b1, b2 | true)"
HOTEL_ISSUES = [
    {
        "id": "issue-1",
        "context": "Booking",
        "name": "guestsWithinCapacity",
        "expression": GUESTS_OCL,
        "original_text": GUESTS_OCL,
        "reason": (f"Warning: Invalid OCL syntax in '{GUESTS_OCL}': Property "
                   "'guests' not found in context 'Booking' (did you mean 'self.guest'?)"),
        "code": "parse_error",
    },
    {
        "id": "issue-2",
        "context": "Room",
        "name": "noOverlappingBookings",
        "expression": OVERLAP_OCL,
        "reason": "Warning: Property 'bookings' not found in context 'Room'",
        "code": "parse_error",
    },
]


class _Planner:
    """A real-provider stand-in whose planner submits ``tasks``."""

    model = "mock"
    _client = object()
    planning_model = None

    def __init__(self, tasks):
        self.usage = UsageTracker("mock")
        self.tasks = tasks
        self.prompts = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.prompts.append(messages[0]["content"])
        block = type("B", (), {"type": "tool_use", "name": force_tool,
                               "input": {"tasks": [{"task": t, "kind": "change"} for t in self.tasks]}})()
        return {"stop_reason": "tool_use", "content": [block]}


def _orchestrator(tmp_path, issues, generator="generate_fastapi_backend", modify=False):
    model = DomainModel(name="Hotel", types={Class(name="Booking"), Class(name="Room")})
    model.conversion_issues = [dict(issue) for issue in issues]
    orch = LLMOrchestrator(llm_client=_Planner([]), domain_model=model, output_dir=str(tmp_path))
    orch._generator_used = generator
    orch._modify_mode = modify
    orch._instructions = "Build the hotel API."
    return orch


def _checklist(orch, planner_tasks):
    """The Phase 2 checklist as ``_run_phase2`` seeds it."""
    gap = gap_analyzer.analyze_gaps_via_llm(
        instructions="Build the hotel API.", generator_used=orch._generator_used,
        domain_model=orch.domain_model, inventory="", llm_client=_Planner(planner_tasks),
        workspace_files=[], action_endpoints=[],
    )
    orch.executor.set_tasks(merge_action_tasks(orch._deterministic_gap_tasks() + (gap or []), []))
    return orch.executor._tasks


@pytest.mark.parametrize("planner_tasks", [[], ["Add a search box to the bookings page"]])
def test_a_rejected_constraint_is_one_checklist_item(tmp_path, planner_tasks):
    tasks = _checklist(_orchestrator(tmp_path, HOTEL_ISSUES), planner_tasks)

    for name in ("guestsWithinCapacity", "noOverlappingBookings"):
        assert len([t for t in tasks if name in t["text"]]) == 1, [t["text"][:60] for t in tasks]


def test_the_item_carries_the_ocl_the_reason_and_where_to_put_it(tmp_path):
    tasks = _checklist(_orchestrator(tmp_path, HOTEL_ISSUES), [])
    task = next(t for t in tasks if "guestsWithinCapacity" in t["text"])

    assert GUESTS_OCL in task["text"]
    # The rejection reason is the model-vs-spec delta: it names the property
    # that did not resolve, so the agent can write the check against the end
    # the model actually declares.
    assert "did you mean 'self.guest'?" in task["text"]
    assert any("router handler" in note for note in task["planning_notes"])
    # A rule navigating a collection reads related rows: never a Pydantic validator.
    overlap = next(t for t in tasks if "noOverlappingBookings" in t["text"])
    assert any("CANNOT run in a Pydantic validator" in note for note in overlap["planning_notes"])


@pytest.mark.parametrize("generator", [
    "generate_fastapi_backend", "generate_web_app", "generate_django", "generate_python_classes", None,
])
@pytest.mark.parametrize("modify", [False, True])
def test_every_rejected_constraint_keeps_an_item(tmp_path, generator, modify):
    """The gap analyzer no longer seeds these, so the harness must on every path,
    including issues without a name or an expression."""
    issues = HOTEL_ISSUES + [
        {"id": "issue-3", "context": "Room", "name": "", "expression": "context Room inv: true"},
        {"id": "issue-4", "context": "Room", "name": "unnamedRule", "expression": ""},
    ]
    seeded = [t["text"] if isinstance(t, dict) else t
              for t in _orchestrator(tmp_path, issues, generator, modify)._deterministic_gap_tasks()]

    recover = [t for t in seeded if t.startswith("Recover model constraint")]
    assert len(recover) == len(issues)
    for name in ("guestsWithinCapacity", "noOverlappingBookings", "unnamedRule"):
        assert any(name in t for t in recover)


def test_the_gap_analyzer_adds_no_item_of_its_own(tmp_path):
    model = _orchestrator(tmp_path, HOTEL_ISSUES).domain_model
    assert gap_analyzer.analyze_gaps_via_llm(
        instructions="Build the hotel API.", generator_used="generate_fastapi_backend",
        domain_model=model, inventory="", llm_client=_Planner([]),
    ) == []


def test_nothing_is_seeded_when_conversion_rejected_nothing(tmp_path):
    seeded = _orchestrator(tmp_path, [])._deterministic_gap_tasks()
    assert not [t for t in seeded if "Recover model constraint" in str(t)]


def test_the_planner_model_json_quotes_a_rejected_expression_once(tmp_path):
    model = _orchestrator(tmp_path, HOTEL_ISSUES).domain_model
    payload = gap_analyzer._safe_serialize_model(model)

    assert payload.count("self.guests->size() <= 4") == 1
    issue = next(i for i in json.loads(payload)["conversion_issues"] if i["name"] == "guestsWithinCapacity")
    assert issue["expression"] == GUESTS_OCL
    assert "did you mean 'self.guest'?" in issue["reason"]
    # The model data itself is not rewritten.
    assert model.conversion_issues[0]["original_text"] == GUESTS_OCL


def test_an_original_text_that_differs_from_the_expression_is_kept():
    issue = {"name": "r", "expression": "context A inv r: self.x > 0",
             "original_text": "context A inv r: self.x > 0 -- positive", "reason": "Warning: odd"}
    compact = gap_analyzer._planner_conversion_issue(issue)
    assert compact["original_text"] == issue["original_text"]
    assert compact["reason"] == issue["reason"]
