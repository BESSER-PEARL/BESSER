"""Tests for the Phase-2 gap analyser (``gap_analyzer.py``).

--- from test_gap_contract_and_phase2_skip.py ------------------------
Tests for the gap-analyser output contract and the Phase-2 skip.

The contract (see gap_analyzer module docstring):
  * ``None``  — analysis failed (LLM error, unparseable reply, mock client)
  * ``[]``    — the model judged the scaffold sufficient → Phase 2 may skip
  * ``[...]`` — focused task list

Also covers the supporting plumbing: always-valid-JSON model truncation,
whole-text task parsing, the relaxed write_file guardrail, and the
GENERATOR_TOOLS ↔ ToolExecutor._handlers parity (including qiskit).

--- from test_gap_false_gaps.py --------------------------------------
The gap analyser must not propose work the model already carries.

Live run, 2026-09-17: the planner emitted "Add 'commercialStatus'
enumeration with literals AWAITING_PAYMENT, CONFIRMED, CANCELLED" while the
model carried BookingCommercialStatus with exactly those literals. The data
was in the prompt and the instruction was right; the MATCH failed on the
name, which is the unreliable part, while the identical member set - the
strongest evidence of sameness - was ignored. The same run's six method
tasks were correct (declared in the model, scaffolded as HTTP 501 stubs)
and must keep being emitted.

--- from test_gap_spec_vs_model_diff.py ------------------------------
The gap analyser must diff the SPEC against the MODEL, not only the code.

Observed live on run ``ca48a6dd`` (2026-09-17). The user's hotel spec named
five booking actions, two independent status dimensions, four validity rules
and a price that belongs to the booking-room *link*. The modelling step
captured two methods, one merged enum, zero constraints and no link class.
The gap analyser then produced four tasks, all cosmetic:

    "Generated enhanced employee page with improved styling"
    "Added HotelNavigation component"
    "Created personalized Home page"
    "Improved Person page with clean card-based layout and Tailwind"

Nothing about the missing behaviour. The analyser is the only stage that
holds the request and the model side by side, so whatever it does not
notice there is lost for the rest of the run — Phase 2 works from the
checklist, and Phase 3 validates the code against the model.

The prompt used to say only "List the missing or incorrect work", which
invites a code-vs-model reading. These tests pin the spec-vs-model pass and
the framing that makes it load-bearing.

"""
from __future__ import annotations


# ==========================================================================
# from test_gap_contract_and_phase2_skip.py
# ==========================================================================

import json
import os

import pytest

from besser.generators.llm import gap_analyzer
from besser.generators.llm.gap_analyzer import (
    _parse_task_array,
    _safe_serialize_model,
    analyze_gaps_via_llm,
)
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.tools import GENERATOR_TOOLS


class _MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockClient:
    """Duck-typed client WITHOUT ``_client`` — gap analyser must skip it."""

    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")
        self.chat_calls = 0

    def chat(self, system, messages, tools):
        self.chat_calls += 1
        return {"stop_reason": "end_turn", "content": [_MockBlock("text", text="Done")]}


# ----------------------------------------------------------------------
# Output contract
# ----------------------------------------------------------------------


def test_mock_client_returns_none_not_empty():
    """A skipped analysis must be None (failure), never [] (no work)."""
    result = analyze_gaps_via_llm(
        instructions="add auth",
        generator_used="generate_fastapi_backend",
        domain_model=None,
        inventory="3 files",
        llm_client=_MockClient(),
    )
    assert result is None


def test_no_generator_fallback_mentions_failure_reason():
    result = analyze_gaps_via_llm(
        instructions="build it",
        generator_used=None,
        domain_model=None,
        inventory="",
        llm_client=_MockClient(),
        generator_failure="generate_sqlalchemy: reserved name 'Base'",
    )
    assert isinstance(result, list) and len(result) == 1
    assert "generate_sqlalchemy: reserved name 'Base'" in result[0]
    assert "avoid the cause" in result[0]


def test_no_generator_fallback_without_failure_keeps_legacy_text():
    result = analyze_gaps_via_llm(
        instructions="build it",
        generator_used=None,
        domain_model=None,
        inventory="",
        llm_client=_MockClient(),
    )
    assert isinstance(result, list) and len(result) == 1
    assert "No BESSER generator was used" in result[0]


# ----------------------------------------------------------------------
# Parsing / truncation plumbing
# ----------------------------------------------------------------------


def test_parse_task_array_survives_brackets_inside_tasks():
    text = '["update models[0] in api.py", "add CHECK (age > [0]) constraint"]'
    parsed = _parse_task_array(text)
    assert parsed == [
        "update models[0] in api.py",
        "add CHECK (age > [0]) constraint",
    ]


def test_parse_task_array_fenced_json():
    text = '```json\n["task one", "task two"]\n```'
    assert _parse_task_array(text) == ["task one", "task two"]


def test_safe_serialize_model_truncation_is_always_valid_json(monkeypatch):
    """A model far over budget must still serialize to parseable JSON."""
    huge = {
        "classes": [
            {
                "name": f"Class{i}",
                "attributes": [
                    {"name": f"attr{j}", "type": "str", "visibility": "public",
                     "doc": "x" * 50}
                    for j in range(30)
                ],
                "methods": [{"name": f"m{j}", "params": []} for j in range(20)],
                "inherited_attributes": [{"name": "base"} for _ in range(20)],
                "inherited_methods": [{"name": "bm"} for _ in range(20)],
            }
            for i in range(200)
        ],
    }
    monkeypatch.setattr(gap_analyzer, "serialize_domain_model", lambda _m: huge)
    payload = _safe_serialize_model(object())
    data = json.loads(payload)  # must never raise
    assert data.get("__truncated__") is True
    assert len(payload) <= gap_analyzer._MAX_MODEL_JSON_CHARS + 100


# ----------------------------------------------------------------------
# Phase-2 skip
# ----------------------------------------------------------------------


class _ExplodingClient(_MockClient):
    """chat() must never be reached when Phase 2 is skipped."""

    def chat(self, system, messages, tools):  # pragma: no cover - guard
        raise AssertionError("Phase 2 was not skipped — chat() was called")


def _make_orchestrator(tmp_path, client):
    class _SM:
        name = "DummySM"

    return LLMOrchestrator(
        llm_client=client,
        state_machines=[_SM()],
        output_dir=str(tmp_path),
    )


def test_phase2_skipped_when_gap_tasks_empty_and_generator_ran(tmp_path, monkeypatch):
    client = _ExplodingClient()
    orch = _make_orchestrator(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )
    progress: list[tuple] = []
    orch.on_progress = lambda *a: progress.append(a)

    orch._run_phase2("build a library api", extra_issues=[])

    assert orch.total_turns == 0
    assert orch._phase2_exited_cleanly is True
    assert any(a[1] == "__customize_skipped__" for a in progress)


def test_phase2_not_skipped_when_analysis_failed(tmp_path, monkeypatch):
    """None (failure) must keep today's behavior: loop without checklist."""
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: None,
    )

    orch._run_phase2("build a library api", extra_issues=[])

    assert client.chat_calls == 1  # entered the loop, got end_turn
    assert orch.total_turns == 1


def test_phase2_not_skipped_when_no_generator(tmp_path, monkeypatch):
    """[] without a deterministic scaffold is not trusted as 'done'."""
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    assert orch._generator_used is None
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )

    orch._run_phase2("build a library api", extra_issues=[])

    assert client.chat_calls == 1


def test_phase2_not_skipped_when_scoped_issues_present(tmp_path, monkeypatch):
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )

    orch._run_phase2(
        "build a library api",
        extra_issues=["Fix syntax error in api.py line 3"],
    )

    assert client.chat_calls == 1


# ----------------------------------------------------------------------
# write_file guardrail relaxation
# ----------------------------------------------------------------------


def test_write_file_allowed_after_two_modifies(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    rel = "backend/api.py"
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write("line_a = 1\nline_b = 2\n")
    executor._generator_files.add(rel)
    executor._read_file({"path": rel})      # seen this run: the generator-file guardrail is what fires

    # Cold write on a small generated file → rejected, with both escapes named
    cold = json.loads(executor.execute("write_file", {"path": rel, "content": "x = 1\n"}))
    assert "error" in cold
    assert "modify_file" in cold["error"]
    assert "delete_file" in cold["error"]

    for old, new in (("line_a = 1", "line_a = 10"), ("line_b = 2", "line_b = 20")):
        result = json.loads(executor.execute(
            "modify_file", {"path": rel, "old_text": old, "new_text": new},
        ))
        assert result.get("status") == "modified"

    # After two demonstrated modify attempts, the rewrite is allowed
    rewrite = json.loads(executor.execute(
        "write_file", {"path": rel, "content": "x = 1\n"},
    ))
    assert rewrite.get("status") == "written"


# ----------------------------------------------------------------------
# Tool registry parity (incl. the qiskit registration)
# ----------------------------------------------------------------------


def test_every_generator_tool_has_an_executor_handler():
    handler_names = set(ToolExecutor._handlers.keys())
    for tool in GENERATOR_TOOLS:
        assert tool["name"] in handler_names, f"{tool['name']} has no handler"


def test_qiskit_without_circuit_surfaces_clean_error(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path), quantum_circuit=None)
    result = json.loads(executor.execute("generate_qiskit", {}))
    assert "error" in result
    assert "quantum circuit" in result["error"].lower()


def test_target_generator_override_is_binding(tmp_path):
    """A caller-specified generator must win without any LLM call."""
    client = _ExplodingClient()  # chat() raises if reached

    class _SM:
        name = "DummySM"

    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[_SM()],
        output_dir=str(tmp_path),
        target_generator="generate_fastapi_backend",
    )
    assert orch._select_generator("anything at all") == "generate_fastapi_backend"


def test_explicit_no_generator_override_is_binding(tmp_path):
    """An approved from-scratch plan must skip the paid selector call."""
    client = _ExplodingClient()  # chat() raises if reached

    class _SM:
        name = "DummySM"

    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[_SM()],
        output_dir=str(tmp_path),
        target_generator=None,
        target_generator_bound=True,
    )
    assert orch._select_generator("anything at all") is None


def test_resume_seeds_prior_cost(tmp_path, monkeypatch):
    """A resumed run's cost cap must cover what the crashed run spent."""
    tracker = UsageTracker("mock-model")
    tracker.seed_cost(1.25)
    assert abs(tracker.estimated_cost - 1.25) < 1e-9
    # Token-based accounting still works on top of the seed
    tracker.output_tokens += 100_000  # sonnet fallback pricing: $1.50
    assert tracker.estimated_cost > 1.25


def test_gap_sanitizer_drops_scaffold_demolition_and_rival_framework():
    """Devstral A/B live finding: the planner proposed deleting the react
    scaffold and installing Flask. The sanitizer must drop those, keep
    honest tasks, and respect a USER-requested rival framework."""
    from besser.generators.llm.gap_analyzer import _sanitize_tasks

    tasks = [
        "delete react frontend scaffold as not requested by user",
        "install Flask backend framework requirements in backend directory",
        "create book model class that implements Book concept",
        "add login route to the main application",
    ]
    kept = _sanitize_tasks(tasks, "generate_web_app", "Build a library web app")
    assert kept == [
        "create book model class that implements Book concept",
        "add login route to the main application",
    ]

    # User explicitly asked for flask -> rival mention is legitimate.
    kept2 = _sanitize_tasks(
        ["create flask blueprint for API endpoints"],
        "generate_fastapi_backend",
        "build me a flask backend",
    )
    assert kept2 == ["create flask blueprint for API endpoints"]

    # From-scratch runs (no scaffold) keep everything non-demolition.
    kept3 = _sanitize_tasks(["use flask for the app"], None, "make an app")
    assert kept3 == ["use flask for the app"]


# ==========================================================================
# from test_gap_false_gaps.py
# ==========================================================================


from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, Method,
    PrimitiveDataType, Property,
)

ENUM_TASKS = [
    "Add 'commercialStatus' enumeration with literals AWAITING_PAYMENT, CONFIRMED, CANCELLED",
    "Add 'physicalStatus' enumeration with literals NOT_ARRIVED, CHECKED_IN, CHECKED_OUT",
]
METHOD_TASKS = [
    "Implement Booking.produceBill in booking_methods.py (scaffold returns HTTP 501)",
    "Implement Booking.registerArrival in booking_methods.py (scaffold returns HTTP 501)",
    "Implement Booking.registerDeparture in booking_methods.py (scaffold returns HTTP 501)",
    "Implement Booking.cancel in booking_methods.py (scaffold returns HTTP 501)",
    "Implement Booking.computeAmountOwed in booking_methods.py (scaffold returns HTTP 501)",
    "Implement Bill.registerPayment in bill_methods.py (scaffold returns HTTP 501)",
]


def _enum(name: str, *literals: str) -> Enumeration:
    return Enumeration(name=name, literals={EnumerationLiteral(name=lit) for lit in literals})


def _booking_model() -> DomainModel:
    """Minimal hand-built version of the live project: two enumerations,
    two classes with declared (unimplemented) methods."""
    str_type = PrimitiveDataType("str")
    booking = Class(name="Booking")
    booking.attributes = {Property(name="ref", type=str_type)}
    booking.methods = {
        Method(name=n) for n in (
            "produceBill", "registerArrival", "registerDeparture", "cancel", "computeAmountOwed",
        )
    }
    bill = Class(name="Bill")
    bill.attributes = {Property(name="number", type=str_type)}
    bill.methods = {Method(name="registerPayment")}
    return DomainModel(name="Hotel", types={
        booking, bill,
        _enum("BookingCommercialStatus", "AWAITING_PAYMENT", "CONFIRMED", "CANCELLED"),
        _enum("BookingPhysicalStatus", "NOT_ARRIVED", "CHECKED_IN", "CHECKED_OUT"),
    })


class TestPresentEnumerationsAreNotGaps:

    def test_present_enumerations_are_dropped_and_declared_methods_kept(self):
        """Pins the live run: the two enumeration tasks go, all six method
        tasks stay, order preserved."""
        from besser.generators.llm.gap_analyzer import _drop_present_enumerations

        kept = _drop_present_enumerations(ENUM_TASKS + METHOD_TASKS, _booking_model())
        assert kept == METHOD_TASKS

    def test_match_ignores_the_proposed_name_and_casing(self):
        from besser.generators.llm.gap_analyzer import _drop_present_enumerations

        tasks = [
            "Create a new enum Status (awaiting_payment, confirmed, cancelled) for bookings",
            "Define enumeration PhysicalState: NOT_ARRIVED | CHECKED_IN | CHECKED_OUT",
        ]
        assert _drop_present_enumerations(tasks, _booking_model()) == []

    def test_a_genuinely_new_enumeration_is_still_a_gap(self):
        from besser.generators.llm.gap_analyzer import _drop_present_enumerations

        task = "Add 'refundStatus' enumeration with literals REQUESTED, APPROVED, PAID"
        assert _drop_present_enumerations([task], _booking_model()) == [task]

    def test_using_an_existing_enumeration_is_real_work(self):
        """Naming every literal is not the same as proposing the enumeration."""
        from besser.generators.llm.gap_analyzer import _drop_present_enumerations

        task = (
            "Add a cancel endpoint that moves BookingCommercialStatus from "
            "AWAITING_PAYMENT or CONFIRMED to CANCELLED"
        )
        assert _drop_present_enumerations([task], _booking_model()) == [task]

    def test_no_model_means_no_filtering(self):
        from besser.generators.llm.gap_analyzer import _drop_present_enumerations

        assert _drop_present_enumerations(list(ENUM_TASKS), None) == ENUM_TASKS

    def test_filter_is_wired_into_the_analyser(self):
        """End to end through analyze_gaps_via_llm with a planner that
        returns the live task list verbatim."""
        from besser.generators.llm.gap_analyzer import analyze_gaps_via_llm

        class Planner:
            _client = object()      # looks like a real provider

            def chat(self, system, messages, tools):
                return {"content": [{"type": "text", "text": json.dumps(ENUM_TASKS + METHOD_TASKS)}]}

        tasks = analyze_gaps_via_llm(
            instructions="Hotel bookings with commercial and physical status.",
            generator_used="generate_fastapi_backend",
            domain_model=_booking_model(),
            inventory="backend/main_api.py 1200\nbackend/booking_methods.py 400",
            llm_client=Planner(),
        )
        assert tasks == METHOD_TASKS


class TestMatchingInstruction:

    def test_prompt_says_match_on_meaning_and_member_sets(self):
        """Belt to the filter's braces: the prompt must stop equating
        'appears in the model' with 'has this exact name'."""
        from besser.generators.llm.gap_analyzer import _build_user_prompt

        prompt = _build_user_prompt("req", "generate_fastapi_backend", "{}", "inv").lower()
        assert "member sets" in prompt
        assert "not on exact names" in prompt
        assert "prefix" in prompt


# ==========================================================================
# from test_gap_spec_vs_model_diff.py
# ==========================================================================


from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Method,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.llm.gap_analyzer import _SYSTEM_PROMPT, analyze_gaps_via_llm

# An abridged form of the live request, keeping every construct the model
# lost: five named actions, two status dimensions, prose rules, and a fact
# that belongs to the relationship.
SPEC = """
Build a web application to manage a small hotel.

A booking offers five actions: produce the bill, check the guest in, check
the guest out, cancel the booking, and compute the amount due.

A booking has a commercial status: awaiting payment, confirmed, or
cancelled. Separately it has a physical status: not arrived, checked in, or
checked out.

The total number of guests must not exceed the combined capacity of the
rooms booked. A room cannot be double-booked for overlapping dates. Email
and phone must be valid. The arrival date must not be after the departure
date.

For each room in a booking we record the price actually agreed for that
room in that booking, which may differ from the standard price.
"""


def _under_captured_model() -> DomainModel:
    """What the modelling step actually produced: 2 of 5 methods, the two
    status dimensions merged into one enum with invented members, no
    constraints, and no link class carrying the agreed price."""
    status = Enumeration(
        name="BookingStatus",
        literals={
            EnumerationLiteral(name="BOOKED"),
            EnumerationLiteral(name="CHECKED_IN"),
            EnumerationLiteral(name="CHECKED_OUT"),
            EnumerationLiteral(name="CANCELLED"),
            EnumerationLiteral(name="NO_SHOW"),
        },
    )

    booking = Class(name="Booking")
    booking.attributes = {Property(name="reference", type=StringType)}
    booking.methods = {Method(name="checkIn"), Method(name="calculateTotal")}

    room = Class(name="Room")
    room.attributes = {Property(name="number", type=StringType)}

    booked = Property(name="rooms", type=room, multiplicity=Multiplicity(1, "*"))
    booked_by = Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation(name="booking_room", ends={booked, booked_by})

    return DomainModel(
        name="Hotel",
        types={booking, room, status},
        associations={assoc},
    )


class _CapturingClient:
    """Records the planner prompt, then returns an empty task list."""

    model = "capture-model"

    def __init__(self):
        self.usage = UsageTracker("capture-model")
        self._client = object()  # makes the analyser treat it as real
        self.system = None
        self.prompt = None

    def chat(self, system, messages, tools):
        self.system = system
        self.prompt = messages[-1]["content"]
        return {
            "stop_reason": "end_turn",
            "content": [type("B", (), {"type": "text", "text": "[]"})()],
        }


@pytest.fixture
def planner_prompt() -> str:
    client = _CapturingClient()
    result = analyze_gaps_via_llm(
        instructions=SPEC,
        generator_used="generate_fastapi_backend",
        domain_model=_under_captured_model(),
        inventory="backend/main.py 1.2K\nfrontend/src/App.tsx 3.4K",
        llm_client=client,
    )
    assert result == [], "fixture expects the stubbed empty reply"
    assert client.prompt, "the planner was never called"
    return client.prompt


def test_prompt_carries_the_spec_and_the_model_side_by_side(planner_prompt):
    """The diff is only possible if both halves actually reach the model."""
    assert "produce the bill" in planner_prompt
    assert "awaiting payment" in planner_prompt
    assert "BookingStatus" in planner_prompt  # the model JSON
    assert "checkIn" in planner_prompt


def test_prompt_asks_for_a_spec_vs_model_pass(planner_prompt):
    """The regression this file exists for: the instruction used to be a
    bare 'list the missing or incorrect work', which reads as code-vs-model
    and produced four styling tasks against a spec with 12 real gaps."""
    low = planner_prompt.lower()
    assert "user request vs domain model" in low
    assert "does not appear is a gap" in low
    # The four categories the live run dropped, each named explicitly.
    assert "operations an entity must support" in low
    assert "status or state vocabularies" in low
    assert "rules, limits and validity conditions" in low
    assert "belong to a relationship" in low


def test_prompt_orders_spec_gaps_first(planner_prompt):
    """With a 16-task cap, ordering decides what survives truncation."""
    assert "Pass-1 gaps FIRST" in planner_prompt
    assert planner_prompt.index("PASS 1") < planner_prompt.index("PASS 2")


def test_two_status_dimensions_must_not_merge(planner_prompt):
    low = planner_prompt.lower()
    assert "two enumerations, not one" in low
    assert "never invented" in low


def test_system_prompt_denies_the_model_authority_over_the_spec():
    """The analyser has to be told the model is lossy; otherwise a model
    that omits a requirement reads as a requirement that does not exist."""
    assert "THE DOMAIN MODEL IS NOT THE SPEC" in _SYSTEM_PROMPT
    low = _SYSTEM_PROMPT.lower()
    assert "the user request is the authority" in low
    assert "you are the only step that sees" in low


def test_spec_fits_the_instruction_budget():
    """A clipped spec silently removes the requirements being diffed."""
    assert len(SPEC) < gap_analyzer._MAX_INSTRUCTIONS_CHARS
