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

In a recorded run the planner emitted "Add 'commercialStatus'
enumeration with literals AWAITING_PAYMENT, CONFIRMED, CANCELLED" while the
model carried BookingCommercialStatus with exactly those literals. The data
was in the prompt and the instruction was right; the MATCH failed on the
name, which is the unreliable part, while the identical member set - the
strongest evidence of sameness - was ignored. The same run's six method
tasks were correct (declared in the model, scaffolded as HTTP 501 stubs)
and must keep being emitted.

--- from test_gap_spec_vs_model_diff.py ------------------------------
The gap analyser must diff the SPEC against the MODEL, not only the code.

Observed live on a recorded run. The user's hotel spec named
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

from besser.spec_driven_agent.planning import gap_analyzer
from besser.spec_driven_agent.planning.gap_analyzer import (
    _parse_task_array,
    _safe_serialize_model,
    analyze_gaps_via_llm,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import GENERATOR_TOOLS


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
    # Model-loss obligations survive the final bare-class fallback, even when
    # the diagnostics themselves exceed the ordinary model-summary budget.
    issues = [{"code": "unsupported_ocl", "original_text": "context Room inv: " + "x" * 13_000}]
    huge["conversion_issues"] = issues
    reduced = json.loads(_safe_serialize_model(object()))
    assert reduced["conversion_issues"] == issues
    assert reduced["__truncated__"] is True
    assert all(set(cls) == {"name"} for cls in reduced["classes"])

    class BrokenModel:
        conversion_issues = issues

        def get_classes(self):
            return [type("C", (), {"name": "Room"})()]

    def broken_serializer(_model):
        raise ValueError("unsupported element")

    monkeypatch.setattr(gap_analyzer, "serialize_domain_model", broken_serializer)
    assert json.loads(_safe_serialize_model(BrokenModel()))["conversion_issues"] == issues


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
        "besser.spec_driven_agent.pipeline.orchestrator.analyze_gaps_via_llm",
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
        "besser.spec_driven_agent.pipeline.orchestrator.analyze_gaps_via_llm",
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
        "besser.spec_driven_agent.pipeline.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )

    orch._run_phase2("build a library api", extra_issues=[])

    assert client.chat_calls == 1


def test_phase2_not_skipped_when_scoped_issues_present(tmp_path, monkeypatch):
    client = _MockClient()
    orch = _make_orchestrator(tmp_path, client)
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.spec_driven_agent.pipeline.orchestrator.analyze_gaps_via_llm",
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


def _generated_file(tmp_path, lines: int, read: bool = True) -> tuple[ToolExecutor, str]:
    """A generator-owned file of ``lines`` lines, read this run unless told not to."""
    executor = ToolExecutor(workspace=str(tmp_path))
    rel = "backend/api.py"
    full = os.path.join(str(tmp_path), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write("".join(f"line_{n} = {n}\n" for n in range(lines)))
    executor._generator_files.add(rel)
    if read:
        executor._read_file({"path": rel})   # seen this run: the SIZE guardrail is what fires
    return executor, rel


def test_write_file_allowed_after_two_modifies(tmp_path):
    """The size rule inverted: this is now the LARGE-file case.

    It used to be the small-file case (<=200 lines had to be modified twice
    before a rewrite unlocked, while anything larger could be rewritten
    cold). That is backwards: a small file is the one a model can reproduce
    faithfully from one read, and a large one is where a rewrite drops code.
    So the threshold swapped sides, and this test swapped with it - the
    assertion "two modify attempts unlock the rewrite" is unchanged, it just
    now guards the files where it matters.

    The rejection text changed with it and the assertion follows the text:
    "delete_file + write_file" is gone because wholesale replacement is the
    very thing the guard exists to discourage, and the message now names the
    unlock rule instead. Neither escape was removed from the executor.
    """
    executor, rel = _generated_file(tmp_path, 300)

    cold = json.loads(executor.execute("write_file", {"path": rel, "content": "x = 1\n"}))
    assert "error" in cold
    assert "modify_file" in cold["error"]
    assert "after two modify_file attempts" in cold["error"]

    for n in (0, 1):
        result = json.loads(executor.execute(
            "modify_file",
            {"path": rel, "old_text": f"line_{n} = {n}", "new_text": f"line_{n} = {n + 10}"},
        ))
        assert result.get("status") == "modified"

    # After two demonstrated modify attempts, the rewrite is allowed
    rewrite = json.loads(executor.execute(
        "write_file", {"path": rel, "content": "x = 1\n"},
    ))
    assert rewrite.get("status") == "written"


def test_a_small_generated_file_may_be_rewritten_once_it_has_been_read(tmp_path):
    """The other half of the inversion: <=200 lines unlocks with no modifies.

    This is the tier the edit ladder now escalates into after two refused
    edits, so it has to be reachable without first spending two refusals
    against the guardrail itself. It is unlocked by the READ, not by the
    size alone: an unread file is still refused, which is the guarantee the
    prompt's "Do not rewrite a file you have not read this run" rests on.

    Adopted for robustness, not throughput: the 96-run A/B behind the change
    moved no pass rate (12/48 vs 16/48, p=0.501) and is claimed only for
    lost scaffold code (27 items across 6 apps vs 0, p=0.027).
    """
    executor, rel = _generated_file(tmp_path, 50)

    rewrite = json.loads(executor.execute("write_file", {"path": rel, "content": "x = 1\n"}))
    assert rewrite.get("status") == "written", rewrite

    unread_executor, unread_rel = _generated_file(tmp_path / "second", 50, read=False)
    refused = json.loads(unread_executor.execute(
        "write_file", {"path": unread_rel, "content": "x = 1\n"},
    ))
    assert "error" in refused, refused
    assert "you have not read it this run" in refused["error"]


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
    from besser.spec_driven_agent.planning.gap_analyzer import _sanitize_tasks

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
        from besser.spec_driven_agent.planning.gap_analyzer import _drop_present_enumerations

        kept = _drop_present_enumerations(ENUM_TASKS + METHOD_TASKS, _booking_model())
        assert kept == METHOD_TASKS

    def test_match_ignores_the_proposed_name_and_casing(self):
        from besser.spec_driven_agent.planning.gap_analyzer import _drop_present_enumerations

        tasks = [
            "Create a new enum Status (awaiting_payment, confirmed, cancelled) for bookings",
            "Define enumeration PhysicalState: NOT_ARRIVED | CHECKED_IN | CHECKED_OUT",
        ]
        assert _drop_present_enumerations(tasks, _booking_model()) == []

    def test_a_genuinely_new_enumeration_is_still_a_gap(self):
        from besser.spec_driven_agent.planning.gap_analyzer import _drop_present_enumerations

        task = "Add 'refundStatus' enumeration with literals REQUESTED, APPROVED, PAID"
        assert _drop_present_enumerations([task], _booking_model()) == [task]

    def test_using_an_existing_enumeration_is_real_work(self):
        """Naming every literal is not the same as proposing the enumeration."""
        from besser.spec_driven_agent.planning.gap_analyzer import _drop_present_enumerations

        task = (
            "Add a cancel endpoint that moves BookingCommercialStatus from "
            "AWAITING_PAYMENT or CONFIRMED to CANCELLED"
        )
        assert _drop_present_enumerations([task], _booking_model()) == [task]

    def test_no_model_means_no_filtering(self):
        from besser.spec_driven_agent.planning.gap_analyzer import _drop_present_enumerations

        assert _drop_present_enumerations(list(ENUM_TASKS), None) == ENUM_TASKS

    def test_filter_is_wired_into_the_analyser(self):
        """End to end through analyze_gaps_via_llm with a planner that
        returns the live task list verbatim."""
        from besser.spec_driven_agent.planning.gap_analyzer import analyze_gaps_via_llm

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
        from besser.spec_driven_agent.planning.gap_analyzer import _build_user_prompt

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
from besser.spec_driven_agent.planning.gap_analyzer import _SYSTEM_PROMPT, analyze_gaps_via_llm

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


def test_full_spec_and_appended_ledger_reach_the_planner():
    """User-input limits must not silently clip accepted text or derived context."""
    from besser.spec_driven_agent.planning.specification import MAX_SPECIFICATION_CHARS

    tail = "\nThe final requirement is to release rooms when cancellation succeeds."
    original = "x" * (MAX_SPECIFICATION_CHARS - len(tail)) + tail
    augmented = original + "\n\nREQUIREMENTS LEDGER:\nR1. " + tail
    prompt = gap_analyzer._build_user_prompt(augmented, "generate_web_app", "{}", "")
    assert f"USER REQUEST:\n{augmented}\n\n" in prompt
    assert "instructions truncated" not in prompt


def test_action_inventory_finds_real_handlers_not_comments_or_orm_names(tmp_path):
    from besser.spec_driven_agent.planning.action_inventory import (
        action_gap_tasks, action_implementation_issues, collect_action_endpoints,
        format_action_inventory,
    )

    source = tmp_path / "backend" / "routers" / "order_methods.py"
    source.parent.mkdir(parents=True)
    source.write_text('''\
@router.post("/order/{order_id}/methods/approve/")
async def execute_order_approve(order_id):
    raise HTTPException(status_code=501, detail="No implementation")

@router.post("/order/{order_id}/methods/archive/")
def execute_order_archive(order_id):
    # Previously raised HTTPException(status_code=501).
    return archive_order(order_id)

@router.post("/order/{order_id}/methods/refund/")
def execute_order_refund(order_id):
    raise NotImplementedError
''', encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text(
        "class Order:\n    def approve(self):\n        return True\n", encoding="utf-8",
    )

    endpoints = collect_action_endpoints(tmp_path)
    assert [(item.action, item.stub_reason) for item in endpoints] == [
        ("approve", "HTTP 501"), ("archive", None), ("refund", "NotImplementedError"),
    ]
    assert "backend/routers/order_methods.py" in format_action_inventory(endpoints)
    tasks = action_gap_tasks(tmp_path, endpoints)
    assert len(tasks) == 2
    assert not any(item["verify"]() for item in tasks)
    issues = action_implementation_issues(tmp_path, endpoints)
    assert len(issues) == 2
    assert all(item.startswith("action contract: backend/routers/order_methods.py line ") for item in issues)


def test_action_verifier_rejects_missing_routes_and_invalid_source(tmp_path):
    from besser.spec_driven_agent.planning.action_inventory import (
        action_gap_tasks, action_implementation_issues, collect_action_endpoints,
    )

    path = tmp_path / "order_methods.py"
    decorator = ('router = APIRouter(prefix="/public")\n'
                 '@router.post("/order/{order_id}/methods/approve/")\n')
    signature = "async def execute_order_approve(order_id):\n"
    path.write_text(decorator + signature + "    pass\n", encoding="utf-8")
    expected = collect_action_endpoints(tmp_path)
    task = action_gap_tasks(tmp_path, expected)[0]
    assert task["verify"]() is False
    for body in (
        '    """TODO"""\n    ...\n',
        "    return JSONResponse(status_code=501, content={})\n",
        "    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED)\n",
    ):
        path.write_text(decorator + signature + body, encoding="utf-8")
        assert task["verify"]() is False
    path.write_text(decorator + signature + "    return await approve_order(order_id)\n", encoding="utf-8")
    assert task["verify"]() is True
    assert action_implementation_issues(tmp_path, expected) == []
    for body in (
        "    if unsupported(order_id):\n        raise HTTPException(status_code=501)\n"
        "    return await approve_order(order_id)\n",
        "    if supported(order_id):\n        return await approve_order(order_id)\n"
        "    raise HTTPException(status_code=501)\n",
        "    def unused_helper():\n        raise NotImplementedError\n"
        "    return await approve_order(order_id)\n",
        "    try:\n        raise HTTPException(status_code=501)\n"
        "    except HTTPException:\n        return await fallback_approval(order_id)\n",
    ):
        path.write_text(decorator + signature + body, encoding="utf-8")
        assert task["verify"]() is True  # No unconditional placeholder is proven.
    renamed = (decorator + signature + "    return await approve_order(order_id)\n").replace(
        "execute_order_approve", "approve_order_action",
    )
    path.write_text(renamed, encoding="utf-8")
    assert task["verify"]() is True  # Renaming Python code preserves the route.
    path.write_text(renamed.replace('prefix="/public"', 'prefix="/admin"'), encoding="utf-8")
    assert task["verify"]() is False  # Equal decorator strings are not equal mounted routes.
    path.write_text(renamed + renamed.replace("approve_order_action", "duplicate_action"), encoding="utf-8")
    assert task["verify"]() is False  # Never choose between ambiguous replacements.
    for source in (signature + "    return True\n", decorator + signature + "    invalid !\n"):
        path.write_text(source, encoding="utf-8")
        assert task["verify"]() is False
        assert len(action_implementation_issues(tmp_path, expected)) == 1
    path.unlink()
    assert task["verify"]() is False


def test_gap_planner_receives_action_handler_inventory_and_corrects_wrong_layer(tmp_path):
    from besser.spec_driven_agent.planning.action_inventory import (
        action_gap_tasks, collect_action_endpoints, merge_action_tasks,
    )

    path = tmp_path / "order_methods.py"
    path.write_text('''\
@router.post("/order/{order_id}/methods/approve/")
async def execute_order_approve(order_id):
    raise HTTPException(status_code=501)
''', encoding="utf-8")
    endpoints = collect_action_endpoints(tmp_path)

    class Planner(_CapturingClient):
        def chat(self, system, messages, tools):
            self.prompt = messages[-1]["content"]
            return {"content": [{"type": "text", "text": json.dumps([
                "Implement approve in sql_alchemy.py",
            ])}]}

    planner = Planner()
    tasks = analyze_gaps_via_llm(
        instructions="An order may be approved once.", generator_used="generate_web_app",
        domain_model=None, inventory="order_methods.py", llm_client=planner,
        workspace_files=["order_methods.py", "sql_alchemy.py"], action_endpoints=endpoints,
    )
    assert "UNIMPLEMENTED: HTTP 501" in planner.prompt
    assert "execute_order_approve" in planner.prompt
    assert len(tasks) == 1
    assert "ACTION HANDOFF" in tasks[0] and "order_methods.py" in tasks[0]
    assert "same-named ORM method alone does not connect" in tasks[0]
    canonical = action_gap_tasks(tmp_path, endpoints)
    merged = merge_action_tasks(canonical + tasks, endpoints)
    assert len(merged) == 1  # Not one task per planner + one per harness.
    assert merged[0]["text"] == canonical[0]["text"]  # Stable checkpoint key.
    assert merged[0]["planning_notes"] == tasks  # No requested behavior lost.
    assert merged[0]["verify"]() is False
    assert "planning_notes" not in canonical[0]  # Inputs are not mutated.

    # A planner omission must not erase the independently seeded obligation.
    empty_planner = _CapturingClient()
    assert analyze_gaps_via_llm(
        instructions="Approve orders.", generator_used="generate_web_app",
        domain_model=None, inventory="order_methods.py", llm_client=empty_planner,
        action_endpoints=endpoints,
    ) == []
    assert len(action_gap_tasks(tmp_path, endpoints)) == 1

    # Two entities may have a same-named action: never collapse those tasks
    # using name similarity, nor discard an unrelated rule or multi-action task.
    (tmp_path / "invoice_methods.py").write_text(
        path.read_text(encoding="utf-8").replace("order", "invoice"), encoding="utf-8",
    )
    both = collect_action_endpoints(tmp_path)
    multi = gap_analyzer._note_action_placement(["Implement approve for orders and invoices"], both)
    other = "Validate an order's contact details"
    merged = merge_action_tasks(action_gap_tasks(tmp_path, both) + multi + [other], both)
    assert len(merged) == 4
    assert multi[0] in merged and other in merged


def test_action_task_names_the_instrument_rather_than_only_saying_verify(tmp_path):
    """Measured over 211 completed runs: 1,189 of 2,985 checklist
    items say "verif*" and NOT ONE names a tool (``write_file`` is the only
    tool name that appears anywhere, in 159). Over 214 runs gpt-5.6 called
    ``test_api`` 13.1 times a run and Qwen 0.4 - 6 of 66 Qwen runs used it at
    all. The checklist gates completion and is re-listed verbatim by the
    end-turn nudge, so the item has to carry the call itself.
    """
    from besser.spec_driven_agent.planning.action_inventory import (
        action_gap_tasks, action_implementation_issues, collect_action_endpoints,
        format_action_inventory,
    )

    source = tmp_path / "backend" / "routers" / "bill_methods.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        '@router.post("/bill/{bill_id}/methods/registerPayment/")\n'
        "async def execute_bill_registerPayment(bill_id):\n"
        '    raise HTTPException(status_code=501, detail="No implementation")\n',
        encoding="utf-8",
    )
    endpoints = collect_action_endpoints(tmp_path)
    text = action_gap_tasks(tmp_path, endpoints)[0]["text"]

    assert "test_api" in text
    # still actionable: route, file, function and the 501 fact all survive
    for fact in ("POST", "/bill/{bill_id}/methods/registerPayment/",
                 "backend/routers/bill_methods.py", "execute_bill_registerPayment",
                 "HTTP 501"):
        assert fact in text, fact
    # and it gates the item on the scenario, not on the structural check
    assert "not done until" in text
    assert "structural check" not in text

    assert "test_api" in format_action_inventory(endpoints)
    assert "test_api" in action_implementation_issues(tmp_path, endpoints)[0]
