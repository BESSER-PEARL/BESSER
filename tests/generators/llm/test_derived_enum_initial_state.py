"""The owner's directive, verbatim: "if the model have issue with the spec
bad from the modeling agent but the spec agent should fix it like I use you
claude code" - when the user's prose specification and the B-UML model
disagree, the Spec-Driven Agent closes the gap in the GENERATED CODE. It
must not ship the model's silence as a decision, and it must not leave the
gap as an unresolved checklist item forever.

Concrete instance (hotel spec, commit 143b7656): the model marks
``Booking.totalPrice``, ``physicalStatus`` and ``commercialStatus`` as
``isDerived``. Commit 143b7656 made the generator default a required
derived int/float/str/bool attribute to an unambiguous zero at INSERT time
(so ``totalPrice`` needs nothing further), but deliberately left an enum
unset - the model declares ``BookingPhysicalStatus.CHECKED_IN`` first
(alphabetically), while the spec says a booking "starts out with the guests
not yet arrived". Using the first-declared literal would have shipped every
new booking already checked in. The generator cannot know the initial
state; the spec can, and nothing before this change told the Phase 2 agent
to look for it.

The enum column is nullable now, so the create route no longer dies on it -
but a booking still starts with no status, which is not a state the spec
describes. The task stays; only its reason changed.
"""

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, PrimitiveDataType, Property,
)
from besser.generators.llm.gap_analyzer import (
    _note_derived_enum_initial_state,
    analyze_gaps_via_llm,
)
from besser.generators.llm.llm_client import UsageTracker


HOTEL_SPEC = (
    "Two states are followed for every booking, and neither is set by hand. "
    "A booking begins awaiting payment. Every new booking starts out with "
    "the guests not yet arrived. The total price of the booking is likewise "
    "not typed in. It is worked out from the agreed prices of the rooms."
)


def _commercial_status() -> Enumeration:
    return Enumeration(name="BookingCommercialStatus", literals={
        EnumerationLiteral(name="AWAITING_PAYMENT"),
        EnumerationLiteral(name="CONFIRMED"),
        EnumerationLiteral(name="CANCELLED"),
    })


def _physical_status() -> Enumeration:
    # Alphabetically CHECKED_IN sorts first - the "first declared literal"
    # every layer in this codebase agrees on (model_serializer.py sorts
    # literals by name, and so does the generated Python Enum class).
    return Enumeration(name="BookingPhysicalStatus", literals={
        EnumerationLiteral(name="CHECKED_IN"),
        EnumerationLiteral(name="CHECKED_OUT"),
        EnumerationLiteral(name="NOT_ARRIVED"),
    })


def _hotel_model() -> DomainModel:
    commercial = _commercial_status()
    physical = _physical_status()
    float_type = PrimitiveDataType("float")
    booking = Class(name="Booking")
    booking.attributes = {
        Property(name="totalPrice", type=float_type, is_derived=True),
        Property(name="commercialStatus", type=commercial, is_derived=True),
        Property(name="physicalStatus", type=physical, is_derived=True),
    }
    return DomainModel(name="Hotel", types={booking, commercial, physical})


class TestDerivedEnumWithNoInitialValueIsFlagged:

    def test_both_enum_attributes_are_flagged_quoting_the_spec(self):
        tasks = _note_derived_enum_initial_state(_hotel_model(), HOTEL_SPEC)
        by_attr = {t.split(" is a derived")[0]: t for t in tasks}

        assert set(by_attr) == {"Booking.commercialStatus", "Booking.physicalStatus"}

        commercial_task = by_attr["Booking.commercialStatus"]
        assert "SPEC DECIDES" in commercial_task
        assert "A booking\nbegins awaiting payment." in commercial_task \
            or "A booking begins awaiting payment." in commercial_task
        assert "BookingCommercialStatus.AWAITING_PAYMENT" in commercial_task

        physical_task = by_attr["Booking.physicalStatus"]
        assert "SPEC DECIDES" in physical_task
        assert "BookingPhysicalStatus.NOT_ARRIVED" in physical_task
        # The model's first-declared (alphabetical) literal must be named as
        # a warning, never adopted as the resolution.
        assert "CHECKED_IN" in physical_task
        assert "set the initial value to BookingPhysicalStatus.CHECKED_IN" not in physical_task

    def test_the_task_no_longer_claims_the_create_route_is_dead(self):
        """The column is nullable now, so the create route works and the task
        must not say otherwise -- an agent told "every create request will
        violate NOT NULL" is being pointed at a bug that no longer exists, and
        the cheapest way to silence it is the first-literal default that was
        measured and rejected. What is still open is the semantic one."""
        tasks = _note_derived_enum_initial_state(_hotel_model(), HOTEL_SPEC)
        physical = next(t for t in tasks if t.startswith("Booking.physicalStatus"))
        assert "NOT NULL" not in physical
        assert "nullable" in physical
        assert "starts with no physicalStatus at all" in physical

    def test_derived_float_attribute_already_defaulted_is_not_flagged(self):
        """commit 143b7656 already gives totalPrice a server default (0.0) -
        the generator's own fix. Nothing further is genuinely open here."""
        tasks = _note_derived_enum_initial_state(_hotel_model(), HOTEL_SPEC)
        assert not any(t.startswith("Booking.totalPrice") for t in tasks)

    def test_int_str_bool_derived_attrs_are_never_flagged(self):
        for type_name in ("int", "str", "bool"):
            cls = Class(name="X")
            cls.attributes = {
                Property(name="n", type=PrimitiveDataType(type_name), is_derived=True),
            }
            model = DomainModel(name="M", types={cls})
            assert _note_derived_enum_initial_state(model, "no spec text") == []

    def test_optional_derived_enum_needs_nothing(self):
        status = _commercial_status()
        cls = Class(name="X")
        cls.attributes = {
            Property(name="s", type=status, is_derived=True, is_optional=True),
        }
        model = DomainModel(name="M", types={cls, status})
        assert _note_derived_enum_initial_state(model, HOTEL_SPEC) == []

    def test_explicit_model_default_needs_nothing(self):
        """A modeller who set an explicit default has already decided."""
        status = _commercial_status()
        cls = Class(name="X")
        cls.attributes = {
            Property(name="s", type=status, is_derived=True, default_value="AWAITING_PAYMENT"),
        }
        model = DomainModel(name="M", types={cls, status})
        assert _note_derived_enum_initial_state(model, HOTEL_SPEC) == []

    def test_no_matching_spec_sentence_falls_back_to_a_generic_instruction(self):
        """The gap is still real even when the spec text can't be located
        automatically - it must not be silently dropped."""
        status = _commercial_status()
        cls = Class(name="X")
        cls.attributes = {Property(name="s", type=status, is_derived=True)}
        model = DomainModel(name="M", types={cls, status})

        tasks = _note_derived_enum_initial_state(model, "This spec never mentions it.")
        assert len(tasks) == 1
        assert "SPEC DECIDES" in tasks[0]
        assert "Re-read the user's specification" in tasks[0]
        assert "do not guess" in tasks[0]

    def test_no_domain_model_returns_nothing(self):
        assert _note_derived_enum_initial_state(None, HOTEL_SPEC) == []

    def test_no_enumerations_in_model_returns_nothing(self):
        cls = Class(name="X")
        cls.attributes = {Property(name="n", type=PrimitiveDataType("int"), is_derived=True)}
        model = DomainModel(name="M", types={cls})
        assert _note_derived_enum_initial_state(model, HOTEL_SPEC) == []


class _CapturingClient:
    """A planner that judges the scaffold already sufficient (``[]``)."""

    model = "capture-model"

    def __init__(self):
        self.usage = UsageTracker("capture-model")
        self._client = object()  # looks like a real provider

    def chat(self, system, messages, tools):
        return {
            "stop_reason": "end_turn",
            "content": [type("B", (), {"type": "text", "text": "[]"})()],
        }


class TestWiredIntoTheAnalyser:

    def test_survives_even_when_the_planner_says_nothing_is_needed(self):
        """The regression this change exists to close: an empty planner
        list can short-circuit Phase 2 entirely (see the module docstring
        and orchestrator._run_phase2's ``gap_tasks == []`` skip). A
        genuine spec-vs-model gap must force the list non-empty so that
        skip never fires."""
        tasks = analyze_gaps_via_llm(
            instructions=HOTEL_SPEC,
            generator_used="generate_fastapi_backend",
            domain_model=_hotel_model(),
            inventory="backend/main_api.py 1200",
            llm_client=_CapturingClient(),
        )
        assert tasks != []
        assert any(t.startswith("Booking.commercialStatus") for t in tasks)
        assert any(t.startswith("Booking.physicalStatus") for t in tasks)

    def test_harness_owned_tasks_come_first(self):
        """Deterministic, spec-decided work should not be buried behind an
        arbitrary number of planner tasks."""

        class Planner(_CapturingClient):
            def chat(self, system, messages, tools):
                return {"content": [{"type": "text", "text": '["do something else"]'}]}

        tasks = analyze_gaps_via_llm(
            instructions=HOTEL_SPEC,
            generator_used="generate_fastapi_backend",
            domain_model=_hotel_model(),
            inventory="backend/main_api.py 1200",
            llm_client=Planner(),
        )
        assert tasks[0].startswith("Booking.")
        assert tasks[-1] == "do something else"


class TestPromptStatesTheSpecOutranksTheModel:
    """Rule 3 of the Phase 2 system prompt (prompt_builder.py) already said
    the specification is "the behavior authority", but never told the agent
    to (a) prefer the spec explicitly when the two disagree, or (b) say so
    instead of silently picking one. This pins the strengthened rule."""

    def test_precedence_and_disclosure_language_present(self):
        from besser.generators.llm.prompt_builder import build_system_prompt

        cls = Class(name="Booking")
        cls.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
        model = DomainModel(name="M", types={cls})

        prompt = build_system_prompt(
            model, None, None, inventory="Generator `None` produced 0 files:",
            instructions=HOTEL_SPEC, max_turns=10,
        )
        assert "OUTRANKS the model" in prompt
        assert "say so in your plan" in prompt
