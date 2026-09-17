"""The gap analyser must not propose work the model already carries.

Live run, 2026-09-17: the planner emitted "Add 'commercialStatus'
enumeration with literals AWAITING_PAYMENT, CONFIRMED, CANCELLED" while the
model carried BookingCommercialStatus with exactly those literals. The data
was in the prompt and the instruction was right; the MATCH failed on the
name, which is the unreliable part, while the identical member set - the
strongest evidence of sameness - was ignored. The same run's six method
tasks were correct (declared in the model, scaffolded as HTTP 501 stubs)
and must keep being emitted.
"""

import json

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
